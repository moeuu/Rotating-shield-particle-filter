"""Baseline PF demo without shielding, with real-time visualization."""

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

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import load_or_generate_obstacle_grid
from spectrum.pipeline import SpectralDecomposer
from visualization.realtime_viz import RealTimePFVisualizer, build_frame_from_pf
from evaluation_metrics import compute_metrics, print_metrics_report

from baseline_pf.measurement import BaselineMeasurement
from baseline_pf.particle_filter import BaselinePF, BaselinePFConfig
from baseline_pf.planning import generate_measurement_positions

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "baseline_pf"
DEFAULT_SOURCE_CONFIG = ROOT / "source_layouts" / "demo_sources.json"
DEFAULT_OBSTACLE_CONFIG = ROOT / "obstacle_layouts" / "demo_obstacles.json"
MEASUREMENT_TIME_S = 30.0


def _build_demo_sources() -> list[PointSource]:
    """Return a default set of point sources."""
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
                "Each source must include 'isotope', 'position', and "
                "'intensity_cps_1m'."
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


def run_baseline_pf(
    *,
    live: bool = True,
    total_time_s: float = 300.0,
    sources: list[PointSource] | None = None,
    obstacle_layout_path: str | None = DEFAULT_OBSTACLE_CONFIG.as_posix(),
    obstacle_seed: int | None = None,
    detect_threshold_abs: float = 30.0,
    detect_threshold_rel: float = 0.2,
    eval_match_radius_m: float = 0.5,
) -> None:
    """Run the baseline PF using evenly spaced measurements."""
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=10.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    sources = _build_demo_sources() if sources is None else sources
    decomposer = SpectralDecomposer()
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
    positions = generate_measurement_positions(
        env,
        obstacle_grid,
        total_time_s,
        measurement_time_s=MEASUREMENT_TIME_S,
    )
    isotopes = list(decomposer.isotope_names)
    pf_config = BaselinePFConfig(
        num_particles=800,
        resample_threshold=0.5,
        position_sigma=0.4,
        strength_sigma=0.3,
        min_strength=0.01,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
    )
    pf = BaselinePF(
        isotopes=isotopes,
        config=pf_config,
        rng=np.random.default_rng(0),
    )

    true_src = {}
    true_strengths = {}
    for iso in isotopes:
        pos_list = [
            np.array(src.position, dtype=float)
            for src in sources
            if src.isotope == iso
        ]
        str_list = [src.intensity_cps_1m for src in sources if src.isotope == iso]
        if pos_list:
            true_src[iso] = np.vstack(pos_list)
        if str_list:
            true_strengths[iso] = float(np.max(str_list))

    viz = RealTimePFVisualizer(
        isotopes=isotopes,
        world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z),
        true_sources=true_src,
        true_strengths=true_strengths,
        obstacle_grid=obstacle_grid,
        show_counts=False,
    )
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    estimate_mode = "mmse"
    estimate_min_strength = 100.0
    if live:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

    elapsed = 0.0
    for step_idx, pose in enumerate(positions):
        env_step = EnvironmentConfig(detector_position=tuple(pose))
        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env_step,
            acquisition_time=MEASUREMENT_TIME_S,
            rng=np.random.default_rng(123 + step_idx),
        )
        counts, _ = decomposer.isotope_counts_with_detection(
            spectrum,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
        )
        measurement = BaselineMeasurement(
            counts_by_isotope=counts,
            live_time_s=MEASUREMENT_TIME_S,
            detector_position=pose,
            pose_idx=step_idx,
            RFe=np.eye(3),
            RPb=np.eye(3),
        )
        pf.update_all(pose, counts, MEASUREMENT_TIME_S)
        elapsed += MEASUREMENT_TIME_S
        frame = build_frame_from_pf(
            pf,
            measurement,
            step_index=step_idx,
            time_sec=elapsed,
            estimate_mode=estimate_mode,
            min_est_strength=estimate_min_strength,
        )
        viz.update(frame)
        print(f"[step {step_idx}] pose={pose.tolist()} measurement={counts}")
        step_path = out_dir / f"step_{step_idx:04d}_pf.png"
        viz.save_final(step_path.as_posix())
        if live:
            plt.pause(0.05)

    pf_out_path = out_dir / "result_baseline_pf.png"
    est_out_path = out_dir / "result_baseline_estimates.png"
    viz.save_final(pf_out_path.as_posix())
    viz.save_estimates_only(est_out_path.as_posix())
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
    for iso, state in pf.estimate_all().items():
        est_list: list[dict[str, float | list[float]]] = []
        positions = np.asarray(state.positions, dtype=float)
        strengths = np.asarray(state.strengths, dtype=float)
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
