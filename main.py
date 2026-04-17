"""CLI entry point for the real-time PF demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_demo import (
    DEFAULT_OBSTACLE_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    load_sources_from_json,
    run_live_pf,
)


def main() -> None:
    """Parse CLI arguments and run the real-time PF demo."""
    parser = argparse.ArgumentParser(description="Real-time rotating-shield PF visualization demo.")
    parser.add_argument(
        "--max-steps",
        "--steps",
        dest="max_steps",
        type=int,
        default=None,
        help="Maximum number of measurement steps (default: run until convergence).",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=15,
        help="Maximum number of measurement poses (default: 15).",
    )
    parser.add_argument(
        "--pose-candidates",
        type=int,
        default=64,
        help="Number of candidate poses to generate per step (default: 64).",
    )
    parser.add_argument(
        "--pose-min-dist",
        type=float,
        default=3.0,
        help="Minimum distance (m) from visited poses for candidates (default: 3.0).",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help=(
            "Disable interactive updating (still saves results/result_pf.png and "
            "results/result_spectrum.png by default)."
        ),
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag appended to result output filenames (ex: ex5 -> result_pf_ex5.png).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an interactive window (forces Agg backend and disables live updates).",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default=DEFAULT_SOURCE_CONFIG.as_posix(),
        help="Path to a JSON file that defines the point sources.",
    )
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default=DEFAULT_OBSTACLE_CONFIG.as_posix(),
        help="Path to a JSON file that defines blocked grid cells.",
    )
    parser.add_argument(
        "--environment-mode",
        type=str,
        default="fixed",
        choices=("fixed", "random"),
        help=(
            "Environment generation mode: fixed loads the obstacle JSON, "
            "random creates a fresh obstacle layout at startup."
        ),
    )
    parser.add_argument(
        "--obstacle-seed",
        type=int,
        default=None,
        help="RNG seed used when creating a fixed missing layout or a random startup layout.",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Disable obstacles during pose selection and visualization.",
    )
    parser.add_argument(
        "--ig-threshold-mode",
        type=str,
        default="relative_pose",
        choices=("absolute", "relative_max", "relative_pose"),
        help="IG threshold mode: absolute or relative to max IG.",
    )
    parser.add_argument(
        "--ig-threshold-rel",
        type=float,
        default=0.02,
        help="Relative IG threshold fraction for dynamic modes.",
    )
    parser.add_argument(
        "--ig-threshold-min",
        type=float,
        default=None,
        help="Minimum IG threshold floor (defaults to config value).",
    )
    parser.add_argument(
        "--detect-threshold-abs",
        type=float,
        default=30.0,
        help="Absolute detection threshold for peak-matched activity (counts).",
    )
    parser.add_argument(
        "--detect-threshold",
        dest="detect_threshold_abs",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --detect-threshold-abs.",
    )
    parser.add_argument(
        "--detect-threshold-rel",
        type=float,
        default=0.2,
        help="Relative detection threshold as a fraction of max peak-matched activity.",
    )
    parser.add_argument(
        "--detect-consecutive",
        type=int,
        default=20,
        help="Consecutive detections required to enable an isotope.",
    )
    parser.add_argument(
        "--detect-min-steps",
        type=int,
        default=None,
        help="Minimum steps before locking detected isotopes (defaults to detect_consecutive).",
    )
    parser.add_argument(
        "--eval-match-radius",
        type=float,
        default=0.5,
        help="Match radius (m) for evaluation metrics.",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="spectrum",
        choices=("spectrum", "expected"),
        help="Counts to pass to the PF: spectrum (default) or expected.",
    )
    parser.add_argument(
        "--birth",
        action="store_true",
        help="Enable birth/death/split/merge moves (default: disabled).",
    )
    parser.add_argument(
        "--merge-prob",
        type=float,
        default=None,
        help="Merge proposal probability when birth/death is enabled (default: 0.4).",
    )
    parser.add_argument(
        "--merge-distance-max",
        type=float,
        default=None,
        help="Max distance (m) to merge nearby sources (default: 1.0).",
    )
    parser.add_argument(
        "--merge-delta-ll-threshold",
        type=float,
        default=None,
        help="Log-likelihood threshold for merge acceptance (default: -1.0).",
    )
    parser.add_argument(
        "--cluster-eps-m",
        type=float,
        default=None,
        help="Clustering epsilon (m) for output estimates (default: 1.2).",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Maximum number of sources per isotope (defaults to 3 with --birth, else 1).",
    )
    parser.add_argument(
        "--temper-max-resamples",
        type=int,
        default=2,
        help="Max resamples per observation during tempering (default: 2).",
    )
    parser.add_argument(
        "--no-roughen-on-temper-resample",
        action="store_true",
        help="Disable roughening on resamples triggered inside tempering.",
    )
    parser.add_argument(
        "--roughening-k",
        type=float,
        default=None,
        help="Override roughening coefficient k (optional).",
    )
    parser.add_argument(
        "--min-sigma-pos",
        type=float,
        default=None,
        help="Override minimum roughening sigma (optional).",
    )
    parser.add_argument(
        "--max-sigma-pos",
        type=float,
        default=None,
        help="Override maximum roughening sigma (optional).",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        help="Enable per-isotope convergence gating (default: disabled).",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="analytic",
        choices=("analytic", "isaacsim", "geant4"),
        help="Simulation backend used for observation generation.",
    )
    parser.add_argument(
        "--sim-config",
        type=str,
        default=None,
        help="Optional JSON config path for the selected simulation backend.",
    )
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=None,
        help="Blender executable path used by --environment-mode random.",
    )
    parser.add_argument(
        "--blender-output",
        type=str,
        default=None,
        help="Optional USD output path for the Blender-generated random environment.",
    )
    parser.add_argument(
        "--blender-timeout-s",
        type=float,
        default=120.0,
        help="Timeout for Blender random environment generation.",
    )
    parser.add_argument(
        "--robot-speed",
        type=float,
        default=0.5,
        help="Nominal robot travel speed in m/s used for mission-time accounting.",
    )
    parser.add_argument(
        "--rotation-overhead-s",
        type=float,
        default=0.5,
        help="Fixed shield actuation overhead per measurement in seconds.",
    )
    args = parser.parse_args()
    if args.max_sources is None:
        args.max_sources = 3 if args.birth else 1
    pf_overrides: dict[str, object] = {
        "max_sources": args.max_sources,
        "max_resamples_per_observation": args.temper_max_resamples,
    }
    if args.merge_prob is not None:
        pf_overrides["merge_prob"] = float(args.merge_prob)
    elif args.birth:
        pf_overrides["merge_prob"] = 0.4
    if args.merge_distance_max is not None:
        pf_overrides["merge_distance_max"] = float(args.merge_distance_max)
    elif args.birth:
        pf_overrides["merge_distance_max"] = 1.0
    if args.merge_delta_ll_threshold is not None:
        pf_overrides["merge_delta_ll_threshold"] = float(args.merge_delta_ll_threshold)
    elif args.birth:
        pf_overrides["merge_delta_ll_threshold"] = -1.0
    if args.cluster_eps_m is not None:
        pf_overrides["cluster_eps_m"] = float(args.cluster_eps_m)
    elif args.birth:
        pf_overrides["cluster_eps_m"] = 1.2
    if args.no_roughen_on_temper_resample:
        pf_overrides["disable_regularize_on_temper_resample"] = True
    if args.roughening_k is not None:
        pf_overrides["roughening_k"] = float(args.roughening_k)
    if args.min_sigma_pos is not None:
        pf_overrides["min_sigma_pos"] = float(args.min_sigma_pos)
    if args.max_sigma_pos is not None:
        pf_overrides["max_sigma_pos"] = float(args.max_sigma_pos)
    sources = None
    if args.source_config:
        source_path = Path(args.source_config)
        if not source_path.is_absolute():
            source_path = (ROOT / source_path).resolve()
        if source_path.exists():
            try:
                sources = load_sources_from_json(source_path)
                print(f"Loaded {len(sources)} sources from {source_path}")
            except (OSError, ValueError) as exc:
                print(f"Failed to load sources from {source_path}: {exc}")
        else:
            print(f"Source config not found: {source_path}. Using built-in demo sources.")
    run_live_pf(
        live=not (args.no_live or args.headless),
        max_steps=args.max_steps,
        max_poses=args.max_poses,
        sources=sources,
        detect_threshold_abs=args.detect_threshold_abs,
        detect_threshold_rel=args.detect_threshold_rel,
        detect_consecutive=args.detect_consecutive,
        detect_min_steps=args.detect_min_steps,
        ig_threshold_mode=args.ig_threshold_mode,
        ig_threshold_rel=args.ig_threshold_rel,
        ig_threshold_min=args.ig_threshold_min,
        environment_mode=args.environment_mode,
        obstacle_layout_path=None if args.no_obstacles else args.obstacle_config,
        obstacle_seed=args.obstacle_seed,
        eval_match_radius_m=args.eval_match_radius,
        count_mode=args.count,
        birth_enabled=args.birth,
        pf_config_overrides=pf_overrides,
        output_tag=args.output_tag,
        pose_candidates=args.pose_candidates,
        pose_min_dist=args.pose_min_dist,
        converge=args.converge,
        sim_backend=args.sim_backend,
        sim_config_path=args.sim_config,
        blender_executable=args.blender_executable,
        blender_output_path=args.blender_output,
        blender_timeout_s=args.blender_timeout_s,
        nominal_motion_speed_m_s=args.robot_speed,
        rotation_overhead_s=args.rotation_overhead_s,
    )


if __name__ == "__main__":
    main()
