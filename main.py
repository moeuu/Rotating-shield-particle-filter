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

from realtime_demo import DEFAULT_SOURCE_CONFIG, load_sources_from_json, run_live_pf


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
        "--no-live",
        action="store_true",
        help="Disable interactive updating (still saves results/result_pf.png and results/result_spectrum.png).",
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
        default=0.1,
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
        default=10,
        help="Consecutive detections required to enable an isotope.",
    )
    parser.add_argument(
        "--detect-min-steps",
        type=int,
        default=None,
        help="Minimum steps before locking detected isotopes (defaults to detect_consecutive).",
    )
    args = parser.parse_args()
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
        sources=sources,
        detect_threshold_abs=args.detect_threshold_abs,
        detect_threshold_rel=args.detect_threshold_rel,
        detect_consecutive=args.detect_consecutive,
        detect_min_steps=args.detect_min_steps,
        ig_threshold_mode=args.ig_threshold_mode,
        ig_threshold_rel=args.ig_threshold_rel,
        ig_threshold_min=args.ig_threshold_min,
    )


if __name__ == "__main__":
    main()
