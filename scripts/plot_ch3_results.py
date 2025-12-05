"""Plot Chapter 3 experiment logs (error vs. scenario, shield vs. no-shield)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_logs(path: Path) -> List[Dict[str, float]]:
    logs: List[Dict[str, float]] = []
    for file in sorted(path.glob("*.jsonl")):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
    return logs


def _aggregate(logs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by_scn: Dict[str, List[Dict[str, float]]] = {}
    for entry in logs:
        by_scn.setdefault(entry["scenario"], []).append(entry)
    summary: Dict[str, Dict[str, float]] = {}
    for scn, values in by_scn.items():
        pos_err = np.array([v["position_error"] for v in values], dtype=float)
        str_err = np.array([v["strength_error"] for v in values], dtype=float)
        iso_acc = np.array([v["iso_accuracy"] for v in values], dtype=float)
        summary[scn] = {
            "pos_mean": float(np.nanmean(pos_err)),
            "pos_q50": float(np.nanmedian(pos_err)),
            "str_mean": float(np.nanmean(str_err)),
            "iso_mean": float(np.nanmean(iso_acc)),
        }
    return summary


def plot_summary(summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    scenarios = list(summary.keys())
    pos = [summary[s]["pos_mean"] for s in scenarios]
    iso = [summary[s]["iso_mean"] for s in scenarios]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(scenarios, pos, color="#4C72B0")
    ax[0].set_ylabel("Mean position error (m)")
    ax[0].set_title("Position error vs scenario")
    ax[0].set_xticklabels(scenarios, rotation=20, ha="right")

    ax[1].bar(scenarios, iso, color="#55A868")
    ax[1].set_ylabel("Isotope ID accuracy")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Isotope accuracy vs scenario")
    ax[1].set_xticklabels(scenarios, rotation=20, ha="right")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ch3_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chapter 3 experiment results.")
    parser.add_argument("--logdir", type=Path, default=Path("results/ch3_experiments"), help="Directory with JSONL logs.")
    parser.add_argument("--output", type=Path, default=Path("results/ch3_experiments"), help="Directory for plots.")
    parser.add_argument(
        "--ig-threshold", type=float, default=1e-3, help="Information gain threshold used during convergence."
    )
    parser.add_argument(
        "--credible-volume-threshold",
        type=float,
        default=1e-3,
        help="Credible region volume threshold used during convergence.",
    )
    args = parser.parse_args()
    logs = _load_logs(args.logdir)
    if not logs:
        raise SystemExit(f"No logs found under {args.logdir}")
    summary = _aggregate(logs)
    plot_summary(summary, args.output)
    print(
        f"Convergence thresholds: IG<{args.ig_threshold}, credible_volume<{args.credible_volume_threshold} "
        "(configure in RotatingShieldPFConfig for runs)."
    )


if __name__ == "__main__":
    main()
