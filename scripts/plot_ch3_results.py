"""Plot Chapter 3 experiment logs (error vs. scenario, shield vs. no-shield)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_logs(path: Path) -> List[Dict[str, float]]:
    """Load all JSONL experiment logs under a directory."""
    logs: List[Dict[str, float]] = []
    for file in sorted(path.glob("*.jsonl")):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
    return logs


def _numeric_array(values: List[object]) -> np.ndarray:
    """Convert a list of scalars to a numeric array with NaN for missing values."""
    cleaned: List[float] = []
    for value in values:
        if value is None:
            cleaned.append(np.nan)
            continue
        cleaned.append(float(value))
    return np.asarray(cleaned, dtype=float)


def _nanquantile(values: np.ndarray, q: float) -> float:
    """Return a quantile while tolerating empty/all-NaN arrays."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _aggregate(logs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate Monte Carlo logs into scenario-level summary statistics."""
    by_scn: Dict[str, List[Dict[str, float]]] = {}
    for entry in logs:
        by_scn.setdefault(entry["scenario"], []).append(entry)
    summary: Dict[str, Dict[str, float]] = {}
    for scn, values in by_scn.items():
        pos_err = _numeric_array([v.get("position_error") for v in values])
        str_err = _numeric_array([v.get("strength_error") for v in values])
        iso_acc = _numeric_array([v.get("iso_accuracy") for v in values])
        success = _numeric_array([v.get("trial_success", False) for v in values])
        pos_target = _numeric_array([v.get("position_within_target", False) for v in values])
        fp_count = _numeric_array([v.get("fp_count", np.nan) for v in values])
        fn_count = _numeric_array([v.get("fn_count", np.nan) for v in values])
        uncertainty = _numeric_array([v.get("global_uncertainty", np.nan) for v in values])
        mismatch_labels = sorted({str(v.get("mismatch_label", "matched")) for v in values})
        estimator_mu = dict(values[0].get("estimator_mu_by_isotope", {}))
        observation_mu = dict(values[0].get("observation_mu_by_isotope", {}))
        summary[scn] = {
            "num_trials": int(len(values)),
            "position_mean": float(np.nanmean(pos_err)),
            "position_q25": _nanquantile(pos_err, 0.25),
            "position_q50": _nanquantile(pos_err, 0.50),
            "position_q75": _nanquantile(pos_err, 0.75),
            "position_iqr": _nanquantile(pos_err, 0.75) - _nanquantile(pos_err, 0.25),
            "position_target_m": float(values[0].get("position_target_m", np.nan)),
            "position_within_target_rate": float(np.nanmean(pos_target)),
            "trial_success_rate": float(np.nanmean(success)),
            "strength_mean": float(np.nanmean(str_err)),
            "strength_q50": _nanquantile(str_err, 0.50),
            "iso_mean": float(np.nanmean(iso_acc)),
            "fp_mean": float(np.nanmean(fp_count)),
            "fn_mean": float(np.nanmean(fn_count)),
            "uncertainty_q50": _nanquantile(uncertainty, 0.50),
            "mismatch_label": ",".join(mismatch_labels),
            "estimator_mu_by_isotope": estimator_mu,
            "observation_mu_by_isotope": observation_mu,
        }
    return summary


def plot_summary(summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Plot the scenario summary using robust Monte Carlo statistics."""
    scenarios = list(summary.keys())
    x = np.arange(len(scenarios))
    pos_q50 = np.asarray([summary[s]["position_q50"] for s in scenarios], dtype=float)
    pos_q25 = np.asarray([summary[s]["position_q25"] for s in scenarios], dtype=float)
    pos_q75 = np.asarray([summary[s]["position_q75"] for s in scenarios], dtype=float)
    pos_yerr = np.vstack((pos_q50 - pos_q25, pos_q75 - pos_q50))
    success = np.asarray([summary[s]["trial_success_rate"] for s in scenarios], dtype=float)
    iso = np.asarray([summary[s]["iso_mean"] for s in scenarios], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(x, pos_q50, color="#4C72B0", yerr=pos_yerr, capsize=4)
    axes[0].set_ylabel("Median position error (m)")
    axes[0].set_title("Position error (IQR)")

    axes[1].bar(x, success, color="#DD8452")
    axes[1].set_ylabel("Trial success rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Localization success")

    axes[2].bar(x, iso, color="#55A868")
    axes[2].set_ylabel("Within-radius isotope accuracy")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Isotope accuracy")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios, rotation=20, ha="right")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ch3_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def save_summary_json(summary: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    """Persist the aggregated Monte Carlo summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ch3_summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return out_path


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
    summary_path = save_summary_json(summary, args.output)
    print(f"Saved summary to {summary_path}")
    print(
        f"Convergence thresholds: IG<{args.ig_threshold}, credible_volume<{args.credible_volume_threshold} "
        "(configure in RotatingShieldPFConfig for runs)."
    )


if __name__ == "__main__":
    main()
