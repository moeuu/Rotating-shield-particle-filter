"""Analyze PF run logs for shield attenuation comparison metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

ISOTOPES = ("Co-60", "Cs-137", "Eu-154")


def parse_count_map(text: str) -> dict[str, float]:
    """Parse a compact ``{name:value}`` count map from a log line."""
    values: dict[str, float] = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        values[key.strip()] = float(raw_value.strip())
    return values


def count_entropy(counts: dict[str, float]) -> float:
    """Return normalized isotope count entropy in [0, 1]."""
    vector = np.asarray([max(float(counts.get(iso, 0.0)), 0.0) for iso in ISOTOPES], dtype=float)
    total = float(np.sum(vector))
    if total <= 0.0:
        return 0.0
    probabilities = vector / total
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(entropy / math.log(len(ISOTOPES)))


def dominant_fraction(counts: dict[str, float]) -> float:
    """Return the largest isotope fraction in one decomposed count vector."""
    vector = np.asarray([max(float(counts.get(iso, 0.0)), 0.0) for iso in ISOTOPES], dtype=float)
    total = float(np.sum(vector))
    if total <= 0.0:
        return 0.0
    return float(np.max(vector) / total)


def parse_pose_key(text: str) -> tuple[float, float, float] | None:
    """Parse a rounded detector pose key from a step line."""
    match = re.search(r"pose=\[([^\]]+)\]", text)
    if match is None:
        return None
    values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", match.group(1))]
    if len(values) != 3:
        return None
    return tuple(round(value, 3) for value in values)


def pairwise_log_signature_separation(vectors: list[dict[str, float]]) -> float:
    """Return mean pairwise log-count separation for a shield posture set."""
    if len(vectors) < 2:
        return 0.0
    rows = np.asarray(
        [[max(float(vector.get(iso, 0.0)), 0.0) for iso in ISOTOPES] for vector in vectors],
        dtype=float,
    )
    rows = np.log1p(rows)
    distances: list[float] = []
    for index in range(len(rows)):
        for other in range(index + 1, len(rows)):
            distances.append(float(np.linalg.norm(rows[index] - rows[other])))
    return float(np.mean(distances)) if distances else 0.0


def parse_final_metrics(lines: list[str]) -> dict[str, dict[str, float]]:
    """Parse final PF localization and cardinality metrics from log lines."""
    metrics: dict[str, dict[str, float]] = {}
    current: str | None = None
    for line in lines:
        isotope_match = re.search(r"\[Isotope: ([^\]]+)\]", line)
        if isotope_match is not None:
            current = isotope_match.group(1).strip()
            metrics[current] = {}
            continue
        if current is None:
            continue
        cardinality = re.search(r"GT=(\d+), EST=(\d+), Assigned=(\d+), TP=(\d+), FP=(\d+), FN=(\d+)", line)
        if cardinality is not None:
            names = ("gt", "est", "assigned", "tp", "fp", "fn")
            for name, value in zip(names, cardinality.groups(), strict=True):
                metrics[current][name] = float(value)
            continue
        position = re.search(r"Position error \[m\]: mean=([0-9.eE+-]+)", line)
        if position is not None:
            metrics[current]["position_error_mean_m"] = float(position.group(1))
            continue
        intensity = re.search(r"Intensity rel error \[%\]: mean=([0-9.eE+-]+)", line)
        if intensity is not None:
            metrics[current]["intensity_rel_error_mean_pct"] = float(intensity.group(1))
    return metrics


def analyze_log(path: Path) -> dict[str, Any]:
    """Analyze a PF run log and return comparison-ready metrics."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    counts_by_step: dict[int, dict[str, float]] = {}
    pose_by_step: dict[int, tuple[float, float, float]] = {}
    planner_signature: list[float] = []
    planner_signature_utility: list[float] = []
    live_times: list[float] = []
    total_spectrum_counts: list[float] = []
    for line in lines:
        decomp_match = re.search(
            r"\[step (\d+)\] geant4_decomposition .*?total_spectrum_counts=([0-9.eE+-]+) "
            r"response_poisson=\{([^}]*)\}",
            line,
        )
        if decomp_match is not None:
            step = int(decomp_match.group(1))
            total_spectrum_counts.append(float(decomp_match.group(2)))
            counts_by_step[step] = parse_count_map(decomp_match.group(3))
            continue
        pose_match = re.search(r"\[step (\d+)\] pose=", line)
        if pose_match is not None:
            step = int(pose_match.group(1))
            pose_key = parse_pose_key(line)
            if pose_key is not None:
                pose_by_step[step] = pose_key
            signature = re.search(r" signature=([0-9.eE+-]+)", line)
            if signature is not None:
                planner_signature.append(float(signature.group(1)))
            signature_utility = re.search(r" signature_utility=([0-9.eE+-]+)", line)
            if signature_utility is not None:
                planner_signature_utility.append(float(signature_utility.group(1)))
            live_time = re.search(r" live_time_s=([0-9.eE+-]+)", line)
            if live_time is not None:
                live_times.append(float(live_time.group(1)))
    counts = [counts_by_step[step] for step in sorted(counts_by_step)]
    pose_vectors: dict[tuple[float, float, float], list[dict[str, float]]] = {}
    for step, vector in counts_by_step.items():
        pose_key = pose_by_step.get(step)
        if pose_key is None:
            continue
        pose_vectors.setdefault(pose_key, []).append(vector)
    observed_separations = [
        pairwise_log_signature_separation(vectors)
        for vectors in pose_vectors.values()
        if len(vectors) >= 2
    ]
    response_sums = {
        isotope: float(sum(vector.get(isotope, 0.0) for vector in counts))
        for isotope in ISOTOPES
    }
    entropies = [count_entropy(vector) for vector in counts]
    dominant_fractions = [dominant_fraction(vector) for vector in counts]
    final_metrics = parse_final_metrics(lines)
    return {
        "log_path": str(path),
        "steps": len(counts),
        "poses_with_multiple_postures": len(observed_separations),
        "response_poisson_sums": response_sums,
        "response_poisson_balance_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "response_poisson_balance_entropy_min": float(np.min(entropies)) if entropies else 0.0,
        "response_poisson_dominant_fraction_mean": float(np.mean(dominant_fractions)) if dominant_fractions else 0.0,
        "response_poisson_dominant_fraction_max": float(np.max(dominant_fractions)) if dominant_fractions else 0.0,
        "observed_shield_signature_log_l2_mean": (
            float(np.mean(observed_separations)) if observed_separations else 0.0
        ),
        "observed_shield_signature_log_l2_min": (
            float(np.min(observed_separations)) if observed_separations else 0.0
        ),
        "planner_signature_mean": float(np.mean(planner_signature)) if planner_signature else 0.0,
        "planner_signature_utility_mean": (
            float(np.mean(planner_signature_utility)) if planner_signature_utility else 0.0
        ),
        "live_time_total_s": float(np.sum(live_times)) if live_times else 0.0,
        "live_time_mean_s": float(np.mean(live_times)) if live_times else 0.0,
        "total_spectrum_counts_sum": float(np.sum(total_spectrum_counts)) if total_spectrum_counts else 0.0,
        "final_pf_metrics": final_metrics,
    }


def main() -> None:
    """Parse CLI arguments and print log-analysis metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="PF run log paths.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact table.")
    args = parser.parse_args()
    results = [analyze_log(path) for path in args.logs]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    for result in results:
        final = result["final_pf_metrics"]
        print(Path(result["log_path"]).name)
        print(
            "  "
            f"steps={result['steps']} live={result['live_time_total_s']:.1f}s "
            f"balance_entropy_mean={result['response_poisson_balance_entropy_mean']:.3f} "
            f"dominant_fraction_mean={result['response_poisson_dominant_fraction_mean']:.3f} "
            f"shield_signature_log_l2_mean={result['observed_shield_signature_log_l2_mean']:.3f}"
        )
        for isotope in ISOTOPES:
            isotope_metrics = final.get(isotope, {})
            position = isotope_metrics.get("position_error_mean_m", float("nan"))
            fp = isotope_metrics.get("fp", float("nan"))
            fn = isotope_metrics.get("fn", float("nan"))
            print(f"  {isotope}: pos_err={position:.3f}m FP={fp:.0f} FN={fn:.0f}")


if __name__ == "__main__":
    main()
