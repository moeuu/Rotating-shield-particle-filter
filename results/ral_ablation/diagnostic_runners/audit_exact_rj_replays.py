"""Audit an exact-RJ replay without mutating any replay artifact.

During a replay, pass ``--progress-log`` to validate station health without
loading truth. After completion, also pass ``--baseline-dir`` and ``--truth``
to print a paired comparison using the same location assignment and threshold
definitions for both replays. The script writes JSON only to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


ISOTOPES = ("Co-60", "Cs-137", "Eu-154")
THRESHOLDS_M = (0.5, 1.0, 2.0, 3.0)


def _parse_args() -> argparse.Namespace:
    """Parse truth-free health and optional completed-replay comparison inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--heuristic-posterior", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--exit-file", type=Path)
    parser.add_argument("--expected-stations", type=int, default=20)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return the byte-level SHA-256 of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a complete JSONL file as objects."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _load_progress_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load only complete station rows from a possibly growing replay log."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last_partial = line_number == len(lines) and not text.endswith("\n")
            if not is_last_partial:
                issues.append(f"invalid_json_line_{line_number}:{exc.msg}")
            continue
        if (
            isinstance(payload, dict)
            and "station_id" in payload
            and "structural_timing_s" in payload
        ):
            rows.append(payload)
    return rows, issues


def _nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    """Return paths of every non-finite float in a nested JSON-like value."""
    if isinstance(value, float):
        return [] if math.isfinite(value) else [prefix]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_nonfinite_paths(child, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_nonfinite_paths(child, f"{prefix}[{index}]"))
        return paths
    return []


def _metric(
    timing: Mapping[str, Any],
    primary: str,
    fallback: str | None = None,
) -> float | None:
    """Return a numeric timing metric with an optional legacy fallback name."""
    value = timing.get(primary)
    if value is None and fallback is not None:
        value = timing.get(fallback)
    return None if value is None else float(value)


def _proposal_summary(
    timing: Mapping[str, Any],
    *,
    prefix: str,
    fallback_prefix: str | None = None,
) -> dict[str, float | None]:
    """Return attempted, accepted, and rate fields for one proposal family."""
    accepted = _metric(
        timing,
        f"{prefix}_accepted",
        None if fallback_prefix is None else f"{fallback_prefix}_accepted",
    )
    attempted = _metric(
        timing,
        f"{prefix}_attempted",
        None if fallback_prefix is None else f"{fallback_prefix}_attempted",
    )
    rate = None
    if accepted is not None and attempted not in (None, 0.0):
        rate = accepted / attempted
    return {
        "attempted": attempted,
        "accepted": accepted,
        "acceptance_rate": rate,
    }


def _audit_station_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_stations: int,
) -> dict[str, Any]:
    """Audit causal station rows and summarize proposal/weight diagnostics."""
    materialized = list(rows)
    issues: list[str] = []
    nonfinite = _nonfinite_paths(materialized)
    if nonfinite:
        issues.extend(f"nonfinite:{path}" for path in nonfinite)
    station_ids = [int(row["station_id"]) for row in materialized]
    if station_ids != list(range(len(materialized))):
        issues.append(f"noncontiguous_station_ids:{station_ids}")
    isotopes: dict[str, Any] = {}
    for isotope in ISOTOPES:
        station_payloads: list[dict[str, Any]] = []
        for row in materialized:
            raw_timing = row.get("structural_timing_s", {}).get(isotope)
            if not isinstance(raw_timing, Mapping):
                issues.append(f"missing_timing:{isotope}:station_{row['station_id']}")
                continue
            proposals = {
                "birth": _proposal_summary(
                    raw_timing,
                    prefix="rj_birth",
                ),
                "death": _proposal_summary(
                    raw_timing,
                    prefix="rj_death",
                ),
                "global_position": _proposal_summary(
                    raw_timing,
                    prefix="rj_global_position",
                    fallback_prefix="rj_position",
                ),
                "local_position": _proposal_summary(
                    raw_timing,
                    prefix="rj_local_position",
                ),
                "strength": _proposal_summary(
                    raw_timing,
                    prefix="rj_strength",
                ),
            }
            for proposal_name, proposal in proposals.items():
                accepted = proposal["accepted"]
                attempted = proposal["attempted"]
                if accepted is not None and accepted < 0.0:
                    issues.append(
                        f"negative_accepted:{isotope}:{row['station_id']}:"
                        f"{proposal_name}"
                    )
                if attempted is not None and attempted < 0.0:
                    issues.append(
                        f"negative_attempted:{isotope}:{row['station_id']}:"
                        f"{proposal_name}"
                    )
                if (
                    accepted is not None
                    and attempted is not None
                    and accepted > attempted
                ):
                    issues.append(
                        f"accepted_exceeds_attempted:{isotope}:"
                        f"{row['station_id']}:{proposal_name}"
                    )
            max_weight_diff = _metric(
                raw_timing,
                "outer_log_weight_max_abs_diff",
            )
            array_equal = _metric(
                raw_timing,
                "outer_log_weight_array_equal",
            )
            weights_preserved = _metric(raw_timing, "weights_preserved")
            if max_weight_diff not in (None, 0.0):
                issues.append(
                    f"nonzero_weight_diff:{isotope}:{row['station_id']}:"
                    f"{max_weight_diff}"
                )
            if array_equal not in (None, 1.0):
                issues.append(
                    f"weight_array_not_equal:{isotope}:{row['station_id']}:"
                    f"{array_equal}"
                )
            if weights_preserved != 1.0:
                issues.append(
                    f"weights_not_preserved:{isotope}:{row['station_id']}:"
                    f"{weights_preserved}"
                )
            station_payloads.append(
                {
                    "station_id": int(row["station_id"]),
                    "record_index": int(row["record_index"]),
                    "proposals": proposals,
                    "outer_log_weight_max_abs_diff": max_weight_diff,
                    "outer_log_weight_array_equal": array_equal,
                    "weights_preserved": weights_preserved,
                }
            )
        isotopes[isotope] = station_payloads
    return {
        "healthy": not issues,
        "issues": issues,
        "stations_completed": len(materialized),
        "expected_stations": int(expected_stations),
        "complete": len(materialized) == expected_stations,
        "last_station_id": None if not materialized else station_ids[-1],
        "isotopes": isotopes,
    }


def _process_status(
    pid: int | None,
    exit_file: Path | None,
) -> dict[str, Any]:
    """Return optional process liveness and persistent-session exit status."""
    alive = None
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        else:
            alive = True
    exit_code = None
    if exit_file is not None and exit_file.exists():
        exit_code = int(exit_file.read_text(encoding="utf-8").strip())
    return {
        "pid": pid,
        "alive": alive,
        "exit_file": None if exit_file is None else str(exit_file),
        "exit_code": exit_code,
    }


def _posterior_metrics(
    posterior: Mapping[str, Any],
    truth_sources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute final truth metrics with deterministic one-to-one assignment."""
    isotope_metrics: dict[str, Any] = {}
    all_distances: list[float] = []
    truth_count = 0
    estimate_count = 0
    for isotope in ISOTOPES:
        truth = [
            source
            for source in truth_sources
            if source.get("isotope") == isotope
        ]
        payload = posterior["isotopes"][isotope]
        modes = payload["modes"]
        if len(modes) != int(payload["map_cardinality"]):
            raise ValueError(f"{isotope} mode count does not equal MAP K")
        truth_xyz = np.asarray([source["position"] for source in truth], float)
        estimate_xyz = np.asarray(
            [mode["position_mean_xyz"] for mode in modes],
            float,
        )
        distances = np.linalg.norm(
            truth_xyz[:, None, :] - estimate_xyz[None, :, :],
            axis=2,
        )
        truth_indices, estimate_indices = linear_sum_assignment(distances)
        pairs: list[dict[str, Any]] = []
        assigned_distances: list[float] = []
        for truth_index, estimate_index in zip(
            truth_indices,
            estimate_indices,
            strict=True,
        ):
            distance = float(distances[truth_index, estimate_index])
            truth_strength = float(
                truth[truth_index]["intensity_cps_1m"]
            )
            estimate_strength = float(
                modes[estimate_index]["strength_mean_cps_1m"]
            )
            assigned_distances.append(distance)
            pairs.append(
                {
                    "truth_index": int(truth_index),
                    "estimate_mode_index": int(estimate_index),
                    "distance_m": distance,
                    "truth_strength_cps_1m": truth_strength,
                    "estimate_strength_cps_1m": estimate_strength,
                    "strength_relative_error": (
                        estimate_strength - truth_strength
                    )
                    / truth_strength,
                }
            )
        truth_total = sum(
            float(source["intensity_cps_1m"])
            for source in truth
        )
        estimate_total = sum(
            float(mode["strength_mean_cps_1m"])
            for mode in modes
        )
        cardinality = {
            int(key): float(value)
            for key, value in payload["cardinality_distribution"].items()
        }
        isotope_metrics[isotope] = {
            "truth_k": len(truth),
            "map_k": int(payload["map_cardinality"]),
            "p_truth_k": cardinality.get(len(truth), 0.0),
            "p_k0": cardinality.get(0, 0.0),
            "posterior_mean_k": sum(
                k * probability
                for k, probability in cardinality.items()
            ),
            "cardinality_distribution": {
                str(key): value
                for key, value in sorted(cardinality.items())
            },
            "truth_total_strength_cps_1m": truth_total,
            "map_modes_total_strength_cps_1m": estimate_total,
            "total_strength_relative_error": (
                estimate_total - truth_total
            )
            / truth_total,
            "assigned_pairs": pairs,
            "unmatched_estimate_mode_indices": sorted(
                set(range(len(modes))) - set(map(int, estimate_indices))
            ),
            "matched_distance_mean_m": float(np.mean(assigned_distances)),
            "matched_distance_rmse_m": float(
                np.sqrt(np.mean(np.square(assigned_distances)))
            ),
            "matched_distance_max_m": float(np.max(assigned_distances)),
        }
        all_distances.extend(assigned_distances)
        truth_count += len(truth)
        estimate_count += len(modes)
    threshold_metrics: dict[str, Any] = {}
    for threshold in THRESHOLDS_M:
        true_positive = sum(
            distance <= threshold
            for distance in all_distances
        )
        false_negative = truth_count - true_positive
        false_positive = estimate_count - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        threshold_metrics[str(threshold)] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "f1": 0.0 if denominator == 0 else 2 * true_positive / denominator,
        }
    return {
        "isotopes": isotope_metrics,
        "aggregate": {
            "truth_source_count": truth_count,
            "map_mode_count": estimate_count,
            "matched_distance_mean_m": float(np.mean(all_distances)),
            "matched_distance_rmse_m": float(
                np.sqrt(np.mean(np.square(all_distances)))
            ),
            "matched_distance_max_m": float(np.max(all_distances)),
            "threshold_metrics": threshold_metrics,
        },
    }


def _replay_manifest(replay_dir: Path) -> dict[str, Any]:
    """Return persisted hashes and semantic identity for one completed replay."""
    files = (
        "counterfactual_contract.json",
        "pf_diagnostics.json",
        "pf_posterior.json",
        "pf_trace.jsonl",
        "station_diagnostics.jsonl",
    )
    missing = [name for name in files if not (replay_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete replay {replay_dir}; missing {missing}"
        )
    contract = _load_json(replay_dir / "counterfactual_contract.json")
    diagnostics = _load_json(replay_dir / "pf_diagnostics.json")
    posterior = _load_json(replay_dir / "pf_posterior.json")
    trace = _load_jsonl(replay_dir / "pf_trace.jsonl")
    stations = _load_jsonl(replay_dir / "station_diagnostics.jsonl")
    return {
        "path": str(replay_dir),
        "artifact_sha256": {
            name: _sha256(replay_dir / name)
            for name in files
        },
        "contract": contract,
        "diagnostics": diagnostics,
        "posterior": posterior,
        "trace": trace,
        "stations": stations,
    }


def _completed_replay_audit(
    candidate_dir: Path,
    baseline_dir: Path,
    truth_path: Path,
    heuristic_posterior_path: Path | None,
    *,
    expected_stations: int,
) -> dict[str, Any]:
    """Return paired baseline/candidate truth metrics and integrity checks."""
    candidate = _replay_manifest(candidate_dir)
    baseline = _replay_manifest(baseline_dir)
    truth_payload = _load_json(truth_path)
    truth_sources = truth_payload["sources"]
    candidate_health = _audit_station_rows(
        candidate["stations"],
        expected_stations=expected_stations,
    )
    baseline_health = _audit_station_rows(
        baseline["stations"],
        expected_stations=expected_stations,
    )
    candidate_metrics = _posterior_metrics(
        candidate["posterior"],
        truth_sources,
    )
    baseline_metrics = _posterior_metrics(
        baseline["posterior"],
        truth_sources,
    )
    paired_checks = {
        "same_measurement_log_sha256": (
            candidate["contract"]["measurement_log_sha256"]
            == baseline["contract"]["measurement_log_sha256"]
        ),
        "same_random_seed": (
            candidate["contract"]["random_seed"]
            == baseline["contract"]["random_seed"]
        ),
        "same_record_count": (
            candidate["contract"]["record_count"]
            == baseline["contract"]["record_count"]
        ),
        "same_counterfactual_config_sha256": (
            candidate["contract"]["counterfactual_pf_config_sha256"]
            == baseline["contract"]["counterfactual_pf_config_sha256"]
        ),
        "candidate_final_state_hash_matches_trace": (
            candidate["diagnostics"]["final_state_sha256"]
            == candidate["trace"][-1]["state_sha256"]
        ),
        "baseline_final_state_hash_matches_trace": (
            baseline["diagnostics"]["final_state_sha256"]
            == baseline["trace"][-1]["state_sha256"]
        ),
    }
    heuristic = None
    if heuristic_posterior_path is not None:
        heuristic_posterior = _load_json(heuristic_posterior_path)
        heuristic = {
            "path": str(heuristic_posterior_path),
            "sha256": _sha256(heuristic_posterior_path),
            "same_measurement_log_sha256": (
                heuristic_posterior["measurement_log_sha256"]
                == candidate["contract"]["measurement_log_sha256"]
            ),
            "map_k": {
                isotope: int(
                    heuristic_posterior["isotopes"][isotope][
                        "map_cardinality"
                    ]
                )
                for isotope in ISOTOPES
            },
            "p_k0": {
                isotope: float(
                    heuristic_posterior["isotopes"][isotope][
                        "cardinality_distribution"
                    ]["0"]
                )
                for isotope in ISOTOPES
            },
        }
    return {
        "schema_version": 1,
        "method": {
            "location_matching": (
                "Minimum-total-Euclidean-distance one-to-one assignment "
                "independently per isotope"
            ),
            "thresholds_m": list(THRESHOLDS_M),
            "truth_evaluation_started_after_candidate_completion": True,
        },
        "truth": {
            "path": str(truth_path),
            "sha256": _sha256(truth_path),
        },
        "paired_checks": paired_checks,
        "candidate": {
            "artifact_sha256": candidate["artifact_sha256"],
            "health": candidate_health,
            "metrics": candidate_metrics,
        },
        "baseline": {
            "artifact_sha256": baseline["artifact_sha256"],
            "health": baseline_health,
            "metrics": baseline_metrics,
        },
        "heuristic": heuristic,
    }


def main() -> int:
    """Print a truth-free health audit or completed paired replay comparison."""
    args = _parse_args()
    output: dict[str, Any] = {
        "schema_version": 1,
        "candidate_dir": str(args.candidate_dir),
    }
    health: dict[str, Any] | None = None
    if args.progress_log is not None:
        rows, parse_issues = _load_progress_rows(args.progress_log)
        health = _audit_station_rows(
            rows,
            expected_stations=args.expected_stations,
        )
        health["issues"] = parse_issues + health["issues"]
        health["healthy"] = not health["issues"]
        output["progress_log"] = {
            "path": str(args.progress_log),
            "sha256_snapshot": _sha256(args.progress_log),
            "health": health,
        }
    output["process"] = _process_status(args.pid, args.exit_file)
    if args.baseline_dir is not None or args.truth is not None:
        if args.baseline_dir is None or args.truth is None:
            raise ValueError("--baseline-dir and --truth must be used together")
        output["completed_comparison"] = _completed_replay_audit(
            args.candidate_dir,
            args.baseline_dir,
            args.truth,
            args.heuristic_posterior,
            expected_stations=args.expected_stations,
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    if health is not None and not health["healthy"]:
        return 2
    process = output["process"]
    if process["exit_code"] not in (None, 0):
        return 3
    if (
        process["pid"] is not None
        and process["alive"] is False
        and process["exit_code"] is None
    ):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
