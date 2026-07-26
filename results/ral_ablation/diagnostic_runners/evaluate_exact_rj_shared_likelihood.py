"""Evaluate completed exact-RJ reports under one shared PF likelihood.

The diagnostic is read-only. It reconstructs the immutable count-domain
measurement bundle, evaluates continuous truth, feasible nearest-patch truth,
and saved posterior report modes, then writes JSON only to stdout. Saved modes
are posterior summaries rather than a joint particle checkpoint, so their
likelihood is explicitly labelled as a report-point diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from pf.particle_filter import MeasurementData
from pf.pure_estimator import _ordered_surface_dictionary_sha256
from pf.replay import build_replay_estimator
from pf.state import IsotopeState
from runtime.measurement_log import MeasurementLog, load_measurement_log


def _parse_args() -> argparse.Namespace:
    """Parse immutable replay and truth inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-posterior", required=True, type=Path)
    parser.add_argument("--baseline-posterior", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a file's byte-level SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _counterfactual_log(log: MeasurementLog) -> MeasurementLog:
    """Remove the historical estimator block without changing observations."""
    runtime_config = dict(log.runtime_config)
    removed = runtime_config.pop("effective_pf_replay", None)
    if removed is None:
        raise RuntimeError("Measurement log lacks effective_pf_replay.")
    return replace(log, runtime_config=runtime_config)


def _covariance_mapping(
    dense: np.ndarray,
    isotopes: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Convert a dense isotope covariance to the estimator mapping contract."""
    covariance = np.asarray(dense, dtype=float)
    expected = (len(isotopes), len(isotopes))
    if covariance.shape != expected:
        raise ValueError(f"Expected isotope covariance shape {expected}.")
    return {
        row_isotope: {
            column_isotope: float(covariance[row_index, column_index])
            for column_index, column_isotope in enumerate(isotopes)
        }
        for row_index, row_isotope in enumerate(isotopes)
    }


def _measurement_data(
    estimator: Any,
    log: MeasurementLog,
    isotope: str,
) -> tuple[MeasurementData, dict[str, Any]]:
    """Reconstruct the runtime count-covariance history for one isotope."""
    isotopes = tuple(str(value) for value in log.run_manifest["isotopes"])
    projected_variances: list[float] = []
    observed: list[float] = []
    for record in log.records:
        if record.isotope_counts is None:
            raise RuntimeError("Every record must contain isotope counts.")
        counts = {
            name: float(record.isotope_counts[name])
            for name in isotopes
        }
        if record.isotope_count_covariance is None:
            raw_variances = {
                name: max(float(counts[name]), 1.0)
                for name in isotopes
            }
            covariance = None
        else:
            dense = np.asarray(record.isotope_count_covariance, dtype=float)
            covariance = _covariance_mapping(dense, isotopes)
            raw_variances = {
                name: float(dense[index, index])
                for index, name in enumerate(isotopes)
            }
        effective, _ = estimator._project_observation_covariance_to_variance(
            counts,
            raw_variances,
            covariance,
        )
        if effective is None:
            raise RuntimeError("Observation covariance projection returned None.")
        observed.append(float(counts[isotope]))
        projected_variances.append(float(effective[isotope]))

    station_ids = np.asarray(
        [int(record.station_id) for record in log.records],
        dtype=np.int64,
    )
    filt = estimator.filters[isotope]
    routes = np.empty(len(log.records), dtype="<U32")
    route_by_station: dict[str, str] = {}
    for station_id in np.unique(station_ids):
        rows = np.flatnonzero(station_ids == int(station_id))
        route = (
            "count_covariance"
            if filt._sequence_covariance_enabled(int(rows.size), None)
            else "count"
        )
        routes[rows] = route
        route_by_station[str(int(station_id))] = route
    data = MeasurementData(
        z_k=np.asarray(observed, dtype=float),
        observation_variances=np.asarray(projected_variances, dtype=float),
        detector_positions=np.asarray(
            [record.detector_pose_xyz for record in log.records],
            dtype=float,
        ),
        fe_indices=np.asarray(
            [record.fe_orientation_index for record in log.records],
            dtype=np.int64,
        ),
        pb_indices=np.asarray(
            [record.pb_orientation_index for record in log.records],
            dtype=np.int64,
        ),
        live_times=np.asarray(
            [record.live_time_s for record in log.records],
            dtype=float,
        ),
        station_sequence_ids=station_ids,
        runtime_likelihood_routes=routes,
        observation_count_covariance=None,
    )
    return data, {
        "route_by_station": route_by_station,
        "unique_routes": sorted(set(routes.tolist())),
        "projected_observation_variance_min": float(
            np.min(data.observation_variances)
        ),
        "projected_observation_variance_median": float(
            np.median(data.observation_variances)
        ),
        "projected_observation_variance_max": float(
            np.max(data.observation_variances)
        ),
    }


def _state_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    background_cps: float,
    position_key: str,
    strength_key: str,
) -> IsotopeState:
    """Build one evaluation state without fitting any parameter."""
    positions = np.asarray(
        [row[position_key] for row in rows],
        dtype=float,
    ).reshape(-1, 3)
    strengths = np.asarray(
        [row[strength_key] for row in rows],
        dtype=float,
    ).reshape(-1)
    if positions.shape[0] != strengths.size:
        raise ValueError("Position and strength counts differ.")
    return IsotopeState(
        num_sources=int(strengths.size),
        positions=positions,
        strengths=strengths,
        background=float(background_cps),
    )


def _evaluate_state(
    filt: Any,
    data: MeasurementData,
    state: IsotopeState,
) -> dict[str, Any]:
    """Evaluate one fixed state under the exact structural likelihood."""
    _, expected = filt._lambda_components(state, data)
    observed = np.asarray(data.z_k, dtype=float)
    residual = observed - expected
    correlation = (
        float(np.corrcoef(expected, observed)[0, 1])
        if expected.size > 1
        and float(np.std(expected)) > 0.0
        and float(np.std(observed)) > 0.0
        else None
    )
    return {
        "source_count": int(state.num_sources),
        "background_cps": float(state.background),
        "total_strength_cps_1m": float(np.sum(state.strengths)),
        "shared_structural_log_likelihood": float(
            filt._structural_count_log_likelihood_np(data, expected)
        ),
        "predicted_to_observed_total_ratio": float(
            np.sum(expected) / max(float(np.sum(observed)), 1.0e-300)
        ),
        "predicted_observed_correlation": correlation,
        "residual_rmse_counts": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _nearest_distinct_patches(
    truth_rows: Sequence[Mapping[str, Any]],
    centers_xyz: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Snap truth positions to a minimum-distance distinct patch assignment."""
    truth_positions = np.asarray(
        [row["position"] for row in truth_rows],
        dtype=float,
    ).reshape(-1, 3)
    centers = np.asarray(centers_xyz, dtype=float)
    distances = np.linalg.norm(
        truth_positions[:, None, :] - centers[None, :, :],
        axis=2,
    )
    truth_indices, patch_indices = linear_sum_assignment(distances)
    if len(truth_indices) != len(truth_rows):
        raise RuntimeError("Failed to assign every truth source to a patch.")
    patch_by_truth = {
        int(truth_index): int(patch_index)
        for truth_index, patch_index in zip(
            truth_indices,
            patch_indices,
            strict=True,
        )
    }
    snapped_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for truth_index, row in enumerate(truth_rows):
        patch_index = patch_by_truth[truth_index]
        snapped_rows.append(
            {
                "position": centers[patch_index].tolist(),
                "intensity_cps_1m": float(row["intensity_cps_1m"]),
            }
        )
        assignments.append(
            {
                "truth_index": int(truth_index),
                "patch_index": int(patch_index),
                "distance_m": float(distances[truth_index, patch_index]),
                "truth_position_xyz": [
                    float(value) for value in truth_positions[truth_index]
                ],
                "patch_position_xyz": [
                    float(value) for value in centers[patch_index]
                ],
            }
        )
    return snapped_rows, assignments


def _report_rows(posterior: Mapping[str, Any], isotope: str) -> list[dict[str, Any]]:
    """Extract fixed mean-valued report modes for one isotope."""
    isotope_payload = posterior["isotopes"][isotope]
    modes = isotope_payload["modes"]
    if len(modes) != int(isotope_payload["map_cardinality"]):
        raise ValueError(f"{isotope} report modes do not match MAP cardinality.")
    return [
        {
            "position": mode["position_mean_xyz"],
            "intensity_cps_1m": mode["strength_mean_cps_1m"],
            "report_mode_index": int(mode_index),
        }
        for mode_index, mode in enumerate(modes)
    ]


def _truth_matched_report_subset(
    truth_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any]]:
    """Drop unmatched report modes using only deterministic spatial assignment."""
    if len(report_rows) < len(truth_rows):
        return None, {
            "available": False,
            "reason": "report_cardinality_below_truth",
        }
    truth_positions = np.asarray(
        [row["position"] for row in truth_rows],
        dtype=float,
    )
    report_positions = np.asarray(
        [row["position"] for row in report_rows],
        dtype=float,
    )
    distances = np.linalg.norm(
        truth_positions[:, None, :] - report_positions[None, :, :],
        axis=2,
    )
    truth_indices, report_indices = linear_sum_assignment(distances)
    selected = sorted(int(index) for index in report_indices)
    return [report_rows[index] for index in selected], {
        "available": True,
        "selection_semantics": (
            "minimum-distance truth assignment; no likelihood optimization "
            "and no strength refit"
        ),
        "selected_report_mode_indices": selected,
        "dropped_report_mode_indices": sorted(
            set(range(len(report_rows))) - set(selected)
        ),
        "assignment": [
            {
                "truth_index": int(truth_index),
                "report_mode_index": int(report_index),
                "distance_m": float(distances[truth_index, report_index]),
            }
            for truth_index, report_index in zip(
                truth_indices,
                report_indices,
                strict=True,
            )
        ],
    }


def _isotope_audit(
    estimator: Any,
    log: MeasurementLog,
    isotope: str,
    truth_rows: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate truth and both saved report points for one isotope."""
    filt = estimator.filters[isotope]
    data, data_manifest = _measurement_data(estimator, log, isotope)
    background = float(filt._background_level())
    patches = filt._structural_rj_surface_patches
    if patches is None:
        raise RuntimeError("Exact-RJ surface dictionary is unavailable.")
    snapped_rows, patch_assignments = _nearest_distinct_patches(
        truth_rows,
        patches.centers_xyz,
    )
    candidate_rows = _report_rows(candidate, isotope)
    baseline_rows = _report_rows(baseline, isotope)
    candidate_subset, candidate_subset_manifest = _truth_matched_report_subset(
        truth_rows,
        candidate_rows,
    )
    baseline_subset, baseline_subset_manifest = _truth_matched_report_subset(
        truth_rows,
        baseline_rows,
    )
    state_rows: dict[str, Sequence[Mapping[str, Any]]] = {
        "continuous_truth": truth_rows,
        "nearest_patch_truth": snapped_rows,
        "candidate_report_point": candidate_rows,
        "baseline_report_point": baseline_rows,
    }
    if candidate_subset is not None:
        state_rows["candidate_truth_matched_subset"] = candidate_subset
    if baseline_subset is not None:
        state_rows["baseline_truth_matched_subset"] = baseline_subset
    states = {
        label: _evaluate_state(
            filt,
            data,
            _state_from_rows(
                rows,
                background_cps=background,
                position_key="position",
                strength_key="intensity_cps_1m",
            ),
        )
        for label, rows in state_rows.items()
    }
    continuous_ll = states["continuous_truth"][
        "shared_structural_log_likelihood"
    ]
    snapped_ll = states["nearest_patch_truth"][
        "shared_structural_log_likelihood"
    ]
    candidate_ll = states["candidate_report_point"][
        "shared_structural_log_likelihood"
    ]
    baseline_ll = states["baseline_report_point"][
        "shared_structural_log_likelihood"
    ]
    deltas: dict[str, float] = {
        "continuous_truth_minus_nearest_patch_truth": float(
            continuous_ll - snapped_ll
        ),
        "nearest_patch_truth_minus_candidate_report_point": float(
            snapped_ll - candidate_ll
        ),
        "nearest_patch_truth_minus_baseline_report_point": float(
            snapped_ll - baseline_ll
        ),
        "candidate_report_point_minus_baseline_report_point": float(
            candidate_ll - baseline_ll
        ),
    }
    if candidate_subset is not None:
        deltas[
            "candidate_full_minus_truth_matched_subset"
        ] = float(
            candidate_ll
            - states["candidate_truth_matched_subset"][
                "shared_structural_log_likelihood"
            ]
        )
    if baseline_subset is not None:
        deltas[
            "baseline_full_minus_truth_matched_subset"
        ] = float(
            baseline_ll
            - states["baseline_truth_matched_subset"][
                "shared_structural_log_likelihood"
            ]
        )
    return {
        "measurement_data": data_manifest,
        "nearest_patch_truth_assignment": patch_assignments,
        "states": states,
        "log_likelihood_deltas": deltas,
        "candidate_truth_matched_subset": candidate_subset_manifest,
        "baseline_truth_matched_subset": baseline_subset_manifest,
    }


def main() -> int:
    """Print the shared-likelihood audit as canonical human-readable JSON."""
    args = _parse_args()
    original_log = load_measurement_log(args.measurement_log)
    config = _load_object(args.config)
    configured_use_gpu = bool(config.get("use_gpu", False))
    evaluation_config = dict(config)
    evaluation_config["use_gpu"] = False
    replay_log = _counterfactual_log(original_log)
    estimator = build_replay_estimator(
        replay_log,
        evaluation_config,
        profile="pf_strict",
        seed=int(args.seed),
    )
    estimator._ensure_kernel_cache()
    candidate = _load_object(args.candidate_posterior)
    baseline = _load_object(args.baseline_posterior)
    truth_payload = _load_object(args.truth)
    truth_sources = truth_payload["sources"]
    isotopes = tuple(str(value) for value in original_log.run_manifest["isotopes"])

    patch_manifests: dict[str, Any] = {}
    for isotope in isotopes:
        patches = estimator.filters[isotope]._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError(f"{isotope} lacks an exact-RJ patch dictionary.")
        patch_manifests[isotope] = {
            "patch_count": int(patches.patch_count),
            "total_area_m2": float(np.sum(patches.areas_m2)),
            "ordered_centers_areas_sha256": _ordered_surface_dictionary_sha256(
                patches.centers_xyz,
                patches.areas_m2,
            ),
        }

    payload = {
        "schema_version": 1,
        "diagnostic": "completed_exact_rj_shared_likelihood_report_audit",
        "input_sha256": {
            "measurement_log": original_log.log_sha256,
            "config_file": _sha256(args.config),
            "candidate_posterior": _sha256(args.candidate_posterior),
            "baseline_posterior": _sha256(args.baseline_posterior),
            "truth": _sha256(args.truth),
        },
        "method": {
            "truth_read_after_replay_completion": True,
            "likelihood": (
                "PF structural likelihood with the reconstructed runtime "
                "count/count-covariance route, projected isotope extraction "
                "variance, configured "
                "transport uncertainty, station-view common-mode covariance, "
                "and configured shield-shape terms"
            ),
            "response": (
                "shared ContinuousKernel with spherical-octant Fe/Pb geometry, "
                "obstacle attenuation, detector aperture, and logged response scale"
            ),
            "background": "configured fixed per-isotope PF background",
            "parameter_fitting": "none",
            "report_point_warning": (
                "Saved modes are posterior marginal summaries, not a jointly "
                "sampled particle or maximum-likelihood checkpoint."
            ),
            "compute_backend": {
                "replay_config_use_gpu": configured_use_gpu,
                "evaluation_use_gpu": False,
                "reason": (
                    "CPU float64 evaluation avoids interference from an unrelated "
                    "GPU process; it changes scheduling only, not model semantics."
                ),
            },
        },
        "surface_patch_dictionary": patch_manifests,
        "isotopes": {
            isotope: _isotope_audit(
                estimator,
                original_log,
                isotope,
                [
                    source
                    for source in truth_sources
                    if source["isotope"] == isotope
                ],
                candidate,
                baseline,
            )
            for isotope in isotopes
        },
    }
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
