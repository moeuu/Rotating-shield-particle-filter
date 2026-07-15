"""Replay the current response-Poisson guard from validation records."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pf.likelihood import (
    DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS,
    DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
    count_likelihood_variance,
)
from measurement.observation_model import build_runtime_observation_model
from measurement.continuous_kernels import transport_response_factor_from_payload
from spectrum.pipeline import IsotopeCountEstimate
from spectrum.pipeline import SpectralDecomposer
from spectrum.response_truth_calibration import fit_local_knn_response_truth_calibration
from spectrum.response_truth_calibration import response_truth_calibration_scale
from spectrum.runtime_config import spectrum_config_from_runtime_config

RESPONSE_VS_TRUTH_GATE = (0.04, 0.12)
PF_TARGET_VS_TRUTH_GATE = (0.03, 0.10)
RESPONSE_VS_PF_TARGET_GATE = (0.05, 0.15)


def _float_value(value: object, default: float = float("nan")) -> float:
    """Return a finite-compatible float parsed from a CSV value."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_error(count: float, truth: float) -> float:
    """Return an absolute relative error for a positive truth count."""
    if truth <= 0.0:
        return float("nan")
    return abs(float(count) - float(truth)) / float(truth)


def _truth_relative_difference(first: float, second: float, truth: float) -> float:
    """Return a truth-normalized absolute difference between two counts."""
    if truth <= 0.0:
        return float("nan")
    return abs(float(first) - float(second)) / float(truth)


def _truth_relative_spread(
    first: float,
    second: float,
    truth: float,
) -> float:
    """Return the truth-normalized spread across truth and two counts."""
    if truth <= 0.0:
        return float("nan")
    values = np.asarray([float(first), float(second), float(truth)], dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float((np.max(values) - np.min(values)) / float(truth))


def _summary(values: list[float]) -> dict[str, float]:
    """Return standard residual summary statistics."""
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _count_summary(values: list[float]) -> dict[str, float]:
    """Return count summary statistics with common false-positive thresholds."""
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    result = _summary([float(value) for value in arr])
    for threshold in (1000.0, 5000.0, 10000.0, 20000.0):
        result[f"ge_{int(threshold)}"] = float(np.sum(arr >= threshold))
    return result


def _quality_gate(
    summary: dict[str, float],
    *,
    mean_threshold: float,
    p95_threshold: float,
) -> dict[str, object]:
    """Return a mean/p95 quality gate result without failing on max tails."""
    mean = float(summary.get("mean", float("inf")))
    p95 = float(summary.get("p95", float("inf")))
    return {
        "mean_threshold": float(mean_threshold),
        "p95_threshold": float(p95_threshold),
        "mean": mean,
        "p95": p95,
        "max_tail_report_only": float(summary.get("max", float("nan"))),
        "passed": bool(mean <= float(mean_threshold) and p95 <= float(p95_threshold)),
    }


def _count_semantics_quality_gates(
    *,
    response_summary: dict[str, float],
    target_summary: dict[str, float],
    response_target_summary: dict[str, float],
) -> dict[str, object]:
    """Return the agreed response/PF count-semantics quality gates."""
    response_mean, response_p95 = RESPONSE_VS_TRUTH_GATE
    target_mean, target_p95 = PF_TARGET_VS_TRUTH_GATE
    response_target_mean, response_target_p95 = RESPONSE_VS_PF_TARGET_GATE
    return {
        "current_response_poisson_vs_truth": _quality_gate(
            response_summary,
            mean_threshold=response_mean,
            p95_threshold=response_p95,
        ),
        "target_pf_counts_vs_truth": _quality_gate(
            target_summary,
            mean_threshold=target_mean,
            p95_threshold=target_p95,
        ),
        "current_response_vs_target_pf_counts": _quality_gate(
            response_target_summary,
            mean_threshold=response_target_mean,
            p95_threshold=response_target_p95,
        ),
    }


def _metric_bucket() -> dict[str, list[float]]:
    """Return an empty grouped-metric bucket for replay diagnostics."""
    return {
        "response_vs_truth": [],
        "target_vs_truth": [],
        "response_vs_target": [],
        "triad_spread": [],
        "signed_response_vs_truth": [],
        "signed_target_vs_truth": [],
        "response_pf_z_vs_truth": [],
        "response_pf_z_vs_target": [],
        "target_pf_z_vs_truth": [],
        "response_poisson_variance": [],
        "pf_likelihood_variance": [],
    }


def _low_count_bin_key(truth: float, truth_min: float) -> str:
    """Return the low-count diagnostic bin for a truth count."""
    truth_value = max(float(truth), 0.0)
    if truth_value >= float(truth_min):
        return f"truth_ge_{int(float(truth_min))}"
    for threshold in (1000.0, 5000.0, float(truth_min)):
        if truth_value < threshold:
            return f"truth_lt_{int(threshold)}"
    return f"truth_lt_{int(float(truth_min))}"


def _append_count_bin_metrics(
    buckets: dict[str, dict[str, list[float]]],
    *,
    truth: float,
    response_count: float,
    target_count: float,
    truth_min: float,
) -> None:
    """Append residual metrics to the matching truth-count diagnostic bin."""
    key = _low_count_bin_key(truth, truth_min)
    bucket = buckets.setdefault(key, _metric_bucket())
    response_error = _relative_error(response_count, truth)
    target_error = _relative_error(target_count, truth)
    response_target_error = _truth_relative_difference(
        response_count,
        target_count,
        truth,
    )
    if np.isfinite(response_error):
        bucket["response_vs_truth"].append(float(response_error))
    if np.isfinite(target_error):
        bucket["target_vs_truth"].append(float(target_error))
    if np.isfinite(response_target_error):
        bucket["response_vs_target"].append(float(response_target_error))


def _summarize_count_bins(
    buckets: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    """Return sorted low-count diagnostic summaries."""
    return {
        key: _summarize_metric_bucket(bucket)
        for key, bucket in sorted(buckets.items())
    }


def _line_bic_bucket(selection: dict[str, object]) -> str:
    """Return a stable line-model BIC bucket for grouped replay diagnostics."""
    if bool(selection.get("selected", False)):
        return "line_selected"
    if not bool(selection.get("requested", False)):
        return "line_disabled"
    delta = _float_value(selection.get("bic_delta_line_minus_isotope"), 0.0)
    if delta < -3500.0:
        return "line_margin_win"
    if delta < 0.0:
        return "line_weak_win"
    return "isotope_bic_selected"


def _summarize_metric_bucket(bucket: dict[str, list[float]]) -> dict[str, object]:
    """Return summary statistics for one grouped replay-diagnostic bucket."""
    response = _summary(bucket.get("response_vs_truth", []))
    target = _summary(bucket.get("target_vs_truth", []))
    response_target = _summary(bucket.get("response_vs_target", []))
    triad = _summary(bucket.get("triad_spread", []))
    response_z_truth = _summary(bucket.get("response_pf_z_vs_truth", []))
    response_z_target = _summary(bucket.get("response_pf_z_vs_target", []))
    target_z_truth = _summary(bucket.get("target_pf_z_vs_truth", []))
    response_variance = _summary(bucket.get("response_poisson_variance", []))
    pf_variance = _summary(bucket.get("pf_likelihood_variance", []))
    signed_response = np.asarray(
        [
            value
            for value in bucket.get("signed_response_vs_truth", [])
            if np.isfinite(value)
        ],
        dtype=float,
    )
    signed_target = np.asarray(
        [
            value
            for value in bucket.get("signed_target_vs_truth", [])
            if np.isfinite(value)
        ],
        dtype=float,
    )
    response_values = np.asarray(
        [
            value
            for value in bucket.get("response_vs_truth", [])
            if np.isfinite(value)
        ],
        dtype=float,
    )
    return {
        "n": float(response.get("n", 0.0)),
        "score": float(
            max(
                float(response.get("p95", 0.0)),
                float(target.get("p95", 0.0)),
                float(response_target.get("p95", 0.0)),
                float(triad.get("p95", 0.0)),
            )
        ),
        "response_vs_truth": response,
        "target_vs_truth": target,
        "response_vs_target": response_target,
        "triad_spread": triad,
        "response_pf_z_vs_truth": response_z_truth,
        "response_pf_z_vs_target": response_z_target,
        "target_pf_z_vs_truth": target_z_truth,
        "response_poisson_variance": response_variance,
        "pf_likelihood_variance": pf_variance,
        "tail_ge_20pct": float(np.sum(response_values >= 0.2)),
        "signed_response_vs_truth_mean": (
            float(np.mean(signed_response)) if signed_response.size else float("nan")
        ),
        "signed_target_vs_truth_mean": (
            float(np.mean(signed_target)) if signed_target.size else float("nan")
        ),
    }


def _ranked_group_summaries(
    grouped: dict[tuple[object, ...], dict[str, list[float]]],
    labels: tuple[str, ...],
    *,
    min_n: int = 1,
    top: int = 20,
    rank_metric: str = "score",
) -> list[dict[str, object]]:
    """Return ranked grouped replay summaries with label fields attached."""
    rows: list[dict[str, object]] = []
    for key, bucket in grouped.items():
        summary = _summarize_metric_bucket(bucket)
        if float(summary.get("n", 0.0)) < float(min_n):
            continue
        row = {label: value for label, value in zip(labels, key)}
        row.update(summary)
        rows.append(row)
    rows.sort(
        key=lambda item: _group_rank_value(item, rank_metric),
        reverse=True,
    )
    return rows[: max(0, int(top))]


def _group_rank_value(item: dict[str, object], rank_metric: str) -> float:
    """Return a sortable group-summary value for the requested metric."""
    value = item.get(rank_metric, 0.0)
    if isinstance(value, dict):
        return float(value.get("p95", 0.0))
    return float(value)


def _coverage(values: list[float]) -> dict[str, float]:
    """Return empirical coverage under common sigma thresholds."""
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    result = {"n": float(arr.size)}
    for threshold in (1.0, 2.0, 3.0, 5.0, 10.0):
        result[f"le_{threshold:g}"] = float(np.mean(arr <= threshold))
    return result


def _best_count_source(
    truth: float,
    candidates: dict[str, float],
) -> str:
    """Return the candidate count source with the smallest truth-relative error."""
    valid = {
        name: _relative_error(count, truth)
        for name, count in candidates.items()
        if np.isfinite(float(count))
    }
    if not valid:
        return "none"
    return min(valid, key=valid.__getitem__)


def _case_linear_index(case_name: str) -> int:
    """Return the zero-based validation case index encoded in a case name."""
    try:
        scenario = int(case_name.split("_pair", 1)[0].split("_")[-1])
        pair = int(case_name.split("_pair", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return -1
    return scenario * 64 + pair


def _case_scenario_key(case_name: str) -> str:
    """Return the validation scenario key without shield-pair expansion."""
    name = str(case_name)
    if "_pair" in name:
        return name.split("_pair", 1)[0]
    return name


def _shield_pair_id_from_row(case_name: str, row: dict[str, str]) -> int:
    """Return the shield pair id encoded in a record row or case name."""
    fe_index = _float_value(row.get("fe_index"))
    pb_index = _float_value(row.get("pb_index"))
    if np.isfinite(fe_index) and np.isfinite(pb_index):
        return int(fe_index) * 8 + int(pb_index)
    try:
        return int(case_name.split("_pair", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return -1


def _load_response_rows(records_path: Path) -> list[dict[str, str]]:
    """Load response-Poisson rows from a validation records CSV."""
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == "response_poisson"
        ]


def _rows_by_case_and_isotope(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Return validation CSV rows keyed by case and isotope."""
    return {
        (str(row.get("case", "")), str(row.get("isotope", ""))): row
        for row in rows
    }


def _replay_case(
    decomposer: SpectralDecomposer,
    rows: list[dict[str, str]],
) -> tuple[dict[str, IsotopeCountEstimate], dict[str, dict[str, object]]]:
    """Replay the current response-Poisson guard for one validation case."""
    requested = [str(row["isotope"]) for row in rows]
    estimates: dict[str, IsotopeCountEstimate] = {}
    variances: dict[str, float] = {}
    photopeak_counts: dict[str, float] = {}
    photopeak_variances: dict[str, float] = {}
    chi2_values: list[float] = []
    for row in rows:
        isotope = str(row["isotope"])
        raw_count = max(
            _float_value(
                row.get("response_poisson_raw_coefficient"),
                _float_value(row.get("estimated_counts"), 0.0),
            ),
            0.0,
        )
        variance = max(
            _float_value(row.get("estimated_variance"), max(raw_count, 1.0)),
            1.0,
        )
        estimates[isotope] = IsotopeCountEstimate(
            isotope=isotope,
            counts=raw_count,
            variance=variance,
            method="response_poisson",
        )
        variances[isotope] = variance
        photopeak_counts[isotope] = max(
            _float_value(row.get("response_poisson_photopeak_count"), 0.0),
            0.0,
        )
        photopeak_variances[isotope] = max(
            _float_value(row.get("response_poisson_photopeak_variance"), 1.0),
            1.0,
        )
        chi2 = _float_value(row.get("response_poisson_reduced_chi2"))
        if np.isfinite(chi2):
            chi2_values.append(float(chi2))
    reduced_chi2 = (
        float(np.median(np.asarray(chi2_values, dtype=float)))
        if chi2_values
        else float("nan")
    )
    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        photopeak_counts,
        photopeak_variances,
        reduced_chi2=reduced_chi2,
        requested=requested,
    )
    return estimates, diagnostics


def _runtime_likelihood_settings(runtime_config: dict[str, object]) -> dict[str, float]:
    """Return PF likelihood variance settings from nested or legacy config keys."""
    payload = runtime_config.get("pf_count_likelihood", {})
    likelihood = payload if isinstance(payload, dict) else {}

    def value(name: str, default: float) -> float:
        """Return one likelihood parameter value."""
        if name in likelihood:
            return float(likelihood[name])
        legacy_name = f"pf_{name}"
        if legacy_name in runtime_config:
            return float(runtime_config[legacy_name])
        return float(default)

    return {
        "transport_model_rel_sigma": max(
            value(
                "transport_model_rel_sigma",
                DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
            ),
            0.0,
        ),
        "transport_model_abs_sigma": max(
            value(
                "transport_model_abs_sigma",
                DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA,
            ),
            0.0,
        ),
        "spectrum_count_rel_sigma": max(
            value(
                "spectrum_count_rel_sigma",
                DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
            ),
            0.0,
        ),
        "spectrum_count_abs_sigma": max(
            value(
                "spectrum_count_abs_sigma",
                DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA,
            ),
            0.0,
        ),
        "low_count_abs_sigma": max(
            value("low_count_abs_sigma", DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA),
            0.0,
        ),
        "low_count_transition_counts": max(
            value(
                "low_count_transition_counts",
                DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS,
            ),
            0.0,
        ),
    }


def _runtime_variance_ceiling_settings(
    runtime_config: dict[str, object],
) -> dict[str, float | bool]:
    """Return response-Poisson observation-variance ceiling settings."""
    return {
        "enabled": bool(
            runtime_config.get("response_poisson_count_variance_ceiling_enable", True)
        ),
        "max_rel_sigma": max(
            float(
                runtime_config.get(
                    "response_poisson_count_variance_max_rel_sigma",
                    0.15,
                )
            ),
            0.0,
        ),
        "max_abs_sigma": max(
            float(
                runtime_config.get(
                    "response_poisson_count_variance_max_abs_sigma",
                    40.0,
                )
            ),
            0.0,
        ),
        "preserve_diagnostic_floors": bool(
            runtime_config.get(
                "response_poisson_count_variance_preserve_diagnostic_floors",
                True,
            )
        ),
        "preserve_guard_floors": bool(
            runtime_config.get(
                "response_poisson_count_variance_preserve_guard_floors",
                True,
            )
        ),
    }


def _cap_response_poisson_variance(
    count: float,
    variance: float,
    ceiling_settings: dict[str, float | bool],
    *,
    diagnostic_payload: dict[str, object] | None = None,
) -> float:
    """Apply the runtime response-Poisson variance ceiling to one count."""
    base_variance = max(float(variance), 1.0)
    if not bool(ceiling_settings.get("enabled", True)):
        return base_variance
    count_value = max(float(count), 0.0)
    rel_sigma = max(float(ceiling_settings.get("max_rel_sigma", 0.15)), 0.0)
    abs_sigma = max(float(ceiling_settings.get("max_abs_sigma", 40.0)), 0.0)
    reference_count = max(count_value, 1.0)
    variance_ceiling = max(
        count_value,
        (rel_sigma * reference_count) ** 2,
        abs_sigma**2,
        1.0,
    )
    if base_variance <= variance_ceiling:
        return float(base_variance)
    preserved_floor = _preserved_response_poisson_variance_floor(
        diagnostic_payload,
        preserve_guard=bool(ceiling_settings.get("preserve_guard_floors", True)),
        base_variance=base_variance,
    )
    return float(max(variance_ceiling, min(base_variance, preserved_floor)))


def _preserved_response_poisson_variance_floor(
    diagnostic_payload: dict[str, object] | None,
    *,
    preserve_guard: bool,
    base_variance: float,
) -> float:
    """Return replay variance that runtime guard diagnostics would preserve."""
    if not preserve_guard or not isinstance(diagnostic_payload, dict):
        return 0.0
    guarded_variance = _finite_mapping_float(diagnostic_payload, "guarded_variance")
    preserved = 0.0 if guarded_variance is None else max(guarded_variance, 0.0)
    reason = str(diagnostic_payload.get("reason", ""))
    if not reason:
        return float(preserved)
    poisson_count = _finite_mapping_float(diagnostic_payload, "poisson_count")
    photopeak_count = _finite_mapping_float(diagnostic_payload, "photopeak_count")
    if photopeak_count is None:
        photopeak_count = _finite_mapping_float(diagnostic_payload, "photo_count")
    if poisson_count is not None and photopeak_count is not None:
        preserved = max(preserved, (poisson_count - photopeak_count) ** 2)
    if "photo_count" in diagnostic_payload or "suppressed" in diagnostic_payload:
        preserved = max(preserved, float(base_variance))
    return float(max(preserved, 0.0))


def _finite_mapping_float(
    payload: dict[str, object],
    key: str,
) -> float | None:
    """Return a finite float from a replay diagnostic mapping."""
    value = payload.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _current_target_counts_from_results(
    *,
    results_path: Path,
    runtime_config: dict[str, object],
    isotopes: list[str],
) -> dict[tuple[str, str], float]:
    """Project saved source-term diagnostics through the current runtime model."""
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=tuple(isotopes),
    )
    model = observation_model.transport_response_model
    if not isinstance(model, dict) or bool(model.get("enabled", True)) is False:
        return {}
    by_isotope = model.get("by_isotope", {})
    if not isinstance(by_isotope, dict):
        return {}
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list):
        return {}
    projected: dict[tuple[str, str], float] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        case = result.get("case", {})
        if not isinstance(case, dict) or not bool(
            case.get("include_in_accuracy_summary", True),
        ):
            continue
        case_name = str(case.get("name", ""))
        if not case_name:
            continue
        detector_xyz = case.get("detector_pose_xyz", ())
        pair_id = int(case.get("fe_index", 0)) * 8 + int(case.get("pb_index", 0))
        per_isotope = result.get("per_isotope", {})
        if not isinstance(per_isotope, dict):
            continue
        for isotope, item in per_isotope.items():
            if not isinstance(item, dict):
                continue
            payload = by_isotope.get(str(isotope), {})
            if not isinstance(payload, dict):
                payload = {}
            total = 0.0
            for source_row in item.get("target_pf_count_diagnostics", []):
                if not isinstance(source_row, dict):
                    continue
                terms = source_row.get("transport_response_terms", [])
                if not isinstance(terms, list):
                    continue
                for term in terms:
                    if not isinstance(term, dict):
                        continue
                    counts = _transport_response_base_counts(term)
                    distance = _source_distance_feature(
                        source_row,
                        term,
                        detector_xyz,
                    )
                    shield_tau = _float_value(
                        term.get("shield_tau_feature"),
                        0.0,
                    )
                    total += counts * transport_response_factor_from_payload(
                        payload,
                        pair_id=pair_id,
                        shield_tau_feature=shield_tau,
                        obstacle_tau_feature=_float_value(
                            term.get("obstacle_tau_feature"),
                            0.0,
                        ),
                        fe_tau_feature=_float_value(term.get("fe_tau_feature"), 0.0),
                        pb_tau_feature=_float_value(term.get("pb_tau_feature"), 0.0),
                        distance_feature=distance,
                        distance_shield_feature=_float_value(
                            term.get("distance_shield_feature"),
                            distance * shield_tau,
                        ),
                    )
            projected[(case_name, str(isotope))] = float(total)
    return projected


def _transport_response_base_counts(term: dict[str, object]) -> float:
    """Return source-scale-adjusted base counts for response projection."""
    for key in ("scaled_base_counts", "scaled_counts", "counts"):
        counts = _float_value(term.get(key))
        if np.isfinite(counts):
            return max(float(counts), 0.0)
    return 0.0


def _source_distance_feature(
    source_row: dict[str, object],
    term: dict[str, object],
    detector_xyz: object,
) -> float:
    """Return the source-detector distance feature for replay projection."""
    for value in (
        term.get("distance_feature"),
        term.get("source_distance_m"),
        source_row.get("source_distance_m"),
        source_row.get("distance_feature"),
    ):
        distance = _float_value(value)
        if np.isfinite(distance) and distance >= 0.0:
            return float(distance)
    detector = np.asarray(detector_xyz, dtype=float).reshape(-1)
    position = np.asarray(source_row.get("position_xyz", []), dtype=float).reshape(-1)
    if detector.size != 3 or position.size != 3:
        return 0.0
    distance = float(np.linalg.norm(detector - position))
    return distance if np.isfinite(distance) and distance >= 0.0 else 0.0


def _split_response_truth_records_by_scenario(
    records: list[dict[str, object]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split response-truth calibration records by scenario key."""
    fraction = min(max(float(holdout_fraction), 0.0), 0.9)
    if fraction <= 0.0 or len(records) < 2:
        return list(records), []
    scenarios = sorted({str(record.get("scenario", "")) for record in records})
    if len(scenarios) < 2:
        return _split_response_truth_records_by_index(
            records,
            holdout_fraction=fraction,
            seed=int(seed),
        )
    rng = np.random.default_rng(int(seed))
    shuffled = list(scenarios)
    rng.shuffle(shuffled)
    holdout_count = int(round(float(len(shuffled)) * fraction))
    holdout_count = min(max(holdout_count, 1), len(shuffled) - 1)
    holdout_scenarios = set(shuffled[:holdout_count])
    train: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for record in records:
        if str(record.get("scenario", "")) in holdout_scenarios:
            holdout.append(record)
        else:
            train.append(record)
    return train, holdout


def _split_response_truth_records_by_index(
    records: list[dict[str, object]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split response-truth calibration records by row index."""
    fraction = min(max(float(holdout_fraction), 0.0), 0.9)
    if fraction <= 0.0 or len(records) < 2:
        return list(records), []
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(records))
    rng.shuffle(indices)
    holdout_count = int(round(float(len(records)) * fraction))
    holdout_count = min(max(holdout_count, 1), len(records) - 1)
    holdout_indices = set(int(value) for value in indices[:holdout_count])
    train: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if index in holdout_indices:
            holdout.append(record)
        else:
            train.append(record)
    return train, holdout


def _evaluate_response_truth_payload(
    records: list[dict[str, object]],
    payload: dict[str, object],
) -> dict[str, object]:
    """Evaluate a response-truth calibration payload on records."""
    calibrated_errors: list[float] = []
    uncalibrated_errors: list[float] = []
    scales: list[float] = []
    worst_rows: list[dict[str, object]] = []
    for record in records:
        isotope = str(record.get("isotope", ""))
        response_count = _float_value(record.get("response_count"))
        truth = _float_value(record.get("truth_count"))
        if not isotope or truth <= 0.0 or response_count <= 0.0:
            continue
        scale = response_truth_calibration_scale(
            payload,
            isotope=isotope,
            shield_pair_id=int(record.get("shield_pair_id", -1)),
            count=response_count,
            diagnostics={
                "coefficients": {isotope: record.get("raw_count", response_count)},
                "photopeak_counts": {
                    isotope: record.get("photopeak_count", response_count)
                },
                "reduced_chi2": record.get("reduced_chi2", 1.0),
            },
            spectrum_total=_float_value(record.get("spectrum_total")),
        )
        calibrated_count = float(response_count * scale)
        calibrated_error = _relative_error(calibrated_count, truth)
        uncalibrated_error = _relative_error(response_count, truth)
        calibrated_errors.append(calibrated_error)
        uncalibrated_errors.append(uncalibrated_error)
        scales.append(float(scale))
        worst_rows.append(
            {
                "case": str(record.get("case", "")),
                "scenario": str(record.get("scenario", "")),
                "isotope": isotope,
                "shield_pair_id": int(record.get("shield_pair_id", -1)),
                "truth": truth,
                "uncalibrated_count": response_count,
                "calibrated_count": calibrated_count,
                "scale": float(scale),
                "uncalibrated_error": uncalibrated_error,
                "calibrated_error": calibrated_error,
            }
        )
    worst_rows.sort(key=lambda row: float(row["calibrated_error"]), reverse=True)
    calibrated_summary = _summary(calibrated_errors)
    return {
        "uncalibrated_response_poisson": _summary(uncalibrated_errors),
        "calibrated_response_poisson": calibrated_summary,
        "quality_gate": _quality_gate(calibrated_summary),
        "scale_summary": _summary(scales),
        "worst_rows": worst_rows[:10],
    }


def _response_truth_holdout_validation(
    calibration_records: list[dict[str, object]],
    *,
    holdout_fraction: float,
    holdout_seed: int,
) -> dict[str, object]:
    """Fit response-truth calibration on train records and evaluate holdout."""
    train_records, holdout_records = _split_response_truth_records_by_scenario(
        calibration_records,
        holdout_fraction=float(holdout_fraction),
        seed=int(holdout_seed),
    )
    if not train_records or not holdout_records:
        return {
            "enabled": False,
            "reason": "empty_train_or_holdout",
            "train_records": len(train_records),
            "holdout_records": len(holdout_records),
        }
    payload = fit_local_knn_response_truth_calibration(
        train_records,
        neighbor_count=8,
        min_group_points=8,
        metadata={
            "purpose": "scenario_holdout_response_truth_calibration",
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(holdout_seed),
        },
    )
    return {
        "enabled": True,
        "holdout_fraction": float(holdout_fraction),
        "holdout_seed": int(holdout_seed),
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "train_scenarios": len(
            {str(record.get("scenario", "")) for record in train_records}
        ),
        "holdout_scenarios": len(
            {str(record.get("scenario", "")) for record in holdout_records}
        ),
        "fit_groups": len(dict(payload.get("groups", {}))),
        "train_validation": _evaluate_response_truth_payload(train_records, payload),
        "holdout_validation": _evaluate_response_truth_payload(
            holdout_records,
            payload,
        ),
    }


def recompute_spectrum_records(
    *,
    records_path: Path,
    config_path: Path,
    spectra_path: Path,
    results_json_path: Path | None = None,
    use_current_target: bool = True,
    truth_min: float,
    checkpoints: list[int],
    top: int,
    write_response_truth_calibration_path: Path | None = None,
    response_truth_holdout_fraction: float = 0.0,
    response_truth_holdout_seed: int = 20260608,
) -> dict[str, object]:
    """Recompute response-Poisson counts from saved spectra with current config."""
    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    spectrum_config = spectrum_config_from_runtime_config(runtime_config)
    response_truth_calibration: dict[str, object] | None = None
    decomposer = SpectralDecomposer(spectrum_config)
    likelihood_settings = _runtime_likelihood_settings(runtime_config)
    variance_ceiling_settings = _runtime_variance_ceiling_settings(runtime_config)
    rows = _load_response_rows(records_path)
    rows_by_key = _rows_by_case_and_isotope(rows)
    isotopes = sorted({str(row["isotope"]) for row in rows})
    if results_json_path is None:
        inferred_results_json = records_path.with_name("results.json")
    else:
        inferred_results_json = results_json_path
    current_targets = (
        _current_target_counts_from_results(
            results_path=inferred_results_json,
            runtime_config=runtime_config,
            isotopes=isotopes,
        )
        if use_current_target and inferred_results_json.exists()
        else {}
    )

    current_errors: list[float] = []
    current_raw_errors: list[float] = []
    current_photopeak_errors: list[float] = []
    recorded_response_errors: list[float] = []
    recorded_target_errors: list[float] = []
    target_errors: list[float] = []
    recorded_response_target_mismatch: list[float] = []
    current_response_target_mismatch: list[float] = []
    recorded_triad_spread: list[float] = []
    current_triad_spread: list[float] = []
    current_absent_target_counts: list[float] = []
    current_absent_truth_counts: list[float] = []
    response_pf_z_truth: list[float] = []
    response_pf_z_target: list[float] = []
    target_pf_z_truth: list[float] = []
    line_selection: Counter[str] = Counter()
    guard_reasons: Counter[str] = Counter()
    best_count_sources: Counter[str] = Counter()
    tail_best_count_sources: Counter[str] = Counter()
    truth_count_bins: dict[str, dict[str, list[float]]] = {}
    absent_target_methods: Counter[str] = Counter()
    absent_truth_methods: Counter[str] = Counter()
    absent_target_guard_reasons: Counter[str] = Counter()
    absent_truth_guard_reasons: Counter[str] = Counter()
    grouped_by_isotope: dict[tuple[object, ...], dict[str, list[float]]] = (
        defaultdict(_metric_bucket)
    )
    grouped_by_pair: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        _metric_bucket
    )
    grouped_by_pair_isotope: dict[tuple[object, ...], dict[str, list[float]]] = (
        defaultdict(_metric_bucket)
    )
    grouped_by_guard_reason: dict[tuple[object, ...], dict[str, list[float]]] = (
        defaultdict(_metric_bucket)
    )
    grouped_by_method: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        _metric_bucket
    )
    grouped_by_best_source: dict[tuple[object, ...], dict[str, list[float]]] = (
        defaultdict(_metric_bucket)
    )
    grouped_by_line_bic_bucket: dict[tuple[object, ...], dict[str, list[float]]] = (
        defaultdict(_metric_bucket)
    )
    checkpoint_rows: dict[int, dict[str, list[float]]] = {
        int(checkpoint): {"recorded": [], "current": [], "target": []}
        for checkpoint in checkpoints
    }
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[str(row.get("case", ""))].append(row)
    ranked_rows: list[dict[str, object]] = []
    absent_target_rows: list[dict[str, object]] = []
    absent_truth_rows: list[dict[str, object]] = []
    calibration_records: list[dict[str, object]] = []
    response_truth_calibration_rows = 0
    with np.load(spectra_path) as spectra:
        for case_name in spectra.files:
            spectrum = np.asarray(spectra[case_name], dtype=float)
            spectrum_total = float(np.sum(np.clip(spectrum, a_min=0.0, a_max=None)))
            case_rows = rows_by_case.get(str(case_name), [])
            live_time_s = 1.0
            for row in case_rows:
                value = _float_value(row.get("dwell_time_s"))
                if np.isfinite(value) and value > 0.0:
                    live_time_s = float(value)
                    break
            estimates = decomposer.compute_response_poisson_estimates(
                spectrum,
                isotopes=isotopes,
                live_time_s=live_time_s,
            )
            diagnostics = dict(decomposer.last_response_poisson_diagnostics)
            response_coefficients = dict(diagnostics.get("coefficients", {}))
            photopeak_counts = dict(diagnostics.get("photopeak_counts", {}))
            selection = dict(diagnostics.get("line_model_selection", {}))
            selected = str(selection.get("selected", False))
            line_selection[selected] += 1
            guard = dict(diagnostics.get("crosstalk_count_guard", {}))
            low_snr = dict(diagnostics.get("low_snr_photopeak_suppression", {}))
            for isotope, item in guard.items():
                if isinstance(item, dict):
                    guard_reasons[str(item.get("reason", ""))] += 1
            case_index = _case_linear_index(case_name)
            uncalibrated_variances = {
                isotope: max(float(estimate.variance), 1.0)
                for isotope, estimate in estimates.items()
            }
            for isotope in isotopes:
                row = rows_by_key.get((case_name, isotope))
                if row is None:
                    continue
                truth = _float_value(row.get("transport_truth_counts"))
                estimate = estimates.get(isotope)
                if estimate is None:
                    continue
                uncalibrated_count = float(estimate.counts)
                current_count = float(uncalibrated_count)
                raw_count = max(
                    _float_value(
                        response_coefficients.get(isotope),
                        uncalibrated_count,
                    ),
                    0.0,
                )
                photopeak_count = _float_value(photopeak_counts.get(isotope))
                current_variance = max(
                    float(uncalibrated_variances.get(isotope, estimate.variance)),
                    1.0,
                )
                row_guard = guard.get(isotope, {})
                row_low_snr = low_snr.get(isotope, {})
                diagnostic_payload = (
                    row_guard
                    if isinstance(row_guard, dict) and row_guard
                    else row_low_snr
                    if isinstance(row_low_snr, dict) and row_low_snr
                    else None
                )
                current_variance = _cap_response_poisson_variance(
                    current_count,
                    current_variance,
                    variance_ceiling_settings,
                    diagnostic_payload=diagnostic_payload,
                )
                recorded_count = _float_value(row.get("estimated_counts"))
                recorded_target_count = _float_value(row.get("target_pf_counts"))
                target_count = current_targets.get(
                    (case_name, isotope),
                    recorded_target_count,
                )
                guard_reason = (
                    str(row_guard.get("reason", "none"))
                    if isinstance(row_guard, dict) and row_guard
                    else "none"
                )
                pair_id = _shield_pair_id_from_row(case_name, row)
                calibration_scale = 1.0
                if calibration_scale != 1.0:
                    response_truth_calibration_rows += 1
                absent_summary = {
                    "case": case_name,
                    "isotope": isotope,
                    "truth": truth,
                    "target_count": target_count,
                    "recorded_target_count": recorded_target_count,
                    "current_count": current_count,
                    "uncalibrated_current_count": uncalibrated_count,
                    "current_raw_count": raw_count,
                    "current_photopeak_count": photopeak_count,
                    "response_truth_calibration_scale": calibration_scale,
                    "method": str(estimate.method),
                    "guard_reason": guard_reason,
                    "guard": row_guard if isinstance(row_guard, dict) else {},
                    "line_model_selected": selection.get("selected", False),
                    "line_model_reason": selection.get("reason", ""),
                    "line_bic_delta": selection.get(
                        "bic_delta_line_minus_isotope",
                        None,
                    ),
                }
                if target_count < float(truth_min):
                    current_absent_target_counts.append(current_count)
                    absent_target_methods[str(estimate.method)] += 1
                    absent_target_guard_reasons[guard_reason] += 1
                    absent_target_rows.append(dict(absent_summary))
                _append_count_bin_metrics(
                    truth_count_bins,
                    truth=truth,
                    response_count=current_count,
                    target_count=target_count,
                    truth_min=truth_min,
                )
                if truth < float(truth_min):
                    current_absent_truth_counts.append(current_count)
                    absent_truth_methods[str(estimate.method)] += 1
                    absent_truth_guard_reasons[guard_reason] += 1
                    absent_truth_rows.append(dict(absent_summary))
                    continue
                calibration_records.append(
                    {
                        "case": case_name,
                        "scenario": _case_scenario_key(case_name),
                        "isotope": isotope,
                        "shield_pair_id": pair_id,
                        "truth_count": truth,
                        "response_count": uncalibrated_count,
                        "raw_count": raw_count,
                        "photopeak_count": photopeak_count,
                        "reduced_chi2": diagnostics.get("reduced_chi2", 1.0),
                        "spectrum_total": spectrum_total,
                    }
                )
                current_error = _relative_error(current_count, truth)
                raw_error = _relative_error(raw_count, truth)
                photopeak_error = _relative_error(photopeak_count, truth)
                recorded_error = _relative_error(recorded_count, truth)
                recorded_target_error = _relative_error(recorded_target_count, truth)
                target_error = _relative_error(target_count, truth)
                recorded_target_mismatch = _truth_relative_difference(
                    recorded_count,
                    recorded_target_count,
                    truth,
                )
                current_target_mismatch = _truth_relative_difference(
                    current_count,
                    target_count,
                    truth,
                )
                recorded_spread = _truth_relative_spread(
                    recorded_count,
                    recorded_target_count,
                    truth,
                )
                current_spread = _truth_relative_spread(
                    current_count,
                    target_count,
                    truth,
                )
                current_errors.append(current_error)
                current_raw_errors.append(raw_error)
                if np.isfinite(photopeak_error):
                    current_photopeak_errors.append(photopeak_error)
                best_source = _best_count_source(
                    truth,
                    {
                        "final": current_count,
                        "raw": raw_count,
                        "photopeak": photopeak_count,
                    },
                )
                best_count_sources[best_source] += 1
                if current_error >= 0.2:
                    tail_best_count_sources[best_source] += 1
                line_bic_bucket = _line_bic_bucket(selection)
                recorded_response_errors.append(recorded_error)
                recorded_target_errors.append(recorded_target_error)
                target_errors.append(target_error)
                recorded_response_target_mismatch.append(recorded_target_mismatch)
                current_response_target_mismatch.append(current_target_mismatch)
                recorded_triad_spread.append(recorded_spread)
                current_triad_spread.append(current_spread)
                method = str(estimate.method)
                pf_variance = float("nan")
                response_pf_z_truth_value = float("nan")
                response_pf_z_target_value = float("nan")
                target_pf_z_truth_value = float("nan")
                if np.isfinite(target_count) and target_count > 0.0:
                    pf_variance = float(
                        count_likelihood_variance(
                            np.array([current_count], dtype=float),
                            np.array([target_count], dtype=float),
                            observation_count_variance=np.array(
                                [current_variance],
                                dtype=float,
                            ),
                            **likelihood_settings,
                        )[0]
                    )
                    response_pf_z_truth_value = float(
                        abs(current_count - truth) / np.sqrt(pf_variance)
                    )
                    response_pf_z_target_value = float(
                        abs(current_count - target_count) / np.sqrt(pf_variance)
                    )
                    target_pf_z_truth_value = float(
                        abs(target_count - truth) / np.sqrt(pf_variance)
                    )
                    response_pf_z_truth.append(response_pf_z_truth_value)
                    response_pf_z_target.append(response_pf_z_target_value)
                    target_pf_z_truth.append(target_pf_z_truth_value)
                for key, grouped in (
                    ((isotope,), grouped_by_isotope),
                    ((pair_id,), grouped_by_pair),
                    ((pair_id, isotope), grouped_by_pair_isotope),
                    ((guard_reason,), grouped_by_guard_reason),
                    ((method,), grouped_by_method),
                    ((best_source,), grouped_by_best_source),
                    ((line_bic_bucket,), grouped_by_line_bic_bucket),
                ):
                    bucket = grouped[key]
                    bucket["response_vs_truth"].append(current_error)
                    bucket["target_vs_truth"].append(target_error)
                    bucket["response_vs_target"].append(current_target_mismatch)
                    bucket["triad_spread"].append(current_spread)
                    bucket["signed_response_vs_truth"].append(
                        (current_count - truth) / max(float(truth), 1.0e-12)
                    )
                    bucket["signed_target_vs_truth"].append(
                        (target_count - truth) / max(float(truth), 1.0e-12)
                    )
                    bucket["response_pf_z_vs_truth"].append(response_pf_z_truth_value)
                    bucket["response_pf_z_vs_target"].append(response_pf_z_target_value)
                    bucket["target_pf_z_vs_truth"].append(target_pf_z_truth_value)
                    bucket["response_poisson_variance"].append(current_variance)
                    bucket["pf_likelihood_variance"].append(pf_variance)
                for checkpoint, buckets in checkpoint_rows.items():
                    if 0 <= case_index < int(checkpoint):
                        buckets["recorded"].append(recorded_error)
                        buckets["current"].append(current_error)
                        buckets["target"].append(target_error)
                ranked_rows.append(
                    {
                        "case": case_name,
                        "isotope": isotope,
                        "truth": truth,
                        "recorded_count": recorded_count,
                        "current_count": current_count,
                        "uncalibrated_current_count": uncalibrated_count,
                        "current_raw_count": raw_count,
                        "current_photopeak_count": photopeak_count,
                        "response_truth_calibration_scale": calibration_scale,
                        "response_poisson_variance": current_variance,
                        "pf_likelihood_variance": pf_variance,
                        "response_pf_z_vs_truth": response_pf_z_truth_value,
                        "response_pf_z_vs_target": response_pf_z_target_value,
                        "target_pf_z_vs_truth": target_pf_z_truth_value,
                        "recorded_target_count": recorded_target_count,
                        "target_count": target_count,
                        "recorded_error": recorded_error,
                        "current_error": current_error,
                        "current_raw_error": raw_error,
                        "current_photopeak_error": photopeak_error,
                        "best_count_source": best_source,
                        "recorded_target_error": recorded_target_error,
                        "target_error": target_error,
                        "recorded_response_target_mismatch": recorded_target_mismatch,
                        "current_response_target_mismatch": current_target_mismatch,
                        "recorded_triad_spread": recorded_spread,
                        "current_triad_spread": current_spread,
                        "line_model_selected": selection.get("selected", False),
                        "line_model_reason": selection.get("reason", ""),
                        "line_bic_bucket": line_bic_bucket,
                        "line_bic_delta": selection.get(
                            "bic_delta_line_minus_isotope",
                            None,
                        ),
                        "method": method,
                        "guard": row_guard if isinstance(row_guard, dict) else {},
                    }
                )
    ranked_rows.sort(key=lambda item: float(item["current_error"]), reverse=True)
    triad_rows = sorted(
        ranked_rows,
        key=lambda item: max(
            float(item["current_error"]),
            float(item["target_error"]),
            float(item["current_response_target_mismatch"]),
        ),
        reverse=True,
    )
    target_rows = sorted(
        ranked_rows,
        key=lambda item: float(item["target_error"]),
        reverse=True,
    )
    absent_target_rows.sort(
        key=lambda item: float(item["current_count"]),
        reverse=True,
    )
    absent_truth_rows.sort(
        key=lambda item: float(item["current_count"]),
        reverse=True,
    )
    current_response_summary = _summary(current_errors)
    target_summary = _summary(target_errors)
    response_target_summary = _summary(current_response_target_mismatch)
    calibration_write_info: dict[str, object] = {
        "configured": response_truth_calibration is not None,
        "applied_rows": float(response_truth_calibration_rows),
    }
    if write_response_truth_calibration_path is not None:
        calibration_payload = fit_local_knn_response_truth_calibration(
            calibration_records,
            neighbor_count=8,
            min_group_points=8,
            metadata={
                "source_records": str(records_path),
                "source_spectra": str(spectra_path),
                "source_config": str(config_path),
                "truth_min": float(truth_min),
                "purpose": "response_poisson_to_transport_truth_count_semantics",
            },
        )
        target_path = Path(write_response_truth_calibration_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        calibration_write_info.update(
            {
                "written_path": str(target_path),
                "fit_records": float(len(calibration_records)),
                "fit_groups": float(
                    len(dict(calibration_payload.get("groups", {})))
                ),
            }
        )
    holdout_validation: dict[str, object] = {
        "enabled": False,
        "reason": "disabled",
    }
    if float(response_truth_holdout_fraction) > 0.0:
        holdout_validation = _response_truth_holdout_validation(
            calibration_records,
            holdout_fraction=float(response_truth_holdout_fraction),
            holdout_seed=int(response_truth_holdout_seed),
        )
    return {
        "records": str(records_path),
        "config": str(config_path),
        "spectra": str(spectra_path),
        "results_json": str(inferred_results_json) if current_targets else None,
        "target_source": (
            "current_runtime_transport_response_model"
            if current_targets
            else "records_csv_target_pf_counts"
        ),
        "truth_min": float(truth_min),
        "recorded_response_poisson": _summary(recorded_response_errors),
        "current_response_poisson": current_response_summary,
        "current_raw_response_poisson": _summary(current_raw_errors),
        "current_photopeak_nnls": _summary(current_photopeak_errors),
        "response_truth_calibration": calibration_write_info,
        "response_truth_holdout_validation": holdout_validation,
        "quality_gates": _count_semantics_quality_gates(
            response_summary=current_response_summary,
            target_summary=target_summary,
            response_target_summary=response_target_summary,
        ),
        "truth_count_bins": _summarize_count_bins(truth_count_bins),
        "current_best_count_source_counts": dict(best_count_sources),
        "current_tail_ge_20pct_best_count_source_counts": dict(
            tail_best_count_sources,
        ),
        "current_absent_by_target_counts": _count_summary(
            current_absent_target_counts,
        ),
        "current_absent_by_truth_counts": _count_summary(
            current_absent_truth_counts,
        ),
        "current_absent_by_target_method_counts": dict(absent_target_methods),
        "current_absent_by_truth_method_counts": dict(absent_truth_methods),
        "current_absent_by_target_guard_reason_counts": dict(
            absent_target_guard_reasons,
        ),
        "current_absent_by_truth_guard_reason_counts": dict(
            absent_truth_guard_reasons,
        ),
        "recorded_target_pf_counts": _summary(recorded_target_errors),
        "target_pf_counts": target_summary,
        "recorded_response_vs_recorded_target_pf_counts": _summary(
            recorded_response_target_mismatch
        ),
        "current_response_vs_target_pf_counts": response_target_summary,
        "recorded_truth_response_target_spread": _summary(recorded_triad_spread),
        "current_truth_response_target_spread": _summary(current_triad_spread),
        "standard_pf_likelihood_z": {
            "settings": likelihood_settings,
            "variance_ceiling": variance_ceiling_settings,
            "target_source": (
                "current_runtime_transport_response_model"
                if current_targets
                else "records_csv_target_pf_counts"
            ),
            "current_response_vs_transport_truth": _summary(response_pf_z_truth),
            "current_response_vs_transport_truth_coverage": _coverage(
                response_pf_z_truth
            ),
            "current_response_vs_target_pf": _summary(response_pf_z_target),
            "current_response_vs_target_pf_coverage": _coverage(response_pf_z_target),
            "target_pf_vs_transport_truth": _summary(target_pf_z_truth),
            "target_pf_vs_transport_truth_coverage": _coverage(target_pf_z_truth),
        },
        "line_model_selected_counts": dict(line_selection),
        "guard_reasons": dict(guard_reasons),
        "current_by_isotope": _ranked_group_summaries(
            grouped_by_isotope,
            ("isotope",),
            top=max(int(top), 10),
        ),
        "current_by_isotope_target_rank": _ranked_group_summaries(
            grouped_by_isotope,
            ("isotope",),
            top=max(int(top), 10),
            rank_metric="target_vs_truth",
        ),
        "current_by_shield_pair": _ranked_group_summaries(
            grouped_by_pair,
            ("pair_id",),
            top=max(int(top), 10),
        ),
        "current_by_shield_pair_target_rank": _ranked_group_summaries(
            grouped_by_pair,
            ("pair_id",),
            top=max(int(top), 10),
            rank_metric="target_vs_truth",
        ),
        "current_by_shield_pair_isotope": _ranked_group_summaries(
            grouped_by_pair_isotope,
            ("pair_id", "isotope"),
            top=max(int(top), 10),
        ),
        "current_by_shield_pair_isotope_target_rank": _ranked_group_summaries(
            grouped_by_pair_isotope,
            ("pair_id", "isotope"),
            top=max(int(top), 10),
            rank_metric="target_vs_truth",
        ),
        "current_by_guard_reason": _ranked_group_summaries(
            grouped_by_guard_reason,
            ("guard_reason",),
            top=max(int(top), 10),
        ),
        "current_by_method": _ranked_group_summaries(
            grouped_by_method,
            ("method",),
            top=max(int(top), 10),
        ),
        "current_by_best_count_source": _ranked_group_summaries(
            grouped_by_best_source,
            ("best_count_source",),
            top=max(int(top), 10),
        ),
        "current_by_line_bic_bucket": _ranked_group_summaries(
            grouped_by_line_bic_bucket,
            ("line_bic_bucket",),
            top=max(int(top), 10),
        ),
        "checkpoints": {
            str(checkpoint): {
                label: _summary(values)
                for label, values in buckets.items()
            }
            for checkpoint, buckets in checkpoint_rows.items()
        },
        "worst_current_rows": ranked_rows[: max(0, int(top))],
        "worst_target_rows": target_rows[: max(0, int(top))],
        "worst_triad_rows": triad_rows[: max(0, int(top))],
        "worst_absent_by_target_rows": absent_target_rows[: max(0, int(top))],
        "worst_absent_by_truth_rows": absent_truth_rows[: max(0, int(top))],
    }


def replay_records(
    *,
    records_path: Path,
    config_path: Path,
    results_json_path: Path | None = None,
    use_current_target: bool = True,
    truth_min: float,
    checkpoints: list[int],
    top: int,
) -> dict[str, object]:
    """Replay current guard behavior and return aggregate diagnostics."""
    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    decomposer = SpectralDecomposer(spectrum_config_from_runtime_config(runtime_config))
    likelihood_settings = _runtime_likelihood_settings(runtime_config)
    variance_ceiling_settings = _runtime_variance_ceiling_settings(runtime_config)
    rows = _load_response_rows(records_path)
    isotopes = sorted({str(row["isotope"]) for row in rows})
    if results_json_path is None:
        inferred_results_json = records_path.with_name("results.json")
    else:
        inferred_results_json = results_json_path
    current_targets = (
        _current_target_counts_from_results(
            results_path=inferred_results_json,
            runtime_config=runtime_config,
            isotopes=isotopes,
        )
        if use_current_target and inferred_results_json.exists()
        else {}
    )
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[str(row["case"])].append(row)

    old_errors: list[float] = []
    projected_errors: list[float] = []
    recorded_target_errors: list[float] = []
    target_errors: list[float] = []
    recorded_response_target_mismatch: list[float] = []
    projected_response_target_mismatch: list[float] = []
    old_recorded_triad_spread: list[float] = []
    projected_current_triad_spread: list[float] = []
    projected_recorded_pf_z_target: list[float] = []
    recorded_target_pf_z_truth: list[float] = []
    old_observation_z_truth: list[float] = []
    projected_observation_z_truth: list[float] = []
    projected_pf_z_truth: list[float] = []
    projected_pf_z_target: list[float] = []
    target_pf_z_truth: list[float] = []
    checkpoint_rows: dict[int, dict[str, list[float]]] = {
        int(checkpoint): {"old": [], "projected": [], "target": []}
        for checkpoint in checkpoints
    }
    truth_count_bins: dict[str, dict[str, list[float]]] = {}
    reasons: Counter[str] = Counter()
    changed_rows = 0
    large_regressions: list[dict[str, object]] = []
    ranked_rows: list[dict[str, object]] = []
    for case_name, case_rows in rows_by_case.items():
        projected, diagnostics = _replay_case(decomposer, case_rows)
        for isotope, diagnostic in diagnostics.items():
            reasons[str(diagnostic.get("reason", ""))] += 1
        case_index = _case_linear_index(case_name)
        for row in case_rows:
            truth = _float_value(row.get("transport_truth_counts"))
            isotope = str(row["isotope"])
            old_count = _float_value(row.get("estimated_counts"))
            projected_count = float(projected[isotope].counts)
            projected_variance = _cap_response_poisson_variance(
                projected_count,
                float(projected[isotope].variance),
                variance_ceiling_settings,
                diagnostic_payload=diagnostics.get(isotope, {}),
            )
            old_variance = max(
                _float_value(row.get("estimated_variance"), max(old_count, 1.0)),
                1.0,
            )
            old_variance = _cap_response_poisson_variance(
                old_count,
                old_variance,
                variance_ceiling_settings,
            )
            recorded_target_count = _float_value(row.get("target_pf_counts"))
            target_count = current_targets.get(
                (case_name, isotope),
                recorded_target_count,
            )
            _append_count_bin_metrics(
                truth_count_bins,
                truth=truth,
                response_count=projected_count,
                target_count=target_count,
                truth_min=truth_min,
            )
            if truth < float(truth_min):
                continue
            old_error = _relative_error(old_count, truth)
            projected_error = _relative_error(projected_count, truth)
            target_error = _relative_error(target_count, truth)
            recorded_target_error = _relative_error(recorded_target_count, truth)
            recorded_target_mismatch = _truth_relative_difference(
                old_count,
                recorded_target_count,
                truth,
            )
            projected_target_mismatch = _truth_relative_difference(
                projected_count,
                target_count,
                truth,
            )
            old_recorded_spread = _truth_relative_spread(
                old_count,
                recorded_target_count,
                truth,
            )
            projected_current_spread = _truth_relative_spread(
                projected_count,
                target_count,
                truth,
            )
            old_z_truth = abs(old_count - truth) / np.sqrt(old_variance)
            projected_z_truth = abs(projected_count - truth) / np.sqrt(
                projected_variance
            )
            old_observation_z_truth.append(float(old_z_truth))
            projected_observation_z_truth.append(float(projected_z_truth))
            if np.isfinite(recorded_target_count) and recorded_target_count > 0.0:
                recorded_pf_variance = float(
                    count_likelihood_variance(
                        np.array([projected_count], dtype=float),
                        np.array([recorded_target_count], dtype=float),
                        observation_count_variance=np.array(
                            [projected_variance],
                            dtype=float,
                        ),
                        **likelihood_settings,
                    )[0]
                )
                projected_recorded_pf_z_target.append(
                    float(
                        abs(projected_count - recorded_target_count)
                        / np.sqrt(recorded_pf_variance)
                    )
                )
                recorded_target_pf_z_truth.append(
                    float(
                        abs(recorded_target_count - truth)
                        / np.sqrt(recorded_pf_variance)
                    )
                )
            pf_variance = float("nan")
            projected_pf_z_truth_value = float("nan")
            projected_pf_z_target_value = float("nan")
            target_pf_z_truth_value = float("nan")
            if np.isfinite(target_count) and target_count > 0.0:
                pf_variance = float(
                    count_likelihood_variance(
                        np.array([projected_count], dtype=float),
                        np.array([target_count], dtype=float),
                        observation_count_variance=np.array(
                            [projected_variance],
                            dtype=float,
                        ),
                        **likelihood_settings,
                    )[0]
                )
                projected_pf_z_truth_value = float(
                    abs(projected_count - truth) / np.sqrt(pf_variance)
                )
                projected_pf_z_target_value = float(
                    abs(projected_count - target_count) / np.sqrt(pf_variance)
                )
                target_pf_z_truth_value = float(
                    abs(target_count - truth) / np.sqrt(pf_variance)
                )
                projected_pf_z_truth.append(projected_pf_z_truth_value)
                projected_pf_z_target.append(projected_pf_z_target_value)
                target_pf_z_truth.append(target_pf_z_truth_value)
            old_errors.append(old_error)
            projected_errors.append(projected_error)
            recorded_target_errors.append(recorded_target_error)
            target_errors.append(target_error)
            recorded_response_target_mismatch.append(recorded_target_mismatch)
            projected_response_target_mismatch.append(projected_target_mismatch)
            old_recorded_triad_spread.append(old_recorded_spread)
            projected_current_triad_spread.append(projected_current_spread)
            for checkpoint, buckets in checkpoint_rows.items():
                if 0 <= case_index < int(checkpoint):
                    buckets["old"].append(old_error)
                    buckets["projected"].append(projected_error)
                    buckets["target"].append(target_error)
            if abs(projected_count - old_count) > max(
                1.0e-6,
                1.0e-9 * max(abs(old_count), 1.0),
            ):
                changed_rows += 1
            row_summary = {
                "case": case_name,
                "isotope": isotope,
                "truth": truth,
                "old_count": old_count,
                "projected_count": projected_count,
                "recorded_target_count": recorded_target_count,
                "target_count": target_count,
                "photopeak_count": _float_value(
                    row.get("response_poisson_photopeak_count"),
                    0.0,
                ),
                "old_variance": old_variance,
                "projected_variance": projected_variance,
                "pf_likelihood_variance": pf_variance,
                "old_observation_z_vs_truth": old_z_truth,
                "projected_observation_z_vs_truth": projected_z_truth,
                "projected_pf_z_vs_truth": projected_pf_z_truth_value,
                "projected_pf_z_vs_target": projected_pf_z_target_value,
                "target_pf_z_vs_truth": target_pf_z_truth_value,
                "old_error": old_error,
                "projected_error": projected_error,
                "target_error": target_error,
                "recorded_target_error": recorded_target_error,
                "recorded_response_target_mismatch": recorded_target_mismatch,
                "projected_response_target_mismatch": projected_target_mismatch,
                "old_recorded_triad_spread": old_recorded_spread,
                "projected_current_triad_spread": projected_current_spread,
                "guard": diagnostics.get(isotope, {}),
            }
            ranked_rows.append(row_summary)
            if projected_error > old_error + 0.05:
                large_regressions.append(row_summary)

    ranked_rows.sort(key=lambda item: float(item["projected_error"]), reverse=True)
    large_regressions.sort(
        key=lambda item: float(item["projected_error"]) - float(item["old_error"]),
        reverse=True,
    )
    triad_rows = sorted(
        ranked_rows,
        key=lambda item: max(
            float(item["projected_error"]),
            float(item["target_error"]),
            float(item["projected_response_target_mismatch"]),
        ),
        reverse=True,
    )
    return {
        "records": str(records_path),
        "config": str(config_path),
        "results_json": str(inferred_results_json) if current_targets else None,
        "target_source": (
            "current_runtime_transport_response_model"
            if current_targets
            else "records_csv_target_pf_counts"
        ),
        "truth_min": float(truth_min),
        "old": _summary(old_errors),
        "projected_current_guard": _summary(projected_errors),
        "recorded_target_pf_counts": _summary(recorded_target_errors),
        "target_pf_counts": _summary(target_errors),
        "recorded_response_vs_recorded_target_pf_counts": _summary(
            recorded_response_target_mismatch
        ),
        "projected_response_vs_target_pf_counts": _summary(
            projected_response_target_mismatch
        ),
        "quality_gates": _count_semantics_quality_gates(
            response_summary=_summary(projected_errors),
            target_summary=_summary(target_errors),
            response_target_summary=_summary(projected_response_target_mismatch),
        ),
        "truth_count_bins": _summarize_count_bins(truth_count_bins),
        "old_recorded_truth_response_target_spread": _summary(
            old_recorded_triad_spread
        ),
        "projected_current_truth_response_target_spread": _summary(
            projected_current_triad_spread
        ),
        "observation_z_vs_transport_truth": {
            "old": _summary(old_observation_z_truth),
            "old_coverage": _coverage(old_observation_z_truth),
            "projected_current_guard": _summary(projected_observation_z_truth),
            "projected_current_guard_coverage": _coverage(
                projected_observation_z_truth
            ),
        },
        "standard_pf_likelihood_z": {
            "settings": likelihood_settings,
            "variance_ceiling": variance_ceiling_settings,
            "target_source": (
                "current_runtime_transport_response_model"
                if current_targets
                else "records_csv_target_pf_counts"
            ),
            "projected_response_vs_transport_truth": _summary(projected_pf_z_truth),
            "projected_response_vs_transport_truth_coverage": _coverage(
                projected_pf_z_truth
            ),
            "projected_response_vs_target_pf": _summary(projected_pf_z_target),
            "projected_response_vs_target_pf_coverage": _coverage(
                projected_pf_z_target
            ),
            "target_pf_vs_transport_truth": _summary(target_pf_z_truth),
            "target_pf_vs_transport_truth_coverage": _coverage(target_pf_z_truth),
            "projected_response_vs_recorded_target_pf": _summary(
                projected_recorded_pf_z_target
            ),
            "recorded_target_pf_vs_transport_truth": _summary(
                recorded_target_pf_z_truth
            ),
        },
        "changed_rows": int(changed_rows),
        "guard_reasons": dict(reasons),
        "large_regressions": int(len(large_regressions)),
        "checkpoints": {
            str(checkpoint): {
                label: _summary(values)
                for label, values in buckets.items()
            }
            for checkpoint, buckets in checkpoint_rows.items()
        },
        "worst_projected_rows": ranked_rows[: max(0, int(top))],
        "worst_triad_rows": triad_rows[: max(0, int(top))],
        "worst_regressions": large_regressions[: max(0, int(top))],
    }


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay the current response-Poisson crosstalk guard from a "
            "validation records.csv file without rerunning Geant4."
        )
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(
            "results/spectrum_validation/"
            "count_model_dominanceguard_transport_20260608_001510/records.csv"
        ),
        help="Validation records.csv path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
        ),
        help="Runtime config used to construct SpectrumConfig.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help=(
            "Optional validation results.json path used to project current PF "
            "targets through the runtime transport-response model. Defaults to "
            "the records.csv sibling when present."
        ),
    )
    parser.add_argument(
        "--spectra",
        type=Path,
        default=None,
        help="Optional spectra.npz path for --recompute-spectra.",
    )
    parser.add_argument(
        "--recompute-spectra",
        action="store_true",
        help=(
            "Recompute response_poisson counts from saved spectra.npz with the "
            "current runtime config instead of replaying only the saved guard "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--write-response-truth-calibration",
        type=Path,
        default=None,
        help=(
            "Write a diagnostic detector-response calibration that maps "
            "recomputed response_poisson counts onto transport_truth_counts. "
            "This is for holdout analysis only and is not a runtime artifact. "
            "Only valid with --recompute-spectra."
        ),
    )
    parser.add_argument(
        "--response-truth-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "Scenario-level holdout fraction for fitting a fresh response-truth "
            "calibration on train records and evaluating held-out records."
        ),
    )
    parser.add_argument(
        "--response-truth-holdout-seed",
        type=int,
        default=20260608,
        help="Random seed for the response-truth scenario holdout split.",
    )
    parser.add_argument(
        "--no-current-target",
        action="store_true",
        help=(
            "Use the records.csv target_pf_counts column instead of current "
            "projection."
        ),
    )
    parser.add_argument(
        "--truth-min",
        type=float,
        default=10000.0,
        help="Minimum transport truth count included in accuracy statistics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        action="append",
        default=[256, 320, 512, 1024, 2048],
        help="Cumulative case-count checkpoint to summarize; may be repeated.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of worst rows to include.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> None:
    """Run the response-Poisson guard replay CLI."""
    args = _parser().parse_args()
    if bool(args.recompute_spectra):
        spectra_path = (
            args.spectra
            if args.spectra is not None
            else Path(args.records).with_name("spectra.npz")
        )
        result = recompute_spectrum_records(
            records_path=args.records,
            config_path=args.config,
            spectra_path=spectra_path,
            results_json_path=args.results_json,
            use_current_target=not bool(args.no_current_target),
            truth_min=float(args.truth_min),
            checkpoints=[int(value) for value in args.checkpoint],
            top=int(args.top),
            write_response_truth_calibration_path=(
                args.write_response_truth_calibration
            ),
            response_truth_holdout_fraction=float(
                args.response_truth_holdout_fraction
            ),
            response_truth_holdout_seed=int(args.response_truth_holdout_seed),
        )
    else:
        if args.write_response_truth_calibration is not None:
            raise ValueError(
                "--write-response-truth-calibration requires --recompute-spectra"
            )
        if float(args.response_truth_holdout_fraction) > 0.0:
            raise ValueError(
                "--response-truth-holdout-fraction requires --recompute-spectra"
            )
        result = replay_records(
            records_path=args.records,
            config_path=args.config,
            results_json_path=args.results_json,
            use_current_target=not bool(args.no_current_target),
            truth_min=float(args.truth_min),
            checkpoints=[int(value) for value in args.checkpoint],
            top=int(args.top),
        )
    indent = 2 if bool(args.pretty) else None
    print(json.dumps(result, indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
