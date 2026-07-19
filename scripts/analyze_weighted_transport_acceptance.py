"""Analyze predeclared acceptance gates for weighted Geant4 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ISOTOPES = ("Cs-137", "Co-60", "Eu-154")
EXPECTED_FRESH_CASES = 64
EXPECTED_MAX_SAMPLING_FRACTION = 1.0
EXPECTED_TARGET_SAMPLED_PRIMARIES = 1_500_000.0
MIN_SAMPLING_FRACTION = 1.0e-6
MIN_EXPECTED_SAMPLED_PRIMARIES = 100_000.0
MIN_EFFECTIVE_ENTRIES = 2_000.0
MIN_ISOTOPE_DETECTED_EFFECTIVE_ENTRIES = 100.0
MAX_ABS_PRIMARY_POISSON_Z = 8.0
MAX_RUNTIME_P50_S = 10.0
MAX_RUNTIME_P95_S = 12.0
MAX_RUNTIME_S = 15.0
MAX_POOLED_ABS_Z = 3.0
MAX_INDIVIDUAL_ABS_Z = 5.0
MAX_POOLED_ABS_RELATIVE_BIAS = 0.02
MIN_RESPONSE_Q_PER_DOF = 0.50
MAX_RESPONSE_Q_PER_DOF = 1.50
MAX_SMALL_SAMPLE_RESPONSE_Q_PER_DOF = 4.0
MIN_ISOTOPE_MEAN_Z2 = 0.45
MAX_ISOTOPE_MEAN_Z2 = 1.75
MAX_ISOTOPE_ABS_RELATIVE_BIAS = 0.02
MIN_NOMINAL_95_COVERAGE = 0.90
MAX_NOMINAL_95_COVERAGE = 0.99
MIN_COVERAGE_SAMPLE_SIZE = 16
MAX_PF_FORWARD_MEAN_ABS_RELATIVE_ERROR = 0.05
MAX_PF_FORWARD_MEDIAN_ABS_RELATIVE_ERROR = 0.03
MAX_PF_FORWARD_P95_ABS_RELATIVE_ERROR = 0.15
MAX_PF_FORWARD_ABS_RELATIVE_ERROR = 0.50
MAX_RESPONSE_MEAN_ABS_RELATIVE_ERROR = 0.08
MAX_RESPONSE_MEDIAN_ABS_RELATIVE_ERROR = 0.05
MAX_RESPONSE_P95_ABS_RELATIVE_ERROR = 0.25
MAX_RESPONSE_ABS_RELATIVE_ERROR = 1.00
MAX_RESPONSE_ISOTOPE_MEAN_ABS_RELATIVE_ERROR = 0.10
MAX_RESPONSE_ISOTOPE_P95_ABS_RELATIVE_ERROR = 0.30
MIN_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN = 0.10
MAX_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN = 0.30
MAX_LIKELIHOOD_SCALE_OVER_TARGET_P95 = 0.75
MAX_LIKELIHOOD_SQUARED_DISTANCE_PER_DOF = 4.0
MAX_NORMALIZED_RESIDUAL_RMS = 2.5
MAX_NORMALIZED_RESIDUAL_MEAN_ABS = 1.0
PSD_RELATIVE_TOLERANCE = 1.0e-8

THRESHOLDS = {
    "expected_fresh_cases": EXPECTED_FRESH_CASES,
    "expected_max_sampling_fraction": EXPECTED_MAX_SAMPLING_FRACTION,
    "expected_target_sampled_primaries": EXPECTED_TARGET_SAMPLED_PRIMARIES,
    "min_expected_sampled_primaries": MIN_EXPECTED_SAMPLED_PRIMARIES,
    "min_effective_entries": MIN_EFFECTIVE_ENTRIES,
    "min_isotope_detected_effective_entries": (MIN_ISOTOPE_DETECTED_EFFECTIVE_ENTRIES),
    "max_abs_primary_poisson_z": MAX_ABS_PRIMARY_POISSON_Z,
    "max_runtime_p50_s": MAX_RUNTIME_P50_S,
    "max_runtime_p95_s": MAX_RUNTIME_P95_S,
    "max_runtime_s": MAX_RUNTIME_S,
    "max_pooled_abs_z": MAX_POOLED_ABS_Z,
    "max_individual_abs_z": MAX_INDIVIDUAL_ABS_Z,
    "max_pooled_abs_relative_bias": MAX_POOLED_ABS_RELATIVE_BIAS,
    "min_response_q_per_dof": MIN_RESPONSE_Q_PER_DOF,
    "max_response_q_per_dof": MAX_RESPONSE_Q_PER_DOF,
    "max_small_sample_response_q_per_dof": (MAX_SMALL_SAMPLE_RESPONSE_Q_PER_DOF),
    "min_isotope_mean_z2": MIN_ISOTOPE_MEAN_Z2,
    "max_isotope_mean_z2": MAX_ISOTOPE_MEAN_Z2,
    "max_isotope_abs_relative_bias": MAX_ISOTOPE_ABS_RELATIVE_BIAS,
    "min_nominal_95_coverage": MIN_NOMINAL_95_COVERAGE,
    "max_nominal_95_coverage": MAX_NOMINAL_95_COVERAGE,
    "min_coverage_sample_size": MIN_COVERAGE_SAMPLE_SIZE,
    "max_pf_forward_mean_abs_relative_error": (MAX_PF_FORWARD_MEAN_ABS_RELATIVE_ERROR),
    "max_pf_forward_median_abs_relative_error": (
        MAX_PF_FORWARD_MEDIAN_ABS_RELATIVE_ERROR
    ),
    "max_pf_forward_p95_abs_relative_error": (MAX_PF_FORWARD_P95_ABS_RELATIVE_ERROR),
    "max_pf_forward_abs_relative_error": MAX_PF_FORWARD_ABS_RELATIVE_ERROR,
    "max_response_mean_abs_relative_error": (MAX_RESPONSE_MEAN_ABS_RELATIVE_ERROR),
    "max_response_median_abs_relative_error": (MAX_RESPONSE_MEDIAN_ABS_RELATIVE_ERROR),
    "max_response_p95_abs_relative_error": (MAX_RESPONSE_P95_ABS_RELATIVE_ERROR),
    "max_response_abs_relative_error": MAX_RESPONSE_ABS_RELATIVE_ERROR,
    "max_response_isotope_mean_abs_relative_error": (
        MAX_RESPONSE_ISOTOPE_MEAN_ABS_RELATIVE_ERROR
    ),
    "max_response_isotope_p95_abs_relative_error": (
        MAX_RESPONSE_ISOTOPE_P95_ABS_RELATIVE_ERROR
    ),
    "min_likelihood_scale_over_target_median": (
        MIN_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN
    ),
    "max_likelihood_scale_over_target_median": (
        MAX_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN
    ),
    "max_likelihood_scale_over_target_p95": (MAX_LIKELIHOOD_SCALE_OVER_TARGET_P95),
    "max_likelihood_squared_distance_per_dof": (
        MAX_LIKELIHOOD_SQUARED_DISTANCE_PER_DOF
    ),
    "max_normalized_residual_rms": MAX_NORMALIZED_RESIDUAL_RMS,
    "max_normalized_residual_mean_abs": MAX_NORMALIZED_RESIDUAL_MEAN_ABS,
    "psd_relative_tolerance": PSD_RELATIVE_TOLERANCE,
}


def _finite_float(value: object) -> float:
    """Return a finite float or raise ``ValueError``."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Expected a finite number, got {value!r}.")
    return numeric


def _metadata_bool(value: object) -> bool:
    """Parse one metadata boolean without accepting arbitrary truthy values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Expected a metadata boolean, got {value!r}.")


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return stable descriptive quantiles for finite values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _gate(
    value: object,
    *,
    passed: bool,
    criterion: str,
    applicable: bool = True,
    detail: str = "",
) -> dict[str, object]:
    """Build one machine-readable acceptance-gate record."""
    return {
        "applicable": bool(applicable),
        "passed": bool(passed) if applicable else None,
        "criterion": criterion,
        "value": value,
        "detail": detail,
    }


def _case_name(record: Mapping[str, Any]) -> str:
    """Return a validation record's case name."""
    case = record.get("case", {})
    if not isinstance(case, Mapping):
        return ""
    return str(case.get("name", ""))


def _physical_case_payload(record: Mapping[str, Any]) -> dict[str, object]:
    """Return the fields that uniquely specify a physical validation case."""
    case = record.get("case", {})
    if not isinstance(case, Mapping):
        return {}
    keys = (
        "detector_pose_xyz",
        "dwell_time_s",
        "fe_index",
        "pb_index",
        "sources",
        "obstacle_cells",
        "obstacle_instances",
    )
    return {key: case.get(key) for key in keys}


def _payload_digest(payload: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_by_name(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Index records by case name and report duplicate or empty names."""
    indexed: dict[str, Mapping[str, Any]] = {}
    invalid: list[str] = []
    for index, record in enumerate(records):
        name = _case_name(record)
        if not name or name in indexed:
            invalid.append(name or f"<empty:{index}>")
            continue
        indexed[name] = record
    return indexed, invalid


def analyze_case_matching(
    accelerated: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Validate case-name and physical-case identity for a paired analysis."""
    accelerated_by_name, accelerated_invalid = _records_by_name(accelerated)
    reference_by_name, reference_invalid = _records_by_name(reference)
    accelerated_names = set(accelerated_by_name)
    reference_names = set(reference_by_name)
    common = sorted(accelerated_names & reference_names)
    physical_mismatches: list[dict[str, str]] = []
    for name in common:
        accelerated_payload = _physical_case_payload(accelerated_by_name[name])
        reference_payload = _physical_case_payload(reference_by_name[name])
        if accelerated_payload != reference_payload:
            physical_mismatches.append(
                {
                    "case": name,
                    "accelerated_sha256": _payload_digest(accelerated_payload),
                    "reference_sha256": _payload_digest(reference_payload),
                }
            )
    return {
        "accelerated_count": len(accelerated),
        "reference_count": len(reference),
        "paired_count": len(common),
        "accelerated_invalid_names": accelerated_invalid,
        "reference_invalid_names": reference_invalid,
        "missing_from_accelerated": sorted(reference_names - accelerated_names),
        "missing_from_reference": sorted(accelerated_names - reference_names),
        "physical_mismatches": physical_mismatches,
        "names_match": bool(
            not accelerated_invalid
            and not reference_invalid
            and accelerated_names <= reference_names
        ),
        "physical_cases_match": not physical_mismatches,
        "paired_case_names": common,
    }


def _likelihood_semantics_errors(record: Mapping[str, Any]) -> list[str]:
    """Return PF likelihood-semantics errors retained in one record."""
    diagnostics = record.get("pf_count_likelihood_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return ["missing pf_count_likelihood_diagnostics"]
    specs = diagnostics.get("likelihood_spec_by_isotope", {})
    if not isinstance(specs, Mapping):
        return ["missing likelihood_spec_by_isotope"]
    errors: list[str] = []
    for isotope in ISOTOPES:
        spec = specs.get(isotope, {})
        if not isinstance(spec, Mapping):
            errors.append(f"{isotope}: missing likelihood spec")
            continue
        if spec.get("observation_count_variance_semantics") != "complete_statistical":
            errors.append(
                f"{isotope}: observation variance is not complete_statistical"
            )
    guards = diagnostics.get("runtime_likelihood_guards", {})
    if not isinstance(guards, Mapping):
        errors.append("missing runtime_likelihood_guards")
        return errors
    if guards.get("observation_count_variance_semantics") != "complete_statistical":
        errors.append("runtime observation variance is not complete_statistical")
    for key in (
        "direct_spectrum_likelihood_enable",
        "shield_contrast_likelihood_enable",
        "shield_view_ratio_likelihood_enable",
    ):
        if guards.get(key) is not False:
            errors.append(f"{key}: expected false, got {guards.get(key)!r}")
    return errors


def _transport_metadata_errors(record: Mapping[str, Any]) -> list[str]:
    """Validate strict weighted-history and variance provenance for one case."""
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ["metadata is missing"]
    errors: list[str] = []
    try:
        effective_fraction_for_mode = _finite_float(
            metadata["primary_sampling_fraction"]
        )
    except (KeyError, TypeError, ValueError):
        effective_fraction_for_mode = float("nan")
    history_is_thinned = bool(
        np.isfinite(effective_fraction_for_mode)
        and effective_fraction_for_mode < 1.0 - 1.0e-12
    )

    expected_strings = {
        "backend": "geant4",
        "engine_mode": "external",
        "source_rate_model": "detector_cps_1m",
        "expected_primary_semantics": "detector_equivalent_histories",
        "transport_history_mode": (
            "weighted_thinning" if history_is_thinned else "full_unit_weight"
        ),
        "spectrum_variance_semantics": ("compound_poisson_sumw2_includes_counting"),
        "spectrum_variance_dead_time_propagation": "fixed_observed_scale",
        "intensity_cps_1m_definition": "net_detector_count_rate_at_1m",
    }
    for key, expected in expected_strings.items():
        if metadata.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {metadata.get(key)!r}")

    expected_bools = {
        "accelerated_weighted_transport_enable": True,
        "primary_sampling_budget_enabled": True,
        "history_thinning_enabled": history_is_thinned,
        "transport_tally_weighted": history_is_thinned,
        "weighted_transport": history_is_thinned,
        "source_bias_weighted_transport": False,
        "poisson_background": True,
        "theory_tvl_attenuation": False,
        "multithreaded_run_manager": True,
    }
    for key, expected in expected_bools.items():
        try:
            observed = _metadata_bool(metadata.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key}: missing or invalid boolean")
            continue
        if observed != expected:
            errors.append(f"{key}: expected {expected}, got {observed}")

    try:
        fraction = _finite_float(metadata["primary_sampling_fraction"])
        requested_fraction = _finite_float(
            metadata["requested_primary_sampling_fraction"]
        )
        target_sampled = _finite_float(metadata["target_sampled_primaries"])
        weight = _finite_float(metadata["primary_history_weight"])
        expected_unthinned = _finite_float(metadata["expected_unthinned_primaries"])
        expected_detector = _finite_float(
            metadata["expected_detector_equivalent_primaries"]
        )
        expected_sampled = _finite_float(metadata["expected_sampled_primaries"])
        tau = _finite_float(metadata["dead_time_tau_s"])
        scale = _finite_float(metadata["dead_time_observed_scale"])
        dwell = _finite_float(metadata["dwell_time_s"])
        pre_total = _finite_float(metadata["pre_dead_time_total_spectrum_counts"])
        pre_sumw2 = _finite_float(metadata["pre_dead_time_weighted_spectrum_sumw2"])
        post_sumw2 = _finite_float(metadata["weighted_spectrum_sumw2"])
        effective = _finite_float(metadata["weighted_spectrum_effective_entries"])
        post_total = _finite_float(metadata["total_spectrum_counts"])
        actual_primaries = _finite_float(record["num_primaries"])
        requested_threads = int(metadata["requested_threads"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"missing or invalid numeric provenance: {exc}")
        return errors

    if not np.isclose(
        requested_fraction,
        EXPECTED_MAX_SAMPLING_FRACTION,
        rtol=0.0,
        atol=1.0e-12,
    ):
        errors.append(
            "requested_primary_sampling_fraction: expected "
            f"{EXPECTED_MAX_SAMPLING_FRACTION}, got {requested_fraction}"
        )
    if not np.isclose(
        target_sampled,
        EXPECTED_TARGET_SAMPLED_PRIMARIES,
        rtol=0.0,
        atol=1.0e-6,
    ):
        errors.append(
            "target_sampled_primaries: expected "
            f"{EXPECTED_TARGET_SAMPLED_PRIMARIES}, got {target_sampled}"
        )
    expected_fraction = max(
        MIN_SAMPLING_FRACTION,
        min(
            requested_fraction,
            target_sampled / max(expected_unthinned, 1.0e-300),
        ),
    )
    if not np.isclose(fraction, expected_fraction, rtol=1.0e-12, atol=1.0e-12):
        errors.append(
            "primary_sampling_fraction is inconsistent with the configured "
            "maximum and sampled-primary budget"
        )
    expected_resolution = (
        "target_budget_limited"
        if target_sampled / max(expected_unthinned, 1.0e-300)
        < requested_fraction
        else "maximum_fraction_limited"
    )
    if metadata.get("primary_sampling_fraction_resolution") != expected_resolution:
        errors.append(
            "primary_sampling_fraction_resolution: expected "
            f"{expected_resolution!r}, got "
            f"{metadata.get('primary_sampling_fraction_resolution')!r}"
        )
    if requested_threads != 32:
        errors.append(f"requested_threads: expected 32, got {requested_threads}")
    if not np.isclose(weight, 1.0 / fraction, rtol=1.0e-12, atol=1.0e-12):
        errors.append("primary_history_weight is not reciprocal to sampling fraction")
    if not np.isclose(
        expected_unthinned,
        expected_detector,
        rtol=1.0e-12,
        atol=1.0e-6,
    ):
        errors.append("unthinned and detector-equivalent primary expectations differ")
    if not np.isclose(
        expected_sampled,
        expected_unthinned * fraction,
        rtol=1.0e-12,
        atol=1.0e-6,
    ):
        errors.append("sampled-primary expectation is inconsistent with thinning")
    if tau < 0.0 or dwell <= 0.0 or not 0.0 < scale <= 1.0:
        errors.append("dead-time constants are outside their physical ranges")
    expected_scale = 1.0 / (1.0 + pre_total * tau / max(dwell, 1.0e-300))
    if not np.isclose(scale, expected_scale, rtol=1.0e-10, atol=1.0e-12):
        errors.append("dead-time observed scale is internally inconsistent")
    if (
        min(pre_total, pre_sumw2, post_sumw2, effective, post_total, actual_primaries)
        < 0.0
    ):
        errors.append("count or variance provenance contains a negative value")
    if not np.isclose(
        post_sumw2,
        pre_sumw2 * scale**2,
        rtol=1.0e-10,
        atol=1.0e-6,
    ):
        errors.append("post-dead-time sumw2 is inconsistent with pre-dead-time sumw2")
    expected_effective = post_total**2 / max(post_sumw2, 1.0e-300)
    if post_sumw2 > 0.0 and not np.isclose(
        effective,
        expected_effective,
        rtol=1.0e-8,
        atol=1.0e-6,
    ):
        errors.append("weighted effective entries are inconsistent with sumw2")
    raw_total = _finite_float(record.get("raw_total_spectrum_counts", post_total))
    if not np.isclose(raw_total, post_total, rtol=1.0e-9, atol=1.0e-3):
        errors.append("recorded raw total disagrees with transport metadata")
    errors.extend(_likelihood_semantics_errors(record))
    return errors


def analyze_transport_provenance(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate strict per-case weighted-transport provenance checks."""
    failures: list[dict[str, object]] = []
    for record in records:
        errors = _transport_metadata_errors(record)
        if errors:
            failures.append({"case": _case_name(record), "errors": errors})
    return {
        "checked_cases": len(records),
        "valid_cases": len(records) - len(failures),
        "all_valid": bool(records) and not failures,
        "failures": failures,
    }


def _full_reference_provenance_errors(
    record: Mapping[str, Any],
) -> list[str]:
    """Validate that one full-history reference uses the current covariance path."""
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ["metadata is missing"]
    errors: list[str] = []
    expected_strings = {
        "backend": "geant4",
        "engine_mode": "external",
        "source_rate_model": "detector_cps_1m",
        "transport_history_mode": "full_unit_weight",
        "spectrum_variance_semantics": ("compound_poisson_sumw2_includes_counting"),
        "spectrum_variance_dead_time_propagation": "fixed_observed_scale",
    }
    for key, expected in expected_strings.items():
        if metadata.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {metadata.get(key)!r}")
    expected_bools = {
        "history_thinning_enabled": False,
        "transport_tally_weighted": False,
        "weighted_transport": False,
        "source_bias_weighted_transport": False,
        "multithreaded_run_manager": True,
    }
    for key, expected in expected_bools.items():
        try:
            observed = _metadata_bool(metadata.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key}: missing or invalid boolean")
            continue
        if observed != expected:
            errors.append(f"{key}: expected {expected}, got {observed}")
    try:
        fraction = _finite_float(metadata["primary_sampling_fraction"])
        weight = _finite_float(metadata["primary_history_weight"])
        requested_threads = int(metadata["requested_threads"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"missing or invalid full-history numeric provenance: {exc}")
    else:
        if not np.isclose(fraction, 1.0, rtol=0.0, atol=1.0e-12):
            errors.append(f"primary_sampling_fraction: expected 1, got {fraction}")
        if not np.isclose(weight, 1.0, rtol=0.0, atol=1.0e-12):
            errors.append(f"primary_history_weight: expected 1, got {weight}")
        if requested_threads != 32:
            errors.append(f"requested_threads: expected 32, got {requested_threads}")
    diagnostics = record.get("response_poisson_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        errors.append("response_poisson_diagnostics is missing")
    else:
        semantics = str(diagnostics.get("count_covariance_semantics", ""))
        if not semantics.startswith("compound_poisson_sumw2_"):
            errors.append(
                "response count covariance does not use current compound-Poisson path"
            )
        if (
            diagnostics.get("count_covariance_complete_transport_provenance")
            is not True
        ):
            errors.append(
                "response count covariance lacks complete transport provenance"
            )
    return errors


def analyze_full_reference_provenance(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate current full-history covariance provenance checks."""
    failures: list[dict[str, object]] = []
    for record in records:
        errors = _full_reference_provenance_errors(record)
        if errors:
            failures.append({"case": _case_name(record), "errors": errors})
    return {
        "checked_cases": len(records),
        "valid_cases": len(records) - len(failures),
        "all_valid": bool(records) and not failures,
        "failures": failures,
    }


def analyze_sampling_quality(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Summarize sampled-primary and weighted-effective-history quality."""
    expected_values: list[float] = []
    effective_values: list[float] = []
    detected_effective_values: list[float] = []
    poisson_z_values: list[float] = []
    invalid_cases: list[str] = []
    for record in records:
        metadata = record.get("metadata", {})
        try:
            if not isinstance(metadata, Mapping):
                raise ValueError("metadata is missing")
            expected = _finite_float(metadata["expected_sampled_primaries"])
            actual = _finite_float(record["num_primaries"])
            effective = _finite_float(metadata["weighted_spectrum_effective_entries"])
            history_weight = _finite_float(metadata["primary_history_weight"])
            dead_time_scale = _finite_float(metadata["dead_time_observed_scale"])
            if expected <= 0.0 or effective < 0.0:
                raise ValueError(
                    "nonpositive expectation or negative effective entries"
                )
            if history_weight <= 0.0 or dead_time_scale <= 0.0:
                raise ValueError("invalid history weight or dead-time scale")
            case_detected_effective = [
                _finite_float(metadata[f"transport_detected_counts_{isotope}"])
                / (history_weight * dead_time_scale)
                for isotope in ISOTOPES
            ]
        except (KeyError, TypeError, ValueError):
            invalid_cases.append(_case_name(record))
            continue
        expected_values.append(expected)
        effective_values.append(effective)
        detected_effective_values.extend(case_detected_effective)
        poisson_z_values.append((actual - expected) / np.sqrt(expected))
    return {
        "invalid_cases": invalid_cases,
        "expected_sampled_primaries": _quantiles(expected_values),
        "effective_entries": _quantiles(effective_values),
        "isotope_detected_effective_entries": _quantiles(detected_effective_values),
        "primary_poisson_z": _quantiles(poisson_z_values),
        "max_abs_primary_poisson_z": (
            float(np.max(np.abs(poisson_z_values))) if poisson_z_values else None
        ),
    }


def _formal_covariance(
    record: Mapping[str, Any],
) -> tuple[NDArray[np.float64], list[str]]:
    """Read one formal response covariance in the canonical isotope order."""
    payload = record.get("response_poisson_covariance", {})
    if not isinstance(payload, Mapping):
        raise ValueError("response_poisson_covariance is missing")
    formal = payload.get("formal", {})
    if not isinstance(formal, Mapping):
        raise ValueError("formal covariance is missing")
    matrix = np.empty((len(ISOTOPES), len(ISOTOPES)), dtype=float)
    for row_index, row_name in enumerate(ISOTOPES):
        row = formal.get(row_name, {})
        if not isinstance(row, Mapping):
            raise ValueError(f"formal covariance row {row_name} is missing")
        for column_index, column_name in enumerate(ISOTOPES):
            matrix[row_index, column_index] = _finite_float(row[column_name])
    return matrix, list(ISOTOPES)


def _covariance_health(matrix: NDArray[np.float64]) -> dict[str, object]:
    """Return finite, symmetry, positive-diagonal, and PSD diagnostics."""
    finite = bool(np.all(np.isfinite(matrix)))
    symmetric = bool(
        finite and np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-8)
    )
    symmetric_matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = (
        np.linalg.eigvalsh(symmetric_matrix)
        if finite
        else np.full(matrix.shape[0], np.nan)
    )
    scale = max(float(np.max(np.abs(np.diag(symmetric_matrix)))), 1.0)
    tolerance = PSD_RELATIVE_TOLERANCE * scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    positive_diagonal = bool(finite and np.all(np.diag(matrix) > 0.0))
    psd = bool(finite and minimum_eigenvalue >= -tolerance)
    return {
        "finite": finite,
        "symmetric": symmetric,
        "positive_diagonal": positive_diagonal,
        "psd": psd,
        "minimum_eigenvalue": minimum_eigenvalue,
        "psd_absolute_tolerance": tolerance,
        "valid": finite and symmetric and positive_diagonal and psd,
    }


def analyze_covariance_health(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate formal response-covariance health across cases."""
    failures: list[dict[str, object]] = []
    minimum_eigenvalues: list[float] = []
    for record in records:
        try:
            matrix, _ = _formal_covariance(record)
            health = _covariance_health(matrix)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"case": _case_name(record), "error": str(exc)})
            continue
        minimum_eigenvalues.append(float(health["minimum_eigenvalue"]))
        if not bool(health["valid"]):
            failures.append({"case": _case_name(record), "health": health})
    return {
        "checked_cases": len(records),
        "valid_cases": len(records) - len(failures),
        "all_valid": bool(records) and not failures,
        "minimum_eigenvalue": (
            float(min(minimum_eigenvalues)) if minimum_eigenvalues else None
        ),
        "failures": failures,
    }


def _response_counts(record: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return response-Poisson isotope counts in canonical order."""
    diagnostics = record.get("response_poisson_diagnostics", {})
    counts = diagnostics.get("counts", {}) if isinstance(diagnostics, Mapping) else {}
    if not isinstance(counts, Mapping):
        counts = {}
    values: list[float] = []
    for isotope in ISOTOPES:
        value = counts.get(isotope)
        if value is None:
            per_isotope = record.get("per_isotope", {})
            isotope_payload = (
                per_isotope.get(isotope, {}) if isinstance(per_isotope, Mapping) else {}
            )
            methods = (
                isotope_payload.get("method_counts", {})
                if isinstance(isotope_payload, Mapping)
                else {}
            )
            value = (
                methods.get("response_poisson")
                if isinstance(methods, Mapping)
                else None
            )
        values.append(_finite_float(value))
    return np.asarray(values, dtype=float)


def _transport_truth_counts(record: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return Geant4 transport-truth isotope counts in canonical order."""
    per_isotope = record.get("per_isotope", {})
    if not isinstance(per_isotope, Mapping):
        raise ValueError("per_isotope is missing")
    return np.asarray(
        [
            _finite_float(per_isotope[isotope]["transport_truth_counts"])
            for isotope in ISOTOPES
        ],
        dtype=float,
    )


def _quadratic_form(
    residual: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> tuple[float, int]:
    """Return a PSD-aware generalized quadratic form and numerical rank."""
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    threshold = PSD_RELATIVE_TOLERANCE * scale
    positive = eigenvalues > threshold
    if not np.any(positive):
        raise ValueError("covariance has zero numerical rank")
    projected = eigenvectors[:, positive].T @ residual
    value = float(np.sum(projected**2 / eigenvalues[positive]))
    return value, int(np.sum(positive))


def _paired_records(
    accelerated: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return name-aligned accelerated/reference records."""
    accelerated_by_name, _ = _records_by_name(accelerated)
    reference_by_name, _ = _records_by_name(reference)
    return [
        (accelerated_by_name[name], reference_by_name[name])
        for name in sorted(set(accelerated_by_name) & set(reference_by_name))
    ]


def _raw_total_variance(record: Mapping[str, Any]) -> float:
    """Return the post-dead-time sumw2 variance of a raw spectrum total."""
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata is missing")
    variance = _finite_float(metadata["weighted_spectrum_sumw2"])
    if variance <= 0.0:
        raise ValueError("raw-total variance is not positive")
    return variance


def analyze_paired_raw_totals(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, object]:
    """Compare paired raw totals using independent sumw2 variances."""
    differences: list[float] = []
    variances: list[float] = []
    z_values: list[float] = []
    reference_total = 0.0
    invalid_cases: list[str] = []
    for accelerated, reference in pairs:
        try:
            accelerated_total = _finite_float(accelerated["raw_total_spectrum_counts"])
            reference_count = _finite_float(reference["raw_total_spectrum_counts"])
            variance = _raw_total_variance(accelerated) + _raw_total_variance(reference)
            if variance <= 0.0:
                raise ValueError("paired variance is not positive")
        except (KeyError, TypeError, ValueError) as exc:
            invalid_cases.append(f"{_case_name(accelerated)}: {exc}")
            continue
        difference = accelerated_total - reference_count
        differences.append(difference)
        variances.append(variance)
        z_values.append(difference / np.sqrt(variance))
        reference_total += reference_count
    pooled_difference = float(np.sum(differences)) if differences else float("nan")
    pooled_variance = float(np.sum(variances)) if variances else float("nan")
    return {
        "variance_semantics": (
            "independent accelerated and full-history post-dead-time sumw2"
        ),
        "valid_pairs": len(differences),
        "invalid_cases": invalid_cases,
        "pooled_difference": pooled_difference if differences else None,
        "pooled_relative_bias": (
            pooled_difference / reference_total if reference_total > 0.0 else None
        ),
        "pooled_z": (
            pooled_difference / np.sqrt(pooled_variance)
            if differences and pooled_variance > 0.0
            else None
        ),
        "per_case_z": _quantiles(z_values),
        "mean_z2": (float(np.mean(np.square(z_values))) if z_values else None),
        "max_abs_z": (float(np.max(np.abs(z_values))) if z_values else None),
        "nominal_95_coverage": (
            float(np.mean(np.abs(z_values) <= 1.959963984540054)) if z_values else None
        ),
    }


def _residual_calibration_summary(
    rows: Sequence[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ],
) -> dict[str, object]:
    """Summarize multivariate and isotope-wise residual calibration."""
    q_total = 0.0
    dof_total = 0
    invalid_rows = 0
    per_isotope_z: dict[str, list[float]] = {isotope: [] for isotope in ISOTOPES}
    difference_sums = np.zeros(len(ISOTOPES), dtype=float)
    reference_sums = np.zeros(len(ISOTOPES), dtype=float)
    for residual, covariance, reference_counts in rows:
        try:
            q_value, rank = _quadratic_form(residual, covariance)
        except (ValueError, np.linalg.LinAlgError):
            invalid_rows += 1
            continue
        q_total += q_value
        dof_total += rank
        diagonal = np.diag(covariance)
        for index, isotope in enumerate(ISOTOPES):
            if diagonal[index] > 0.0 and np.isfinite(diagonal[index]):
                per_isotope_z[isotope].append(
                    float(residual[index] / np.sqrt(diagonal[index]))
                )
        difference_sums += residual
        reference_sums += reference_counts
    isotope_summary: dict[str, object] = {}
    for index, isotope in enumerate(ISOTOPES):
        values = np.asarray(per_isotope_z[isotope], dtype=float)
        isotope_summary[isotope] = {
            "count": int(values.size),
            "mean_z": float(np.mean(values)) if values.size else None,
            "mean_z2": float(np.mean(values**2)) if values.size else None,
            "max_abs_z": float(np.max(np.abs(values))) if values.size else None,
            "nominal_95_coverage": (
                float(np.mean(np.abs(values) <= 1.959963984540054))
                if values.size
                else None
            ),
            "pooled_relative_bias": (
                float(difference_sums[index] / reference_sums[index])
                if reference_sums[index] > 0.0
                else None
            ),
        }
    return {
        "valid_rows": len(rows) - invalid_rows,
        "invalid_rows": invalid_rows,
        "q_total": q_total,
        "dof_total": dof_total,
        "q_per_dof": q_total / dof_total if dof_total > 0 else None,
        "per_isotope": isotope_summary,
    }


def analyze_paired_response(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, object]:
    """Compare paired response estimates with both formal covariances."""
    rows: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    invalid_cases: list[str] = []
    for accelerated, reference in pairs:
        try:
            accelerated_counts = _response_counts(accelerated)
            reference_counts = _response_counts(reference)
            accelerated_covariance, _ = _formal_covariance(accelerated)
            reference_covariance, _ = _formal_covariance(reference)
            covariance = accelerated_covariance + reference_covariance
            if not bool(_covariance_health(covariance)["valid"]):
                raise ValueError("sum of paired formal covariances is invalid")
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            invalid_cases.append(f"{_case_name(accelerated)}: {exc}")
            continue
        rows.append(
            (
                accelerated_counts - reference_counts,
                covariance,
                reference_counts,
            )
        )
    summary = _residual_calibration_summary(rows)
    summary["invalid_cases"] = invalid_cases
    summary["covariance_semantics"] = (
        "accelerated formal covariance plus independent full-history formal covariance"
    )
    return summary


def _fresh_error_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Summarize PF-forward and response errors against transport truth."""
    forward: dict[str, list[float]] = {isotope: [] for isotope in ISOTOPES}
    response: dict[str, list[float]] = {isotope: [] for isotope in ISOTOPES}
    rows: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    invalid_cases: list[str] = []
    likelihood_invalid_cases: list[str] = []
    likelihood_scales: list[float] = []
    likelihood_squared_distance = 0.0
    likelihood_degrees_of_freedom = 0
    likelihood_normalized: dict[str, list[float]] = {
        isotope: [] for isotope in ISOTOPES
    }
    for record in records:
        try:
            truth = _transport_truth_counts(record)
            estimated = _response_counts(record)
            covariance, _ = _formal_covariance(record)
            per_isotope = record["per_isotope"]
            case_forward: list[float] = []
            case_response: list[float] = []
            for index, isotope in enumerate(ISOTOPES):
                isotope_payload = per_isotope[isotope]
                case_forward.append(
                    _finite_float(
                        isotope_payload["pf_target_relative_error_vs_transport_truth"]
                    )
                )
                case_response.append(
                    float((estimated[index] - truth[index]) / max(truth[index], 1.0))
                )
        except (KeyError, TypeError, ValueError) as exc:
            invalid_cases.append(f"{_case_name(record)}: {exc}")
        else:
            for index, isotope in enumerate(ISOTOPES):
                forward[isotope].append(case_forward[index])
                response[isotope].append(case_response[index])
            rows.append((estimated - truth, covariance, truth))
        try:
            diagnostics = record["pf_count_likelihood_diagnostics"]
            target = diagnostics["targets"]["runtime_pf_forward"]
            scales = target["likelihood_scale_over_target_by_isotope"]
            normalized = target["normalized_residual_by_isotope"]
            case_squared_distance = _finite_float(target["diagonal_squared_distance"])
            case_degrees_of_freedom = int(target["degrees_of_freedom"])
            case_scales = [_finite_float(scales[isotope]) for isotope in ISOTOPES]
            case_normalized = [
                _finite_float(normalized[isotope]) for isotope in ISOTOPES
            ]
            if case_squared_distance < 0.0 or case_degrees_of_freedom != len(ISOTOPES):
                raise ValueError(
                    "likelihood squared distance or three-isotope degrees of "
                    "freedom is invalid"
                )
        except (KeyError, TypeError, ValueError) as exc:
            likelihood_invalid_cases.append(f"{_case_name(record)}: {exc}")
        else:
            likelihood_squared_distance += case_squared_distance
            likelihood_degrees_of_freedom += case_degrees_of_freedom
            likelihood_scales.extend(case_scales)
            for index, isotope in enumerate(ISOTOPES):
                likelihood_normalized[isotope].append(case_normalized[index])

    def summarize(values: Mapping[str, Sequence[float]]) -> dict[str, object]:
        """Summarize signed relative-error samples by isotope."""
        output: dict[str, object] = {}
        for isotope in ISOTOPES:
            array = np.asarray(values[isotope], dtype=float)
            output[isotope] = {
                "count": int(array.size),
                "signed_mean": float(np.mean(array)) if array.size else None,
                "mean_abs": float(np.mean(np.abs(array))) if array.size else None,
                "median_abs": float(np.median(np.abs(array))) if array.size else None,
                "p95_abs": (
                    float(np.percentile(np.abs(array), 95.0)) if array.size else None
                ),
                "max_abs": float(np.max(np.abs(array))) if array.size else None,
            }
        pooled = np.concatenate(
            [np.asarray(values[isotope], dtype=float) for isotope in ISOTOPES]
        )
        output["pooled"] = {
            "count": int(pooled.size),
            "signed_mean": float(np.mean(pooled)) if pooled.size else None,
            "mean_abs": float(np.mean(np.abs(pooled))) if pooled.size else None,
            "median_abs": float(np.median(np.abs(pooled))) if pooled.size else None,
            "p95_abs": (
                float(np.percentile(np.abs(pooled), 95.0)) if pooled.size else None
            ),
            "max_abs": float(np.max(np.abs(pooled))) if pooled.size else None,
        }
        return output

    normalized_summary: dict[str, object] = {}
    for isotope in ISOTOPES:
        values = np.asarray(likelihood_normalized[isotope], dtype=float)
        normalized_summary[isotope] = {
            "count": int(values.size),
            "rms": float(np.sqrt(np.mean(values**2))) if values.size else None,
            "mean_abs": float(np.mean(np.abs(values))) if values.size else None,
        }

    return {
        "invalid_cases": invalid_cases,
        "pf_forward_relative_error": summarize(forward),
        "response_relative_error": summarize(response),
        "response_formal_scale_diagnostic": {
            **_residual_calibration_summary(rows),
            "interpretation": (
                "diagnostic only: response estimates and transport truth share "
                "histories, while their cross-covariance is not retained"
            ),
        },
        "pf_likelihood_scale": {
            "invalid_cases": likelihood_invalid_cases,
            "scale_over_target": _quantiles(likelihood_scales),
            "squared_distance_total": float(likelihood_squared_distance),
            "degrees_of_freedom_total": int(likelihood_degrees_of_freedom),
            "squared_distance_per_dof": (
                float(likelihood_squared_distance / likelihood_degrees_of_freedom)
                if likelihood_degrees_of_freedom > 0
                else None
            ),
            "normalized_residual_by_isotope": normalized_summary,
        },
    }


def _fresh_completion(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Validate one complete 8-by-8 shield-pair sweep."""
    pair_ids: list[int] = []
    base_names: set[str] = set()
    invalid_cases: list[str] = []
    for record in records:
        case = record.get("case", {})
        try:
            if not isinstance(case, Mapping):
                raise ValueError("case is missing")
            fe_index = int(case["fe_index"])
            pb_index = int(case["pb_index"])
            if not 0 <= fe_index < 8 or not 0 <= pb_index < 8:
                raise ValueError("shield index is outside [0, 7]")
            pair_ids.append(fe_index * 8 + pb_index)
            base_names.add(str(case.get("name", "")).split("_pair", maxsplit=1)[0])
        except (KeyError, TypeError, ValueError) as exc:
            invalid_cases.append(f"{_case_name(record)}: {exc}")
    unique_pairs = sorted(set(pair_ids))
    return {
        "observed_cases": len(records),
        "expected_cases": EXPECTED_FRESH_CASES,
        "invalid_cases": invalid_cases,
        "base_scenarios": sorted(base_names),
        "unique_pair_count": len(unique_pairs),
        "missing_pair_ids": sorted(set(range(64)) - set(unique_pairs)),
        "duplicate_pair_count": len(pair_ids) - len(unique_pairs),
        "complete": bool(
            len(records) == EXPECTED_FRESH_CASES
            and not invalid_cases
            and len(base_names) == 1
            and unique_pairs == list(range(64))
        ),
    }


def _base_gates(
    records: Sequence[Mapping[str, Any]],
    speed: Mapping[str, Any],
    provenance: Mapping[str, Any],
    sampling: Mapping[str, Any],
    covariance: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Build gates common to paired and fresh analyses."""
    expected_quantiles = sampling["expected_sampled_primaries"]
    effective_quantiles = sampling["effective_entries"]
    detected_effective_quantiles = sampling["isotope_detected_effective_entries"]
    primary_z = sampling["max_abs_primary_poisson_z"]
    valid_sampling = not sampling["invalid_cases"] and len(records) > 0
    return {
        "transport_provenance": _gate(
            provenance["valid_cases"],
            passed=bool(provenance["all_valid"]),
            criterion="all accelerated cases have strict weighted-transport provenance",
        ),
        "minimum_expected_sampled_primaries": _gate(
            expected_quantiles["min"],
            passed=bool(
                valid_sampling
                and expected_quantiles["min"] is not None
                and expected_quantiles["min"] >= MIN_EXPECTED_SAMPLED_PRIMARIES
            ),
            criterion=f">= {MIN_EXPECTED_SAMPLED_PRIMARIES:g} per case",
        ),
        "minimum_effective_entries": _gate(
            effective_quantiles["min"],
            passed=bool(
                valid_sampling
                and effective_quantiles["min"] is not None
                and effective_quantiles["min"] >= MIN_EFFECTIVE_ENTRIES
            ),
            criterion=f">= {MIN_EFFECTIVE_ENTRIES:g} per case",
        ),
        "minimum_isotope_detected_effective_entries": _gate(
            detected_effective_quantiles["min"],
            passed=bool(
                valid_sampling
                and detected_effective_quantiles["min"] is not None
                and detected_effective_quantiles["min"]
                >= MIN_ISOTOPE_DETECTED_EFFECTIVE_ENTRIES
            ),
            criterion=(
                f">= {MIN_ISOTOPE_DETECTED_EFFECTIVE_ENTRIES:g} per case and isotope"
            ),
        ),
        "sampled_primary_poisson_consistency": _gate(
            primary_z,
            passed=bool(
                valid_sampling
                and primary_z is not None
                and primary_z <= MAX_ABS_PRIMARY_POISSON_Z
            ),
            criterion=f"maximum absolute z <= {MAX_ABS_PRIMARY_POISSON_Z:g}",
        ),
        "runtime_p50": _gate(
            speed["p50"],
            passed=bool(speed["p50"] is not None and speed["p50"] <= MAX_RUNTIME_P50_S),
            criterion=f"<= {MAX_RUNTIME_P50_S:g} s",
        ),
        "runtime_p95": _gate(
            speed["p95"],
            passed=bool(speed["p95"] is not None and speed["p95"] <= MAX_RUNTIME_P95_S),
            criterion=f"<= {MAX_RUNTIME_P95_S:g} s",
        ),
        "runtime_max": _gate(
            speed["max"],
            passed=bool(speed["max"] is not None and speed["max"] <= MAX_RUNTIME_S),
            criterion=f"<= {MAX_RUNTIME_S:g} s",
        ),
        "formal_covariance_health": _gate(
            covariance["valid_cases"],
            passed=bool(covariance["all_valid"]),
            criterion="all formal covariance matrices are finite, symmetric, PSD, and positive-diagonal",
        ),
    }


def _paired_gates(
    matching: Mapping[str, Any],
    raw: Mapping[str, Any],
    response: Mapping[str, Any],
    reference_provenance: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Build gates specific to matched accelerated/full-history records."""
    paired_count = int(matching["paired_count"])
    expected_pair_count = int(matching["accelerated_count"])
    distributional = paired_count >= MIN_COVERAGE_SAMPLE_SIZE
    q_per_dof = response["q_per_dof"]
    paired_analysis_complete = bool(
        expected_pair_count > 0
        and paired_count == expected_pair_count
        and int(raw["valid_pairs"]) == expected_pair_count
        and not raw["invalid_cases"]
        and int(response["valid_rows"]) == expected_pair_count
        and int(response["invalid_rows"]) == 0
        and not response["invalid_cases"]
        and int(response["dof_total"]) == expected_pair_count * len(ISOTOPES)
        and all(
            int(response["per_isotope"][isotope]["count"]) == expected_pair_count
            for isotope in ISOTOPES
        )
    )
    gates = {
        "full_reference_provenance": _gate(
            reference_provenance["valid_cases"],
            passed=bool(reference_provenance["all_valid"]),
            criterion=(
                "all supplied references are current unit-weight full-history "
                "records with complete compound-Poisson covariance provenance"
            ),
        ),
        "paired_case_names": _gate(
            matching["paired_count"],
            passed=bool(matching["names_match"]),
            criterion="every accelerated case name has one unique reference match",
        ),
        "paired_physical_cases": _gate(
            len(matching["physical_mismatches"]),
            passed=bool(matching["physical_cases_match"]),
            criterion="all paired physical case payloads are identical",
        ),
        "paired_analysis_completeness": _gate(
            {
                "expected_pairs": expected_pair_count,
                "matched_pairs": paired_count,
                "raw_valid_pairs": raw["valid_pairs"],
                "raw_invalid_cases": len(raw["invalid_cases"]),
                "response_valid_rows": response["valid_rows"],
                "response_invalid_rows": response["invalid_rows"],
                "response_invalid_cases": len(response["invalid_cases"]),
                "response_dof_total": response["dof_total"],
            },
            passed=paired_analysis_complete,
            criterion=(
                "every matched pair yields one raw comparison and one full-rank "
                "three-isotope response comparison without invalid cases or rows"
            ),
        ),
        "raw_total_pooled_z": _gate(
            raw["pooled_z"],
            passed=bool(
                raw["pooled_z"] is not None and abs(raw["pooled_z"]) <= MAX_POOLED_ABS_Z
            ),
            criterion=f"absolute pooled z <= {MAX_POOLED_ABS_Z:g}",
        ),
        "raw_total_pooled_bias": _gate(
            raw["pooled_relative_bias"],
            passed=bool(
                raw["pooled_relative_bias"] is not None
                and abs(raw["pooled_relative_bias"]) <= MAX_POOLED_ABS_RELATIVE_BIAS
            ),
            criterion=(
                f"absolute pooled relative bias <= {MAX_POOLED_ABS_RELATIVE_BIAS:g}"
            ),
        ),
        "response_q_per_dof": _gate(
            q_per_dof,
            passed=bool(
                q_per_dof is not None
                and (
                    MIN_RESPONSE_Q_PER_DOF <= q_per_dof <= MAX_RESPONSE_Q_PER_DOF
                    if distributional
                    else q_per_dof <= MAX_SMALL_SAMPLE_RESPONSE_Q_PER_DOF
                )
            ),
            criterion=(
                f"{MIN_RESPONSE_Q_PER_DOF:g} <= paired formal-covariance "
                f"Q/dof <= {MAX_RESPONSE_Q_PER_DOF:g}"
                if distributional
                else (
                    "small-sample gross check: paired formal-covariance "
                    f"Q/dof <= {MAX_SMALL_SAMPLE_RESPONSE_Q_PER_DOF:g}"
                )
            ),
        ),
        "raw_total_max_abs_z": _gate(
            raw["max_abs_z"],
            passed=bool(
                raw["max_abs_z"] is not None
                and raw["max_abs_z"] <= MAX_INDIVIDUAL_ABS_Z
            ),
            criterion=f"maximum individual absolute z <= {MAX_INDIVIDUAL_ABS_Z:g}",
        ),
    }
    raw_count = int(raw["valid_pairs"])
    gates["raw_total_mean_z2"] = _gate(
        raw["mean_z2"],
        passed=bool(
            raw["mean_z2"] is not None
            and MIN_RESPONSE_Q_PER_DOF <= raw["mean_z2"] <= MAX_RESPONSE_Q_PER_DOF
        ),
        criterion=(
            f"{MIN_RESPONSE_Q_PER_DOF:g} <= mean z-squared "
            f"<= {MAX_RESPONSE_Q_PER_DOF:g}"
        ),
        applicable=raw_count >= MIN_COVERAGE_SAMPLE_SIZE,
        detail=f"requires at least {MIN_COVERAGE_SAMPLE_SIZE} paired cases",
    )
    gates["raw_total_nominal_95_coverage"] = _gate(
        raw["nominal_95_coverage"],
        passed=bool(
            raw["nominal_95_coverage"] is not None
            and MIN_NOMINAL_95_COVERAGE
            <= raw["nominal_95_coverage"]
            <= MAX_NOMINAL_95_COVERAGE
        ),
        criterion=(
            f"{MIN_NOMINAL_95_COVERAGE:g} <= coverage <= {MAX_NOMINAL_95_COVERAGE:g}"
        ),
        applicable=raw_count >= MIN_COVERAGE_SAMPLE_SIZE,
        detail=f"requires at least {MIN_COVERAGE_SAMPLE_SIZE} paired cases",
    )
    isotope_payload = response["per_isotope"]
    for isotope in ISOTOPES:
        values = isotope_payload[isotope]
        prefix = isotope.replace("-", "_").lower()
        gates[f"response_{prefix}_mean_z2"] = _gate(
            values["mean_z2"],
            passed=bool(
                values["mean_z2"] is not None
                and MIN_ISOTOPE_MEAN_Z2 <= values["mean_z2"] <= MAX_ISOTOPE_MEAN_Z2
            ),
            criterion=(
                f"{MIN_ISOTOPE_MEAN_Z2:g} <= mean z-squared <= {MAX_ISOTOPE_MEAN_Z2:g}"
            ),
            applicable=int(values["count"]) >= MIN_COVERAGE_SAMPLE_SIZE,
            detail=(
                f"distributional z-squared gate requires at least "
                f"{MIN_COVERAGE_SAMPLE_SIZE} paired cases"
            ),
        )
        gates[f"response_{prefix}_max_abs_z"] = _gate(
            values["max_abs_z"],
            passed=bool(
                values["max_abs_z"] is not None
                and values["max_abs_z"] <= MAX_INDIVIDUAL_ABS_Z
            ),
            criterion=f"maximum individual absolute z <= {MAX_INDIVIDUAL_ABS_Z:g}",
        )
        gates[f"response_{prefix}_pooled_bias"] = _gate(
            values["pooled_relative_bias"],
            passed=bool(
                values["pooled_relative_bias"] is not None
                and abs(values["pooled_relative_bias"]) <= MAX_ISOTOPE_ABS_RELATIVE_BIAS
            ),
            criterion=f"absolute relative bias <= {MAX_ISOTOPE_ABS_RELATIVE_BIAS:g}",
        )
        gates[f"response_{prefix}_nominal_95_coverage"] = _gate(
            values["nominal_95_coverage"],
            passed=bool(
                values["nominal_95_coverage"] is not None
                and MIN_NOMINAL_95_COVERAGE
                <= values["nominal_95_coverage"]
                <= MAX_NOMINAL_95_COVERAGE
            ),
            criterion=(
                f"{MIN_NOMINAL_95_COVERAGE:g} <= coverage "
                f"<= {MAX_NOMINAL_95_COVERAGE:g}"
            ),
            applicable=int(values["count"]) >= MIN_COVERAGE_SAMPLE_SIZE,
            detail=f"requires at least {MIN_COVERAGE_SAMPLE_SIZE} paired cases",
        )
    return gates


def _fresh_gates(
    completion: Mapping[str, Any],
    accuracy: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Build gates specific to a fresh independent 64-pair validation."""
    expected_cases = EXPECTED_FRESH_CASES
    expected_isotope_rows = expected_cases * len(ISOTOPES)
    covariance_calibration = accuracy["response_formal_scale_diagnostic"]
    likelihood = accuracy["pf_likelihood_scale"]
    forward_summary = accuracy["pf_forward_relative_error"]
    response_summary = accuracy["response_relative_error"]
    analysis_complete = bool(
        not accuracy["invalid_cases"]
        and not likelihood["invalid_cases"]
        and int(forward_summary["pooled"]["count"]) == expected_isotope_rows
        and int(response_summary["pooled"]["count"]) == expected_isotope_rows
        and all(
            int(forward_summary[isotope]["count"]) == expected_cases
            and int(response_summary[isotope]["count"]) == expected_cases
            and int(likelihood["normalized_residual_by_isotope"][isotope]["count"])
            == expected_cases
            for isotope in ISOTOPES
        )
        and int(covariance_calibration["valid_rows"]) == expected_cases
        and int(covariance_calibration["invalid_rows"]) == 0
        and int(covariance_calibration["dof_total"]) == expected_isotope_rows
        and int(likelihood["scale_over_target"]["count"]) == expected_isotope_rows
        and int(likelihood["degrees_of_freedom_total"]) == expected_isotope_rows
    )
    gates = {
        "fresh_all_64_shield_pairs": _gate(
            completion["observed_cases"],
            passed=bool(completion["complete"]),
            criterion="exactly one physical scenario with every Fe/Pb pair 0..63",
        ),
        "fresh_analysis_completeness": _gate(
            {
                "expected_cases": expected_cases,
                "accuracy_invalid_cases": len(accuracy["invalid_cases"]),
                "likelihood_invalid_cases": len(likelihood["invalid_cases"]),
                "forward_rows": forward_summary["pooled"]["count"],
                "response_rows": response_summary["pooled"]["count"],
                "response_covariance_valid_rows": covariance_calibration["valid_rows"],
                "response_covariance_invalid_rows": covariance_calibration[
                    "invalid_rows"
                ],
                "response_covariance_dof": covariance_calibration["dof_total"],
                "likelihood_scale_rows": likelihood["scale_over_target"]["count"],
                "likelihood_dof": likelihood["degrees_of_freedom_total"],
            },
            passed=analysis_complete,
            criterion=(
                "all 64 cases yield complete three-isotope accuracy, covariance, "
                "and PF-likelihood rows with 192 total degrees of freedom"
            ),
        ),
    }
    forward = accuracy["pf_forward_relative_error"]["pooled"]
    response = accuracy["response_relative_error"]["pooled"]
    pooled_limits = {
        "pf_forward": (
            forward,
            MAX_PF_FORWARD_MEAN_ABS_RELATIVE_ERROR,
            MAX_PF_FORWARD_MEDIAN_ABS_RELATIVE_ERROR,
            MAX_PF_FORWARD_P95_ABS_RELATIVE_ERROR,
            MAX_PF_FORWARD_ABS_RELATIVE_ERROR,
        ),
        "response": (
            response,
            MAX_RESPONSE_MEAN_ABS_RELATIVE_ERROR,
            MAX_RESPONSE_MEDIAN_ABS_RELATIVE_ERROR,
            MAX_RESPONSE_P95_ABS_RELATIVE_ERROR,
            MAX_RESPONSE_ABS_RELATIVE_ERROR,
        ),
    }
    for prefix, (
        values,
        mean_limit,
        median_limit,
        p95_limit,
        max_limit,
    ) in pooled_limits.items():
        for metric_name, limit in (
            ("mean_abs", mean_limit),
            ("median_abs", median_limit),
            ("p95_abs", p95_limit),
            ("max_abs", max_limit),
        ):
            value = values[metric_name]
            gates[f"fresh_{prefix}_{metric_name}"] = _gate(
                value,
                passed=bool(value is not None and value <= limit),
                criterion=f"<= {limit:g}",
            )
    for isotope in ISOTOPES:
        values = accuracy["response_relative_error"][isotope]
        prefix = isotope.replace("-", "_").lower()
        gates[f"fresh_response_{prefix}_mean_abs"] = _gate(
            values["mean_abs"],
            passed=bool(
                values["mean_abs"] is not None
                and values["mean_abs"] <= MAX_RESPONSE_ISOTOPE_MEAN_ABS_RELATIVE_ERROR
            ),
            criterion=f"<= {MAX_RESPONSE_ISOTOPE_MEAN_ABS_RELATIVE_ERROR:g}",
        )
        gates[f"fresh_response_{prefix}_p95_abs"] = _gate(
            values["p95_abs"],
            passed=bool(
                values["p95_abs"] is not None
                and values["p95_abs"] <= MAX_RESPONSE_ISOTOPE_P95_ABS_RELATIVE_ERROR
            ),
            criterion=f"<= {MAX_RESPONSE_ISOTOPE_P95_ABS_RELATIVE_ERROR:g}",
        )
    scale = likelihood["scale_over_target"]
    gates["fresh_likelihood_scale_median"] = _gate(
        scale["p50"],
        passed=bool(
            scale["p50"] is not None
            and MIN_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN
            <= scale["p50"]
            <= MAX_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN
        ),
        criterion=(
            f"{MIN_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN:g} <= median "
            f"<= {MAX_LIKELIHOOD_SCALE_OVER_TARGET_MEDIAN:g}"
        ),
    )
    gates["fresh_likelihood_scale_p95"] = _gate(
        scale["p95"],
        passed=bool(
            scale["p95"] is not None
            and scale["p95"] <= MAX_LIKELIHOOD_SCALE_OVER_TARGET_P95
        ),
        criterion=f"<= {MAX_LIKELIHOOD_SCALE_OVER_TARGET_P95:g}",
    )
    gates["fresh_likelihood_squared_distance_per_dof"] = _gate(
        likelihood["squared_distance_per_dof"],
        passed=bool(
            likelihood["squared_distance_per_dof"] is not None
            and likelihood["squared_distance_per_dof"]
            <= MAX_LIKELIHOOD_SQUARED_DISTANCE_PER_DOF
            and not likelihood["invalid_cases"]
        ),
        criterion=f"<= {MAX_LIKELIHOOD_SQUARED_DISTANCE_PER_DOF:g}",
    )
    for isotope in ISOTOPES:
        values = likelihood["normalized_residual_by_isotope"][isotope]
        prefix = isotope.replace("-", "_").lower()
        gates[f"fresh_likelihood_{prefix}_rms"] = _gate(
            values["rms"],
            passed=bool(
                values["rms"] is not None
                and values["rms"] <= MAX_NORMALIZED_RESIDUAL_RMS
            ),
            criterion=f"<= {MAX_NORMALIZED_RESIDUAL_RMS:g}",
        )
        gates[f"fresh_likelihood_{prefix}_mean_abs"] = _gate(
            values["mean_abs"],
            passed=bool(
                values["mean_abs"] is not None
                and values["mean_abs"] <= MAX_NORMALIZED_RESIDUAL_MEAN_ABS
            ),
            criterion=f"<= {MAX_NORMALIZED_RESIDUAL_MEAN_ABS:g}",
        )
    gates["fresh_response_q_per_dof_diagnostic"] = _gate(
        covariance_calibration["q_per_dof"],
        passed=bool(
            covariance_calibration["q_per_dof"] is not None
            and covariance_calibration["q_per_dof"] <= MAX_RESPONSE_Q_PER_DOF
        ),
        criterion=f"response-vs-truth formal-covariance Q/dof <= {MAX_RESPONSE_Q_PER_DOF:g}",
        applicable=False,
        detail=(
            "not an acceptance gate because response/truth cross-covariance from "
            "shared histories is unavailable"
        ),
    )
    return gates


def analyze_acceptance(
    accelerated: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Analyze paired or fresh weighted-transport acceptance without tuning."""
    records = list(accelerated)
    speed = _quantiles(
        [_finite_float(record.get("runtime_s", float("nan"))) for record in records]
    )
    provenance = analyze_transport_provenance(records)
    sampling = analyze_sampling_quality(records)
    covariance = analyze_covariance_health(records)
    gates = _base_gates(records, speed, provenance, sampling, covariance)
    output: dict[str, object] = {
        "schema_version": 1,
        "mode": "paired" if reference is not None else "fresh",
        "thresholds": dict(THRESHOLDS),
        "accelerated_case_count": len(records),
        "speed_runtime_s": speed,
        "transport_provenance": provenance,
        "sampling_quality": sampling,
        "formal_covariance_health": covariance,
    }
    if reference is not None:
        reference_records = list(reference)
        reference_provenance = analyze_full_reference_provenance(reference_records)
        matching = analyze_case_matching(records, reference_records)
        pairs = _paired_records(records, reference_records)
        raw = analyze_paired_raw_totals(pairs)
        paired_response = analyze_paired_response(pairs)
        output.update(
            {
                "full_reference_provenance": reference_provenance,
                "case_matching": matching,
                "paired_raw_total": raw,
                "paired_response": paired_response,
            }
        )
        gates.update(
            _paired_gates(
                matching,
                raw,
                paired_response,
                reference_provenance,
            )
        )
    else:
        completion = _fresh_completion(records)
        accuracy = _fresh_error_summary(records)
        output.update({"completion": completion, "fresh_accuracy": accuracy})
        gates.update(_fresh_gates(completion, accuracy))
    output["gates"] = gates
    output["failed_gates"] = [
        name
        for name, gate in gates.items()
        if bool(gate["applicable"]) and not bool(gate["passed"])
    ]
    output["overall_pass"] = not output["failed_gates"]
    return output


def _load_results(path: Path) -> list[Mapping[str, Any]]:
    """Load a validation ``results.json`` list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(record, Mapping) for record in payload
    ):
        raise ValueError(f"{path} must contain a JSON list of result objects.")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accelerated-results",
        type=Path,
        required=True,
        help="Weighted-transport validation results.json.",
    )
    parser.add_argument(
        "--full-reference",
        type=Path,
        default=None,
        help="Optional matched full-history results.json for paired analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output acceptance JSON path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acceptance analysis and return nonzero when a gate fails."""
    args = parse_args(argv)
    accelerated = _load_results(args.accelerated_results)
    reference = (
        _load_results(args.full_reference) if args.full_reference is not None else None
    )
    report = analyze_acceptance(accelerated, reference)
    report["inputs"] = {
        "accelerated_results": args.accelerated_results.as_posix(),
        "full_reference": (
            args.full_reference.as_posix() if args.full_reference is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if bool(report["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
