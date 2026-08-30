"""Standardized truth-aware cluster accuracy for completed PF runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluation.source_normalization import Source, normalize_sources
from pf.cardinality_policy import (
    HARD_CAP_POSTERIOR_MASS_LIMIT,
    hard_cap_mass_is_acceptable,
)


@dataclass(frozen=True, slots=True)
class ClusterAccuracyCriteria:
    """Declare fixed operational tolerances for cluster-level PF scoring."""

    cluster_assignment_radius_m: float = 0.5
    maximum_position_error_m: float = 0.5
    maximum_relative_strength_error: float = 0.25
    response_indistinguishable_cosine: float = 0.995

    def __post_init__(self) -> None:
        """Reject invalid or non-finite evaluation tolerances."""
        positive = {
            "cluster_assignment_radius_m": self.cluster_assignment_radius_m,
            "maximum_position_error_m": self.maximum_position_error_m,
            "maximum_relative_strength_error": (
                self.maximum_relative_strength_error
            ),
        }
        for name, value in positive.items():
            resolved = float(value)
            if not np.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, resolved)
        probabilities = {
            "response_indistinguishable_cosine": (
                self.response_indistinguishable_cosine
            ),
        }
        for name, value in probabilities.items():
            resolved = float(value)
            if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, resolved)

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return the canonical outcome-independent policy payload."""
        return {
            "policy_name": "cluster_accuracy_without_raw_cardinality_scoring",
            "cluster_assignment_radius_m": self.cluster_assignment_radius_m,
            "maximum_position_error_m": self.maximum_position_error_m,
            "maximum_relative_strength_error": (
                self.maximum_relative_strength_error
            ),
            "response_indistinguishable_cosine": (
                self.response_indistinguishable_cosine
            ),
            "maximum_hard_cap_posterior_mass": HARD_CAP_POSTERIOR_MASS_LIMIT,
            "raw_component_cardinality_is_accuracy_target": False,
        }

    @property
    def maximum_hard_cap_posterior_mass(self) -> float:
        """Return the single hard-cap limit shared with live health checks."""
        return HARD_CAP_POSTERIOR_MASS_LIMIT

    @property
    def sha256(self) -> str:
        """Return the canonical criteria digest recorded with every result."""
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_CLUSTER_ACCURACY_CRITERIA = ClusterAccuracyCriteria()
_EVALUATION_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_family",
        "source_run_id",
        "measurement_log_sha256",
        "hard_max_sources_per_isotope",
        "response_signature_semantics",
        "truth_read",
        "isotopes",
    }
)
_EVALUATION_ISOTOPE_FIELDS = frozenset(
    {
        "mode_label_indices",
        "mode_positions_xyz_m",
        "mode_strengths_cps_1m",
        "normalized_response_signatures_measurement_by_mode",
    }
)
_RESPONSE_SIGNATURE_SEMANTICS = (
    "normalized_same_isotope_expected_count_by_completed_measurement"
)


def _pairwise_euclidean(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a finite pairwise 3-D Euclidean distance matrix."""
    left = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    if np.any(~np.isfinite(distances)):
        raise ValueError("Source positions produced non-finite distances.")
    return distances


def _normalized_columns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return finite unit-norm columns, preserving exact zero columns."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("Response signatures must be a finite matrix.")
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > 0.0,
    )


def _posterior_sources(
    posterior_isotope: Mapping[str, object] | None,
) -> tuple[list[Source], Mapping[str, object]]:
    """Extract ordered PF modes and their cardinality distribution."""
    if posterior_isotope is None:
        return [], {}
    modes = posterior_isotope.get("modes")
    distribution = posterior_isotope.get("cardinality_distribution")
    if not isinstance(modes, Sequence) or isinstance(modes, (str, bytes)):
        raise ValueError("PF posterior modes must be an array.")
    if not isinstance(distribution, Mapping):
        raise ValueError("PF posterior cardinality distribution must be an object.")
    sources = normalize_sources(
        [
            {
                "position": mode.get("position_medoid_xyz"),
                "strength": mode.get("strength_representative_cps_1m"),
            }
            for mode in modes
            if isinstance(mode, Mapping)
        ]
    )
    if len(sources) != len(modes):
        raise ValueError("Every PF posterior mode must be an object.")
    if any(source.strength <= 0.0 for source in sources):
        raise ValueError("Every reported PF source strength must be positive.")
    return sources, distribution


def _validated_signatures(
    evaluation_isotope: Mapping[str, object] | None,
    estimates: Sequence[Source],
) -> NDArray[np.float64]:
    """Validate response columns against the published ordered PF modes."""
    if evaluation_isotope is None:
        if estimates:
            raise ValueError("PF response signatures are missing for estimates.")
        return np.zeros((0, 0), dtype=np.float64)
    if set(evaluation_isotope) != _EVALUATION_ISOTOPE_FIELDS:
        raise ValueError(
            "PF evaluation isotope fields do not match schema version 1."
        )
    raw_labels = evaluation_isotope.get("mode_label_indices")
    raw_positions = evaluation_isotope.get("mode_positions_xyz_m")
    raw_strengths = evaluation_isotope.get("mode_strengths_cps_1m")
    raw_signatures = evaluation_isotope.get(
        "normalized_response_signatures_measurement_by_mode"
    )
    positions = np.asarray(raw_positions, dtype=np.float64)
    strengths = np.asarray(raw_strengths, dtype=np.float64)
    signatures = np.asarray(raw_signatures, dtype=np.float64)
    if not isinstance(raw_labels, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_labels
    ):
        raise ValueError("PF evaluation mode labels must be integer arrays.")
    if len(raw_labels) != len(estimates) or len(set(raw_labels)) != len(raw_labels):
        raise ValueError("PF evaluation mode labels must be unique and aligned.")
    expected_positions = np.asarray(
        [source.pos for source in estimates],
        dtype=np.float64,
    ).reshape(len(estimates), 3)
    expected_strengths = np.asarray(
        [source.strength for source in estimates],
        dtype=np.float64,
    )
    if (
        positions.shape != expected_positions.shape
        or strengths.shape != expected_strengths.shape
        or signatures.ndim != 2
        or signatures.shape[1] != len(estimates)
        or np.any(~np.isfinite(signatures))
        or np.any(signatures < 0.0)
        or not np.array_equal(positions, expected_positions)
        or not np.array_equal(strengths, expected_strengths)
    ):
        raise ValueError(
            "PF evaluation response signatures do not match posterior modes."
        )
    normalized = _normalized_columns(signatures)
    if estimates and np.any(np.linalg.norm(normalized, axis=0) == 0.0):
        raise ValueError("PF evaluation response signatures must be nonzero.")
    return normalized


def _cluster_representative_index(
    estimates: Sequence[Source],
    member_indices: NDArray[np.int64],
) -> int:
    """Return a truth-independent strength-weighted medoid for one cluster."""
    members = np.asarray(member_indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(
        [estimates[int(index)].pos for index in members],
        dtype=np.float64,
    )
    strengths = np.asarray(
        [estimates[int(index)].strength for index in members],
        dtype=np.float64,
    )
    distances = _pairwise_euclidean(positions, positions)
    weights = strengths if float(np.sum(strengths)) > 0.0 else np.ones_like(strengths)
    costs = np.sum(distances * weights[None, :], axis=1)
    return int(members[int(np.argmin(costs))])


def _hard_cap_mass(
    distribution: Mapping[str, object],
    hard_max_sources: int,
) -> float:
    """Return validated posterior mass at the configured hard capacity."""
    total = 0.0
    mass = 0.0
    for raw_cardinality, raw_probability in distribution.items():
        if (
            not isinstance(raw_cardinality, str)
            or not raw_cardinality.isdigit()
            or str(int(raw_cardinality)) != raw_cardinality
            or int(raw_cardinality) > hard_max_sources
        ):
            raise ValueError("Cardinality keys must be canonical nonnegative integers.")
        if isinstance(raw_probability, bool) or not isinstance(
            raw_probability,
            (int, float),
        ):
            raise ValueError("Cardinality probabilities must be numeric.")
        probability = float(raw_probability)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError(
                "Cardinality probabilities must be finite and nonnegative."
            )
        total += probability
        if int(raw_cardinality) == hard_max_sources:
            mass = probability
    if distribution and not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("Cardinality distribution must sum to one.")
    return float(mass)


def compute_cluster_accuracy_evaluation(
    truth_by_isotope: Mapping[str, Sequence[object]],
    posterior_payload: Mapping[str, object],
    evaluation_input_payload: Mapping[str, object],
    *,
    criteria: ClusterAccuracyCriteria = DEFAULT_CLUSTER_ACCURACY_CRITERIA,
) -> dict[str, Any]:
    """Score every true source against a local aggregate of estimated modes.

    The small source-count loops only serialize per-source audit rows; all
    distance, response, and aggregation calculations are vectorized.  Truth is
    an evaluation-only input and cannot alter PF state or posterior cardinality.
    """
    if set(evaluation_input_payload) != _EVALUATION_INPUT_FIELDS:
        raise ValueError("PF evaluation input fields do not match schema version 1.")
    if (
        evaluation_input_payload.get("schema_version") != 1
        or evaluation_input_payload.get("artifact_family")
        != "pf_post_run_cluster_evaluation_input"
        or evaluation_input_payload.get("truth_read") is not False
        or evaluation_input_payload.get("response_signature_semantics")
        != _RESPONSE_SIGNATURE_SEMANTICS
    ):
        raise ValueError("PF evaluation input contract is invalid.")
    source_run_id = evaluation_input_payload.get("source_run_id")
    measurement_log_sha256 = evaluation_input_payload.get(
        "measurement_log_sha256"
    )
    if (
        not isinstance(source_run_id, str)
        or not source_run_id
        or source_run_id.strip() != source_run_id
    ):
        raise ValueError("PF evaluation input must declare a nonempty run_id.")
    if (
        not isinstance(measurement_log_sha256, str)
        or len(measurement_log_sha256) != 64
        or any(value not in "0123456789abcdef" for value in measurement_log_sha256)
    ):
        raise ValueError("PF evaluation input MeasurementLog hash is invalid.")
    posterior_isotopes = posterior_payload.get("isotopes")
    evaluation_isotopes = evaluation_input_payload.get("isotopes")
    hard_max_sources = evaluation_input_payload.get(
        "hard_max_sources_per_isotope"
    )
    if not isinstance(posterior_isotopes, Mapping):
        raise ValueError("PF posterior must contain isotope results.")
    if not isinstance(evaluation_isotopes, Mapping):
        raise ValueError("PF evaluation input must contain isotope signatures.")
    for name, values in (
        ("private truth", truth_by_isotope),
        ("PF posterior", posterior_isotopes),
        ("PF evaluation input", evaluation_isotopes),
    ):
        if any(
            not isinstance(isotope, str) or not isotope
            for isotope in values
        ):
            raise ValueError(f"{name} isotope keys must be nonempty strings.")
    if (
        isinstance(hard_max_sources, bool)
        or not isinstance(hard_max_sources, int)
        or hard_max_sources < 1
    ):
        raise ValueError("PF evaluation input must declare a positive hard cap.")
    isotope_names = sorted(
        set(str(value) for value in truth_by_isotope)
        | set(str(value) for value in posterior_isotopes)
        | set(str(value) for value in evaluation_isotopes)
    )
    isotope_results: dict[str, object] = {}
    all_position_errors: list[float] = []
    all_relative_strength_errors: list[float] = []
    global_failures: list[str] = []
    raw_truth_count = 0
    raw_estimate_count = 0
    for isotope in isotope_names:
        truth = normalize_sources(truth_by_isotope.get(isotope, ()))
        if any(source.strength <= 0.0 for source in truth):
            raise ValueError("Private truth source strengths must be positive.")
        posterior_row = posterior_isotopes.get(isotope)
        if posterior_row is not None and not isinstance(posterior_row, Mapping):
            raise ValueError(f"PF posterior isotope {isotope} must be an object.")
        estimates, cardinality_distribution = _posterior_sources(posterior_row)
        evaluation_row = evaluation_isotopes.get(isotope)
        if evaluation_row is not None and not isinstance(evaluation_row, Mapping):
            raise ValueError(f"PF evaluation isotope {isotope} must be an object.")
        signatures = _validated_signatures(evaluation_row, estimates)
        truth_positions = np.asarray(
            [source.pos for source in truth],
            dtype=np.float64,
        ).reshape(len(truth), 3)
        estimate_positions = np.asarray(
            [source.pos for source in estimates],
            dtype=np.float64,
        ).reshape(len(estimates), 3)
        distances = _pairwise_euclidean(truth_positions, estimate_positions)
        nearest_truth = np.full(len(estimates), -1, dtype=np.int64)
        nearest_distance = np.full(len(estimates), np.inf, dtype=np.float64)
        if truth and estimates:
            nearest_truth = np.argmin(distances, axis=0).astype(np.int64)
            nearest_distance = distances[
                nearest_truth,
                np.arange(len(estimates), dtype=np.int64),
            ]
        assigned = (
            np.isfinite(nearest_distance)
            & (nearest_distance <= criteria.cluster_assignment_radius_m)
        )
        truth_rows: list[dict[str, object]] = []
        cluster_signatures: list[NDArray[np.float64]] = []
        covered_truth_indices: list[int] = []
        isotope_failures: list[str] = []
        isotope_position_errors: list[float] = []
        isotope_relative_strength_errors: list[float] = []
        for truth_index, truth_source in enumerate(truth):
            members = np.flatnonzero(assigned & (nearest_truth == truth_index)).astype(
                np.int64,
                copy=False,
            )
            covered = bool(members.size)
            representative_index: int | None = None
            position_error: float | None = None
            estimated_strength = 0.0
            relative_strength_error: float | None = None
            representative_position: list[float] | None = None
            position_error_vector: list[float] | None = None
            absolute_strength_error: float | None = None
            if covered:
                representative_index = _cluster_representative_index(
                    estimates,
                    members,
                )
                position_error = float(
                    np.linalg.norm(
                        estimates[representative_index].pos - truth_source.pos
                    )
                )
                representative_position = estimates[
                    representative_index
                ].pos.tolist()
                position_error_vector = (
                    estimates[representative_index].pos - truth_source.pos
                ).tolist()
                estimated_strength = float(
                    np.sum(
                        [estimates[int(index)].strength for index in members],
                        dtype=np.float64,
                    )
                )
                relative_strength_error = abs(
                    estimated_strength - truth_source.strength
                ) / truth_source.strength
                absolute_strength_error = abs(
                    estimated_strength - truth_source.strength
                )
                combined_signature = np.sum(
                    signatures[:, members]
                    * np.asarray(
                        [estimates[int(index)].strength for index in members],
                        dtype=np.float64,
                    )[None, :],
                    axis=1,
                    keepdims=True,
                )
                cluster_signatures.append(_normalized_columns(combined_signature)[:, 0])
                covered_truth_indices.append(truth_index)
                all_position_errors.append(position_error)
                all_relative_strength_errors.append(relative_strength_error)
                isotope_position_errors.append(position_error)
                isotope_relative_strength_errors.append(
                    relative_strength_error
                )
            position_passed = bool(
                position_error is not None
                and position_error <= criteria.maximum_position_error_m
            )
            strength_passed = bool(
                relative_strength_error is not None
                and relative_strength_error
                <= criteria.maximum_relative_strength_error
            )
            if not covered:
                isotope_failures.append(f"missing_truth_cluster:{truth_index}")
            else:
                if not position_passed:
                    isotope_failures.append(f"position_error:{truth_index}")
                if not strength_passed:
                    isotope_failures.append(f"strength_error:{truth_index}")
            truth_rows.append(
                {
                    "truth_source_index": truth_index,
                    "truth_position_xyz_m": truth_source.pos.tolist(),
                    "truth_strength_cps_1m": truth_source.strength,
                    "assigned_estimate_indices": [int(value) for value in members],
                    "assigned_raw_component_count": int(members.size),
                    "representative_estimate_index": representative_index,
                    "representative_estimated_position_xyz_m": (
                        representative_position
                    ),
                    "position_error_xyz_m": position_error_vector,
                    "representative_position_error_m": position_error,
                    "combined_estimated_strength_cps_1m": estimated_strength,
                    "combined_absolute_strength_error_cps_1m": (
                        absolute_strength_error
                    ),
                    "combined_relative_strength_error": relative_strength_error,
                    "covered": covered,
                    "position_accuracy_passed": position_passed,
                    "strength_accuracy_passed": strength_passed,
                    "source_accuracy_passed": bool(
                        covered and position_passed and strength_passed
                    ),
                }
            )
        remote_indices = np.flatnonzero(~assigned).astype(np.int64, copy=False)
        cluster_signature_matrix = (
            np.stack(cluster_signatures, axis=1)
            if cluster_signatures
            else np.zeros((signatures.shape[0], 0), dtype=np.float64)
        )
        remote_rows: list[dict[str, object]] = []
        response_distinct_remote_count = 0
        for estimate_index in remote_indices.tolist():
            maximum_cosine = (
                float(np.max(signatures[:, estimate_index] @ cluster_signature_matrix))
                if cluster_signature_matrix.shape[1]
                else 0.0
            )
            response_distinct = bool(
                maximum_cosine < criteria.response_indistinguishable_cosine
            )
            response_distinct_remote_count += int(response_distinct)
            remote_rows.append(
                {
                    "estimate_index": int(estimate_index),
                    "position_xyz_m": estimates[estimate_index].pos.tolist(),
                    "strength_cps_1m": estimates[estimate_index].strength,
                    "nearest_truth_index": (
                        int(nearest_truth[estimate_index])
                        if nearest_truth[estimate_index] >= 0
                        else None
                    ),
                    "nearest_truth_distance_m": (
                        float(nearest_distance[estimate_index])
                        if np.isfinite(nearest_distance[estimate_index])
                        else None
                    ),
                    "maximum_cosine_to_covered_cluster_response": maximum_cosine,
                    "response_distinct": response_distinct,
                }
            )
        if response_distinct_remote_count:
            isotope_failures.append("response_distinct_remote_components")
        hard_cap_mass = _hard_cap_mass(
            cardinality_distribution,
            hard_max_sources,
        )
        hard_cap_passed = hard_cap_mass_is_acceptable(hard_cap_mass)
        if not hard_cap_passed:
            isotope_failures.append("hard_cardinality_cap_saturation")
        isotope_results[isotope] = {
            "passed": not isotope_failures,
            "failure_reasons": isotope_failures,
            "raw_truth_count": len(truth),
            "raw_estimate_component_count": len(estimates),
            "raw_component_cardinality_scored": False,
            "covered_truth_cluster_count": len(covered_truth_indices),
            "response_distinct_remote_component_count": (
                response_distinct_remote_count
            ),
            "hard_cap_posterior_mass": hard_cap_mass,
            "hard_cap_saturation_passed": hard_cap_passed,
            "metrics": {
                "matched_truth_source_count": len(isotope_position_errors),
                "position_error_mean_m": (
                    float(np.mean(isotope_position_errors))
                    if isotope_position_errors
                    else None
                ),
                "position_error_max_m": (
                    float(np.max(isotope_position_errors))
                    if isotope_position_errors
                    else None
                ),
                "relative_strength_error_mean": (
                    float(np.mean(isotope_relative_strength_errors))
                    if isotope_relative_strength_errors
                    else None
                ),
                "relative_strength_error_max": (
                    float(np.max(isotope_relative_strength_errors))
                    if isotope_relative_strength_errors
                    else None
                ),
            },
            "truth_sources": truth_rows,
            "remote_estimates": remote_rows,
        }
        global_failures.extend(f"{isotope}:{reason}" for reason in isotope_failures)
        raw_truth_count += len(truth)
        raw_estimate_count += len(estimates)
    position_array = np.asarray(all_position_errors, dtype=np.float64)
    strength_array = np.asarray(all_relative_strength_errors, dtype=np.float64)
    return {
        "schema_version": 1,
        "artifact_family": "completed_pf_cluster_accuracy_evaluation",
        "criteria": criteria.to_dict(),
        "criteria_sha256": criteria.sha256,
        "truth_used_only_after_completion": True,
        "changes_pf_state_or_cardinality": False,
        "passed": not global_failures,
        "failure_reasons": global_failures,
        "global": {
            "truth_source_count": raw_truth_count,
            "raw_estimate_component_count": raw_estimate_count,
            "raw_component_cardinality_scored": False,
            "matched_truth_source_count": int(position_array.size),
            "position_error_mean_m": (
                float(np.mean(position_array)) if position_array.size else None
            ),
            "position_error_max_m": (
                float(np.max(position_array)) if position_array.size else None
            ),
            "relative_strength_error_mean": (
                float(np.mean(strength_array)) if strength_array.size else None
            ),
            "relative_strength_error_max": (
                float(np.max(strength_array)) if strength_array.size else None
            ),
        },
        "isotopes": isotope_results,
    }


__all__ = [
    "ClusterAccuracyCriteria",
    "DEFAULT_CLUSTER_ACCURACY_CRITERIA",
    "compute_cluster_accuracy_evaluation",
]
