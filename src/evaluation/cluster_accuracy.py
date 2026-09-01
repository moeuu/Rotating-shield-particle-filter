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
    """Declare fixed prospective targets for split-aware PF scoring."""

    position_target_m: float = 0.5
    split_assignment_radius_multiplier: float = 4.0
    same_isotope_separation_fraction: float = 0.5
    maximum_relative_strength_error: float = 0.25
    response_indistinguishable_cosine: float = 0.995

    def __post_init__(self) -> None:
        """Reject invalid or non-finite evaluation tolerances."""
        positive = {
            "position_target_m": self.position_target_m,
            "split_assignment_radius_multiplier": (
                self.split_assignment_radius_multiplier
            ),
            "maximum_relative_strength_error": (
                self.maximum_relative_strength_error
            ),
        }
        for name, value in positive.items():
            resolved = float(value)
            if not np.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, resolved)
        if self.split_assignment_radius_multiplier <= 1.0:
            raise ValueError(
                "split_assignment_radius_multiplier must exceed one."
            )
        probabilities = {
            "same_isotope_separation_fraction": (
                self.same_isotope_separation_fraction
            ),
            "response_indistinguishable_cosine": (
                self.response_indistinguishable_cosine
            ),
        }
        for name, value in probabilities.items():
            resolved = float(value)
            if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, resolved)
        if self.same_isotope_separation_fraction <= 0.0:
            raise ValueError(
                "same_isotope_separation_fraction must be positive."
            )
        if self.same_isotope_separation_fraction > 0.5:
            raise ValueError(
                "same_isotope_separation_fraction must not exceed 0.5."
            )

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return the canonical outcome-independent policy payload."""
        return {
            "policy_name": (
                "split_aware_cluster_accuracy_without_raw_cardinality_scoring"
            ),
            "position_target_m": self.position_target_m,
            "split_assignment_radius_multiplier": (
                self.split_assignment_radius_multiplier
            ),
            "maximum_split_assignment_radius_m": (
                self.maximum_split_assignment_radius_m
            ),
            "same_isotope_separation_fraction": (
                self.same_isotope_separation_fraction
            ),
            "maximum_relative_strength_error": (
                self.maximum_relative_strength_error
            ),
            "response_indistinguishable_cosine": (
                self.response_indistinguishable_cosine
            ),
            "maximum_hard_cap_posterior_mass": HARD_CAP_POSTERIOR_MASS_LIMIT,
            "raw_component_cardinality_is_accuracy_target": False,
            "merged_source_count_semantics": (
                "one_per_truth_cluster_plus_response_distinct_remote"
            ),
            "response_indistinguishable_remote_counts_as_source": False,
            "split_assignment_uses_response_signatures": False,
            "merged_position_summary": "strength_weighted_centroid",
            "position_target_metric": (
                "strength_weighted_rms_distance_to_truth"
            ),
        }

    @property
    def maximum_split_assignment_radius_m(self) -> float:
        """Return the largest distance eligible for split assignment."""
        return self.position_target_m * self.split_assignment_radius_multiplier

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
    """Return a truth-independent display medoid for one cluster."""
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


def _truth_assignment_radii(
    truth_positions: NDArray[np.float64],
    criteria: ClusterAccuracyCriteria,
) -> NDArray[np.float64]:
    """Return nonoverlapping split-assignment radii for true sources."""
    positions = np.asarray(truth_positions, dtype=np.float64).reshape(-1, 3)
    radii = np.full(
        positions.shape[0],
        criteria.maximum_split_assignment_radius_m,
        dtype=np.float64,
    )
    if positions.shape[0] <= 1:
        return radii
    separations = _pairwise_euclidean(positions, positions)
    np.fill_diagonal(separations, np.inf)
    nearest_separation = np.min(separations, axis=1)
    return np.minimum(
        radii,
        criteria.same_isotope_separation_fraction * nearest_separation,
    )


def _unique_nearest_truth_mask(
    distances: NDArray[np.float64],
    nearest_distance: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return estimates with one numerically unique nearest true source."""
    matrix = np.asarray(distances, dtype=np.float64)
    nearest = np.asarray(nearest_distance, dtype=np.float64).reshape(-1)
    if matrix.shape[1] != nearest.size:
        raise ValueError("Nearest-distance arrays are not aligned.")
    if matrix.shape[0] <= 1:
        return np.ones(nearest.size, dtype=np.bool_)
    ties = np.isclose(
        matrix,
        nearest[None, :],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    return np.sum(ties, axis=0) == 1


def _weighted_cluster_geometry(
    estimates: Sequence[Source],
    member_indices: NDArray[np.int64],
    truth_position: NDArray[np.float64],
    position_target_m: float,
) -> dict[str, object]:
    """Return centroid, spread, and truth-RMS geometry for one split cluster."""
    members = np.asarray(member_indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(
        [estimates[int(index)].pos for index in members],
        dtype=np.float64,
    )
    strengths = np.asarray(
        [estimates[int(index)].strength for index in members],
        dtype=np.float64,
    )
    total_strength = float(np.sum(strengths, dtype=np.float64))
    if total_strength <= 0.0:
        raise ValueError("Split-cluster strength must be positive.")
    normalized_weights = strengths / total_strength
    centroid = np.sum(
        positions * normalized_weights[:, None],
        axis=0,
    )
    truth = np.asarray(truth_position, dtype=np.float64).reshape(3)
    centroid_error_vector = centroid - truth
    centroid_error = float(np.linalg.norm(centroid_error_vector))
    truth_distances = np.linalg.norm(positions - truth[None, :], axis=1)
    rms_truth_error = float(
        np.sqrt(np.sum(normalized_weights * np.square(truth_distances)))
    )
    centroid_distances = np.linalg.norm(
        positions - centroid[None, :],
        axis=1,
    )
    rms_dispersion = float(
        np.sqrt(np.sum(normalized_weights * np.square(centroid_distances)))
    )
    within_target_fraction = float(
        np.sum(
            normalized_weights[truth_distances <= float(position_target_m)],
            dtype=np.float64,
        )
    )
    return {
        "centroid": centroid,
        "centroid_error_vector": centroid_error_vector,
        "centroid_error": centroid_error,
        "rms_truth_error": rms_truth_error,
        "rms_dispersion": rms_dispersion,
        "maximum_component_truth_error": float(np.max(truth_distances)),
        "within_target_strength_fraction": within_target_fraction,
        "total_strength": total_strength,
    }


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
    """Score every true source against a bounded split-aware mode aggregate.

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
    all_centroid_position_errors: list[float] = []
    all_rms_position_errors: list[float] = []
    all_spatial_dispersions: list[float] = []
    all_relative_strength_errors: list[float] = []
    global_accuracy_failures: list[str] = []
    global_detection_failures: list[str] = []
    global_hard_cap_failures: list[str] = []
    raw_truth_count = 0
    raw_estimate_count = 0
    total_truth_associated_merged_source_count = 0
    total_merged_estimated_source_count = 0
    total_split_truth_cluster_count = 0
    total_raw_assigned_component_count = 0
    total_raw_component_count_reduction_by_merging = 0
    total_unassigned_remote_component_count = 0
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
        assignment_radii = _truth_assignment_radii(truth_positions, criteria)
        nearest_truth = np.full(len(estimates), -1, dtype=np.int64)
        nearest_distance = np.full(len(estimates), np.inf, dtype=np.float64)
        assignment_radius_by_estimate = np.full(
            len(estimates),
            np.nan,
            dtype=np.float64,
        )
        unique_nearest_truth = np.zeros(len(estimates), dtype=np.bool_)
        if truth and estimates:
            nearest_truth = np.argmin(distances, axis=0).astype(np.int64)
            nearest_distance = distances[
                nearest_truth,
                np.arange(len(estimates), dtype=np.int64),
            ]
            assignment_radius_by_estimate = assignment_radii[nearest_truth]
            unique_nearest_truth = _unique_nearest_truth_mask(
                distances,
                nearest_distance,
            )
        assigned = unique_nearest_truth & (
            nearest_distance <= assignment_radius_by_estimate
        )
        truth_rows: list[dict[str, object]] = []
        cluster_signatures: list[NDArray[np.float64]] = []
        associated_truth_indices: list[int] = []
        isotope_accuracy_failures: list[str] = []
        isotope_detection_failures: list[str] = []
        isotope_centroid_position_errors: list[float] = []
        isotope_rms_position_errors: list[float] = []
        isotope_spatial_dispersions: list[float] = []
        isotope_relative_strength_errors: list[float] = []
        split_truth_cluster_count = 0
        raw_assigned_component_count = 0
        for truth_index, truth_source in enumerate(truth):
            members = np.flatnonzero(assigned & (nearest_truth == truth_index)).astype(
                np.int64,
                copy=False,
            )
            core_members = members[
                nearest_distance[members] <= criteria.position_target_m
            ]
            extended_members = members[
                nearest_distance[members] > criteria.position_target_m
            ]
            associated = bool(members.size)
            is_split_cluster = bool(members.size > 1)
            split_truth_cluster_count += int(is_split_cluster)
            raw_assigned_component_count += int(members.size)
            display_medoid_index: int | None = None
            display_medoid_position: list[float] | None = None
            display_medoid_error: float | None = None
            merged_position: list[float] | None = None
            merged_position_error_vector: list[float] | None = None
            centroid_position_error: float | None = None
            rms_position_error: float | None = None
            spatial_dispersion: float | None = None
            maximum_component_position_error: float | None = None
            within_target_strength_fraction: float | None = None
            estimated_strength = 0.0
            relative_strength_error: float | None = None
            absolute_strength_error: float | None = None
            if associated:
                display_medoid_index = _cluster_representative_index(
                    estimates,
                    members,
                )
                display_medoid_position = estimates[
                    display_medoid_index
                ].pos.tolist()
                display_medoid_error = float(
                    np.linalg.norm(
                        estimates[display_medoid_index].pos - truth_source.pos
                    )
                )
                geometry = _weighted_cluster_geometry(
                    estimates,
                    members,
                    truth_source.pos,
                    criteria.position_target_m,
                )
                centroid = np.asarray(geometry["centroid"], dtype=np.float64)
                centroid_error_vector = np.asarray(
                    geometry["centroid_error_vector"],
                    dtype=np.float64,
                )
                merged_position = centroid.tolist()
                merged_position_error_vector = centroid_error_vector.tolist()
                centroid_position_error = float(geometry["centroid_error"])
                rms_position_error = float(geometry["rms_truth_error"])
                spatial_dispersion = float(geometry["rms_dispersion"])
                maximum_component_position_error = float(
                    geometry["maximum_component_truth_error"]
                )
                within_target_strength_fraction = float(
                    geometry["within_target_strength_fraction"]
                )
                estimated_strength = float(geometry["total_strength"])
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
                associated_truth_indices.append(truth_index)
                all_centroid_position_errors.append(centroid_position_error)
                all_rms_position_errors.append(rms_position_error)
                all_spatial_dispersions.append(spatial_dispersion)
                all_relative_strength_errors.append(relative_strength_error)
                isotope_centroid_position_errors.append(
                    centroid_position_error
                )
                isotope_rms_position_errors.append(rms_position_error)
                isotope_spatial_dispersions.append(spatial_dispersion)
                isotope_relative_strength_errors.append(
                    relative_strength_error
                )
            centroid_target_met = bool(
                centroid_position_error is not None
                and centroid_position_error <= criteria.position_target_m
            )
            position_target_met = bool(
                rms_position_error is not None
                and rms_position_error <= criteria.position_target_m
            )
            strength_target_met = bool(
                relative_strength_error is not None
                and relative_strength_error
                <= criteria.maximum_relative_strength_error
            )
            if not associated:
                failure = f"missing_truth_cluster:{truth_index}"
                isotope_detection_failures.append(failure)
                isotope_accuracy_failures.append(failure)
            else:
                if not position_target_met:
                    isotope_accuracy_failures.append(
                        f"position_target_not_met:{truth_index}"
                    )
                if not strength_target_met:
                    isotope_accuracy_failures.append(
                        f"strength_target_not_met:{truth_index}"
                    )
            truth_rows.append(
                {
                    "truth_source_index": truth_index,
                    "truth_position_xyz_m": truth_source.pos.tolist(),
                    "truth_strength_cps_1m": truth_source.strength,
                    "effective_split_assignment_radius_m": float(
                        assignment_radii[truth_index]
                    ),
                    "assigned_estimate_indices": [int(value) for value in members],
                    "core_estimate_indices": [
                        int(value) for value in core_members
                    ],
                    "extended_split_estimate_indices": [
                        int(value) for value in extended_members
                    ],
                    "assigned_component_truth_distances_m": [
                        float(nearest_distance[int(value)]) for value in members
                    ],
                    "assigned_raw_component_count": int(members.size),
                    "is_split_cluster": is_split_cluster,
                    "merged_source_count_contribution": int(associated),
                    "display_medoid_estimate_index": display_medoid_index,
                    "display_medoid_position_xyz_m": display_medoid_position,
                    "display_medoid_position_error_m": display_medoid_error,
                    "merged_position_xyz_m": merged_position,
                    "merged_position_error_xyz_m": (
                        merged_position_error_vector
                    ),
                    "merged_centroid_position_error_m": (
                        centroid_position_error
                    ),
                    "strength_weighted_rms_position_error_m": (
                        rms_position_error
                    ),
                    "strength_weighted_spatial_dispersion_m": (
                        spatial_dispersion
                    ),
                    "maximum_assigned_component_position_error_m": (
                        maximum_component_position_error
                    ),
                    "within_position_target_strength_fraction": (
                        within_target_strength_fraction
                    ),
                    "combined_estimated_strength_cps_1m": estimated_strength,
                    "combined_absolute_strength_error_cps_1m": (
                        absolute_strength_error
                    ),
                    "combined_relative_strength_error": relative_strength_error,
                    "associated": associated,
                    "merged_centroid_position_target_met": (
                        centroid_target_met
                    ),
                    "position_target_met": position_target_met,
                    "strength_target_met": strength_target_met,
                    "source_accuracy_target_met": bool(
                        associated
                        and position_target_met
                        and strength_target_met
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
            if nearest_truth[estimate_index] < 0:
                exclusion_reason = "no_same_isotope_truth"
            elif not unique_nearest_truth[estimate_index]:
                exclusion_reason = "equidistant_truth_ambiguity"
            else:
                exclusion_reason = "outside_split_assignment_radius"
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
                    "effective_split_assignment_radius_m": (
                        float(assignment_radius_by_estimate[estimate_index])
                        if np.isfinite(
                            assignment_radius_by_estimate[estimate_index]
                        )
                        else None
                    ),
                    "assignment_exclusion_reason": exclusion_reason,
                    "maximum_cosine_to_associated_cluster_response": (
                        maximum_cosine
                    ),
                    "response_distinct": response_distinct,
                }
            )
        if response_distinct_remote_count:
            isotope_accuracy_failures.append(
                "response_distinct_remote_components"
            )
        truth_associated_merged_source_count = len(associated_truth_indices)
        merged_estimated_source_count = (
            truth_associated_merged_source_count
            + response_distinct_remote_count
        )
        raw_component_count_reduction_by_merging = (
            raw_assigned_component_count
            - truth_associated_merged_source_count
        )
        hard_cap_mass = _hard_cap_mass(
            cardinality_distribution,
            hard_max_sources,
        )
        hard_cap_passed = hard_cap_mass_is_acceptable(hard_cap_mass)
        isotope_hard_cap_failures = (
            [] if hard_cap_passed else ["hard_cardinality_cap_saturation"]
        )
        isotope_results[isotope] = {
            "truth_source_detection_status": (
                "pass" if not isotope_detection_failures else "failed"
            ),
            "truth_source_detection_failure_reasons": (
                isotope_detection_failures
            ),
            "accuracy_status": (
                "pass" if not isotope_accuracy_failures else "failed"
            ),
            "accuracy_failure_reasons": isotope_accuracy_failures,
            "hard_cap_sampler_quality_status": (
                "pass" if hard_cap_passed else "failed"
            ),
            "hard_cap_sampler_quality_failure_reasons": (
                isotope_hard_cap_failures
            ),
            "raw_truth_count": len(truth),
            "raw_estimate_component_count": len(estimates),
            "raw_component_cardinality_scored": False,
            "associated_truth_source_count": len(associated_truth_indices),
            "truth_associated_merged_source_count": (
                truth_associated_merged_source_count
            ),
            "merged_estimated_source_count": merged_estimated_source_count,
            "split_truth_cluster_count": split_truth_cluster_count,
            "raw_assigned_component_count": raw_assigned_component_count,
            "raw_component_count_reduction_by_merging": (
                raw_component_count_reduction_by_merging
            ),
            "unassigned_remote_component_count": int(remote_indices.size),
            "core_covered_truth_source_count": sum(
                bool(row["core_estimate_indices"]) for row in truth_rows
            ),
            "position_target_met_truth_source_count": sum(
                bool(row["position_target_met"]) for row in truth_rows
            ),
            "strength_target_met_truth_source_count": sum(
                bool(row["strength_target_met"]) for row in truth_rows
            ),
            "response_distinct_remote_component_count": (
                response_distinct_remote_count
            ),
            "hard_cap_posterior_mass": hard_cap_mass,
            "hard_cap_saturation_passed": hard_cap_passed,
            "metrics": {
                "associated_truth_source_count": len(
                    isotope_rms_position_errors
                ),
                "merged_centroid_position_error_mean_m": (
                    float(np.mean(isotope_centroid_position_errors))
                    if isotope_centroid_position_errors
                    else None
                ),
                "merged_centroid_position_error_max_m": (
                    float(np.max(isotope_centroid_position_errors))
                    if isotope_centroid_position_errors
                    else None
                ),
                "strength_weighted_rms_position_error_mean_m": (
                    float(np.mean(isotope_rms_position_errors))
                    if isotope_rms_position_errors
                    else None
                ),
                "strength_weighted_rms_position_error_max_m": (
                    float(np.max(isotope_rms_position_errors))
                    if isotope_rms_position_errors
                    else None
                ),
                "strength_weighted_spatial_dispersion_mean_m": (
                    float(np.mean(isotope_spatial_dispersions))
                    if isotope_spatial_dispersions
                    else None
                ),
                "strength_weighted_spatial_dispersion_max_m": (
                    float(np.max(isotope_spatial_dispersions))
                    if isotope_spatial_dispersions
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
        global_accuracy_failures.extend(
            f"{isotope}:{reason}" for reason in isotope_accuracy_failures
        )
        global_detection_failures.extend(
            f"{isotope}:{reason}" for reason in isotope_detection_failures
        )
        global_hard_cap_failures.extend(
            f"{isotope}:{reason}" for reason in isotope_hard_cap_failures
        )
        raw_truth_count += len(truth)
        raw_estimate_count += len(estimates)
        total_truth_associated_merged_source_count += (
            truth_associated_merged_source_count
        )
        total_merged_estimated_source_count += merged_estimated_source_count
        total_split_truth_cluster_count += split_truth_cluster_count
        total_raw_assigned_component_count += raw_assigned_component_count
        total_raw_component_count_reduction_by_merging += (
            raw_component_count_reduction_by_merging
        )
        total_unassigned_remote_component_count += int(remote_indices.size)
    centroid_position_array = np.asarray(
        all_centroid_position_errors,
        dtype=np.float64,
    )
    rms_position_array = np.asarray(all_rms_position_errors, dtype=np.float64)
    spatial_dispersion_array = np.asarray(
        all_spatial_dispersions,
        dtype=np.float64,
    )
    strength_array = np.asarray(all_relative_strength_errors, dtype=np.float64)
    return {
        "schema_version": 3,
        "artifact_family": "completed_pf_cluster_accuracy_evaluation",
        "criteria": criteria.to_dict(),
        "criteria_sha256": criteria.sha256,
        "truth_used_only_after_completion": True,
        "changes_pf_state_or_cardinality": False,
        "truth_source_detection_status": (
            "pass" if not global_detection_failures else "failed"
        ),
        "truth_source_detection_failure_reasons": global_detection_failures,
        "accuracy_status": (
            "pass" if not global_accuracy_failures else "failed"
        ),
        "accuracy_failure_reasons": global_accuracy_failures,
        "hard_cap_sampler_quality_status": (
            "pass" if not global_hard_cap_failures else "failed"
        ),
        "hard_cap_sampler_quality_failure_reasons": global_hard_cap_failures,
        "global": {
            "truth_source_count": raw_truth_count,
            "raw_estimate_component_count": raw_estimate_count,
            "raw_component_cardinality_scored": False,
            "associated_truth_source_count": int(rms_position_array.size),
            "truth_associated_merged_source_count": (
                total_truth_associated_merged_source_count
            ),
            "merged_estimated_source_count": (
                total_merged_estimated_source_count
            ),
            "split_truth_cluster_count": total_split_truth_cluster_count,
            "raw_assigned_component_count": (
                total_raw_assigned_component_count
            ),
            "raw_component_count_reduction_by_merging": (
                total_raw_component_count_reduction_by_merging
            ),
            "unassigned_remote_component_count": (
                total_unassigned_remote_component_count
            ),
            "merged_centroid_position_error_mean_m": (
                float(np.mean(centroid_position_array))
                if centroid_position_array.size
                else None
            ),
            "merged_centroid_position_error_max_m": (
                float(np.max(centroid_position_array))
                if centroid_position_array.size
                else None
            ),
            "strength_weighted_rms_position_error_mean_m": (
                float(np.mean(rms_position_array))
                if rms_position_array.size
                else None
            ),
            "strength_weighted_rms_position_error_max_m": (
                float(np.max(rms_position_array))
                if rms_position_array.size
                else None
            ),
            "strength_weighted_spatial_dispersion_mean_m": (
                float(np.mean(spatial_dispersion_array))
                if spatial_dispersion_array.size
                else None
            ),
            "strength_weighted_spatial_dispersion_max_m": (
                float(np.max(spatial_dispersion_array))
                if spatial_dispersion_array.size
                else None
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
