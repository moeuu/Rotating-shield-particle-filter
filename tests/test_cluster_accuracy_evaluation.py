"""Tests for standardized post-run cluster accuracy scoring."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.cluster_accuracy import (
    ClusterAccuracyCriteria,
    DEFAULT_CLUSTER_ACCURACY_CRITERIA,
    compute_cluster_accuracy_evaluation,
)
from pf.cardinality_policy import HARD_CAP_POSTERIOR_MASS_LIMIT


def _posterior(
    modes: list[tuple[list[float], float]],
    *,
    hard_cap_mass: float = 0.01,
) -> dict[str, Any]:
    """Return one compact posterior payload for cluster evaluation."""
    return {
        "isotopes": {
            "Cs-137": {
                "cardinality_distribution": {
                    "3": 1.0 - hard_cap_mass,
                    "8": hard_cap_mass,
                },
                "modes": [
                    {
                        "position_medoid_xyz": position,
                        "strength_representative_cps_1m": strength,
                    }
                    for position, strength in modes
                ],
            }
        }
    }


def _input(
    modes: list[tuple[list[float], float]],
    signatures: list[list[float]],
) -> dict[str, Any]:
    """Return aligned truth-free response-signature input."""
    return {
        "schema_version": 1,
        "artifact_family": "pf_post_run_cluster_evaluation_input",
        "source_run_id": "test-run",
        "measurement_log_sha256": "a" * 64,
        "hard_max_sources_per_isotope": 8,
        "response_signature_semantics": (
            "normalized_same_isotope_expected_count_by_completed_measurement"
        ),
        "truth_read": False,
        "isotopes": {
            "Cs-137": {
                "mode_label_indices": list(range(len(modes))),
                "mode_positions_xyz_m": [position for position, _ in modes],
                "mode_strengths_cps_1m": [strength for _, strength in modes],
                "normalized_response_signatures_measurement_by_mode": signatures,
            }
        },
    }


def test_local_splits_are_aggregated_without_scoring_raw_cardinality() -> None:
    """Two local components may represent one accurate physical source."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
            {"position": [5.0, 0.0, 0.0], "strength": 200.0},
        ]
    }
    modes = [
        ([0.1, 0.0, 0.0], 60.0),
        ([-0.1, 0.0, 0.0], 40.0),
        ([5.2, 0.0, 0.0], 210.0),
    ]
    signatures = [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(modes, signatures),
    )

    assert result["accuracy_status"] == "pass"
    assert result["schema_version"] == 3
    assert result["hard_cap_sampler_quality_status"] == "pass"
    isotope = result["isotopes"]["Cs-137"]
    assert isotope["raw_estimate_component_count"] == 3
    assert isotope["raw_component_cardinality_scored"] is False
    assert isotope["associated_truth_source_count"] == 2
    first = isotope["truth_sources"][0]
    assert first["assigned_raw_component_count"] == 2
    assert first["core_estimate_indices"] == [0, 1]
    assert first["extended_split_estimate_indices"] == []
    assert first["combined_estimated_strength_cps_1m"] == 100.0
    assert first["merged_position_xyz_m"] == pytest.approx([0.02, 0.0, 0.0])
    assert first["merged_position_error_xyz_m"] == pytest.approx(
        [0.02, 0.0, 0.0]
    )
    assert first["merged_centroid_position_error_m"] == pytest.approx(0.02)
    assert first["strength_weighted_rms_position_error_m"] == pytest.approx(
        0.1
    )
    assert first["strength_weighted_spatial_dispersion_m"] == pytest.approx(
        0.0979795897
    )
    assert first["combined_absolute_strength_error_cps_1m"] == 0.0
    assert isotope["metrics"]["associated_truth_source_count"] == 2


def test_extended_splits_contribute_to_strength_and_position_metrics() -> None:
    """Moderately separated components must remain in truth scoring."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [
        ([0.1, 0.0, 0.0], 60.0),
        ([1.2, 0.0, 0.0], 40.0),
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(
            modes,
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
    )

    isotope = result["isotopes"]["Cs-137"]
    source = isotope["truth_sources"][0]
    assert result["truth_source_detection_status"] == "pass"
    assert source["assigned_estimate_indices"] == [0, 1]
    assert source["core_estimate_indices"] == [0]
    assert source["extended_split_estimate_indices"] == [1]
    assert source["effective_split_assignment_radius_m"] == pytest.approx(1.5)
    assert source["combined_estimated_strength_cps_1m"] == pytest.approx(100.0)
    assert source["merged_position_xyz_m"] == pytest.approx([0.54, 0.0, 0.0])
    assert source["strength_weighted_rms_position_error_m"] == pytest.approx(
        0.7628892449
    )
    assert source["position_target_met"] is False
    assert "position_target_not_met:0" in isotope["accuracy_failure_reasons"]


def test_rms_position_target_cannot_be_hidden_by_centroid_cancellation() -> None:
    """A centered but widely split cluster must retain its localization spread."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [
        ([-1.0, 0.0, 0.0], 50.0),
        ([1.0, 0.0, 0.0], 50.0),
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(modes, [[1.0, 1.0]]),
    )

    source = result["isotopes"]["Cs-137"]["truth_sources"][0]
    assert source["merged_centroid_position_error_m"] == pytest.approx(0.0)
    assert source["merged_centroid_position_target_met"] is True
    assert source["strength_weighted_spatial_dispersion_m"] == pytest.approx(
        1.0
    )
    assert source["strength_weighted_rms_position_error_m"] == pytest.approx(
        1.0
    )
    assert source["position_target_met"] is False


def test_split_assignment_is_capped_by_same_isotope_truth_separation() -> None:
    """A broad split gate must not overlap neighboring same-isotope truths."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
            {"position": [2.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [
        ([0.0, 0.0, 0.0], 100.0),
        ([2.0, 0.0, 0.0], 100.0),
        ([-1.2, 0.0, 0.0], 10.0),
    ]
    signatures = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(modes, signatures),
    )

    isotope = result["isotopes"]["Cs-137"]
    assert isotope["accuracy_status"] == "pass"
    assert isotope["truth_sources"][0][
        "effective_split_assignment_radius_m"
    ] == pytest.approx(1.0)
    assert isotope["remote_estimates"][0]["estimate_index"] == 2
    assert isotope["remote_estimates"][0][
        "assignment_exclusion_reason"
    ] == "outside_split_assignment_radius"


def test_equidistant_component_remains_an_audited_remote_ambiguity() -> None:
    """Truth ordering must not decide an exactly ambiguous split assignment."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
            {"position": [2.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [
        ([0.0, 0.0, 0.0], 100.0),
        ([2.0, 0.0, 0.0], 100.0),
        ([1.0, 0.0, 0.0], 10.0),
    ]
    signatures = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(modes, signatures),
    )

    remote = result["isotopes"]["Cs-137"]["remote_estimates"][0]
    assert remote["estimate_index"] == 2
    assert remote["assignment_exclusion_reason"] == (
        "equidistant_truth_ambiguity"
    )


def test_response_distinct_remote_component_fails_cluster_accuracy() -> None:
    """A remote response basis cannot be hidden by an accurate local cluster."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [
        ([0.1, 0.0, 0.0], 100.0),
        ([5.0, 5.0, 0.0], 30.0),
    ]
    signatures = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes),
        _input(modes, signatures),
    )

    assert result["accuracy_status"] == "failed"
    assert result["hard_cap_sampler_quality_status"] == "pass"
    isotope = result["isotopes"]["Cs-137"]
    assert isotope["response_distinct_remote_component_count"] == 1
    assert "response_distinct_remote_components" in isotope[
        "accuracy_failure_reasons"
    ]


def test_hard_cap_mass_marks_sampler_failure_not_accuracy_failure() -> None:
    """Hard-cap saturation remains a sampler failure outside raw-K scoring."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [([0.0, 0.0, 0.0], 100.0)]

    result = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes, hard_cap_mass=0.2),
        _input(modes, [[1.0]]),
    )

    assert result["accuracy_status"] == "pass"
    assert result["hard_cap_sampler_quality_status"] == "failed"
    isotope = result["isotopes"]["Cs-137"]
    assert isotope["hard_cap_saturation_passed"] is False
    assert "hard_cardinality_cap_saturation" in isotope[
        "hard_cap_sampler_quality_failure_reasons"
    ]


def test_post_run_uses_the_same_inclusive_hard_cap_limit_as_live_health() -> None:
    """Post-run scoring must consume the shared non-configurable 5% limit."""
    truth = {
        "Cs-137": [
            {"position": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    modes = [([0.0, 0.0, 0.0], 100.0)]

    at_limit = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(modes, hard_cap_mass=HARD_CAP_POSTERIOR_MASS_LIMIT),
        _input(modes, [[1.0]]),
    )
    above_limit = compute_cluster_accuracy_evaluation(
        truth,
        _posterior(
            modes,
            hard_cap_mass=HARD_CAP_POSTERIOR_MASS_LIMIT + 1.0e-6,
        ),
        _input(modes, [[1.0]]),
    )

    assert (
        DEFAULT_CLUSTER_ACCURACY_CRITERIA.maximum_hard_cap_posterior_mass
        == HARD_CAP_POSTERIOR_MASS_LIMIT
    )
    assert at_limit["isotopes"]["Cs-137"]["hard_cap_saturation_passed"] is True
    assert (
        above_limit["isotopes"]["Cs-137"]["hard_cap_saturation_passed"]
        is False
    )


def test_evaluation_input_schema_rejects_unknown_fields() -> None:
    """Standard scoring must not ignore misspelled or retired input fields."""
    modes = [([0.0, 0.0, 0.0], 100.0)]
    evaluation_input = _input(modes, [[1.0]])
    evaluation_input["ignored_threshold"] = 1.0

    with pytest.raises(ValueError, match="schema version 1"):
        compute_cluster_accuracy_evaluation(
            {"Cs-137": [{"position": [0.0, 0.0, 0.0], "strength": 100.0}]},
            _posterior(modes),
            evaluation_input,
        )


def test_evaluation_rejects_noncanonical_cardinality_and_response_values() -> None:
    """Aliases and nonphysical response columns must fail instead of coercing."""
    truth = {
        "Cs-137": [{"position": [0.0, 0.0, 0.0], "strength": 100.0}]
    }
    modes = [([0.0, 0.0, 0.0], 100.0)]
    posterior = _posterior(modes)
    posterior["isotopes"]["Cs-137"]["cardinality_distribution"] = {
        "1": 0.9,
        "08": 0.1,
    }
    with pytest.raises(ValueError, match="canonical"):
        compute_cluster_accuracy_evaluation(
            truth,
            posterior,
            _input(modes, [[1.0]]),
        )

    with pytest.raises(ValueError, match="response signatures"):
        compute_cluster_accuracy_evaluation(
            truth,
            _posterior(modes),
            _input(modes, [[-1.0]]),
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"split_assignment_radius_multiplier": 1.0},
        {"same_isotope_separation_fraction": 0.0},
        {"same_isotope_separation_fraction": 0.500001},
    ),
)
def test_split_assignment_criteria_reject_overlapping_or_local_gates(
    kwargs: dict[str, float],
) -> None:
    """The standard split gate must be broader than a target and nonoverlapping."""
    with pytest.raises(ValueError):
        ClusterAccuracyCriteria(**kwargs)
