"""Tests for standardized post-run cluster accuracy scoring."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.cluster_accuracy import (
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
    assert result["hard_cap_sampler_quality_status"] == "pass"
    isotope = result["isotopes"]["Cs-137"]
    assert isotope["raw_estimate_component_count"] == 3
    assert isotope["raw_component_cardinality_scored"] is False
    assert isotope["covered_truth_cluster_count"] == 2
    first = isotope["truth_sources"][0]
    assert first["assigned_raw_component_count"] == 2
    assert first["combined_estimated_strength_cps_1m"] == 100.0
    assert first["representative_estimated_position_xyz_m"] == [0.1, 0.0, 0.0]
    assert first["position_error_xyz_m"] == [0.1, 0.0, 0.0]
    assert first["combined_absolute_strength_error_cps_1m"] == 0.0
    assert isotope["metrics"]["matched_truth_source_count"] == 2


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
