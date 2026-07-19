"""Tests for weighted Geant4 transport acceptance analysis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.analyze_weighted_transport_acceptance import analyze_acceptance


ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


def _covariance(diagonal: float = 10_000.0) -> dict[str, dict[str, float]]:
    """Return a diagonal isotope covariance payload."""
    return {
        row: {column: diagonal if row == column else 0.0 for column in ISOTOPES}
        for row in ISOTOPES
    }


def _case(name: str, pair_id: int) -> dict[str, Any]:
    """Return a compact deterministic physical validation case."""
    return {
        "name": name,
        "detector_pose_xyz": [1.0, 2.0, 0.5],
        "dwell_time_s": 30.0,
        "fe_index": pair_id // 8,
        "pb_index": pair_id % 8,
        "sources": [
            {
                "isotope": "Cs-137",
                "position_xyz": [3.0, 4.0, 0.1],
                "intensity_cps_1m": 1_000_000.0,
            }
        ],
        "obstacle_cells": [[2, 3]],
        "obstacle_instances": [],
    }


def _accelerated_record(name: str, pair_id: int = 0) -> dict[str, Any]:
    """Return one internally consistent weighted-transport result record."""
    counts = {"Cs-137": 3_000.0, "Co-60": 2_000.0, "Eu-154": 5_000.0}
    covariance = _covariance()
    return {
        "case": _case(name, pair_id),
        "runtime_s": 2.0,
        "num_primaries": 1_500_000.0,
        "raw_total_spectrum_counts": 10_000.0,
        "metadata": {
            "backend": "geant4",
            "engine_mode": "external",
            "source_rate_model": "detector_cps_1m",
            "expected_primary_semantics": "detector_equivalent_histories",
            "transport_history_mode": "weighted_thinning",
            "spectrum_variance_semantics": ("compound_poisson_sumw2_includes_counting"),
            "spectrum_variance_dead_time_propagation": "fixed_observed_scale",
            "intensity_cps_1m_definition": "net_detector_count_rate_at_1m",
            "accelerated_weighted_transport_enable": True,
            "history_thinning_enabled": True,
            "transport_tally_weighted": True,
            "weighted_transport": True,
            "source_bias_weighted_transport": False,
            "poisson_background": True,
            "theory_tvl_attenuation": False,
            "primary_sampling_fraction": 0.15,
            "requested_primary_sampling_fraction": 1.0,
            "primary_history_weight": 1.0 / 0.15,
            "target_sampled_primaries": 1_500_000,
            "primary_sampling_budget_enabled": True,
            "primary_sampling_fraction_resolution": "target_budget_limited",
            "expected_detector_equivalent_primaries": 10_000_000.0,
            "expected_unthinned_primaries": 10_000_000.0,
            "expected_sampled_primaries": 1_500_000.0,
            "dead_time_tau_s": 0.0,
            "dead_time_observed_scale": 1.0,
            "dwell_time_s": 30.0,
            "pre_dead_time_total_spectrum_counts": 10_000.0,
            "pre_dead_time_weighted_spectrum_sumw2": 10_000.0,
            "weighted_spectrum_sumw2": 10_000.0,
            "weighted_spectrum_effective_entries": 10_000.0,
            "total_spectrum_counts": 10_000.0,
            "requested_threads": 32,
            "multithreaded_run_manager": True,
            "transport_detected_counts_Cs-137": 500_000.0,
            "transport_detected_counts_Co-60": 500_000.0,
            "transport_detected_counts_Eu-154": 500_000.0,
        },
        "pf_count_likelihood_diagnostics": {
            "runtime_likelihood_guards": {
                "observation_count_variance_semantics": "complete_statistical",
                "direct_spectrum_likelihood_enable": False,
                "shield_contrast_likelihood_enable": False,
                "shield_view_ratio_likelihood_enable": False,
            },
            "likelihood_spec_by_isotope": {
                isotope: {
                    "observation_count_variance_semantics": "complete_statistical"
                }
                for isotope in ISOTOPES
            },
            "targets": {
                "runtime_pf_forward": {
                    "degrees_of_freedom": 3,
                    "diagonal_squared_distance": 0.0,
                    "likelihood_scale_over_target_by_isotope": {
                        isotope: 0.15 for isotope in ISOTOPES
                    },
                    "normalized_residual_by_isotope": {
                        isotope: 0.0 for isotope in ISOTOPES
                    },
                }
            },
        },
        "response_poisson_diagnostics": {"counts": counts},
        "response_poisson_covariance": {
            "isotope_order": list(ISOTOPES),
            "formal": covariance,
        },
        "per_isotope": {
            isotope: {
                "transport_truth_counts": counts[isotope],
                "pf_target_relative_error_vs_transport_truth": 0.0,
                "method_counts": {"response_poisson": counts[isotope]},
            }
            for isotope in ISOTOPES
        },
    }


def _reference_record(name: str, pair_id: int = 0) -> dict[str, Any]:
    """Return a matched full-history reference record."""
    record = _accelerated_record(name, pair_id)
    record["metadata"] = {
        "backend": "geant4",
        "engine_mode": "external",
        "source_rate_model": "detector_cps_1m",
        "transport_history_mode": "full_unit_weight",
        "spectrum_variance_semantics": "compound_poisson_sumw2_includes_counting",
        "spectrum_variance_dead_time_propagation": "fixed_observed_scale",
        "history_thinning_enabled": False,
        "transport_tally_weighted": False,
        "weighted_transport": False,
        "source_bias_weighted_transport": False,
        "multithreaded_run_manager": True,
        "primary_sampling_fraction": 1.0,
        "primary_history_weight": 1.0,
        "requested_threads": 32,
        "weighted_spectrum_sumw2": 10_000.0,
    }
    record["response_poisson_diagnostics"].update(
        {
            "count_covariance_semantics": (
                "compound_poisson_sumw2_dead_time_delta_jacobian_"
                "response_folded_factored"
            ),
            "count_covariance_complete_transport_provenance": True,
        }
    )
    return record


def test_paired_acceptance_passes_consistent_weighted_result() -> None:
    """Paired gates pass for identical unbiased estimates and healthy metadata."""
    accelerated = [_accelerated_record("multi_iso_0000")]
    reference = [_reference_record("multi_iso_0000")]

    report = analyze_acceptance(accelerated, reference)

    assert report["overall_pass"] is True
    assert report["case_matching"]["physical_cases_match"] is True
    assert report["full_reference_provenance"]["all_valid"] is True
    assert report["paired_raw_total"]["pooled_z"] == 0.0
    assert report["paired_response"]["q_per_dof"] == 0.0
    assert report["gates"]["paired_analysis_completeness"]["passed"] is True
    assert report["gates"]["raw_total_nominal_95_coverage"]["applicable"] is False


def test_transport_provenance_accepts_maximum_fraction_limited_budget() -> None:
    """Dim scenes may use the configured maximum below the history budget."""
    accelerated = _accelerated_record("multi_iso_0000")
    metadata = accelerated["metadata"]
    metadata["expected_detector_equivalent_primaries"] = 1_000_000.0
    metadata["expected_unthinned_primaries"] = 1_000_000.0
    metadata["expected_sampled_primaries"] = 1_000_000.0
    metadata["primary_sampling_fraction"] = 1.0
    metadata["primary_history_weight"] = 1.0
    metadata["primary_sampling_fraction_resolution"] = "maximum_fraction_limited"
    metadata["transport_history_mode"] = "full_unit_weight"
    metadata["history_thinning_enabled"] = False
    metadata["transport_tally_weighted"] = False
    metadata["weighted_transport"] = False
    accelerated["num_primaries"] = 1_000_000.0

    report = analyze_acceptance(
        [accelerated],
        [_reference_record("multi_iso_0000")],
    )

    assert report["transport_provenance"]["all_valid"] is True


def test_paired_acceptance_fails_mismatched_case_and_provenance() -> None:
    """Physical-case mismatch and an unauthorized fraction fail closed."""
    accelerated = _accelerated_record("multi_iso_0000")
    accelerated["case"]["detector_pose_xyz"] = [9.0, 2.0, 0.5]
    accelerated["metadata"]["primary_sampling_fraction"] = 0.03
    reference = _reference_record("multi_iso_0000")

    report = analyze_acceptance([accelerated], [reference])

    assert report["overall_pass"] is False
    assert report["gates"]["paired_physical_cases"]["passed"] is False
    assert report["gates"]["transport_provenance"]["passed"] is False


def test_paired_acceptance_rejects_legacy_full_reference_provenance() -> None:
    """A reference without current response-covariance provenance fails closed."""
    reference = _reference_record("multi_iso_0000")
    del reference["response_poisson_diagnostics"][
        "count_covariance_complete_transport_provenance"
    ]

    report = analyze_acceptance(
        [_accelerated_record("multi_iso_0000")],
        [reference],
    )

    assert report["overall_pass"] is False
    assert report["full_reference_provenance"]["all_valid"] is False
    assert report["gates"]["full_reference_provenance"]["passed"] is False


def test_paired_acceptance_rejects_an_unanalyzable_matched_pair() -> None:
    """Every matched raw and response row must be successfully analyzed."""
    accelerated = _accelerated_record("multi_iso_0000")
    del accelerated["raw_total_spectrum_counts"]

    report = analyze_acceptance(
        [accelerated],
        [_reference_record("multi_iso_0000")],
    )

    assert report["overall_pass"] is False
    assert report["paired_raw_total"]["valid_pairs"] == 0
    assert report["paired_raw_total"]["invalid_cases"]
    assert report["gates"]["paired_analysis_completeness"]["passed"] is False


def test_paired_acceptance_requires_a_reference_for_every_accelerated_case() -> None:
    """Paired completeness counts accelerated cases without reference matches."""
    accelerated = [
        _accelerated_record("multi_iso_0000"),
        _accelerated_record("multi_iso_0001"),
    ]

    report = analyze_acceptance(
        accelerated,
        [_reference_record("multi_iso_0000")],
    )

    completeness = report["gates"]["paired_analysis_completeness"]
    assert report["overall_pass"] is False
    assert completeness["value"]["expected_pairs"] == 2
    assert completeness["value"]["matched_pairs"] == 1
    assert completeness["passed"] is False


def test_paired_acceptance_rejects_an_invalid_response_row() -> None:
    """A matched pair with an unreadable response covariance fails completeness."""
    accelerated = _accelerated_record("multi_iso_0000")
    del accelerated["response_poisson_covariance"]["formal"]["Eu-154"]

    report = analyze_acceptance(
        [accelerated],
        [_reference_record("multi_iso_0000")],
    )

    assert report["overall_pass"] is False
    assert report["paired_response"]["valid_rows"] == 0
    assert report["paired_response"]["invalid_cases"]
    assert report["gates"]["paired_analysis_completeness"]["passed"] is False


def test_fresh_acceptance_requires_and_accepts_all_64_pairs() -> None:
    """Fresh acceptance recognizes exactly one complete 8-by-8 pair sweep."""
    records = [
        _accelerated_record(f"multi_iso_0000_pair{pair_id:02d}", pair_id)
        for pair_id in range(64)
    ]

    report = analyze_acceptance(records)

    assert report["overall_pass"] is True
    assert report["completion"]["complete"] is True
    assert report["completion"]["unique_pair_count"] == 64
    assert report["formal_covariance_health"]["all_valid"] is True
    assert report["gates"]["fresh_analysis_completeness"]["passed"] is True


def test_fresh_acceptance_rejects_incomplete_accuracy_rows() -> None:
    """A malformed accuracy case cannot be hidden by the remaining valid rows."""
    records = [
        _accelerated_record(f"multi_iso_0000_pair{pair_id:02d}", pair_id)
        for pair_id in range(64)
    ]
    del records[0]["per_isotope"]["Eu-154"][
        "pf_target_relative_error_vs_transport_truth"
    ]

    report = analyze_acceptance(records)

    completeness = report["gates"]["fresh_analysis_completeness"]
    assert report["overall_pass"] is False
    assert report["fresh_accuracy"]["invalid_cases"]
    assert (
        report["fresh_accuracy"]["pf_forward_relative_error"]["pooled"]["count"]
        == 63 * 3
    )
    assert completeness["passed"] is False


def test_fresh_acceptance_rejects_incomplete_likelihood_dof() -> None:
    """The likelihood gate requires three retained degrees of freedom per case."""
    records = [
        _accelerated_record(f"multi_iso_0000_pair{pair_id:02d}", pair_id)
        for pair_id in range(64)
    ]
    target = records[0]["pf_count_likelihood_diagnostics"]["targets"][
        "runtime_pf_forward"
    ]
    target["degrees_of_freedom"] = 2

    report = analyze_acceptance(records)

    assert report["overall_pass"] is False
    assert (
        report["fresh_accuracy"]["pf_likelihood_scale"]["degrees_of_freedom_total"]
        == 63 * 3
    )
    assert report["fresh_accuracy"]["pf_likelihood_scale"]["invalid_cases"]
    assert report["gates"]["fresh_analysis_completeness"]["passed"] is False


def test_fresh_acceptance_rejects_an_invalid_likelihood_case() -> None:
    """A malformed likelihood row fails even when all accuracy rows are valid."""
    records = [
        _accelerated_record(f"multi_iso_0000_pair{pair_id:02d}", pair_id)
        for pair_id in range(64)
    ]
    target = records[0]["pf_count_likelihood_diagnostics"]["targets"][
        "runtime_pf_forward"
    ]
    del target["normalized_residual_by_isotope"]["Co-60"]

    report = analyze_acceptance(records)

    likelihood = report["fresh_accuracy"]["pf_likelihood_scale"]
    assert report["overall_pass"] is False
    assert likelihood["invalid_cases"]
    assert likelihood["scale_over_target"]["count"] == 63 * 3
    assert report["gates"]["fresh_analysis_completeness"]["passed"] is False


def test_covariance_health_rejects_non_psd_formal_covariance() -> None:
    """A finite but indefinite response covariance fails the health gate."""
    records = [
        _accelerated_record(f"multi_iso_0000_pair{pair_id:02d}", pair_id)
        for pair_id in range(64)
    ]
    broken = deepcopy(records[0])
    formal = broken["response_poisson_covariance"]["formal"]
    formal["Cs-137"]["Co-60"] = 20_000.0
    formal["Co-60"]["Cs-137"] = 20_000.0
    records[0] = broken

    report = analyze_acceptance(records)

    assert report["overall_pass"] is False
    assert report["gates"]["formal_covariance_health"]["passed"] is False
    assert report["formal_covariance_health"]["valid_cases"] == 63
