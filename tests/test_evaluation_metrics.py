"""Tests for evaluation metrics matching and counts."""

import json

import pytest

from evaluation_metrics import compute_metrics, save_metrics_json


def test_compute_metrics_counts_with_gate() -> None:
    """Outside-radius assignments must remain unmatched in standard metrics."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
            {"pos": [5.0, 0.0, 0.0], "strength": 200.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.1, 0.0, 0.0], "strength": 110.0},
            {"pos": [10.0, 0.0, 0.0], "strength": 50.0},
        ]
    }
    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=0.5)
    counts = metrics["isotopes"]["Cs-137"]["counts"]
    assert counts["gt"] == 2
    assert counts["est"] == 2
    assert counts["assigned"] == 1
    assert counts["assigned_all"] == 2
    assert counts["matched"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 1
    assert len(metrics["isotopes"]["Cs-137"]["matches"]) == 1
    assert metrics["matching_policy"]["strength_used_for_assignment"] is False
    assert metrics["matching_policy"]["outside_radius_behavior"] == "unmatched"


def test_threshold_matching_is_recomputed_without_strength_cost() -> None:
    """Every threshold should maximize valid spatial matches independently."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
            {"pos": [3.0, 0.0, 0.0], "strength": 1000.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.1, 0.0, 0.0], "strength": 100.0},
            {"pos": [-0.1, 0.0, 0.0], "strength": 1000.0},
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        distance_thresholds_m=(0.5, 3.0),
    )

    isotope = metrics["isotopes"]["Cs-137"]
    assert isotope["threshold_counts"]["0.5m"]["tp"] == 1
    assert isotope["threshold_counts"]["3m"]["tp"] == 2


def test_compute_metrics_position_target_flag() -> None:
    """Position-error summaries should include the fixed 0.5 m target check."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.3, 0.0, 0.0], "strength": 95.0},
        ]
    }
    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=0.5)
    position_error = metrics["isotopes"]["Cs-137"]["position_error"]
    assert position_error["target_m"] == 0.5
    assert position_error["within_target"] is True


def test_compute_metrics_reports_threshold_precision_recall() -> None:
    """Metrics should expose localization-threshold counts for paper tables."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
            {"pos": [5.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.5, 0.0, 0.0], "strength": 100.0},
            {"pos": [8.0, 0.0, 0.0], "strength": 100.0},
            {"pos": [20.0, 0.0, 0.0], "strength": 100.0},
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        distance_thresholds_m=(1.0, 3.0),
    )

    data = metrics["isotopes"]["Cs-137"]
    assert data["counts"]["source_count_error"] == 1
    assert data["counts"]["source_count_abs_error"] == 1
    assert data["threshold_counts"]["1m"]["tp"] == 1
    assert data["threshold_counts"]["1m"]["precision"] == 1 / 3
    assert data["threshold_counts"]["1m"]["recall"] == 1 / 2
    assert data["threshold_counts"]["3m"]["tp"] == 2
    assert data["threshold_counts"]["3m"]["precision"] == 2 / 3
    assert data["threshold_counts"]["3m"]["recall"] == 1.0


def test_compute_metrics_default_thresholds_include_half_meter() -> None:
    """Default paper metrics should include 0.5, 1, 2, and 3 m gates."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.4, 0.0, 0.0], "strength": 100.0},
        ]
    }

    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=0.5)
    threshold_counts = metrics["isotopes"]["Cs-137"]["threshold_counts"]

    assert list(threshold_counts.keys()) == ["0.5m", "1m", "2m", "3m"]
    assert threshold_counts["0.5m"]["tp"] == 1


def test_compute_metrics_reports_3d_surface_cardinality_and_strength_axes() -> None:
    """Expanded metrics should keep geometry, model order, and bias separate."""
    gt_by_iso = {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 0.0],
                "strength": 100.0,
                "surface_kind": "floor",
            },
            {
                "pos": [1.0, 0.0, 3.0],
                "strength": 200.0,
                "surface_kind": "ceiling",
            },
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {
                "pos": [0.3, 0.4, 0.0],
                "strength": 110.0,
                "surface_kind": "floor",
            },
            {
                "pos": [1.0, 0.0, 2.5],
                "strength": 150.0,
                "surface_kind": "wall",
            },
            {
                "pos": [20.0, 20.0, 0.0],
                "strength": 10.0,
                "surface_kind": "floor",
            },
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=1.0,
        close_pair_distance_m=4.0,
    )

    isotope = metrics["isotopes"]["Cs-137"]
    assert isotope["position_error"]["median"] == 0.5
    assert isotope["position_error"]["p95"] == 0.5
    assert isotope["xy_error"]["max"] == 0.5
    assert isotope["z_abs_error"]["max"] == 0.5
    assert isotope["intensity_rel_error_pct"]["p95"] == 24.25
    assert isotope["detection"]["precision"] == 2 / 3
    assert isotope["detection"]["recall"] == 1.0
    assert isotope["detection"]["f1"] == 0.8
    assert isotope["counts"]["cardinality_exact_match"] is False
    separation = isotope["same_isotope_close_pair_separation"]
    assert separation["eligible_pair_count"] == 1
    assert separation["separation_rate"] == 1.0
    surface = isotope["surface_classification"]
    assert surface["ceiling_localization_recall"] == 1.0
    assert surface["ceiling_classification_recall"] == 0.0
    assert metrics["global"]["detection"]["false_source_count"] == 1


def test_close_pair_requires_distinct_estimated_separation() -> None:
    """Two duplicate modes must not count as resolving a close truth pair."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [-0.2, 0.0, 0.0], "strength": 100.0},
            {"pos": [0.2, 0.0, 0.0], "strength": 100.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
            {"pos": [0.0, 0.0, 0.0], "strength": 100.0},
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        close_pair_distance_m=1.0,
        close_pair_min_estimated_separation_m=0.5,
    )

    separation = metrics["isotopes"]["Cs-137"][
        "same_isotope_close_pair_separation"
    ]
    assert separation["matched_pair_count"] == 1
    assert separation["separated_pair_count"] == 0
    assert separation["separation_rate"] == 0.0
    assert separation["pairs"][0]["estimated_separation_m"] == 0.0
    assert separation["pairs"][0]["resolved"] is False


def test_all_surface_labels_and_posterior_scores_are_reported() -> None:
    """Obstacle surfaces should participate in hard and probabilistic scoring."""
    gt_by_iso = {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 1.0],
                "strength": 100.0,
                "surface_kind": "obstacle_top",
            }
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 1.0],
                "strength": 100.0,
                "surface_kind": "obstacle_top",
            }
        ]
    }
    uncertainty = {
        "Cs-137": [
            {
                "mode_index": 0,
                "surface_posterior_available": True,
                "surface_kind_posterior": {
                    "floor": 0.2,
                    "wall": 0.0,
                    "ceiling": 0.0,
                    "obstacle_side": 0.0,
                    "obstacle_top": 0.8,
                    "off_surface": 0.0,
                },
            }
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        uncertainty_by_iso=uncertainty,
    )

    surface = metrics["isotopes"]["Cs-137"]["surface_classification"]
    assert set(surface["labels"]) == {
        "floor",
        "wall",
        "ceiling",
        "obstacle_side",
        "obstacle_top",
        "off_surface",
    }
    assert surface["confusion_matrix"]["obstacle_top"]["obstacle_top"] == 1
    assert surface["posterior_scoring"]["multiclass_brier_score"] == pytest.approx(
        0.08
    )
    assert surface["posterior_scoring"]["negative_log_score"] == pytest.approx(
        0.22314355131420976
    )


def test_compute_metrics_reports_cross_isotope_confusion() -> None:
    """Spatially correct but isotope-swapped hotspots should populate confusion."""
    gt_by_iso = {
        "Cs-137": [{"pos": [0.0, 0.0, 0.0], "strength": 100.0}],
        "Co-60": [{"pos": [5.0, 0.0, 0.0], "strength": 100.0}],
    }
    est_by_iso = {
        "Cs-137": [{"pos": [5.0, 0.0, 0.0], "strength": 100.0}],
        "Co-60": [{"pos": [0.0, 0.0, 0.0], "strength": 100.0}],
    }

    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=0.5)

    confusion = metrics["global"]["isotope_confusion"]
    assert confusion["localized_pair_count"] == 2
    assert confusion["correct_isotope_count"] == 0
    assert confusion["accuracy_among_localized"] == 0.0
    assert confusion["matrix"]["Cs-137"]["Co-60"] == 1
    assert confusion["matrix"]["Co-60"]["Cs-137"] == 1


def test_compute_metrics_reports_posterior_interval_coverage() -> None:
    """Matched truth should be checked against both ellipsoid and z intervals."""
    gt_by_iso = {
        "Cs-137": [{"pos": [0.0, 0.0, 1.0], "strength": 100.0}],
    }
    est_by_iso = {
        "Cs-137": [{"pos": [0.0, 0.0, 1.0], "strength": 100.0}],
    }
    uncertainty = {
        "Cs-137": [
            {
                "mode_index": 0,
                "weighted_mean_xyz_m": [0.0, 0.0, 1.0],
                "z_quantiles_m": {"q05": 0.8, "q50": 1.0, "q95": 1.2},
                "ellipsoid_90": {
                    "semi_axis_lengths_m": [0.5, 0.5, 0.5],
                    "orientation_matrix_xyz_by_axis": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        uncertainty_by_iso=uncertainty,
    )

    coverage = metrics["global"]["uncertainty_coverage"]
    assert coverage["position_interval_evaluable_count"] == 1
    assert coverage["position_interval_90_coverage"] == 1.0
    assert coverage["z_interval_evaluable_count"] == 1
    assert coverage["z_interval_90_coverage"] == 1.0
    assert "Gaussian-equivalent" in coverage["position_interval_interpretation"]
    assert "not an empirical" in coverage["position_interval_interpretation"]
    json.dumps(metrics, allow_nan=False)


def test_uncertainty_coverage_counts_misses_and_excludes_bad_references() -> None:
    """End-to-end coverage counts misses but excludes invalid posterior references."""
    gt_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 1.0], "strength": 100.0},
            {"pos": [2.0, 0.0, 1.0], "strength": 100.0},
            {"pos": [4.0, 0.0, 1.0], "strength": 100.0},
            {"pos": [8.0, 0.0, 1.0], "strength": 100.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {"pos": [0.0, 0.0, 1.0], "strength": 100.0},
            {"pos": [2.0, 0.0, 1.0], "strength": 100.0},
            {"pos": [4.0, 0.0, 1.0], "strength": 100.0},
        ]
    }
    common = {
        "weighted_mean_xyz_m": [0.0, 0.0, 1.0],
        "z_quantiles_m": {"q05": 0.5, "q50": 1.0, "q95": 1.5},
        "ellipsoid_90": {
            "semi_axis_lengths_m": [0.5, 0.5, 0.5],
            "orientation_matrix_xyz_by_axis": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }
    uncertainty = {
        "Cs-137": [
            {
                **common,
                "mode_index": 0,
                "existence_mass": 0.9,
                "reference_consistent": True,
                "posterior_support_available": True,
            },
            {
                **common,
                "mode_index": 1,
                "reference_consistent": False,
                "posterior_support_available": True,
            },
            {
                **common,
                "mode_index": 2,
                "reference_consistent": True,
                "posterior_support_available": False,
            },
        ]
    }

    metrics = compute_metrics(
        gt_by_iso,
        est_by_iso,
        match_radius_m=0.5,
        uncertainty_by_iso=uncertainty,
    )

    coverage = metrics["isotopes"]["Cs-137"]["uncertainty_coverage"]
    assert coverage["position_interval_evaluable_count"] == 1
    assert coverage["position_interval_90_coverage"] == 1.0
    assert coverage["position_interval_end_to_end_evaluable_count"] == 2
    assert coverage["position_interval_90_coverage_end_to_end"] == 0.5
    assert coverage["excluded_diagnostic_count_by_reason"] == {
        "reference_inconsistent": 1,
        "posterior_support_unavailable": 1,
    }
    existence = coverage["existence_mass_calibration"]
    assert existence["evaluable_count"] == 2
    assert existence["brier_score"] == pytest.approx(0.505)


def test_global_cardinality_and_empty_availability_are_explicit() -> None:
    """Global cardinality should retain absolute over- and under-count totals."""
    metrics = compute_metrics(
        {"Cs-137": [], "Co-60": [{"pos": [0, 0, 0], "strength": 1.0}]},
        {"Cs-137": [{"pos": [0, 0, 0], "strength": 1.0}], "Co-60": []},
        match_radius_m=0.5,
    )
    cardinality = metrics["global"]["cardinality"]
    assert cardinality["source_count_error"] == 0
    assert cardinality["source_count_abs_error"] == 2
    assert cardinality["overestimated_source_count"] == 1
    assert cardinality["underestimated_source_count"] == 1
    assert cardinality["exact_match_rate"] == 0.0
    assert cardinality["per_evaluable_isotope_exact_match_rate"] == 0.0

    empty = compute_metrics({}, {}, match_radius_m=0.5)
    assert empty["available"] is False
    assert empty["global"]["cardinality"]["available"] is False
    assert empty["global"]["cardinality"]["all_isotopes_exact_match"] is None


def test_cardinality_run_indicator_excludes_empty_empty_isotopes() -> None:
    """Run exact-match and per-isotope rate must not be inflated by empty keys."""
    metrics = compute_metrics(
        {
            "Cs-137": [{"pos": [0, 0, 0], "strength": 1.0}],
            "Co-60": [],
            "Eu-154": [],
        },
        {
            "Cs-137": [{"pos": [0, 0, 0], "strength": 1.0}],
            "Co-60": [{"pos": [1, 0, 0], "strength": 1.0}],
            "Eu-154": [],
        },
        match_radius_m=0.5,
    )

    cardinality = metrics["global"]["cardinality"]
    assert cardinality["exact_match_rate"] == 0.0
    assert cardinality["evaluable_isotope_count"] == 2
    assert cardinality["exact_match_isotope_count"] == 1
    assert cardinality["per_evaluable_isotope_exact_match_rate"] == 0.5
    assert cardinality["excluded_empty_empty_isotopes"] == ["Eu-154"]


def test_room_surface_three_class_excludes_non_room_truth_with_metadata() -> None:
    """The 3-class metric should score room surfaces and disclose exclusions."""
    gt_by_iso = {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 0.0],
                "strength": 1.0,
                "surface_kind": "floor",
            },
            {
                "pos": [1.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "wall",
            },
            {
                "pos": [2.0, 0.0, 3.0],
                "strength": 1.0,
                "surface_kind": "ceiling",
            },
            {
                "pos": [3.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "obstacle_side",
            },
            {
                "pos": [4.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "reference",
            },
            {"pos": [5.0, 0.0, 1.0], "strength": 1.0},
        ]
    }
    est_by_iso = {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 0.0],
                "strength": 1.0,
                "surface_kind": "floor",
            },
            {
                "pos": [1.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "ceiling",
            },
            {
                "pos": [2.0, 0.0, 3.0],
                "strength": 1.0,
                "surface_kind": "obstacle_top",
            },
            {
                "pos": [3.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "obstacle_side",
            },
            {
                "pos": [4.0, 0.0, 1.0],
                "strength": 1.0,
                "surface_kind": "reference",
            },
            {"pos": [5.0, 0.0, 1.0], "strength": 1.0},
        ]
    }

    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=0.1)
    room = metrics["global"]["surface_classification"]["room_surface_3class"]

    assert room["labels"] == ["floor", "wall", "ceiling"]
    assert room["eligible_truth_count"] == 3
    assert room["evaluable_localized_count"] == 2
    assert room["confusion_matrix"]["floor"]["floor"] == 1
    assert room["confusion_matrix"]["wall"]["ceiling"] == 1
    assert room["per_class"]["floor"]["recall"] == 1.0
    assert room["per_class"]["wall"]["recall"] == 0.0
    assert room["per_class"]["ceiling"]["recall"] == 0.0
    assert room["accuracy_among_evaluable_localized"] == 0.5
    assert room["end_to_end_accuracy"] == pytest.approx(1.0 / 3.0)
    assert room["excluded"]["truth_count_by_category"] == {
        "obstacle": 1,
        "other": 0,
        "reference": 1,
        "unknown": 1,
    }
    assert room["excluded"]["prediction_count_by_category"]["obstacle"] == 1


def test_metric_inputs_and_json_output_reject_nonfinite_values(tmp_path) -> None:
    """Metric inputs and persistence must not emit permissive NaN JSON."""
    with pytest.raises(ValueError, match="finite"):
        compute_metrics(
            {"Cs-137": [{"pos": [float("nan"), 0, 0], "strength": 1.0}]},
            {},
            match_radius_m=0.5,
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_metrics(
            {"Cs-137": [{"pos": [0, 0, 0], "strength": -1.0}]},
            {},
            match_radius_m=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_metrics({}, {}, match_radius_m=float("inf"))
    with pytest.raises(ValueError):
        save_metrics_json({"invalid": float("nan")}, str(tmp_path / "bad.json"))
