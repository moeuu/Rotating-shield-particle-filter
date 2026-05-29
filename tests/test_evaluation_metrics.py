"""Tests for evaluation metrics matching and counts."""

from evaluation_metrics import compute_metrics


def test_compute_metrics_counts_with_gate() -> None:
    """Closest matching should pair estimates even when far."""
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
    assert counts["assigned"] == 2
    assert counts["matched"] == 2
    assert counts["fp"] == 0
    assert counts["fn"] == 0


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
