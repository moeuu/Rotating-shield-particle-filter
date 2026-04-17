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
