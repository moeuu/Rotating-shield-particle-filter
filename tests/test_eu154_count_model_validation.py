"""Tests for Eu-154 count-model residual validation tooling."""

from __future__ import annotations

import json

from scripts.validate_eu154_count_model import summarize_eu154_count_model


def test_summarize_eu154_count_model_reads_summary(tmp_path) -> None:
    """Eu-154 validator should compute relative residual fields from summaries."""
    summary_path = tmp_path / "result_summary_test.json"
    summary_path.write_text(
        json.dumps(
            {
                "measurements_completed": 12,
                "isotope_count_residual_diagnostics": {
                    "Eu-154": {
                        "reported_source_count": 2,
                        "observed_total_counts": 100.0,
                        "predicted_total_counts": 70.0,
                        "positive_residual_total_counts": 30.0,
                        "negative_residual_total_counts": 0.0,
                        "residual_chi2": 9.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_eu154_count_model([summary_path.as_posix()])

    assert len(rows) == 1
    assert rows[0]["measurements_completed"] == 12
    assert rows[0]["reported_source_count"] == 2
    assert rows[0]["relative_bias"] == -0.3
    assert rows[0]["absolute_relative_error"] == 0.3
    assert rows[0]["underprediction_fraction"] == 0.3
