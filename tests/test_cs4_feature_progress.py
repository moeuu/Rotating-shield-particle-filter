"""Tests for persistent feature-validation progress summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import summarize_cs4_feature_progress as progress


def test_build_status_tracks_each_seed_independently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A completed seed must not hide pending runs of the same variant."""
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    rows = [
        {"case": "cs4", "variant": "feature_all_on", "seed": "101"},
        {"case": "cs4", "variant": "feature_all_on", "seed": "202"},
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "variant", "seed"))
        writer.writeheader()
        writer.writerows(rows)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    completed_tag = "cs4_feature_all_on_seed_101"
    (results_dir / f"result_summary_{completed_tag}.json").write_text(
        json.dumps({"measurements_completed": 3}),
        encoding="utf-8",
    )

    status = progress._build_status(
        manifest_path=manifest_path,
        log_path=tmp_path / "run.log",
    )

    assert status["completed_count"] == 1
    assert status["pending_count"] == 1
    assert list(status["completed_variants"]) == [completed_tag]
    assert status["pending_variants"] == ["cs4_feature_all_on_seed_202"]
    assert status["current_variant"] == "cs4_feature_all_on_seed_202"
