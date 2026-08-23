"""Tests for the explicit post-run private-truth join."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.private_truth import load_private_truth_for_completed_result


def _write(path: Path, payload: dict[str, object]) -> None:
    """Write one compact JSON test artifact."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _truth(run_id: str) -> dict[str, object]:
    """Return one valid private truth-manifest payload."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_profile": "private-profile",
        "scene_seed": 1234,
        "scene_rng_provenance": {"algorithm": "test"},
        "sources": [
            {
                "isotope": "Cs-137",
                "position": [1.0, 2.0, 3.0],
                "intensity_cps_1m": 100.0,
            }
        ],
    }


def test_private_truth_joins_only_after_exact_completed_run_id(tmp_path: Path) -> None:
    """Evaluation may load private truth only for its completed PF run."""
    result_path = tmp_path / "result.json"
    truth_path = tmp_path / "truth.json"
    _write(result_path, {"status": "complete", "run_id": "opaque-run"})
    _write(truth_path, _truth("opaque-run"))

    joined = load_private_truth_for_completed_result(result_path, truth_path)

    assert joined.run_id == "opaque-run"
    assert joined.scene_seed == 1234
    assert len(joined.sources) == 1


@pytest.mark.parametrize(
    ("status", "truth_run_id", "message"),
    (
        ("running", "opaque-run", "completed"),
        ("complete", "different-run", "differs"),
    ),
)
def test_private_truth_join_fails_closed(
    tmp_path: Path,
    status: str,
    truth_run_id: str,
    message: str,
) -> None:
    """Incomplete or cross-run evaluation joins must be rejected."""
    result_path = tmp_path / "result.json"
    truth_path = tmp_path / "truth.json"
    _write(result_path, {"status": status, "run_id": "opaque-run"})
    _write(truth_path, _truth(truth_run_id))

    with pytest.raises(ValueError, match=message):
        load_private_truth_for_completed_result(result_path, truth_path)
