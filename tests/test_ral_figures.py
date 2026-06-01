"""Tests for RA-L manuscript figure generation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts import build_ral_figures as figures


def _write_summary(
    path: Path,
    *,
    tag: str,
    estimates: int = 1,
    position_error: float = 0.5,
) -> Path:
    """Write a compact result summary fixture."""
    trace_path = path.with_name(f"{tag}_trace.jsonl")
    trace_records = [
        {
            "robot_position": [1.0, 1.0, 0.5],
            "step_index": 0,
        },
        {
            "robot_position": [2.0, 2.5, 0.5],
            "step_index": 1,
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in trace_records) + "\n",
        encoding="utf-8",
    )
    payload = {
        "ground_truth_sources": {
            "Cs-137": [
                {"pos": [1.0, 3.0, 2.0], "strength": 30000.0},
            ]
        },
        "estimated_sources": {
            "Cs-137": [
                {"pos": [1.2, 3.1, 2.0], "strength": 29000.0}
                for _ in range(estimates)
            ]
        },
        "match_metrics": {
            "isotopes": {
                "Cs-137": {
                    "counts": {
                        "assigned": min(estimates, 1),
                        "est": estimates,
                        "fp": max(estimates - 1, 0),
                        "fn": max(1 - estimates, 0),
                    },
                    "position_error": {"mean": position_error},
                    "intensity_rel_error_pct": {"mean": 8.0},
                }
            }
        },
        "measurements_completed": 8,
        "mission_metrics": {
            "path_segments": [
                {
                    "waypoints_xyz": [
                        [1.0, 1.0, 0.5],
                        [1.0, 2.0, 0.5],
                        [2.0, 2.5, 0.5],
                    ],
                    "distance_m": 2.1,
                    "obstacle_aware": True,
                }
            ]
        },
        "final_particle_cloud": {
            "Cs-137": {
                "positions": [
                    [1.0, 3.0, 2.0],
                    [1.2, 3.1, 2.0],
                    [1.3, 3.2, 2.1],
                ],
                "weights": [0.5, 0.3, 0.2],
            }
        },
        "output_paths": {
            "intermediate_estimate_trace_jsonl": trace_path.as_posix(),
        },
        "remaining_measurement_estimates": [
            {
                "current_station_count": 1,
                "estimated_remaining_stations": 2,
                "isotope_details": {"Cs-137": {"map_source_count": 0}},
            },
            {
                "current_station_count": 2,
                "estimated_remaining_stations": 1,
                "isotope_details": {"Cs-137": {"map_source_count": estimates}},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_render_concept_figures_write_files(tmp_path: Path) -> None:
    """Concept figure rendering should write nonempty PDF files."""
    fig1 = figures.render_problem_setting(tmp_path / "fig1.pdf")
    fig2 = figures.render_method_overview(tmp_path / "fig2.pdf")

    assert fig1.exists()
    assert fig2.exists()
    assert fig1.stat().st_size > 1000
    assert fig2.stat().st_size > 1000


def test_render_experiment_summary_writes_png(tmp_path: Path) -> None:
    """Experiment summary rendering should create a readable RA-L-scale PNG."""
    proposed = _write_summary(
        tmp_path / "result_summary_mix9_multi_isotope_cardinality_proposed_seed_test.json",
        tag="proposed",
        estimates=1,
        position_error=0.4,
    )
    baseline = _write_summary(
        tmp_path / "result_summary_mix9_multi_isotope_cardinality_baseline_passive_equal_time_no_shield_seed_test.json",
        tag="baseline",
        estimates=2,
        position_error=1.1,
    )

    output = figures.render_experiment_summary([proposed, baseline], tmp_path / "summary.png")

    assert output.exists()
    with Image.open(output) as image:
        assert image.size[0] > 1200
        assert image.size[1] > 900
