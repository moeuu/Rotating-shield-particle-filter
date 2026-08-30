"""Tests for evidence-bounded RA-L manuscript figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_ral_figures as figures


def _write_json(path: Path, payload: object) -> None:
    """Write one JSON fixture and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_completed_run(root: Path, *, status: str = "complete") -> Path:
    """Write one minimal durable completed-run artifact set."""
    run_id = "fixture-run"
    pf_output = root / "pf_output"
    measurement_log = root / "measurement_log"
    pf_output.mkdir(parents=True, exist_ok=True)
    measurement_log.mkdir(parents=True, exist_ok=True)
    _write_json(
        pf_output / "closed_loop_result.json",
        {
            "status": status,
            "run_id": run_id,
            "record_count": 4,
            "station_count": 2,
        },
    )
    _write_json(
        root / "truth_manifest.json",
        {
            "run_id": run_id,
            "sources": [
                {
                    "isotope": "Cs-137",
                    "position": [1.0, 2.0, 3.0],
                    "intensity_cps_1m": 1_000_000.0,
                },
                {
                    "isotope": "Co-60",
                    "position": [8.0, 12.0, 1.0],
                    "intensity_cps_1m": 1_200_000.0,
                },
            ],
        },
    )
    _write_json(
        measurement_log / "environment.json",
        {
            "size_x": 10.0,
            "size_y": 15.0,
            "size_z": 5.0,
            "obstacle_grid": {
                "cell_size": 1.0,
                "origin": [0.0, 0.0],
                "blocked_cells": [[4, 5], [4, 6]],
                "transport_boxes_m": [[4.0, 5.0, 0.0, 5.0, 7.0, 2.0]],
            },
        },
    )
    _write_json(
        pf_output / "pf_posterior.json",
        {
            "provenance": {"estimator_commit": "fixture-predecessor"},
            "isotopes": {
                "Cs-137": {
                    "modes": [
                        {
                            "label_index": 0,
                            "position_medoid_xyz": [1.2, 2.1, 3.0],
                            "strength_representative_cps_1m": 900_000.0,
                        },
                        {
                            "label_index": 1,
                            "position_medoid_xyz": [7.0, 1.0, 5.0],
                            "strength_representative_cps_1m": 310_000.0,
                        },
                    ]
                },
                "Co-60": {
                    "modes": [
                        {
                            "label_index": 0,
                            "position_medoid_xyz": [8.1, 11.8, 1.1],
                            "strength_representative_cps_1m": 1_250_000.0,
                        }
                    ]
                },
            },
        },
    )
    trace_rows = []
    for station_id, cs_cardinality in enumerate((2, 8)):
        trace_rows.append(
            {
                "station_id": station_id,
                "posterior_snapshot": {
                    "isotopes": {
                        "Cs-137": {
                            "map_cardinality": cs_cardinality,
                            "cardinality_distribution": {
                                str(cs_cardinality): 0.9,
                                "8": 0.1 if cs_cardinality == 8 else 0.0,
                            },
                        },
                        "Co-60": {
                            "map_cardinality": 1,
                            "cardinality_distribution": {"1": 1.0, "8": 0.0},
                        },
                    }
                },
            }
        )
    (pf_output / "pf_station_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace_rows) + "\n",
        encoding="utf-8",
    )
    np.savez(
        measurement_log / "observations.npz",
        station_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
        detector_pose_xyz=np.asarray(
            [
                [1.0, 1.0, 0.5],
                [1.0, 1.0, 0.5],
                [2.0, 3.0, 0.5],
                [2.0, 3.0, 0.5],
            ],
            dtype=np.float64,
        ),
        fe_orientation_index=np.asarray([0, 1, 2, 3], dtype=np.int64),
        pb_orientation_index=np.asarray([4, 5, 6, 7], dtype=np.int64),
        live_time_s=np.full(4, 20.0, dtype=np.float64),
    )
    positions_cs = np.asarray(
        [
            [[1.1, 2.0, 3.0], [7.0, 1.0, 5.0]],
            [[1.2, 2.1, 3.0], [7.1, 1.0, 5.0]],
            [[1.0, 2.2, 3.1], [6.9, 1.1, 4.9]],
            [[1.1, 1.9, 3.0], [7.0, 0.9, 5.0]],
        ],
        dtype=np.float64,
    )
    positions_co = np.asarray(
        [
            [[8.0, 12.0, 1.0], [0.0, 0.0, 0.0]],
            [[8.1, 11.9, 1.1], [0.0, 0.0, 0.0]],
            [[7.9, 12.1, 1.0], [0.0, 0.0, 0.0]],
            [[8.0, 11.8, 0.9], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    np.savez(
        pf_output / "pf_particles.npz",
        isotope_names=np.asarray(["Cs-137", "Co-60"]),
        weights_n=np.full(4, 0.25, dtype=np.float64),
        isotope_000_positions_nk3=positions_cs,
        isotope_000_source_mask_nk=np.ones((4, 2), dtype=bool),
        isotope_001_positions_nk3=positions_co,
        isotope_001_source_mask_nk=np.asarray(
            [[True, False], [True, False], [True, False], [True, False]],
            dtype=bool,
        ),
    )
    return root


def test_render_concept_figures_write_files(tmp_path: Path) -> None:
    """Concept figure rendering should write nonempty PDF files."""
    fig1 = figures.render_problem_setting(tmp_path / "fig1.pdf")
    fig2 = figures.render_method_overview(tmp_path / "fig2.pdf")

    assert fig1.exists()
    assert fig2.exists()
    assert fig1.stat().st_size > 1000
    assert fig2.stat().st_size > 1000


def test_completed_run_loader_and_figure_are_auditable(tmp_path: Path) -> None:
    """A verified run should produce metrics, a PDF, and a provenance manifest."""
    run_dir = _write_completed_run(tmp_path / "run")
    bundle = figures.load_completed_run(run_dir)
    output = figures.render_completed_run_summary(run_dir, tmp_path / "result.pdf")
    provenance = figures.write_figure_provenance(
        [output],
        tmp_path / "figure_provenance.json",
        completed_run_dir=run_dir,
    )
    metrics = figures.completed_run_metrics(bundle)

    assert output.exists()
    assert output.stat().st_size > 1000
    assert provenance.exists()
    assert metrics["source_count"] == 2
    assert metrics["position_pass_count"] == 2
    assert metrics["joint_position_strength_pass_count"] == 2
    assert metrics["final_hard_cap_mass"]["Cs-137"] == pytest.approx(0.1)


def test_completed_run_loader_fails_closed_on_incomplete_run(tmp_path: Path) -> None:
    """Incomplete runs must not be rendered as completed paper evidence."""
    run_dir = _write_completed_run(tmp_path / "run", status="failed")

    with pytest.raises(ValueError, match="not complete"):
        figures.load_completed_run(run_dir)


def test_completed_run_loader_fails_closed_on_run_id_mismatch(tmp_path: Path) -> None:
    """Truth and public result identifiers must match before visualization."""
    run_dir = _write_completed_run(tmp_path / "run")
    truth_path = run_dir / "truth_manifest.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["run_id"] = "different-run"
    _write_json(truth_path, truth)

    with pytest.raises(ValueError, match="run_id values differ"):
        figures.load_completed_run(run_dir)
