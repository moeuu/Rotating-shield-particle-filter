"""Tests for evidence-bounded RA-L manuscript figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import build_ral_figures as figures


def _write_json(path: Path, payload: object) -> None:
    """Write one JSON fixture and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_completed_run(
    root: Path,
    *,
    execution_status: str = "complete",
) -> Path:
    """Write one minimal durable completed-run artifact set."""
    run_id = "fixture-run"
    pf_output = root / "pf_output"
    measurement_log = root / "measurement_log"
    pf_output.mkdir(parents=True, exist_ok=True)
    measurement_log.mkdir(parents=True, exist_ok=True)
    _write_json(
        pf_output / "closed_loop_result.json",
        {
            "schema_version": 2,
            "execution_status": execution_status,
            "sampler_quality_status": "failed",
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
            "provenance": {
                "estimator_commit": "fixture-predecessor",
                "measurement_log_sha256": "fixture-log",
            },
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
    metadata_rows = [
        {
            "run_id": run_id,
            "step_id": step_id,
            "array_index": step_id,
            "station_id": 0 if step_id < 2 else 1,
            "metadata": (
                {
                    "travel_waypoints_xyz": [
                        [1.0, 1.0, 0.5],
                        [1.4, 1.5, 0.5],
                        [2.0, 3.0, 0.5],
                    ]
                }
                if step_id == 2
                else {}
            ),
        }
        for step_id in range(4)
    ]
    (measurement_log / "observation_metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
        encoding="utf-8",
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
    _write_json(
        pf_output / "pf_figure_data.json",
        {
            "schema_version": 1,
            "artifact_family": "pf_result_figure_data",
            "run_identity": {
                "run_id": run_id,
                "measurement_log_sha256": "fixture-log",
            },
            "coordinate_system": {
                "frame": "MeasurementLog world frame",
                "axis_order": ["x", "y", "z"],
                "length_unit": "m",
            },
            "route": {
                "schema_version": 1,
                "travel_path_segments_xyz": [
                    [[1.0, 1.0, 0.5], [1.4, 1.5, 0.5], [2.0, 3.0, 0.5]]
                ],
                "measurement_stations": [],
                "current_detector_position_xyz": [2.0, 3.0, 0.5],
                "latest_spectrum_counts": None,
                "energy_bin_edges_keV": None,
                "latest_step_id": 3,
            },
            "truth_included": False,
        },
    )
    return root


def _write_split_aware_evaluation(root: Path) -> Path:
    """Write one schema-v3 split-aware evaluation for the completed fixture."""
    cs_truth = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    cs_positions = np.asarray(
        [[1.2, 2.1, 3.0], [7.0, 1.0, 5.0]],
        dtype=np.float64,
    )
    cs_strengths = np.asarray([900_000.0, 310_000.0], dtype=np.float64)
    cs_total = float(np.sum(cs_strengths))
    cs_centroid = np.sum(cs_strengths[:, None] * cs_positions, axis=0) / cs_total
    cs_centroid_error = float(np.linalg.norm(cs_centroid - cs_truth))
    cs_rms_error = float(
        np.sqrt(
            np.sum(cs_strengths * np.sum(np.square(cs_positions - cs_truth), axis=1))
            / cs_total
        )
    )
    co_truth = np.asarray([8.0, 12.0, 1.0], dtype=np.float64)
    co_position = np.asarray([8.1, 11.8, 1.1], dtype=np.float64)
    co_error = float(np.linalg.norm(co_position - co_truth))
    evaluation_path = root / "split_aware_evaluation.json"
    _write_json(
        evaluation_path,
        {
            "schema_version": 3,
            "artifact_family": "completed_pf_cluster_accuracy_evaluation",
            "execution_status": "complete",
            "changes_pf_state_or_cardinality": False,
            "run_identity": {
                "run_id": "fixture-run",
                "measurement_log_sha256": "fixture-log",
            },
            "criteria": {
                "merged_position_summary": "strength_weighted_centroid",
                "position_target_metric": ("strength_weighted_rms_distance_to_truth"),
                "merged_source_count_semantics": (
                    "one_per_truth_cluster_plus_response_distinct_remote"
                ),
            },
            "isotopes": {
                "Co-60": {
                    "truth_sources": [
                        {
                            "truth_source_index": 0,
                            "assigned_estimate_indices": [0],
                            "assigned_raw_component_count": 1,
                            "merged_position_xyz_m": co_position.tolist(),
                            "combined_estimated_strength_cps_1m": 1_250_000.0,
                            "merged_centroid_position_error_m": co_error,
                            "strength_weighted_rms_position_error_m": co_error,
                            "combined_relative_strength_error": (
                                50_000.0 / 1_200_000.0
                            ),
                        }
                    ]
                },
                "Cs-137": {
                    "truth_sources": [
                        {
                            "truth_source_index": 0,
                            "assigned_estimate_indices": [0, 1],
                            "assigned_raw_component_count": 2,
                            "merged_position_xyz_m": cs_centroid.tolist(),
                            "combined_estimated_strength_cps_1m": cs_total,
                            "merged_centroid_position_error_m": cs_centroid_error,
                            "strength_weighted_rms_position_error_m": cs_rms_error,
                            "combined_relative_strength_error": 0.21,
                        }
                    ]
                },
            },
        },
    )
    return evaluation_path


def test_render_concept_figures_write_files(tmp_path: Path) -> None:
    """Authenticated Isaac figure composition should write nonempty PDFs."""
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    image_paths: list[Path] = []
    for index in range(4):
        image_path = capture_dir / f"shield_{index}.png"
        plt.imsave(image_path, np.full((240, 320, 3), 0.5 + index * 0.05))
        image_paths.append(image_path)
    environment_path = capture_dir / "environment.png"
    plt.imsave(environment_path, np.full((240, 400, 3), 0.7))
    provenance_path = capture_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "detector_sequence": {
                    "recorded_pair_ids": [1, 45, 27, 49],
                }
            }
        ),
        encoding="utf-8",
    )

    fig1 = figures.render_experiment_environment(
        tmp_path / "fig1.pdf",
        image_path=environment_path,
    )
    fig2 = figures.render_detector_shield_sequence(
        tmp_path / "fig2.pdf",
        image_paths=tuple(image_paths),
        provenance_path=provenance_path,
    )

    assert fig1.exists()
    assert fig2.exists()
    assert fig1.stat().st_size > 1000
    assert fig2.stat().st_size > 1000


def test_detector_sequence_arrows_are_between_adjacent_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each sequence arrow should occupy its own inter-panel column."""
    image_paths: list[Path] = []
    for index in range(4):
        image_path = tmp_path / f"shield_{index}.png"
        plt.imsave(image_path, np.full((240, 320, 3), 0.5 + index * 0.05))
        image_paths.append(image_path)
    provenance_path = tmp_path / "provenance.json"
    _write_json(
        provenance_path,
        {"detector_sequence": {"recorded_pair_ids": [1, 45, 27, 49]}},
    )
    captured: dict[str, plt.Figure] = {}

    def capture_figure(fig: plt.Figure, output_path: Path) -> Path:
        """Retain the unsaved figure so its final layout can be inspected."""
        captured["figure"] = fig
        return output_path

    monkeypatch.setattr(figures, "save_figure", capture_figure)
    figures.render_detector_shield_sequence(
        tmp_path / "fig2.pdf",
        image_paths=tuple(image_paths),
        provenance_path=provenance_path,
    )

    fig = captured["figure"]
    panel_axes = [ax for ax in fig.axes if not ax.get_label().startswith("sequence-")]
    arrow_axes = [ax for ax in fig.axes if ax.get_label().startswith("sequence-")]
    assert len(panel_axes) == 4
    assert len(arrow_axes) == 3
    for index, arrow_ax in enumerate(arrow_axes):
        left_box = panel_axes[index].get_position()
        arrow_box = arrow_ax.get_position()
        right_box = panel_axes[index + 1].get_position()
        assert left_box.x1 <= arrow_box.x0 < arrow_box.x1 <= right_box.x0
        assert len(arrow_ax.texts) == 1
    plt.close(fig)


def test_capture_provenance_sources_are_hash_verified(tmp_path: Path) -> None:
    """Paper provenance must transitively bind the raw Isaac/Geant4 inputs."""
    source = tmp_path / "actual-geant4-tracks.json"
    source.write_text("native steps\n", encoding="utf-8")
    provenance = tmp_path / "capture-provenance.json"
    _write_json(
        provenance,
        {
            "source_files": [
                {
                    "path": source.resolve().as_posix(),
                    "sha256": figures._sha256(source),
                }
            ]
        },
    )

    assert figures._verified_capture_source_paths(provenance) == [source.resolve()]
    source.write_text("altered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from provenance"):
        figures._verified_capture_source_paths(provenance)


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
    assert len(bundle.route_segments_xyz) == 1
    np.testing.assert_allclose(
        bundle.route_segments_xyz[0],
        np.asarray([[1.0, 1.0, 0.5], [1.4, 1.5, 0.5], [2.0, 3.0, 0.5]]),
    )
    source_paths = {
        row["path"]
        for row in json.loads(provenance.read_text(encoding="utf-8"))["source_files"]
    }
    figure_data_path = run_dir / "pf_output" / "pf_figure_data.json"
    assert figure_data_path.resolve().as_posix() in source_paths


def test_split_aware_current_run_figure_uses_merged_source_metrics(
    tmp_path: Path,
) -> None:
    """A schema-v3 result should plot raw modes but score merged RMS errors."""
    run_dir = _write_completed_run(tmp_path / "run")
    evaluation_path = _write_split_aware_evaluation(run_dir)
    bundle = figures.load_split_aware_completed_run(run_dir, evaluation_path)
    output = figures.render_completed_run_summary(
        run_dir,
        tmp_path / "current_result.pdf",
        split_aware_evaluation=evaluation_path,
    )
    scene_output = figures.render_completed_run_scene(
        run_dir,
        tmp_path / "current_scene.pdf",
        split_aware_evaluation=evaluation_path,
    )
    provenance = figures.write_figure_provenance(
        [output, scene_output],
        tmp_path / "current_provenance.json",
        completed_run_dir=run_dir,
        split_aware_evaluation=evaluation_path,
    )
    metrics = figures.completed_run_metrics(bundle)

    assert bundle.predecessor_code is False
    assert len(bundle.split_aware_results) == 2
    assert bundle.split_aware_results[1].assigned_component_indices == (1, 2)
    assert metrics["evidence_status"] == "completed_proposed_split_aware_result"
    assert metrics["position_pass_count"] == 1
    assert metrics["joint_position_strength_pass_count"] == 1
    assert output.exists()
    assert scene_output.exists()
    assert provenance.exists()
    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert "manuscript_scene" in provenance_payload["transformations"]
    source_paths = {
        row["path"]
        for row in provenance_payload["source_files"]
    }
    assert evaluation_path.resolve().as_posix() in source_paths


def test_result_projections_share_station_order_without_duplicate_room_frame(
    tmp_path: Path,
) -> None:
    """Both paper projections must share labels and avoid a mystery room box."""
    run_dir = _write_completed_run(tmp_path / "run")
    evaluation_path = _write_split_aware_evaluation(run_dir)
    bundle = figures.load_split_aware_completed_run(run_dir, evaluation_path)
    figure, (floor_axis, elevation_axis) = plt.subplots(1, 2)

    with mock.patch.object(
        floor_axis,
        "scatter",
        wraps=floor_axis.scatter,
    ) as floor_scatter:
        figures._plot_projection(
            floor_axis,
            bundle,
            projection="xy",
            title="Recorded floor map",
            label_truth_ids=False,
            emphasize_stations=True,
            show_station_labels=True,
            show_truth_estimate_links=False,
            show_raw_components=False,
            use_cui_source_markers=True,
        )
    figures._plot_projection(
        elevation_axis,
        bundle,
        projection="xz",
        title="Recorded elevation map",
        label_truth_ids=False,
        show_route_segments=False,
        emphasize_stations=True,
        show_station_labels=True,
        show_truth_estimate_links=False,
    )

    expected_station_labels = {"0", "1"}
    assert {text.get_text() for text in floor_axis.texts}.issuperset(
        expected_station_labels
    )
    assert {text.get_text() for text in elevation_axis.texts}.issuperset(
        expected_station_labels
    )
    assert len(floor_axis.lines) == 1
    assert not elevation_axis.lines
    source_markers = [
        call.kwargs.get("marker")
        for call in floor_scatter.call_args_list
        if call.kwargs.get("marker") is not None
    ]
    assert source_markers.count("*") == len(bundle.truth_sources)
    assert source_markers.count("x") == len(bundle.split_aware_results)
    assert "P" not in source_markers
    assert "X" not in source_markers
    room_width, room_height, _ = bundle.room_xyz_m
    full_room_patches = [
        patch
        for patch in floor_axis.patches
        if hasattr(patch, "get_xy")
        and tuple(float(value) for value in patch.get_xy()) == (0.0, 0.0)
        and float(patch.get_width()) == pytest.approx(room_width)
        and float(patch.get_height()) == pytest.approx(room_height)
    ]
    assert not full_room_patches
    plt.close(figure)


def test_split_aware_current_run_figure_rejects_altered_aggregation(
    tmp_path: Path,
) -> None:
    """Figure input must recompute merged quantities from bound PF components."""
    run_dir = _write_completed_run(tmp_path / "run")
    evaluation_path = _write_split_aware_evaluation(run_dir)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["isotopes"]["Cs-137"]["truth_sources"][0][
        "combined_estimated_strength_cps_1m"
    ] += 1.0
    _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="combined strength"):
        figures.load_split_aware_completed_run(run_dir, evaluation_path)


def test_completed_run_loader_fails_closed_on_incomplete_run(tmp_path: Path) -> None:
    """Incomplete runs must not be rendered as completed paper evidence."""
    run_dir = _write_completed_run(
        tmp_path / "run",
        execution_status="failed",
    )

    with pytest.raises(ValueError, match="not complete"):
        figures.load_completed_run(run_dir)


def test_completed_run_loader_uses_authenticated_log_route_fallback(
    tmp_path: Path,
) -> None:
    """A legacy run may recover only exact waypoints saved in its MeasurementLog."""
    run_dir = _write_completed_run(tmp_path / "run")
    (run_dir / "pf_output" / "pf_figure_data.json").unlink()

    bundle = figures.load_completed_run(run_dir)

    assert len(bundle.route_segments_xyz) == 1
    np.testing.assert_array_equal(
        bundle.route_segments_xyz[0],
        np.asarray(
            [[1.0, 1.0, 0.5], [1.4, 1.5, 0.5], [2.0, 3.0, 0.5]]
        ),
    )


def test_completed_run_loader_fails_closed_on_run_id_mismatch(tmp_path: Path) -> None:
    """Truth and public result identifiers must match before visualization."""
    run_dir = _write_completed_run(tmp_path / "run")
    truth_path = run_dir / "truth_manifest.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["run_id"] = "different-run"
    _write_json(truth_path, truth)

    with pytest.raises(ValueError, match="run_id values differ"):
        figures.load_completed_run(run_dir)
