"""Tests for truth-safe CUI evaluation rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualization.realtime_viz import (
    CUISplitPFVisualizer,
    PFFrame,
)


def test_cui_truth_is_hidden_until_explicit_evaluation_update(
    tmp_path: Path,
) -> None:
    """CUI truth attachment must be explicit and copy evaluation arrays."""
    visualizer = CUISplitPFVisualizer(
        isotopes=["Cs-137"],
        output_dir=tmp_path,
        true_sources={},
        true_strengths={},
    )
    assert visualizer.true_sources == {}
    assert "truth: hidden" in visualizer.index_path.read_text(encoding="utf-8")
    source = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    strength = np.asarray([400_000.0], dtype=np.float64)

    visualizer.set_truth(
        {"Cs-137": source},
        {"Cs-137": strength},
    )
    source[:] = -1.0
    strength[:] = -1.0

    np.testing.assert_array_equal(
        visualizer.true_sources["Cs-137"],
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        visualizer.true_strengths["Cs-137"],
        np.asarray([400_000.0], dtype=np.float64),
    )
    assert (
        "truth: visible (evaluation overlay only; not provided to PF/planner)"
        in visualizer.index_path.read_text(encoding="utf-8")
    )


def test_cui_writes_plain_and_neighborhood_labeled_pf_images(
    tmp_path: Path,
) -> None:
    """CUI output must retain plain PF images and add source-labeled images."""
    visualizer = CUISplitPFVisualizer(
        isotopes=["Co-60"],
        output_dir=tmp_path,
        world_bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 3.0),
        true_sources={
            "Co-60": np.asarray(
                [[1.0, 1.0, 0.5], [5.0, 5.0, 0.5]],
                dtype=float,
            )
        },
        true_strengths={"Co-60": np.asarray([1.0, 1.0], dtype=float)},
        source_label_neighborhood_m=1.0,
    )
    frame = PFFrame(
        step_index=3,
        time=120.0,
        robot_position=np.asarray([2.0, 2.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.asarray([1.0, 0.0, 0.0], dtype=float),
        RPb=np.asarray([0.0, 1.0, 0.0], dtype=float),
        duration=30.0,
        particle_positions={"Co-60": np.zeros((0, 3), dtype=float)},
        particle_weights={"Co-60": np.zeros(0, dtype=float)},
        estimated_sources={
            "Co-60": np.asarray(
                [
                    [1.2, 1.0, 0.5],
                    [1.4, 1.0, 0.5],
                    [8.0, 8.0, 0.5],
                ],
                dtype=float,
            )
        },
        estimated_strengths={
            "Co-60": np.asarray([1.0, 1.0, 1.0], dtype=float)
        },
    )

    truth_entries, estimate_entries = visualizer._source_label_entries(
        frame,
        "Co-60",
    )
    assert [label for _, label in truth_entries] == ["Co-1 T", "Co-2 T"]
    assert [label for _, label in estimate_entries] == [
        "Co-1 E1",
        "Co-1 E2",
        "Co remote-1",
    ]

    visualizer.update(frame)

    assert (tmp_path / "pf_3d_step_0003.png").is_file()
    assert (tmp_path / "pf_3d_labeled_step_0003.png").is_file()
    assert visualizer.latest_pf_path.is_file()
    assert visualizer.latest_pf_labeled_path.is_file()
    assert "latest_pf_3d_labeled.png" in visualizer.index_path.read_text(
        encoding="utf-8"
    )
