"""Tests for truth-safe CUI evaluation rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.realtime_viz import (
    CUISplitPFVisualizer,
    PFFrame,
    RealTimePFVisualizer,
    _shield_material_normal,
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


def test_realtime_visualizer_accepts_runtime_shield_normal_vectors(
    tmp_path: Path,
) -> None:
    """A runtime octant normal must render without a matrix-shape failure."""
    incoming_fe = np.asarray([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    incoming_pb = np.asarray([-1.0, 1.0, -1.0], dtype=float) / np.sqrt(3.0)
    frame = PFFrame(
        step_index=0,
        time=30.0,
        robot_position=np.asarray([1.0, 1.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=incoming_fe,
        RPb=incoming_pb,
        duration=30.0,
        particle_positions={"Cs-137": np.zeros((0, 3), dtype=float)},
        particle_weights={"Cs-137": np.zeros(0, dtype=float)},
        estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
        estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
    )
    visualizer = RealTimePFVisualizer(isotopes=["Cs-137"])
    output_path = tmp_path / "runtime_normal_frame.png"
    try:
        visualizer.update(frame)
        visualizer.save_final(output_path.as_posix())
    finally:
        plt.close(visualizer.fig)

    assert output_path.is_file()
    np.testing.assert_allclose(
        _shield_material_normal(incoming_fe),
        -incoming_fe,
    )


def test_legacy_shield_rotation_uses_positive_octant_centre() -> None:
    """Legacy matrices must use the local octant centre instead of local Z."""
    np.testing.assert_allclose(
        _shield_material_normal(np.eye(3, dtype=float)),
        np.ones(3, dtype=float) / np.sqrt(3.0),
    )
