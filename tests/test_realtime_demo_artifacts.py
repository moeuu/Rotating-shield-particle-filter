"""Tests for stable final visualization artifact publication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from realtime_demo import (
    _prepare_final_visualization_frame,
    _publish_final_cui_split_views,
)
from visualization.realtime_viz import PFFrame


def test_prepare_final_visualization_frame_preserves_travel_segment() -> None:
    """Final posterior projection must not connect station poses directly."""
    saved_segment = np.asarray(
        [
            [1.0, 1.0, 0.5],
            [1.0, 2.0, 0.5],
            [2.0, 2.0, 0.5],
        ],
        dtype=float,
    )
    frame = PFFrame(
        step_index=0,
        time=30.0,
        robot_position=np.asarray([2.0, 2.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.asarray([1.0, 0.0, 0.0], dtype=float),
        RPb=np.asarray([0.0, 1.0, 0.0], dtype=float),
        duration=30.0,
        particle_positions={"Cs-137": np.zeros((0, 3), dtype=float)},
        particle_weights={"Cs-137": np.zeros(0, dtype=float)},
        estimated_sources={"Cs-137": np.zeros((0, 3), dtype=float)},
        estimated_strengths={"Cs-137": np.zeros(0, dtype=float)},
        path_waypoints_xyz=saved_segment.copy(),
    )

    _prepare_final_visualization_frame(
        frame,
        step_index=7,
        elapsed_s=240.0,
        final_estimates={
            "Cs-137": (
                np.asarray([[4.0, 5.0, 0.5]], dtype=float),
                np.asarray([600_000.0], dtype=float),
            )
        },
    )

    np.testing.assert_array_equal(frame.path_waypoints_xyz, saved_segment)
    np.testing.assert_array_equal(
        frame.estimated_sources["Cs-137"],
        np.asarray([[4.0, 5.0, 0.5]], dtype=float),
    )
    assert frame.step_index == 7
    assert frame.time == 240.0


def test_publish_final_cui_split_views_copies_all_images(
    tmp_path: Path,
) -> None:
    """Completed CUI images must include plain and labeled PF results."""
    latest_robot = tmp_path / "cui" / "latest_robot_2d.png"
    latest_pf = tmp_path / "cui" / "latest_pf_3d.png"
    latest_pf_labeled = tmp_path / "cui" / "latest_pf_3d_labeled.png"
    latest_robot.parent.mkdir(parents=True)
    latest_robot.write_bytes(b"robot-png")
    latest_pf.write_bytes(b"pf-png")
    latest_pf_labeled.write_bytes(b"pf-labeled-png")
    final_robot = tmp_path / "results" / "result_robot_2d_case.png"
    final_pf = tmp_path / "results" / "result_pf_3d_case.png"
    final_pf_labeled = tmp_path / "results" / "result_pf_3d_labeled_case.png"

    _publish_final_cui_split_views(
        source_robot_path=latest_robot,
        source_pf_path=latest_pf,
        source_pf_labeled_path=latest_pf_labeled,
        final_robot_path=final_robot,
        final_pf_path=final_pf,
        final_pf_labeled_path=final_pf_labeled,
    )

    assert final_robot.read_bytes() == b"robot-png"
    assert final_pf.read_bytes() == b"pf-png"
    assert final_pf_labeled.read_bytes() == b"pf-labeled-png"


def test_publish_final_cui_split_views_rejects_missing_source(
    tmp_path: Path,
) -> None:
    """Publication must fail rather than emit an incomplete final image pair."""
    latest_robot = tmp_path / "latest_robot_2d.png"
    latest_robot.write_bytes(b"robot-png")

    with pytest.raises(RuntimeError, match="latest_pf_3d.png"):
        _publish_final_cui_split_views(
            source_robot_path=latest_robot,
            source_pf_path=tmp_path / "latest_pf_3d.png",
            source_pf_labeled_path=tmp_path / "latest_pf_3d_labeled.png",
            final_robot_path=tmp_path / "result_robot_2d.png",
            final_pf_path=tmp_path / "result_pf_3d.png",
            final_pf_labeled_path=tmp_path / "result_pf_3d_labeled.png",
        )

    assert not (tmp_path / "result_robot_2d.png").exists()
    assert not (tmp_path / "result_pf_3d.png").exists()
    assert not (tmp_path / "result_pf_3d_labeled.png").exists()
