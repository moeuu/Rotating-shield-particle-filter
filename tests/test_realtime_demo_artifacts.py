"""Tests for stable final visualization artifact publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from realtime_demo import _publish_final_cui_split_views


def test_publish_final_cui_split_views_copies_both_images(
    tmp_path: Path,
) -> None:
    """Completed CUI images must be copied to stable final result paths."""
    latest_robot = tmp_path / "cui" / "latest_robot_2d.png"
    latest_pf = tmp_path / "cui" / "latest_pf_3d.png"
    latest_robot.parent.mkdir(parents=True)
    latest_robot.write_bytes(b"robot-png")
    latest_pf.write_bytes(b"pf-png")
    final_robot = tmp_path / "results" / "result_robot_2d_case.png"
    final_pf = tmp_path / "results" / "result_pf_3d_case.png"

    _publish_final_cui_split_views(
        source_robot_path=latest_robot,
        source_pf_path=latest_pf,
        final_robot_path=final_robot,
        final_pf_path=final_pf,
    )

    assert final_robot.read_bytes() == b"robot-png"
    assert final_pf.read_bytes() == b"pf-png"


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
            final_robot_path=tmp_path / "result_robot_2d.png",
            final_pf_path=tmp_path / "result_pf_3d.png",
        )

    assert not (tmp_path / "result_robot_2d.png").exists()
    assert not (tmp_path / "result_pf_3d.png").exists()
