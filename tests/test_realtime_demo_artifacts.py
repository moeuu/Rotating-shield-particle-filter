"""Tests for stable final visualization artifact publication."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from visualization.artifacts import (
    prepare_final_visualization_frame,
    publish_final_cui_split_views,
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

    final_frame = prepare_final_visualization_frame(
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

    assert final_frame is not frame
    np.testing.assert_array_equal(final_frame.path_waypoints_xyz, saved_segment)
    np.testing.assert_array_equal(
        final_frame.estimated_sources["Cs-137"],
        np.asarray([[4.0, 5.0, 0.5]], dtype=float),
    )
    assert final_frame.step_index == 7
    assert final_frame.time == 240.0
    assert frame.step_index == 0
    assert frame.time == 30.0


def test_publish_final_cui_split_views_copies_all_images(
    tmp_path: Path,
) -> None:
    """Completed CUI artifacts must include all five browser panels."""
    latest_overview = tmp_path / "cui" / "latest_experiment_overview.png"
    latest_robot = tmp_path / "cui" / "latest_robot_2d.png"
    latest_pf = tmp_path / "cui" / "latest_pf_3d.png"
    latest_pf_labeled = tmp_path / "cui" / "latest_pf_3d_labeled.png"
    latest_spectrum = tmp_path / "cui" / "latest_spectrum.png"
    latest_robot.parent.mkdir(parents=True)
    latest_overview.write_bytes(b"overview-png")
    latest_robot.write_bytes(b"robot-png")
    latest_pf.write_bytes(b"pf-png")
    latest_pf_labeled.write_bytes(b"pf-labeled-png")
    latest_spectrum.write_bytes(b"spectrum-png")
    final_overview = tmp_path / "results" / "result_overview_case.png"
    final_robot = tmp_path / "results" / "result_robot_2d_case.png"
    final_pf = tmp_path / "results" / "result_pf_3d_case.png"
    final_pf_labeled = tmp_path / "results" / "result_pf_3d_labeled_case.png"
    final_spectrum = tmp_path / "results" / "result_spectrum_case.png"

    publish_final_cui_split_views(
        source_overview_path=latest_overview,
        source_robot_path=latest_robot,
        source_pf_path=latest_pf,
        source_pf_labeled_path=latest_pf_labeled,
        source_spectrum_path=latest_spectrum,
        final_overview_path=final_overview,
        final_robot_path=final_robot,
        final_pf_path=final_pf,
        final_pf_labeled_path=final_pf_labeled,
        final_spectrum_path=final_spectrum,
    )

    assert final_overview.read_bytes() == b"overview-png"
    assert final_robot.read_bytes() == b"robot-png"
    assert final_pf.read_bytes() == b"pf-png"
    assert final_pf_labeled.read_bytes() == b"pf-labeled-png"
    assert final_spectrum.read_bytes() == b"spectrum-png"


def test_publish_final_cui_split_views_keeps_three_view_compatibility(
    tmp_path: Path,
) -> None:
    """Legacy callers may continue publishing the original three views."""
    sources = tuple(tmp_path / "cui" / f"source-{index}.png" for index in range(3))
    targets = tuple(tmp_path / "results" / f"target-{index}.png" for index in range(3))
    for index, source in enumerate(sources):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{index}".encode())

    publish_final_cui_split_views(
        source_robot_path=sources[0],
        source_pf_path=sources[1],
        source_pf_labeled_path=sources[2],
        final_robot_path=targets[0],
        final_pf_path=targets[1],
        final_pf_labeled_path=targets[2],
    )

    assert [target.read_bytes() for target in targets] == [
        b"source-0",
        b"source-1",
        b"source-2",
    ]


def test_publish_final_cui_split_views_rejects_missing_source(
    tmp_path: Path,
) -> None:
    """Publication must fail rather than emit an incomplete final image pair."""
    latest_robot = tmp_path / "latest_robot_2d.png"
    latest_robot.write_bytes(b"robot-png")

    with pytest.raises(RuntimeError, match="latest_pf_3d.png"):
        publish_final_cui_split_views(
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


def test_publish_final_cui_split_views_stages_every_copy_before_replacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy failure must preserve every previously published final image."""
    sources = tuple(tmp_path / "cui" / f"source-{index}.png" for index in range(3))
    targets = tuple(tmp_path / "results" / f"target-{index}.png" for index in range(3))
    for index, path in enumerate(sources):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"new-{index}".encode())
    for index, path in enumerate(targets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"old-{index}".encode())

    real_copyfile = shutil.copyfile
    copy_count = 0

    def fail_on_second_copy(source: Path, target: Path) -> str:
        """Raise on the second copy after allowing the first stage file."""
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("synthetic copy failure")
        return real_copyfile(source, target)

    monkeypatch.setattr("visualization.artifacts.shutil.copyfile", fail_on_second_copy)

    with pytest.raises(OSError, match="synthetic copy failure"):
        publish_final_cui_split_views(
            source_robot_path=sources[0],
            source_pf_path=sources[1],
            source_pf_labeled_path=sources[2],
            final_robot_path=targets[0],
            final_pf_path=targets[1],
            final_pf_labeled_path=targets[2],
        )

    assert [path.read_bytes() for path in targets] == [
        b"old-0",
        b"old-1",
        b"old-2",
    ]
    assert not tuple((tmp_path / "results").glob(".*.tmp-*"))
