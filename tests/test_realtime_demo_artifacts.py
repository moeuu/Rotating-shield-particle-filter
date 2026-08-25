"""Tests for stable final visualization artifact publication."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from visualization.artifacts import publish_final_cui_split_views


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


def test_publish_final_cui_split_views_rejects_missing_source(
    tmp_path: Path,
) -> None:
    """Publication must fail rather than emit an incomplete final image pair."""
    latest_robot = tmp_path / "latest_robot_2d.png"
    latest_robot.write_bytes(b"robot-png")

    with pytest.raises(RuntimeError, match="latest_pf_3d.png"):
        publish_final_cui_split_views(
            source_overview_path=tmp_path / "latest_experiment_overview.png",
            source_robot_path=latest_robot,
            source_pf_path=tmp_path / "latest_pf_3d.png",
            source_pf_labeled_path=tmp_path / "latest_pf_3d_labeled.png",
            source_spectrum_path=tmp_path / "latest_spectrum.png",
            final_overview_path=tmp_path / "result_overview.png",
            final_robot_path=tmp_path / "result_robot_2d.png",
            final_pf_path=tmp_path / "result_pf_3d.png",
            final_pf_labeled_path=tmp_path / "result_pf_3d_labeled.png",
            final_spectrum_path=tmp_path / "result_spectrum.png",
        )

    assert not (tmp_path / "result_robot_2d.png").exists()
    assert not (tmp_path / "result_pf_3d.png").exists()
    assert not (tmp_path / "result_pf_3d_labeled.png").exists()


def test_publish_final_cui_split_views_stages_every_copy_before_replacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy failure must preserve every previously published final image."""
    sources = tuple(tmp_path / "cui" / f"source-{index}.png" for index in range(5))
    targets = tuple(tmp_path / "results" / f"target-{index}.png" for index in range(5))
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
            source_overview_path=sources[0],
            source_robot_path=sources[1],
            source_pf_path=sources[2],
            source_pf_labeled_path=sources[3],
            source_spectrum_path=sources[4],
            final_overview_path=targets[0],
            final_robot_path=targets[1],
            final_pf_path=targets[2],
            final_pf_labeled_path=targets[3],
            final_spectrum_path=targets[4],
        )

    assert [path.read_bytes() for path in targets] == [
        b"old-0",
        b"old-1",
        b"old-2",
        b"old-3",
        b"old-4",
    ]
    assert not tuple((tmp_path / "results").glob(".*.tmp-*"))
