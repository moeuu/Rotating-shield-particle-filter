"""Tests for RA-L supplementary video generation helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from PIL import Image
import pytest

from scripts import build_ral_supplementary_video as video


def _write_test_image(path: Path, color: tuple[int, int, int]) -> Path:
    """Write a deterministic RGB test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 90), color).save(path)
    return path


def test_write_storyboard_frames_creates_numbered_pngs(tmp_path: Path) -> None:
    """Storyboard rendering should produce fixed-resolution numbered frames."""
    image_path = _write_test_image(tmp_path / "input.png", (20, 70, 120))
    segment = video.VideoSegment(
        title="Test segment",
        caption="This caption describes the generated frame sequence.",
        image_path=image_path,
    )

    frames = video.write_storyboard_frames(
        [segment],
        tmp_path / "frames",
        fps=2,
        seconds_per_segment=1.5,
        resolution=(320, 180),
    )

    assert [path.name for path in frames] == [
        "frame_00000.png",
        "frame_00001.png",
        "frame_00002.png",
    ]
    with Image.open(frames[0]) as frame:
        assert frame.size == (320, 180)


def test_build_ffmpeg_command_uses_h264_mp4(tmp_path: Path) -> None:
    """The encoder command should target H.264 MP4 output."""
    command = video.build_ffmpeg_command(
        tmp_path / "frames",
        tmp_path / "out.mp4",
        fps=12,
    )

    assert command[0] == "ffmpeg"
    assert "-framerate" in command
    assert "12" in command
    assert "libx264" in command
    assert str(tmp_path / "out.mp4") == command[-1]


def test_encode_mp4_invokes_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """MP4 encoding should call ffmpeg with the generated frame pattern."""
    frames_dir = tmp_path / "frames"
    _write_test_image(frames_dir / "frame_00000.png", (120, 70, 20))
    output_path = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        """Capture the ffmpeg command without running an external process."""
        calls.append(command)
        assert check is True
        output_path.write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.subprocess, "run", fake_run)

    encoded = video.encode_mp4(frames_dir, output_path, fps=6)

    assert encoded == output_path
    assert output_path.read_bytes() == b"fake mp4"
    assert calls
    assert str(frames_dir / "frame_%05d.png") in calls[0]
