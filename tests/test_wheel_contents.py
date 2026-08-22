"""Tests for the installed wheel's public CLI module closure."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_cli_runtime_helpers_are_in_wheel(tmp_path: Path) -> None:
    """The live console entry point must not import an omitted top-level module."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    wheels = tuple(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode("utf-8")

    assert "pf/closed_loop.py" in names
    assert "pf/cui_runtime.py" in names
    assert "rotating-shield-pf-live = pf.closed_loop:main" in entry_points
