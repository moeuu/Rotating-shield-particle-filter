"""Tests for Blender-driven environment generation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from measurement.obstacles import ObstacleGrid
from sim.blender_environment import generate_blender_environment_usd


def test_generate_blender_environment_usd_invokes_blender(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The helper should write a manifest and call Blender in background mode."""
    commands: list[list[str]] = []

    def _fake_which(name: str) -> str:
        """Resolve the fake Blender executable."""
        assert name == "blender"
        return "/usr/bin/blender"

    def _fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Record the command and create the expected output USD file."""
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("sim.blender_environment.shutil.which", _fake_which)
    monkeypatch.setattr("sim.blender_environment.subprocess.run", _fake_run)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 1),),
    )
    output_path = tmp_path / "generated.usda"
    base_usd_path = tmp_path / "manchester_base.usda"
    base_usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    result = generate_blender_environment_usd(
        grid=grid,
        output_path=output_path,
        room_size_xyz=(2.0, 2.0, 3.0),
        base_usd_path=base_usd_path,
    )

    assert result == output_path.resolve()
    assert commands
    assert commands[0][:3] == ["/usr/bin/blender", "--background", "--python"]
    manifest_path = output_path.with_suffix(".manifest.json")
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert '"obstacle_cells":' in manifest_text
    assert base_usd_path.as_posix() in manifest_text


def test_blender_generator_uses_semantic_wall_group() -> None:
    """Generated USD environments should group room boundaries under Wall."""
    root = Path(__file__).resolve().parents[1]
    generator_source = (root / "scripts" / "generate_blender_environment.py").read_text(
        encoding="utf-8"
    )

    assert '_ensure_empty("Wall", transport_group="wall")' in generator_source
    assert 'parent=wall_group' in generator_source
    assert 'transport_group="wall"' in generator_source
    assert '_ensure_empty("Obstacles", transport_group="obstacle")' in generator_source
