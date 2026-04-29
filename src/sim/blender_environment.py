"""Helpers for generating random environment USD files with Blender."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from measurement.obstacles import ObstacleGrid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER_SCRIPT = ROOT / "scripts" / "generate_blender_environment.py"


def resolve_blender_executable(blender_executable: str | None = None) -> str:
    """Return the Blender executable path or raise a clear error."""
    candidate = blender_executable or os.environ.get("BLENDER") or "blender"
    if any(separator in candidate for separator in ("/", "\\")):
        path = Path(os.path.expandvars(candidate)).expanduser()
        if path.exists():
            return path.as_posix()
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            "Blender executable was not found. Install Blender, add it to PATH, "
            "set BLENDER=/path/to/blender, or pass --blender-executable /path/to/blender."
        )
    return resolved


def write_blender_environment_manifest(
    path: Path,
    *,
    grid: ObstacleGrid,
    room_size_xyz: tuple[float, float, float],
    obstacle_height_m: float,
    obstacle_material: str,
    base_usd_path: Path | None = None,
    traversability_output_path: Path | None = None,
    robot_radius_m: float = 0.35,
    traversability_reachable_from_xy: tuple[float, float] | None = None,
    traversability_blocking_z_range_m: tuple[float, float] = (0.05, 2.0),
) -> dict[str, Any]:
    """Write the Blender environment manifest consumed by the generator script."""
    payload: dict[str, Any] = {
        "room_size_xyz": [float(value) for value in room_size_xyz],
        "obstacle_origin_xy": [float(value) for value in grid.origin],
        "obstacle_cell_size_m": float(grid.cell_size),
        "obstacle_grid_shape": [int(value) for value in grid.grid_shape],
        "obstacle_cells": [list(cell) for cell in grid.blocked_cells],
        "obstacle_height_m": float(obstacle_height_m),
        "obstacle_material": str(obstacle_material),
    }
    if base_usd_path is not None:
        payload["base_usd_path"] = base_usd_path.expanduser().resolve().as_posix()
    if traversability_output_path is not None:
        payload["traversability_output_path"] = (
            traversability_output_path.expanduser().resolve().as_posix()
        )
        payload["traversability_robot_radius_m"] = float(robot_radius_m)
        payload["traversability_cell_size_m"] = float(grid.cell_size)
        payload["traversability_blocking_z_range_m"] = [
            float(traversability_blocking_z_range_m[0]),
            float(traversability_blocking_z_range_m[1]),
        ]
        if traversability_reachable_from_xy is not None:
            payload["traversability_reachable_from_xy"] = [
                float(traversability_reachable_from_xy[0]),
                float(traversability_reachable_from_xy[1]),
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def generate_blender_environment_usd(
    *,
    grid: ObstacleGrid,
    output_path: Path,
    room_size_xyz: tuple[float, float, float],
    obstacle_height_m: float = 2.0,
    obstacle_material: str = "concrete",
    base_usd_path: Path | None = None,
    traversability_output_path: Path | None = None,
    robot_radius_m: float = 0.35,
    traversability_reachable_from_xy: tuple[float, float] | None = None,
    traversability_blocking_z_range_m: tuple[float, float] = (0.05, 2.0),
    blender_executable: str | None = None,
    script_path: Path = DEFAULT_BLENDER_SCRIPT,
    timeout_s: float = 120.0,
) -> Path:
    """Generate a USD environment file by running Blender in background mode."""
    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    write_blender_environment_manifest(
        manifest_path,
        grid=grid,
        room_size_xyz=room_size_xyz,
        obstacle_height_m=obstacle_height_m,
        obstacle_material=obstacle_material,
        base_usd_path=base_usd_path,
        traversability_output_path=traversability_output_path,
        robot_radius_m=robot_radius_m,
        traversability_reachable_from_xy=traversability_reachable_from_xy,
        traversability_blocking_z_range_m=traversability_blocking_z_range_m,
    )
    executable = resolve_blender_executable(blender_executable)
    command = [
        executable,
        "--background",
        "--python",
        script_path.expanduser().resolve().as_posix(),
        "--",
        "--input",
        manifest_path.as_posix(),
        "--output",
        output_path.as_posix(),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = "\n".join(part for part in (stderr, stdout) if part)
        raise RuntimeError(f"Blender environment generation failed:\n{details}")
    if not output_path.exists():
        raise RuntimeError(f"Blender did not create the expected USD file: {output_path}")
    return output_path
