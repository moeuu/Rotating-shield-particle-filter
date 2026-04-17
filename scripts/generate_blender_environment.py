"""Generate a random environment USD file from a manifest using Blender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


def _build_parser() -> argparse.ArgumentParser:
    """Build the Blender-side argument parser."""
    parser = argparse.ArgumentParser(description="Generate a Blender-authored USD environment.")
    parser.add_argument("--input", required=True, help="Input environment manifest JSON.")
    parser.add_argument("--output", required=True, help="Output USD/USDA file path.")
    return parser


def _parse_blender_args() -> argparse.Namespace:
    """Parse arguments after Blender's -- separator."""
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return _build_parser().parse_args(script_args)


def _load_manifest(path: Path) -> dict:
    """Load the JSON manifest that defines the generated environment."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Environment manifest must be a JSON object.")
    return payload


def _clear_scene() -> None:
    """Remove all objects from the active Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    """Create a simple diffuse material."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _add_cube(
    name: str,
    *,
    size_xyz: tuple[float, float, float],
    center_xyz: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Add a cube with the requested dimensions and center."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_xyz)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size_xyz
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    obj["simbridge_material"] = material.name
    return obj


def _author_room(payload: dict, concrete: bpy.types.Material) -> None:
    """Create the room shell objects in Blender."""
    size_x, size_y, size_z = (float(value) for value in payload["room_size_xyz"])
    wall_height = min(3.0, size_z)
    wall_thickness = 0.1
    _add_cube(
        "Floor",
        size_xyz=(size_x, size_y, 0.1),
        center_xyz=(0.5 * size_x, 0.5 * size_y, -0.05),
        material=concrete,
    )
    _add_cube(
        "NorthWall",
        size_xyz=(size_x, wall_thickness, wall_height),
        center_xyz=(0.5 * size_x, size_y + 0.5 * wall_thickness, 0.5 * wall_height),
        material=concrete,
    )
    _add_cube(
        "SouthWall",
        size_xyz=(size_x, wall_thickness, wall_height),
        center_xyz=(0.5 * size_x, -0.5 * wall_thickness, 0.5 * wall_height),
        material=concrete,
    )
    _add_cube(
        "EastWall",
        size_xyz=(wall_thickness, size_y, wall_height),
        center_xyz=(size_x + 0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
        material=concrete,
    )
    _add_cube(
        "WestWall",
        size_xyz=(wall_thickness, size_y, wall_height),
        center_xyz=(-0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
        material=concrete,
    )


def _author_obstacles(payload: dict, concrete: bpy.types.Material) -> None:
    """Create obstacle box objects in Blender from blocked grid cells."""
    origin_x, origin_y = (float(value) for value in payload["obstacle_origin_xy"])
    cell_size = float(payload["obstacle_cell_size_m"])
    height = float(payload.get("obstacle_height_m", 2.0))
    for index, cell in enumerate(payload.get("obstacle_cells", [])):
        ix, iy = (int(value) for value in cell)
        x0 = origin_x + float(ix) * cell_size
        y0 = origin_y + float(iy) * cell_size
        _add_cube(
            f"Obstacle_{index:04d}",
            size_xyz=(cell_size, cell_size, height),
            center_xyz=(x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, 0.5 * height),
            material=concrete,
        )


def _export_usd(output_path: Path) -> None:
    """Export the Blender scene to USD under /World/Environment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
            root_prim_path="/World/Environment",
        )
    except TypeError:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
        )


def main() -> None:
    """Generate the Blender scene and export it to USD."""
    args = _parse_blender_args()
    payload = _load_manifest(Path(args.input).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()
    _clear_scene()
    concrete = _material("ConcreteMaterial", (0.55, 0.58, 0.60, 1.0))
    _author_room(payload, concrete)
    _author_obstacles(payload, concrete)
    _export_usd(output_path)


if __name__ == "__main__":
    main()
