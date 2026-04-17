"""Convert Manchester Gazebo SDF assets into a USD scene using Blender."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Vector


@dataclass
class ConversionContext:
    """Carry SDF conversion settings and model search roots."""

    model_roots: tuple[Path, ...]
    default_material_name: str
    root_prim_path: str
    missing_assets: list[str] = field(default_factory=list)


def _build_parser() -> argparse.ArgumentParser:
    """Build the Blender-side argument parser."""
    parser = argparse.ArgumentParser(description="Convert a Manchester SDF asset to USD.")
    parser.add_argument("--input", required=True, help="Input SDF file or model directory.")
    parser.add_argument("--output", required=True, help="Output USD/USDA file path.")
    parser.add_argument(
        "--model-root",
        action="append",
        default=[],
        help="Directory used to resolve model:// URIs; can be passed multiple times.",
    )
    parser.add_argument(
        "--root-prim-path",
        default="/World/Environment",
        help="USD root prim path used for exported static environment geometry.",
    )
    parser.add_argument(
        "--default-material",
        default="ConcreteMaterial",
        help="Fallback material assigned to SDF primitives without imported materials.",
    )
    return parser


def _parse_blender_args() -> argparse.Namespace:
    """Parse arguments after Blender's -- separator."""
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return _build_parser().parse_args(script_args)


def _clear_scene() -> None:
    """Remove all objects from the active Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _sanitize_name(value: str) -> str:
    """Return a Blender-safe object name token."""
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_")
    return token or "Object"


def _local_name(tag: str) -> str:
    """Return an XML tag name without its namespace."""
    return str(tag).rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    """Return direct child elements with the requested local name."""
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    """Return the first direct child with the requested local name."""
    matches = _children(element, name)
    return matches[0] if matches else None


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    """Return the first descendant with the requested local name."""
    for child in element.iter():
        if child is not element and _local_name(child.tag) == name:
            return child
    return None


def _text(element: ET.Element | None, default: str = "") -> str:
    """Return stripped XML element text with a default fallback."""
    if element is None or element.text is None:
        return default
    return element.text.strip()


def _float_values(text: str, expected: int, default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse a whitespace-separated float vector."""
    if not text:
        return default
    values = tuple(float(part) for part in text.split())
    if len(values) < expected:
        values = values + default[len(values) :]
    return values[:expected]


def _pose_matrix(pose_text: str) -> Matrix:
    """Convert an SDF pose string into a Blender transform matrix."""
    x, y, z, roll, pitch, yaw = _float_values(
        pose_text,
        6,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    translation = Matrix.Translation(Vector((x, y, z)))
    rotation = Euler((roll, pitch, yaw), "XYZ").to_matrix().to_4x4()
    return translation @ rotation


def _scale_matrix(scale_xyz: tuple[float, float, float]) -> Matrix:
    """Return a 4x4 scale matrix."""
    matrix = Matrix.Identity(4)
    matrix[0][0] = float(scale_xyz[0])
    matrix[1][1] = float(scale_xyz[1])
    matrix[2][2] = float(scale_xyz[2])
    return matrix


def _material(name: str) -> bpy.types.Material:
    """Return a simple diffuse fallback material."""
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.diffuse_color = (0.55, 0.58, 0.60, 1.0)
    return material


def _assign_default_material(obj: bpy.types.Object, material_name: str) -> None:
    """Assign a fallback material to mesh objects that have no material."""
    if obj.type != "MESH":
        return
    if not obj.data.materials:
        obj.data.materials.append(_material(material_name))
    obj["simbridge_material"] = material_name


def _input_sdf_path(input_path: Path) -> Path:
    """Resolve a CLI input path into an SDF file path."""
    if input_path.is_file():
        return input_path
    model_sdf = input_path / "model.sdf"
    if model_sdf.exists():
        return model_sdf
    sdf_paths = sorted(input_path.rglob("*.sdf"))
    if not sdf_paths:
        raise FileNotFoundError(f"No SDF file found under {input_path}")
    return sdf_paths[0]


def _model_search_roots(input_sdf: Path, model_roots: list[str]) -> tuple[Path, ...]:
    """Build an ordered set of directories used for model URI resolution."""
    roots: list[Path] = []
    for value in model_roots:
        roots.append(Path(value).expanduser().resolve())
    roots.extend([input_sdf.parent, input_sdf.parent.parent])
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def _resolve_model_uri(uri: str, context: ConversionContext, current_sdf: Path) -> Path | None:
    """Resolve a Gazebo model URI or relative mesh path."""
    cleaned = uri.strip()
    if not cleaned:
        return None
    if cleaned.startswith("file://"):
        file_path = Path(cleaned.removeprefix("file://")).expanduser()
        return file_path.resolve() if file_path.exists() else None
    if cleaned.startswith("model://"):
        relative = Path(cleaned.removeprefix("model://").lstrip("/"))
        parts = relative.parts
        for root in context.model_roots:
            candidate = (root / relative).resolve()
            if candidate.exists():
                return candidate
            if parts and root.name == parts[0]:
                candidate = (root / Path(*parts[1:])).resolve() if len(parts) > 1 else root
                if candidate.exists():
                    return candidate
        return None
    relative_path = (current_sdf.parent / cleaned).resolve()
    return relative_path if relative_path.exists() else None


def _resolve_included_sdf(uri: str, context: ConversionContext, current_sdf: Path) -> Path | None:
    """Resolve an SDF include URI into a model.sdf path."""
    resolved = _resolve_model_uri(uri, context, current_sdf)
    if resolved is None:
        return None
    if resolved.is_dir():
        model_sdf = resolved / "model.sdf"
        return model_sdf if model_sdf.exists() else None
    return resolved if resolved.suffix.lower() == ".sdf" else None


def _import_mesh_file(mesh_path: Path) -> list[bpy.types.Object]:
    """Import a mesh file and return the newly created Blender objects."""
    before = set(bpy.data.objects)
    suffix = mesh_path.suffix.lower()
    if suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=mesh_path.as_posix())
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path.as_posix())
    elif suffix == ".obj":
        _import_obj(mesh_path)
    elif suffix == ".stl":
        _import_stl(mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
    return [obj for obj in bpy.data.objects if obj not in before]


def _import_obj(mesh_path: Path) -> None:
    """Import an OBJ file across supported Blender versions."""
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=mesh_path.as_posix())
        return
    bpy.ops.import_scene.obj(filepath=mesh_path.as_posix())


def _import_stl(mesh_path: Path) -> None:
    """Import an STL file across supported Blender versions."""
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=mesh_path.as_posix())
        return
    bpy.ops.import_mesh.stl(filepath=mesh_path.as_posix())


def _parent_imported_objects(
    objects: list[bpy.types.Object],
    *,
    name: str,
    matrix_world: Matrix,
    material_name: str,
) -> None:
    """Parent imported root objects under a transform empty."""
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    empty.matrix_world = matrix_world
    imported = set(objects)
    for obj in objects:
        _assign_default_material(obj, material_name)
        if obj.parent not in imported:
            obj.parent = empty


def _author_mesh_visual(
    visual: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    current_sdf: Path,
    name_prefix: str,
) -> None:
    """Import one SDF mesh visual."""
    mesh = _first_descendant(visual, "mesh")
    if mesh is None:
        return
    uri = _text(_first_child(mesh, "uri"))
    mesh_path = _resolve_model_uri(uri, context, current_sdf)
    if mesh_path is None:
        context.missing_assets.append(uri)
        return
    scale = _float_values(_text(_first_child(mesh, "scale")), 3, (1.0, 1.0, 1.0))
    objects = _import_mesh_file(mesh_path)
    visual_name = _sanitize_name(str(visual.get("name") or mesh_path.stem))
    _parent_imported_objects(
        objects,
        name=f"{name_prefix}_{visual_name}",
        matrix_world=transform @ _scale_matrix((scale[0], scale[1], scale[2])),
        material_name=context.default_material_name,
    )


def _author_box_visual(
    visual: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    name_prefix: str,
) -> None:
    """Author one SDF box visual as a Blender cube."""
    size_text = _text(_first_descendant(visual, "size"), "1 1 1")
    size = _float_values(size_text, 3, (1.0, 1.0, 1.0))
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.name = f"{name_prefix}_{_sanitize_name(str(visual.get('name') or 'Box'))}"
    obj.matrix_world = transform @ _scale_matrix((size[0], size[1], size[2]))
    _assign_default_material(obj, context.default_material_name)


def _author_cylinder_visual(
    visual: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    name_prefix: str,
) -> None:
    """Author one SDF cylinder visual as a Blender cylinder."""
    cylinder = _first_descendant(visual, "cylinder")
    if cylinder is None:
        return
    radius = float(_text(_first_child(cylinder, "radius"), "0.5"))
    length = float(_text(_first_child(cylinder, "length"), "1.0"))
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=1.0)
    obj = bpy.context.object
    obj.name = f"{name_prefix}_{_sanitize_name(str(visual.get('name') or 'Cylinder'))}"
    obj.matrix_world = transform @ _scale_matrix((radius, radius, length))
    _assign_default_material(obj, context.default_material_name)


def _author_sphere_visual(
    visual: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    name_prefix: str,
) -> None:
    """Author one SDF sphere visual as a Blender UV sphere."""
    sphere = _first_descendant(visual, "sphere")
    if sphere is None:
        return
    radius = float(_text(_first_child(sphere, "radius"), "0.5"))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.object
    obj.name = f"{name_prefix}_{_sanitize_name(str(visual.get('name') or 'Sphere'))}"
    obj.matrix_world = transform @ _scale_matrix((radius, radius, radius))
    _assign_default_material(obj, context.default_material_name)


def _author_visual(
    visual: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    current_sdf: Path,
    name_prefix: str,
) -> None:
    """Author one SDF visual element."""
    geometry = _first_child(visual, "geometry")
    if geometry is None:
        return
    visual_transform = transform @ _pose_matrix(_text(_first_child(visual, "pose")))
    if _first_descendant(geometry, "mesh") is not None:
        _author_mesh_visual(visual, visual_transform, context, current_sdf, name_prefix)
        return
    if _first_descendant(geometry, "box") is not None:
        _author_box_visual(visual, visual_transform, context, name_prefix)
        return
    if _first_descendant(geometry, "cylinder") is not None:
        _author_cylinder_visual(visual, visual_transform, context, name_prefix)
        return
    if _first_descendant(geometry, "sphere") is not None:
        _author_sphere_visual(visual, visual_transform, context, name_prefix)


def _parse_link(
    link: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    current_sdf: Path,
    model_name: str,
) -> None:
    """Parse an SDF link and author its visual geometry."""
    link_name = _sanitize_name(str(link.get("name") or "Link"))
    link_transform = transform @ _pose_matrix(_text(_first_child(link, "pose")))
    for visual in _children(link, "visual"):
        _author_visual(
            visual,
            link_transform,
            context,
            current_sdf,
            f"{model_name}_{link_name}",
        )


def _parse_model(
    model: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    current_sdf: Path,
    stack: tuple[Path, ...],
) -> None:
    """Parse an SDF model element and its nested includes."""
    model_name = _sanitize_name(str(model.get("name") or current_sdf.parent.name))
    model_transform = transform @ _pose_matrix(_text(_first_child(model, "pose")))
    for include in _children(model, "include"):
        _parse_include(include, model_transform, context, current_sdf, stack)
    for link in _children(model, "link"):
        _parse_link(link, model_transform, context, current_sdf, model_name)


def _parse_include(
    include: ET.Element,
    transform: Matrix,
    context: ConversionContext,
    current_sdf: Path,
    stack: tuple[Path, ...],
) -> None:
    """Parse an SDF include element by loading the referenced model.sdf."""
    uri = _text(_first_child(include, "uri"))
    included_sdf = _resolve_included_sdf(uri, context, current_sdf)
    if included_sdf is None:
        context.missing_assets.append(uri)
        return
    include_transform = transform @ _pose_matrix(_text(_first_child(include, "pose")))
    _parse_sdf_file(included_sdf, include_transform, context, stack)


def _parse_sdf_file(
    sdf_path: Path,
    transform: Matrix,
    context: ConversionContext,
    stack: tuple[Path, ...] = (),
) -> None:
    """Parse one SDF file and author its visual geometry."""
    sdf_path = sdf_path.expanduser().resolve()
    if sdf_path in stack:
        return
    root = ET.parse(sdf_path).getroot()
    next_stack = stack + (sdf_path,)
    containers = _children(root, "world") or [root]
    for container in containers:
        for include in _children(container, "include"):
            _parse_include(include, transform, context, sdf_path, next_stack)
        for model in _children(container, "model"):
            _parse_model(model, transform, context, sdf_path, next_stack)


def _export_usd(output_path: Path, root_prim_path: str) -> None:
    """Export the Blender scene to USD."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
            root_prim_path=root_prim_path,
        )
    except TypeError:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
        )


def main() -> None:
    """Run the SDF to USD conversion."""
    args = _parse_blender_args()
    input_path = Path(args.input).expanduser().resolve()
    input_sdf = _input_sdf_path(input_path)
    context = ConversionContext(
        model_roots=_model_search_roots(input_sdf, args.model_root),
        default_material_name=str(args.default_material),
        root_prim_path=str(args.root_prim_path),
    )
    _clear_scene()
    _material(context.default_material_name)
    _parse_sdf_file(input_sdf, Matrix.Identity(4), context)
    if context.missing_assets:
        missing = "\n".join(sorted(set(context.missing_assets)))
        raise FileNotFoundError(f"Could not resolve referenced SDF assets:\n{missing}")
    _export_usd(Path(args.output).expanduser().resolve(), context.root_prim_path)


if __name__ == "__main__":
    main()
