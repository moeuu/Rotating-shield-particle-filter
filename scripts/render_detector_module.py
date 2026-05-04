import math
from pathlib import Path

import bpy
from mathutils import Vector


RAW_PNG = Path("/tmp/ral_detector_raw.png")


def clear_scene():
    """Remove all default Blender objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name, color, roughness=0.45, alpha=1.0, metallic=0.0):
    """Create a material with optional alpha transparency."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Alpha"].default_value = alpha
    mat.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1
    mat.show_transparent_back = True
    return mat


def sph(r, theta, phi, signs=(1, 1, 1)):
    """Convert octant spherical coordinates to a Blender vector."""
    sx, sy, sz = signs
    return Vector(
        (
            sx * r * math.sin(theta) * math.cos(phi),
            sy * r * math.sin(theta) * math.sin(phi),
            sz * r * math.cos(theta),
        )
    )


def make_octant_shell(name, r_inner, r_outer, signs, material):
    """Create one exact spherical-octant shell around the detector."""
    n_theta = 28
    n_phi = 28
    theta0, theta1 = 0.0, math.pi / 2.0
    phi0, phi1 = 0.0, math.pi / 2.0
    vertices = []

    for r in (r_outer, r_inner):
        for i in range(n_theta + 1):
            theta = theta0 + (theta1 - theta0) * i / n_theta
            for j in range(n_phi + 1):
                phi = phi0 + (phi1 - phi0) * j / n_phi
                vertices.append(tuple(sph(r, theta, phi, signs)))

    def idx(layer, i, j):
        return layer * (n_theta + 1) * (n_phi + 1) + i * (n_phi + 1) + j

    faces = []

    for i in range(n_theta):
        for j in range(n_phi):
            faces.append(
                (
                    idx(0, i, j),
                    idx(0, i + 1, j),
                    idx(0, i + 1, j + 1),
                    idx(0, i, j + 1),
                )
            )
            faces.append(
                (
                    idx(1, i, j + 1),
                    idx(1, i + 1, j + 1),
                    idx(1, i + 1, j),
                    idx(1, i, j),
                )
            )

    for i in range(n_theta):
        faces.append(
            (idx(0, i, 0), idx(1, i, 0), idx(1, i + 1, 0), idx(0, i + 1, 0))
        )
        faces.append(
            (
                idx(0, i + 1, n_phi),
                idx(1, i + 1, n_phi),
                idx(1, i, n_phi),
                idx(0, i, n_phi),
            )
        )

    for j in range(n_phi):
        faces.append(
            (idx(0, 0, j + 1), idx(1, 0, j + 1), idx(1, 0, j), idx(0, 0, j))
        )
        faces.append(
            (
                idx(0, n_theta, j),
                idx(1, n_theta, j),
                idx(1, n_theta, j + 1),
                idx(0, n_theta, j + 1),
            )
        )

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj, (theta0, theta1, phi0, phi1, signs)


def add_poly_curve(name, points, material, bevel_depth=0.01):
    """Add a small beveled polyline curve."""
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    poly = curve.splines.new("POLY")
    poly.points.add(len(points) - 1)
    for p, co in zip(poly.points, points):
        p.co = (co.x, co.y, co.z, 1.0)
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def add_shell_edges(prefix, r_inner, r_outer, angles, material):
    """Draw boundary edges so the octant-shell shape remains legible in print."""
    theta0, theta1, phi0, phi1, signs = angles
    samples = 52
    for r in (r_outer, r_inner):
        for theta, tag in ((theta0, "theta0"), (theta1, "theta1")):
            points = [
                sph(r, theta, phi0 + (phi1 - phi0) * i / samples, signs)
                for i in range(samples + 1)
            ]
            add_poly_curve(f"{prefix}_{tag}_{r:.2f}", points, material)
        for phi, tag in ((phi0, "phi0"), (phi1, "phi1")):
            points = [
                sph(r, theta0 + (theta1 - theta0) * i / samples, phi, signs)
                for i in range(samples + 1)
            ]
            add_poly_curve(f"{prefix}_{tag}_{r:.2f}", points, material)

    axis_points = [
        (theta1, phi0),
        (theta1, phi1),
        (theta0, phi0),
    ]
    for theta, phi in axis_points:
        add_poly_curve(
            f"{prefix}_radial",
            [sph(r_inner, theta, phi, signs), sph(r_outer, theta, phi, signs)],
            material,
            bevel_depth=0.012,
        )


def look_at(obj, target):
    """Aim an object at a target point."""
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_scene():
    """Build the detector-shield render scene."""
    clear_scene()

    mat_detector = make_material(
        "CeBr3 cyan",
        (0.0, 0.86, 0.95, 0.33),
        roughness=0.12,
        alpha=0.33,
    )
    mat_fe = make_material(
        "Fe shield yellow",
        (0.95, 0.68, 0.08, 1.0),
        roughness=0.38,
    )
    mat_pb = make_material(
        "Pb shield light gray",
        (0.78, 0.82, 0.87, 0.86),
        roughness=0.32,
        alpha=0.86,
    )
    mat_edge = make_material(
        "dark outline",
        (0.03, 0.03, 0.03, 1.0),
        roughness=0.6,
    )

    detector_radius = 0.72
    fe_thickness = 0.30
    pb_thickness = 0.20
    fe_inner = detector_radius
    fe_outer = detector_radius + fe_thickness
    pb_inner = detector_radius + fe_thickness
    pb_outer = pb_inner + pb_thickness

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96,
        ring_count=48,
        radius=detector_radius,
        location=(0, 0, 0),
    )
    detector = bpy.context.object
    detector.name = "CeBr3 detector"
    detector.data.materials.append(mat_detector)
    bpy.ops.object.shade_smooth()

    _, fe_angles = make_octant_shell(
        "Fe spherical-octant shield",
        fe_inner,
        fe_outer,
        (-1, -1, -1),
        mat_fe,
    )
    _, pb_angles = make_octant_shell(
        "Pb spherical-octant shield",
        pb_inner,
        pb_outer,
        (1, 1, 1),
        mat_pb,
    )
    add_shell_edges("Fe_edges", fe_inner, fe_outer, fe_angles, mat_edge)
    add_shell_edges("Pb_edges", pb_inner, pb_outer, pb_angles, mat_edge)

    bpy.ops.object.light_add(type="AREA", location=(-3.0, -4.0, 5.0))
    key = bpy.context.object
    key.name = "large softbox"
    key.data.energy = 450
    key.data.size = 5.0
    bpy.ops.object.light_add(type="POINT", location=(3.0, 4.0, 3.0))
    fill = bpy.context.object
    fill.name = "fill light"
    fill.data.energy = 55

    bpy.ops.object.camera_add(location=(4.4, -6.0, 3.0))
    camera = bpy.context.object
    look_at(camera, (0.0, 0.0, -0.05))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 4.3
    bpy.context.scene.camera = camera

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 96
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.world.color = (1, 1, 1)
    bpy.context.scene.render.resolution_x = 1800
    bpy.context.scene.render.resolution_y = 1120
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(RAW_PNG)


if __name__ == "__main__":
    build_scene()
    bpy.ops.render.render(write_still=True)
