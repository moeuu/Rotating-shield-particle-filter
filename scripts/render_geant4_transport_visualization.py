"""Render Isaac Sim figures with Geant4-style photon-track overlays.

The script creates two visual-only manuscript assets: direct source-to-detector
transport and obstacle attenuation/scatter transport. The scene geometry is
authored in Isaac Sim, while the photon tracks are illustrative Geant4-style
visual primitives. Runtime transport, PF observations, and Geant4 settings are
not modified by this script.
"""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_ROOT = ROOT / "results" / "ral_isaac_figures"
SERIF_FONT_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.isaacsim_app.app import IsaacSimApplication  # noqa: E402
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription  # noqa: E402
from sim.protocol import SimulationCommand  # noqa: E402


SOURCE = (1.2, 3.3, 0.85)
ROBOT_XY = (7.1, 3.25)
DETECTOR = (7.1, 3.25, 0.72)
OBSTACLE_CENTER = (4.05, 3.28, 0.95)
OBSTACLE_SIZE = (0.92, 1.85, 1.9)
ROOM_SIZE_XYZ = (8.4, 6.8, 3.2)

NORMAL_RAY_COLOR = (0.0, 0.95, 0.16)
DETECTED_RAY_COLOR = (1.0, 0.84, 0.05)
ATTENUATED_RAY_COLOR = (0.01, 0.01, 0.01)
SCATTERED_RAY_COLOR = (0.0, 0.9, 1.0)


def _app_config() -> dict[str, object]:
    """Return an Isaac Sim configuration tuned for transport visualization."""
    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "detector_height_m": DETECTOR[2],
        "obstacle_height_m": OBSTACLE_SIZE[2],
        "robot_animation_time_scale": 0.0,
        "lighting": {
            "dome_intensity": 1100.0,
            "color_rgb": [1.0, 0.99, 0.96],
            "interior_lights": [
                {
                    "position_xyz": [2.2, 1.1, 3.6],
                    "intensity": 70000.0,
                    "radius_m": 0.05,
                },
                {
                    "position_xyz": [6.2, 5.6, 3.6],
                    "intensity": 90000.0,
                    "radius_m": 0.05,
                },
            ],
        },
        "stage_visual_rules": [
            {
                "path_prefix": "/World/Environment/Wall/Floor",
                "color_rgb": [0.60, 0.62, 0.62],
                "opacity": 1.0,
                "roughness": 0.75,
            },
            {
                "path_prefix": "/World/Environment/Wall",
                "color_rgb": [0.66, 0.70, 0.72],
                "opacity": 0.16,
                "roughness": 0.85,
            },
            {
                "path_prefix": "/World/SimBridge/Sources",
                "color_rgb": [1.0, 0.02, 0.00],
                "opacity": 1.0,
                "roughness": 0.22,
                "emissive_scale": 8.0,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Body",
                "color_rgb": [0.24, 0.29, 0.34],
                "opacity": 1.0,
                "roughness": 0.5,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Detector",
                "color_rgb": [0.0, 0.86, 1.0],
                "opacity": 1.0,
                "roughness": 0.25,
                "emissive_scale": 3.5,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/FeShield",
                "color_rgb": [0.96, 0.58, 0.08],
                "opacity": 1.0,
                "roughness": 0.45,
                "emissive_scale": 0.55,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/PbShield",
                "color_rgb": [0.62, 0.66, 0.76],
                "opacity": 1.0,
                "roughness": 0.45,
                "emissive_scale": 0.35,
            },
            {
                "path_prefix": "/World/SimBridge/Transport/ConcreteSlab",
                "color_rgb": [0.42, 0.45, 0.47],
                "opacity": 0.68,
                "roughness": 0.86,
            },
        ],
        "stage_material_rules": [
            {"path_prefix": "/World/Environment", "material": "concrete"},
            {"path_prefix": "/World/SimBridge/Transport/ConcreteSlab", "material": "concrete"},
        ],
    }


def _scene_description() -> SceneDescription:
    """Create the compact source-detector scene used for both captures."""
    return SceneDescription(
        room_size_xyz=ROOM_SIZE_XYZ,
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(0, 0),
        obstacle_material="concrete",
        obstacle_cells=[],
        author_obstacle_prims=False,
        author_room_boundary_prims=True,
        sources=[SourceDescription("Cs-137", SOURCE, 30000.0)],
        usd_path=None,
        use_config_usd_fallback=False,
    )


def _command(step_id: int) -> SimulationCommand:
    """Create a still robot command for the visualization scene."""
    return SimulationCommand(
        step_id=step_id,
        target_pose_xyz=(ROBOT_XY[0], ROBOT_XY[1], 0.0),
        target_base_yaw_rad=math.pi,
        fe_orientation_index=0,
        pb_orientation_index=6,
        dwell_time_s=30.0,
    )


def _backend(app: IsaacSimApplication):
    """Return the real Isaac Sim stage backend from the application."""
    backend = app._stage_backend  # noqa: SLF001
    if backend is None:
        raise RuntimeError("Isaac Sim backend is not available.")
    return backend


def _pump(app: IsaacSimApplication, frames: int = 24) -> None:
    """Advance Isaac Sim several frames so render state settles."""
    for _ in range(frames):
        app.update()


def _set_camera(
    app: IsaacSimApplication,
    path: str,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    focal_length_mm: float,
) -> None:
    """Create and update one Isaac Sim camera."""
    _backend(app).set_camera_view(
        path,
        eye_xyz=eye,
        target_xyz=target,
        focal_length_mm=focal_length_mm,
    )
    _pump(app, frames=18)


def _capture(
    *,
    camera_path: str,
    output_dir: Path,
    name: str,
    resolution: tuple[int, int],
) -> Path:
    """Capture one RGB render product from an Isaac Sim camera."""
    import omni.replicator.core as rep  # type: ignore

    capture_dir = output_dir / f"capture_{name}"
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(camera_path, resolution)
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=str(capture_dir), rgb=True)
    writer.attach(render_product)
    for _ in range(2):
        rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()
    candidates = sorted(capture_dir.glob("rgb*.png"))
    if not candidates:
        raise RuntimeError(f"Replicator did not write an RGB image in {capture_dir}")
    final_path = output_dir / f"{name}.png"
    shutil.copy2(candidates[-1], final_path)
    return final_path


def _save_pdf(image_path: Path, pdf_path: Path) -> None:
    """Write a PNG image as a single-page PDF."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=300.0)


def _font(
    size_px: int,
    *,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a Times-compatible serif font for in-figure labels."""
    font_path = SERIF_FONT_BOLD if bold else SERIF_FONT_REGULAR
    try:
        return ImageFont.truetype(font_path, size_px)
    except OSError:
        return ImageFont.load_default()


def _text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    outline_rgb: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Draw a compact white label box and return its bounds."""
    left, top = xy
    bbox = draw.multiline_textbbox((left, top), text, font=font, spacing=4)
    padding_x = 12
    padding_y = 8
    box = (
        bbox[0] - padding_x,
        bbox[1] - padding_y,
        bbox[2] + padding_x,
        bbox[3] + padding_y,
    )
    draw.rounded_rectangle(
        box,
        radius=4,
        fill=(255, 255, 255, 232),
        outline=outline_rgb + (255,),
        width=3,
    )
    draw.multiline_text((left, top), text, fill=(15, 15, 15, 255), font=font, spacing=4)
    return box


def _callout(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    label_xy: tuple[int, int],
    target_xy: tuple[int, int],
    color_rgb: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw a label with a pointer line to the target feature."""
    box = _text_box(draw, label_xy, text, font, outline_rgb=color_rgb)
    start_x, start_y = _callout_anchor(box, target_xy)
    draw.line(
        (start_x, start_y, target_xy[0], target_xy[1]),
        fill=color_rgb + (255,),
        width=4,
    )
    radius = 8
    draw.ellipse(
        (
            target_xy[0] - radius,
            target_xy[1] - radius,
            target_xy[0] + radius,
            target_xy[1] + radius,
        ),
        fill=color_rgb + (255,),
    )


def _callout_anchor(
    box: tuple[int, int, int, int],
    target_xy: tuple[int, int],
) -> tuple[int, int]:
    """Return the box-edge point nearest to the target without crossing text."""
    left, top, right, bottom = box
    target_x, target_y = target_xy
    inset = 14
    if target_y < top:
        return min(max(target_x, left + inset), right - inset), top
    if target_y > bottom:
        return min(max(target_x, left + inset), right - inset), bottom
    if target_x < left:
        return left, min(max(target_y, top + inset), bottom - inset)
    return right, min(max(target_y, top + inset), bottom - inset)


def _legend(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[int, int],
    rows: tuple[tuple[str, tuple[int, int, int]], ...],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw an in-panel photon-track color legend."""
    left, top = xy
    line_len = 66
    row_h = 48
    width = 520
    height = 22 + row_h * len(rows)
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=4,
        fill=(255, 255, 255, 228),
        outline=(35, 35, 35, 255),
        width=2,
    )
    for index, (label, color) in enumerate(rows):
        y = top + 22 + index * row_h
        draw.line(
            (left + 18, y + 16, left + 18 + line_len, y + 16),
            fill=color + (255,),
            width=5,
        )
        draw.text((left + 102, y), label, fill=(15, 15, 15, 255), font=font)


def _annotate_transport_capture(image_path: Path, *, obstructed: bool) -> Path:
    """Add publication-style labels to a transport capture."""
    with Image.open(image_path).convert("RGBA") as image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        label_font = _font(50)
        legend_font = _font(42)
        _legend(
            draw,
            xy=(42, 42),
            rows=(
                ("emitted gamma ray", (0, 205, 60)),
                ("detected gamma ray", (222, 172, 0)),
                ("attenuated gamma ray", (0, 0, 0)),
                ("scattered gamma ray", (0, 188, 205)),
            ),
            font=legend_font,
        )
        _callout(
            draw,
            text="radiation\nsource",
            label_xy=(62, 378),
            target_xy=(286, 430),
            color_rgb=(190, 55, 45),
            font=label_font,
        )
        _callout(
            draw,
            text="CeBr3\ndetector",
            label_xy=(1040, 320),
            target_xy=(1164, 462),
            color_rgb=(0, 142, 165),
            font=label_font,
        )
        _callout(
            draw,
            text="rotating Pb/Fe\noctant shields",
            label_xy=(970, 640),
            target_xy=(1128, 480),
            color_rgb=(130, 110, 35),
            font=label_font,
        )
        if obstructed:
            _callout(
                draw,
                text="concrete\nobstacle",
                label_xy=(590, 246),
                target_xy=(704, 430),
                color_rgb=(80, 80, 80),
                font=label_font,
            )
            _callout(
                draw,
                text="scattering\nsite",
                label_xy=(816, 170),
                target_xy=(818, 342),
                color_rgb=(0, 150, 170),
                font=label_font,
            )
        annotated = Image.alpha_composite(image, overlay).convert("RGB")
        annotated.save(image_path)
    return image_path


def _offset_detector_points(
    count: int,
    *,
    radius: float = 0.19,
) -> list[tuple[float, float, float]]:
    """Return deterministic target points distributed over the detector face."""
    offsets: list[tuple[float, float, float]] = []
    for index in range(count):
        angle = 2.399963 * index
        radial = radius * math.sqrt((index + 0.5) / count)
        offsets.append(
            (
                DETECTOR[0],
                DETECTOR[1] + radial * math.cos(angle),
                DETECTOR[2] + radial * math.sin(angle),
            )
        )
    return offsets


def _point_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    """Interpolate between two 3-D points."""
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
        start[2] + (end[2] - start[2]) * fraction,
    )


def _with_offset(
    point: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Add a small deterministic offset to a 3-D point."""
    return (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2])


def _room_boundary_endpoint(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the point where a source ray reaches the room boundary."""
    bounds_min = (0.08, 0.08, 0.08)
    bounds_max = (
        ROOM_SIZE_XYZ[0] - 0.08,
        ROOM_SIZE_XYZ[1] - 0.08,
        ROOM_SIZE_XYZ[2] - 0.08,
    )
    candidates: list[float] = []
    for origin, component, lower, upper in zip(
        SOURCE,
        direction,
        bounds_min,
        bounds_max,
        strict=True,
    ):
        if abs(component) < 1e-8:
            continue
        limit = upper if component > 0.0 else lower
        distance = (limit - origin) / component
        if distance > 0.0:
            candidates.append(distance)
    scale = min(candidates) if candidates else 1.0
    return (
        SOURCE[0] + direction[0] * scale,
        SOURCE[1] + direction[1] * scale,
        SOURCE[2] + direction[2] * scale,
    )


def _normal_emission_endpoints() -> list[tuple[float, float, float]]:
    """Return endpoints that make source emission look omnidirectional."""
    directions: list[tuple[float, float, float]] = []

    for index in range(28):
        angle = 2.0 * math.pi * index / 28
        directions.append(
            (
                math.cos(angle),
                0.18 * math.sin(2.0 * angle),
                0.72 * math.sin(angle),
            )
        )

    for index in range(16):
        angle = 2.0 * math.pi * index / 16
        directions.append(
            (
                0.78 * math.cos(angle),
                math.sin(angle),
                0.22 * math.sin(3.0 * angle),
            )
        )

    endpoints: list[tuple[float, float, float]] = []
    for direction in directions:
        length = math.sqrt(sum(component * component for component in direction))
        unit = tuple(component / length for component in direction)
        endpoints.append(_room_boundary_endpoint(unit))
    return endpoints


def _author_normal_emission_tracks(app: IsaacSimApplication) -> None:
    """Add green omnidirectional source-emission tracks for visualization."""
    backend = _backend(app)
    for index, endpoint in enumerate(_normal_emission_endpoints()):
        midpoint = _point_between(SOURCE, endpoint, 0.48)
        offset = (
            0.0,
            0.035 * math.sin(index * 1.7),
            0.035 * math.cos(index * 1.1),
        )
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/NormalEmission_{index:02d}",
            points_xyz=(SOURCE, _with_offset(midpoint, offset), endpoint),
            color_rgb=NORMAL_RAY_COLOR,
            width_m=0.012,
        )


def _author_common_markers(app: IsaacSimApplication) -> None:
    """Add stable source, detector, and label-free reference markers."""
    backend = _backend(app)
    backend.ensure_sphere(
        "/World/SimBridge/Transport/SourceHalo",
        radius_m=0.32,
        translation_xyz=SOURCE,
        color_rgb=(1.0, 0.08, 0.02),
        material="air",
    )
    backend.ensure_sphere(
        "/World/SimBridge/Transport/DetectorHitVolume",
        radius_m=0.28,
        translation_xyz=DETECTOR,
        color_rgb=(0.0, 0.86, 1.0),
        material="air",
    )
    backend.step()


def _author_direct_tracks(app: IsaacSimApplication) -> None:
    """Add direct Geant4-style source-to-detector photon tracks."""
    backend = _backend(app)
    backend.remove_prim("/World/SimBridge/Transport")
    backend.ensure_xform("/World/SimBridge/Transport")
    _author_common_markers(app)
    _author_normal_emission_tracks(app)
    for index, target in enumerate(_offset_detector_points(18, radius=0.18)):
        mid = _point_between(SOURCE, target, 0.55)
        offset = (
            0.0,
            0.08 * math.sin(index * 1.7),
            0.05 * math.cos(index * 1.3),
        )
        points = (SOURCE, _with_offset(mid, offset), target)
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/DirectPhoton_{index:02d}",
            points_xyz=points,
            color_rgb=DETECTED_RAY_COLOR,
            width_m=0.025,
        )
        if index % 3 == 0:
            backend.ensure_sphere(
                f"/World/SimBridge/Transport/DetectorHit_{index:02d}",
                radius_m=0.045,
                translation_xyz=target,
                color_rgb=DETECTED_RAY_COLOR,
                material="air",
            )
    backend.step()


def _obstacle_entry_point(target: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the approximate front-face intersection with the concrete obstacle."""
    front_x = OBSTACLE_CENTER[0] - 0.5 * OBSTACLE_SIZE[0]
    fraction = (front_x - SOURCE[0]) / (target[0] - SOURCE[0])
    return _point_between(SOURCE, target, fraction)


def _obstacle_exit_point(target: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return the approximate back-face intersection with the concrete obstacle."""
    back_x = OBSTACLE_CENTER[0] + 0.5 * OBSTACLE_SIZE[0]
    fraction = (back_x - SOURCE[0]) / (target[0] - SOURCE[0])
    return _point_between(SOURCE, target, fraction)


def _author_concrete_obstacle(app: IsaacSimApplication) -> None:
    """Add the semi-transparent concrete obstacle used in the scatter view."""
    backend = _backend(app)
    backend.ensure_box(
        "/World/SimBridge/Transport/ConcreteSlab",
        size_xyz=OBSTACLE_SIZE,
        translation_xyz=OBSTACLE_CENTER,
        color_rgb=(0.42, 0.45, 0.47),
        material="concrete",
        transport_group="obstacle",
    )
    backend.apply_visual_material(
        "/World/SimBridge/Transport/ConcreteSlab",
        color_rgb=(0.42, 0.45, 0.47),
        opacity=0.68,
        roughness=0.86,
        emissive_scale=0.0,
    )


def _author_obstacle_tracks(app: IsaacSimApplication) -> None:
    """Add attenuation, transmission, and scatter photon-track overlays."""
    backend = _backend(app)
    backend.remove_prim("/World/SimBridge/Transport")
    backend.ensure_xform("/World/SimBridge/Transport")
    _author_common_markers(app)
    _author_concrete_obstacle(app)
    _author_normal_emission_tracks(app)

    detector_targets = _offset_detector_points(20, radius=0.2)
    detected_indices = {1, 6, 11, 15, 18}
    for index, target in enumerate(detector_targets):
        entry = _obstacle_entry_point(target)
        entry = _with_offset(
            entry,
            (
                0.0,
                0.12 * math.sin(index * 1.1),
                0.10 * math.cos(index * 1.45),
            ),
        )
        exit_point = _obstacle_exit_point(target)
        exit_point = _with_offset(
            exit_point,
            (
                0.0,
                0.05 * math.sin(index * 1.3),
                0.05 * math.cos(index * 1.6),
            ),
        )
        if index in detected_indices:
            backend.ensure_polyline(
                f"/World/SimBridge/Transport/DetectedThroughObstacle_{index:02d}",
                points_xyz=(SOURCE, entry, exit_point, target),
                color_rgb=DETECTED_RAY_COLOR,
                width_m=0.022,
            )
            backend.ensure_sphere(
                f"/World/SimBridge/Transport/DetectedHit_{index:02d}",
                radius_m=0.043,
                translation_xyz=target,
                color_rgb=DETECTED_RAY_COLOR,
                material="air",
            )
            continue

        absorbed_stop = _point_between(entry, exit_point, 0.58)
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/AttenuatedPhoton_{index:02d}",
            points_xyz=(SOURCE, entry, absorbed_stop),
            color_rgb=ATTENUATED_RAY_COLOR,
            width_m=0.024,
        )
        backend.ensure_sphere(
            f"/World/SimBridge/Transport/Absorbed_{index:02d}",
            radius_m=0.04,
            translation_xyz=absorbed_stop,
            color_rgb=ATTENUATED_RAY_COLOR,
            material="air",
        )

    scatter_origins = [
        _with_offset(OBSTACLE_CENTER, (0.08, -0.46, 0.18)),
        _with_offset(OBSTACLE_CENTER, (0.02, 0.38, 0.28)),
        _with_offset(OBSTACLE_CENTER, (0.14, -0.18, -0.22)),
        _with_offset(OBSTACLE_CENTER, (0.12, 0.18, -0.04)),
        _with_offset(OBSTACLE_CENTER, (0.05, 0.0, 0.42)),
    ]
    scatter_ends = [
        (5.3, 1.28, 1.35),
        (5.0, 5.55, 1.5),
        (5.6, 2.1, 0.38),
        (5.8, 4.7, 0.88),
        (4.9, 3.6, 2.25),
    ]
    for index, (origin, end) in enumerate(zip(scatter_origins, scatter_ends, strict=True)):
        knee = _with_offset(_point_between(origin, end, 0.45), (0.18, 0.0, 0.18))
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/ScatteredPhoton_{index:02d}",
            points_xyz=(origin, knee, end),
            color_rgb=SCATTERED_RAY_COLOR,
            width_m=0.022,
        )
        backend.ensure_sphere(
            f"/World/SimBridge/Transport/ScatterPoint_{index:02d}",
            radius_m=0.035,
            translation_xyz=origin,
            color_rgb=SCATTERED_RAY_COLOR,
            material="air",
        )
    backend.step()


def _compose_panels(
    direct_path: Path,
    obstructed_path: Path,
    output_path: Path,
) -> Path:
    """Compose the two transport captures into one labeled figure."""
    with (
        Image.open(direct_path).convert("RGB") as direct,
        Image.open(obstructed_path).convert("RGB") as obstructed,
    ):
        label_h = 82
        gap = 20
        tile_w, tile_h = direct.size
        canvas = Image.new("RGB", (2 * tile_w + gap, tile_h + label_h), "white")
        canvas.paste(direct, (0, label_h))
        canvas.paste(obstructed, (tile_w + gap, label_h))
        draw = ImageDraw.Draw(canvas)
        font = _font(56)
        draw.text(
            (18, 16),
            "(a) Omnidirectional emission and detector hits",
            fill=(20, 20, 20),
            font=font,
        )
        draw.text(
            (tile_w + gap + 18, 16),
            "(b) Obstacle attenuation and scatter",
            fill=(20, 20, 20),
            font=font,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return output_path


def main() -> None:
    """Render the transport visualization panels and composite figure."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = IsaacSimApplication(use_mock=False, app_config=_app_config())
    try:
        app.reset(_scene_description())
        app.step(_command(step_id=0))

        _author_direct_tracks(app)
        _set_camera(
            app,
            "/World/SimBridge/Transport/Camera",
            eye=(4.15, -5.25, 4.55),
            target=(4.15, 3.25, 0.82),
            focal_length_mm=20.0,
        )
        direct = _capture(
            camera_path="/World/SimBridge/Transport/Camera",
            output_dir=OUTPUT_ROOT,
            name="geant4_direct_transport",
            resolution=(1450, 900),
        )
        _annotate_transport_capture(direct, obstructed=False)

        _author_obstacle_tracks(app)
        _set_camera(
            app,
            "/World/SimBridge/Transport/Camera",
            eye=(4.15, -5.25, 4.55),
            target=(4.15, 3.25, 0.82),
            focal_length_mm=20.0,
        )
        obstructed = _capture(
            camera_path="/World/SimBridge/Transport/Camera",
            output_dir=OUTPUT_ROOT,
            name="geant4_obstacle_scatter",
            resolution=(1450, 900),
        )
        _annotate_transport_capture(obstructed, obstructed=True)

        composite = _compose_panels(
            direct,
            obstructed,
            OUTPUT_ROOT / "geant4_transport_visualization.png",
        )
        for image_path in (direct, obstructed, composite):
            _save_pdf(image_path, image_path.with_suffix(".pdf"))
    finally:
        app.close()


if __name__ == "__main__":
    main()
