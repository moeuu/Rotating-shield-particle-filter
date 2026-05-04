"""Render RA-L manuscript figures from an Isaac Sim scene.

The script authors a deterministic cluttered 3-D scene, captures the robot,
detector-shield module, and environment overview using Isaac Sim, and writes
PDF files into the RA-L manuscript figure directories.
Only visual prims are added here; this script does not alter runtime transport,
PF observations, or Geant4 settings.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LATEX_ROOT = ROOT.parent / "latex" / "projects" / "ieee-ra-l-letter"
OUTPUT_ROOT = ROOT / "results" / "ral_isaac_figures"
SERIF_FONT_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.isaacsim_app.app import IsaacSimApplication  # noqa: E402
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription  # noqa: E402
from sim.protocol import SimulationCommand  # noqa: E402


def _obstacle_cells() -> list[tuple[int, int]]:
    """Return a deterministic cluttered obstacle layout for the paper figures."""
    cells: set[tuple[int, int]] = set()
    cells.update((3, y) for y in range(2, 8))
    cells.update((6, y) for y in range(7, 13))
    cells.update((x, 10) for x in range(1, 5))
    cells.update((x, 4) for x in range(6, 9))
    cells.update({(1, 13), (2, 13), (8, 2), (8, 3), (4, 14), (5, 14)})
    cells.update({(1, 5), (2, 5), (7, 12), (8, 12), (5, 1)})
    return sorted(cells)


def _scene_description() -> SceneDescription:
    """Create the deterministic Isaac Sim scene used for all captures."""
    sources = [
        SourceDescription("Cs-137", (8.2, 13.3, 0.85), 30000.0),
        SourceDescription("Co-60", (2.1, 11.6, 0.85), 18000.0),
        SourceDescription("Eu-154", (7.4, 5.8, 0.85), 12000.0),
    ]
    return SceneDescription(
        room_size_xyz=(10.0, 16.0, 4.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 16),
        obstacle_material="concrete",
        obstacle_cells=_obstacle_cells(),
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
        sources=sources,
        usd_path=None,
        use_config_usd_fallback=False,
    )


def _app_config() -> dict[str, object]:
    """Return the visual Isaac Sim app configuration for manuscript captures."""
    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "detector_height_m": 0.72,
        "obstacle_height_m": 1.8,
        "robot_animation_time_scale": 0.0,
        "lighting": {
            "dome_intensity": 1400.0,
            "color_rgb": [0.98, 0.99, 1.0],
            "interior_lights": [
                {
                    "position_xyz": [2.0, 2.0, 3.7],
                    "intensity": 80000.0,
                    "radius_m": 0.05,
                },
                {
                    "position_xyz": [7.8, 7.0, 3.7],
                    "intensity": 90000.0,
                    "radius_m": 0.05,
                },
                {
                    "position_xyz": [4.5, 14.0, 3.7],
                    "intensity": 80000.0,
                    "radius_m": 0.05,
                },
            ],
        },
        "stage_visual_rules": [
            {
                "path_prefix": "/World/Environment/Wall/Floor",
                "color_rgb": [0.62, 0.65, 0.67],
                "opacity": 1.0,
                "roughness": 0.75,
            },
            {
                "path_prefix": "/World/Environment/Wall",
                "color_rgb": [0.62, 0.67, 0.70],
                "opacity": 0.18,
                "roughness": 0.85,
            },
            {
                "path_prefix": "/World/SimBridge/Obstacles",
                "color_rgb": [0.40, 0.43, 0.46],
                "opacity": 1.0,
                "roughness": 0.78,
            },
            {
                "path_prefix": "/World/SimBridge/Sources",
                "color_rgb": [1.0, 0.04, 0.02],
                "opacity": 1.0,
                "roughness": 0.25,
                "emissive_scale": 5.0,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Body",
                "color_rgb": [0.23, 0.29, 0.34],
                "opacity": 1.0,
                "roughness": 0.48,
                "emissive_scale": 0.2,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Detector",
                "color_rgb": [0.0, 0.85, 1.0],
                "opacity": 1.0,
                "roughness": 0.25,
                "emissive_scale": 2.8,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/FeShield",
                "color_rgb": [0.96, 0.56, 0.08],
                "opacity": 1.0,
                "roughness": 0.45,
                "emissive_scale": 0.7,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/PbShield",
                "color_rgb": [0.62, 0.66, 0.76],
                "opacity": 1.0,
                "roughness": 0.45,
                "emissive_scale": 0.4,
            },
        ],
        "stage_material_rules": [
            {"path_prefix": "/World/Environment", "material": "concrete"},
            {"path_prefix": "/World/SimBridge/Obstacles", "material": "concrete"},
        ],
    }


def _command(
    *,
    step_id: int,
    pose_xy: tuple[float, float],
    yaw: float,
    fe: int,
    pb: int,
) -> SimulationCommand:
    """Create a robot command for a still manuscript capture."""
    return SimulationCommand(
        step_id=step_id,
        target_pose_xyz=(pose_xy[0], pose_xy[1], 0.0),
        target_base_yaw_rad=yaw,
        fe_orientation_index=fe,
        pb_orientation_index=pb,
        dwell_time_s=30.0,
    )


def _pump(app: IsaacSimApplication, frames: int = 24) -> None:
    """Advance Isaac Sim several frames so render state settles."""
    for _ in range(frames):
        app.update()


def _backend(app: IsaacSimApplication):
    """Return the real Isaac Sim stage backend from the application."""
    backend = app._stage_backend  # noqa: SLF001
    if backend is None:
        raise RuntimeError("Isaac Sim backend is not available.")
    return backend


def _author_context_prims(
    app: IsaacSimApplication,
    *,
    include_robot_path: bool,
) -> None:
    """Add visual-only measurement path and source-ray guides for captures."""
    backend = _backend(app)
    if include_robot_path:
        path_points = (
            (1.2, 1.4, 0.08),
            (2.4, 3.1, 0.08),
            (4.3, 5.9, 0.08),
            (5.5, 8.2, 0.08),
            (7.1, 10.4, 0.08),
            (8.0, 12.2, 0.08),
        )
        backend.ensure_polyline(
            "/World/SimBridge/View/RobotPath",
            points_xyz=path_points,
            color_rgb=(0.05, 0.32, 1.0),
            width_m=0.045,
        )
        for index, point in enumerate(path_points):
            backend.ensure_sphere(
                f"/World/SimBridge/View/Measurement_{index:02d}",
                radius_m=0.085,
                translation_xyz=point,
                color_rgb=(0.05, 0.32, 1.0),
                material="air",
            )
    detector = (4.3, 5.9, 0.72)
    for index, source in enumerate(_scene_description().sources):
        source_marker = (source.position_xyz[0], source.position_xyz[1], 2.08)
        backend.ensure_sphere(
            f"/World/SimBridge/View/SourceMarker_{index:02d}",
            radius_m=0.17,
            translation_xyz=source_marker,
            color_rgb=(1.0, 0.05, 0.02),
            material="air",
        )
        backend.ensure_polyline(
            f"/World/SimBridge/View/GammaRay_{index:02d}",
            points_xyz=(source_marker, detector),
            color_rgb=(1.0, 0.86, 0.05),
            width_m=0.026,
        )
    backend.step()


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


def _font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a Times-compatible serif font for figure annotations."""
    try:
        return ImageFont.truetype(SERIF_FONT_REGULAR, size_px)
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
    """Draw a compact in-figure label box and return its bounds."""
    left, top = xy
    bbox = draw.multiline_textbbox((left, top), text, font=font, spacing=4)
    pad_x = 12
    pad_y = 8
    box = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
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


def _callout_many(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    label_xy: tuple[int, int],
    target_xys: tuple[tuple[int, int], ...],
    color_rgb: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw one label with pointer lines to several target features."""
    box = _text_box(draw, label_xy, text, font, outline_rgb=color_rgb)
    radius = 8
    for target_xy in target_xys:
        start_x, start_y = _callout_anchor(box, target_xy)
        draw.line(
            (start_x, start_y, target_xy[0], target_xy[1]),
            fill=color_rgb + (255,),
            width=4,
        )
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
    """Draw a small color legend inside a figure panel."""
    left, top = xy
    swatch_w = 68
    pad_x = 22
    pad_y = 18
    text_sizes = [draw.textbbox((0, 0), label, font=font) for label, _ in rows]
    text_width = max((bbox[2] - bbox[0] for bbox in text_sizes), default=0)
    text_height = max((bbox[3] - bbox[1] for bbox in text_sizes), default=28)
    row_h = max(56, text_height + 20)
    width = pad_x * 3 + swatch_w + text_width
    height = pad_y * 2 + row_h * len(rows)
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=4,
        fill=(255, 255, 255, 228),
        outline=(35, 35, 35, 255),
        width=2,
    )
    for index, (label, color) in enumerate(rows):
        y = top + pad_y + index * row_h
        center_y = y + row_h // 2
        draw.line(
            (left + pad_x, center_y, left + pad_x + swatch_w, center_y),
            fill=color + (255,),
            width=5,
        )
        draw.text(
            (left + pad_x * 2 + swatch_w, center_y - text_height // 2),
            label,
            fill=(15, 15, 15, 255),
            font=font,
        )


def _annotate_capture(image_path: Path, kind: str) -> Path:
    """Add publication-style component labels to one manuscript capture."""
    with Image.open(image_path).convert("RGBA") as image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        label_font = _font(60)
        legend_font = _font(58)
        if kind == "detector":
            _callout(
                draw,
                text="CeBr3\ndetector",
                label_xy=(980, 210),
                target_xy=(832, 566),
                color_rgb=(0, 140, 165),
                font=label_font,
            )
            _callout(
                draw,
                text="Fe octant\nshield",
                label_xy=(445, 735),
                target_xy=(683, 712),
                color_rgb=(210, 150, 0),
                font=label_font,
            )
            _callout(
                draw,
                text="Pb octant\nshield",
                label_xy=(950, 670),
                target_xy=(948, 520),
                color_rgb=(95, 105, 130),
                font=label_font,
            )
            _callout(
                draw,
                text="mobile\nbase",
                label_xy=(970, 880),
                target_xy=(850, 930),
                color_rgb=(65, 80, 90),
                font=label_font,
            )
        elif kind == "problem":
            _callout(
                draw,
                text="robot +\nNondirectional Detector\n+ Fe/Pb shields",
                label_xy=(300, 720),
                target_xy=(812, 675),
                color_rgb=(0, 120, 150),
                font=label_font,
            )
            _callout(
                draw,
                text="cluttered 3-D environment",
                label_xy=(1025, 805),
                target_xy=(1210, 620),
                color_rgb=(80, 80, 80),
                font=label_font,
            )
            _callout_many(
                draw,
                text="radiation sources",
                label_xy=(1255, 150),
                target_xys=((1195, 88), (620, 158), (1205, 523)),
                color_rgb=(190, 55, 45),
                font=label_font,
            )
            _callout(
                draw,
                text="gamma\nrays",
                label_xy=(1265, 390),
                target_xy=(1010, 565),
                color_rgb=(210, 170, 0),
                font=label_font,
            )
        elif kind == "environment":
            _legend(
                draw,
                xy=(44, 44),
                rows=(
                    ("measurement path", (0, 88, 210)),
                    ("gamma path", (220, 180, 0)),
                    ("blocked cell", (85, 92, 96)),
                ),
                font=legend_font,
            )
            _callout(
                draw,
                text="traversable\ncorridor",
                label_xy=(1120, 705),
                target_xy=(890, 610),
                color_rgb=(85, 85, 85),
                font=label_font,
            )
            _callout(
                draw,
                text="robot\nstation",
                label_xy=(640, 800),
                target_xy=(756, 710),
                color_rgb=(0, 120, 150),
                font=label_font,
            )
            _callout(
                draw,
                text="concrete\nobstacle",
                label_xy=(1130, 250),
                target_xy=(1220, 422),
                color_rgb=(80, 80, 80),
                font=label_font,
            )
        annotated = Image.alpha_composite(image, overlay).convert("RGB")
        annotated.save(image_path)
    return image_path


def _copy_to_manuscript() -> None:
    """Copy generated PDFs into the current RA-L manuscript figure locations."""
    copies = {
        "problem_setting.pdf": (
            LATEX_ROOT / "sections/01_introduction/figures/ProblemSetting.pdf"
        ),
    }
    for source_name, destination in copies.items():
        shutil.copy2(OUTPUT_ROOT / source_name, destination)


def main() -> None:
    """Render all Isaac Sim manuscript captures and update the LaTeX figures."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = IsaacSimApplication(use_mock=False, app_config=_app_config())
    try:
        app.reset(_scene_description())
        app.step(_command(step_id=0, pose_xy=(4.3, 5.9), yaw=1.05, fe=0, pb=6))

        _set_camera(
            app,
            "/World/SimBridge/View/DetectorCamera",
            eye=(6.35, 3.35, 2.35),
            target=(4.25, 5.92, 0.78),
            focal_length_mm=44.0,
        )
        detector = _capture(
            camera_path="/World/SimBridge/View/DetectorCamera",
            output_dir=OUTPUT_ROOT,
            name="detector_module",
            resolution=(1600, 1050),
        )
        _annotate_capture(detector, "detector")

        app.step(_command(step_id=20, pose_xy=(4.3, 5.9), yaw=1.05, fe=0, pb=6))
        _author_context_prims(app, include_robot_path=False)
        _set_camera(
            app,
            "/World/SimBridge/View/ProblemCamera",
            eye=(5.1, -5.6, 11.7),
            target=(5.0, 8.0, 0.2),
            focal_length_mm=22.0,
        )
        problem = _capture(
            camera_path="/World/SimBridge/View/ProblemCamera",
            output_dir=OUTPUT_ROOT,
            name="problem_setting",
            resolution=(1800, 1050),
        )
        _annotate_capture(problem, "problem")

        _set_camera(
            app,
            "/World/SimBridge/View/EnvironmentCamera",
            eye=(5.0, -3.6, 9.2),
            target=(5.0, 8.0, 0.1),
            focal_length_mm=23.0,
        )
        _author_context_prims(app, include_robot_path=True)
        environment = _capture(
            camera_path="/World/SimBridge/View/EnvironmentCamera",
            output_dir=OUTPUT_ROOT,
            name="simulation_environment",
            resolution=(1700, 1150),
        )
        _annotate_capture(environment, "environment")

        for image_path in (problem, detector, environment):
            _save_pdf(image_path, image_path.with_suffix(".pdf"))
        _copy_to_manuscript()
    finally:
        app.close()


if __name__ == "__main__":
    main()
