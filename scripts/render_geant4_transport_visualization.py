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
        room_size_xyz=(8.4, 6.8, 3.2),
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


def _offset_detector_points(count: int, *, radius: float = 0.19) -> list[tuple[float, float, float]]:
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
            color_rgb=(1.0, 0.84, 0.05),
            width_m=0.024,
        )
        if index % 3 == 0:
            backend.ensure_sphere(
                f"/World/SimBridge/Transport/DetectorHit_{index:02d}",
                radius_m=0.045,
                translation_xyz=target,
                color_rgb=(0.05, 1.0, 0.48),
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

    detector_targets = _offset_detector_points(20, radius=0.2)
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
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/IncomingPhoton_{index:02d}",
            points_xyz=(SOURCE, entry),
            color_rgb=(1.0, 0.82, 0.03),
            width_m=0.024,
        )
        if index not in {1, 6, 11, 15, 18}:
            backend.ensure_sphere(
                f"/World/SimBridge/Transport/Absorbed_{index:02d}",
                radius_m=0.04,
                translation_xyz=entry,
                color_rgb=(1.0, 0.06, 0.02),
                material="air",
            )

    for out_index, target_index in enumerate((1, 6, 11, 15, 18)):
        target = detector_targets[target_index]
        exit_point = _obstacle_exit_point(target)
        exit_point = _with_offset(
            exit_point,
            (
                0.0,
                0.06 * math.sin(out_index * 1.2),
                0.06 * math.cos(out_index * 1.4),
            ),
        )
        backend.ensure_polyline(
            f"/World/SimBridge/Transport/TransmittedPhoton_{out_index:02d}",
            points_xyz=(exit_point, target),
            color_rgb=(0.05, 0.74, 1.0),
            width_m=0.017,
        )
        backend.ensure_sphere(
            f"/World/SimBridge/Transport/TransmittedHit_{out_index:02d}",
            radius_m=0.043,
            translation_xyz=target,
            color_rgb=(0.05, 1.0, 0.48),
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
            color_rgb=(1.0, 0.34, 0.02),
            width_m=0.019,
        )
        backend.ensure_sphere(
            f"/World/SimBridge/Transport/ScatterPoint_{index:02d}",
            radius_m=0.035,
            translation_xyz=origin,
            color_rgb=(1.0, 0.42, 0.03),
            material="air",
        )
    backend.step()


def _compose_panels(
    direct_path: Path,
    obstructed_path: Path,
    output_path: Path,
) -> Path:
    """Compose the two transport captures into one labeled figure."""
    with Image.open(direct_path).convert("RGB") as direct, Image.open(obstructed_path).convert(
        "RGB"
    ) as obstructed:
        label_h = 82
        gap = 20
        tile_w, tile_h = direct.size
        canvas = Image.new("RGB", (2 * tile_w + gap, tile_h + label_h), "white")
        canvas.paste(direct, (0, label_h))
        canvas.paste(obstructed, (tile_w + gap, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 44)
        except OSError:
            font = ImageFont.load_default()
        draw.text((18, 16), "(a) Direct source-to-detector transport", fill=(20, 20, 20), font=font)
        draw.text(
            (tile_w + gap + 18, 16),
            "(b) Concrete attenuation and scatter",
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
