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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LATEX_ROOT = ROOT.parent / "latex" / "projects" / "ieee-ra-l-letter"
OUTPUT_ROOT = ROOT / "results" / "ral_isaac_figures"

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


def _author_context_prims(app: IsaacSimApplication) -> None:
    """Add visual-only path, measurement, and ray guides for the captures."""
    backend = _backend(app)
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
        end = source.position_xyz
        backend.ensure_polyline(
            f"/World/SimBridge/View/GammaRay_{index:02d}",
            points_xyz=(end, detector),
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


def _copy_to_manuscript() -> None:
    """Copy generated PDFs into the RA-L manuscript figure locations."""
    copies = {
        "problem_setting.pdf": LATEX_ROOT / "sections/01_introduction/figures/ProblemSetting.pdf",
        "detector_module.pdf": LATEX_ROOT / "sections/03_method/figures/Detector.pdf",
        "simulation_environment.pdf": (
            LATEX_ROOT / "sections/04_experiments/figures/Simulation_environment.pdf"
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

        app.step(_command(step_id=20, pose_xy=(4.3, 5.9), yaw=1.05, fe=0, pb=6))
        _author_context_prims(app)
        _set_camera(
            app,
            "/World/SimBridge/View/ProblemCamera",
            eye=(5.3, -4.6, 6.4),
            target=(5.2, 8.2, 0.7),
            focal_length_mm=26.0,
        )
        problem = _capture(
            camera_path="/World/SimBridge/View/ProblemCamera",
            output_dir=OUTPUT_ROOT,
            name="problem_setting",
            resolution=(1800, 1050),
        )

        _set_camera(
            app,
            "/World/SimBridge/View/EnvironmentCamera",
            eye=(5.0, -3.6, 9.2),
            target=(5.0, 8.0, 0.1),
            focal_length_mm=23.0,
        )
        environment = _capture(
            camera_path="/World/SimBridge/View/EnvironmentCamera",
            output_dir=OUTPUT_ROOT,
            name="simulation_environment",
            resolution=(1700, 1150),
        )

        for image_path in (problem, detector, environment):
            _save_pdf(image_path, image_path.with_suffix(".pdf"))
        _copy_to_manuscript()
    finally:
        app.close()


if __name__ == "__main__":
    main()
