"""Capture authenticated RA-L scene and detector views in Isaac Sim.

The scene capture is reconstructed from one completed MeasurementLog and its
private truth manifest. Three directionally diverse emitted gamma tracks per
source are selected from separately saved native Geant4 step trajectories at
the displayed recorded pose. The detector sequence uses four spatially legible
Fe/Pb pairs that were actually acquired at one adaptive station. Visual
annotations do not alter transport or inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT.parent / "Rotating-shield-simulation-runtime"
PF_SRC = ROOT / "src"
RUNTIME_SRC = RUNTIME_ROOT / "src"
OUTPUT_ROOT = ROOT / "results" / "ral_isaac_figures"
DEFAULT_RUN_ID = "ral_a3fde7067c4ac222_proposed"
DEFAULT_RUN_DIR = ROOT / "results" / "ral_ablation" / "runs" / DEFAULT_RUN_ID
DEFAULT_MEASUREMENT_LOG = (
    ROOT / "results" / "ral_ablation" / "measurement_logs" / DEFAULT_RUN_ID
)
DEFAULT_TRUTH_MANIFEST = (
    RUNTIME_ROOT
    / "private_runs"
    / "ral_ablation"
    / "truth_manifests"
    / f"{DEFAULT_RUN_ID}.json"
)
DEFAULT_GEANT4_TRACKS = (
    RUNTIME_ROOT
    / "private_runs"
    / "ral_ablation"
    / "figure_tracks"
    / f"{DEFAULT_RUN_ID}_station_05.json"
)
EMISSION_TRACKS_PER_SOURCE = 3

for import_root in (ROOT, PF_SRC, RUNTIME_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from measurement.obstacle_assets import obstacle_instances_from_dicts  # noqa: E402
from measurement.shielding import (  # noqa: E402
    physical_shield_normal_from_orientation_index,
)
from sim.isaacsim_app.app import IsaacSimApplication  # noqa: E402
from sim.isaacsim_app.estimator_visualizer import ISOTOPE_COLORS  # noqa: E402
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription  # noqa: E402
from sim.shield_geometry import SHIELD_CONTACT_RADIUS_M  # noqa: E402
from sim.protocol import SimulationCommand  # noqa: E402


@dataclass(frozen=True, slots=True)
class CaptureInputs:
    """Contain authenticated inputs required for the Isaac Sim captures."""

    run_id: str
    run_dir: Path
    measurement_log_dir: Path
    truth_manifest_path: Path
    environment: dict[str, Any]
    truth: dict[str, Any]
    station_positions_xyz: np.ndarray
    station_yaw_rad: np.ndarray
    station_pair_ids: tuple[tuple[int, ...], ...]
    route_segments_xyz: tuple[np.ndarray, ...]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_geant4_tracks(path: Path) -> dict[str, Any]:
    """Load an authenticated actual-Geant4 trajectory artifact."""
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    if payload.get("schema_version") != 1:
        raise ValueError("The Geant4 trajectory artifact has an unknown schema.")
    if payload.get("artifact_semantics") != (
        "actual native Geant4 primary-gamma step endpoints; no drawn or "
        "interpolated particle histories"
    ):
        raise ValueError("The trajectory artifact is not actual Geant4 step data.")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or not validation.get(
        "all_points_are_native_step_endpoints"
    ):
        raise ValueError("The trajectory artifact lacks native-step validation.")
    return payload


def _yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    """Return the planar yaw represented by one WXYZ quaternion."""
    w_value, x_value, y_value, z_value = (
        float(value) for value in np.asarray(quaternion, dtype=np.float64)
    )
    return math.atan2(
        2.0 * (w_value * z_value + x_value * y_value),
        1.0 - 2.0 * (y_value * y_value + z_value * z_value),
    )


def _load_route_segments(metadata_path: Path, run_id: str) -> tuple[np.ndarray, ...]:
    """Load exact persisted travel-waypoint segments from MeasurementLog rows."""
    segments: list[np.ndarray] = []
    for line_number, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") != run_id:
            raise ValueError(
                f"{metadata_path}:{line_number} has a different run_id."
            )
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(
                f"{metadata_path}:{line_number} lacks a metadata object."
            )
        raw_waypoints = metadata.get("travel_waypoints_xyz")
        if raw_waypoints is None:
            continue
        waypoints = np.asarray(raw_waypoints, dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 3
            or len(waypoints) < 2
            or np.any(~np.isfinite(waypoints))
        ):
            raise ValueError(
                f"{metadata_path}:{line_number} has invalid travel waypoints."
            )
        segments.append(waypoints)
    if not segments:
        raise ValueError("The completed MeasurementLog contains no saved route.")
    return tuple(segments)


def load_capture_inputs(
    *,
    run_dir: Path,
    measurement_log_dir: Path,
    truth_manifest_path: Path,
) -> CaptureInputs:
    """Load and cross-check one completed run for deterministic rendering."""
    run_dir = Path(run_dir).expanduser().resolve()
    measurement_log_dir = Path(measurement_log_dir).expanduser().resolve()
    truth_manifest_path = Path(truth_manifest_path).expanduser().resolve()
    result = _read_json(run_dir / "closed_loop_result.json")
    environment = _read_json(measurement_log_dir / "environment.json")
    truth = _read_json(truth_manifest_path)
    if result.get("execution_status") != "complete":
        raise ValueError("Isaac capture requires a completed closed-loop run.")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("The completed result lacks a valid run_id.")
    if truth.get("run_id") != run_id:
        raise ValueError("The truth manifest is not bound to the completed run.")

    observations_path = measurement_log_dir / "observations.npz"
    with np.load(observations_path, allow_pickle=False) as observations:
        station_ids = np.asarray(observations["station_id"], dtype=np.int64)
        poses = np.asarray(observations["detector_pose_xyz"], dtype=np.float64)
        quaternions = np.asarray(
            observations["detector_quat_wxyz"], dtype=np.float64
        )
        fe_indices = np.asarray(
            observations["fe_orientation_index"], dtype=np.int64
        )
        pb_indices = np.asarray(
            observations["pb_orientation_index"], dtype=np.int64
        )
    record_count = int(result.get("record_count", -1))
    station_count = int(result.get("station_count", -1))
    if (
        station_ids.shape != (record_count,)
        or poses.shape != (record_count, 3)
        or quaternions.shape != (record_count, 4)
        or fe_indices.shape != (record_count,)
        or pb_indices.shape != (record_count,)
    ):
        raise ValueError("Observation arrays disagree with the completed result.")
    if np.any(~np.isfinite(poses)) or np.any(~np.isfinite(quaternions)):
        raise ValueError("Observation poses contain nonfinite values.")
    expected_stations = np.arange(station_count, dtype=np.int64)
    if not np.array_equal(np.unique(station_ids), expected_stations):
        raise ValueError("Observation station identifiers are not contiguous.")
    first_rows = np.asarray(
        [int(np.flatnonzero(station_ids == station)[0]) for station in expected_stations]
    )
    station_positions = poses[first_rows]
    station_yaw = np.asarray(
        [_yaw_from_quaternion_wxyz(quaternions[row]) for row in first_rows],
        dtype=np.float64,
    )
    station_pairs: list[tuple[int, ...]] = []
    for station in expected_stations:
        mask = station_ids == station
        pair_ids = tuple(
            int(value) for value in (fe_indices[mask] * 8 + pb_indices[mask])
        )
        if len(pair_ids) != 8:
            raise ValueError("Every rendered station must contain exactly eight views.")
        station_pairs.append(pair_ids)
    route_segments = _load_route_segments(
        measurement_log_dir / "observation_metadata.jsonl",
        run_id,
    )
    return CaptureInputs(
        run_id=run_id,
        run_dir=run_dir,
        measurement_log_dir=measurement_log_dir,
        truth_manifest_path=truth_manifest_path,
        environment=environment,
        truth=truth,
        station_positions_xyz=station_positions,
        station_yaw_rad=station_yaw,
        station_pair_ids=tuple(station_pairs),
        route_segments_xyz=route_segments,
    )


def _scene_description(inputs: CaptureInputs) -> SceneDescription:
    """Create an Isaac scene from authenticated environment and truth artifacts."""
    environment = inputs.environment
    obstacle_grid = environment.get("obstacle_grid")
    if not isinstance(obstacle_grid, dict):
        raise TypeError("environment.obstacle_grid must be an object.")
    raw_instances = environment.get("obstacle_instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("The current scene lacks physical obstacle instances.")
    raw_sources = inputs.truth.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("The truth manifest lacks source records.")
    sources = [
        SourceDescription(
            isotope=str(source["isotope"]),
            position_xyz=tuple(float(value) for value in source["position"]),
            intensity_cps_1m=float(source["intensity_cps_1m"]),
            transport_position_xyz=tuple(
                float(value) for value in source["transport_position"]
            ),
            surface_chart_id=int(source["surface_chart_id"]),
            surface_uv=tuple(float(value) for value in source["surface_uv"]),
            surface_normal_xyz=tuple(
                float(value) for value in source["surface_normal"]
            ),
            surface_emission_policy_sha256=str(
                source["surface_emission_policy_sha256"]
            ),
        )
        for source in raw_sources
    ]
    return SceneDescription(
        room_size_xyz=tuple(
            float(environment[field]) for field in ("size_x", "size_y", "size_z")
        ),
        obstacle_origin_xy=tuple(float(value) for value in obstacle_grid["origin"]),
        obstacle_cell_size_m=float(obstacle_grid["cell_size"]),
        obstacle_grid_shape=tuple(int(value) for value in obstacle_grid["grid_shape"]),
        obstacle_material="concrete",
        obstacle_cells=[
            tuple(int(value) for value in cell)
            for cell in obstacle_grid.get("blocked_cells", [])
        ],
        obstacle_instances=obstacle_instances_from_dicts(raw_instances),
        author_obstacle_prims=True,
        author_room_boundary_prims=False,
        sources=sources,
        usd_path=None,
        use_config_usd_fallback=False,
    )


def _material_visual_rules(environment: dict[str, Any]) -> list[dict[str, object]]:
    """Return subtle material-specific colors for exact obstacle components."""
    colors = {
        "concrete": [0.39, 0.41, 0.43],
        "steel": [0.23, 0.29, 0.34],
        "aluminum": [0.56, 0.60, 0.63],
        "lead": [0.38, 0.40, 0.45],
    }
    rules: list[dict[str, object]] = []
    raw_instances = environment.get("obstacle_instances", [])
    if not isinstance(raw_instances, list):
        return rules
    for instance in raw_instances:
        if not isinstance(instance, dict):
            continue
        instance_name = str(instance.get("name", ""))
        components = instance.get("components", [])
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            material = str(component.get("material", "")).lower()
            color = colors.get(material, [0.45, 0.47, 0.49])
            component_name = str(component.get("name", ""))
            rules.append(
                {
                    "path_prefix": (
                        "/World/SimBridge/Obstacles/"
                        f"{instance_name}/{component_name}"
                    ),
                    "color_rgb": color,
                    "opacity": 1.0,
                    "roughness": 0.72,
                }
            )
    return rules


def _app_config(inputs: CaptureInputs) -> dict[str, object]:
    """Return a high-quality Isaac Sim configuration for manuscript captures."""
    source_rules = [
        {
            "path_prefix": "/World/SimBridge/Sources/Cs_137",
            "color_rgb": [1.0, 0.04, 0.03],
            "opacity": 1.0,
            "roughness": 0.20,
            "emissive_scale": 7.0,
        },
        {
            "path_prefix": "/World/SimBridge/Sources/Co_60",
            "color_rgb": [0.03, 0.42, 1.0],
            "opacity": 1.0,
            "roughness": 0.20,
            "emissive_scale": 7.0,
        },
    ]
    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "detector_height_m": 0.72,
        "obstacle_height_m": 1.8,
        "robot_animation_time_scale": 0.0,
        "lighting": {
            "dome_intensity": 1125.0,
            "color_rgb": [0.98, 0.99, 1.0],
            "interior_lights": [
                {
                    "position_xyz": [1.4, 1.5, 6.8],
                    "intensity": 65000.0,
                    "radius_m": 0.08,
                },
                {
                    "position_xyz": [8.7, 8.0, 7.2],
                    "intensity": 80000.0,
                    "radius_m": 0.08,
                },
                {
                    "position_xyz": [3.5, 14.0, 6.5],
                    "intensity": 70000.0,
                    "radius_m": 0.08,
                },
            ],
        },
        "stage_visual_rules": [
            {
                "path_prefix": "/World/Environment/Wall/Floor",
                "color_rgb": [0.73, 0.75, 0.76],
                "opacity": 1.0,
                "roughness": 0.78,
            },
            {
                "path_prefix": "/World/Environment/Wall",
                "color_rgb": [0.74, 0.78, 0.80],
                "opacity": 0.10,
                "roughness": 0.88,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Body",
                "color_rgb": [0.18, 0.23, 0.28],
                "opacity": 1.0,
                "roughness": 0.48,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/Detector",
                "color_rgb": [0.00, 0.84, 0.95],
                "opacity": 1.0,
                "roughness": 0.20,
                "emissive_scale": 3.2,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/FeShield",
                "color_rgb": [0.95, 0.56, 0.06],
                "opacity": 1.0,
                "roughness": 0.42,
                "emissive_scale": 0.45,
            },
            {
                "path_prefix": "/World/SimBridge/Robot/PbShield",
                "color_rgb": [0.72, 0.75, 0.82],
                "opacity": 1.0,
                "roughness": 0.42,
                "emissive_scale": 0.25,
            },
            *source_rules,
            *_material_visual_rules(inputs.environment),
        ],
        "stage_material_rules": [
            {"path_prefix": "/World/Environment", "material": "concrete"},
        ],
    }


def _command(
    inputs: CaptureInputs,
    *,
    station_index: int,
    pair_id: int,
    step_id: int,
) -> SimulationCommand:
    """Create a still command from one recorded station and orientation pair."""
    if station_index < 0 or station_index >= len(inputs.station_positions_xyz):
        raise ValueError("station_index is outside the recorded run.")
    if pair_id < 0 or pair_id >= 64:
        raise ValueError("pair_id must lie in [0, 63].")
    return SimulationCommand(
        step_id=step_id,
        target_pose_xyz=tuple(
            float(value) for value in inputs.station_positions_xyz[station_index]
        ),
        target_base_yaw_rad=float(inputs.station_yaw_rad[station_index]),
        fe_orientation_index=pair_id // 8,
        pb_orientation_index=pair_id % 8,
        dwell_time_s=20.0,
    )


def _backend(app: IsaacSimApplication):
    """Return the real stage backend from one Isaac Sim application."""
    backend = app._stage_backend  # noqa: SLF001
    if backend is None:
        raise RuntimeError("Isaac Sim stage backend is unavailable.")
    return backend


def _pump(app: IsaacSimApplication, frames: int = 24) -> None:
    """Advance Isaac Sim until authored render state has settled."""
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
    """Create or update one deterministic Isaac Sim camera."""
    _backend(app).set_camera_view(
        path,
        eye_xyz=eye,
        target_xyz=target,
        focal_length_mm=focal_length_mm,
    )
    _pump(app, frames=20)


def _capture(
    *,
    camera_path: str,
    output_dir: Path,
    name: str,
    resolution: tuple[int, int],
) -> Path:
    """Capture one raw RGB render from an Isaac Sim camera."""
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
        raise RuntimeError(f"Isaac Sim wrote no RGB image in {capture_dir}.")
    final_path = output_dir / f"{name}.png"
    shutil.copy2(candidates[-1], final_path)
    return final_path


def _trajectory_points(track: dict[str, Any]) -> np.ndarray:
    """Return one finite native Geant4 trajectory as an ``N x 3`` array."""
    points = np.asarray(track.get("points_xyz_m"), dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or np.any(~np.isfinite(points))
    ):
        raise ValueError("A saved Geant4 track contains invalid step endpoints.")
    return points


def _trajectory_length_m(track: dict[str, Any]) -> float:
    """Return the polyline length of one actual Geant4 trajectory."""
    points = _trajectory_points(track)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _trajectory_initial_direction(track: dict[str, Any]) -> np.ndarray:
    """Return the first nonzero unit direction of one saved trajectory."""
    deltas = np.diff(_trajectory_points(track), axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    nonzero = np.flatnonzero(lengths > 1.0e-12)
    if nonzero.size == 0:
        raise ValueError("A saved Geant4 track has no nonzero transport step.")
    index = int(nonzero[0])
    return deltas[index] / lengths[index]


def _trajectory_identity(track: dict[str, Any]) -> dict[str, object]:
    """Return fields that uniquely identify one rendered native trajectory."""
    return {
        key: track[key]
        for key in (
            "mode",
            "source_index",
            "isotope",
            "primary_batch_index",
            "primary_history_index",
            "bias_branch_lineage_id",
            "track_id",
            "initial_energy_keV",
            "raw_step_count",
            "detector_entered",
            "interacted",
            "points_truncated",
        )
    } | {
        "point_count": int(len(_trajectory_points(track))),
        "trajectory_length_m": _trajectory_length_m(track),
    }


def _select_geant4_tracks(
    inputs: CaptureInputs,
    artifact: dict[str, Any],
    *,
    station_index: int,
) -> tuple[dict[str, Any], ...]:
    """Select diverse actual emission tracks for every displayed source."""
    if artifact.get("run_id") != inputs.run_id:
        raise ValueError("Geant4 tracks are bound to a different run.")
    if int(artifact.get("station_index", -1)) != station_index:
        raise ValueError("Geant4 tracks were not recorded at the rendered station.")
    expected_pair = inputs.station_pair_ids[station_index][0]
    if int(artifact.get("recorded_pair_id", -1)) != expected_pair:
        raise ValueError("Geant4 tracks use a different recorded shield pair.")
    detector = np.asarray(
        artifact.get("detector_pose_xyz_m"), dtype=np.float64
    )
    if not np.allclose(
        detector,
        inputs.station_positions_xyz[station_index],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("Geant4 track detector pose differs from the MeasurementLog.")
    modes = artifact.get("modes")
    if not isinstance(modes, dict):
        raise TypeError("Geant4 trajectory modes must be an object.")
    isotropic = modes.get("isotropic_emission")
    if not isinstance(isotropic, dict):
        raise TypeError("The isotropic Geant4 trajectory mode is required.")
    isotropic_tracks = isotropic.get("tracks")
    if not isinstance(isotropic_tracks, list):
        raise TypeError("The isotropic Geant4 trajectory mode needs a track array.")

    emission_selection: list[dict[str, Any]] = []
    for source_index in range(len(inputs.truth["sources"])):
        candidates = [
            track
            for track in isotropic_tracks
            if int(track.get("source_index", -1)) == source_index
            and not bool(track.get("points_truncated", True))
            and _trajectory_length_m(track) >= 0.75
        ]
        if not candidates:
            raise ValueError(
                f"No readable actual Geant4 emission track for source {source_index}."
            )
        if len(candidates) < EMISSION_TRACKS_PER_SOURCE:
            raise ValueError(
                "Insufficient readable Geant4 emission tracks for source "
                f"{source_index}: need {EMISSION_TRACKS_PER_SOURCE}."
            )
        anchor = min(
            candidates,
            key=lambda track: (
                abs(_trajectory_length_m(track) - 3.0),
                int(track["primary_history_index"]),
            ),
        )
        selected = [anchor]
        selected_directions = [_trajectory_initial_direction(anchor)]
        remaining = [track for track in candidates if track is not anchor]
        while len(selected) < EMISSION_TRACKS_PER_SOURCE:
            next_track = min(
                remaining,
                key=lambda track: (
                    -min(
                        1.0
                        - float(
                            np.clip(
                                np.dot(
                                    _trajectory_initial_direction(track),
                                    direction,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                        for direction in selected_directions
                    ),
                    abs(_trajectory_length_m(track) - 3.0),
                    int(track["primary_history_index"]),
                ),
            )
            selected.append(next_track)
            selected_directions.append(_trajectory_initial_direction(next_track))
            remaining.remove(next_track)
        emission_selection.extend(selected)
    return tuple(emission_selection)


def _author_run_context(
    app: IsaacSimApplication,
    inputs: CaptureInputs,
    geant4_tracks: dict[str, Any],
    *,
    station_index: int,
) -> tuple[dict[str, Any], ...]:
    """Author the saved route, sources, and selected actual Geant4 tracks."""
    backend = _backend(app)
    context_root = "/World/SimBridge/View/AuthenticatedRun"
    backend.remove_prim(context_root)
    backend.ensure_xform(context_root)
    for index, segment in enumerate(inputs.route_segments_xyz):
        backend.ensure_polyline(
            f"{context_root}/Route_{index:02d}",
            points_xyz=tuple(
                tuple(float(value) for value in row) for row in segment
            ),
            color_rgb=(0.00, 0.62, 0.70),
            width_m=0.035,
        )
    for index, position in enumerate(inputs.station_positions_xyz):
        backend.ensure_sphere(
            f"{context_root}/Station_{index:02d}",
            radius_m=0.065,
            translation_xyz=tuple(float(value) for value in position),
            color_rgb=(0.04, 0.04, 0.04),
            material="air",
        )
    for index, source in enumerate(inputs.truth["sources"]):
        isotope = str(source["isotope"])
        backend.ensure_sphere(
            f"{context_root}/Source_{index:02d}",
            radius_m=0.145,
            translation_xyz=tuple(float(value) for value in source["position"]),
            color_rgb=ISOTOPE_COLORS.get(isotope, (1.0, 0.8, 0.05)),
            material="air",
        )
    emission_tracks = _select_geant4_tracks(
        inputs,
        geant4_tracks,
        station_index=station_index,
    )
    for index, track in enumerate(emission_tracks):
        points = _trajectory_points(track)
        backend.ensure_polyline(
            f"{context_root}/Geant4Emission_{index:02d}",
            points_xyz=tuple(
                tuple(float(value) for value in point) for point in points
            ),
            color_rgb=(0.18, 0.82, 0.26),
            width_m=0.030,
        )
    backend.step()
    return emission_tracks


def _author_room_context(app: IsaacSimApplication, inputs: CaptureInputs) -> None:
    """Author a solid ground plane and unobtrusive wireframe room boundary."""
    backend = _backend(app)
    room_x = float(inputs.environment["size_x"])
    room_y = float(inputs.environment["size_y"])
    room_z = float(inputs.environment["size_z"])
    root = "/World/SimBridge/View/RoomContext"
    backend.remove_prim(root)
    backend.ensure_xform(root)
    backend.ensure_box(
        f"{root}/Ground",
        size_xyz=(room_x, room_y, 0.08),
        translation_xyz=(0.5 * room_x, 0.5 * room_y, -0.04),
        color_rgb=(0.73, 0.75, 0.76),
        material="concrete",
    )
    lower = (
        (0.0, 0.0, 0.01),
        (room_x, 0.0, 0.01),
        (room_x, room_y, 0.01),
        (0.0, room_y, 0.01),
        (0.0, 0.0, 0.01),
    )
    upper = tuple((x_value, y_value, room_z) for x_value, y_value, _ in lower)
    backend.ensure_polyline(
        f"{root}/LowerBoundary",
        points_xyz=lower,
        color_rgb=(0.34, 0.37, 0.40),
        width_m=0.018,
    )
    backend.ensure_polyline(
        f"{root}/UpperBoundary",
        points_xyz=upper,
        color_rgb=(0.44, 0.47, 0.50),
        width_m=0.014,
    )
    for index, (x_value, y_value) in enumerate(
        ((0.0, 0.0), (room_x, 0.0), (room_x, room_y), (0.0, room_y))
    ):
        backend.ensure_polyline(
            f"{root}/VerticalBoundary_{index:02d}",
            points_xyz=((x_value, y_value, 0.0), (x_value, y_value, room_z)),
            color_rgb=(0.44, 0.47, 0.50),
            width_m=0.014,
        )
    backend.step()


def _author_studio(app: IsaacSimApplication) -> None:
    """Author a neutral inspection floor and background for head close-ups."""
    backend = _backend(app)
    backend.remove_prim("/World/SimBridge/View/AuthenticatedRun")
    backend.remove_prim("/World/SimBridge/View/Studio")
    backend.ensure_xform("/World/SimBridge/View/Studio")
    backend.ensure_box(
        "/World/SimBridge/View/Studio/Floor",
        size_xyz=(5.0, 5.0, 0.08),
        translation_xyz=(0.0, 0.0, -0.04),
        color_rgb=(0.70, 0.73, 0.75),
        material="concrete",
    )
    backend.ensure_box(
        "/World/SimBridge/View/Studio/Backdrop",
        size_xyz=(5.0, 0.08, 4.0),
        translation_xyz=(0.0, 1.65, 2.0),
        color_rgb=(0.42, 0.46, 0.50),
        material="concrete",
    )
    backend.step()


def _author_studio_telescoping_mast(
    app: IsaacSimApplication,
    *,
    detector_height_m: float,
) -> None:
    """Render a slim two-stage mast below the detector for close-up views."""
    if not math.isfinite(detector_height_m) or detector_height_m <= 0.20:
        raise ValueError("detector_height_m must exceed 0.20 m.")
    backend = _backend(app)
    robot_root = "/World/SimBridge/Robot"
    mount_top_m = detector_height_m - SHIELD_CONTACT_RADIUS_M
    upper_top_m = mount_top_m - 0.012
    collar_center_m = min(0.60, 0.62 * upper_top_m)
    lower_bottom_m = 0.10
    lower_top_m = collar_center_m + 0.015
    upper_bottom_m = collar_center_m - 0.010
    backend.ensure_box(
        f"{robot_root}/Mast",
        size_xyz=(0.045, 0.045, lower_top_m - lower_bottom_m),
        translation_xyz=(
            0.0,
            0.0,
            0.5 * (lower_bottom_m + lower_top_m),
        ),
        color_rgb=(0.22, 0.25, 0.28),
        material="steel",
    )
    backend.ensure_box(
        f"{robot_root}/StudioMastUpper",
        size_xyz=(0.022, 0.022, upper_top_m - upper_bottom_m),
        translation_xyz=(
            0.0,
            0.0,
            0.5 * (upper_bottom_m + upper_top_m),
        ),
        color_rgb=(0.54, 0.58, 0.61),
        material="steel",
    )
    backend.ensure_box(
        f"{robot_root}/StudioMastCollar",
        size_xyz=(0.060, 0.060, 0.040),
        translation_xyz=(0.0, 0.0, collar_center_m),
        color_rgb=(0.14, 0.17, 0.20),
        material="steel",
    )
    backend.ensure_box(
        f"{robot_root}/StudioHeadMount",
        size_xyz=(0.032, 0.032, 0.024),
        translation_xyz=(0.0, 0.0, mount_top_m - 0.012),
        color_rgb=(0.18, 0.21, 0.24),
        material="steel",
    )
    backend.step()


def _studio_command(pair_id: int, step_id: int) -> SimulationCommand:
    """Create one fixed-pose shield command for the detector inspection view."""
    return SimulationCommand(
        step_id=step_id,
        target_pose_xyz=(0.0, 0.0, 1.05),
        target_base_yaw_rad=0.0,
        fe_orientation_index=pair_id // 8,
        pb_orientation_index=pair_id % 8,
        dwell_time_s=20.0,
    )


def _select_spatially_legible_pairs(
    recorded_pair_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Select four recorded pairs whose two octants separate in the camera."""
    eye = np.asarray((0.90, -1.22, 1.48), dtype=np.float64)
    target = np.asarray((0.0, 0.0, 1.02), dtype=np.float64)
    view = eye - target
    view /= np.linalg.norm(view)
    camera_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    camera_right = np.cross(view, camera_up)
    camera_right /= np.linalg.norm(camera_right)
    camera_vertical = np.cross(camera_right, view)
    camera_vertical /= np.linalg.norm(camera_vertical)

    scores: list[tuple[float, int]] = []
    for recorded_index, pair_id in enumerate(recorded_pair_ids):
        fe_normal = physical_shield_normal_from_orientation_index(pair_id // 8)
        pb_normal = physical_shield_normal_from_orientation_index(pair_id % 8)
        fe_projection = np.asarray(
            (
                np.dot(fe_normal, camera_right),
                np.dot(fe_normal, camera_vertical),
            ),
            dtype=np.float64,
        )
        pb_projection = np.asarray(
            (
                np.dot(pb_normal, camera_right),
                np.dot(pb_normal, camera_vertical),
            ),
            dtype=np.float64,
        )
        separation = float(np.linalg.norm(fe_projection - pb_projection))
        visibility = max(0.0, float(np.dot(fe_normal, view))) + max(
            0.0, float(np.dot(pb_normal, view))
        )
        scores.append((separation + 0.4 * visibility, recorded_index))
    selected_indices = {
        recorded_index
        for _, recorded_index in sorted(scores, reverse=True)[:4]
    }
    return tuple(
        pair_id
        for recorded_index, pair_id in enumerate(recorded_pair_ids)
        if recorded_index in selected_indices
    )


def _artifact_record(path: Path) -> dict[str, object]:
    """Return one path, size, and digest record for provenance."""
    resolved = Path(path).resolve()
    return {
        "path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_provenance(
    inputs: CaptureInputs,
    *,
    output_dir: Path,
    output_paths: list[Path],
    environment_station_index: int,
    shield_station_index: int,
    all_shield_pair_ids: tuple[int, ...],
    shield_pair_ids: tuple[int, ...],
    geant4_track_artifact_path: Path,
    emission_tracks: tuple[dict[str, Any], ...],
) -> Path:
    """Write enough capture metadata to reproduce every raw render."""
    source_paths = [
        inputs.run_dir / "closed_loop_result.json",
        inputs.run_dir / "planner_audit.jsonl",
        inputs.measurement_log_dir / "environment.json",
        inputs.measurement_log_dir / "observations.npz",
        inputs.measurement_log_dir / "observation_metadata.jsonl",
        inputs.truth_manifest_path,
        geant4_track_artifact_path,
        Path(__file__),
        RUNTIME_ROOT / "src" / "sim" / "isaacsim_app" / "scene_builder.py",
        RUNTIME_ROOT / "src" / "sim" / "shield_geometry.py",
        RUNTIME_ROOT / "src" / "measurement" / "detector_geometry.py",
    ]
    payload = {
        "schema_version": 1,
        "run_id": inputs.run_id,
        "source_files": [_artifact_record(path) for path in source_paths],
        "outputs": [_artifact_record(path) for path in output_paths],
        "environment_capture": {
            "station_index": environment_station_index,
            "detector_pose_xyz_m": inputs.station_positions_xyz[
                environment_station_index
            ].tolist(),
            "pair_id": inputs.station_pair_ids[environment_station_index][0],
            "camera": {
                "eye_xyz_m": [14.5, -13.0, 13.0],
                "target_xyz_m": [5.0, 7.6, 1.45],
                "focal_length_mm": 25.0,
                "resolution_px": [2400, 1400],
            },
            "route_semantics": "exact persisted travel_waypoints_xyz",
            "station_semantics": "recorded detector_pose_xyz",
            "source_semantics": "private truth overlay for contextual paper figure",
            "geant4_track_artifact": _artifact_record(
                geant4_track_artifact_path
            ),
            "geant4_track_semantics": (
                "actual native Geant4 primary-gamma step endpoints selected "
                "without coordinate interpolation"
            ),
            "displayed_isotropic_emission_tracks": [
                _trajectory_identity(track) for track in emission_tracks
            ],
            "track_display_selection": (
                "three actual isotropic tracks per source: one approximately "
                "3 m anchor followed by two tracks maximizing the minimum "
                "initial-direction separation; deterministic history-index ties"
            ),
        },
        "detector_sequence": {
            "recorded_station_index": shield_station_index,
            "all_recorded_pair_ids": list(all_shield_pair_ids),
            "selected_pair_ids": list(shield_pair_ids),
            "orientation_pairs": [
                {"fe": pair_id // 8, "pb": pair_id % 8}
                for pair_id in shield_pair_ids
            ],
            "studio_detector_pose_xyz_m": [0.0, 0.0, 1.05],
            "camera": {
                "eye_xyz_m": [0.90, -1.22, 1.48],
                "target_xyz_m": [0.0, 0.0, 1.02],
                "focal_length_mm": 68.0,
                "resolution_px": [1400, 1050],
            },
            "note": (
                "The neutral studio changes only visual context; detector and "
                "shield geometry use the runtime scene builder. The support is "
                "rendered as a slim two-stage mast at the commanded detector "
                "height; this visual housing is not transport geometry. All "
                "eight recorded pairs are retained as candidate renders. The "
                "four displayed pairs maximize projected Fe/Pb separation and "
                "camera visibility while preserving acquisition order."
            ),
        },
        "renderer": "Isaac Sim RayTracedLighting",
        "randomness": "none",
    }
    output_path = Path(output_dir) / "isaac_capture_provenance.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse authenticated capture paths and selected run indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--measurement-log-dir",
        type=Path,
        default=DEFAULT_MEASUREMENT_LOG,
    )
    parser.add_argument(
        "--truth-manifest",
        type=Path,
        default=DEFAULT_TRUTH_MANIFEST,
    )
    parser.add_argument(
        "--geant4-tracks",
        type=Path,
        default=DEFAULT_GEANT4_TRACKS,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--environment-station",
        type=int,
        default=5,
        help="Recorded station shown with the robot in the scene overview.",
    )
    parser.add_argument(
        "--shield-station",
        type=int,
        default=1,
        help="Recorded adaptive station supplying the four displayed pair IDs.",
    )
    return parser.parse_args()


def main() -> None:
    """Capture the current scene and a four-view recorded shield sequence."""
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_capture_inputs(
        run_dir=args.run_dir,
        measurement_log_dir=args.measurement_log_dir,
        truth_manifest_path=args.truth_manifest,
    )
    geant4_track_path = Path(args.geant4_tracks).expanduser().resolve()
    geant4_tracks = _load_geant4_tracks(geant4_track_path)
    environment_station = int(args.environment_station)
    shield_station = int(args.shield_station)
    if shield_station <= 0:
        raise ValueError("The shield sequence must use a posterior-adaptive station.")
    all_shield_pair_ids = inputs.station_pair_ids[shield_station]
    shield_pair_ids = _select_spatially_legible_pairs(all_shield_pair_ids)
    generated: list[Path] = []
    provenance_path: Path | None = None
    app = IsaacSimApplication(use_mock=False, app_config=_app_config(inputs))
    try:
        app.reset(_scene_description(inputs))
        _author_room_context(app, inputs)
        environment_pair_id = inputs.station_pair_ids[environment_station][0]
        app.step(
            _command(
                inputs,
                station_index=environment_station,
                pair_id=environment_pair_id,
                step_id=0,
            )
        )
        emission_tracks = _author_run_context(
            app,
            inputs,
            geant4_tracks,
            station_index=environment_station,
        )
        _set_camera(
            app,
            "/World/SimBridge/View/EnvironmentCamera",
            eye=(14.5, -13.0, 13.0),
            target=(5.0, 7.6, 1.45),
            focal_length_mm=25.0,
        )
        generated.append(
            _capture(
                camera_path="/World/SimBridge/View/EnvironmentCamera",
                output_dir=output_dir,
                name="experiment_environment",
                resolution=(2400, 1400),
            )
        )

        studio_scene = SceneDescription(
            room_size_xyz=(5.0, 5.0, 4.0),
            author_obstacle_prims=False,
            author_room_boundary_prims=False,
            sources=[],
            usd_path=None,
            use_config_usd_fallback=False,
        )
        app.reset(studio_scene)
        _author_studio(app)
        _set_camera(
            app,
            "/World/SimBridge/View/DetectorSequenceCamera",
            eye=(0.90, -1.22, 1.48),
            target=(0.0, 0.0, 1.02),
            focal_length_mm=68.0,
        )
        candidate_paths: dict[int, Path] = {}
        for recorded_index, pair_id in enumerate(all_shield_pair_ids):
            app.step(_studio_command(pair_id, step_id=100 + recorded_index))
            _author_studio_telescoping_mast(app, detector_height_m=1.05)
            _pump(app, frames=18)
            candidate_path = _capture(
                camera_path="/World/SimBridge/View/DetectorSequenceCamera",
                output_dir=output_dir,
                name=(
                    f"shield_candidate_{recorded_index:02d}_pair_{pair_id:02d}"
                ),
                resolution=(1400, 1050),
            )
            candidate_paths[pair_id] = candidate_path
            generated.append(candidate_path)
        for view_index, pair_id in enumerate(shield_pair_ids):
            selected_path = output_dir / f"shield_sequence_{view_index:02d}.png"
            shutil.copy2(candidate_paths[pair_id], selected_path)
            generated.append(selected_path)
        provenance_path = _write_provenance(
            inputs,
            output_dir=output_dir,
            output_paths=generated,
            environment_station_index=environment_station,
            shield_station_index=shield_station,
            all_shield_pair_ids=all_shield_pair_ids,
            shield_pair_ids=shield_pair_ids,
            geant4_track_artifact_path=geant4_track_path,
            emission_tracks=emission_tracks,
        )
        for output in generated:
            print(f"Wrote {output}", flush=True)
        print(f"Wrote {provenance_path}", flush=True)
    finally:
        app.close()
    if provenance_path is None:
        raise RuntimeError("Isaac Sim closed before capture provenance was written.")


if __name__ == "__main__":
    main()
