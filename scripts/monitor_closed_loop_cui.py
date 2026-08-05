"""Attach a truth-free CUI dashboard to an already-running PF closed loop."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from runtime.adaptive import AdaptiveCandidateProvider, cui_truth_overlay_from_scene
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.runtime import load_runtime_config

from cui_runtime import ensure_cui_view_server
from visualization.realtime_viz import CUISplitPFVisualizer, PFFrame


def _json_lines(path: Path) -> list[dict[str, object]]:
    """Read complete JSON objects from one append-only controller trace."""
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("Controller trace entries must be JSON objects.")
        records.append(value)
    return records


def _record_paths(stream_dir: Path) -> list[Path]:
    """Return staged MeasurementLog record files in acquisition order."""
    return sorted(stream_dir.glob("record_*.npz"))


def _read_record(path: Path) -> dict[str, np.ndarray]:
    """Load one native MeasurementLog record without reading truth artifacts."""
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _load_json_object(path: Path) -> dict[str, object]:
    """Load one finite JSON object from a CUI support artifact."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _latest_records_by_station(
    record_paths: Sequence[Path],
) -> list[dict[str, np.ndarray]]:
    """Return the latest shield view at each acquired station."""
    by_station: dict[int, dict[str, np.ndarray]] = {}
    for path in record_paths:
        record = _read_record(path)
        station_id = int(record["station_id"].reshape(-1)[0])
        by_station[station_id] = record
    return [by_station[station_id] for station_id in sorted(by_station)]


def _truth_arrays_from_scenario(
    scenario_path: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return CUI-only source truth arrays from a private runtime scenario."""
    if scenario_path is None:
        return {}, {}
    scenario = _load_json_object(scenario_path)
    scene = scenario.get("scene")
    if not isinstance(scene, dict):
        raise TypeError("Private scenario scene must be a JSON object.")
    overlay = cui_truth_overlay_from_scene(build_scene_description(scene))
    sources_raw = overlay["true_sources"]
    strengths_raw = overlay["true_strengths"]
    if not isinstance(sources_raw, Mapping) or not isinstance(
        strengths_raw,
        Mapping,
    ):
        raise TypeError("CUI truth overlay must contain source mappings.")
    sources = {
        str(isotope): np.asarray(values, dtype=np.float64).reshape((-1, 3))
        for isotope, values in sources_raw.items()
    }
    strengths = {
        str(isotope): np.asarray(values, dtype=np.float64).reshape(-1)
        for isotope, values in strengths_raw.items()
    }
    return sources, strengths


def _runtime_config_for_routes(
    stream_dir: Path,
    scenario_path: Path | None,
) -> Mapping[str, object] | None:
    """Load physical motion configuration for CUI route reconstruction."""
    if scenario_path is not None:
        scenario = _load_json_object(scenario_path)
        raw_config_path = scenario.get("runtime_config_path")
        if isinstance(raw_config_path, str):
            config_path = (scenario_path.parent / raw_config_path).resolve()
            return load_runtime_config(config_path)
    resolved_path = stream_dir / "runtime_config.resolved.json"
    if resolved_path.exists():
        return _load_json_object(resolved_path)
    return None


def _route_waypoints(
    provider: AdaptiveCandidateProvider,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray | None:
    """Return a CUI route segment between two station poses."""
    start = tuple(
        float(value)
        for value in np.asarray(start_pose, dtype=float).reshape(3)
    )
    target = tuple(
        float(value)
        for value in np.asarray(target_pose, dtype=float).reshape(3)
    )
    if float(np.linalg.norm(np.asarray(target) - np.asarray(start))) <= 1.0e-12:
        return None
    try:
        waypoints = provider.travel_waypoints_xyz(start, target)
    except (RuntimeError, ValueError):
        waypoints = (start, target)
    arr = np.asarray(waypoints, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        return None
    return arr


def _estimate_arrays(
    trace_entry: Mapping[str, object],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract truth-free posterior mode locations and strengths for the CUI."""
    snapshot = trace_entry.get("posterior_snapshot")
    if not isinstance(snapshot, Mapping):
        raise TypeError("Controller trace must include a posterior snapshot.")
    isotopes_raw = snapshot.get("isotopes")
    if not isinstance(isotopes_raw, Mapping):
        raise TypeError("Posterior snapshot isotopes must be a mapping.")
    isotopes = [str(isotope) for isotope in isotopes_raw]
    positions: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    for isotope in isotopes:
        estimate = isotopes_raw[isotope]
        if not isinstance(estimate, Mapping):
            raise TypeError("Posterior isotope summary must be a mapping.")
        modes = estimate.get("modes", [])
        if not isinstance(modes, list):
            raise TypeError("Posterior isotope modes must be a list.")
        mode_positions: list[list[float]] = []
        mode_strengths: list[float] = []
        for mode in modes:
            if not isinstance(mode, Mapping):
                continue
            position = mode.get("position_medoid_xyz")
            if not isinstance(position, list) or len(position) != 3:
                continue
            mode_positions.append([float(value) for value in position])
            mode_strengths.append(
                float(
                    mode.get(
                        "strength_representative_cps_1m",
                        mode.get("strength_median_cps_1m", 0.0),
                    )
                )
            )
        positions[isotope] = np.asarray(mode_positions, dtype=np.float64).reshape(
            (-1, 3)
        )
        strengths[isotope] = np.asarray(mode_strengths, dtype=np.float64)
    return isotopes, positions, strengths


def _frame_from_record(
    record: Mapping[str, np.ndarray],
    *,
    positions: Mapping[str, np.ndarray],
    strengths: Mapping[str, np.ndarray],
    elapsed_time_s: float,
    path_waypoints_xyz: np.ndarray | None = None,
) -> PFFrame:
    """Build a CUI-only frame from one truth-free native observation record."""
    normals = np.asarray(generate_octant_orientations(), dtype=np.float64)
    fe_index = int(record["fe_orientation_index"].reshape(-1)[0])
    pb_index = int(record["pb_orientation_index"].reshape(-1)[0])
    edges = np.asarray(record["energy_bin_edges_keV"], dtype=np.float64).reshape(-1)
    counts = np.asarray(record["spectrum_counts"], dtype=np.int64).reshape(-1)
    return PFFrame(
        step_index=int(record["step_id"].reshape(-1)[0]),
        time=float(elapsed_time_s),
        robot_position=np.asarray(
            record["detector_pose_xyz"], dtype=np.float64
        ).reshape(-1, 3)[0],
        robot_orientation=None,
        RFe=normals[fe_index],
        RPb=normals[pb_index],
        duration=float(record["live_time_s"].reshape(-1)[0]),
        particle_positions={
            isotope: np.zeros((0, 3), dtype=np.float64)
            for isotope in positions
        },
        particle_weights={
            isotope: np.zeros(0, dtype=np.float64) for isotope in positions
        },
        estimated_sources={
            isotope: np.asarray(values, dtype=np.float64)
            for isotope, values in positions.items()
        },
        estimated_strengths={
            isotope: np.asarray(values, dtype=np.float64)
            for isotope, values in strengths.items()
        },
        path_waypoints_xyz=path_waypoints_xyz,
        spectrum_energy_keV=0.5 * (edges[:-1] + edges[1:]),
        spectrum_counts=counts,
    )


def _elapsed_time(records: Sequence[Mapping[str, np.ndarray]]) -> float:
    """Return the physical mission time represented by the staged records."""
    return float(
        sum(
            float(record["live_time_s"].reshape(-1)[0])
            + float(record["travel_time_s"].reshape(-1)[0])
            + float(record["shield_actuation_time_s"].reshape(-1)[0])
            for record in records
        )
    )


def render_current_view(
    trace_path: Path,
    stream_dir: Path,
    output_dir: Path,
    scenario_path: Path | None = None,
) -> bool:
    """Render the latest fully assimilated posterior and acquired stations once."""
    trace = _json_lines(trace_path)
    record_paths = _record_paths(stream_dir)
    records = _latest_records_by_station(record_paths)
    if not trace or not records:
        return False
    environment_path = stream_dir / "environment.json"
    environment = _load_json_object(environment_path)
    if not isinstance(environment, Mapping):
        raise TypeError("Runtime environment must be a JSON object.")
    obstacle_raw = environment.get("obstacle_grid")
    obstacle_grid = (
        None
        if obstacle_raw is None
        else ObstacleGrid.from_dict(dict(obstacle_raw))
    )
    isotopes, positions, strengths = _estimate_arrays(trace[-1])
    true_sources, true_strengths = _truth_arrays_from_scenario(scenario_path)
    route_provider = AdaptiveCandidateProvider(
        environment,
        obstacle_grid,
        runtime_config=_runtime_config_for_routes(stream_dir, scenario_path),
    )
    visualizer = CUISplitPFVisualizer(
        isotopes=isotopes,
        output_dir=output_dir,
        world_bounds=(
            0.0,
            float(environment["size_x"]),
            0.0,
            float(environment["size_y"]),
            0.0,
            float(environment["size_z"]),
        ),
        true_sources=true_sources,
        true_strengths=true_strengths,
        obstacle_grid=obstacle_grid,
    )
    elapsed_time_s = _elapsed_time([_read_record(path) for path in record_paths])
    previous_pose = np.asarray(
        environment["detector_position"],
        dtype=np.float64,
    ).reshape(3)
    for record in records:
        current_pose = np.asarray(
            record["detector_pose_xyz"],
            dtype=np.float64,
        ).reshape(-1, 3)[0]
        visualizer.update(
            _frame_from_record(
                record,
                positions=positions,
                strengths=strengths,
                elapsed_time_s=elapsed_time_s,
                path_waypoints_xyz=_route_waypoints(
                    route_provider,
                    previous_pose,
                    current_pose,
                ),
            )
        )
        previous_pose = current_pose
    return True


def _signature(trace_path: Path, stream_dir: Path) -> tuple[int, int, int]:
    """Return a small state signature that changes with new CUI inputs."""
    trace_mtime = 0 if not trace_path.exists() else trace_path.stat().st_mtime_ns
    records = _record_paths(stream_dir)
    if not records:
        return (trace_mtime, 0, 0)
    latest = records[-1]
    return (trace_mtime, len(records), latest.stat().st_mtime_ns)


def _parse_args() -> argparse.Namespace:
    """Parse one monitor invocation without importing simulation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Continuously refresh the CUI without interacting with Geant4 or the PF."""
    args = _parse_args()
    if args.poll_s <= 0.0:
        raise ValueError("--poll-s must be positive.")
    output_dir = args.output_dir.resolve()
    url = ensure_cui_view_server(output_dir)
    print(f"CUI monitor URL: {url}", flush=True)
    previous_signature: tuple[int, int, int] | None = None
    while True:
        signature = _signature(args.trace, args.stream_dir)
        if signature != previous_signature:
            if render_current_view(
                args.trace,
                args.stream_dir,
                output_dir,
                args.scenario,
            ):
                print(
                    "CUI monitor rendered "
                    f"records={signature[1]} trace_mtime_ns={signature[0]}",
                    flush=True,
                )
            previous_signature = signature
        if args.once:
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
