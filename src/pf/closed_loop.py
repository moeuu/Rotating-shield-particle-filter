"""PF-owned control of estimator-neutral adaptive acquisition."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from measurement.obstacles import ObstacleGrid
from runtime.adaptive_client import (
    AdaptiveRuntimeClient,
    adaptive_step_request,
    candidate_index_for_pose,
    parse_adaptive_record,
    parse_candidate_snapshot,
    parse_run_context,
)
from runtime.measurement_log import load_measurement_log
from runtime.provenance import canonical_json_bytes

from cui_runtime import (
    ensure_cui_view_server,
    resolve_cui_split_view_enabled,
)

from pf.replay import (
    bind_finalized_measurement_log,
    build_live_estimator,
    load_pf_config,
    measurement_record_to_spectrum_input,
)
from pf.isotope_gate import FullSpectrumIsotopeGate
from planning.audit import PlannerAuditWriter, build_planner_audit
from planning.configuration import dss_config_from_pf_settings
from planning.dss_pp import (
    DSSPPConfig,
    DSSPPResult,
    ShieldProgram,
    build_shield_program_library,
    select_dss_pp_next_station,
)
from pf.runtime_defaults import DEFAULT_CUI_SPLIT_VIEW_DIR
from visualization.realtime_viz import (
    AsyncCUISplitPFVisualizer,
    build_frame_from_pf,
)

ROOT = Path(__file__).resolve().parents[2]


def _exact_integer(value: object, *, name: str, minimum: int) -> int:
    """Return one exact integer satisfying an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _finite_positive(value: object, *, name: str) -> float:
    """Return one finite, strictly positive real value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return parsed


def _strict_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    """Require an exact controller-facing protocol schema."""
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields disagree: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


@dataclass(frozen=True, slots=True)
class PFControlBudget:
    """Keep PF mission limits separate from physical runtime configuration."""

    max_stations: int
    max_measurements: int
    views_per_station: int
    live_time_s: float
    adaptive_stop: bool
    minimum_stop_stations: int
    runtime_refinement_top_k: int
    planner_audit_top_k: int

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        planner: DSSPPConfig,
    ) -> "PFControlBudget":
        """Resolve PF-owned station, view, and stopping limits."""
        max_stations = _exact_integer(
            settings.get("mission_stop_max_poses", 20),
            name="mission_stop_max_poses",
            minimum=1,
        )
        max_measurements = _exact_integer(
            settings.get(
                "measurement_budget_max_steps",
                max_stations * int(planner.program_length),
            ),
            name="measurement_budget_max_steps",
            minimum=1,
        )
        configured_views = _exact_integer(
            settings.get("orientation_k", int(planner.program_length)),
            name="orientation_k",
            minimum=1,
        )
        if configured_views != int(planner.program_length):
            raise ValueError(
                "orientation_k and dss_pp.program_length must agree for one "
                "station likelihood block."
            )
        if max_measurements < configured_views:
            raise ValueError(
                "measurement_budget_max_steps must accommodate at least one "
                "complete station program."
            )
        adaptive_stop = settings.get("adaptive_mission_stop", False)
        if not isinstance(adaptive_stop, bool):
            raise TypeError("adaptive_mission_stop must be a boolean.")
        refinement = _exact_integer(
            settings.get("runtime_candidate_refinement_top_k", 0),
            name="runtime_candidate_refinement_top_k",
            minimum=0,
        )
        audit_top_k = _exact_integer(
            settings.get(
                "planner_audit_top_k",
                max(10, int(planner.diagnostic_ranked_node_limit)),
            ),
            name="planner_audit_top_k",
            minimum=0,
        )
        return cls(
            max_stations=max_stations,
            max_measurements=max_measurements,
            views_per_station=configured_views,
            live_time_s=_finite_positive(
                settings.get("measurement_live_time_s", planner.live_time_s),
                name="measurement_live_time_s",
            ),
            adaptive_stop=adaptive_stop,
            minimum_stop_stations=_exact_integer(
                settings.get("mission_stop_min_convergence_poses", 4),
                name="mission_stop_min_convergence_poses",
                minimum=1,
            ),
            runtime_refinement_top_k=refinement,
            planner_audit_top_k=audit_top_k,
        )


@dataclass(frozen=True, slots=True)
class PFClosedLoopResult:
    """Describe one completed PF-controlled acquisition and posterior."""

    measurement_log_path: Path
    pf_output_dir: Path
    run_id: str
    record_count: int
    station_count: int
    stop_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return one strict JSON-safe result payload."""
        return {
            "schema_version": 1,
            "status": "complete",
            "control_mode": "pf_closed_loop",
            "measurement_log_path": self.measurement_log_path.as_posix(),
            "pf_output_dir": self.pf_output_dir.as_posix(),
            "run_id": self.run_id,
            "record_count": self.record_count,
            "station_count": self.station_count,
            "stop_reason": self.stop_reason,
        }


def _obstacle_grid(context: object) -> ObstacleGrid | None:
    """Return the truth-free embedded obstacle map for PF route scoring."""
    environment = getattr(context, "environment")
    raw = environment.get("obstacle_grid")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("context.environment.obstacle_grid must be an object.")
    return ObstacleGrid.from_dict(raw)


def _bounds(context: object) -> tuple[np.ndarray, np.ndarray]:
    """Return finite room bounds from the runtime handshake."""
    environment = getattr(context, "environment")
    upper = np.asarray(
        [environment["size_x"], environment["size_y"], environment["size_z"]],
        dtype=np.float64,
    )
    if upper.shape != (3,) or np.any(~np.isfinite(upper)) or np.any(upper <= 0.0):
        raise ValueError("Runtime environment bounds must be finite and positive.")
    return np.zeros(3, dtype=np.float64), upper


def _height_bounds(context: object) -> tuple[float, float] | None:
    """Return runtime-owned detector height limits when declared."""
    environment = getattr(context, "environment")
    raw = environment.get("adaptive_measurement", {})
    if not isinstance(raw, Mapping):
        raise TypeError("environment.adaptive_measurement must be a mapping.")
    if "detector_height_min_m" not in raw or "detector_height_max_m" not in raw:
        return None
    return (
        float(raw["detector_height_min_m"]),
        float(raw["detector_height_max_m"]),
    )


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Durably append one finite JSON object to a controller trace."""
    line = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        if os.write(descriptor, line) != len(line):
            raise OSError("PF controller trace append was incomplete.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_cui_bool(
    settings: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    """Return one boolean CUI setting without accepting truthy substitutes."""
    value = settings.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean.")
    return value


def _strict_cui_host(settings: Mapping[str, object]) -> str:
    """Return the configured network bind host for the CUI server."""
    value = settings.get("cui_split_view_host", "0.0.0.0")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("cui_split_view_host must be a nonempty string.")
    return value


def _strict_cui_port(settings: Mapping[str, object]) -> int:
    """Return a valid TCP port for the CUI server."""
    value = settings.get("cui_split_view_port", 8877)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cui_split_view_port must be an integer.")
    if value < 1 or value > 65535:
        raise ValueError("cui_split_view_port must be between 1 and 65535.")
    return int(value)


def _cui_truth_display_mode(settings: Mapping[str, object]) -> str:
    """Return the requested evaluation-truth display mode for the CUI only."""
    value = settings.get("cui_truth_display_mode", "post_run")
    if not isinstance(value, str):
        raise TypeError("cui_truth_display_mode must be a string.")
    mode = value.strip()
    if mode not in {"hidden", "evaluation_live", "post_run"}:
        raise ValueError(
            "cui_truth_display_mode must be hidden, evaluation_live, or post_run."
        )
    return mode


def _cui_output_dir(settings: Mapping[str, object]) -> Path:
    """Resolve the shared browser-served CUI output directory."""
    raw = settings.get("cui_split_view_dir", DEFAULT_CUI_SPLIT_VIEW_DIR)
    output_dir = Path(str(raw)).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir.resolve()


def _start_cui_split_view(
    settings: Mapping[str, object],
    *,
    isotopes: Sequence[str],
    room_bounds: tuple[np.ndarray, np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    output_hook: Any,
) -> AsyncCUISplitPFVisualizer | None:
    """Start the truth-free asynchronous CUI renderer for one PF run."""
    if not resolve_cui_split_view_enabled(settings, save_outputs=True):
        return None
    lower, upper = room_bounds
    output_dir = _cui_output_dir(settings)
    visualizer = AsyncCUISplitPFVisualizer(
        isotopes=[str(isotope) for isotope in isotopes],
        output_dir=output_dir,
        world_bounds=(
            float(lower[0]),
            float(upper[0]),
            float(lower[1]),
            float(upper[1]),
            float(lower[2]),
            float(upper[2]),
        ),
        obstacle_grid=obstacle_grid,
        max_particles_per_isotope=settings.get(
            "cui_split_view_max_particles_per_isotope"
        ),
    )
    output_hook(
        "CUI split visualization enabled: "
        f"{visualizer.index_path.as_posix()} "
        "(latest_robot_2d.png, latest_pf_3d.png)"
    )
    output_hook("CUI split visualization rendering: async process")
    if _strict_cui_bool(settings, "cui_split_view_serve", True):
        public_host_raw = settings.get("cui_split_view_public_host")
        if public_host_raw is not None and (
            not isinstance(public_host_raw, str) or not public_host_raw.strip()
        ):
            raise TypeError(
                "cui_split_view_public_host must be a nonempty string when set."
            )
        split_url = ensure_cui_view_server(
            output_dir,
            host=_strict_cui_host(settings),
            port=_strict_cui_port(settings),
            public_host=(None if public_host_raw is None else str(public_host_raw)),
        )
        output_hook(f"CUI split visualization URL: {split_url}")
    return visualizer


def _truth_arrays_from_cui_overlay(
    payload: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return CUI-only truth arrays from the private runtime overlay payload."""
    truth = payload.get("truth")
    if truth is None:
        return {}, {}
    if not isinstance(truth, Mapping):
        raise TypeError("CUI overlay truth must be an object or null.")
    sources_raw = truth.get("true_sources")
    strengths_raw = truth.get("true_strengths")
    if not isinstance(sources_raw, Mapping) or not isinstance(
        strengths_raw,
        Mapping,
    ):
        raise TypeError("CUI overlay truth sources and strengths must be objects.")
    if set(sources_raw) != set(strengths_raw):
        raise ValueError("CUI overlay truth isotope sets differ.")
    sources: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    for isotope, values in sources_raw.items():
        positions = np.asarray(values, dtype=np.float64)
        if positions.size == 0:
            positions = positions.reshape((0, 3))
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or np.any(~np.isfinite(positions))
        ):
            raise ValueError("CUI overlay truth positions must have shape (N, 3).")
        strength = np.asarray(strengths_raw[isotope], dtype=np.float64).reshape(-1)
        if strength.shape != (positions.shape[0],) or np.any(~np.isfinite(strength)):
            raise ValueError("CUI overlay truth strengths must align with sources.")
        sources[str(isotope)] = positions
        strengths[str(isotope)] = strength
    return sources, strengths


def _record_path_waypoints(record: object) -> np.ndarray | None:
    """Return truth-free travel waypoints from one runtime record metadata."""
    metadata = getattr(record, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("travel_waypoints_xyz")
    if raw is None:
        return None
    waypoints = np.asarray(raw, dtype=np.float64)
    if (
        waypoints.ndim != 2
        or waypoints.shape[1] != 3
        or np.any(~np.isfinite(waypoints))
    ):
        raise ValueError("record.metadata.travel_waypoints_xyz must have shape (N, 3).")
    if waypoints.shape[0] < 2:
        return None
    return waypoints.copy()


def _publish_cui_frame(
    visualizer: AsyncCUISplitPFVisualizer,
    estimator: object,
    record: object,
    *,
    elapsed_time_s: float,
    record_measurement: bool,
) -> None:
    """Queue one truth-free PF CUI frame without changing PF state."""
    energy_edges = np.asarray(record.energy_bin_edges_keV, dtype=np.float64)
    spectrum_counts = np.asarray(record.spectrum_counts, dtype=np.int64)
    if energy_edges.ndim != 1 or energy_edges.size != spectrum_counts.size + 1:
        raise ValueError("Adaptive record has incompatible spectrum bin edges.")
    normals = np.asarray(estimator.normals, dtype=np.float64)
    fe_index = int(record.fe_orientation_index)
    pb_index = int(record.pb_orientation_index)
    if (
        normals.ndim != 2
        or normals.shape[1] != 3
        or fe_index < 0
        or pb_index < 0
        or fe_index >= normals.shape[0]
        or pb_index >= normals.shape[0]
    ):
        raise ValueError("Adaptive record shield orientation is out of range.")
    frame = build_frame_from_pf(
        estimator,
        int(record.step_id),
        float(elapsed_time_s),
        detector_position=np.asarray(record.detector_pose_xyz, dtype=np.float64),
        live_time_s=float(record.live_time_s),
        RFe=normals[fe_index],
        RPb=normals[pb_index],
        spectrum_energy_keV=0.5 * (energy_edges[:-1] + energy_edges[1:]),
        spectrum_counts=spectrum_counts,
    )
    path_waypoints = _record_path_waypoints(record)
    if path_waypoints is not None:
        frame.path_waypoints_xyz = path_waypoints
    frame.record_measurement = bool(record_measurement)
    visualizer.update(frame)


def _particle_diagnostics(estimator: object) -> dict[str, object]:
    """Extract particle-adequacy evidence without simulation truth."""
    raw = estimator.step_diagnostics(top_k=0, include_estimates=False)
    keep = (
        "particle_count",
        "current_ess",
        "current_ess_ratio",
        "temper_resamples",
        "temper_min_ess",
        "joint_guided_initialization_ess",
        "station_unique_ancestor_count",
        "cumulative_unique_ancestor_count",
        "r_probability_by_count",
        "transition_weight_mass",
        "structural_rejection_diagnostics",
        "joint_cross_isotope_rejection_diagnostics",
        "joint_cross_isotope_state_rejection_diagnostics",
        "joint_smc_soft_budget_exceeded",
    )
    isotopes = {
        str(isotope): {key: values.get(key) for key in keep}
        for isotope, values in raw.items()
    }
    configured_count = int(estimator.pf_config.num_particles)
    target_ratio = float(estimator.pf_config.target_ess_ratio)
    guided_ratios = [
        float(values["joint_guided_initialization_ess"]) / configured_count
        for values in isotopes.values()
        if values.get("joint_guided_initialization_ess") is not None
    ]
    ancestry_counts = [
        int(values["cumulative_unique_ancestor_count"])
        for values in isotopes.values()
        if values.get("cumulative_unique_ancestor_count") is not None
    ]
    evidence = {
        "configured_particle_count": configured_count,
        "target_ess_ratio": target_ratio,
        "minimum_guided_initialization_ess_ratio": (
            None if not guided_ratios else float(min(guided_ratios))
        ),
        "minimum_cumulative_unique_ancestor_count": (
            None if not ancestry_counts else int(min(ancestry_counts))
        ),
        "diversity_warning": bool(
            (guided_ratios and min(guided_ratios) < target_ratio)
            or (ancestry_counts and min(ancestry_counts) <= 1)
        ),
        "interpretation": (
            "A warning means particle diversity may be insufficient, but "
            "particle count alone is not identified without independent-seed "
            "2k/4k/8k replay stability."
        ),
    }
    return {"assessment": evidence, "isotopes": isotopes}


def _live_posterior_summary(estimator: object) -> dict[str, object]:
    """Return a truth-free, explicitly non-publishable station summary.

    A publishable ``PFPosteriorSnapshot`` is intentionally unavailable until
    the runtime finalizes MeasurementLog v2 and supplies its immutable digest.
    The live controller therefore serializes only the current PF point
    estimates; final provenance remains exclusive to ``pf_posterior.json``.
    """
    raw = estimator.posterior_point_estimate()
    if not isinstance(raw, Mapping):
        raise TypeError("PF live point estimates must be an isotope mapping.")
    isotopes: dict[str, object] = {}
    for isotope, estimate in raw.items():
        to_dict = getattr(estimate, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("Every PF live point estimate must be serializable.")
        isotopes[str(isotope)] = to_dict()
    return {
        "schema_version": 1,
        "publishable": False,
        "provenance_status": "awaiting_finalized_measurement_log_digest",
        "isotopes": isotopes,
    }


def _bootstrap_program(estimator: object, planner: DSSPPConfig) -> ShieldProgram:
    """Choose a PF-owned balanced first-station shield program."""
    programs = build_shield_program_library(
        estimator.normals,
        program_length=int(planner.program_length),
        max_programs=int(planner.max_programs),
    )
    if not programs:
        raise RuntimeError("PF shield program library is empty.")
    return programs[0]


def _register_station_pose(
    estimator: object,
    records: Sequence[object],
    *,
    station_id: int,
) -> int:
    """Register one single-pose station and return its estimator pose index."""
    if not records:
        raise ValueError("A PF station must contain at least one record.")
    poses = np.asarray([record.detector_pose_xyz for record in records], dtype=float)
    quaternions = np.asarray(
        [record.detector_quat_wxyz for record in records],
        dtype=float,
    )
    if not np.all(poses == poses[0]) or not np.all(quaternions == quaternions[0]):
        raise ValueError("Every view in a PF station must share one detector pose.")
    pose = poses[0]
    if station_id == 0 and not estimator.measurements and len(estimator.poses) == 1:
        estimator.poses[0] = pose.copy()
        estimator.kernel_cache = None
        pose_index = 0
    else:
        estimator.add_measurement_pose(pose, reset_filters=False)
        pose_index = len(estimator.poses) - 1
    return int(pose_index)


def _assimilate_station(
    estimator: object,
    records: Sequence[object],
    *,
    station_id: int,
    contract_hash: str,
) -> None:
    """Assimilate one durably staged, single-pose station block."""
    pose_index = _register_station_pose(
        estimator,
        records,
        station_id=station_id,
    )
    estimator.update_spectrum_station(
        tuple(measurement_record_to_spectrum_input(record) for record in records),
        pose_idx=pose_index,
        generative_contract_hash_sha256=contract_hash,
    )


def _plan(
    estimator: object,
    candidates: Mapping[str, object],
    *,
    current_pose: np.ndarray,
    visited_poses: Sequence[np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    room_bounds: tuple[np.ndarray, np.ndarray],
    height_bounds: tuple[float, float] | None,
    planner: DSSPPConfig,
    rng: np.random.Generator,
) -> DSSPPResult:
    """Rank only runtime-authored physical actions with PF-specific utility."""
    return select_dss_pp_next_station(
        estimator,
        np.asarray(candidates["candidate_poses_xyz"], dtype=np.float64),
        current_pose,
        current_pair_id=int(candidates["current_pair_id"]),
        visited_poses_xyz=np.asarray(visited_poses, dtype=np.float64),
        map_api=obstacle_grid,
        bounds_xyz=room_bounds,
        continuous_height_bounds_m=height_bounds,
        config=planner,
        rng=rng,
        candidate_motion_times_s=np.asarray(
            candidates["travel_costs"],
            dtype=np.float64,
        ),
    )


def _refine_and_replan(
    client: AdaptiveRuntimeClient,
    estimator: object,
    candidates: Mapping[str, object],
    initial: DSSPPResult,
    *,
    refinement_top_k: int,
    current_pose: np.ndarray,
    visited_poses: Sequence[np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    room_bounds: tuple[np.ndarray, np.ndarray],
    height_bounds: tuple[float, float] | None,
    planner: DSSPPConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, object], DSSPPResult]:
    """Optionally request runtime-owned local poses and rerank them exactly."""
    if refinement_top_k <= 0:
        return dict(candidates), initial
    ranked = initial.diagnostics.get("ranked_nodes", [])
    seed_indices: list[int] = []
    for node in ranked:
        if not isinstance(node, Mapping):
            continue
        index = candidate_index_for_pose(candidates, node["pose_xyz"])
        if index not in seed_indices:
            seed_indices.append(index)
        if len(seed_indices) >= refinement_top_k:
            break
    if not seed_indices:
        return dict(candidates), initial
    event = client.request({"type": "refine", "candidate_indices": seed_indices})
    _strict_fields(event, {"type", "candidates"}, name="refine event")
    if event.get("type") != "candidates":
        raise ValueError("Shared runtime did not return refined candidates.")
    refined = parse_candidate_snapshot(event["candidates"])
    result = _plan(
        estimator,
        refined,
        current_pose=current_pose,
        visited_poses=visited_poses,
        obstacle_grid=obstacle_grid,
        room_bounds=room_bounds,
        height_bounds=height_bounds,
        planner=planner,
        rng=rng,
    )
    return refined, result


def _write_final_outputs(
    output_dir: Path,
    *,
    estimator: object,
    log: object,
    result: PFClosedLoopResult,
    budget: PFControlBudget,
) -> None:
    """Publish the final posterior and controller provenance atomically per file."""
    posterior = estimator.posterior_snapshot().to_dict()
    diagnostics = {
        "schema_version": 1,
        "estimator_family": "pure_particle_filter",
        "measurement_log_sha256": log.log_sha256,
        "record_count": len(log.records),
        "station_count": result.station_count,
        "stop_reason": result.stop_reason,
        "control_budget": asdict(budget),
        "posterior_convergence": estimator.posterior_convergence_diagnostics(),
        "posterior_predictive_check": estimator.posterior_predictive_check(),
        "structural_transition_provenance": (
            estimator.structural_transition_diagnostics()
        ),
        "last_pf_step_diagnostics": estimator.step_diagnostics(
            top_k=0,
            include_estimates=False,
        ),
        "detected_isotope_gate": getattr(
            estimator,
            "detected_isotope_gate_diagnostics",
            None,
        ),
        "candidate_isotopes": list(
            getattr(estimator, "candidate_isotopes", estimator.isotopes)
        ),
        "active_isotopes": list(estimator.joint_isotope_order()),
    }
    (output_dir / "pf_posterior.json").write_bytes(canonical_json_bytes(posterior))
    (output_dir / "pf_diagnostics.json").write_bytes(canonical_json_bytes(diagnostics))
    (output_dir / "closed_loop_result.json").write_bytes(
        canonical_json_bytes(result.to_dict())
    )


def run_pf_closed_loop(
    scenario_path: str | Path,
    *,
    runtime_root: str | Path,
    pf_config_path: str | Path,
    output_dir: str | Path,
    profile: str = "pf_strict",
    seed: int = 0,
    private_scene_profile: str | None = None,
    output_hook: Any = print,
) -> PFClosedLoopResult:
    """Run a PF-specific closed loop over the common adaptive runtime API."""
    settings, config_hash = load_pf_config(pf_config_path)
    planner = dss_config_from_pf_settings(
        settings,
        runtime_owned_candidates=True,
    )
    budget = PFControlBudget.from_settings(settings, planner)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace PF output {target}.")
    target.mkdir(parents=True)
    planner_writer = PlannerAuditWriter(target / "planner_audit.jsonl")
    controller_trace = target / "pf_station_trace.jsonl"
    client = AdaptiveRuntimeClient(
        scenario_path,
        runtime_root=runtime_root,
        private_scene_profile=private_scene_profile,
        output_hook=output_hook,
    )
    estimator = None
    cui_split_viz: AsyncCUISplitPFVisualizer | None = None
    try:
        ready = client.read_event()
        _strict_fields(
            ready,
            {"type", "schema_version", "context", "candidates", "bootstrap"},
            name="ready event",
        )
        if ready.get("type") != "ready" or ready.get("schema_version") != 1:
            raise ValueError("Shared runtime returned an incompatible handshake.")
        context = parse_run_context(ready["context"])
        candidates = parse_candidate_snapshot(ready["candidates"])
        bootstrap = ready["bootstrap"]
        if not isinstance(bootstrap, Mapping):
            raise TypeError("Runtime bootstrap must be a mapping.")
        _strict_fields(
            bootstrap,
            {"candidate_index", "fe_orientation_index", "pb_orientation_index"},
            name="bootstrap",
        )
        detected_only_raw = settings.get(
            "pf_detected_isotopes_only",
            settings.get("detected_isotopes_only", False),
        )
        if not isinstance(detected_only_raw, bool):
            raise TypeError("pf_detected_isotopes_only must be a boolean.")
        initial_estimator_settings = dict(settings)
        if detected_only_raw:
            initial_estimator_settings["num_particles"] = 1
            initial_estimator_settings["variable_cardinality"] = False
            initial_estimator_settings["init_num_sources"] = (0, 0)
        estimator = build_live_estimator(
            context,
            initial_estimator_settings,
            profile=profile,
            seed=seed,
            runtime_root=runtime_root,
            config_hash=config_hash,
        )
        isotope_gate = (
            FullSpectrumIsotopeGate(
                candidate_isotopes=tuple(context.isotopes),
                false_activation_probability=float(
                    settings.get(
                        "detected_isotope_false_activation_probability",
                        1.0e-3,
                    )
                ),
            )
            if detected_only_raw
            else None
        )
        detection_estimator = estimator if isotope_gate is not None else None
        obstacle_grid = _obstacle_grid(context)
        room_bounds = _bounds(context)
        height_bounds = _height_bounds(context)
        cui_truth_mode = _cui_truth_display_mode(settings)
        cui_split_viz = _start_cui_split_view(
            settings,
            isotopes=getattr(context, "isotopes"),
            room_bounds=room_bounds,
            obstacle_grid=obstacle_grid,
            output_hook=output_hook,
        )
        if cui_split_viz is not None and cui_truth_mode == "evaluation_live":
            truth_sources, truth_strengths = _truth_arrays_from_cui_overlay(
                client.request_cui_overlay(include_truth=True)
            )
            cui_split_viz.set_truth(truth_sources, truth_strengths)
            output_hook(
                "CUI split visualization truth overlay: evaluation_live "
                "(private runtime CUI channel; not estimator/planner input)"
            )
        contract_hash = str(
            context.runtime_config["full_spectrum_contract_hash_sha256"]
        )
        planner_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0xD55A11])
        )
        current_program = _bootstrap_program(estimator, planner)
        current_pose = np.asarray(
            candidates["candidate_poses_xyz"][int(bootstrap["candidate_index"])],
            dtype=np.float64,
        )
        visited: list[np.ndarray] = []
        record_count = 0
        cui_elapsed_time_s = 0.0
        last_cui_record: object | None = None
        station_id = 0
        station_history: list[tuple[object, ...]] = []
        gate_diagnostics: dict[str, object] | None = None
        stop_reason = "maximum_station_budget"
        planner_writer.append(
            {
                "schema_version": 1,
                "station_id": 0,
                "selection_mode": "pf_prior_balanced_bootstrap",
                "selected_pose_xyz": current_pose.tolist(),
                "selected_program": {
                    "name": current_program.name,
                    "kind": current_program.kind,
                    "pair_ids": list(current_program.pair_ids),
                },
                "selected_score": None,
                "selected_information_gain": None,
                "best_exact_information_gain": None,
                "total_action_count": 0,
                "selected_proxy_rank": 0,
                "exact_action_count": 0,
                "proxy_action_count": 0,
                "planning_particle_count": 0,
                "score_leader": None,
                "information_gain_leader": None,
                "top_ranked_actions": [],
                "shortlist_certificate": {
                    "available": False,
                    "winner_exceeds_excluded_bound": False,
                    "evaluated_objective_lower_bound": None,
                    "excluded_objective_upper_bound": None,
                },
                "exact_eig_seed": None,
                "mc_seed_rank_stability": {
                    "status": "not_applicable_before_first_observation"
                },
            }
        )
        while station_id < budget.max_stations:
            if record_count + len(current_program.pair_ids) > budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                break
            station_records = []
            for view_index, pair_id in enumerate(current_program.pair_ids):
                candidate_index = candidate_index_for_pose(candidates, current_pose)
                fe_index, pb_index = divmod(int(pair_id), 8)
                event = client.request(
                    adaptive_step_request(
                        candidate_index=candidate_index,
                        fe_orientation_index=fe_index,
                        pb_orientation_index=pb_index,
                        dwell_time_s=budget.live_time_s,
                        station_id=station_id,
                        station_complete=(
                            view_index == len(current_program.pair_ids) - 1
                        ),
                    )
                )
                _strict_fields(
                    event,
                    {"type", "record", "candidates"},
                    name="record event",
                )
                if event.get("type") != "record":
                    raise ValueError("Shared runtime did not return a record event.")
                record = parse_adaptive_record(event["record"])
                if int(record.station_id) != station_id:
                    raise ValueError("Runtime record station_id changed unexpectedly.")
                station_records.append(record)
                candidates = parse_candidate_snapshot(event["candidates"])
                record_count += 1
                cui_elapsed_time_s += (
                    float(record.live_time_s)
                    + float(record.travel_time_s)
                    + float(record.shield_actuation_time_s)
                )
                if cui_split_viz is not None:
                    _publish_cui_frame(
                        cui_split_viz,
                        estimator,
                        record,
                        elapsed_time_s=cui_elapsed_time_s,
                        record_measurement=True,
                    )
                    last_cui_record = record
            if station_records[-1].metadata.get("station_complete") is not True:
                raise ValueError("Runtime omitted the final station marker.")
            station_history.append(tuple(station_records))
            assimilation_start_s = time.perf_counter()
            if isotope_gate is None:
                _assimilate_station(
                    estimator,
                    station_records,
                    station_id=station_id,
                    contract_hash=contract_hash,
                )
            else:
                assert detection_estimator is not None
                detection_pose_index = _register_station_pose(
                    detection_estimator,
                    station_records,
                    station_id=station_id,
                )
                score_grids = (
                    detection_estimator.full_spectrum_isotope_detection_score_grids(
                        tuple(
                            measurement_record_to_spectrum_input(record)
                            for record in station_records
                        ),
                        pose_idx=detection_pose_index,
                        generative_contract_hash_sha256=contract_hash,
                    )
                )
                gate_diagnostics = isotope_gate.update(score_grids)
                detection_estimator.detected_isotope_gate_diagnostics = gate_diagnostics
                active_isotopes = tuple(
                    isotope
                    for isotope in context.isotopes
                    if isotope in isotope_gate.active_isotopes
                )
                if gate_diagnostics["newly_active_isotopes"]:
                    estimator = build_live_estimator(
                        context,
                        settings,
                        profile=profile,
                        seed=seed,
                        runtime_root=runtime_root,
                        config_hash=config_hash,
                        inference_isotopes=active_isotopes,
                    )
                    for replay_station_id, replay_station in enumerate(station_history):
                        _assimilate_station(
                            estimator,
                            replay_station,
                            station_id=replay_station_id,
                            contract_hash=contract_hash,
                        )
                    estimator.detected_isotope_gate_diagnostics = gate_diagnostics
                    output_hook(
                        "Spectrum-detected isotope PF set active: "
                        f"{list(active_isotopes)}; rebuilt from "
                        f"{len(station_history)} truth-free station(s)."
                    )
                elif active_isotopes:
                    _assimilate_station(
                        estimator,
                        station_records,
                        station_id=station_id,
                        contract_hash=contract_hash,
                    )
                else:
                    output_hook(
                        "No isotope has crossed the truth-free full-spectrum "
                        "activation threshold; PF assimilation remains deferred."
                    )
                if active_isotopes:
                    estimator.detected_isotope_gate_diagnostics = gate_diagnostics
            assimilation_elapsed_s = time.perf_counter() - assimilation_start_s
            current_pose = np.asarray(
                station_records[-1].detector_pose_xyz,
                dtype=np.float64,
            )
            visited.append(current_pose.copy())
            posterior_snapshot = _live_posterior_summary(estimator)
            if cui_split_viz is not None:
                _publish_cui_frame(
                    cui_split_viz,
                    estimator,
                    station_records[-1],
                    elapsed_time_s=cui_elapsed_time_s,
                    record_measurement=False,
                )
                last_cui_record = station_records[-1]
            _append_jsonl(
                controller_trace,
                {
                    "schema_version": 1,
                    "station_id": station_id,
                    "record_count": record_count,
                    "pose_xyz": current_pose.tolist(),
                    "pair_ids": [int(value) for value in current_program.pair_ids],
                    "pf_update_elapsed_s": float(assimilation_elapsed_s),
                    "detected_isotope_gate": gate_diagnostics,
                    "particle_adequacy": _particle_diagnostics(estimator),
                    "posterior_snapshot": posterior_snapshot,
                },
            )
            completed_stations = station_id + 1
            if record_count >= budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                break
            if completed_stations >= budget.max_stations:
                stop_reason = "maximum_station_budget"
                break
            if (
                budget.adaptive_stop
                and completed_stations >= budget.minimum_stop_stations
                and estimator.posterior_convergence_diagnostics().get(
                    "ready",
                    False,
                )
            ):
                stop_reason = "intrinsic_surface_posterior_converged"
                break
            planned = _plan(
                estimator,
                candidates,
                current_pose=current_pose,
                visited_poses=visited,
                obstacle_grid=obstacle_grid,
                room_bounds=room_bounds,
                height_bounds=height_bounds,
                planner=planner,
                rng=planner_rng,
            )
            candidates, planned = _refine_and_replan(
                client,
                estimator,
                candidates,
                planned,
                refinement_top_k=budget.runtime_refinement_top_k,
                current_pose=current_pose,
                visited_poses=visited,
                obstacle_grid=obstacle_grid,
                room_bounds=room_bounds,
                height_bounds=height_bounds,
                planner=planner,
                rng=planner_rng,
            )
            station_id += 1
            planner_writer.append(
                build_planner_audit(
                    station_id=station_id,
                    result=planned,
                    top_k=budget.planner_audit_top_k,
                )
            )
            current_pose = np.asarray(planned.next_pose, dtype=np.float64)
            current_program = planned.shield_program
        if (
            cui_split_viz is not None
            and cui_truth_mode == "post_run"
            and last_cui_record is not None
        ):
            truth_sources, truth_strengths = _truth_arrays_from_cui_overlay(
                client.request_cui_overlay(include_truth=True)
            )
            cui_split_viz.set_truth(truth_sources, truth_strengths)
            _publish_cui_frame(
                cui_split_viz,
                estimator,
                last_cui_record,
                elapsed_time_s=cui_elapsed_time_s,
                record_measurement=False,
            )
            output_hook(
                "CUI split visualization truth overlay: post_run "
                "(private runtime CUI channel; not estimator/planner input)"
            )
        published = client.finalize()
        _strict_fields(
            published,
            {"type", "path", "record_count"},
            name="published event",
        )
        if published.get("type") != "published":
            raise ValueError("Shared runtime did not publish MeasurementLog v2.")
        log = load_measurement_log(published["path"])
        if int(published["record_count"]) != len(log.records):
            raise RuntimeError("Published MeasurementLog record count is inconsistent.")
        if isotope_gate is not None and not isotope_gate.active_isotopes:
            raise RuntimeError(
                "No candidate isotope crossed the truth-free full-spectrum "
                "activation threshold before the acquisition budget ended."
            )
        bind_finalized_measurement_log(estimator, log)
        result = PFClosedLoopResult(
            measurement_log_path=log.path.resolve(),
            pf_output_dir=target,
            run_id=log.run_id,
            record_count=len(log.records),
            station_count=len({record.station_id for record in log.records}),
            stop_reason=stop_reason,
        )
        _write_final_outputs(
            target,
            estimator=estimator,
            log=log,
            result=result,
            budget=budget,
        )
        return result
    except BaseException:
        client.abort()
        raise
    finally:
        if cui_split_viz is not None:
            cui_split_viz.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public PF adaptive-controller command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--private-scene-profile",
        choices=("ral-mix9", "ral-cs4-co3-eu0"),
        default=None,
    )
    args = parser.parse_args(None if argv is None else list(argv))
    result = run_pf_closed_loop(
        args.scenario,
        runtime_root=args.runtime_root,
        pf_config_path=args.config,
        output_dir=args.output_dir,
        profile=args.profile,
        seed=args.seed,
        private_scene_profile=args.private_scene_profile,
    )
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PFClosedLoopResult",
    "PFControlBudget",
    "run_pf_closed_loop",
]
