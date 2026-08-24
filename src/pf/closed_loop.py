"""PF-owned control of estimator-neutral adaptive acquisition."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from measurement.obstacles import ObstacleGrid
from runtime.adaptive_client import (
    AdaptiveCandidateSnapshot,
    AdaptiveRefineRequest,
    AdaptiveRuntimeClient,
    AdaptiveStepRequest,
    candidate_index_for_pose,
)
from runtime.cui import CUIRoute, CUI_URL_MESSAGE_PREFIX, cui_route_from_records
from runtime.cui_components import CUITruthDisplayMode
from runtime.artifacts import DurableJSONLWriter
from runtime.experiment_profiles import (
    AcquisitionContract,
    acquisition_contract_from_environment,
)
from runtime.measurement_log import (
    MeasurementLogRecord,
    MeasurementLogView,
    load_measurement_log,
)
from runtime.provenance import canonical_json_bytes

from pf.atomic_io import atomic_write_bytes
from pf.control_policy import PFControlPolicy, validate_control_policy
from pf.cui_runtime import (
    ensure_cui_view_server,
    resolve_cui_split_view_enabled,
)
from pf.configuration import load_pf_config
from pf.isotope_gate import FullSpectrumIsotopeGate
from pf.live_session import (
    assimilate_persisted_station,
    bind_published_measurement_log,
    build_live_estimator,
    live_posterior_summary,
    measurement_record_to_station_input,
    register_persisted_station_pose,
)
from pf.live_resume import reconstruct_live_resume_state
from pf.runtime_defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)
from planning.audit import PlannerAuditWriter, build_planner_audit
from planning.configuration import dss_config_from_pf_settings
from planning.bootstrap_program import build_balanced_bootstrap_program
from planning.dss_pp import DSSPPConfig, DSSPPResult, select_dss_pp_next_station
from planning.program_types import ShieldProgram
from visualization.artifacts import publish_final_cui_split_views
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


@dataclass(frozen=True, slots=True)
class PFControlBudget:
    """Combine runtime acquisition limits with PF-owned stopping controls."""

    max_stations: int
    max_measurements: int
    views_per_station: int
    live_time_s: float
    adaptive_stop_enabled: bool
    stop_assessment_start_station: int
    stop_required_consecutive_stations: int
    runtime_refinement_top_k: int
    planner_audit_top_k: int

    @classmethod
    def from_runtime_contract(
        cls,
        settings: Mapping[str, Any],
        planner: DSSPPConfig,
        acquisition_contract: AcquisitionContract,
    ) -> "PFControlBudget":
        """Resolve runtime limits without accepting estimator-side overrides."""
        configured_views = int(acquisition_contract.views_per_station)
        if configured_views != int(planner.program_length):
            raise ValueError(
                "Runtime views_per_station and planner program_length must agree."
            )
        if acquisition_contract.max_measurements < configured_views:
            raise ValueError(
                "Runtime max_measurements must accommodate one complete station."
            )
        adaptive_stop = settings.get("adaptive_stop", {})
        if not isinstance(adaptive_stop, Mapping):
            raise TypeError("adaptive_stop must be an object.")
        adaptive_stop_enabled = adaptive_stop.get("enabled", False)
        if not isinstance(adaptive_stop_enabled, bool):
            raise TypeError("adaptive_stop.enabled must be a boolean.")
        assessment_start = _exact_integer(
            adaptive_stop.get("assessment_start_station", 10),
            name="adaptive_stop.assessment_start_station",
            minimum=1,
        )
        required_consecutive = _exact_integer(
            adaptive_stop.get("required_consecutive_stations", 3),
            name="adaptive_stop.required_consecutive_stations",
            minimum=1,
        )
        earliest_stop_station = assessment_start + required_consecutive - 1
        if adaptive_stop_enabled and earliest_stop_station > int(
            acquisition_contract.max_stations
        ):
            raise ValueError(
                "adaptive_stop cannot accumulate its required consecutive "
                "stations before the runtime max_stations limit."
            )
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
            max_stations=acquisition_contract.max_stations,
            max_measurements=acquisition_contract.max_measurements,
            views_per_station=configured_views,
            live_time_s=acquisition_contract.live_time_s,
            adaptive_stop_enabled=adaptive_stop_enabled,
            stop_assessment_start_station=assessment_start,
            stop_required_consecutive_stations=required_consecutive,
            runtime_refinement_top_k=refinement,
            planner_audit_top_k=audit_top_k,
        )

    @property
    def earliest_adaptive_stop_station(self) -> int:
        """Return the first station where a complete ready streak can stop."""
        return (
            self.stop_assessment_start_station
            + self.stop_required_consecutive_stations
            - 1
        )


@dataclass(slots=True)
class AdaptiveStopTracker:
    """Track consecutive model-native stop decisions across station updates."""

    budget: PFControlBudget
    consecutive_ready_stations: int = 0
    last_station_count: int = 0

    def assess(
        self,
        estimator: object,
        *,
        station_count: int,
    ) -> dict[str, object]:
        """Assess one new station and return a traceable stopping decision."""
        count = _exact_integer(station_count, name="station_count", minimum=1)
        if count != self.last_station_count + 1:
            raise ValueError(
                "Adaptive-stop stations must be assessed once in consecutive order."
            )
        self.last_station_count = count
        enabled = bool(self.budget.adaptive_stop_enabled)
        eligible = bool(
            enabled and count >= self.budget.stop_assessment_start_station
        )
        diagnostics: dict[str, Any] | None = None
        instantaneous_ready: bool | None = None
        if eligible:
            raw_diagnostics = estimator.posterior_convergence_diagnostics()
            if not isinstance(raw_diagnostics, Mapping):
                raise TypeError(
                    "posterior_convergence_diagnostics must return a mapping."
                )
            diagnostics = dict(raw_diagnostics)
            raw_ready = diagnostics.get("ready")
            if not isinstance(raw_ready, bool):
                raise TypeError(
                    "posterior convergence ready must be a boolean."
                )
            instantaneous_ready = raw_ready
            if instantaneous_ready:
                self.consecutive_ready_stations += 1
            else:
                self.consecutive_ready_stations = 0
        else:
            self.consecutive_ready_stations = 0
        stop_ready = bool(
            eligible
            and self.consecutive_ready_stations
            >= self.budget.stop_required_consecutive_stations
        )
        return {
            "enabled": enabled,
            "assessed": eligible,
            "assessment_start_station": (
                self.budget.stop_assessment_start_station
            ),
            "required_consecutive_stations": (
                self.budget.stop_required_consecutive_stations
            ),
            "earliest_stop_station": (
                self.budget.earliest_adaptive_stop_station
            ),
            "instantaneous_ready": instantaneous_ready,
            "consecutive_ready_stations": self.consecutive_ready_stations,
            "stop_ready": stop_ready,
            "posterior_convergence": diagnostics,
        }


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
    value = settings.get("cui_split_view_host", DEFAULT_CUI_SPLIT_VIEW_HOST)
    if not isinstance(value, str) or not value.strip():
        raise TypeError("cui_split_view_host must be a nonempty string.")
    return value


def _strict_cui_port(settings: Mapping[str, object]) -> int:
    """Return a valid TCP port for the CUI server."""
    value = settings.get("cui_split_view_port", DEFAULT_CUI_SPLIT_VIEW_PORT)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cui_split_view_port must be an integer.")
    if value < 1 or value > 65535:
        raise ValueError("cui_split_view_port must be between 1 and 65535.")
    return int(value)


def _cui_truth_display_mode(settings: Mapping[str, object]) -> str:
    """Require a truth-free CUI inside the estimator-owned controller."""
    value = settings.get("cui_truth_display_mode", "hidden")
    if not isinstance(value, str):
        raise TypeError("cui_truth_display_mode must be a string.")
    try:
        mode = CUITruthDisplayMode(value.strip())
    except ValueError as exc:
        raise ValueError(
            "cui_truth_display_mode must be hidden, evaluation_live, or post_run."
        ) from exc
    if mode is not CUITruthDisplayMode.HIDDEN:
        raise ValueError(
            "PF closed-loop CUI must keep truth hidden; truth overlays belong to "
            "a separate post-estimation evaluator."
        )
    return mode.value


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
        "(latest_experiment_overview.png, latest_robot_2d.png, "
        "latest_pf_3d.png, latest_pf_3d_labeled.png, latest_spectrum.png)"
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
        output_hook(f"{CUI_URL_MESSAGE_PREFIX} {split_url}")
    return visualizer


def _publish_cui_frame(
    visualizer: AsyncCUISplitPFVisualizer,
    estimator: object,
    record: MeasurementLogRecord,
    route_records: list[MeasurementLogRecord],
    *,
    elapsed_time_s: float,
    record_measurement: bool,
) -> None:
    """Queue one truth-free PF CUI frame without changing PF state."""
    if record_measurement:
        route: CUIRoute = cui_route_from_records((*route_records, record))
        route_records.append(record)
    else:
        if not route_records or route_records[-1].step_id != record.step_id:
            raise ValueError(
                "Posterior-only CUI redraw must reference the latest routed record."
            )
        route = cui_route_from_records(route_records)
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
    if route.travel_path_segments_xyz:
        frame.path_waypoints_xyz = route.travel_path_segments_xyz[-1].copy()
    frame.cui_route = route
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
            "2k/4k/8k live-session stability."
        ),
    }
    return {"assessment": evidence, "isotopes": isotopes}


def _live_posterior_summary(estimator: object) -> dict[str, object]:
    """Return a truth-free, explicitly non-publishable station summary.

    A publishable ``PFPosteriorSnapshot`` is intentionally unavailable until
    the runtime finalizes MeasurementLog and supplies its immutable digest.
    The live controller therefore serializes only the current PF point
    estimates; final provenance remains exclusive to ``pf_posterior.json``.
    """
    return live_posterior_summary(estimator)


def _external_shield_program(
    estimator: object,
    planner: DSSPPConfig,
    control_policy: PFControlPolicy | None,
    *,
    pose_index: int,
    current_pair_id: int | None,
) -> ShieldProgram | None:
    """Resolve one injected shield program without importing experiment code."""
    if control_policy is None:
        return None
    return control_policy.select_shield_program(
        total_pairs=int(len(estimator.normals) ** 2),
        program_length=int(planner.program_length),
        pose_index=pose_index,
        current_pair_id=current_pair_id,
    )


def _bootstrap_program(
    estimator: object,
    planner: DSSPPConfig,
    control_policy: PFControlPolicy | None,
) -> ShieldProgram:
    """Choose an injected or balanced first-station shield program."""
    baseline = _external_shield_program(
        estimator,
        planner,
        control_policy,
        pose_index=0,
        current_pair_id=None,
    )
    if baseline is not None:
        return baseline
    return build_balanced_bootstrap_program(
        num_orientations=int(len(estimator.normals)),
        program_length=int(planner.program_length),
    )


def _register_station_pose(
    estimator: object,
    records: Sequence[object],
    *,
    station_id: int,
) -> int:
    """Register one single-pose station and return its estimator pose index."""
    return register_persisted_station_pose(
        estimator,
        records,
        station_id=station_id,
    )


def _assimilate_station(
    estimator: object,
    records: Sequence[object],
    *,
    station_id: int,
    contract_hash: str,
) -> None:
    """Assimilate one durably staged, single-pose station block."""
    assimilate_persisted_station(
        estimator,
        records,
        station_id=station_id,
        generative_contract_hash_sha256=contract_hash,
    )


def _plan(
    estimator: object,
    candidates: AdaptiveCandidateSnapshot,
    *,
    current_pose: np.ndarray,
    visited_poses: Sequence[np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    room_bounds: tuple[np.ndarray, np.ndarray],
    height_bounds: tuple[float, float] | None,
    planner: DSSPPConfig,
    rng: np.random.Generator,
    settings: Mapping[str, Any],
    station_index: int,
    control_policy: PFControlPolicy | None,
) -> DSSPPResult:
    """Rank runtime actions under the standard or injected control policy."""
    candidate_poses = np.asarray(
        candidates.candidate_poses_xyz,
        dtype=np.float64,
    )
    visited_array = np.asarray(visited_poses, dtype=np.float64)
    if visited_array.size == 0:
        visited_array = visited_array.reshape((0, 3))
    baseline_program = _external_shield_program(
        estimator,
        planner,
        control_policy,
        pose_index=station_index,
        current_pair_id=candidates.current_pair_id,
    )
    baseline_path = (
        None
        if control_policy is None
        else control_policy.select_path(
            candidate_poses_xyz=candidate_poses,
            current_pose_xyz=current_pose,
            visited_poses_xyz=visited_array,
            bounds_xyz=room_bounds,
        )
    )
    if baseline_path is not None:
        if baseline_program is None:
            raise ValueError(
                "A baseline path policy requires an explicit baseline shield policy."
            )
        return DSSPPResult(
            next_pose=baseline_path.next_pose,
            next_pose_index=baseline_path.candidate_index,
            shield_program=baseline_program,
            score=baseline_path.score,
            sequence=(),
            diagnostics={
                "selection_mode": "external_control_path",
                "external_path_policy": baseline_path.policy_name,
                "planning_particle_count": 0,
                "ranked_nodes": [],
                "component_leaders": {},
                "planning_eig_shortlist": {},
            },
        )
    active_planner = planner
    if baseline_program is not None:
        active_planner = replace(
            planner,
            forced_program_pair_ids=baseline_program.pair_ids,
        )
    return select_dss_pp_next_station(
        estimator,
        candidate_poses,
        current_pose,
        current_pair_id=candidates.current_pair_id,
        visited_poses_xyz=np.asarray(visited_poses, dtype=np.float64),
        map_api=obstacle_grid,
        bounds_xyz=room_bounds,
        continuous_height_bounds_m=height_bounds,
        config=active_planner,
        rng=rng,
        candidate_motion_times_s=np.asarray(
            candidates.travel_costs,
            dtype=np.float64,
        ),
        candidate_horizontal_travel_times_s=np.asarray(
            candidates.horizontal_travel_times_s,
            dtype=np.float64,
        )
        if candidates.has_motion_time_components
        else None,
        candidate_mast_vertical_times_s=np.asarray(
            candidates.mast_vertical_times_s,
            dtype=np.float64,
        )
        if candidates.has_motion_time_components
        else None,
        candidate_settling_times_s=np.asarray(
            candidates.settling_times_s,
            dtype=np.float64,
        )
        if candidates.has_motion_time_components
        else None,
    )


def _planner_rng(seed: int, station_index: int) -> np.random.Generator:
    """Return a station-addressed planner stream that supports exact resume."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xD55A11, int(station_index)])
    )


def _refine_and_replan(
    client: AdaptiveRuntimeClient,
    estimator: object,
    candidates: AdaptiveCandidateSnapshot,
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
    settings: Mapping[str, Any],
    station_index: int,
    control_policy: PFControlPolicy | None,
) -> tuple[AdaptiveCandidateSnapshot, DSSPPResult]:
    """Optionally request runtime-owned local poses and rerank them exactly."""
    if refinement_top_k <= 0 or (
        control_policy is not None and control_policy.has_fixed_path
    ):
        return candidates, initial
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
        return candidates, initial
    event = client.refine_candidates(AdaptiveRefineRequest.from_indices(seed_indices))
    refined = event.candidates
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
        settings=settings,
        station_index=station_index,
        control_policy=control_policy,
    )
    return refined, result


def _write_final_outputs(
    output_dir: Path,
    *,
    estimator: object,
    log: object,
    result: PFClosedLoopResult,
    budget: PFControlBudget,
    adaptive_stop_status: Mapping[str, object] | None,
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
        "adaptive_stop_status": (
            None if adaptive_stop_status is None else dict(adaptive_stop_status)
        ),
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
    atomic_write_bytes(
        output_dir / "pf_posterior.json",
        canonical_json_bytes(posterior),
    )
    atomic_write_bytes(
        output_dir / "pf_diagnostics.json",
        canonical_json_bytes(diagnostics),
    )
    atomic_write_bytes(
        output_dir / "closed_loop_result.json",
        canonical_json_bytes(result.to_dict()),
    )


def run_pf_closed_loop(
    session_socket: str | Path,
    *,
    runtime_root: str | Path,
    pf_config_path: str | Path,
    output_dir: str | Path,
    profile: str = "pf_strict",
    seed: int = 0,
    control_policy: PFControlPolicy | None = None,
    output_hook: Any = print,
) -> PFClosedLoopResult:
    """Run a PF closed loop over an opaque truth-free runtime session socket."""
    validate_control_policy(control_policy)
    settings, config_hash = load_pf_config(pf_config_path)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace PF output {target}.")
    target.mkdir(parents=True)
    resources = ExitStack()
    client: AdaptiveRuntimeClient | None = None
    estimator = None
    cui_split_viz: AsyncCUISplitPFVisualizer | None = None
    completed_result: PFClosedLoopResult | None = None
    cui_frame_rendered = False
    cui_route_records: list[MeasurementLogRecord] = []
    try:
        planner_writer = PlannerAuditWriter(target / "planner_audit.jsonl")
        resources.callback(planner_writer.close)
        controller_writer = DurableJSONLWriter(
            target / "pf_station_trace.jsonl",
            mode=0o644,
        )
        resources.callback(controller_writer.close)
        client = AdaptiveRuntimeClient.connect(
            session_socket,
            output_hook=output_hook,
        )
        resources.callback(client.close)
        ready = client.handshake()
        schema_version = ready.schema_version
        context = ready.context
        candidates = ready.candidates
        bootstrap = ready.bootstrap
        acquisition_contract = acquisition_contract_from_environment(
            context.environment
        )
        planner = dss_config_from_pf_settings(
            settings,
            acquisition_contract=acquisition_contract,
        )
        budget = PFControlBudget.from_runtime_contract(
            settings,
            planner,
            acquisition_contract,
        )
        stop_tracker = AdaptiveStopTracker(budget)
        latest_adaptive_stop_status: dict[str, object] | None = None
        detected_only_raw = settings.get(
            "pf_detected_isotopes_only",
            settings.get("detected_isotopes_only", False),
        )
        if not isinstance(detected_only_raw, bool):
            raise TypeError("pf_detected_isotopes_only must be a boolean.")
        if schema_version == 2 and detected_only_raw:
            raise ValueError(
                "Adaptive resume currently requires pf_detected_isotopes_only=false."
            )
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
        _cui_truth_display_mode(settings)
        cui_split_viz = _start_cui_split_view(
            settings,
            isotopes=getattr(context, "isotopes"),
            room_bounds=room_bounds,
            obstacle_grid=obstacle_grid,
            output_hook=output_hook,
        )
        contract_hash = str(
            context.runtime_config["full_spectrum_contract_hash_sha256"]
        )
        current_program = _bootstrap_program(
            estimator,
            planner,
            control_policy,
        )
        visited: list[np.ndarray]
        station_history: list[tuple[MeasurementLogRecord, ...]]
        gate_diagnostics: dict[str, object] | None = None
        stop_reason = "maximum_station_budget"
        continue_acquisition = True
        if schema_version == 1:
            assert bootstrap is not None
            current_pose = np.asarray(
                candidates.candidate_poses_xyz[bootstrap.candidate_index],
                dtype=np.float64,
            )
            visited = []
            record_count = 0
            cui_elapsed_time_s = 0.0
            station_id = 0
            station_history = []
            planner_writer.append(
                {
                    "schema_version": 1,
                    "station_id": 0,
                    "selection_mode": (
                        "external_control_bootstrap"
                        if current_program.kind == "external_control"
                        else "pf_prior_balanced_bootstrap"
                    ),
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
        else:
            prefix = ready.resume
            assert prefix is not None
            resume_station_view = MeasurementLogView.from_records(
                context,
                prefix.records,
            ).station_view()
            cui_route_records.extend(resume_station_view.records)
            resume_state = reconstruct_live_resume_state(
                resume_station_view,
                next_station_id=prefix.next_station_id,
                expected_views_per_station=budget.views_per_station,
            )
            station_history = [tuple(station) for station in resume_state.stations]
            for prefix_station_id, prefix_station in enumerate(station_history):
                _assimilate_station(
                    estimator,
                    prefix_station,
                    station_id=prefix_station_id,
                    contract_hash=contract_hash,
                )
                latest_adaptive_stop_status = stop_tracker.assess(
                    estimator,
                    station_count=prefix_station_id + 1,
                )
            if cui_split_viz is not None:
                _publish_cui_frame(
                    cui_split_viz,
                    estimator,
                    resume_station_view.records[-1],
                    cui_route_records,
                    elapsed_time_s=resume_state.elapsed_time_s,
                    record_measurement=False,
                )
                cui_frame_rendered = True
            current_pose = resume_state.current_pose.copy()
            visited = [pose.copy() for pose in resume_state.visited_poses]
            record_count = resume_state.record_count
            cui_elapsed_time_s = resume_state.elapsed_time_s
            station_id = resume_state.next_station_id
            if record_count >= budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                continue_acquisition = False
            elif station_id >= budget.max_stations:
                stop_reason = "maximum_station_budget"
                continue_acquisition = False
            elif bool(
                latest_adaptive_stop_status
                and latest_adaptive_stop_status["stop_ready"]
            ):
                stop_reason = "intrinsic_surface_posterior_converged"
                continue_acquisition = False
            if continue_acquisition:
                resume_rng = _planner_rng(seed, station_id)
                planned = _plan(
                    estimator,
                    candidates,
                    current_pose=current_pose,
                    visited_poses=visited,
                    obstacle_grid=obstacle_grid,
                    room_bounds=room_bounds,
                    height_bounds=height_bounds,
                    planner=planner,
                    rng=resume_rng,
                    settings=settings,
                    station_index=station_id,
                    control_policy=control_policy,
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
                    rng=resume_rng,
                    settings=settings,
                    station_index=station_id,
                    control_policy=control_policy,
                )
                planner_writer.append(
                    build_planner_audit(
                        station_id=station_id,
                        result=planned,
                        top_k=budget.planner_audit_top_k,
                    )
                )
                current_pose = np.asarray(planned.next_pose, dtype=np.float64)
                current_program = planned.shield_program
            output_hook(
                "Resumed PF from "
                f"{record_count} truth-free records at station {station_id}."
            )
        while continue_acquisition and station_id < budget.max_stations:
            if record_count + len(current_program.pair_ids) > budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                break
            station_records = []
            for view_index, pair_id in enumerate(current_program.pair_ids):
                candidate_index = candidate_index_for_pose(candidates, current_pose)
                fe_index, pb_index = divmod(int(pair_id), 8)
                event = client.acquire(
                    AdaptiveStepRequest(
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
                record = event.record
                if int(record.station_id) != station_id:
                    raise ValueError("Runtime record station_id changed unexpectedly.")
                station_records.append(record)
                candidates = event.candidates
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
                        cui_route_records,
                        elapsed_time_s=cui_elapsed_time_s,
                        record_measurement=True,
                    )
                    cui_frame_rendered = True
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
                            measurement_record_to_station_input(record)
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
                    for prior_station_id, prior_station in enumerate(station_history):
                        _assimilate_station(
                            estimator,
                            prior_station,
                            station_id=prior_station_id,
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
            completed_stations = station_id + 1
            latest_adaptive_stop_status = stop_tracker.assess(
                estimator,
                station_count=completed_stations,
            )
            posterior_snapshot = _live_posterior_summary(estimator)
            if cui_split_viz is not None:
                _publish_cui_frame(
                    cui_split_viz,
                    estimator,
                    station_records[-1],
                    cui_route_records,
                    elapsed_time_s=cui_elapsed_time_s,
                    record_measurement=False,
                )
                cui_frame_rendered = True
            controller_writer.append(
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
                    "adaptive_stop": latest_adaptive_stop_status,
                }
            )
            if record_count >= budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                break
            if completed_stations >= budget.max_stations:
                stop_reason = "maximum_station_budget"
                break
            if latest_adaptive_stop_status["stop_ready"]:
                stop_reason = "intrinsic_surface_posterior_converged"
                break
            station_planner_rng = _planner_rng(seed, completed_stations)
            planned = _plan(
                estimator,
                candidates,
                current_pose=current_pose,
                visited_poses=visited,
                obstacle_grid=obstacle_grid,
                room_bounds=room_bounds,
                height_bounds=height_bounds,
                planner=planner,
                rng=station_planner_rng,
                settings=settings,
                station_index=completed_stations,
                control_policy=control_policy,
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
                rng=station_planner_rng,
                settings=settings,
                station_index=completed_stations,
                control_policy=control_policy,
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
        published = client.finalize_log()
        log = load_measurement_log(published.path)
        if published.record_count != len(log.records):
            raise RuntimeError("Published MeasurementLog record count is inconsistent.")
        if isotope_gate is not None and not isotope_gate.active_isotopes:
            raise RuntimeError(
                "No candidate isotope crossed the truth-free full-spectrum "
                "activation threshold before the acquisition budget ended."
            )
        live_records = tuple(
            record for station_records in station_history for record in station_records
        )
        bind_published_measurement_log(
            estimator,
            log,
            live_records=live_records,
        )
        result = PFClosedLoopResult(
            measurement_log_path=log.path.resolve(),
            pf_output_dir=target,
            run_id=log.run_id,
            record_count=len(log.records),
            station_count=log.station_view().station_count,
            stop_reason=stop_reason,
        )
        _write_final_outputs(
            target,
            estimator=estimator,
            log=log,
            result=result,
            budget=budget,
            adaptive_stop_status=latest_adaptive_stop_status,
        )
        completed_result = result
        return result
    except BaseException:
        if client is not None:
            client.abort()
        raise
    finally:
        try:
            if cui_split_viz is not None:
                cui_split_viz.close()
                artifact_paths = (
                    getattr(cui_split_viz, "latest_overview_path", None),
                    getattr(cui_split_viz, "latest_robot_path", None),
                    getattr(cui_split_viz, "latest_pf_path", None),
                    getattr(cui_split_viz, "latest_pf_labeled_path", None),
                    getattr(cui_split_viz, "latest_spectrum_path", None),
                )
                if (
                    completed_result is not None
                    and cui_frame_rendered
                    and all(isinstance(path, Path) for path in artifact_paths)
                ):
                    publish_final_cui_split_views(
                        source_overview_path=artifact_paths[0],
                        source_robot_path=artifact_paths[1],
                        source_pf_path=artifact_paths[2],
                        source_pf_labeled_path=artifact_paths[3],
                        source_spectrum_path=artifact_paths[4],
                        final_overview_path=(target / "final_experiment_overview.png"),
                        final_robot_path=target / "final_robot_2d.png",
                        final_pf_path=target / "final_pf_3d.png",
                        final_pf_labeled_path=target / "final_pf_3d_labeled.png",
                        final_spectrum_path=target / "final_spectrum.png",
                    )
        finally:
            resources.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public PF adaptive-controller command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-socket", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(None if argv is None else list(argv))
    result = run_pf_closed_loop(
        args.session_socket,
        runtime_root=args.runtime_root,
        pf_config_path=args.config,
        output_dir=args.output_dir,
        profile=args.profile,
        seed=args.seed,
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
