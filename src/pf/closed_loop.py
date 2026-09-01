"""PF-owned control of estimator-neutral adaptive acquisition."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from collections.abc import Callable, Mapping, Sequence
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
from runtime.cui import (
    CUIRoute,
    CUIServerHandle,
    CUI_URL_MESSAGE_PREFIX,
    cui_route_from_records,
)
from runtime.artifacts import (
    AtomicBundlePublisher,
    DurableJSONLWriter,
    atomic_write_bytes,
)
from runtime.experiment_profiles import (
    AcquisitionContract,
    acquisition_contract_from_environment,
)
from runtime.measurement_log import (
    MeasurementLogRecord,
    load_measurement_log,
)
from pf.cardinality_policy import HARD_CAP_POSTERIOR_MASS_LIMIT
from pf.control_policy import validate_control_policy
from pf.cui_runtime import (
    resolve_cui_split_view_enabled,
    start_cui_view_server,
)
from pf.live_session import (
    PFLiveSession,
    _strict_live_artifact_json_bytes,
    live_posterior_summary,
    load_production_live_pf_config,
)
from pf.gpu_utils import preflight_compute_backend
from pf.profiles import production_compute_backend_values
from planning.audit import (
    PlannerAuditWriter,
    SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES,
    build_bootstrap_planner_audit,
    build_planner_audit,
)
from planning.configuration import dss_config_from_pf_settings
from planning.bootstrap_program import build_balanced_bootstrap_program
from planning.dss_pp import DSSPPConfig, DSSPPResult, select_dss_pp_next_station
from planning.program_types import ShieldProgram
from visualization.artifacts import publish_final_cui_split_views
from visualization.frame import PFFrame
from visualization.realtime_viz import (
    AsyncCUISplitPFVisualizer,
    build_frame_from_pf,
)


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
    stop_assessment_start_station: int
    stop_required_consecutive_stations: int
    runtime_refinement_top_k: int
    planner_audit_top_k: int

    @classmethod
    def from_runtime_contract(
        cls,
        settings: Mapping[str, Any],
        acquisition_contract: AcquisitionContract,
    ) -> "PFControlBudget":
        """Resolve runtime limits without accepting estimator-side overrides."""
        configured_views = int(acquisition_contract.views_per_station)
        if acquisition_contract.max_measurements < configured_views:
            raise ValueError(
                "Runtime max_measurements must accommodate one complete station."
            )
        adaptive_stop = settings["adaptive_stop"]
        if not isinstance(adaptive_stop, Mapping):
            raise TypeError("adaptive_stop must be an object.")
        assessment_start = _exact_integer(
            adaptive_stop["assessment_start_station"],
            name="adaptive_stop.assessment_start_station",
            minimum=1,
        )
        required_consecutive = _exact_integer(
            adaptive_stop["required_consecutive_stations"],
            name="adaptive_stop.required_consecutive_stations",
            minimum=1,
        )
        earliest_stop_station = assessment_start + required_consecutive - 1
        reachable_station_count = min(
            int(acquisition_contract.max_stations),
            int(acquisition_contract.max_measurements) // configured_views,
        )
        if earliest_stop_station > reachable_station_count:
            raise ValueError(
                "adaptive_stop cannot accumulate its required consecutive "
                "stations before the runtime station/measurement budgets."
            )
        refinement = _exact_integer(
            settings["runtime_candidate_refinement_top_k"],
            name="runtime_candidate_refinement_top_k",
            minimum=0,
        )
        audit_top_k = _exact_integer(
            settings["planner_audit_top_k"],
            name="planner_audit_top_k",
            minimum=0,
        )
        return cls(
            max_stations=acquisition_contract.max_stations,
            max_measurements=acquisition_contract.max_measurements,
            views_per_station=configured_views,
            live_time_s=acquisition_contract.live_time_s,
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


PFPlannerConfig = DSSPPConfig


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
        eligible = bool(count >= self.budget.stop_assessment_start_station)
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
                raise TypeError("posterior convergence ready must be a boolean.")
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
            "assessed": eligible,
            "assessment_start_station": (self.budget.stop_assessment_start_station),
            "required_consecutive_stations": (
                self.budget.stop_required_consecutive_stations
            ),
            "earliest_stop_station": (self.budget.earliest_adaptive_stop_station),
            "instantaneous_ready": instantaneous_ready,
            "consecutive_ready_stations": self.consecutive_ready_stations,
            "stop_ready": stop_ready,
            "posterior_convergence": diagnostics,
        }


def _compact_adaptive_stop_status(
    status: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return only station-varying adaptive-stop decision fields."""
    if status is None:
        return None
    compact = {
        "assessed": status.get("assessed"),
        "instantaneous_ready": status.get("instantaneous_ready"),
        "consecutive_ready_stations": status.get("consecutive_ready_stations"),
        "stop_ready": status.get("stop_ready"),
    }
    retained = {key: value for key, value in compact.items() if value is not None}
    return retained or None


@dataclass(frozen=True, slots=True)
class PFClosedLoopResult:
    """Describe one completed PF-controlled acquisition and posterior."""

    measurement_log_path: Path
    pf_output_dir: Path
    run_id: str
    record_count: int
    station_count: int
    stop_reason: str
    sampler_quality_status: str

    def to_dict(self) -> dict[str, object]:
        """Return one strict JSON-safe result payload."""
        return {
            "schema_version": 2,
            "execution_status": "complete",
            "sampler_quality_status": self.sampler_quality_status,
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
    if not isinstance(raw, Mapping):
        raise TypeError("context.environment.obstacle_grid must be an object.")
    return ObstacleGrid.from_dict(dict(raw))


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


def _require_candidate_anchor(
    candidates: AdaptiveCandidateSnapshot,
    current_pose: np.ndarray,
) -> None:
    """Require runtime motion costs to be anchored to the controller pose."""
    pose = np.asarray(current_pose, dtype=np.float64)
    if pose.shape != (3,) or np.any(~np.isfinite(pose)):
        raise ValueError("Controller current pose must be one finite XYZ row.")
    if tuple(candidates.current_pose_xyz) != tuple(pose.tolist()):
        raise RuntimeError(
            "Runtime candidate motion costs are anchored to another current pose."
        )


def _strict_cui_bool(settings: Mapping[str, object], key: str) -> bool:
    """Return one boolean CUI setting without accepting truthy substitutes."""
    value = settings[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean.")
    return value


def _strict_cui_host(settings: Mapping[str, object]) -> str:
    """Return the configured network bind host for the CUI server."""
    value = settings["cui_split_view_host"]
    if not isinstance(value, str) or not value.strip():
        raise TypeError("cui_split_view_host must be a nonempty string.")
    return value


def _strict_cui_port(settings: Mapping[str, object]) -> int:
    """Return a valid TCP port for the CUI server."""
    value = settings["cui_split_view_port"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cui_split_view_port must be an integer.")
    if value < 1 or value > 65535:
        raise ValueError("cui_split_view_port must be between 1 and 65535.")
    return int(value)


def _strict_cui_public_host(settings: Mapping[str, object]) -> str:
    """Return the explicit browser-facing host for the CUI server."""
    value = settings["cui_split_view_public_host"]
    if not isinstance(value, str) or not value or value == "auto":
        raise TypeError("cui_split_view_public_host must be an explicit host string.")
    return value


def _bind_cui_view_server(
    settings: Mapping[str, object],
    *,
    output_dir: Path,
) -> CUIServerHandle | None:
    """Bind the configured CUI port before any runtime session is opened."""
    enabled = resolve_cui_split_view_enabled(settings)
    serve = _strict_cui_bool(settings, "cui_split_view_serve")
    if serve and not enabled:
        raise ValueError("cui_split_view_serve requires cui_split_view=true.")
    if not serve:
        network_fields = (
            "cui_split_view_host",
            "cui_split_view_port",
            "cui_split_view_public_host",
        )
        non_inert = [name for name in network_fields if settings[name] is not None]
        if non_inert:
            raise ValueError(
                f"Non-serving CUI requires null network fields: {non_inert}."
            )
        if not enabled and (
            _strict_cui_bool(settings, "cui_split_view_save_step_history")
            or settings["cui_split_view_max_particles_per_isotope"] is not None
        ):
            raise ValueError("Disabled CUI requires inert renderer-only settings.")
        return None
    return start_cui_view_server(
        output_dir,
        host=_strict_cui_host(settings),
        port=_strict_cui_port(settings),
        public_host=_strict_cui_public_host(settings),
    )


def _start_cui_split_view(
    settings: Mapping[str, object],
    *,
    output_dir: Path,
    isotopes: Sequence[str],
    room_bounds: tuple[np.ndarray, np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    truth_overlay_socket_path: str | Path | None,
    server_handle: CUIServerHandle | None,
    output_hook: Any,
) -> AsyncCUISplitPFVisualizer | None:
    """Start the renderer after context resolution and a successful server bind."""
    if not resolve_cui_split_view_enabled(settings):
        if truth_overlay_socket_path is not None:
            raise ValueError("A CUI truth overlay socket requires cui_split_view=true.")
        if server_handle is not None:
            raise RuntimeError("Disabled CUI unexpectedly owns a server handle.")
        return None
    lower, upper = room_bounds
    serve = _strict_cui_bool(settings, "cui_split_view_serve")
    if serve:
        if server_handle is None or not isinstance(server_handle.url, str):
            raise RuntimeError(
                "Enabled CUI serving requires its pre-bound owning handle."
            )
    elif server_handle is not None:
        raise RuntimeError("Non-serving CUI unexpectedly owns a server handle.")
    visualizer: AsyncCUISplitPFVisualizer | None = None
    try:
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
            truth_overlay_socket_path=truth_overlay_socket_path,
            max_particles_per_isotope=settings[
                "cui_split_view_max_particles_per_isotope"
            ],
            save_step_history=_strict_cui_bool(
                settings,
                "cui_split_view_save_step_history",
            ),
        )
        output_hook(
            "CUI split visualization enabled: "
            f"{visualizer.index_path.as_posix()} "
            "(latest_experiment_overview.png, latest_robot_2d.png, "
            "latest_pf_3d.png, latest_pf_3d_labeled.png, latest_spectrum.png)"
        )
        output_hook("CUI split visualization rendering: async process")
        if server_handle is not None:
            output_hook(f"{CUI_URL_MESSAGE_PREFIX} {server_handle.url}")
        return visualizer
    except BaseException as exc:
        if visualizer is not None:
            try:
                visualizer.close()
            except BaseException as close_exc:
                exc.add_note(
                    "Secondary CUI renderer cleanup failure: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )
        raise


def _publish_cui_frame(
    visualizer: AsyncCUISplitPFVisualizer,
    estimator: object,
    record: MeasurementLogRecord,
    route_records: list[MeasurementLogRecord],
    *,
    elapsed_time_s: float,
    record_measurement: bool,
    reusable_frame: PFFrame | None = None,
) -> PFFrame:
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
    path_waypoints = (
        None
        if not route.travel_path_segments_xyz
        else route.travel_path_segments_xyz[-1].copy()
    )
    spectrum_energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    detector_position = np.asarray(record.detector_pose_xyz, dtype=np.float64)
    if reusable_frame is None:
        frame = build_frame_from_pf(
            estimator,
            int(record.step_id),
            float(elapsed_time_s),
            detector_position=detector_position,
            spectrum_energy_keV=spectrum_energy,
            spectrum_counts=spectrum_counts,
        )
        frame.path_waypoints_xyz = path_waypoints
        frame.cui_route = route
    else:
        frame = replace(
            reusable_frame,
            step_index=int(record.step_id),
            time=float(elapsed_time_s),
            robot_position=detector_position,
            path_waypoints_xyz=path_waypoints,
            spectrum_energy_keV=spectrum_energy,
            spectrum_counts=spectrum_counts,
            cui_route=route,
        )
    visualizer.update(frame)
    return frame


def _pf_result_figure_data_payload(
    route: CUIRoute,
    *,
    run_id: str,
    measurement_log_sha256: str,
) -> dict[str, object]:
    """Build truth-free numeric data needed to redraw PF result figures."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("PF figure data requires a nonempty run identifier.")
    if not isinstance(measurement_log_sha256, str) or not measurement_log_sha256:
        raise ValueError("PF figure data requires a MeasurementLog identity.")
    return {
        "schema_version": 1,
        "artifact_family": "pf_result_figure_data",
        "run_identity": {
            "run_id": run_id,
            "measurement_log_sha256": measurement_log_sha256,
        },
        "coordinate_system": {
            "frame": "MeasurementLog world frame",
            "axis_order": ["x", "y", "z"],
            "length_unit": "m",
        },
        "route": route.to_payload(),
        "source_artifacts": {
            "measurement_log": [
                "environment.json",
                "forward_model_manifest.json",
                "run_manifest.json",
                "runtime_config.resolved.json",
                "observations.npz",
                "observation_metadata.jsonl",
            ],
            "pf_output": [
                "pf_posterior.json",
                "pf_diagnostics.json",
                "pf_particles.npz",
                "pf_post_run_evaluation_input.json",
                "pf_station_trace.jsonl",
                "pf_station_performance.jsonl",
                "planner_audit.jsonl",
            ],
        },
        "presentation_contract": {
            "numeric_values_rounded": False,
            "route_line_semantics": (
                "Only persisted MeasurementLog travel_waypoints_xyz segments "
                "may be connected as a route."
            ),
            "station_semantics": (
                "Measurement stations are retained independently of route availability."
            ),
        },
        "truth_included": False,
    }


def _particle_diagnostics(estimator: object) -> dict[str, object]:
    """Extract particle-adequacy evidence without simulation truth."""
    raw = estimator.step_diagnostics(
        top_k=0,
        include_estimates=False,
        include_runtime_details=False,
    )
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("PF step diagnostics must contain isotope mappings.")
    expected_isotopes = tuple(str(value) for value in estimator.isotopes)
    if tuple(str(value) for value in raw) != expected_isotopes:
        raise ValueError(
            "PF step diagnostics isotope order differs from the estimator."
        )
    if any(not isinstance(values, Mapping) for values in raw.values()):
        raise TypeError("Every PF isotope diagnostic row must be a mapping.")
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
        "hard_max_sources",
        "transition_weight_mass",
        "joint_smc_wall_time_limit_exceeded",
        "joint_rejuvenation_mixing_incomplete",
        "joint_structural_mixing_incomplete",
    )
    retained = {
        str(isotope): {key: values.get(key) for key in keep}
        for isotope, values in raw.items()
    }
    transition_count_method = getattr(
        estimator,
        "latest_station_adjacent_cardinality_transition_counts",
        None,
    )
    transition_counts = (
        transition_count_method() if callable(transition_count_method) else {}
    )
    if not isinstance(transition_counts, Mapping):
        raise TypeError("PF adjacent-cardinality transition counts must be a mapping.")
    sampler_fields = (
        "joint_smc_wall_time_limit_exceeded",
        "joint_rejuvenation_mixing_incomplete",
        "joint_structural_mixing_incomplete",
    )
    for isotope, values in retained.items():
        if any(type(values[name]) is not bool for name in sampler_fields):
            raise TypeError(
                f"PF isotope {isotope} omits Boolean sampler-health evidence."
            )
    isotopes = {}
    for isotope, values in retained.items():
        transition_mass = values["transition_weight_mass"]
        aggregate_transition_mass: dict[str, float] = {}
        if isinstance(transition_mass, Mapping):
            attempted = sum(
                float(value)
                for name, value in transition_mass.items()
                if str(name).endswith("_attempted_weight_mass")
            )
            accepted = sum(
                float(value)
                for name, value in transition_mass.items()
                if str(name).endswith("_accepted_weight_mass")
            )
            if any(
                str(name).endswith("_attempted_weight_mass") for name in transition_mass
            ):
                aggregate_transition_mass = {
                    "attempted": float(attempted),
                    "accepted": float(accepted),
                }
        compact = {
            "particle_count": values["particle_count"],
            "current_ess": values["current_ess"],
            "current_ess_ratio": values["current_ess_ratio"],
            "temper_resamples": values["temper_resamples"],
            "temper_min_ess": values["temper_min_ess"],
            "station_unique_ancestor_count": values["station_unique_ancestor_count"],
            "cumulative_unique_ancestor_count": values[
                "cumulative_unique_ancestor_count"
            ],
            "cardinality_distribution": values["r_probability_by_count"],
            "structural_transition_weight_mass": (aggregate_transition_mass or None),
            "adjacent_cardinality_transition_counts": transition_counts.get(isotope),
            "sampler_health": {
                "smc_rejuvenation_wall_time_respected": bool(
                    values["joint_smc_wall_time_limit_exceeded"] is False
                ),
                "rejuvenation_mixing_complete": bool(
                    values["joint_rejuvenation_mixing_incomplete"] is False
                ),
                "structural_mixing_complete": bool(
                    values["joint_structural_mixing_incomplete"] is False
                ),
            },
        }
        hard_cap = values["hard_max_sources"]
        distribution = values["r_probability_by_count"]
        if (
            isinstance(hard_cap, bool)
            or not isinstance(hard_cap, int)
            or hard_cap <= 0
            or not isinstance(distribution, Mapping)
        ):
            raise TypeError(
                f"PF isotope {isotope} omits hard-cap cardinality evidence."
            )
        hard_cap_mass = distribution.get(
            int(hard_cap),
            distribution.get(str(int(hard_cap)), 0.0),
        )
        if (
            isinstance(hard_cap_mass, bool)
            or not isinstance(hard_cap_mass, (int, float))
            or not np.isfinite(float(hard_cap_mass))
            or not 0.0 <= float(hard_cap_mass) <= 1.0 + 1.0e-12
        ):
            raise TypeError(
                f"PF isotope {isotope} has invalid hard-cap posterior mass."
            )
        compact["hard_cap_source_count"] = int(hard_cap)
        compact["hard_cap_posterior_mass"] = float(hard_cap_mass)
        compact["hard_cap_posterior_mass_limit"] = HARD_CAP_POSTERIOR_MASS_LIMIT
        isotopes[isotope] = {
            key: value for key, value in compact.items() if value is not None
        }
    configured_count = int(estimator.pf_config.num_particles)
    target_ratio = float(estimator.pf_config.target_ess_ratio)
    guided_ratios = [
        float(values["joint_guided_initialization_ess"]) / configured_count
        for values in retained.values()
        if values.get("joint_guided_initialization_ess") is not None
    ]
    ancestry_counts = [
        int(values["cumulative_unique_ancestor_count"])
        for values in retained.values()
        if values.get("cumulative_unique_ancestor_count") is not None
    ]
    lineage_collapsed = bool(ancestry_counts and min(ancestry_counts) <= 1)
    rejuvenation_records = getattr(
        estimator,
        "last_joint_rejuvenation_diagnostics",
        (),
    )
    final_rejuvenation = (
        rejuvenation_records[-1]
        if isinstance(rejuvenation_records, Sequence)
        and not isinstance(rejuvenation_records, (str, bytes))
        and rejuvenation_records
        and isinstance(rejuvenation_records[-1], Mapping)
        else None
    )

    def _finite_binary_diagnostic(value: object) -> bool:
        """Return whether a diagnostic is a strict finite numeric zero or one."""
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            and float(value) in (0.0, 1.0)
        )

    recovery_by_isotope: dict[str, dict[str, object]] = {}
    recovery_record_complete = bool(final_rejuvenation is not None)
    if final_rejuvenation is not None:
        configured_recovery_mass = float(
            estimator.pf_config.joint_lineage_recovery_min_surviving_weight_mass
        )
        recorded_recovery_mass = final_rejuvenation.get(
            "lineage_recovery_min_surviving_weight_mass"
        )
        recovery_epoch = final_rejuvenation.get("lineage_recovery_epoch")
        distinct_joint_states = final_rejuvenation.get("distinct_joint_state_count")
        minimum_distinct_joint_states = final_rejuvenation.get(
            "minimum_distinct_joint_states"
        )
        expected_minimum_distinct_joint_states = int(
            np.ceil(target_ratio * configured_count)
        )
        distinct_state_record_valid = bool(
            isinstance(distinct_joint_states, (int, float))
            and not isinstance(distinct_joint_states, bool)
            and np.isfinite(float(distinct_joint_states))
            and float(distinct_joint_states).is_integer()
            and 0.0 <= float(distinct_joint_states) <= configured_count
        )
        minimum_state_record_valid = bool(
            isinstance(minimum_distinct_joint_states, (int, float))
            and not isinstance(minimum_distinct_joint_states, bool)
            and np.isfinite(float(minimum_distinct_joint_states))
            and float(minimum_distinct_joint_states).is_integer()
            and float(minimum_distinct_joint_states)
            == expected_minimum_distinct_joint_states
        )
        recovery_record_complete = bool(
            recovery_record_complete
            and distinct_state_record_valid
            and minimum_state_record_valid
            and float(distinct_joint_states) >= float(minimum_distinct_joint_states)
            and isinstance(recorded_recovery_mass, (int, float))
            and not isinstance(recorded_recovery_mass, bool)
            and np.isfinite(float(recorded_recovery_mass))
            and float(recorded_recovery_mass) == configured_recovery_mass
            and isinstance(recovery_epoch, (int, float))
            and not isinstance(recovery_epoch, bool)
            and np.isfinite(float(recovery_epoch))
            and float(recovery_epoch).is_integer()
            and float(recovery_epoch) >= (1.0 if lineage_collapsed else 0.0)
        )
        for isotope in expected_isotopes:
            certified_count = final_rejuvenation.get(
                f"lineage_recovery_certified_row_count.{isotope}"
            )
            surviving_mass = final_rejuvenation.get(
                f"lineage_recovery_surviving_weight_mass.{isotope}"
            )
            sufficient = final_rejuvenation.get(
                f"lineage_recovery_sufficient.{isotope}"
            )
            row_complete = bool(
                isinstance(certified_count, (int, float))
                and not isinstance(certified_count, bool)
                and np.isfinite(float(certified_count))
                and float(certified_count).is_integer()
                and 0.0 <= float(certified_count) <= configured_count
                and isinstance(surviving_mass, (int, float))
                and not isinstance(surviving_mass, bool)
                and np.isfinite(float(surviving_mass))
                and 0.0 <= float(surviving_mass) <= 1.0 + 1.0e-12
                and _finite_binary_diagnostic(sufficient)
            )
            expected_sufficient = bool(
                row_complete
                and float(certified_count) > 0.0
                and float(surviving_mass) >= configured_recovery_mass
            )
            row_complete = bool(
                row_complete and bool(sufficient == 1.0) is expected_sufficient
            )
            recovery_record_complete = recovery_record_complete and row_complete
            recovery_by_isotope[isotope] = {
                "certified_row_count": certified_count,
                "surviving_weight_mass": surviving_mass,
                "sufficient": bool(sufficient == 1.0),
            }
    recovery_required_recorded = bool(
        final_rejuvenation is not None
        and _finite_binary_diagnostic(
            final_rejuvenation.get("lineage_recovery_required")
        )
        and float(final_rejuvenation["lineage_recovery_required"]) == 1.0
    )
    recovery_sufficient_recorded = bool(
        final_rejuvenation is not None
        and _finite_binary_diagnostic(
            final_rejuvenation.get("lineage_recovery_sufficient")
        )
        and float(final_rejuvenation["lineage_recovery_sufficient"]) == 1.0
    )
    recovery_complete = bool(
        lineage_collapsed
        and recovery_record_complete
        and recovery_required_recorded
        and recovery_sufficient_recorded
        and all(row["sufficient"] is True for row in recovery_by_isotope.values())
    )
    evidence = {
        "configured_particle_count": configured_count,
        "target_ess_ratio": target_ratio,
        "minimum_guided_initialization_ess_ratio": (
            None if not guided_ratios else float(min(guided_ratios))
        ),
        "minimum_cumulative_unique_ancestor_count": (
            None if not ancestry_counts else int(min(ancestry_counts))
        ),
        "diversity_evidence_available": bool(guided_ratios or ancestry_counts),
        "genealogical_collapse_detected": lineage_collapsed,
        "lineage_recovery_required": lineage_collapsed,
        "lineage_recovery_complete": recovery_complete,
        "lineage_recovery_by_isotope": recovery_by_isotope,
        "distinct_joint_state_count": (
            None
            if final_rejuvenation is None
            else final_rejuvenation.get("distinct_joint_state_count")
        ),
        "minimum_distinct_joint_states": (
            None
            if final_rejuvenation is None
            else final_rejuvenation.get("minimum_distinct_joint_states")
        ),
        "diversity_warning": bool(
            (not guided_ratios and not ancestry_counts)
            or (lineage_collapsed and not recovery_complete)
            or (
                not ancestry_counts
                and guided_ratios
                and min(guided_ratios) < target_ratio
            )
        ),
    }
    evidence = {key: value for key, value in evidence.items() if value is not None}
    sampler_health = {
        "smc_rejuvenation_wall_time_respected": all(
            values.get("joint_smc_wall_time_limit_exceeded") is False
            for values in retained.values()
        ),
        "rejuvenation_mixing_complete": all(
            values.get("joint_rejuvenation_mixing_incomplete") is False
            for values in retained.values()
        ),
        "structural_mixing_complete": all(
            values.get("joint_structural_mixing_incomplete") is False
            for values in retained.values()
        ),
    }
    result = {
        "assessment": evidence,
        "sampler_health": sampler_health,
        "isotopes": isotopes,
    }
    result["sampler_quality"] = _sampler_quality_summary(result)
    return result


def _require_plannable_sampler_health(
    particle_adequacy: Mapping[str, object],
) -> None:
    """Validate sampler diagnostics without blocking the next acquisition."""
    raw_health = particle_adequacy.get("sampler_health")
    expected = {
        "smc_rejuvenation_wall_time_respected",
        "rejuvenation_mixing_complete",
        "structural_mixing_complete",
    }
    if not isinstance(raw_health, Mapping) or set(raw_health) != expected:
        raise TypeError(
            "PF particle diagnostics must contain the exact sampler-health gates."
        )
    if any(type(raw_health[name]) is not bool for name in expected):
        raise TypeError("PF sampler-health gates must be booleans.")
    assessment = particle_adequacy.get("assessment")
    if not isinstance(assessment, Mapping):
        raise TypeError("PF particle diagnostics must contain diversity assessment.")
    if (
        type(assessment.get("diversity_evidence_available")) is not bool
        or type(assessment.get("diversity_warning")) is not bool
    ):
        raise TypeError("PF diversity assessment must contain Boolean evidence.")
    quality = particle_adequacy.get("sampler_quality")
    if quality is None:
        return
    if not isinstance(quality, Mapping):
        raise TypeError("PF sampler quality must be an object.")
    if quality.get("status") not in {"pass", "warning", "failed"}:
        raise TypeError("PF sampler quality has an invalid status.")
    reasons = quality.get("reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise TypeError("PF sampler-quality reasons must be a string list.")


def _sampler_quality_summary(
    particle_adequacy: Mapping[str, object],
) -> dict[str, object]:
    """Classify truth-free sampler quality without changing execution status."""
    raw_health = particle_adequacy.get("sampler_health")
    assessment = particle_adequacy.get("assessment")
    isotopes = particle_adequacy.get("isotopes")
    if (
        not isinstance(raw_health, Mapping)
        or not isinstance(assessment, Mapping)
        or not isinstance(isotopes, Mapping)
    ):
        raise TypeError(
            "PF sampler quality requires health, diversity, and isotope rows."
        )
    reasons = sorted(name for name, value in raw_health.items() if value is not True)
    if assessment.get("diversity_evidence_available") is not True:
        reasons.append("particle_diversity_evidence_unavailable")
    elif assessment.get("diversity_warning") is not False:
        reasons.append("particle_diversity_warning")
    hard_cap_isotopes: list[str] = []
    for isotope, raw_row in isotopes.items():
        if not isinstance(isotope, str) or not isinstance(raw_row, Mapping):
            raise TypeError("PF sampler-quality isotope rows are invalid.")
        mass = raw_row.get("hard_cap_posterior_mass")
        limit = raw_row.get("hard_cap_posterior_mass_limit")
        if (
            isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not np.isfinite(float(mass))
            or not np.isfinite(float(limit))
        ):
            raise TypeError("PF sampler-quality hard-cap evidence is invalid.")
        if float(mass) > float(limit):
            hard_cap_isotopes.append(isotope)
    reasons.extend(
        f"hard_cap_posterior_mass_exceeded.{isotope}"
        for isotope in sorted(hard_cap_isotopes)
    )
    status = "failed" if hard_cap_isotopes else "warning" if reasons else "pass"
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
    }


def _shield_view_count_shadow_health(
    *,
    belief_after_station_id: int,
    particle_adequacy: Mapping[str, object],
    posterior_convergence: Mapping[str, object],
) -> dict[str, object]:
    """Return truth-free hard gates for audit-only view-count shortening."""
    if (
        isinstance(belief_after_station_id, bool)
        or not isinstance(belief_after_station_id, int)
        or belief_after_station_id < 0
    ):
        raise ValueError("belief_after_station_id must be nonnegative.")
    assessment = particle_adequacy.get("assessment")
    if not isinstance(assessment, Mapping):
        raise TypeError("particle_adequacy.assessment must be a mapping.")
    sampler_health = posterior_convergence.get("sampler_health")
    innovation = posterior_convergence.get("innovation")
    isotope_rows = posterior_convergence.get("isotopes")
    if not isinstance(sampler_health, Mapping):
        raise TypeError("posterior sampler_health must be a mapping.")
    if not isinstance(innovation, Mapping):
        raise TypeError("posterior innovation must be a mapping.")
    if not isinstance(isotope_rows, Mapping):
        raise TypeError("posterior isotope health must be a mapping.")
    expected_sampler_keys = {
        "smc_rejuvenation_wall_time_respected",
        "rejuvenation_mixing_complete",
        "structural_mixing_complete",
    }
    if set(sampler_health) != expected_sampler_keys or any(
        type(sampler_health[name]) is not bool for name in expected_sampler_keys
    ):
        raise TypeError(
            "posterior sampler_health must contain exactly three Boolean gates."
        )
    expected_innovation_keys = {
        "available",
        "passed",
        "view_count",
        "dimension",
        "renewal_total_max_abs_z",
        "renewal_total_within_confidence",
        "conditional_mark_pearson",
        "conditional_mark_degrees_of_freedom",
        "conditional_mark_tail_probability",
        "conditional_mark_upper_tail_probability",
        "confidence",
    }
    if set(innovation) != expected_innovation_keys or any(
        type(innovation[name]) is not bool for name in ("available", "passed")
    ):
        raise TypeError(
            "posterior innovation must match the exact model-native schema."
        )
    particle_isotopes = particle_adequacy.get("isotopes")
    if not isinstance(particle_isotopes, Mapping) or not particle_isotopes:
        raise TypeError("particle adequacy requires nonempty isotope rows.")
    if tuple(str(key) for key in isotope_rows) != tuple(
        str(key) for key in particle_isotopes
    ):
        raise ValueError(
            "posterior and particle-adequacy isotope rows must match exactly."
        )

    reasons: list[str] = []
    diversity_evidence_available = bool(
        assessment.get("diversity_evidence_available", False)
        or assessment.get("minimum_guided_initialization_ess_ratio") is not None
        or assessment.get("minimum_cumulative_unique_ancestor_count") is not None
    )
    if not diversity_evidence_available:
        reasons.append("particle_diversity_evidence_unavailable")
    elif bool(assessment.get("diversity_warning", True)):
        reasons.append("particle_diversity_warning")
    for name in (
        "smc_rejuvenation_wall_time_respected",
        "rejuvenation_mixing_complete",
        "structural_mixing_complete",
    ):
        if sampler_health.get(name) is not True:
            reasons.append(f"sampler_health:{name}")
    innovation_available = innovation.get("available") is True
    innovation_passed = innovation.get("passed") is True
    if not innovation_available:
        reasons.append("posterior_predictive_innovation_unavailable")
    elif not innovation_passed:
        reasons.append("posterior_predictive_innovation_failed")
    boundary_isotopes: list[str] = []
    for isotope, raw_row in sorted(isotope_rows.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_row, Mapping):
            raise TypeError("posterior isotope health rows must be mappings.")
        gates = raw_row.get("gates")
        if not isinstance(gates, Mapping):
            raise TypeError("posterior isotope gates must be mappings.")
        expected_isotope_gates = {
            "cardinality_not_at_upper_boundary",
            "surface_path_concentration",
        }
        if set(gates) != expected_isotope_gates or any(
            type(gates[name]) is not bool for name in expected_isotope_gates
        ):
            raise TypeError(
                "posterior isotope health must contain the exact Boolean "
                "cardinality boundary gate."
            )
        if gates["cardinality_not_at_upper_boundary"] is not True:
            boundary_isotopes.append(str(isotope))
            reasons.append(f"cardinality_upper_boundary:{isotope}")
    return {
        "policy_schema_version": 1,
        "hard_gate_contract": list(SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES),
        "available": True,
        "passed": not reasons,
        "source_station_id": int(belief_after_station_id),
        "hard_failure_reasons": reasons,
        "truth_used": False,
        "particle_adequacy": {
            "diversity_evidence_available": bool(diversity_evidence_available),
            "diversity_warning": bool(assessment.get("diversity_warning", True)),
            "minimum_guided_initialization_ess_ratio": assessment.get(
                "minimum_guided_initialization_ess_ratio"
            ),
            "minimum_cumulative_unique_ancestor_count": assessment.get(
                "minimum_cumulative_unique_ancestor_count"
            ),
        },
        "sampler_health": dict(sampler_health),
        "posterior_predictive_innovation_available": bool(innovation_available),
        "posterior_predictive_innovation_passed": bool(innovation_passed),
        "cardinality_upper_boundary_isotopes": boundary_isotopes,
    }


def _current_shadow_health(
    estimator: object,
    *,
    planner: PFPlannerConfig,
    belief_after_station_id: int,
    particle_adequacy: Mapping[str, object],
    adaptive_stop_status: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Evaluate view-count health only when the shadow policy is enabled."""
    if not bool(planner.shield_view_count_shadow_enabled):
        return None
    convergence: Mapping[str, object] | None = None
    if adaptive_stop_status is not None:
        raw = adaptive_stop_status.get("posterior_convergence")
        if isinstance(raw, Mapping):
            convergence = raw
    if convergence is None:
        raw = estimator.posterior_convergence_diagnostics()
        if not isinstance(raw, Mapping):
            raise TypeError("posterior_convergence_diagnostics must be a mapping.")
        convergence = raw
    return _shield_view_count_shadow_health(
        belief_after_station_id=int(belief_after_station_id),
        particle_adequacy=particle_adequacy,
        posterior_convergence=convergence,
    )


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
    planner: PFPlannerConfig,
    control_policy: object | None,
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
    planner: PFPlannerConfig,
    control_policy: object | None,
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


def _plan(
    estimator: object,
    candidates: AdaptiveCandidateSnapshot,
    *,
    current_pose: np.ndarray,
    visited_poses: Sequence[np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    room_bounds: tuple[np.ndarray, np.ndarray],
    planner: PFPlannerConfig,
    rng: np.random.Generator,
    station_index: int,
    control_policy: object | None,
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
        visited_poses_xyz=np.asarray(visited_poses, dtype=np.float64),
        map_api=obstacle_grid,
        bounds_xyz=room_bounds,
        config=active_planner,
        rng=rng,
        candidate_motion_times_s=np.asarray(
            candidates.travel_costs,
            dtype=np.float64,
        ),
        candidate_horizontal_travel_times_s=np.asarray(
            candidates.horizontal_travel_times_s,
            dtype=np.float64,
        ),
        candidate_mast_vertical_times_s=np.asarray(
            candidates.mast_vertical_times_s,
            dtype=np.float64,
        ),
        candidate_settling_times_s=np.asarray(
            candidates.settling_times_s,
            dtype=np.float64,
        ),
    )


def _planning_stage_wall_times(result: DSSPPResult) -> dict[str, float]:
    """Return the exact planner stage timings from one native DSS result."""
    if not isinstance(result.diagnostics, Mapping):
        raise TypeError("DSS planning diagnostics must be a mapping.")
    raw = result.diagnostics.get("planning_stage_wall_s")
    if not isinstance(raw, Mapping):
        raise RuntimeError("Native DSS planning omitted stage timing diagnostics.")
    required = {
        "planning_particle_snapshot",
        "geometry_particle_snapshot",
        "signature_mode_extraction",
        "official_mode_projection",
        "node_build_and_eig",
        "total_before_result",
    }
    if set(raw) != required:
        raise RuntimeError("Native DSS planning stage timing schema is incomplete.")
    resolved = {str(key): float(raw[key]) for key in sorted(required)}
    if any(not np.isfinite(value) or value < 0.0 for value in resolved.values()):
        raise RuntimeError("Native DSS planning stage timings are invalid.")
    return resolved


def _planner_rng(seed: int, station_index: int) -> np.random.Generator:
    """Return the deterministic planner stream for one fresh-run station."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), 0xD55A11, int(station_index)])
    )


def _require_refinement_seed_capacity(
    settings: Mapping[str, object],
    candidates: AdaptiveCandidateSnapshot,
) -> None:
    """Reject a live refinement request that the handshake cannot satisfy."""
    top_k = _exact_integer(
        settings["runtime_candidate_refinement_top_k"],
        name="runtime_candidate_refinement_top_k",
        minimum=0,
    )
    candidate_count = len(candidates.candidate_poses_xyz)
    if top_k > candidate_count:
        raise ValueError(
            "runtime_candidate_refinement_top_k exceeds the authenticated "
            f"candidate count ({top_k} > {candidate_count})."
        )


def _require_native_planner_settings(settings: Mapping[str, object]) -> None:
    """Reject fixed-path sentinels when native DSS-PP must execute."""
    if not isinstance(settings.get("dss_pp"), Mapping):
        raise ValueError(
            "Native DSS-PP control requires a complete dss_pp configuration."
        )
    planning_samples = settings.get("planning_eig_samples")
    if (
        isinstance(planning_samples, bool)
        or not isinstance(planning_samples, int)
        or planning_samples < 2
    ):
        raise ValueError("Native DSS-PP control requires planning_eig_samples>=2.")


def _planner_audit_for_mode(
    *,
    station_id: int,
    result: DSSPPResult,
    planner: PFPlannerConfig,
    top_k: int,
    belief_after_station_id: int,
    posterior_health: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build one native DSS-PP audit for the current RA-L control contract."""
    return build_planner_audit(
        station_id=station_id,
        result=result,
        top_k=top_k,
        belief_after_station_id=belief_after_station_id,
        posterior_health=posterior_health,
    )


def _refine_and_replan(
    client: AdaptiveRuntimeClient,
    live_session: PFLiveSession,
    estimator: object,
    candidates: AdaptiveCandidateSnapshot,
    initial: DSSPPResult,
    *,
    refinement_top_k: int,
    current_pose: np.ndarray,
    visited_poses: Sequence[np.ndarray],
    obstacle_grid: ObstacleGrid | None,
    room_bounds: tuple[np.ndarray, np.ndarray],
    planner: PFPlannerConfig,
    rng: np.random.Generator,
    station_index: int,
    control_policy: object | None,
) -> tuple[AdaptiveCandidateSnapshot, DSSPPResult]:
    """Optionally request runtime-owned local poses and rerank them exactly."""
    if refinement_top_k <= 0:
        return candidates, initial
    if not isinstance(initial.diagnostics, Mapping):
        raise TypeError("Candidate refinement requires planner diagnostics.")
    if "ranked_nodes" not in initial.diagnostics:
        raise ValueError(
            "Candidate refinement requires canonical ranked_nodes diagnostics."
        )
    ranked = initial.diagnostics["ranked_nodes"]
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
        raise TypeError("Candidate refinement ranked_nodes must be a sequence.")
    if not ranked:
        raise ValueError("Candidate refinement is enabled but ranked_nodes is empty.")
    seed_indices: list[int] = []
    for node in ranked:
        if not isinstance(node, Mapping):
            raise TypeError("Every candidate-refinement ranked node must be a mapping.")
        if "pose_xyz" not in node:
            raise ValueError("Candidate-refinement ranked node omits pose_xyz.")
        index = candidate_index_for_pose(candidates, node["pose_xyz"])
        if index not in seed_indices:
            seed_indices.append(index)
        if len(seed_indices) >= refinement_top_k:
            break
    if len(seed_indices) != refinement_top_k:
        raise ValueError(
            "Candidate refinement could not resolve exactly the requested "
            f"{refinement_top_k} distinct runtime-authored seed poses."
        )
    event = client.refine_candidates(AdaptiveRefineRequest.from_indices(seed_indices))
    refined = event.candidates
    live_session.receive_refined_candidates(refined)
    result = _plan(
        estimator,
        refined,
        current_pose=current_pose,
        visited_poses=visited_poses,
        obstacle_grid=obstacle_grid,
        room_bounds=room_bounds,
        planner=planner,
        rng=rng,
        station_index=station_index,
        control_policy=control_policy,
    )
    return refined, result


def _completion_diagnostics_extensions(
    *,
    stop_reason: str,
    budget: PFControlBudget,
    adaptive_stop_status: Mapping[str, object] | None,
    particle_adequacy: Mapping[str, object],
) -> dict[str, object]:
    """Return controller diagnostics sealed with the canonical PF state."""
    stop: dict[str, object] = {"reason": stop_reason}
    compact_stop = _compact_adaptive_stop_status(adaptive_stop_status)
    if compact_stop is not None:
        stop["adaptive"] = compact_stop
    quality = particle_adequacy.get("sampler_quality")
    if not isinstance(quality, Mapping):
        raise TypeError("Final PF diagnostics require sampler quality.")
    return {
        "stop": stop,
        "control_budget": asdict(budget),
        "particle_adequacy": dict(particle_adequacy),
    }


def _last_station_trace_posterior(path: Path) -> dict[str, object] | None:
    """Return the latest valid truth-free posterior from a station trace."""
    if path.is_symlink() or not path.is_file():
        return None
    for raw_line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        posterior = record.get("posterior_snapshot")
        if (
            isinstance(posterior, dict)
            and posterior.get("publishable") is False
            and isinstance(posterior.get("isotopes"), Mapping)
        ):
            return posterior
    return None


def _failure_truth_free_posterior(
    estimator: object | None,
    *,
    station_trace_path: Path,
) -> tuple[dict[str, object], str]:
    """Return the current or last completed truth-free failure posterior."""
    current_error: BaseException | None = None
    if estimator is not None:
        try:
            return _live_posterior_summary(estimator), "current_estimator_state"
        except BaseException as exc:
            current_error = exc
    fallback = _last_station_trace_posterior(station_trace_path)
    if fallback is not None:
        return fallback, "last_completed_station"
    message = "No truth-free PF posterior was available for failure diagnosis."
    if current_error is not None:
        raise RuntimeError(message) from current_error
    raise RuntimeError(message)


def _publish_failure_diagnostics(
    *,
    target: Path,
    staging_dir: Path,
    stage_suffix: str,
    estimator: object | None,
    primary_error: BaseException,
) -> Path:
    """Atomically publish a diagnostic-only bundle for a failed live run."""
    if staging_dir.is_symlink() or not staging_dir.is_dir():
        raise FileNotFoundError(
            f"PF failure staging directory is unavailable: {staging_dir}"
        )
    diagnostic_target = target.with_name(
        f"{target.name}.failure-diagnostics-{stage_suffix}"
    )
    station_trace_path = staging_dir / "pf_station_trace.jsonl"
    posterior, posterior_source = _failure_truth_free_posterior(
        estimator,
        station_trace_path=station_trace_path,
    )
    planner_audit_path = staging_dir / "planner_audit.jsonl"
    performance_trace_path = staging_dir / "pf_station_performance.jsonl"
    cui_image_path = staging_dir / "cui_live" / "latest_pf_3d.png"
    planner_available = bool(
        planner_audit_path.is_file() and not planner_audit_path.is_symlink()
    )
    station_trace_available = bool(
        station_trace_path.is_file() and not station_trace_path.is_symlink()
    )
    performance_trace_available = bool(
        performance_trace_path.is_file() and not performance_trace_path.is_symlink()
    )
    cui_image_available = bool(
        cui_image_path.is_file() and not cui_image_path.is_symlink()
    )
    with AtomicBundlePublisher(
        diagnostic_target,
        policy="create",
    ) as publisher:
        publisher.write_bytes(
            "truth_free_posterior.json",
            _strict_live_artifact_json_bytes(
                posterior,
                artifact_name="PF failure truth-free posterior",
            ),
        )
        if planner_available:
            publisher.copy_file(planner_audit_path, "planner_audit.jsonl")
        else:
            publisher.write_bytes("planner_audit.jsonl", b"")
        if station_trace_available:
            publisher.copy_file(station_trace_path, "pf_station_trace.jsonl")
        else:
            publisher.write_bytes("pf_station_trace.jsonl", b"")
        if performance_trace_available:
            publisher.copy_file(
                performance_trace_path,
                "pf_station_performance.jsonl",
            )
        else:
            publisher.write_bytes("pf_station_performance.jsonl", b"")
        if cui_image_available:
            publisher.copy_file(cui_image_path, "last_cui_pf_3d.png")
        publisher.write_bytes(
            "failure_manifest.json",
            _strict_live_artifact_json_bytes(
                {
                    "schema_version": 1,
                    "artifact_family": "pure_pf_failure_diagnostics",
                    "status": "failed",
                    "success_result": False,
                    "truth_read": False,
                    "posterior_publishable": False,
                    "posterior_source": posterior_source,
                    "planner_audit_available": planner_available,
                    "station_trace_available": station_trace_available,
                    "performance_trace_available": (performance_trace_available),
                    "last_cui_image_available": cui_image_available,
                    "error_type": type(primary_error).__name__,
                    "message": str(primary_error),
                },
                artifact_name="PF failure diagnostic manifest",
            ),
        )
        publisher.publish()
    return diagnostic_target


def run_pf_closed_loop(
    session_socket: str | Path,
    *,
    runtime_root: str | Path,
    pf_config_path: str | Path,
    output_dir: str | Path,
    seed: int,
    cui_truth_overlay_socket_path: str | Path | None = None,
    profile: str = "pf_strict",
    control_policy: object | None = None,
    station_boundary_stop_request: Callable[[int], bool] | None = None,
    output_hook: Any = print,
) -> PFClosedLoopResult:
    """Run a PF closed loop over an opaque truth-free runtime session socket."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Production PF seed must be a nonnegative integer.")
    if seed < 0:
        raise ValueError("Production PF seed must be a nonnegative integer.")
    if station_boundary_stop_request is not None and not callable(
        station_boundary_stop_request
    ):
        raise TypeError("station_boundary_stop_request must be callable or null.")
    control_policy_provenance = validate_control_policy(control_policy)
    validated_config = load_production_live_pf_config(
        pf_config_path,
        profile=profile,
    )
    settings = validated_config.settings()
    if control_policy is not None:
        control_policy.validate_pf_settings(settings)
    else:
        _require_native_planner_settings(settings)
    compute_backend = production_compute_backend_values(settings)
    preflight_compute_backend(
        use_gpu=compute_backend["use_gpu"],
        gpu_device=compute_backend["gpu_device"],
        gpu_dtype="float64",
    )
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace PF output {target}.")
    bundle_publisher = AtomicBundlePublisher(target, policy="create")
    staging_dir: Path | None = bundle_publisher.staging_path.resolve()
    resources = ExitStack()
    client: AdaptiveRuntimeClient | None = None
    live_session: PFLiveSession | None = None
    estimator: object | None = None
    cui_server_handle: CUIServerHandle | None = None
    cui_split_viz: AsyncCUISplitPFVisualizer | None = None
    cui_frame_enqueued = False
    cui_route_records: list[MeasurementLogRecord] = []
    stage_suffix = staging_dir.name.rsplit(".bundle-", maxsplit=1)[-1]
    failure_receipt_path = target.with_name(
        f".{target.name}.failure-{stage_suffix}.json"
    )
    try:
        cui_server_handle = _bind_cui_view_server(
            settings,
            output_dir=staging_dir / "cui_live",
        )
        planner_writer = PlannerAuditWriter(staging_dir / "planner_audit.jsonl")
        resources.callback(planner_writer.close)
        controller_writer = DurableJSONLWriter(
            staging_dir / "pf_station_trace.jsonl",
            mode=0o644,
        )
        resources.callback(controller_writer.close)
        performance_writer = DurableJSONLWriter(
            staging_dir / "pf_station_performance.jsonl",
            mode=0o644,
        )
        resources.callback(performance_writer.close)
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
        _require_refinement_seed_capacity(settings, candidates)
        acquisition_contract = acquisition_contract_from_environment(
            context.environment
        )
        live_session = PFLiveSession(
            context,
            validated_config,
            initial_candidates=candidates,
            profile=profile,
            seed=seed,
            runtime_root=runtime_root,
            control_policy_provenance=control_policy_provenance,
        )
        estimator = live_session.estimator
        planner: PFPlannerConfig = dss_config_from_pf_settings(
            settings,
            acquisition_contract=acquisition_contract,
            detector_aperture_samples=int(estimator.detector_aperture_samples),
        )
        budget = PFControlBudget.from_runtime_contract(
            settings,
            acquisition_contract,
        )
        stop_tracker = AdaptiveStopTracker(budget)
        latest_adaptive_stop_status: dict[str, object] | None = None
        obstacle_grid = _obstacle_grid(context)
        room_bounds = _bounds(context)
        cui_split_viz = _start_cui_split_view(
            settings,
            output_dir=staging_dir / "cui_live",
            isotopes=getattr(context, "isotopes"),
            room_bounds=room_bounds,
            obstacle_grid=obstacle_grid,
            truth_overlay_socket_path=cui_truth_overlay_socket_path,
            server_handle=cui_server_handle,
            output_hook=output_hook,
        )
        current_program = _bootstrap_program(
            estimator,
            planner,
            control_policy,
        )
        visited: list[np.ndarray]
        reusable_cui_frame: PFFrame | None = None
        stop_reason = "maximum_station_budget"
        continue_acquisition = True
        if schema_version == 1:
            assert bootstrap is not None
            current_pose = np.asarray(
                candidates.candidate_poses_xyz[bootstrap.candidate_index],
                dtype=np.float64,
            )
            _require_candidate_anchor(candidates, current_pose)
            visited = []
            record_count = live_session.record_count
            cui_elapsed_time_s = 0.0
            station_id = 0
            bootstrap_audit = build_bootstrap_planner_audit(
                station_id=0,
                pose_index=int(bootstrap.candidate_index),
                pose_xyz=current_pose,
                program=current_program,
                shadow_enabled=bool(
                    planner.shield_view_count_shadow_enabled
                    and current_program.kind != "external_control"
                ),
                candidate_view_counts=tuple(
                    planner.shield_view_count_shadow_candidate_counts
                ),
                retention_fraction=float(
                    planner.shield_view_count_shadow_retention_fraction
                ),
                per_comparison_confidence=float(
                    planner.shield_view_count_shadow_per_comparison_confidence
                ),
            )
            planner_writer.append(bootstrap_audit)
        else:
            raise RuntimeError("PF live acquisition requires fresh protocol schema 1.")
        while continue_acquisition and station_id < budget.max_stations:
            if record_count + len(current_program.pair_ids) > budget.max_measurements:
                stop_reason = "maximum_measurement_budget"
                break
            station_records: list[MeasurementLogRecord] = []
            station_wall_start_s = time.perf_counter()
            station_cui_enqueue_elapsed_s = 0.0
            station_cui_particle_rebuilds = 0
            station_cui_particle_reuses = 0
            assimilation_start_s: float | None = None
            for view_index, pair_id in enumerate(current_program.pair_ids):
                candidate_index = candidate_index_for_pose(candidates, current_pose)
                fe_index, pb_index = divmod(int(pair_id), 8)
                request = AdaptiveStepRequest(
                    action_id=record_count,
                    candidate_index=candidate_index,
                    fe_orientation_index=fe_index,
                    pb_orientation_index=pb_index,
                    dwell_time_s=budget.live_time_s,
                    station_id=station_id,
                    station_complete=(view_index == len(current_program.pair_ids) - 1),
                )
                event = client.acquire(request)
                record = event.record
                if request.station_complete:
                    assimilation_start_s = time.perf_counter()
                completed_station = live_session.receive_acquired(
                    record,
                    request=request,
                    request_candidates=candidates,
                    next_candidates=event.candidates,
                )
                if completed_station is not request.station_complete:
                    raise RuntimeError(
                        "PF station completion disagrees with the exact request."
                    )
                station_records.append(record)
                candidates = event.candidates
                record_count = live_session.record_count
                cui_elapsed_time_s += (
                    float(record.live_time_s)
                    + float(record.travel_time_s)
                    + float(record.shield_actuation_time_s)
                )
                if cui_split_viz is not None:
                    cui_enqueue_start_s = time.perf_counter()
                    rebuild_particle_frame = bool(
                        reusable_cui_frame is None or completed_station
                    )
                    reusable_cui_frame = _publish_cui_frame(
                        cui_split_viz,
                        estimator,
                        record,
                        cui_route_records,
                        elapsed_time_s=cui_elapsed_time_s,
                        record_measurement=True,
                        reusable_frame=(
                            None if rebuild_particle_frame else reusable_cui_frame
                        ),
                    )
                    station_cui_enqueue_elapsed_s += (
                        time.perf_counter() - cui_enqueue_start_s
                    )
                    if rebuild_particle_frame:
                        station_cui_particle_rebuilds += 1
                    else:
                        station_cui_particle_reuses += 1
                    cui_frame_enqueued = True
                else:
                    cui_route_records.append(record)
            if assimilation_start_s is None:
                raise RuntimeError(
                    "PF station completed without an assimilation start."
                )
            assimilation_elapsed_s = time.perf_counter() - assimilation_start_s
            current_pose = np.asarray(
                station_records[-1].detector_pose_xyz,
                dtype=np.float64,
            )
            visited.append(current_pose.copy())
            completed_stations = station_id + 1
            boundary_diagnostics_start_s = time.perf_counter()
            phase_start_s = time.perf_counter()
            latest_adaptive_stop_status = stop_tracker.assess(
                estimator,
                station_count=completed_stations,
            )
            adaptive_stop_elapsed_s = time.perf_counter() - phase_start_s
            phase_start_s = time.perf_counter()
            particle_adequacy = _particle_diagnostics(estimator)
            particle_diagnostics_elapsed_s = time.perf_counter() - phase_start_s
            phase_start_s = time.perf_counter()
            shadow_health = _current_shadow_health(
                estimator,
                planner=planner,
                belief_after_station_id=int(station_id),
                particle_adequacy=particle_adequacy,
                adaptive_stop_status=latest_adaptive_stop_status,
            )
            shadow_health_elapsed_s = time.perf_counter() - phase_start_s
            phase_start_s = time.perf_counter()
            posterior_snapshot = _live_posterior_summary(estimator)
            posterior_snapshot_elapsed_s = time.perf_counter() - phase_start_s
            station_trace = {
                "schema_version": 2,
                "station_id": station_id,
                "pf_update_elapsed_s": float(assimilation_elapsed_s),
                "particle_adequacy": particle_adequacy,
                "posterior_snapshot": posterior_snapshot,
            }
            compact_stop = _compact_adaptive_stop_status(latest_adaptive_stop_status)
            if compact_stop is not None:
                station_trace["adaptive_stop"] = compact_stop
            phase_start_s = time.perf_counter()
            controller_writer.append(station_trace)
            station_trace_write_elapsed_s = time.perf_counter() - phase_start_s
            phase_start_s = time.perf_counter()
            station_stop_requested = False
            if station_boundary_stop_request is not None:
                station_stop_requested = station_boundary_stop_request(
                    completed_stations
                )
                if type(station_stop_requested) is not bool:
                    raise TypeError(
                        "station_boundary_stop_request must return a boolean."
                    )
            stop_request_elapsed_s = time.perf_counter() - phase_start_s
            terminal_reason: str | None = None
            if record_count >= budget.max_measurements:
                terminal_reason = "maximum_measurement_budget"
            elif completed_stations >= budget.max_stations:
                terminal_reason = "maximum_station_budget"
            elif latest_adaptive_stop_status["stop_ready"]:
                terminal_reason = "intrinsic_surface_posterior_converged"
            elif station_stop_requested:
                terminal_reason = "station_boundary_stop_requested"
            boundary_diagnostics_elapsed_s = (
                time.perf_counter() - boundary_diagnostics_start_s
            )
            assimilation_stage = getattr(
                estimator,
                "last_pair_sequence_stage_wall_s",
                {},
            )
            if not isinstance(assimilation_stage, Mapping):
                raise TypeError("PF station assimilation timing must be a mapping.")
            performance_record: dict[str, object] = {
                "schema_version": 1,
                "station_id": int(station_id),
                "completed_station_count": int(completed_stations),
                "view_count": int(len(station_records)),
                "timing_s": {
                    "station_wall_through_boundary": float(
                        time.perf_counter() - station_wall_start_s
                    ),
                    "pf_update": float(assimilation_elapsed_s),
                    "pf_update_breakdown": dict(assimilation_stage),
                    "boundary_diagnostics": float(boundary_diagnostics_elapsed_s),
                    "adaptive_stop": float(adaptive_stop_elapsed_s),
                    "particle_health": float(particle_diagnostics_elapsed_s),
                    "shadow_health": float(shadow_health_elapsed_s),
                    "posterior_snapshot": float(posterior_snapshot_elapsed_s),
                    "station_trace_write": float(station_trace_write_elapsed_s),
                    "stop_request_poll": float(stop_request_elapsed_s),
                    "cui_enqueue": float(station_cui_enqueue_elapsed_s),
                },
                "cui_particle_state": {
                    "rebuild_count": int(station_cui_particle_rebuilds),
                    "reuse_count": int(station_cui_particle_reuses),
                },
                "terminal_reason": terminal_reason,
            }
            if terminal_reason is not None:
                stop_reason = terminal_reason
                performance_record["timing_s"]["planning"] = 0.0
                performance_writer.append(performance_record)
                break
            phase_start_s = time.perf_counter()
            _require_plannable_sampler_health(particle_adequacy)
            sampler_health_gate_elapsed_s = time.perf_counter() - phase_start_s
            station_planner_rng = _planner_rng(seed, completed_stations)
            planning_start_s = time.perf_counter()
            planned = _plan(
                estimator,
                candidates,
                current_pose=current_pose,
                visited_poses=visited,
                obstacle_grid=obstacle_grid,
                room_bounds=room_bounds,
                planner=planner,
                rng=station_planner_rng,
                station_index=completed_stations,
                control_policy=control_policy,
            )
            candidates, planned = _refine_and_replan(
                client,
                live_session,
                estimator,
                candidates,
                planned,
                refinement_top_k=budget.runtime_refinement_top_k,
                current_pose=current_pose,
                visited_poses=visited,
                obstacle_grid=obstacle_grid,
                room_bounds=room_bounds,
                planner=planner,
                rng=station_planner_rng,
                station_index=completed_stations,
                control_policy=control_policy,
            )
            planning_elapsed_s = time.perf_counter() - planning_start_s
            planning_stage_wall_s = _planning_stage_wall_times(planned)
            station_id += 1
            planner_writer.append(
                _planner_audit_for_mode(
                    station_id=station_id,
                    result=planned,
                    planner=planner,
                    top_k=budget.planner_audit_top_k,
                    belief_after_station_id=int(station_id - 1),
                    posterior_health=shadow_health,
                )
            )
            current_pose = np.asarray(planned.next_pose, dtype=np.float64)
            current_program = planned.shield_program
            timing = performance_record["timing_s"]
            assert isinstance(timing, dict)
            timing["sampler_health_gate"] = float(sampler_health_gate_elapsed_s)
            timing["planning"] = float(planning_elapsed_s)
            timing["planning_breakdown"] = planning_stage_wall_s
            timing["station_wall_including_planning"] = float(
                time.perf_counter() - station_wall_start_s
            )
            performance_writer.append(performance_record)
        live_session.complete_live_state(
            diagnostics_extensions=_completion_diagnostics_extensions(
                stop_reason=stop_reason,
                budget=budget,
                adaptive_stop_status=latest_adaptive_stop_status,
                particle_adequacy=particle_adequacy,
            )
        )
        published = client.finalize_log()
        log = load_measurement_log(published.path)
        if published.record_count != len(log.records):
            raise RuntimeError("Published MeasurementLog record count is inconsistent.")
        live_session.bind_published_log(log)
        result = PFClosedLoopResult(
            measurement_log_path=log.path.resolve(),
            pf_output_dir=target,
            run_id=log.run_id,
            record_count=len(log.records),
            station_count=log.station_view().station_count,
            stop_reason=stop_reason,
            sampler_quality_status=str(particle_adequacy["sampler_quality"]["status"]),
        )
        resources.close()
        if cui_split_viz is not None:
            visualizer = cui_split_viz
            if not cui_frame_enqueued:
                raise RuntimeError(
                    "Enabled CUI renderer received no frame for this PF run."
                )
            visualizer.close()
            cui_split_viz = None
            artifact_paths = (
                getattr(visualizer, "latest_overview_path", None),
                getattr(visualizer, "latest_robot_path", None),
                getattr(visualizer, "latest_pf_path", None),
                getattr(visualizer, "latest_pf_labeled_path", None),
                getattr(visualizer, "latest_spectrum_path", None),
            )
            if not all(isinstance(path, Path) for path in artifact_paths):
                raise RuntimeError("CUI renderer omitted canonical artifact paths.")
            missing_artifacts = [
                path.as_posix() for path in artifact_paths if not path.is_file()
            ]
            if missing_artifacts:
                raise RuntimeError(
                    "CUI renderer acknowledged its final frame without artifacts: "
                    f"{missing_artifacts}."
                )
            publish_final_cui_split_views(
                source_overview_path=artifact_paths[0],
                source_robot_path=artifact_paths[1],
                source_pf_path=artifact_paths[2],
                source_pf_labeled_path=artifact_paths[3],
                source_spectrum_path=artifact_paths[4],
                final_overview_path=(staging_dir / "final_experiment_overview.png"),
                final_robot_path=staging_dir / "final_robot_2d.png",
                final_pf_path=staging_dir / "final_pf_3d.png",
                final_pf_labeled_path=(staging_dir / "final_pf_3d_labeled.png"),
                final_spectrum_path=staging_dir / "final_spectrum.png",
            )
        if cui_server_handle is not None:
            server_handle = cui_server_handle
            server_handle.close()
            cui_server_handle = None
        atomic_write_bytes(
            staging_dir / "closed_loop_result.json",
            _strict_live_artifact_json_bytes(
                result.to_dict(),
                artifact_name="PF closed-loop result",
            ),
        )
        if len(cui_route_records) != len(log.records):
            raise RuntimeError(
                "PF figure route record count differs from the finalized log."
            )
        figure_route = cui_route_from_records(cui_route_records)
        atomic_write_bytes(
            staging_dir / "pf_figure_data.json",
            _strict_live_artifact_json_bytes(
                _pf_result_figure_data_payload(
                    figure_route,
                    run_id=log.run_id,
                    measurement_log_sha256=log.log_sha256,
                ),
                artifact_name="PF result figure data",
            ),
        )
        live_session._publish_bound_result_into_staging(staging_dir)
        bundle_publisher.publish()
        staging_dir = None
        return result
    except BaseException as exc:
        secondary_failures: list[dict[str, str]] = []
        if client is not None:
            try:
                client.abort()
            except BaseException as abort_exc:
                secondary_failures.append(
                    {
                        "operation": "runtime_abort",
                        "error_type": type(abort_exc).__name__,
                        "message": str(abort_exc),
                    }
                )
        try:
            resources.close()
        except BaseException as close_exc:
            secondary_failures.append(
                {
                    "operation": "resource_close",
                    "error_type": type(close_exc).__name__,
                    "message": str(close_exc),
                }
            )
        if cui_split_viz is not None:
            visualizer = cui_split_viz
            try:
                visualizer.close()
            except BaseException as cui_exc:
                secondary_failures.append(
                    {
                        "operation": "cui_close",
                        "error_type": type(cui_exc).__name__,
                        "message": str(cui_exc),
                    }
                )
            else:
                cui_split_viz = None
        if cui_server_handle is not None:
            server_handle = cui_server_handle
            try:
                server_handle.close()
            except BaseException as server_exc:
                secondary_failures.append(
                    {
                        "operation": "cui_server_close",
                        "error_type": type(server_exc).__name__,
                        "message": str(server_exc),
                    }
                )
            else:
                cui_server_handle = None
        failure_diagnostics_path: Path | None = None
        if staging_dir is not None:
            try:
                failure_diagnostics_path = _publish_failure_diagnostics(
                    target=target,
                    staging_dir=staging_dir,
                    stage_suffix=stage_suffix,
                    estimator=estimator,
                    primary_error=exc,
                )
            except BaseException as diagnostics_exc:
                secondary_failures.append(
                    {
                        "operation": "failure_diagnostics_publish",
                        "error_type": type(diagnostics_exc).__name__,
                        "message": str(diagnostics_exc),
                    }
                )
        try:
            atomic_write_bytes(
                failure_receipt_path,
                _strict_live_artifact_json_bytes(
                    {
                        "schema_version": 2,
                        "status": "failed",
                        "output_bundle_published": target.exists(),
                        "failure_diagnostics_published": (
                            failure_diagnostics_path is not None
                        ),
                        "failure_diagnostics_dir": (
                            None
                            if failure_diagnostics_path is None
                            else failure_diagnostics_path.as_posix()
                        ),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "secondary_failures": secondary_failures,
                    },
                    artifact_name="PF closed-loop failure receipt",
                ),
            )
        except BaseException as marker_exc:
            secondary_failures.append(
                {
                    "operation": "failure_receipt_write",
                    "error_type": type(marker_exc).__name__,
                    "message": str(marker_exc),
                }
            )
        for failure in secondary_failures:
            exc.add_note(
                "Secondary failure during PF abort: "
                f"{failure['operation']}: {failure['error_type']}: "
                f"{failure['message']}"
            )
        raise
    finally:
        if cui_split_viz is not None:
            try:
                cui_split_viz.close()
            except BaseException:
                pass
        if cui_server_handle is not None:
            try:
                cui_server_handle.close()
            except BaseException:
                pass
        resources.close()
        bundle_publisher.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public PF adaptive-controller command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-socket", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cui-truth-overlay-socket", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(None if argv is None else list(argv))
    result = run_pf_closed_loop(
        args.session_socket,
        runtime_root=args.runtime_root,
        cui_truth_overlay_socket_path=args.cui_truth_overlay_socket,
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
