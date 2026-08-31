"""Differential Shield-Signature Path Planning.

DSS-PP plans over a joint robot-pose and shield-program action. It samples
future spectra from the same validated generative model and evaluates them
with the same sole full-spectrum likelihood used by the online PF.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t
from scipy.spatial import cKDTree

from measurement.continuous_kernels import ContinuousKernel
from pf.estimator import JointPlanningParticles, RotatingShieldPFEstimator
from pf.full_spectrum import (
    TRANSPORT_FEATURE_ORDER,
    TorchPredictiveFullSpectrumModel,
    catalog_line_layout_by_isotope,
)
from pf.randomness import named_random_generator, named_stream_seed

# Private algorithm imports support the production planner implementation.
from planning.dss_candidates import _free_cell_centers
from planning.dss_eig import (  # noqa: F401
    _program_pair_id_matrix,
    _program_view_mask,
    _finite_sphere_geometric_terms_batched,
    _information_gain_from_log_likelihood,
    _finite_sample_information_gain_upper_bound,
    _joint_program_action_layout,
    _selected_program_transport_components,
    _full_spectrum_information_gain,
    _dss_eig_state_chunk_size,
    _dss_eig_likelihood_action_chunk_size,
    _dss_eig_action_batch_size,
    _is_dss_eig_memory_error,
    _release_dss_gpu_cache,
    _dss_accelerator_memory_snapshot,
)
from planning.dss_modes import (  # noqa: F401
    _normalise_weights,
    _posterior_mode_weights,
    _isotope_presence_probability,
    _flattened_posterior_mode_weights,
    _presence_weighted_rows,
    _planning_rng,
    _validate_mode_capacity,
    _validate_eig_likelihood_contract,
    _continuous_kernel_for_estimator,
    _weighted_surface_medoid_index,
    _cluster_source_samples,
    extract_signature_modes,
    _official_signature_modes,
)
from planning.dss_spatial import (  # noqa: F401
    _elevation_pair_indices_and_weights,
    _local_orbit_gains_batch,
    _elevation_condition_gains_batch,
    _node_path_lengths_batch,
    _filter_path_reachable_stations,
    _align_candidate_values,
    _coverage_gain_fractions_batch,
    _station_revisit_penalties_batch,
    _bearing_diversity_gain,
    _bearing_diversity_gains_batch,
    _frontier_band_gains_batch,
    _route_turn_penalty,
    _route_turn_penalties_batch,
    _filter_station_separation,
)
from planning.dss_types import (
    DSSPPConfig,
    DSSPPNode,
    DSSPPResult,
    SignatureMode,
    _DeviceJointProgramSpectrumComponents,
    _JointProgramSpectrumComponents,
    _PendingDSSPPNode,
    estimate_lambda_cost,
)
from planning.adaptive_shortlist import select_adaptive_pose_shortlist
from planning.conditional_eig import prepare_conditional_observation_cache
from planning.conditional_greedy import (
    ConditionalGreedyResult,
    evaluate_subset_information_gain_torch,
    select_conditional_greedy_programs,
)
from planning.conditional_memory import plan_conditional_pose_chunk
from planning.pose_scoring import compose_pose_scores
from planning.program_types import ShieldProgram
from planning.shield_view_count_shadow import select_shield_view_count_shadow


__all__ = [
    "DSSPPConfig",
    "DSSPPNode",
    "DSSPPResult",
    "SignatureMode",
    "estimate_lambda_cost",
    "extract_signature_modes",
    "ShieldProgram",
    "select_dss_pp_next_station",
]


def _node_diagnostic_payload(node: DSSPPNode, rank: int) -> dict[str, object]:
    """Return a JSON-serializable diagnostic payload for one DSS-PP node."""
    return {
        "rank": int(rank),
        "pose_index": int(node.pose_index),
        "pose_xyz": [float(value) for value in np.asarray(node.pose_xyz, dtype=float)],
        "program_name": str(node.program.name),
        "program_kind": str(node.program.kind),
        "pair_ids": [int(value) for value in node.program.pair_ids],
        "score": float(node.score),
        "static_score": float(node.static_score),
        "distance_weight": float(node.distance_weight),
        "information_gain": float(node.information_gain),
        "coverage_gain": float(node.coverage_gain),
        "revisit_penalty": float(node.revisit_penalty),
        "bearing_diversity_gain": float(node.bearing_diversity_gain),
        "frontier_gain": float(node.frontier_gain),
        "turn_penalty": float(node.turn_penalty),
        "local_orbit_gain": float(node.local_orbit_gain),
        "elevation_condition_gain": float(node.elevation_condition_gain),
    }


def _component_leader_payloads(
    nodes: Sequence[DSSPPNode],
) -> dict[str, dict[str, object]]:
    """Return best-node diagnostics for individual DSS-PP score components."""
    node_list = list(nodes)
    if not node_list:
        return {}
    selectors: dict[str, Any] = {
        "score": lambda node: float(node.score),
        "information_gain": lambda node: float(node.information_gain),
        "coverage": lambda node: float(node.coverage_gain),
        "bearing_diversity": lambda node: float(node.bearing_diversity_gain),
        "frontier": lambda node: float(node.frontier_gain),
        "local_orbit": lambda node: float(node.local_orbit_gain),
        "elevation_condition": lambda node: float(node.elevation_condition_gain),
    }
    leaders: dict[str, dict[str, object]] = {}
    for name, selector in selectors.items():
        finite_nodes = [
            node for node in node_list if np.isfinite(float(selector(node)))
        ]
        if not finite_nodes:
            continue
        leader = max(finite_nodes, key=lambda node: float(selector(node)))
        payload = _node_diagnostic_payload(leader, 1)
        payload["component_value"] = float(selector(leader))
        leaders[name] = payload
    return leaders


def _full_spectrum_joint_program_components(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    joint_particles: JointPlanningParticles,
    *,
    live_time_s: float,
    detector_aperture_samples: int,
    device_resident: bool = False,
    working_memory_budget_bytes: int | None = None,
) -> _JointProgramSpectrumComponents | _DeviceJointProgramSpectrumComponents:
    """Build batched source-resolved inputs for the shared spectrum model."""
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
        or len(programs) != int(detectors.shape[0])
        or not programs
    ):
        raise ValueError(
            "Full-spectrum DSS actions require one finite detector position "
            "per nonempty shield program."
        )
    view_count = len(programs[0].pair_ids)
    if view_count <= 0 or any(
        len(program.pair_ids) != view_count for program in programs
    ):
        raise ValueError(
            "One full-spectrum DSS batch requires equal nonzero view counts."
        )
    resolved_live_time = float(live_time_s)
    if not np.isfinite(resolved_live_time) or resolved_live_time <= 0.0:
        raise ValueError("DSS live_time_s must be finite and positive.")
    if not isinstance(device_resident, bool):
        raise TypeError("device_resident must be a boolean.")
    if working_memory_budget_bytes is not None and (
        isinstance(working_memory_budget_bytes, bool)
        or not isinstance(working_memory_budget_bytes, (int, np.integer))
        or int(working_memory_budget_bytes) <= 0
    ):
        raise ValueError("working_memory_budget_bytes must be a positive integer.")
    model = estimator.authenticated_full_spectrum_model()
    isotope_order = tuple(str(value) for value in joint_particles.isotope_order)
    if isotope_order != tuple(sorted(str(value) for value in estimator.isotopes)):
        raise ValueError("Joint planning isotope order must equal the estimator order.")
    particle_weights = _normalise_weights(
        np.asarray(joint_particles.weights_n, dtype=np.float64)
    )
    particle_count = int(particle_weights.size)
    line_identity = tuple(model.line_identity)
    line_count = len(line_identity)
    catalog_layout = catalog_line_layout_by_isotope(model, isotope_order)
    feature_order = tuple(str(value) for value in model.transport_feature_order)
    if feature_order != TRANSPORT_FEATURE_ORDER:
        raise ValueError("DSS and PF transport feature orders differ.")
    impact_edges = np.asarray(
        model.detector_impact_parameter_edges_fraction,
        dtype=np.float64,
    )
    impact_phase_count = int(impact_edges.size - 1)
    slot_counts = {
        isotope: int(
            np.asarray(
                joint_particles.strengths_nk_by_isotope[isotope],
                dtype=np.float64,
            ).shape[1]
        )
        for isotope in isotope_order
    }
    source_slot_count = int(sum(slot_counts.values()))
    action_count = int(detectors.shape[0])
    component_shape = (
        action_count,
        particle_count,
        view_count,
        source_slot_count,
        line_count,
    )
    if device_resident:
        import torch

        if not bool(estimator.pf_config.use_gpu):
            raise ValueError(
                "Device-resident DSS components require the configured GPU path."
            )
        component_device = torch.device(str(estimator.pf_config.gpu_device))
        total_components = torch.zeros(
            component_shape,
            device=component_device,
            dtype=torch.float64,
        )
        uncollided_components = torch.zeros_like(total_components)
        feature_components = torch.zeros(
            component_shape + (len(feature_order),),
            device=component_device,
            dtype=torch.float64,
        )
    else:
        total_components = np.zeros(component_shape, dtype=np.float64)
        uncollided_components = np.zeros_like(total_components)
        feature_components = np.zeros(
            component_shape + (len(feature_order),),
            dtype=np.float64,
        )
    pair_ids = _program_pair_id_matrix(programs)
    orientation_count = int(estimator.num_orientations)
    if (
        pair_ids.shape != (action_count, view_count)
        or np.any(pair_ids < 0)
        or np.any(pair_ids >= orientation_count**2)
    ):
        raise ValueError("DSS shield program contains an invalid pair id.")
    if (
        isinstance(detector_aperture_samples, bool)
        or not isinstance(detector_aperture_samples, (int, np.integer))
        or int(detector_aperture_samples) < 1
    ):
        raise ValueError("detector_aperture_samples must be a positive integer.")
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=int(detector_aperture_samples),
    )
    slot_offset = 0
    for isotope in isotope_order:
        positions = np.asarray(
            joint_particles.positions_nk3_by_isotope[isotope],
            dtype=np.float64,
        )
        raw_chart_ids = np.asarray(
            joint_particles.surface_chart_ids_nk_by_isotope[isotope],
        )
        surface_uv = np.asarray(
            joint_particles.surface_uv_nk2_by_isotope[isotope],
            dtype=np.float64,
        )
        strengths = np.asarray(
            joint_particles.strengths_nk_by_isotope[isotope],
            dtype=np.float64,
        )
        source_mask = np.asarray(
            joint_particles.source_mask_nk_by_isotope[isotope],
            dtype=bool,
        )
        slot_count = slot_counts[isotope]
        if (
            positions.shape != (particle_count, slot_count, 3)
            or strengths.shape != (particle_count, slot_count)
            or source_mask.shape != strengths.shape
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strengths.shape
            or surface_uv.shape != strengths.shape + (2,)
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(surface_uv))
            or np.any(~np.isfinite(strengths))
            or np.any(strengths < 0.0)
            or np.any(strengths[~source_mask] != 0.0)
        ):
            raise ValueError(
                "Joint full-spectrum planning particles contain an invalid "
                f"state for {isotope!r}."
            )
        isotope_layout = catalog_layout[isotope]
        global_line_indices = np.asarray(
            isotope_layout.global_columns,
            dtype=np.int64,
        )
        local_line_indices = np.asarray(
            isotope_layout.transport_line_indices,
            dtype=np.int64,
        )
        branching_weights = np.asarray(
            isotope_layout.branching_weights,
            dtype=np.float64,
        )
        (
            configured_branching,
            configured_energies,
            configured_mu_fe,
            configured_mu_pb,
        ) = kernel.line_transport_contract(
            isotope,
            local_line_indices,
        )
        if not all(
            np.allclose(
                actual,
                np.asarray(expected, dtype=np.float64),
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            for actual, expected in (
                (configured_branching, isotope_layout.branching_weights),
                (configured_energies, isotope_layout.energies_keV),
                (configured_mu_fe, isotope_layout.mu_fe_cm_inv),
                (configured_mu_pb, isotope_layout.mu_pb_cm_inv),
            )
        ):
            raise RuntimeError(
                "DSS, PF, and spectrum-model catalog transport rows differ "
                f"for {isotope!r}."
            )
        if slot_count == 0:
            continue
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        active_particle_indices, active_slot_indices = np.nonzero(source_mask)
        if active_particle_indices.size == 0:
            slot_offset += slot_count
            continue
        active_transport_positions = estimator.surface_transport_positions(
            isotope,
            positions[source_mask],
            chart_ids[source_mask],
            surface_uv[source_mask],
        )
        component_arrays = _selected_program_transport_components(
            kernel,
            isotope=isotope,
            detector_positions=detectors,
            pair_ids_av=pair_ids,
            sources=active_transport_positions,
            positive_line_indices=local_line_indices,
            impact_parameter_edges_fraction=impact_edges,
            device_resident=device_resident,
            working_memory_budget_bytes=working_memory_budget_bytes,
        )
        expected_program_shape = (
            action_count,
            view_count,
            int(active_particle_indices.size),
            int(global_line_indices.size),
        )

        def _local_component(
            field_name: str,
            component_values: dict[str, object] = component_arrays,
        ) -> object:
            """Return one validated reshaped physical component."""
            if device_resident:
                import torch

                values = torch.as_tensor(
                    component_values[field_name],
                    device=total_components.device,
                    dtype=total_components.dtype,
                )
                if tuple(values.shape) != expected_program_shape:
                    raise RuntimeError(
                        f"Full-spectrum component {field_name!r} has an invalid shape."
                    )
            else:
                values = np.asarray(
                    component_values[field_name],
                    dtype=np.float64,
                )
                if (
                    values.shape != expected_program_shape
                    or np.any(~np.isfinite(values))
                    or np.any(values < 0.0)
                ):
                    raise RuntimeError(
                        f"Full-spectrum component {field_name!r} is invalid."
                    )
            return values

        total_local = _local_component("total_kernel")
        uncollided_local = _local_component("uncollided_kernel")
        tau_fe = _local_component("tau_fe")
        tau_pb = _local_component("tau_pb")
        tau_obstacle = _local_component("tau_obstacle")
        tau_obstacle_compton = _local_component("tau_obstacle_compton")
        distance_m = _local_component("distance_m")
        raw_impact = component_arrays["uncollided_impact_fractions"]
        expected_impact_shape = expected_program_shape + (impact_phase_count,)
        if device_resident:
            import torch

            impact_fractions = torch.as_tensor(
                raw_impact,
                device=total_components.device,
                dtype=total_components.dtype,
            )
            if tuple(impact_fractions.shape) != expected_impact_shape:
                raise RuntimeError(
                    "Full-spectrum detector-impact component shape is invalid."
                )
        else:
            impact_fractions = np.asarray(raw_impact, dtype=np.float64)
            if (
                impact_fractions.shape != expected_impact_shape
                or np.any(~np.isfinite(impact_fractions))
                or np.any(impact_fractions < 0.0)
            ):
                raise RuntimeError(
                    "Full-spectrum detector-impact component is invalid."
                )
        if device_resident:
            import torch

            source_scale = torch.as_tensor(
                strengths[source_mask],
                device=total_components.device,
                dtype=total_components.dtype,
            ).reshape(1, 1, -1, 1) * torch.as_tensor(
                branching_weights,
                device=total_components.device,
                dtype=total_components.dtype,
            ).reshape(1, 1, 1, -1)
        else:
            source_scale = (
                strengths[source_mask][None, None, :, None]
                * branching_weights[None, None, None, :]
            )
        total_local *= source_scale
        uncollided_local *= source_scale
        local_features = (
            torch.cat(
                (
                    torch.stack(
                        (
                            tau_fe,
                            tau_pb,
                            tau_obstacle,
                            tau_obstacle_compton,
                            distance_m,
                        ),
                        dim=-1,
                    ),
                    impact_fractions,
                ),
                dim=-1,
            )
            if device_resident
            else np.concatenate(
                (
                    np.stack(
                        (
                            tau_fe,
                            tau_pb,
                            tau_obstacle,
                            tau_obstacle_compton,
                            distance_m,
                        ),
                        axis=-1,
                    ),
                    impact_fractions,
                ),
                axis=-1,
            )
        )
        active_global_slots = int(slot_offset) + active_slot_indices
        if device_resident:
            action_target = torch.arange(
                action_count,
                device=total_components.device,
                dtype=torch.long,
            )
            view_target = torch.arange(
                view_count,
                device=total_components.device,
                dtype=torch.long,
            )
            particle_target = torch.as_tensor(
                active_particle_indices,
                device=total_components.device,
                dtype=torch.long,
            )
            slot_target = torch.as_tensor(
                active_global_slots,
                device=total_components.device,
                dtype=torch.long,
            )
            line_target = torch.as_tensor(
                global_line_indices,
                device=total_components.device,
                dtype=torch.long,
            )
        else:
            action_target = np.arange(action_count, dtype=np.int64)
            view_target = np.arange(view_count, dtype=np.int64)
            particle_target = active_particle_indices
            slot_target = active_global_slots
            line_target = global_line_indices
        target = (
            action_target[:, None, None, None],
            particle_target[None, None, :, None],
            view_target[None, :, None, None],
            slot_target[None, None, :, None],
            line_target[None, None, None, :],
        )
        total_components[target] = total_local
        uncollided_components[target] = uncollided_local
        feature_components[target] = local_features
        del (
            component_arrays,
            _local_component,
            total_local,
            uncollided_local,
            tau_fe,
            tau_pb,
            tau_obstacle,
            tau_obstacle_compton,
            distance_m,
            impact_fractions,
            source_scale,
            local_features,
            target,
            action_target,
            view_target,
            particle_target,
            slot_target,
            line_target,
        )
        slot_offset += slot_count
    if device_resident:
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(total_components)),
                torch.any(~torch.isfinite(uncollided_components)),
                torch.any(~torch.isfinite(feature_components)),
                torch.any(total_components < 0.0),
                torch.any(uncollided_components < 0.0),
                torch.any(feature_components < 0.0),
                torch.any(uncollided_components > total_components + 1.0e-10),
            )
        ).any()
        if bool(invalid.item()):
            raise RuntimeError(
                "Full-spectrum DSS device transport components are invalid."
            )
        return _DeviceJointProgramSpectrumComponents(
            total_pnvsl=total_components,
            uncollided_pnvsl=uncollided_components,
            features_pnvslf=feature_components,
            live_times_v=torch.full(
                (view_count,),
                resolved_live_time,
                device=total_components.device,
                dtype=total_components.dtype,
            ),
            contract_hash_sha256=str(model.contract_hash_sha256),
        )
    if np.any(uncollided_components > total_components + 1.0e-10):
        raise RuntimeError("Full-spectrum DSS transport violates uncollided <= total.")
    return _JointProgramSpectrumComponents(
        total_pnvsl=np.ascontiguousarray(total_components),
        uncollided_pnvsl=np.ascontiguousarray(uncollided_components),
        features_pnvslf=np.ascontiguousarray(feature_components),
        live_times_v=np.full(
            view_count,
            resolved_live_time,
            dtype=np.float64,
        ),
        contract_hash_sha256=str(model.contract_hash_sha256),
    )


def _program_information_proxy_for_poses(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    *,
    config: DSSPPConfig,
    joint_particles: JointPlanningParticles,
    rng: np.random.Generator,
    eig_call_seed: int,
    diagnostics: dict[str, object] | None = None,
) -> NDArray[np.float64]:
    """Return reduced-posterior EIG using the exact PF spectrum law."""
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("Proxy detector positions must be finite and shaped Px3.")
    if not programs:
        return np.zeros((detectors.shape[0], 0), dtype=np.float64)
    gains_by_pose = _program_information_gains_for_poses(
        estimator,
        detectors,
        [list(programs) for _ in range(detectors.shape[0])],
        config=config,
        rng=rng,
        joint_particles=joint_particles,
        diagnostics=diagnostics,
        sample_count_override=config.proxy_eig_samples,
        eig_call_seed=eig_call_seed,
        memory_budget_bytes_override=config.proxy_memory_budget_bytes,
    )
    output = np.vstack(gains_by_pose)
    if output.shape != (detectors.shape[0], len(programs)):
        raise RuntimeError("Full-spectrum proxy returned an invalid action layout.")
    if np.any(~np.isfinite(output)) or np.any(output < 0.0):
        raise RuntimeError("Program information ranking proxies are invalid.")
    return output


def _program_information_gains_for_poses(
    estimator: RotatingShieldPFEstimator,
    detector_positions: NDArray[np.float64],
    programs_by_pose: Sequence[Sequence[ShieldProgram]],
    *,
    config: DSSPPConfig,
    rng: np.random.Generator,
    joint_particles: JointPlanningParticles | None = None,
    diagnostics: dict[str, object] | None = None,
    sample_count_override: int | None = None,
    eig_call_seed: int | None = None,
    memory_budget_bytes_override: int | None = None,
) -> list[NDArray[np.float64]]:
    """Return shared full-spectrum EIG for every candidate/program action."""
    pf_config = estimator.pf_config
    isotopes = tuple(sorted(str(value) for value in estimator.isotopes))
    if not isotopes or any(isotope not in estimator.filters for isotope in isotopes):
        raise RuntimeError(
            "Pure PF planning requires one initialized filter per isotope."
        )
    model = estimator.authenticated_full_spectrum_model()
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if detectors.size == 0:
        detectors = np.zeros((0, 3), dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("detector_positions must be finite and shaped Px3.")
    if len(programs_by_pose) != detectors.shape[0]:
        raise ValueError("programs_by_pose must match detector_positions.")
    outputs = [
        np.zeros(len(programs), dtype=np.float64) for programs in programs_by_pose
    ]
    if detectors.shape[0] == 0:
        return outputs
    if not isinstance(rng, np.random.Generator):
        raise TypeError("DSS EIG requires an explicit numpy random generator.")
    if eig_call_seed is None:
        resolved_eig_call_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int64).max,
                endpoint=False,
                dtype=np.int64,
            )
        )
    elif (
        isinstance(eig_call_seed, bool)
        or not isinstance(eig_call_seed, (int, np.integer))
        or int(eig_call_seed) < 0
    ):
        raise ValueError("eig_call_seed must be a nonnegative integer.")
    else:
        resolved_eig_call_seed = int(eig_call_seed)
    if joint_particles is None:
        if config.planning_particles is None:
            joint_particles = estimator.planning_joint_particles()
        else:
            joint_particles = estimator.planning_joint_particles(
                max_particles=config.planning_particles,
                method=config.planning_method,
                rng=rng,
            )
    if tuple(joint_particles.isotope_order) != isotopes:
        raise ValueError("Joint planning snapshot isotope order is inconsistent.")
    if int(np.asarray(joint_particles.weights_n).size) < 2:
        raise ValueError(
            "Exact DSS EIG requires at least two aligned posterior particles."
        )
    configured_samples = (
        pf_config.planning_eig_samples
        if sample_count_override is None
        else sample_count_override
    )
    if (
        isinstance(configured_samples, bool)
        or not isinstance(configured_samples, (int, np.integer))
        or int(configured_samples) <= 0
    ):
        raise ValueError("DSS EIG sample count must be a positive integer.")
    snapshot_index = len(estimator.measurements)
    use_gpu = bool(pf_config.use_gpu)
    gpu_device = str(pf_config.gpu_device)
    if use_gpu and (
        not isinstance(model, TorchPredictiveFullSpectrumModel)
        or not callable(getattr(model, "sample_predictive_torch", None))
    ):
        raise RuntimeError(
            "Torch DSS requires the shared device-resident predictive sampler."
        )
    if use_gpu and not callable(getattr(model, "cross_log_likelihood_torch", None)):
        raise RuntimeError("Torch DSS requires the shared Torch cross likelihood.")
    accelerator_memory_before = _dss_accelerator_memory_snapshot(
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    flattened_programs, action_pose_indices, pair_ids, view_mask, offsets = (
        _joint_program_action_layout(programs_by_pose)
    )
    if not flattened_programs:
        return outputs
    action_lengths = np.sum(view_mask, axis=1, dtype=np.int64)
    flattened_gains = np.zeros(len(flattened_programs), dtype=np.float64)
    particle_count = int(np.asarray(joint_particles.weights_n).size)
    source_slot_count = int(
        sum(
            np.asarray(joint_particles.strengths_nk_by_isotope[isotope]).shape[1]
            for isotope in isotopes
        )
    )
    line_count = len(tuple(model.line_identity))
    feature_count = len(tuple(model.transport_feature_order))
    impact_phase_count = sum(
        str(name).startswith("uncollided_impact_fraction_")
        for name in model.transport_feature_order
    )
    if impact_phase_count <= 0 or feature_count - impact_phase_count != 5:
        raise RuntimeError(
            "DSS full-spectrum transport requires the complete five-scalar "
            "physics and detector-impact feature contract."
        )
    memory_budget_bytes = (
        config.exact_eig_memory_budget_bytes
        if memory_budget_bytes_override is None
        else memory_budget_bytes_override
    )
    if (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, (int, np.integer))
        or int(memory_budget_bytes) <= 0
    ):
        raise ValueError("DSS EIG memory budget must be a positive integer.")
    memory_budget_bytes = int(memory_budget_bytes)
    sample_count = int(configured_samples)
    latent_rng = named_random_generator(
        resolved_eig_call_seed,
        "dss_pp",
        "joint_full_spectrum_eig",
        int(snapshot_index),
        "common_latent_particles",
    )
    common_latent_indices = latent_rng.choice(
        particle_count,
        size=sample_count,
        replace=True,
        p=_normalise_weights(np.asarray(joint_particles.weights_n, dtype=np.float64)),
    ).astype(np.int64, copy=False)
    action_seeds = np.asarray(
        [
            named_stream_seed(
                resolved_eig_call_seed,
                "dss_pp",
                "joint_full_spectrum_eig",
                int(snapshot_index),
                "canonical_action",
                *(float(value).hex() for value in detectors[int(pose_index)]),
                "pairs",
                *(int(pair_id) for pair_id in program.pair_ids),
            )
            & ((1 << 63) - 1)
            for pose_index, program in zip(
                action_pose_indices,
                flattened_programs,
            )
        ],
        dtype=np.int64,
    )
    memory_contracts: list[dict[str, int]] = []
    attempted_action_batch_sizes: list[int] = []
    successful_action_batch_sizes: list[int] = []
    successful_likelihood_action_chunk_sizes: list[int] = []
    successful_response_resident_bytes: list[int] = []
    successful_response_materialization_peak_bytes: list[int] = []
    successful_response_scratch_budget_bytes: list[int] = []
    oom_retry_events: list[dict[str, int]] = []
    for view_count_raw in np.unique(action_lengths):
        view_count = int(view_count_raw)
        selected_actions = np.flatnonzero(action_lengths == view_count)
        if view_count <= 0:
            raise ValueError("Every DSS shield program must contain a view.")
        action_detectors = detectors[action_pose_indices[selected_actions]]
        action_pairs = pair_ids[selected_actions, :view_count]
        lexicographic_keys = tuple(
            [action_pairs[:, column] for column in range(view_count - 1, -1, -1)]
            + [
                action_detectors[:, 2],
                action_detectors[:, 1],
                action_detectors[:, 0],
            ]
        )
        selected_actions = selected_actions[np.lexsort(lexicographic_keys)]
        state_chunk_size = _dss_eig_state_chunk_size(
            model,
            action_count=int(selected_actions.size),
            particle_count=particle_count,
            sample_count=sample_count,
            source_slot_count=max(source_slot_count, 1),
            view_count=view_count,
            memory_budget_bytes=memory_budget_bytes,
        )
        memory_contract: dict[str, int] = {}
        action_batch_size = _dss_eig_action_batch_size(
            model,
            action_count=int(selected_actions.size),
            particle_count=particle_count,
            sample_count=sample_count,
            source_slot_count=max(source_slot_count, 1),
            view_count=view_count,
            line_count=max(line_count, 1),
            feature_count=max(feature_count, 1),
            memory_budget_bytes=memory_budget_bytes,
            state_chunk_size=state_chunk_size,
            diagnostics=memory_contract,
        )
        memory_contracts.append(memory_contract)
        action_start = 0
        while action_start < int(selected_actions.size):
            action_stop = min(
                action_start + action_batch_size,
                int(selected_actions.size),
            )
            action_indices = selected_actions[action_start:action_stop]
            attempted_action_batch_sizes.append(int(action_indices.size))
            likelihood_action_chunk_size = _dss_eig_likelihood_action_chunk_size(
                model,
                action_count=int(action_indices.size),
                particle_count=particle_count,
                sample_count=sample_count,
                source_slot_count=max(source_slot_count, 1),
                view_count=view_count,
                state_chunk_size=state_chunk_size,
                memory_budget_bytes=memory_budget_bytes,
            )
            action_rng = named_random_generator(
                resolved_eig_call_seed,
                "dss_pp",
                "joint_full_spectrum_eig",
                int(snapshot_index),
                "sampler_interface_rng",
            )
            components = None
            retry_after_memory_error = False
            try:
                response_field_bytes = int(
                    int(action_indices.size)
                    * particle_count
                    * view_count
                    * max(source_slot_count, 1)
                    * max(line_count, 1)
                    * np.dtype(np.float64).itemsize
                )
                planner_destination_bytes = int(
                    (2 + max(feature_count, 1)) * response_field_bytes
                )
                # The host generic path preallocates seven scalar response fields
                # plus the complete detector-impact axis. The device path
                # allocates that selected result only after transport, so it is
                # not resident during runtime scratch. Dense all-pair requests
                # use less memory, but these bounds cover every action layout.
                selected_response_field_count = 7 + impact_phase_count
                selected_response_bytes = int(
                    (0 if use_gpu else selected_response_field_count)
                    * response_field_bytes
                )
                runtime_retained_field_count = 8 + impact_phase_count
                runtime_retained_bytes = int(
                    runtime_retained_field_count * response_field_bytes
                )
                response_resident_bytes = int(
                    planner_destination_bytes
                    + selected_response_bytes
                    + runtime_retained_bytes
                )
                response_assembly_peak_bytes = int(
                    response_resident_bytes
                    + (
                        selected_response_field_count
                        if use_gpu
                        else runtime_retained_field_count
                    )
                    * response_field_bytes
                )
                response_materialization_peak_bytes = int(
                    max(
                        response_assembly_peak_bytes,
                        response_resident_bytes,
                    )
                )
                if response_materialization_peak_bytes > memory_budget_bytes:
                    raise MemoryError(
                        "DSS response materialization exceeds the configured "
                        "phase budget."
                    )
                response_scratch_budget_bytes = (
                    memory_budget_bytes - response_resident_bytes
                )
                if response_scratch_budget_bytes <= 0:
                    raise MemoryError(
                        "DSS response buffers exhaust the configured phase "
                        "budget before transport scratch."
                    )
                components = _full_spectrum_joint_program_components(
                    estimator,
                    detectors[action_pose_indices[action_indices]],
                    [flattened_programs[int(index)] for index in action_indices],
                    joint_particles,
                    live_time_s=float(config.live_time_s),
                    detector_aperture_samples=int(config.detector_aperture_samples),
                    device_resident=use_gpu,
                    working_memory_budget_bytes=int(response_scratch_budget_bytes),
                )
                batch_gains = _full_spectrum_information_gain(
                    estimator,
                    components,
                    np.asarray(
                        joint_particles.weights_n,
                        dtype=np.float64,
                    ),
                    sample_count=sample_count,
                    rng=action_rng,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    latent_particle_indices=common_latent_indices,
                    action_seeds_a=action_seeds[action_indices],
                    action_chunk_size=likelihood_action_chunk_size,
                    state_chunk_size=state_chunk_size,
                )
                components = None
                flattened_gains[action_indices] = batch_gains
            except Exception as error:
                components = None
                if not _is_dss_eig_memory_error(error):
                    raise
                error.__traceback__ = None
                failed_action_batch_size = action_stop - action_start
                if failed_action_batch_size <= 1 and state_chunk_size <= 1:
                    raise RuntimeError(
                        "DSS exact EIG exhausted memory for one action even "
                        "after action and state-chunk reduction."
                    ) from error
                retry_after_memory_error = True
            if retry_after_memory_error:
                _release_dss_gpu_cache()
                reduced_action_batch_size = int(action_batch_size)
                reduced_state_chunk_size = int(state_chunk_size)
                if failed_action_batch_size > 1:
                    reduced_action_batch_size = max(
                        1,
                        failed_action_batch_size // 2,
                    )
                else:
                    reduced_state_chunk_size = max(
                        1,
                        state_chunk_size // 2,
                    )
                oom_retry_events.append(
                    {
                        "view_count": int(view_count),
                        "failed_action_batch_size": int(failed_action_batch_size),
                        "retry_action_batch_size": int(reduced_action_batch_size),
                        "failed_state_chunk_size": int(state_chunk_size),
                        "retry_state_chunk_size": int(reduced_state_chunk_size),
                    }
                )
                action_batch_size = int(reduced_action_batch_size)
                state_chunk_size = int(reduced_state_chunk_size)
                continue
            successful_action_batch_sizes.append(int(action_indices.size))
            successful_likelihood_action_chunk_sizes.append(
                int(likelihood_action_chunk_size)
            )
            successful_response_resident_bytes.append(int(response_resident_bytes))
            successful_response_materialization_peak_bytes.append(
                int(response_materialization_peak_bytes)
            )
            successful_response_scratch_budget_bytes.append(
                int(response_scratch_budget_bytes)
            )
            action_start = action_stop
    for pose_index in range(int(detectors.shape[0])):
        action_start = int(offsets[pose_index])
        action_stop = int(offsets[pose_index + 1])
        outputs[pose_index] = np.asarray(
            flattened_gains[action_start:action_stop],
            dtype=np.float64,
        )
    if diagnostics is not None:
        diagnostics.update(
            {
                "backend": "torch" if use_gpu else "numpy",
                "gpu_device": str(gpu_device) if use_gpu else "cpu",
                "bulk_device_resident": bool(use_gpu),
                "memory_budget_bytes": int(memory_budget_bytes),
                "accelerator_memory_before": accelerator_memory_before,
                "accelerator_memory_after": _dss_accelerator_memory_snapshot(
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                ),
                "memory_contracts": memory_contracts,
                "attempted_action_batch_sizes": attempted_action_batch_sizes,
                "successful_action_batch_sizes": successful_action_batch_sizes,
                "successful_likelihood_action_chunk_sizes": (
                    successful_likelihood_action_chunk_sizes
                ),
                "successful_response_resident_bytes": (
                    successful_response_resident_bytes
                ),
                "successful_response_materialization_peak_bytes": (
                    successful_response_materialization_peak_bytes
                ),
                "successful_response_scratch_budget_bytes": (
                    successful_response_scratch_budget_bytes
                ),
                "oom_retry_count": int(len(oom_retry_events)),
                "oom_retry_events": oom_retry_events,
            }
        )
    return outputs


def _static_station_program_score(
    *,
    coverage_norm: float,
    revisit_penalty: float,
    bearing_gain: float,
    frontier_gain: float,
    turn_penalty: float,
    local_orbit_gain: float,
    elevation_condition_gain: float,
    coverage_floor: float,
    config: DSSPPConfig,
) -> float:
    """Return geometry, route, and coverage utility without count reuse."""
    return float(
        float(config.lambda_coverage) * float(coverage_norm)
        + float(config.lambda_bearing_diversity) * float(bearing_gain)
        + float(config.lambda_frontier) * float(frontier_gain)
        + float(config.lambda_local_orbit) * float(local_orbit_gain)
        + float(config.lambda_elevation_condition)
        * float(np.log1p(max(elevation_condition_gain, 0.0)))
        - float(config.eta_revisit) * float(revisit_penalty)
        - float(config.lambda_turn_smoothness) * float(turn_penalty)
        - float(config.coverage_floor_weight)
        * max(0.0, float(coverage_floor) - float(coverage_norm)) ** 2
    )


def _evaluate_pose_index_from_context(
    pose_index_value: int,
    context: Mapping[str, object],
) -> tuple[int, float, list[_PendingDSSPPNode]]:
    """Materialize all program nodes for one already-vectorized station."""
    pose_index = int(pose_index_value)
    candidate_poses = np.asarray(context["candidate_poses"], dtype=float)
    path_lengths = np.asarray(context["path_lengths"], dtype=float)
    programs = cast(Sequence[ShieldProgram], context["programs"])
    config = cast(DSSPPConfig, context["config"])
    coverage_norm = np.asarray(context["coverage_norm"], dtype=float)
    coverage_raw = np.asarray(context["coverage_raw"], dtype=float)
    revisit_penalties = np.asarray(context["revisit_penalties"], dtype=float)
    bearing_gains = np.asarray(context["bearing_gains"], dtype=float)
    frontier_gains = np.asarray(context["frontier_gains"], dtype=float)
    turn_penalties = np.asarray(context["turn_penalties"], dtype=float)
    local_orbit_gains = np.asarray(context["local_orbit_gains"], dtype=float)
    elevation_condition_gains = np.asarray(
        context["elevation_condition_gains"],
        dtype=float,
    )
    coverage_floor = float(context["coverage_floor"])

    local_pending: list[_PendingDSSPPNode] = []
    local_cheap_score = -np.inf
    pose = candidate_poses[pose_index]
    if not np.isfinite(path_lengths[pose_index]):
        return (
            pose_index,
            local_cheap_score,
            local_pending,
        )
    # Every candidate program is compared by the exact joint EIG below.  The
    # static term is deliberately restricted to geometry, route, and coverage
    # so the same prospective counts cannot be scored a second time.
    for program in programs:
        static_score = _static_station_program_score(
            coverage_norm=float(coverage_norm[pose_index]),
            revisit_penalty=float(revisit_penalties[pose_index]),
            bearing_gain=float(bearing_gains[pose_index]),
            frontier_gain=float(frontier_gains[pose_index]),
            turn_penalty=float(turn_penalties[pose_index]),
            local_orbit_gain=float(local_orbit_gains[pose_index]),
            elevation_condition_gain=float(elevation_condition_gains[pose_index]),
            coverage_floor=coverage_floor,
            config=config,
        )
        local_cheap_score = max(float(local_cheap_score), float(static_score))
        local_pending.append(
            _PendingDSSPPNode(
                pose_index=pose_index,
                pose_xyz=pose.copy(),
                program=program,
                static_score=float(static_score),
                coverage_gain=float(coverage_raw[pose_index]),
                revisit_penalty=float(revisit_penalties[pose_index]),
                bearing_diversity_gain=float(bearing_gains[pose_index]),
                frontier_gain=float(frontier_gains[pose_index]),
                turn_penalty=float(turn_penalties[pose_index]),
                local_orbit_gain=float(local_orbit_gains[pose_index]),
                elevation_condition_gain=float(elevation_condition_gains[pose_index]),
            )
        )
    return (
        pose_index,
        local_cheap_score,
        local_pending,
    )


def _materialize_pose_nodes(
    eval_indices: Sequence[int],
    *,
    context: dict[str, object],
) -> list[tuple[int, float, list[_PendingDSSPPNode]]]:
    """Materialize nodes after all numerical candidate terms were batched."""
    return [
        _evaluate_pose_index_from_context(int(index), context) for index in eval_indices
    ]


def _response_equivalent_surface_coverage_masks(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    surface_points_xyz: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    reference_radius_m: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return candidate and visited-station surface-coverage masks.

    Coverage is deliberately a shield-independent spatial exploration term:
    both prospective and visited detector stations use the same finite-detector
    distance-plus-obstacle response before Fe/Pb attenuation. Shield-specific
    evidence is evaluated exactly once by the joint full-spectrum EIG. Keeping
    both sides of this coverage state on the same contract prevents a station
    from being repeatedly rewarded as "new" merely because its executed shield
    pair differs from the optimistic candidate calculation.
    """
    surfaces = np.asarray(surface_points_xyz, dtype=float)
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if surfaces.ndim != 2 or surfaces.shape[1] != 3:
        raise ValueError("surface_points_xyz must be shaped (S, 3).")
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shaped (C, 3).")
    isotope_names = tuple(str(value) for value in estimator.isotopes)
    if not isotope_names:
        raise ValueError("Surface observability requires configured isotopes.")
    radius = float(reference_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("reference_radius_m must be finite and positive.")
    detector_radius = float(kernel.detector_radius_m)
    reference = float(
        _finite_sphere_geometric_terms_batched(
            np.zeros((1, 3), dtype=float),
            np.asarray([[radius, 0.0, 0.0]], dtype=float),
            detector_radius_m=detector_radius,
        )[0, 0]
    )
    if not np.isfinite(reference) or reference <= 0.0:
        raise RuntimeError("Surface observability reference response is invalid.")
    if surfaces.shape[0] == 0:
        return (
            np.zeros((candidates.shape[0], 0), dtype=bool),
            np.zeros(0, dtype=bool),
        )

    candidate_pairs = cKDTree(candidates).sparse_distance_matrix(
        cKDTree(surfaces),
        max_distance=float(np.nextafter(radius, np.inf)),
        output_type="coo_matrix",
    )
    candidate_rows = np.asarray(candidate_pairs.row, dtype=np.int64)
    candidate_surface_ids = np.asarray(
        candidate_pairs.col,
        dtype=np.int64,
    )
    candidate_pair_covered = np.ones(
        candidate_rows.size,
        dtype=bool,
    )
    for isotope in isotope_names:
        values = np.asarray(
            kernel.kernel_values_unshielded_for_detector_source_pairs(
                isotope=isotope,
                detector_positions=candidates[candidate_rows],
                sources=surfaces[candidate_surface_ids],
            ),
            dtype=np.float64,
        ).reshape(-1)
        if (
            values.shape != (candidate_rows.size,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise RuntimeError(
                "Surface observability kernel returned invalid matched "
                "unshielded values."
            )
        candidate_pair_covered &= values >= reference
    candidate_covered = np.zeros(
        (candidates.shape[0], surfaces.shape[0]),
        dtype=bool,
    )
    candidate_covered[
        candidate_rows[candidate_pair_covered],
        candidate_surface_ids[candidate_pair_covered],
    ] = True

    records = tuple(estimator.measurements)
    if not records:
        acquired_covered = np.zeros(surfaces.shape[0], dtype=bool)
    else:
        acquired_detectors = np.unique(
            np.asarray(
                [record.detector_position_xyz_m for record in records],
                dtype=float,
            ).reshape(-1, 3),
            axis=0,
        )
        if np.any(~np.isfinite(acquired_detectors)):
            raise ValueError(
                "Acquired detector positions must contain finite coordinates."
            )
        acquired_pairs = cKDTree(acquired_detectors).sparse_distance_matrix(
            cKDTree(surfaces),
            max_distance=float(np.nextafter(radius, np.inf)),
            output_type="coo_matrix",
        )
        acquired_rows = np.asarray(acquired_pairs.row, dtype=np.int64)
        acquired_surface_ids = np.asarray(
            acquired_pairs.col,
            dtype=np.int64,
        )
        acquired_min_best = np.full(surfaces.shape[0], np.inf, dtype=float)
        for isotope in isotope_names:
            isotope_best = np.zeros(surfaces.shape[0], dtype=float)
            values = np.asarray(
                kernel.kernel_values_unshielded_for_detector_source_pairs(
                    isotope=isotope,
                    detector_positions=acquired_detectors[acquired_rows],
                    sources=surfaces[acquired_surface_ids],
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                values.shape != (acquired_rows.size,)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise RuntimeError(
                    "Surface observability kernel returned invalid matched "
                    "unshielded acquired-station values."
                )
            np.maximum.at(
                isotope_best,
                acquired_surface_ids,
                values,
            )
            acquired_min_best = np.minimum(
                acquired_min_best,
                isotope_best / reference,
            )
        acquired_covered = acquired_min_best >= 1.0
    return candidate_covered, acquired_covered


def _response_equivalent_surface_coverage_gains(
    *,
    kernel: ContinuousKernel,
    estimator: RotatingShieldPFEstimator,
    surface_points_xyz: NDArray[np.float64],
    surface_area_weights_m2: NDArray[np.float64],
    candidate_poses_xyz: NDArray[np.float64],
    reference_radius_m: float,
) -> NDArray[np.float64]:
    """Return new physically observable surface-area fractions by candidate."""
    candidate_covered, acquired_covered = _response_equivalent_surface_coverage_masks(
        kernel=kernel,
        estimator=estimator,
        surface_points_xyz=surface_points_xyz,
        candidate_poses_xyz=candidate_poses_xyz,
        reference_radius_m=reference_radius_m,
    )
    if candidate_covered.shape[1] == 0:
        return np.zeros(candidate_covered.shape[0], dtype=float)
    area_weights = np.asarray(
        surface_area_weights_m2,
        dtype=np.float64,
    ).reshape(-1)
    if (
        area_weights.shape != (candidate_covered.shape[1],)
        or np.any(~np.isfinite(area_weights))
        or np.any(area_weights <= 0.0)
    ):
        raise ValueError(
            "Surface coverage requires one finite positive physical area "
            "weight per quadrature point."
        )
    total_area = float(np.sum(area_weights, dtype=np.float64))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("Surface coverage total physical area must be positive.")
    newly_covered = candidate_covered & ~acquired_covered[None, :]
    return (
        np.einsum(
            "cs,s->c",
            newly_covered,
            area_weights,
            optimize=True,
        )
        / total_area
    )


def _compose_transition_score(
    *,
    node: DSSPPNode,
    previous_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    config: DSSPPConfig,
    travel_time_override_s: float | None = None,
    horizontal_time_override_s: float | None = None,
    mast_vertical_time_override_s: float | None = None,
    settling_time_override_s: float | None = None,
    path_length_override_m: float | None = None,
) -> tuple[float, float]:
    """Return node score and path length for a specific predecessor."""
    if path_length_override_m is None:
        path_length = float(
            _node_path_lengths_batch(
                map_api,
                previous_pose_xyz,
                np.asarray(node.pose_xyz, dtype=np.float64).reshape(1, 3),
            )[0]
        )
    else:
        path_length = float(path_length_override_m)
        if np.isnan(path_length) or path_length < 0.0:
            raise ValueError("path_length_override_m must be nonnegative and not NaN.")
    if not np.isfinite(path_length):
        return -float("inf"), float("inf")
    motion_components = (
        horizontal_time_override_s,
        mast_vertical_time_override_s,
        settling_time_override_s,
    )
    if all(value is None for value in motion_components):
        if travel_time_override_s is not None:
            raise ValueError(
                "A motion-time total cannot be supplied without all components."
            )
        if any(
            float(weight) != 0.0
            for weight in (
                config.lambda_horizontal_time,
                config.lambda_mast_vertical_time,
                config.lambda_settling_time,
            )
        ):
            raise ValueError(
                "Nonzero motion weights require runtime-authored motion components."
            )
        motion_penalty = 0.0
    elif any(value is None for value in motion_components):
        raise ValueError("Motion-time component overrides must be supplied together.")
    else:
        if travel_time_override_s is None:
            raise ValueError(
                "Motion-time components require the exact runtime-authored total."
            )
        travel_time = float(travel_time_override_s)
        if not np.isfinite(travel_time) or travel_time < 0.0:
            raise ValueError("travel_time_override_s must be finite and nonnegative.")
        component_values = tuple(float(value) for value in motion_components)
        if any(not np.isfinite(value) or value < 0.0 for value in component_values):
            raise ValueError(
                "Motion-time component overrides must be finite and nonnegative."
            )
        if sum(component_values) != travel_time:
            raise ValueError("Motion-time component overrides must sum to travel time.")
        motion_penalty = (
            float(config.lambda_horizontal_time) * component_values[0]
            + float(config.lambda_mast_vertical_time) * component_values[1]
            + float(config.lambda_settling_time) * component_values[2]
        )
    score = (
        float(node.static_score)
        - float(node.distance_weight) * float(path_length)
        - float(motion_penalty)
    )
    return float(score), float(path_length)


def _stable_descending_indices(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return deterministic descending indices with source order as tie-break."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(scores)):
        raise ValueError("Shortlist scores must be finite.")
    return np.lexsort(
        (
            np.arange(scores.size, dtype=np.int64),
            -scores,
        )
    ).astype(np.int64, copy=False)


def _exact_eig_shortlist(
    pending: Sequence[_PendingDSSPPNode],
    programs: Sequence[ShieldProgram],
    proxy_information_scores_pp: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> tuple[NDArray[np.int64], NDArray[np.float64], dict[str, int]]:
    """Shortlist poses, then retain every program at each selected pose."""
    pending_nodes = list(pending)
    if not pending_nodes:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            {
                "global": 0,
                "coverage": 0,
                "program_diversity": 0,
                "global_pose_count": 0,
                "coverage_pose_count": 0,
                "shortlisted_pose_count": 0,
            },
        )
    proxy = np.asarray(proxy_information_scores_pp, dtype=np.float64)
    if (
        proxy.ndim != 2
        or proxy.shape[1] != len(programs)
        or np.any(~np.isfinite(proxy))
        or np.any(proxy < 0.0)
    ):
        raise ValueError("Proxy information scores have an invalid shape.")
    program_index = {
        (
            str(program.name),
            tuple(int(value) for value in program.pair_ids),
            str(program.kind),
        ): index
        for index, program in enumerate(programs)
    }
    if len(program_index) != len(programs):
        raise ValueError("DSS shield programs must be unique.")
    ranking_scores = np.zeros(len(pending_nodes), dtype=np.float64)
    pose_indices = np.asarray(
        sorted({int(item.pose_index) for item in pending_nodes}),
        dtype=np.int64,
    )
    pose_row_by_index = {
        int(pose_index): row for row, pose_index in enumerate(pose_indices)
    }
    action_matrix = np.full(
        (pose_indices.size, len(programs)),
        -1,
        dtype=np.int64,
    )
    for index, item in enumerate(pending_nodes):
        key = (
            str(item.program.name),
            tuple(int(value) for value in item.program.pair_ids),
            str(item.program.kind),
        )
        resolved_program_index = program_index.get(key)
        if resolved_program_index is None:
            raise RuntimeError("Pending DSS node references an unknown program.")
        if int(item.pose_index) < 0 or int(item.pose_index) >= proxy.shape[0]:
            raise IndexError("Pending DSS node references an unknown pose.")
        pose_row = int(pose_row_by_index[int(item.pose_index)])
        if int(action_matrix[pose_row, int(resolved_program_index)]) >= 0:
            raise RuntimeError(
                "A DSS pose contains a duplicate pending shield program."
            )
        action_matrix[pose_row, int(resolved_program_index)] = int(index)
        ranking_scores[index] = float(item.static_score) + float(
            config.lambda_eig
        ) * float(proxy[int(item.pose_index), int(resolved_program_index)])
    if np.any(action_matrix < 0):
        raise RuntimeError(
            "Every DSS pose must expose the complete predeclared program library."
        )
    pose_ranking_scores = np.max(ranking_scores[action_matrix], axis=1)
    pose_limit = min(int(config.exact_eig_pose_max), int(pose_indices.size))
    required_action_count = int(pose_limit * len(programs))
    if pose_limit == int(pose_indices.size):
        return (
            np.arange(len(pending_nodes), dtype=np.int64),
            ranking_scores,
            {
                "global": int(len(pending_nodes)),
                "coverage": 0,
                "program_diversity": 0,
                "global_pose_count": int(pose_limit),
                "coverage_pose_count": 0,
                "shortlisted_pose_count": int(pose_limit),
            },
        )
    selected_pose_rows: set[int] = set()
    category_counts = {
        "global": 0,
        "coverage": 0,
        "program_diversity": 0,
        "global_pose_count": 0,
        "coverage_pose_count": 0,
        "shortlisted_pose_count": int(pose_limit),
    }

    coverage_pose_rows = sorted(
        range(int(pose_indices.size)),
        key=lambda pose_row: (
            -float(pending_nodes[int(action_matrix[pose_row, 0])].coverage_gain),
            -float(pose_ranking_scores[pose_row]),
            int(pose_indices[pose_row]),
        ),
    )
    coverage_limit = min(int(config.exact_eig_coverage_reserve), pose_limit)
    for pose_row in coverage_pose_rows[:coverage_limit]:
        selected_pose_rows.add(int(pose_row))
        category_counts["coverage_pose_count"] += 1

    for pose_row_raw in _stable_descending_indices(pose_ranking_scores):
        if len(selected_pose_rows) >= pose_limit:
            break
        pose_row = int(pose_row_raw)
        if pose_row not in selected_pose_rows:
            selected_pose_rows.add(pose_row)
            category_counts["global_pose_count"] += 1
    ordered_pose_rows = sorted(
        selected_pose_rows,
        key=lambda pose_row: (
            -float(pose_ranking_scores[pose_row]),
            int(pose_indices[pose_row]),
        ),
    )
    ordered_actions: list[int] = []
    for pose_row in ordered_pose_rows:
        pose_actions = action_matrix[int(pose_row)]
        local_order = _stable_descending_indices(ranking_scores[pose_actions])
        ordered_actions.extend(int(pose_actions[index]) for index in local_order)
    ordered = np.asarray(ordered_actions, dtype=np.int64)
    if ordered.size != required_action_count:
        raise RuntimeError("Exact-EIG pose shortlist lost a shield program.")
    category_counts["coverage"] = int(
        category_counts["coverage_pose_count"] * len(programs)
    )
    category_counts["global"] = int(
        category_counts["global_pose_count"] * len(programs)
    )
    return ordered, ranking_scores, category_counts


@dataclass(frozen=True, slots=True)
class _ConditionalSearchBatch:
    """Store one batched all-pair program search and its MC evidence."""

    program_pair_ids_al: NDArray[np.int64]
    information_gains_a: NDArray[np.float64]
    selection_sources_a: tuple[str, ...]
    selected_base_kl_samples_aq: NDArray[np.float64]
    selected_combined_kl_samples_a: tuple[NDArray[np.float64], ...]
    shadow_prefix_pair_ids_by_view_count: dict[int, NDArray[np.int64]]
    shadow_prefix_information_gains_by_view_count: dict[
        int,
        NDArray[np.float64],
    ]
    shadow_prefix_kl_samples_by_view_count: dict[
        int,
        NDArray[np.float64],
    ]
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ConditionalPoseEvaluation:
    """Store one or more MC-seed evaluations over aligned pose batches."""

    information_gains_ra: NDArray[np.float64]
    program_pair_ids_ral: NDArray[np.int64]
    selection_sources_r: tuple[tuple[str, ...], ...]
    selected_base_kl_samples_raq: NDArray[np.float64]
    selected_combined_kl_samples_r: tuple[
        tuple[NDArray[np.float64], ...],
        ...,
    ]
    shadow_prefix_pair_ids_by_view_count: dict[
        int,
        NDArray[np.int64],
    ]
    shadow_prefix_information_gains_by_view_count: dict[
        int,
        NDArray[np.float64],
    ]
    shadow_prefix_kl_samples_by_view_count: dict[
        int,
        NDArray[np.float64],
    ]
    diagnostics: dict[str, object]


def _conditional_minimum_response_scratch_budget(
    estimator: RotatingShieldPFEstimator,
    *,
    detector_aperture_samples: int,
    pair_count: int,
) -> int:
    """Return the runtime-owned minimum scratch budget for one source row."""
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=int(detector_aperture_samples),
    )
    estimator_method = getattr(
        kernel,
        "minimum_line_transport_working_memory_budget_bytes",
        None,
    )
    if not callable(estimator_method):
        raise RuntimeError(
            "Conditional DSS requires the runtime response-memory contract."
        )
    budgets = [
        int(
            estimator_method(
                isotope=str(isotope),
                orientation_pair_count=int(pair_count),
                dtype_bytes=np.dtype(np.float64).itemsize,
            )
        )
        for isotope in sorted(str(value) for value in estimator.isotopes)
    ]
    if not budgets or any(value <= 0 for value in budgets):
        raise RuntimeError("Runtime returned an invalid response scratch budget.")
    return int(max(budgets))


def _all_pair_component_programs(
    *,
    action_count: int,
    pair_count: int,
) -> list[ShieldProgram]:
    """Return one dense pair-ID-ordered response request per pose."""
    if action_count <= 0 or pair_count <= 0:
        raise ValueError("All-pair component dimensions must be positive.")
    dense_program = ShieldProgram(
        name="conditional_all_pair_response_cache",
        pair_ids=tuple(range(int(pair_count))),
        kind="internal_all_pair_response_cache",
    )
    return [dense_program] * int(action_count)


def _slice_joint_program_components(
    components: (
        _JointProgramSpectrumComponents | _DeviceJointProgramSpectrumComponents
    ),
    action_indices_a: NDArray[np.int64],
) -> _JointProgramSpectrumComponents | _DeviceJointProgramSpectrumComponents:
    """Select a batched pose subset without changing response semantics."""
    indices = np.asarray(action_indices_a, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.any(indices < 0):
        raise ValueError("Component slicing requires nonempty action indices.")
    if isinstance(components, _DeviceJointProgramSpectrumComponents):
        if indices.size == 1:
            action_selector: object = slice(
                int(indices[0]),
                int(indices[0]) + 1,
            )
        else:
            import torch

            action_selector = torch.as_tensor(
                indices,
                device=components.total_pnvsl.device,
                dtype=torch.long,
            )
        return _DeviceJointProgramSpectrumComponents(
            total_pnvsl=components.total_pnvsl[action_selector],
            uncollided_pnvsl=components.uncollided_pnvsl[action_selector],
            features_pnvslf=components.features_pnvslf[action_selector],
            live_times_v=components.live_times_v,
            contract_hash_sha256=components.contract_hash_sha256,
        )
    host_selector: object = (
        slice(int(indices[0]), int(indices[0]) + 1) if indices.size == 1 else indices
    )
    return _JointProgramSpectrumComponents(
        total_pnvsl=components.total_pnvsl[host_selector],
        uncollided_pnvsl=components.uncollided_pnvsl[host_selector],
        features_pnvslf=components.features_pnvslf[host_selector],
        live_times_v=np.ascontiguousarray(components.live_times_v),
        contract_hash_sha256=components.contract_hash_sha256,
    )


def _conditional_contender_subsets(
    result: ConditionalGreedyResult,
) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    """Return greedy and best one-swap program contenders."""
    contenders = [
        np.asarray(result.greedy_program_pair_ids_al, dtype=np.int64),
    ]
    names = ["greedy"]
    if int(result.one_swap_candidate_count_per_action) > 0:
        contenders.append(
            np.asarray(
                result.one_swap_best_program_pair_ids_al,
                dtype=np.int64,
            )
        )
        names.append("one_swap")
    subsets = np.stack(contenders, axis=1)
    return np.asarray(subsets, dtype=np.int64), tuple(names)


def _distinct_contender_mask(
    subsets_ack: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Mask the first occurrence of each pose-specific program contender."""
    subsets = np.asarray(subsets_ack, dtype=np.int64)
    if subsets.ndim != 3:
        raise ValueError("Program contenders must be shaped action/candidate/view.")
    canonical = np.sort(subsets, axis=2)
    equal = np.all(
        canonical[:, :, np.newaxis, :] == canonical[:, np.newaxis, :, :],
        axis=-1,
    )
    earlier = np.tril(
        np.ones((subsets.shape[1], subsets.shape[1]), dtype=bool),
        k=-1,
    )
    duplicated = np.any(equal & earlier[np.newaxis, :, :], axis=2)
    return np.asarray(~duplicated, dtype=np.bool_)


def _paired_gap_lower_confidence(
    left_aq: NDArray[np.float64],
    right_aq: NDArray[np.float64],
    *,
    confidence: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return paired mean, standard error, and one-sided lower bounds."""
    left = np.asarray(left_aq, dtype=np.float64)
    right = np.asarray(right_aq, dtype=np.float64)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or left.shape[1] < 2
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("Paired MC samples must be finite aligned matrices.")
    gap = left - right
    mean = np.mean(gap, axis=1, dtype=np.float64)
    standard_error = np.std(gap, axis=1, ddof=1) / np.sqrt(float(gap.shape[1]))
    critical = float(student_t.ppf(float(confidence), int(gap.shape[1] - 1)))
    lower = mean - critical * standard_error
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(standard_error, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
    )


def _select_contenders_from_kl_samples(
    subsets_ack: NDArray[np.int64],
    kl_samples_acq: NDArray[np.float64],
    *,
    confidence: float,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Select distinct contenders and identify statistically ambiguous poses."""
    subsets = np.asarray(subsets_ack, dtype=np.int64)
    samples = np.asarray(kl_samples_acq, dtype=np.float64)
    if (
        subsets.ndim != 3
        or samples.ndim != 3
        or subsets.shape[:2] != samples.shape[:2]
        or samples.shape[2] < 2
        or np.any(~np.isfinite(samples))
    ):
        raise ValueError("Contender subsets and KL samples are inconsistent.")
    means = np.mean(samples, axis=2, dtype=np.float64)
    distinct = _distinct_contender_mask(subsets)
    ranked_means = np.where(distinct, means, -np.inf)
    order = np.argsort(-ranked_means, axis=1, kind="stable")
    rows = np.arange(subsets.shape[0], dtype=np.int64)
    best = order[:, 0]
    distinct_count = np.sum(distinct, axis=1, dtype=np.int64)
    runner_position = np.where(distinct_count > 1, 1, 0).astype(
        np.int64,
        copy=False,
    )
    runner = order[rows, runner_position]
    best_samples = samples[rows, best]
    runner_samples = samples[rows, runner]
    _, _, lower = _paired_gap_lower_confidence(
        best_samples,
        runner_samples,
        confidence=float(confidence),
    )
    ambiguous = (distinct_count > 1) & (lower <= 0.0)
    selected_subsets = subsets[rows, best]
    selected_means = means[rows, best]
    return (
        np.asarray(selected_subsets, dtype=np.int64),
        np.asarray(selected_means, dtype=np.float64),
        np.asarray(best, dtype=np.int64),
        np.asarray(ambiguous, dtype=np.bool_),
        np.asarray(lower, dtype=np.float64),
    )


def _conditional_search_with_components(
    *,
    estimator: RotatingShieldPFEstimator,
    components: (
        _JointProgramSpectrumComponents | _DeviceJointProgramSpectrumComponents
    ),
    detector_positions_a3: NDArray[np.float64],
    particle_weights_n: NDArray[np.float64],
    program_length: int,
    sample_count: int,
    eig_call_seed: int,
    stream_name: str,
    enable_one_swap: bool,
    confirm_ambiguous: bool,
    confidence: float,
    shadow_prefix_view_counts: tuple[int, ...] = (),
    capture_shadow_kl_samples: bool = False,
) -> _ConditionalSearchBatch:
    """Search all pairs and hold out MC samples for shadow inference."""
    import torch

    prepared = prepare_conditional_observation_cache(
        estimator,
        components,
        particle_weights_n,
        detector_positions_a3,
        sample_count=int(sample_count),
        eig_call_seed=int(eig_call_seed),
        stream_name=str(stream_name),
    )
    result = select_conditional_greedy_programs(
        prepared.cache,
        particle_weights_n,
        num_orientations=int(estimator.num_orientations),
        program_length=int(program_length),
        enable_one_swap=bool(enable_one_swap),
    )
    contenders, contender_names = _conditional_contender_subsets(result)
    contender_tensor = torch.as_tensor(
        contenders,
        device=getattr(prepared.cache, "device"),
        dtype=torch.long,
    )
    _initial_eig, initial_kl_tensor = evaluate_subset_information_gain_torch(
        prepared.cache,
        contender_tensor,
        particle_weights_n,
    )
    initial_kl = np.asarray(
        initial_kl_tensor.detach().cpu().numpy(),
        dtype=np.float64,
    )
    shadow_programs: dict[int, NDArray[np.int64]] = {}
    shadow_gains: dict[int, NDArray[np.float64]] = {}
    shadow_kl_samples: dict[int, NDArray[np.float64]] = {}
    if bool(capture_shadow_kl_samples) and not shadow_prefix_view_counts:
        raise ValueError("Shadow KL capture requires declared prefix view counts.")
    if shadow_prefix_view_counts:
        resolved_shadow_counts = tuple(
            int(value) for value in shadow_prefix_view_counts
        )
        if (
            tuple(sorted(set(resolved_shadow_counts))) != resolved_shadow_counts
            or resolved_shadow_counts[-1] != int(program_length)
            or resolved_shadow_counts[0] < 1
        ):
            raise ValueError(
                "Shadow prefix view counts must be increasing and end at the "
                "executed program length."
            )
        greedy_programs = np.asarray(
            result.greedy_program_pair_ids_al,
            dtype=np.int64,
        )
        for view_count in resolved_shadow_counts:
            shadow_programs[view_count] = np.asarray(
                greedy_programs[:, :view_count],
                dtype=np.int64,
            ).copy()
            shadow_gains[view_count] = np.asarray(
                result.stages[view_count - 1].selected_information_gain_a,
                dtype=np.float64,
            ).copy()
    (
        selected_programs,
        selected_gains,
        selected_indices,
        ambiguous,
        initial_lower,
    ) = _select_contenders_from_kl_samples(
        contenders,
        initial_kl,
        confidence=float(confidence),
    )
    base_selected_kl = initial_kl[
        np.arange(initial_kl.shape[0], dtype=np.int64),
        selected_indices,
    ].copy()
    selected_combined_kl = [
        np.asarray(values, dtype=np.float64).copy() for values in base_selected_kl
    ]
    # All tensors needed from the first seed are now on the host. Releasing
    # its opaque cache before an independent confirmation prevents two full
    # action caches from overlapping in device memory.
    del _initial_eig, initial_kl_tensor, contender_tensor, prepared
    shadow_holdout_seed: int | None = None
    shadow_holdout_stream_name: str | None = None
    shadow_holdout_wall_s = 0.0
    if bool(capture_shadow_kl_samples):
        shadow_holdout_started = time.perf_counter()
        _release_dss_gpu_cache()
        shadow_holdout_seed = int(
            named_stream_seed(
                int(eig_call_seed),
                "dss_pp",
                "shield_view_count_shadow",
                str(stream_name),
                "independent_holdout",
            )
            & ((1 << 63) - 1)
        )
        shadow_holdout_stream_name = f"{stream_name}_shadow_independent_holdout"
        shadow_holdout = prepare_conditional_observation_cache(
            estimator,
            components,
            particle_weights_n,
            detector_positions_a3,
            sample_count=int(sample_count),
            eig_call_seed=int(shadow_holdout_seed),
            stream_name=shadow_holdout_stream_name,
        )
        # These three fixed horizons have different last dimensions. Each
        # call remains batched over every pose and reuses the same holdout
        # all-64-view cache, latent states, nuisance draws, and MC sample IDs.
        for view_count in resolved_shadow_counts:
            prefix_tensor = torch.as_tensor(
                shadow_programs[view_count][:, np.newaxis, :],
                device=getattr(shadow_holdout.cache, "device"),
                dtype=torch.long,
            )
            _prefix_eig, prefix_kl_tensor = evaluate_subset_information_gain_torch(
                shadow_holdout.cache,
                prefix_tensor,
                particle_weights_n,
            )
            prefix_kl = np.asarray(
                prefix_kl_tensor[:, 0].detach().cpu().numpy(),
                dtype=np.float64,
            )
            shadow_kl_samples[view_count] = prefix_kl
            shadow_gains[view_count] = np.mean(
                prefix_kl,
                axis=1,
                dtype=np.float64,
            )
            del _prefix_eig, prefix_kl_tensor, prefix_tensor
        del shadow_holdout
        _release_dss_gpu_cache()
        shadow_holdout_wall_s = float(time.perf_counter() - shadow_holdout_started)
    confirmation_count = 0
    confirmation_batch_sizes: list[int] = []
    confirmation_component_strategy = "not_requested"
    combined_lower = initial_lower.copy()
    if bool(confirm_ambiguous) and np.any(ambiguous):
        _release_dss_gpu_cache()
        ambiguous_indices = np.flatnonzero(ambiguous).astype(np.int64, copy=False)
        confirmation_count = int(ambiguous_indices.size)
        confirmation_seed = int(
            named_stream_seed(
                int(eig_call_seed),
                "dss_pp",
                "conditional_all_pairs",
                str(stream_name),
                "ambiguous_program_confirmation",
            )
            & ((1 << 63) - 1)
        )
        confirmation_kl = np.empty_like(initial_kl[ambiguous_indices])
        all_actions_ambiguous = bool(
            ambiguous_indices.size == initial_kl.shape[0]
            and np.array_equal(
                ambiguous_indices,
                np.arange(initial_kl.shape[0], dtype=np.int64),
            )
        )
        confirmation_batches = (
            (ambiguous_indices,)
            if all_actions_ambiguous
            else tuple(
                np.asarray([value], dtype=np.int64) for value in ambiguous_indices
            )
        )
        confirmation_component_strategy = (
            "reuse_original_all_actions"
            if all_actions_ambiguous
            else "single_pose_views_without_full_ambiguous_copy"
        )
        confirmation_offset = 0
        for confirmation_indices in confirmation_batches:
            confirmation_batch_sizes.append(int(confirmation_indices.size))
            confirmation_components = (
                components
                if all_actions_ambiguous
                else _slice_joint_program_components(
                    components,
                    confirmation_indices,
                )
            )
            confirmation = prepare_conditional_observation_cache(
                estimator,
                confirmation_components,
                particle_weights_n,
                np.asarray(detector_positions_a3, dtype=np.float64)[
                    confirmation_indices
                ],
                sample_count=int(sample_count),
                eig_call_seed=confirmation_seed,
                stream_name=(f"{stream_name}_ambiguous_program_confirmation"),
            )
            confirmation_contenders = torch.as_tensor(
                contenders[confirmation_indices],
                device=getattr(confirmation.cache, "device"),
                dtype=torch.long,
            )
            _confirmation_eig, confirmation_kl_tensor = (
                evaluate_subset_information_gain_torch(
                    confirmation.cache,
                    confirmation_contenders,
                    particle_weights_n,
                )
            )
            confirmation_stop = confirmation_offset + int(confirmation_indices.size)
            confirmation_kl[confirmation_offset:confirmation_stop] = np.asarray(
                confirmation_kl_tensor.detach().cpu().numpy(),
                dtype=np.float64,
            )
            confirmation_offset = confirmation_stop
            del (
                _confirmation_eig,
                confirmation_kl_tensor,
                confirmation_contenders,
                confirmation,
                confirmation_components,
            )
            _release_dss_gpu_cache()
        combined_kl = np.concatenate(
            (initial_kl[ambiguous_indices], confirmation_kl),
            axis=2,
        )
        (
            confirmed_programs,
            confirmed_gains,
            confirmed_indices,
            _still_ambiguous,
            confirmed_lower,
        ) = _select_contenders_from_kl_samples(
            contenders[ambiguous_indices],
            combined_kl,
            confidence=float(confidence),
        )
        selected_programs[ambiguous_indices] = confirmed_programs
        selected_gains[ambiguous_indices] = confirmed_gains
        selected_indices[ambiguous_indices] = confirmed_indices
        combined_lower[ambiguous_indices] = confirmed_lower
        base_selected_kl[ambiguous_indices] = initial_kl[
            ambiguous_indices,
            confirmed_indices,
        ]
        for local_index, action_index in enumerate(ambiguous_indices):
            selected_combined_kl[int(action_index)] = np.asarray(
                combined_kl[local_index, confirmed_indices[local_index]],
                dtype=np.float64,
            ).copy()
    selected_sources = tuple(contender_names[int(index)] for index in selected_indices)
    diagnostics: dict[str, object] = {
        "greedy_candidate_count_per_pose": int(
            result.greedy_candidate_count_per_action
        ),
        "one_swap_candidate_count_per_pose": int(
            result.one_swap_candidate_count_per_action
        ),
        "initial_selection_sources": list(result.selection_source_a),
        "selected_sources": list(selected_sources),
        "ambiguous_program_pose_count": int(np.count_nonzero(ambiguous)),
        "independently_confirmed_program_pose_count": int(confirmation_count),
        "initial_cache_released_before_confirmation": True,
        "confirmation_component_strategy": str(confirmation_component_strategy),
        "confirmation_pose_batch_sizes": confirmation_batch_sizes,
        "confirmation_additional_component_pose_limit": int(
            0
            if confirmation_component_strategy == "reuse_original_all_actions"
            else 1
            if confirmation_count > 0
            else 0
        ),
        "program_gap_lower_confidence_initial": [
            float(value) for value in initial_lower
        ],
        "program_gap_lower_confidence_combined": [
            float(value) for value in combined_lower
        ],
        "one_swap_applied_count": int(np.count_nonzero(result.one_swap_applied_a)),
        "shadow_prefix_view_counts": [
            int(value) for value in shadow_prefix_view_counts
        ],
        "shadow_prefix_kl_samples_captured": bool(
            capture_shadow_kl_samples and shadow_prefix_view_counts
        ),
        "shadow_prefix_selection_and_evaluation_independent": bool(
            capture_shadow_kl_samples and shadow_prefix_view_counts
        ),
        "shadow_holdout_seed": shadow_holdout_seed,
        "shadow_holdout_stream_name": shadow_holdout_stream_name,
        "shadow_holdout_wall_s": float(shadow_holdout_wall_s),
    }
    return _ConditionalSearchBatch(
        program_pair_ids_al=np.asarray(selected_programs, dtype=np.int64),
        information_gains_a=np.asarray(selected_gains, dtype=np.float64),
        selection_sources_a=selected_sources,
        selected_base_kl_samples_aq=np.asarray(base_selected_kl, dtype=np.float64),
        selected_combined_kl_samples_a=tuple(selected_combined_kl),
        shadow_prefix_pair_ids_by_view_count=shadow_programs,
        shadow_prefix_information_gains_by_view_count=shadow_gains,
        shadow_prefix_kl_samples_by_view_count=shadow_kl_samples,
        diagnostics=diagnostics,
    )


def _evaluate_conditional_pose_batches(
    *,
    estimator: RotatingShieldPFEstimator,
    detector_positions_a3: NDArray[np.float64],
    joint_particles: JointPlanningParticles,
    config: DSSPPConfig,
    sample_count: int,
    eig_call_seeds_r: NDArray[np.int64],
    stream_name: str,
    workload: str,
    working_memory_budget_bytes: int,
    maximum_subset_candidate_count: int,
    enable_one_swap: bool,
    confirm_ambiguous: bool,
) -> _ConditionalPoseEvaluation:
    """Evaluate all-pair searches in response-preserving batched pose chunks."""
    detectors = np.asarray(detector_positions_a3, dtype=np.float64)
    seeds = np.asarray(eig_call_seeds_r)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or detectors.shape[0] <= 0
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("Conditional detector poses must be finite and nonempty.")
    if (
        seeds.ndim != 1
        or seeds.size <= 0
        or not np.issubdtype(seeds.dtype, np.integer)
        or np.any(seeds < 0)
    ):
        raise ValueError("Conditional EIG seeds must be nonnegative integers.")
    pair_count = int(estimator.num_orientations) ** 2
    program_length = int(config.program_length)
    action_count = int(detectors.shape[0])
    replica_count = int(seeds.size)
    source_slot_count = int(
        sum(
            np.asarray(values).shape[1]
            for values in joint_particles.strengths_nk_by_isotope.values()
        )
    )
    model = estimator.authenticated_full_spectrum_model()
    minimum_response_scratch = _conditional_minimum_response_scratch_budget(
        estimator,
        detector_aperture_samples=int(config.detector_aperture_samples),
        pair_count=int(pair_count),
    )
    chunk_plan = plan_conditional_pose_chunk(
        model,
        workload=str(workload),
        requested_pose_count=int(action_count),
        particle_count=int(np.asarray(joint_particles.weights_n).size),
        sample_count=int(sample_count),
        source_slot_count=max(1, source_slot_count),
        pair_count=int(pair_count),
        program_length=int(program_length),
        line_count=len(tuple(model.line_identity)),
        feature_count=len(tuple(model.transport_feature_order)),
        maximum_subset_candidate_count=int(maximum_subset_candidate_count),
        configured_total_budget_bytes=int(working_memory_budget_bytes),
        minimum_response_scratch_budget_bytes=int(minimum_response_scratch),
        use_gpu=bool(estimator.pf_config.use_gpu),
        gpu_device=str(estimator.pf_config.gpu_device),
    )
    pose_chunk_size = int(chunk_plan.pose_chunk_size)
    gains = np.empty((replica_count, action_count), dtype=np.float64)
    programs = np.empty(
        (replica_count, action_count, program_length),
        dtype=np.int64,
    )
    kl_samples = np.empty(
        (replica_count, action_count, int(sample_count)),
        dtype=np.float64,
    )
    shadow_view_counts = (
        tuple(int(value) for value in config.shield_view_count_shadow_candidate_counts)
        if bool(config.shield_view_count_shadow_enabled)
        else ()
    )
    shadow_programs = {
        view_count: np.empty(
            (replica_count, action_count, view_count),
            dtype=np.int64,
        )
        for view_count in shadow_view_counts
    }
    shadow_gains = {
        view_count: np.empty(
            (replica_count, action_count),
            dtype=np.float64,
        )
        for view_count in shadow_view_counts
    }
    capture_shadow_samples = bool(
        config.shield_view_count_shadow_enabled and str(workload) == "exact"
    )
    shadow_samples = {
        view_count: np.empty(
            (replica_count, action_count, int(sample_count)),
            dtype=np.float64,
        )
        for view_count in shadow_view_counts
        if capture_shadow_samples
    }
    source_rows: list[list[str]] = [list() for _ in range(replica_count)]
    combined_kl_rows: list[list[NDArray[np.float64]]] = [
        [] for _ in range(replica_count)
    ]
    chunk_diagnostics: list[dict[str, object]] = []
    response_wall_s = 0.0
    search_wall_s = 0.0
    weights = np.asarray(joint_particles.weights_n, dtype=np.float64)
    oom_retry_events: list[dict[str, object]] = []
    active_pose_chunk_size = int(pose_chunk_size)
    action_start = 0
    while action_start < action_count:
        action_stop = min(action_start + active_pose_chunk_size, action_count)
        chunk_detectors = detectors[action_start:action_stop]
        response_scratch_budget = chunk_plan.response_scratch_budget_for_pose_count(
            int(chunk_detectors.shape[0])
        )
        attempt_started = time.perf_counter()
        components = None
        local_batches: list[_ConditionalSearchBatch] = []
        replica_diagnostics: list[dict[str, object]] = []
        try:
            response_started = time.perf_counter()
            components = _full_spectrum_joint_program_components(
                estimator=estimator,
                detector_positions=chunk_detectors,
                programs=_all_pair_component_programs(
                    action_count=int(chunk_detectors.shape[0]),
                    pair_count=pair_count,
                ),
                joint_particles=joint_particles,
                live_time_s=float(config.live_time_s),
                detector_aperture_samples=int(config.detector_aperture_samples),
                device_resident=bool(estimator.pf_config.use_gpu),
                working_memory_budget_bytes=int(response_scratch_budget),
            )
            response_wall_s += float(time.perf_counter() - response_started)
            for replica_index, seed in enumerate(seeds):
                search_started = time.perf_counter()
                batch = _conditional_search_with_components(
                    estimator=estimator,
                    components=components,
                    detector_positions_a3=chunk_detectors,
                    particle_weights_n=weights,
                    program_length=program_length,
                    sample_count=int(sample_count),
                    eig_call_seed=int(seed),
                    stream_name=f"{stream_name}_replica_{replica_index}",
                    enable_one_swap=bool(enable_one_swap),
                    confirm_ambiguous=bool(confirm_ambiguous),
                    confidence=float(config.proxy_boundary_confidence),
                    shadow_prefix_view_counts=(
                        tuple(config.shield_view_count_shadow_candidate_counts)
                        if bool(config.shield_view_count_shadow_enabled)
                        else ()
                    ),
                    capture_shadow_kl_samples=bool(
                        config.shield_view_count_shadow_enabled
                        and str(workload) == "exact"
                    ),
                )
                search_wall_s += float(time.perf_counter() - search_started)
                local_batches.append(batch)
                replica_diagnostics.append(dict(batch.diagnostics))
        except Exception as error:
            if not _is_dss_eig_memory_error(error):
                raise
            error.__traceback__ = None
            components = None
            local_batches.clear()
            _release_dss_gpu_cache()
            failed_chunk_size = int(chunk_detectors.shape[0])
            if failed_chunk_size <= 1:
                raise RuntimeError(
                    "Conditional DSS exhausted memory for one full-fidelity "
                    "pose after response/cache retry reduction."
                ) from error
            reduced_chunk_size = 2 if failed_chunk_size > 2 else 1
            oom_retry_events.append(
                {
                    "action_start": int(action_start),
                    "failed_pose_chunk_size": int(failed_chunk_size),
                    "retry_pose_chunk_size": int(reduced_chunk_size),
                    "response_scratch_budget_bytes": int(response_scratch_budget),
                    "failed_attempt_wall_s": float(
                        time.perf_counter() - attempt_started
                    ),
                }
            )
            active_pose_chunk_size = int(reduced_chunk_size)
            continue
        for replica_index, batch in enumerate(local_batches):
            gains[replica_index, action_start:action_stop] = batch.information_gains_a
            programs[replica_index, action_start:action_stop] = (
                batch.program_pair_ids_al
            )
            kl_samples[replica_index, action_start:action_stop] = (
                batch.selected_base_kl_samples_aq
            )
            source_rows[replica_index].extend(batch.selection_sources_a)
            combined_kl_rows[replica_index].extend(batch.selected_combined_kl_samples_a)
            for view_count in shadow_view_counts:
                shadow_programs[view_count][
                    replica_index,
                    action_start:action_stop,
                ] = batch.shadow_prefix_pair_ids_by_view_count[view_count]
                shadow_gains[view_count][
                    replica_index,
                    action_start:action_stop,
                ] = batch.shadow_prefix_information_gains_by_view_count[view_count]
                if capture_shadow_samples:
                    shadow_samples[view_count][
                        replica_index,
                        action_start:action_stop,
                    ] = batch.shadow_prefix_kl_samples_by_view_count[view_count]
        components = None
        chunk_diagnostics.append(
            {
                "action_start": int(action_start),
                "action_stop": int(action_stop),
                "pose_chunk_size": int(chunk_detectors.shape[0]),
                "response_scratch_budget_bytes": int(response_scratch_budget),
                "replicas": replica_diagnostics,
            }
        )
        action_start = action_stop
    if (
        np.any(~np.isfinite(gains))
        or np.any(gains < 0.0)
        or np.any(programs < 0)
        or np.any(programs >= pair_count)
        or np.any(~np.isfinite(kl_samples))
    ):
        raise RuntimeError("Conditional pose batching produced invalid results.")
    if any(len(values) != action_count for values in source_rows) or any(
        len(values) != action_count for values in combined_kl_rows
    ):
        raise RuntimeError("Conditional pose batching lost action diagnostics.")
    if shadow_view_counts:
        invalid_shadow = any(
            np.any(programs_by_count < 0)
            or np.any(programs_by_count >= pair_count)
            or np.any(~np.isfinite(shadow_gains[view_count]))
            or np.any(shadow_gains[view_count] < 0.0)
            for view_count, programs_by_count in shadow_programs.items()
        )
        if invalid_shadow:
            raise RuntimeError("Shadow prefix pose batching produced invalid data.")
        if capture_shadow_samples:
            invalid_samples = any(
                np.any(~np.isfinite(values)) or np.any(values < 0.0)
                for values in shadow_samples.values()
            )
            if invalid_samples:
                raise RuntimeError("Shadow prefix KL samples are invalid.")
            for view_count in shadow_view_counts:
                if not np.allclose(
                    np.mean(
                        shadow_samples[view_count],
                        axis=2,
                        dtype=np.float64,
                    ),
                    shadow_gains[view_count],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        "Shadow prefix batch means lost their paired samples."
                    )
    combined_means = np.asarray(
        [
            [float(np.mean(values, dtype=np.float64)) for values in replica]
            for replica in combined_kl_rows
        ],
        dtype=np.float64,
    )
    if not np.allclose(
        combined_means,
        gains,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Conditional EIG means lost their paired MC samples.")
    diagnostics: dict[str, object] = {
        "pose_count": int(action_count),
        "pose_chunk_size": int(pose_chunk_size),
        "pose_chunk_count": int(len(chunk_diagnostics)),
        "replica_count": int(replica_count),
        "sample_count": int(sample_count),
        "particle_count": int(weights.size),
        "pair_count": int(pair_count),
        "program_length": int(program_length),
        "shadow_prefix_view_counts": [int(value) for value in shadow_view_counts],
        "shadow_prefix_kl_samples_captured": bool(capture_shadow_samples),
        "response_wall_s": float(response_wall_s),
        "search_wall_s": float(search_wall_s),
        "wall_s": float(response_wall_s + search_wall_s),
        "memory_chunk_plan": chunk_plan.diagnostics(),
        "oom_retry_count": int(len(oom_retry_events)),
        "oom_retry_events": oom_retry_events,
        "successful_pose_chunk_sizes": [
            int(item["pose_chunk_size"]) for item in chunk_diagnostics
        ],
        "chunks": chunk_diagnostics,
    }
    return _ConditionalPoseEvaluation(
        information_gains_ra=np.asarray(gains, dtype=np.float64),
        program_pair_ids_ral=np.asarray(programs, dtype=np.int64),
        selection_sources_r=tuple(tuple(values) for values in source_rows),
        selected_base_kl_samples_raq=np.asarray(kl_samples, dtype=np.float64),
        selected_combined_kl_samples_r=tuple(
            tuple(np.asarray(values, dtype=np.float64) for values in replica)
            for replica in combined_kl_rows
        ),
        shadow_prefix_pair_ids_by_view_count={
            int(view_count): np.asarray(values, dtype=np.int64)
            for view_count, values in shadow_programs.items()
        },
        shadow_prefix_information_gains_by_view_count={
            int(view_count): np.asarray(values, dtype=np.float64)
            for view_count, values in shadow_gains.items()
        },
        shadow_prefix_kl_samples_by_view_count={
            int(view_count): np.asarray(values, dtype=np.float64)
            for view_count, values in shadow_samples.items()
        },
        diagnostics=diagnostics,
    )


def _evaluate_fixed_programs_in_pose_batches(
    *,
    estimator: RotatingShieldPFEstimator,
    detector_positions_a3: NDArray[np.float64],
    program_pair_ids_al: NDArray[np.int64],
    joint_particles: JointPlanningParticles,
    config: DSSPPConfig,
    sample_count: int,
    eig_call_seed: int,
    stream_name: str,
    working_memory_budget_bytes: int,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Return fixed-program KL samples and memory scheduling diagnostics."""
    import torch

    detectors = np.asarray(detector_positions_a3, dtype=np.float64)
    programs = np.asarray(program_pair_ids_al, dtype=np.int64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or programs.shape != (detectors.shape[0], int(config.program_length))
        or detectors.shape[0] <= 0
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("Fixed-program confirmation inputs are inconsistent.")
    output = np.empty(
        (detectors.shape[0], int(sample_count)),
        dtype=np.float64,
    )
    pair_count = int(estimator.num_orientations) ** 2
    particle_weights = np.asarray(joint_particles.weights_n, dtype=np.float64)
    source_slot_count = int(
        sum(
            np.asarray(values).shape[1]
            for values in joint_particles.strengths_nk_by_isotope.values()
        )
    )
    model = estimator.authenticated_full_spectrum_model()
    minimum_response_scratch = _conditional_minimum_response_scratch_budget(
        estimator,
        detector_aperture_samples=int(config.detector_aperture_samples),
        pair_count=int(pair_count),
    )
    chunk_plan = plan_conditional_pose_chunk(
        model,
        workload="exact",
        requested_pose_count=int(detectors.shape[0]),
        particle_count=int(particle_weights.size),
        sample_count=int(sample_count),
        source_slot_count=max(1, source_slot_count),
        pair_count=int(pair_count),
        program_length=int(config.program_length),
        line_count=len(tuple(model.line_identity)),
        feature_count=len(tuple(model.transport_feature_order)),
        maximum_subset_candidate_count=1,
        configured_total_budget_bytes=int(working_memory_budget_bytes),
        minimum_response_scratch_budget_bytes=int(minimum_response_scratch),
        use_gpu=bool(estimator.pf_config.use_gpu),
        gpu_device=str(estimator.pf_config.gpu_device),
    )
    pose_chunk_size = int(chunk_plan.pose_chunk_size)
    active_pose_chunk_size = int(pose_chunk_size)
    action_start = 0
    oom_retry_events: list[dict[str, int]] = []
    successful_pose_chunk_sizes: list[int] = []
    while action_start < int(detectors.shape[0]):
        action_stop = min(
            action_start + int(active_pose_chunk_size),
            int(detectors.shape[0]),
        )
        chunk_detectors = detectors[action_start:action_stop]
        components = None
        prepared = None
        subset_tensor = None
        information_gain_tensor = None
        kl_tensor = None
        try:
            components = _full_spectrum_joint_program_components(
                estimator,
                chunk_detectors,
                _all_pair_component_programs(
                    action_count=int(chunk_detectors.shape[0]),
                    pair_count=pair_count,
                ),
                joint_particles,
                live_time_s=float(config.live_time_s),
                detector_aperture_samples=int(config.detector_aperture_samples),
                device_resident=bool(estimator.pf_config.use_gpu),
                working_memory_budget_bytes=(
                    chunk_plan.response_scratch_budget_for_pose_count(
                        int(chunk_detectors.shape[0])
                    )
                ),
            )
            prepared = prepare_conditional_observation_cache(
                estimator,
                components,
                particle_weights,
                chunk_detectors,
                sample_count=int(sample_count),
                eig_call_seed=int(eig_call_seed),
                stream_name=str(stream_name),
            )
            subset_tensor = torch.as_tensor(
                programs[action_start:action_stop, np.newaxis, :],
                device=getattr(prepared.cache, "device"),
                dtype=torch.long,
            )
            information_gain_tensor, kl_tensor = evaluate_subset_information_gain_torch(
                prepared.cache,
                subset_tensor,
                particle_weights,
            )
            output[action_start:action_stop] = np.asarray(
                kl_tensor[:, 0].detach().cpu().numpy(),
                dtype=np.float64,
            )
            del (
                information_gain_tensor,
                kl_tensor,
                subset_tensor,
                prepared,
            )
        except Exception as error:
            if not _is_dss_eig_memory_error(error):
                raise
            error.__traceback__ = None
            components = None
            prepared = None
            subset_tensor = None
            information_gain_tensor = None
            kl_tensor = None
            _release_dss_gpu_cache()
            failed_chunk_size = int(chunk_detectors.shape[0])
            if failed_chunk_size <= 1:
                raise RuntimeError(
                    "Fixed-program confirmation exhausted memory for one "
                    "full-fidelity pose."
                ) from error
            reduced_chunk_size = 2 if failed_chunk_size > 2 else 1
            oom_retry_events.append(
                {
                    "action_start": int(action_start),
                    "failed_pose_chunk_size": int(failed_chunk_size),
                    "retry_pose_chunk_size": int(reduced_chunk_size),
                }
            )
            active_pose_chunk_size = int(reduced_chunk_size)
            continue
        components = None
        successful_pose_chunk_sizes.append(int(chunk_detectors.shape[0]))
        action_start = action_stop
    if np.any(~np.isfinite(output)) or np.any(output < 0.0):
        raise RuntimeError("Fixed-program confirmation produced invalid KL samples.")
    diagnostics = chunk_plan.diagnostics()
    diagnostics.update(
        {
            "oom_retry_count": int(len(oom_retry_events)),
            "oom_retry_events": oom_retry_events,
            "successful_pose_chunk_sizes": successful_pose_chunk_sizes,
        }
    )
    return output, diagnostics


def _static_station_scores_batch(
    *,
    coverage_norm_p: NDArray[np.float64],
    revisit_penalties_p: NDArray[np.float64],
    bearing_gains_p: NDArray[np.float64],
    frontier_gains_p: NDArray[np.float64],
    turn_penalties_p: NDArray[np.float64],
    local_orbit_gains_p: NDArray[np.float64],
    elevation_condition_gains_p: NDArray[np.float64],
    coverage_floor: float,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Return vectorized spatial utility without EIG or physical motion."""
    coverage = np.asarray(coverage_norm_p, dtype=np.float64).reshape(-1)
    arrays = tuple(
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (
            revisit_penalties_p,
            bearing_gains_p,
            frontier_gains_p,
            turn_penalties_p,
            local_orbit_gains_p,
            elevation_condition_gains_p,
        )
    )
    if any(values.shape != coverage.shape for values in arrays) or any(
        np.any(~np.isfinite(values)) for values in (coverage,) + arrays
    ):
        raise ValueError("Batched static pose-score components must align.")
    revisit, bearing, frontier, turn, local_orbit, elevation = arrays
    scores = (
        float(config.lambda_coverage) * coverage
        + float(config.lambda_bearing_diversity) * bearing
        + float(config.lambda_frontier) * frontier
        + float(config.lambda_local_orbit) * local_orbit
        + float(config.lambda_elevation_condition)
        * np.log1p(np.maximum(elevation, 0.0))
        - float(config.eta_revisit) * revisit
        - float(config.lambda_turn_smoothness) * turn
        - float(config.coverage_floor_weight)
        * np.square(np.maximum(0.0, float(coverage_floor) - coverage))
    )
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("Batched static pose scoring produced invalid values.")
    return np.asarray(scores, dtype=np.float64)


def _proxy_replica_scores_payload(
    scores_rp: NDArray[np.float64],
    *,
    evaluated: bool,
) -> list[list[float | None]]:
    """Return JSON-safe proxy replicas with missing refinements as null."""
    if not bool(evaluated):
        return []
    scores = np.asarray(scores_rp, dtype=np.float64)
    if scores.ndim != 2 or np.any(np.isinf(scores)):
        raise ValueError("Proxy replica diagnostics may contain finite values or NaN.")
    return [
        [None if not np.isfinite(value) else float(value) for value in replica]
        for replica in scores
    ]


def _shadow_scores_for_view_counts(
    reference_scores_a: NDArray[np.float64],
    reference_information_gains_a: NDArray[np.float64],
    information_gains_la: NDArray[np.float64],
    *,
    config: DSSPPConfig,
) -> NDArray[np.float64]:
    """Compose view-count scores from common deterministic pose utility."""
    scores = np.asarray(reference_scores_a, dtype=np.float64).reshape(-1)
    reference_gains = np.asarray(
        reference_information_gains_a,
        dtype=np.float64,
    ).reshape(-1)
    gains = np.asarray(information_gains_la, dtype=np.float64)
    if (
        gains.ndim != 2
        or gains.shape[1] != scores.size
        or reference_gains.shape != scores.shape
        or np.any(~np.isfinite(gains))
        or np.any(~np.isfinite(scores))
        or np.any(~np.isfinite(reference_gains))
    ):
        raise ValueError("Shadow view-count scores require aligned finite inputs.")
    deterministic = scores - float(config.lambda_eig) * reference_gains
    output = deterministic[np.newaxis, :] + float(config.lambda_eig) * gains
    if np.any(~np.isfinite(output)):
        raise RuntimeError("Shadow view-count scores became nonfinite.")
    return np.asarray(output, dtype=np.float64)


def _shadow_union_pose_indices(
    executed_shortlist_local_indices: NDArray[np.int64],
    proxy_scores_la: NDArray[np.float64],
    *,
    maximum_pose_count: int,
) -> NDArray[np.int64]:
    """Add K-specific proxy leaders without changing the executed shortlist."""
    executed = np.asarray(
        executed_shortlist_local_indices,
        dtype=np.int64,
    ).reshape(-1)
    scores = np.asarray(proxy_scores_la, dtype=np.float64)
    if (
        scores.ndim != 2
        or scores.shape[1] < 1
        or np.any(~np.isfinite(scores))
        or np.any(executed < 0)
        or np.any(executed >= scores.shape[1])
    ):
        raise ValueError("Shadow union shortlist inputs are invalid.")
    limit = min(int(maximum_pose_count), int(scores.shape[1]))
    if limit < executed.size:
        raise ValueError("Shadow union limit cannot truncate the executed shortlist.")
    ranked = np.stack(
        [_stable_descending_indices(row) for row in scores],
        axis=0,
    )
    # Rank-major flattening reserves leaders from every view count before
    # taking deeper candidates. NumPy uniqueness retains the first occurrence.
    interleaved = ranked.T.reshape(-1)
    candidates = np.concatenate((executed, interleaved)).astype(
        np.int64,
        copy=False,
    )
    _, first_positions = np.unique(candidates, return_index=True)
    ordered = candidates[np.sort(first_positions)]
    return np.asarray(ordered[:limit], dtype=np.int64)


def _shadow_holdout_seed_blocks(
    evaluation: _ConditionalPoseEvaluation,
    pose_indices_a: NDArray[np.int64],
) -> list[dict[str, object]]:
    """Return chunk-resolved independent holdout seed provenance."""
    pose_indices = np.asarray(pose_indices_a, dtype=np.int64).reshape(-1)
    chunks = evaluation.diagnostics.get("chunks")
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("Conditional evaluation chunks must be a sequence.")
    blocks: list[dict[str, object]] = []
    for raw_chunk in chunks:
        if not isinstance(raw_chunk, Mapping):
            raise TypeError("Conditional evaluation chunk must be a mapping.")
        start = int(raw_chunk["action_start"])
        stop = int(raw_chunk["action_stop"])
        replicas = raw_chunk.get("replicas")
        if not isinstance(replicas, Sequence) or len(replicas) != 1:
            raise ValueError("Exact shadow chunks require one MC replica.")
        replica = replicas[0]
        if not isinstance(replica, Mapping):
            raise TypeError("Exact shadow replica must be a mapping.")
        holdout_seed = replica.get("shadow_holdout_seed")
        holdout_stream = replica.get("shadow_holdout_stream_name")
        if holdout_seed is None or holdout_stream is None:
            raise RuntimeError("Exact shadow chunk lost its holdout provenance.")
        blocks.append(
            {
                "seed": int(holdout_seed),
                "stream_name": str(holdout_stream),
                "pose_indices": [int(value) for value in pose_indices[start:stop]],
                "samples_per_pose": int(evaluation.diagnostics["sample_count"]),
            }
        )
    return blocks


def _shadow_selected_action_payload(
    *,
    selected_view_counts_a: NDArray[np.int64],
    scores_la: NDArray[np.float64],
    information_gains_la: NDArray[np.float64],
    programs_by_view_count: Mapping[int, NDArray[np.int64]],
    candidate_view_counts: tuple[int, ...],
    pose_indices_a: NDArray[np.int64],
    poses_a3: NDArray[np.float64],
) -> dict[str, object]:
    """Return the highest-scoring exact shadow action for one selection rule."""
    selected = np.asarray(selected_view_counts_a, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores_la, dtype=np.float64)
    gains = np.asarray(information_gains_la, dtype=np.float64)
    pose_indices = np.asarray(pose_indices_a, dtype=np.int64).reshape(-1)
    poses = np.asarray(poses_a3, dtype=np.float64)
    lengths = np.asarray(candidate_view_counts, dtype=np.int64)
    if (
        scores.shape != gains.shape
        or scores.shape != (lengths.size, selected.size)
        or pose_indices.shape != selected.shape
        or poses.shape != (selected.size, 3)
    ):
        raise ValueError("Shadow action payload inputs must align by pose.")
    length_matches = selected[np.newaxis, :] == lengths[:, np.newaxis]
    if not np.all(np.sum(length_matches, axis=0) == 1):
        raise ValueError("Selected shadow view counts lie outside the policy.")
    selected_length_indices = np.argmax(length_matches, axis=0)
    pose_offsets = np.arange(selected.size, dtype=np.int64)
    selected_scores = scores[selected_length_indices, pose_offsets]
    leader = int(_stable_descending_indices(selected_scores)[0])
    view_count = int(selected[leader])
    return {
        "pose_index": int(pose_indices[leader]),
        "pose_xyz": [float(value) for value in poses[leader]],
        "selected_view_count": view_count,
        "pair_ids": [
            int(value) for value in programs_by_view_count[view_count][leader]
        ],
        "information_gain_mean_nat": float(
            gains[selected_length_indices[leader], leader]
        ),
        "pose_score": float(selected_scores[leader]),
    }


def _shadow_exact_diagnostics(
    *,
    pose_indices_a: NDArray[np.int64],
    poses_a3: NDArray[np.float64],
    programs_by_view_count: Mapping[int, NDArray[np.int64]],
    kl_samples_by_view_count: Mapping[int, NDArray[np.float64]],
    reference_scores_a: NDArray[np.float64],
    reference_information_gains_a: NDArray[np.float64],
    config: DSSPPConfig,
) -> dict[str, object]:
    """Build JSON-safe exact paired evidence and hypothetical actions."""
    candidate_counts = tuple(
        int(value) for value in config.shield_view_count_shadow_candidate_counts
    )
    samples = np.stack(
        [
            np.asarray(kl_samples_by_view_count[view_count], dtype=np.float64)
            for view_count in candidate_counts
        ],
        axis=0,
    )
    decision = select_shield_view_count_shadow(
        samples,
        candidate_view_counts=candidate_counts,
        retention_fraction=float(config.shield_view_count_shadow_retention_fraction),
        per_comparison_confidence=float(
            config.shield_view_count_shadow_per_comparison_confidence
        ),
    )
    scores = _shadow_scores_for_view_counts(
        reference_scores_a,
        reference_information_gains_a,
        decision.information_gain_mean_la,
        config=config,
    )
    critical = float(
        student_t.ppf(
            float(config.shield_view_count_shadow_per_comparison_confidence),
            int(decision.sample_count - 1),
        )
    )
    by_view_count: dict[str, object] = {}
    previous_samples = np.zeros_like(samples[0])
    previous_count = 0
    for length_index, view_count in enumerate(candidate_counts):
        current_samples = samples[length_index]
        increment_samples = current_samples - previous_samples
        increment_mean = np.mean(increment_samples, axis=1, dtype=np.float64)
        increment_se = np.std(increment_samples, axis=1, ddof=1) / np.sqrt(
            float(decision.sample_count)
        )
        increment_lcb = increment_mean - critical * increment_se
        added_live_time = float(
            (int(view_count) - int(previous_count)) * float(config.live_time_s)
        )
        retained = decision.retained_fraction_la[length_index]
        payload: dict[str, object] = {
            "pair_ids": np.asarray(
                programs_by_view_count[view_count],
                dtype=np.int64,
            ).tolist(),
            "program_semantics": "nested_conditional_greedy_prefix",
            "information_gain_mean_nat": decision.information_gain_mean_la[
                length_index
            ].tolist(),
            "information_gain_standard_error_nat": (
                decision.information_gain_standard_error_la[length_index].tolist()
            ),
            "retained_fraction_of_reference": [
                None if not np.isfinite(value) else float(value) for value in retained
            ],
            "measurement_live_time_s": float(
                int(view_count) * float(config.live_time_s)
            ),
            "pose_score": scores[length_index].tolist(),
            "nested_prefix_increment": {
                "previous_view_count": int(previous_count),
                "added_view_count": int(view_count) - int(previous_count),
                "mean_nat": increment_mean.tolist(),
                "paired_standard_error_nat": increment_se.tolist(),
                "one_sided_mc_lcb_nat": increment_lcb.tolist(),
                "added_live_time_s": added_live_time,
                "mean_nat_per_added_live_second": (
                    increment_mean / added_live_time
                ).tolist(),
            },
        }
        if length_index < len(candidate_counts) - 1:
            payload["retention_vs_reference"] = {
                "reference_view_count": int(candidate_counts[-1]),
                "threshold_fraction": float(decision.retention_fraction),
                "paired_margin_mean_nat": decision.retention_margin_mean_sa[
                    length_index
                ].tolist(),
                "paired_margin_standard_error_nat": (
                    decision.retention_margin_standard_error_sa[length_index].tolist()
                ),
                "paired_margin_one_sided_mc_lcb_nat": (
                    decision.retention_margin_lower_confidence_sa[length_index].tolist()
                ),
                "point_passed": decision.retention_point_passed_sa[
                    length_index
                ].tolist(),
                "lcb_passed": decision.retention_lcb_passed_sa[length_index].tolist(),
            }
        else:
            payload["retention_vs_reference"] = None
        by_view_count[str(view_count)] = payload
        previous_samples = current_samples
        previous_count = int(view_count)

    pose_indices = np.asarray(pose_indices_a, dtype=np.int64)
    poses = np.asarray(poses_a3, dtype=np.float64)
    return {
        "particle_count": None,
        "sample_count": int(decision.sample_count),
        "pose_count": int(pose_indices.size),
        "pose_indices": pose_indices.tolist(),
        "pose_xyz": poses.tolist(),
        "by_view_count": by_view_count,
        "point_selected_view_count_by_pose": (
            decision.point_selected_view_count_a.tolist()
        ),
        "paired_lcb_selected_view_count_by_pose": (
            decision.lcb_selected_view_count_a.tolist()
        ),
        "monotonicity_warning_by_pose": (decision.monotonicity_warning_a.tolist()),
        "point_rule_action": _shadow_selected_action_payload(
            selected_view_counts_a=decision.point_selected_view_count_a,
            scores_la=scores,
            information_gains_la=decision.information_gain_mean_la,
            programs_by_view_count=programs_by_view_count,
            candidate_view_counts=candidate_counts,
            pose_indices_a=pose_indices,
            poses_a3=poses,
        ),
        "paired_lcb_rule_action": _shadow_selected_action_payload(
            selected_view_counts_a=decision.lcb_selected_view_count_a,
            scores_la=scores,
            information_gains_la=decision.information_gain_mean_la,
            programs_by_view_count=programs_by_view_count,
            candidate_view_counts=candidate_counts,
            pose_indices_a=pose_indices,
            poses_a3=poses,
        ),
    }


def _shadow_proxy_diagnostics(
    *,
    pose_indices_a: NDArray[np.int64],
    poses_a3: NDArray[np.float64],
    evaluation: _ConditionalPoseEvaluation,
    reference_scores_a: NDArray[np.float64],
    config: DSSPPConfig,
    union_pose_indices_a: NDArray[np.int64],
    executed_shortlist_pose_indices_a: NDArray[np.int64],
) -> dict[str, object]:
    """Build aligned all-pose proxy evidence for the shadow audit."""
    candidate_counts = tuple(
        int(value) for value in config.shield_view_count_shadow_candidate_counts
    )
    gains = np.stack(
        [
            evaluation.shadow_prefix_information_gains_by_view_count[view_count][0]
            for view_count in candidate_counts
        ],
        axis=0,
    )
    programs = {
        view_count: np.asarray(
            evaluation.shadow_prefix_pair_ids_by_view_count[view_count][0],
            dtype=np.int64,
        )
        for view_count in candidate_counts
    }
    scores = _shadow_scores_for_view_counts(
        reference_scores_a,
        evaluation.information_gains_ra[0],
        gains,
        config=config,
    )
    reference = gains[-1]
    passed = (
        gains[:-1]
        >= float(config.shield_view_count_shadow_retention_fraction)
        * reference[np.newaxis, :]
    )
    any_passed = np.any(passed, axis=0)
    first_passed = np.argmax(passed, axis=0)
    selected_indices = np.where(
        any_passed,
        first_passed,
        len(candidate_counts) - 1,
    )
    selected_counts = np.asarray(candidate_counts, dtype=np.int64)[selected_indices]
    pose_indices = np.asarray(pose_indices_a, dtype=np.int64)
    poses = np.asarray(poses_a3, dtype=np.float64)
    return {
        "particle_count": int(evaluation.diagnostics["particle_count"]),
        "samples_per_seed": int(evaluation.diagnostics["sample_count"]),
        "pose_count": int(pose_indices.size),
        "stored_pose_scope": "all_valid_poses",
        "pose_indices": pose_indices.tolist(),
        "pose_xyz": poses.tolist(),
        "by_view_count": {
            str(view_count): {
                "pair_ids": programs[view_count].tolist(),
                "program_semantics": "nested_conditional_greedy_prefix",
                "information_gain_mean_nat": gains[length_index].tolist(),
                "pose_score": scores[length_index].tolist(),
            }
            for length_index, view_count in enumerate(candidate_counts)
        },
        "point_selected_view_count_by_pose": selected_counts.tolist(),
        "point_rule_action": _shadow_selected_action_payload(
            selected_view_counts_a=selected_counts,
            scores_la=scores,
            information_gains_la=gains,
            programs_by_view_count=programs,
            candidate_view_counts=candidate_counts,
            pose_indices_a=pose_indices,
            poses_a3=poses,
        ),
        "executed_fixed_8_shortlist_pose_indices": np.asarray(
            executed_shortlist_pose_indices_a,
            dtype=np.int64,
        ).tolist(),
        "view_count_union_exact_pose_indices": np.asarray(
            union_pose_indices_a,
            dtype=np.int64,
        ).tolist(),
    }


def _conditional_pose_ambiguity_mask(
    score_samples_aq: NDArray[np.float64],
    mean_scores_a: NDArray[np.float64],
    *,
    confidence: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return poses whose paired score gap from the leader is not established."""
    samples = np.asarray(score_samples_aq, dtype=np.float64)
    means = np.asarray(mean_scores_a, dtype=np.float64).reshape(-1)
    if (
        samples.ndim != 2
        or samples.shape[0] != means.size
        or samples.shape[1] < 2
        or np.any(~np.isfinite(samples))
        or np.any(~np.isfinite(means))
    ):
        raise ValueError("Pose ambiguity requires finite aligned MC scores.")
    leader = int(_stable_descending_indices(means)[0])
    leader_samples = np.broadcast_to(samples[leader], samples.shape)
    _, _, lower = _paired_gap_lower_confidence(
        leader_samples,
        samples,
        confidence=float(confidence),
    )
    lower[leader] = np.inf
    ambiguous = lower <= 0.0
    ambiguous[leader] = True
    return np.asarray(ambiguous, dtype=np.bool_), np.asarray(lower, dtype=np.float64)


def _build_conditional_nodes(
    *,
    estimator: RotatingShieldPFEstimator,
    candidate_poses: NDArray[np.float64],
    path_lengths: NDArray[np.float64],
    coverage_norm: NDArray[np.float64],
    coverage_raw: NDArray[np.float64],
    revisit_penalties: NDArray[np.float64],
    bearing_gains: NDArray[np.float64],
    frontier_gains: NDArray[np.float64],
    turn_penalties: NDArray[np.float64],
    local_orbit_gains: NDArray[np.float64],
    elevation_condition_gains: NDArray[np.float64],
    coverage_floor: float,
    coverage_support: str,
    coverage_quadrature_diagnostics: dict[str, object] | None,
    config: DSSPPConfig,
    rng: np.random.Generator,
    joint_particles: JointPlanningParticles,
    motion_times: NDArray[np.float64] | None,
    motion_time_components: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None,
    path_length_support: str,
) -> tuple[list[DSSPPNode], dict[str, object]]:
    """Build standard DSS nodes with all-pair conditional program search."""
    conditional_started = time.perf_counter()
    poses = np.asarray(candidate_poses, dtype=np.float64)
    paths = np.asarray(path_lengths, dtype=np.float64).reshape(-1)
    valid_pose_indices = np.flatnonzero(np.isfinite(paths)).astype(
        np.int64,
        copy=False,
    )
    if valid_pose_indices.size == 0:
        return [], {
            "total_action_count": 0,
            "proxy_action_count": 0,
            "exact_action_count": 0,
            "path_length_support": path_length_support,
        }
    static_scores = _static_station_scores_batch(
        coverage_norm_p=coverage_norm,
        revisit_penalties_p=revisit_penalties,
        bearing_gains_p=bearing_gains,
        frontier_gains_p=frontier_gains,
        turn_penalties_p=turn_penalties,
        local_orbit_gains_p=local_orbit_gains,
        elevation_condition_gains_p=elevation_condition_gains,
        coverage_floor=float(coverage_floor),
        config=config,
    )
    exact_seed = int(
        rng.integers(
            0,
            np.iinfo(np.int64).max,
            endpoint=False,
            dtype=np.int64,
        )
    )
    available_pose_count = int(valid_pose_indices.size)
    proxy_seed: int | None = None
    proxy_evaluation: _ConditionalPoseEvaluation | None = None
    refinement_evaluation: _ConditionalPoseEvaluation | None = None
    refinement_pose_count = 0
    proxy_replica_scores = np.full(
        (int(config.proxy_stability_replicates), available_pose_count),
        np.nan,
        dtype=np.float64,
    )
    if available_pose_count <= int(config.exact_eig_pose_min):
        shortlisted_local_indices = np.arange(
            available_pose_count,
            dtype=np.int64,
        )
        shortlist_stop_reason = "all_poses_fit_minimum"
        shortlist_boundaries: list[dict[str, object]] = []
        coverage_reserve_pose = None
    else:
        proxy_particles = estimator.planning_joint_particles(
            max_particles=int(config.proxy_planning_particles),
            method="top_weight",
        )
        proxy_seed = int(
            named_stream_seed(
                exact_seed,
                "dss_pp",
                "conditional_all_pairs",
                "proxy_base",
            )
            & ((1 << 63) - 1)
        )
        proxy_evaluation = _evaluate_conditional_pose_batches(
            estimator=estimator,
            detector_positions_a3=poses[valid_pose_indices],
            joint_particles=proxy_particles,
            config=config,
            sample_count=int(config.proxy_eig_samples),
            eig_call_seeds_r=np.asarray([proxy_seed], dtype=np.int64),
            stream_name="proxy_all_poses",
            workload="proxy",
            working_memory_budget_bytes=int(config.proxy_memory_budget_bytes),
            maximum_subset_candidate_count=(int(estimator.num_orientations) ** 2),
            enable_one_swap=False,
            confirm_ambiguous=False,
        )
        proxy_motion_times = (
            None if motion_times is None else motion_times[valid_pose_indices]
        )
        proxy_motion_components = (
            None
            if motion_time_components is None
            else tuple(values[valid_pose_indices] for values in motion_time_components)
        )
        base_proxy_scores, proxy_distance_weight = compose_pose_scores(
            proxy_evaluation.information_gains_ra[0],
            static_scores[valid_pose_indices],
            paths[valid_pose_indices],
            config=config,
            motion_times_p=proxy_motion_times,
            motion_time_components_p=proxy_motion_components,
        )
        proxy_replica_scores[0] = base_proxy_scores
        refinement_count = min(
            int(config.proxy_stability_refinement_pool),
            available_pose_count,
        )
        refinement_pose_count = int(refinement_count)
        refinement_local_indices = _stable_descending_indices(base_proxy_scores)[
            :refinement_count
        ]
        extra_replica_count = int(config.proxy_stability_replicates) - 1
        if extra_replica_count > 0:
            refinement_seeds = np.asarray(
                [
                    named_stream_seed(
                        exact_seed,
                        "dss_pp",
                        "conditional_all_pairs",
                        "proxy_refinement",
                        int(replica_index),
                    )
                    & ((1 << 63) - 1)
                    for replica_index in range(extra_replica_count)
                ],
                dtype=np.int64,
            )
            refinement_evaluation = _evaluate_conditional_pose_batches(
                estimator=estimator,
                detector_positions_a3=poses[
                    valid_pose_indices[refinement_local_indices]
                ],
                joint_particles=proxy_particles,
                config=config,
                sample_count=int(config.proxy_eig_samples),
                eig_call_seeds_r=refinement_seeds,
                stream_name="proxy_refinement_pool",
                workload="proxy",
                working_memory_budget_bytes=int(config.proxy_memory_budget_bytes),
                maximum_subset_candidate_count=(int(estimator.num_orientations) ** 2),
                enable_one_swap=False,
                confirm_ambiguous=False,
            )
            resolved_proxy_config = replace(
                config,
                lambda_distance=float(proxy_distance_weight),
            )
            for replica_index in range(extra_replica_count):
                refined_scores, _ = compose_pose_scores(
                    refinement_evaluation.information_gains_ra[replica_index],
                    static_scores[valid_pose_indices[refinement_local_indices]],
                    paths[valid_pose_indices[refinement_local_indices]],
                    config=resolved_proxy_config,
                    motion_times_p=(
                        None
                        if motion_times is None
                        else motion_times[valid_pose_indices[refinement_local_indices]]
                    ),
                    motion_time_components_p=(
                        None
                        if motion_time_components is None
                        else tuple(
                            values[valid_pose_indices[refinement_local_indices]]
                            for values in motion_time_components
                        )
                    ),
                )
                proxy_replica_scores[
                    replica_index + 1,
                    refinement_local_indices,
                ] = refined_scores
        shortlist = select_adaptive_pose_shortlist(
            proxy_replica_scores,
            np.asarray(coverage_raw, dtype=np.float64)[valid_pose_indices],
            minimum_pose_count=int(config.exact_eig_pose_min),
            maximum_pose_count=int(config.exact_eig_pose_max),
            pose_count_step=int(config.exact_eig_pose_step),
            coverage_reserve_count=int(config.exact_eig_coverage_reserve),
            boundary_confidence=float(config.proxy_boundary_confidence),
            minimum_top_k_jaccard=float(config.proxy_top_k_jaccard_min),
        )
        shortlisted_local_indices = shortlist.pose_indices
        shortlist_stop_reason = str(shortlist.stop_reason)
        coverage_reserve_pose = (
            None
            if shortlist.coverage_reserve_pose is None
            else int(valid_pose_indices[int(shortlist.coverage_reserve_pose)])
        )
        shortlist_boundaries = [
            {
                "pose_count": int(item.pose_count),
                "boundary_included_pose": (
                    None
                    if item.boundary_included_pose is None
                    else int(valid_pose_indices[item.boundary_included_pose])
                ),
                "boundary_excluded_pose": (
                    None
                    if item.boundary_excluded_pose is None
                    else int(valid_pose_indices[item.boundary_excluded_pose])
                ),
                "paired_gap_mean": item.paired_gap_mean,
                "paired_gap_standard_error": item.paired_gap_standard_error,
                "paired_gap_lower_confidence": item.paired_gap_lower_confidence,
                "mean_top_k_jaccard": float(item.mean_top_k_jaccard),
                "stable": bool(item.stable),
            }
            for item in shortlist.boundary_diagnostics
        ]

    exact_pose_indices = valid_pose_indices[shortlisted_local_indices]
    shadow_union_local_indices = np.asarray(
        shortlisted_local_indices,
        dtype=np.int64,
    )
    if bool(config.shield_view_count_shadow_enabled) and proxy_evaluation is not None:
        shadow_counts = tuple(
            int(value) for value in config.shield_view_count_shadow_candidate_counts
        )
        shadow_proxy_gains = np.stack(
            [
                proxy_evaluation.shadow_prefix_information_gains_by_view_count[
                    view_count
                ][0]
                for view_count in shadow_counts
            ],
            axis=0,
        )
        shadow_proxy_scores = _shadow_scores_for_view_counts(
            base_proxy_scores,
            proxy_evaluation.information_gains_ra[0],
            shadow_proxy_gains,
            config=config,
        )
        shadow_union_local_indices = _shadow_union_pose_indices(
            shortlisted_local_indices,
            shadow_proxy_scores,
            maximum_pose_count=int(config.exact_eig_pose_max),
        )
    shadow_union_pose_indices = valid_pose_indices[shadow_union_local_indices]
    shadow_extra_local_indices = shadow_union_local_indices[
        ~np.isin(shadow_union_local_indices, shortlisted_local_indices)
    ]
    shadow_extra_pose_indices = valid_pose_indices[shadow_extra_local_indices]
    conditional_pair_count = int(estimator.num_orientations) ** 2
    conditional_swap_count = (
        int(config.program_length)
        * (conditional_pair_count - int(config.program_length))
        if bool(config.conditional_greedy_one_swap)
        else 0
    )
    exact_maximum_candidate_count = max(
        conditional_pair_count,
        conditional_swap_count,
    )
    exact_evaluation = _evaluate_conditional_pose_batches(
        estimator=estimator,
        detector_positions_a3=poses[exact_pose_indices],
        joint_particles=joint_particles,
        config=config,
        sample_count=int(estimator.pf_config.planning_eig_samples),
        eig_call_seeds_r=np.asarray([exact_seed], dtype=np.int64),
        stream_name="exact_shortlist",
        workload="exact",
        working_memory_budget_bytes=int(config.exact_eig_memory_budget_bytes),
        maximum_subset_candidate_count=int(exact_maximum_candidate_count),
        enable_one_swap=bool(config.conditional_greedy_one_swap),
        confirm_ambiguous=True,
    )
    exact_gains = exact_evaluation.information_gains_ra[0].copy()
    exact_programs = exact_evaluation.program_pair_ids_ral[0].copy()
    exact_sources = list(exact_evaluation.selection_sources_r[0])
    final_eig_sample_counts = np.asarray(
        [
            int(values.size)
            for values in exact_evaluation.selected_combined_kl_samples_r[0]
        ],
        dtype=np.int64,
    )
    exact_motion_times = (
        None if motion_times is None else motion_times[exact_pose_indices]
    )
    exact_motion_components = (
        None
        if motion_time_components is None
        else tuple(values[exact_pose_indices] for values in motion_time_components)
    )
    exact_scores, exact_distance_weight = compose_pose_scores(
        exact_gains,
        static_scores[exact_pose_indices],
        paths[exact_pose_indices],
        config=config,
        motion_times_p=exact_motion_times,
        motion_time_components_p=exact_motion_components,
    )

    pose_confirmation_count = 0
    pose_confirmation_wall_s = 0.0
    pose_confirmation_memory_plan: dict[str, object] = {}
    pose_gap_lower = np.full(exact_pose_indices.size, np.inf, dtype=np.float64)
    if float(config.lambda_eig) > 0.0 and exact_pose_indices.size > 1:
        base_kl_samples = exact_evaluation.selected_base_kl_samples_raq[0]
        base_program_gains = np.mean(
            base_kl_samples,
            axis=1,
            dtype=np.float64,
        )
        base_pose_scores, _base_distance_weight = compose_pose_scores(
            base_program_gains,
            static_scores[exact_pose_indices],
            paths[exact_pose_indices],
            config=config,
            motion_times_p=exact_motion_times,
            motion_time_components_p=exact_motion_components,
        )
        deterministic_scores = (
            base_pose_scores - float(config.lambda_eig) * base_program_gains
        )
        base_score_samples = (
            deterministic_scores[:, np.newaxis]
            + float(config.lambda_eig) * base_kl_samples
        )
        ambiguous_pose_mask, pose_gap_lower = _conditional_pose_ambiguity_mask(
            base_score_samples,
            base_pose_scores,
            confidence=float(config.proxy_boundary_confidence),
        )
        if int(np.count_nonzero(ambiguous_pose_mask)) > 1:
            pose_confirmation_started = time.perf_counter()
            ambiguous_local_indices = np.flatnonzero(ambiguous_pose_mask).astype(
                np.int64,
                copy=False,
            )
            pose_confirmation_count = int(ambiguous_local_indices.size)
            ambiguous_pose_indices = exact_pose_indices[ambiguous_local_indices]
            confirmation_seed = int(
                named_stream_seed(
                    exact_seed,
                    "dss_pp",
                    "conditional_all_pairs",
                    "ambiguous_pose_confirmation",
                )
                & ((1 << 63) - 1)
            )
            (
                confirmation_kl,
                pose_confirmation_memory_plan,
            ) = _evaluate_fixed_programs_in_pose_batches(
                estimator=estimator,
                detector_positions_a3=poses[ambiguous_pose_indices],
                program_pair_ids_al=exact_programs[ambiguous_local_indices],
                joint_particles=joint_particles,
                config=config,
                sample_count=int(estimator.pf_config.planning_eig_samples),
                eig_call_seed=confirmation_seed,
                stream_name="ambiguous_pose_confirmation",
                working_memory_budget_bytes=int(config.exact_eig_memory_budget_bytes),
            )
            confirmed_gains = np.asarray(
                [
                    np.mean(
                        np.concatenate(
                            (
                                exact_evaluation.selected_combined_kl_samples_r[0][
                                    int(action_index)
                                ],
                                confirmation_kl[local_index],
                            )
                        ),
                        dtype=np.float64,
                    )
                    for local_index, action_index in enumerate(ambiguous_local_indices)
                ],
                dtype=np.float64,
            )
            exact_gains[ambiguous_local_indices] = confirmed_gains
            final_eig_sample_counts[ambiguous_local_indices] += int(
                estimator.pf_config.planning_eig_samples
            )
            exact_scores, exact_distance_weight = compose_pose_scores(
                exact_gains,
                static_scores[exact_pose_indices],
                paths[exact_pose_indices],
                config=config,
                motion_times_p=exact_motion_times,
                motion_time_components_p=exact_motion_components,
            )
            pose_confirmation_wall_s = float(
                time.perf_counter() - pose_confirmation_started
            )

    shadow_diagnostics: dict[str, object] | None = None
    shadow_extra_evaluation: _ConditionalPoseEvaluation | None = None
    if bool(config.shield_view_count_shadow_enabled):
        shadow_extra_scores = np.empty(0, dtype=np.float64)
        shadow_extra_gains = np.empty(0, dtype=np.float64)
        shadow_extra_seed: int | None = None
        if shadow_extra_pose_indices.size:
            shadow_extra_seed = int(
                named_stream_seed(
                    exact_seed,
                    "dss_pp",
                    "shield_view_count_shadow",
                    "exact_union_extra",
                )
                & ((1 << 63) - 1)
            )
            shadow_extra_evaluation = _evaluate_conditional_pose_batches(
                estimator=estimator,
                detector_positions_a3=poses[shadow_extra_pose_indices],
                joint_particles=joint_particles,
                config=config,
                sample_count=int(estimator.pf_config.planning_eig_samples),
                eig_call_seeds_r=np.asarray(
                    [shadow_extra_seed],
                    dtype=np.int64,
                ),
                stream_name="exact_shadow_union_extra",
                workload="exact",
                working_memory_budget_bytes=int(config.exact_eig_memory_budget_bytes),
                maximum_subset_candidate_count=int(conditional_pair_count),
                enable_one_swap=False,
                confirm_ambiguous=False,
            )
            shadow_extra_gains = shadow_extra_evaluation.information_gains_ra[0].copy()
            shadow_extra_scores, _ = compose_pose_scores(
                shadow_extra_gains,
                static_scores[shadow_extra_pose_indices],
                paths[shadow_extra_pose_indices],
                config=replace(
                    config,
                    lambda_distance=float(exact_distance_weight),
                ),
                motion_times_p=(
                    None
                    if motion_times is None
                    else motion_times[shadow_extra_pose_indices]
                ),
                motion_time_components_p=(
                    None
                    if motion_time_components is None
                    else tuple(
                        values[shadow_extra_pose_indices]
                        for values in motion_time_components
                    )
                ),
            )

        shadow_counts = tuple(
            int(value) for value in config.shield_view_count_shadow_candidate_counts
        )
        shadow_exact_programs = {
            view_count: np.concatenate(
                (
                    exact_evaluation.shadow_prefix_pair_ids_by_view_count[view_count][
                        0
                    ],
                    (
                        np.empty((0, view_count), dtype=np.int64)
                        if shadow_extra_evaluation is None
                        else shadow_extra_evaluation.shadow_prefix_pair_ids_by_view_count[
                            view_count
                        ][0]
                    ),
                ),
                axis=0,
            )
            for view_count in shadow_counts
        }
        shadow_exact_samples = {
            view_count: np.concatenate(
                (
                    exact_evaluation.shadow_prefix_kl_samples_by_view_count[view_count][
                        0
                    ],
                    (
                        np.empty(
                            (
                                0,
                                int(estimator.pf_config.planning_eig_samples),
                            ),
                            dtype=np.float64,
                        )
                        if shadow_extra_evaluation is None
                        else shadow_extra_evaluation.shadow_prefix_kl_samples_by_view_count[
                            view_count
                        ][0]
                    ),
                ),
                axis=0,
            )
            for view_count in shadow_counts
        }
        shadow_exact = _shadow_exact_diagnostics(
            pose_indices_a=shadow_union_pose_indices,
            poses_a3=poses[shadow_union_pose_indices],
            programs_by_view_count=shadow_exact_programs,
            kl_samples_by_view_count=shadow_exact_samples,
            reference_scores_a=np.concatenate(
                (exact_scores, shadow_extra_scores),
            ),
            reference_information_gains_a=np.concatenate(
                (exact_gains, shadow_extra_gains),
            ),
            config=config,
        )
        shadow_exact["particle_count"] = int(
            exact_evaluation.diagnostics["particle_count"]
        )
        shadow_exact["prefix_selection_seed_blocks"] = [
            {
                "seed": int(exact_seed),
                "stream_name": "exact_shortlist_replica_0",
                "pose_indices": [int(value) for value in exact_pose_indices],
                "samples_per_pose": int(estimator.pf_config.planning_eig_samples),
            },
            *(
                []
                if shadow_extra_seed is None
                else [
                    {
                        "seed": int(shadow_extra_seed),
                        "stream_name": "exact_shadow_union_extra_replica_0",
                        "pose_indices": [
                            int(value) for value in shadow_extra_pose_indices
                        ],
                        "samples_per_pose": int(
                            estimator.pf_config.planning_eig_samples
                        ),
                    }
                ]
            ),
        ]
        shadow_exact["paired_evaluation_holdout_seed_blocks"] = [
            *_shadow_holdout_seed_blocks(
                exact_evaluation,
                exact_pose_indices,
            ),
            *(
                []
                if shadow_extra_evaluation is None
                else _shadow_holdout_seed_blocks(
                    shadow_extra_evaluation,
                    shadow_extra_pose_indices,
                )
            ),
        ]
        if proxy_evaluation is None:
            shadow_proxy: dict[str, object] = {
                "status": "skipped_all_valid_poses_exact",
                "pose_count": int(available_pose_count),
                "stored_pose_scope": "all_valid_poses_exact_instead",
                "executed_fixed_8_shortlist_pose_indices": [
                    int(value) for value in exact_pose_indices
                ],
                "view_count_union_exact_pose_indices": [
                    int(value) for value in shadow_union_pose_indices
                ],
            }
        else:
            shadow_proxy = _shadow_proxy_diagnostics(
                pose_indices_a=valid_pose_indices,
                poses_a3=poses[valid_pose_indices],
                evaluation=proxy_evaluation,
                reference_scores_a=base_proxy_scores,
                config=config,
                union_pose_indices_a=shadow_union_pose_indices,
                executed_shortlist_pose_indices_a=exact_pose_indices,
            )
            shadow_proxy["status"] = "evaluated"
            if proxy_seed is None:
                raise RuntimeError("Evaluated shadow proxy lost its MC seed.")
            shadow_proxy["selection_seed_blocks"] = [
                {
                    "seed": int(proxy_seed),
                    "stream_name": "proxy_all_poses_replica_0",
                    "pose_indices": [int(value) for value in valid_pose_indices],
                    "samples_per_pose": int(config.proxy_eig_samples),
                }
            ]
        shadow_diagnostics = {
            "schema_version": 1,
            "status": "evaluated",
            "mode": "audit_only_fixed_8_execution",
            "truth_used": False,
            "policy": {
                "candidate_view_counts": [int(value) for value in shadow_counts],
                "reference_view_count": int(shadow_counts[-1]),
                "retention_fraction": float(
                    config.shield_view_count_shadow_retention_fraction
                ),
                "per_comparison_one_sided_confidence": float(
                    config.shield_view_count_shadow_per_comparison_confidence
                ),
                "global_coverage_claimed": False,
                "selection_statistic": (
                    "paired_lcb_of_information_gain_short_minus_retention_"
                    "times_information_gain_reference"
                ),
                "lcb_pass_condition": "strictly_greater_than_zero",
                "program_semantics": "nested_conditional_greedy_prefix",
            },
            "mc_contract": {
                "status": "evaluated",
                "paired_across_view_counts": True,
                "paired_across_poses": False,
                "paired_across_proxy_and_exact": False,
                "prefix_selection_independent_of_exact_lcb_samples": True,
                "selection_bias_control": (
                    "conditional_greedy_prefixes_fixed_before_independent_"
                    "holdout_evaluation"
                ),
                "predictive_pairing": (
                    "same_all_64_view_draw_shared_latent_and_shared_nuisance"
                ),
            },
            "proxy": shadow_proxy,
            "exact": shadow_exact,
            "executed_shortlist_pose_count": int(exact_pose_indices.size),
            "view_count_union_exact_pose_count": int(shadow_union_pose_indices.size),
            "view_count_union_extra_pose_count": int(shadow_extra_pose_indices.size),
            "view_count_union_capacity": int(config.exact_eig_pose_max),
            "shadow_extra_eig_runtime": (
                {}
                if shadow_extra_evaluation is None
                else dict(shadow_extra_evaluation.diagnostics)
            ),
        }

    source_kind = {
        "greedy": "conditional_greedy_all_pairs",
        "one_swap": "conditional_greedy_one_swap",
    }
    nodes = [
        DSSPPNode(
            pose_index=int(pose_index),
            pose_xyz=poses[int(pose_index)].copy(),
            program=ShieldProgram(
                name=f"{exact_sources[row]}_pose_{int(pose_index):03d}",
                pair_ids=tuple(int(value) for value in exact_programs[row]),
                kind=source_kind[str(exact_sources[row])],
            ),
            score=float(exact_scores[row]),
            static_score=float(
                static_scores[int(pose_index)]
                + float(config.lambda_eig) * exact_gains[row]
            ),
            distance_weight=float(exact_distance_weight),
            information_gain=float(exact_gains[row]),
            coverage_gain=float(coverage_raw[int(pose_index)]),
            revisit_penalty=float(revisit_penalties[int(pose_index)]),
            bearing_diversity_gain=float(bearing_gains[int(pose_index)]),
            frontier_gain=float(frontier_gains[int(pose_index)]),
            turn_penalty=float(turn_penalties[int(pose_index)]),
            local_orbit_gain=float(local_orbit_gains[int(pose_index)]),
            elevation_condition_gain=float(elevation_condition_gains[int(pose_index)]),
        )
        for row, pose_index in enumerate(exact_pose_indices)
    ]
    nodes.sort(key=lambda node: (-float(node.score), int(node.pose_index)))
    greedy_count = int(
        int(config.program_length) * (int(estimator.num_orientations) ** 2)
        - int(config.program_length) * (int(config.program_length) - 1) // 2
    )
    swap_count = int(
        int(config.program_length)
        * (int(estimator.num_orientations) ** 2 - int(config.program_length))
        if bool(config.conditional_greedy_one_swap)
        else 0
    )
    contender_count = 1 + int(swap_count > 0)
    program_confirmation_count = int(
        sum(
            int(replica["independently_confirmed_program_pose_count"])
            for chunk in exact_evaluation.diagnostics["chunks"]
            for replica in chunk["replicas"]
        )
    )
    proxy_subset_evaluation_count = int(
        (0 if proxy_evaluation is None else available_pose_count * greedy_count)
        + (
            0
            if refinement_evaluation is None
            else refinement_pose_count
            * (int(config.proxy_stability_replicates) - 1)
            * greedy_count
        )
    )
    exact_subset_evaluation_count = int(
        exact_pose_indices.size * (greedy_count + swap_count + contender_count)
        + program_confirmation_count * contender_count
        + pose_confirmation_count
    )
    diagnostics: dict[str, object] = {
        "candidate_pose_count": int(available_pose_count),
        "total_action_count": int(available_pose_count),
        "path_length_support": str(path_length_support),
        "proxy_action_count": int(proxy_subset_evaluation_count),
        "proxy_subset_evaluation_count": int(proxy_subset_evaluation_count),
        "proxy_particle_count": int(
            0
            if proxy_evaluation is None
            else proxy_evaluation.diagnostics["particle_count"]
        ),
        "proxy_eig_samples": int(config.proxy_eig_samples),
        "exact_action_count": int(exact_pose_indices.size),
        "exact_subset_evaluation_count": int(exact_subset_evaluation_count),
        "shortlisted_pose_count": int(exact_pose_indices.size),
        "programs_per_shortlisted_pose": 1,
        "full_program_sweep_per_shortlisted_pose": False,
        "pose_shortlist_contract": (
            "proxy_conditional_greedy_plus_spatial_minus_motion_with_"
            "paired_mc_stability_adaptive_8_12_16"
        ),
        "program_search_contract": "all_pairs_conditional_greedy_then_one_swap",
        "exact_eig_seed": int(exact_seed),
        "adaptive_exact_eig_round_count": 1,
        "adaptive_exact_eig_exhausted_all_actions": bool(
            exact_pose_indices.size == available_pose_count
        ),
        "adaptive_shortlist_stop_reason": str(shortlist_stop_reason),
        "adaptive_shortlist_boundaries": shortlist_boundaries,
        "adaptive_shortlist_coverage_reserve_pose": coverage_reserve_pose,
        "adaptive_shortlist_pose_indices": [int(value) for value in exact_pose_indices],
        "exact_pose_results": [
            {
                "pose_index": int(pose_index),
                "pose_xyz": [float(value) for value in poses[int(pose_index)]],
                "program_pair_ids": [int(value) for value in exact_programs[row]],
                "selection_source": str(exact_sources[row]),
                "information_gain": float(exact_gains[row]),
                "information_gain_sample_count": int(final_eig_sample_counts[row]),
                "pose_score": float(exact_scores[row]),
            }
            for row, pose_index in enumerate(exact_pose_indices)
        ],
        "proxy_replica_scores": _proxy_replica_scores_payload(
            proxy_replica_scores,
            evaluated=proxy_evaluation is not None,
        ),
        "proxy_eig_runtime": (
            {} if proxy_evaluation is None else dict(proxy_evaluation.diagnostics)
        ),
        "proxy_refinement_eig_runtime": (
            {}
            if refinement_evaluation is None
            else dict(refinement_evaluation.diagnostics)
        ),
        "exact_eig_runtime": dict(exact_evaluation.diagnostics),
        "pose_gap_lower_confidence_initial": [
            None if not np.isfinite(value) else float(value) for value in pose_gap_lower
        ],
        "independently_confirmed_pose_count": int(pose_confirmation_count),
        "pose_confirmation_wall_s": float(pose_confirmation_wall_s),
        "pose_confirmation_memory_chunk_plan": dict(pose_confirmation_memory_plan),
        "conditional_greedy_candidate_count_per_pose": int(greedy_count),
        "one_swap_candidate_count_per_pose": int(swap_count),
        "independently_confirmed_program_pose_count": int(program_confirmation_count),
        "coverage_support": str(coverage_support),
        "coverage_quadrature": coverage_quadrature_diagnostics,
        "eig_shortlist_wall_s": float(time.perf_counter() - conditional_started),
        "shield_view_count_shadow": shadow_diagnostics,
    }
    return nodes, diagnostics


def _build_nodes(
    *,
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    programs: Sequence[ShieldProgram],
    modes_by_isotope: dict[str, list[SignatureMode]],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    config: DSSPPConfig,
    rng: np.random.Generator,
    joint_particles: JointPlanningParticles,
    candidate_motion_times_s: NDArray[np.float64] | None = None,
    candidate_motion_time_components_s: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None = None,
) -> tuple[list[DSSPPNode], dict[str, object]]:
    """Shortlist all actions cheaply, then exactly evaluate a fixed subset."""
    kernel = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=int(config.detector_aperture_samples),
    )
    candidate_poses = np.asarray(candidate_poses_xyz, dtype=float)
    if candidate_poses.ndim != 2 or candidate_poses.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    motion_times = None
    if candidate_motion_times_s is not None:
        motion_times = np.asarray(
            candidate_motion_times_s,
            dtype=np.float64,
        ).reshape(-1)
        if (
            motion_times.shape != (candidate_poses.shape[0],)
            or np.any(~np.isfinite(motion_times))
            or np.any(motion_times < 0.0)
        ):
            raise ValueError(
                "candidate_motion_times_s must align with candidates and "
                "contain finite nonnegative values."
            )
    motion_time_components = None
    if candidate_motion_time_components_s is not None:
        if len(candidate_motion_time_components_s) != 3:
            raise ValueError("Candidate motion times require exactly three components.")
        motion_time_components = tuple(
            np.asarray(values, dtype=np.float64).reshape(-1)
            for values in candidate_motion_time_components_s
        )
        if any(
            values.shape != (candidate_poses.shape[0],)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            for values in motion_time_components
        ):
            raise ValueError(
                "Candidate motion-time components must align with candidates and "
                "contain finite nonnegative values."
            )
        if motion_times is None or not np.array_equal(
            np.sum(np.vstack(motion_time_components), axis=0),
            motion_times,
        ):
            raise ValueError(
                "Candidate motion-time components must sum to motion times."
            )

    def _motion_component_overrides(pose_index: int) -> dict[str, float | None]:
        """Return score-call overrides for one candidate motion quote."""
        if motion_time_components is None:
            return {
                "horizontal_time_override_s": None,
                "mast_vertical_time_override_s": None,
                "settling_time_override_s": None,
            }
        return {
            "horizontal_time_override_s": float(
                motion_time_components[0][int(pose_index)]
            ),
            "mast_vertical_time_override_s": float(
                motion_time_components[1][int(pose_index)]
            ),
            "settling_time_override_s": float(
                motion_time_components[2][int(pose_index)]
            ),
        }

    info_gains = np.zeros(candidate_poses.shape[0], dtype=float)
    map_has_native_motion = map_api is None or callable(
        getattr(map_api, "motion_path_lengths_batch", None)
    )
    runtime_motion_time_only = motion_times is not None and not map_has_native_motion
    if runtime_motion_time_only:
        if config.lambda_distance is None or float(config.lambda_distance) != 0.0:
            raise ValueError(
                "Runtime-only motion times require lambda_distance=0; "
                "time-valued costs cannot be reinterpreted as path lengths."
            )
        path_lengths = np.zeros(candidate_poses.shape[0], dtype=np.float64)
        path_length_support = "runtime_reachable_candidates_with_time_cost_only"
    else:
        path_lengths = _node_path_lengths_batch(
            map_api,
            current_pose_xyz,
            candidate_poses,
        )
        path_length_support = (
            "explicit_obstacle_free_euclidean"
            if map_api is None
            else "runtime_native_motion_geometry"
        )
    surface_quadrature_builder = getattr(
        estimator,
        "surface_atlas_area_quadrature",
        None,
    )
    coverage_quadrature_diagnostics: dict[str, object] | None = None
    coverage_enabled = bool(
        float(config.lambda_coverage) > 0.0
        or float(config.coverage_floor_weight) > 0.0
        or int(config.exact_eig_coverage_reserve) > 0
    )
    if not coverage_enabled:
        coverage_raw = np.zeros(candidate_poses.shape[0], dtype=np.float64)
        coverage_quadrature_diagnostics = {
            "enabled": False,
            "sample_count": 0,
        }
        coverage_support = "disabled"
    elif callable(surface_quadrature_builder):
        if (
            config.coverage_surface_quadrature_max_points is None
            or config.coverage_surface_max_hausdorff_m is None
        ):
            raise ValueError(
                "Enabled coverage requires explicit surface quadrature settings."
            )
        surface_quadrature = surface_quadrature_builder(
            max_points=int(config.coverage_surface_quadrature_max_points),
            maximum_hausdorff_bound_m=float(config.coverage_surface_max_hausdorff_m),
        )
        surface_coverage_points = np.asarray(
            surface_quadrature.positions_s3,
            dtype=np.float64,
        ).reshape(-1, 3)
        surface_area_weights_m2 = np.asarray(
            surface_quadrature.area_weights_m2_s,
            dtype=np.float64,
        ).reshape(-1)
        coverage_raw = _response_equivalent_surface_coverage_gains(
            kernel=kernel,
            estimator=estimator,
            surface_points_xyz=surface_coverage_points,
            surface_area_weights_m2=surface_area_weights_m2,
            candidate_poses_xyz=candidate_poses,
            reference_radius_m=float(config.coverage_radius_m),
        )
        diagnostics_getter = getattr(
            surface_quadrature,
            "diagnostics",
            None,
        )
        if not callable(diagnostics_getter):
            raise TypeError("Surface quadrature must expose completeness diagnostics.")
        coverage_quadrature_diagnostics = dict(diagnostics_getter())
        coverage_support = (
            "complete_chart_center_area_weighted_unshielded_station_coverage"
        )
    elif isinstance(estimator, RotatingShieldPFEstimator):
        raise RuntimeError(
            "Production DSS coverage requires the PF continuous physical "
            "surface atlas; an XY/free-cell fallback is forbidden."
        )
    else:
        # A small deterministic oracle remains available only to unit-test
        # score composition without constructing a production estimator.
        if config.coverage_surface_quadrature_max_points is None:
            raise ValueError(
                "Enabled coverage requires an explicit quadrature point limit."
            )
        surface_coverage_points = _free_cell_centers(
            map_api,
            z_value=float(current_pose_xyz[2]),
            max_cells=int(config.coverage_surface_quadrature_max_points),
            bounds_xyz=bounds_xyz,
        )
        coverage_raw = _coverage_gain_fractions_batch(
            cell_centers_xyz=surface_coverage_points,
            candidate_poses_xyz=candidate_poses,
            visited_poses_xyz=visited_poses_xyz,
            radius_m=float(config.coverage_radius_m),
        )
        coverage_support = "test_only_free_cell_oracle_3d"
    coverage_norm = coverage_raw.copy()
    max_coverage = float(np.max(coverage_norm)) if coverage_norm.size else 0.0
    if max_coverage > 0.0:
        coverage_norm = coverage_norm / max_coverage
    coverage_floor = 0.0
    coverage_floor_quantile = float(config.coverage_floor_quantile)
    if (
        coverage_norm.size
        and float(config.coverage_floor_weight) > 0.0
        and coverage_floor_quantile > 0.0
    ):
        positive_coverage = coverage_norm[coverage_norm > 0.0]
        if positive_coverage.size:
            coverage_floor = float(
                np.quantile(
                    positive_coverage,
                    coverage_floor_quantile,
                )
            )
    revisit_penalties = _station_revisit_penalties_batch(
        candidate_poses,
        visited_poses_xyz,
        min_separation_m=float(config.min_station_separation_m),
    )
    bearing_gains = _bearing_diversity_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        modes_by_isotope,
    )
    frontier_target = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
    )
    frontier_gains = _frontier_band_gains_batch(
        candidate_poses,
        visited_poses_xyz,
        target_radius_m=frontier_target,
    )
    turn_penalties = _route_turn_penalties_batch(
        candidate_poses,
        current_pose_xyz,
        visited_poses_xyz,
    )
    local_orbit_gains = _local_orbit_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    elevation_condition_gains = _elevation_condition_gains_batch(
        candidate_poses,
        modes_by_isotope,
        config=config,
    )
    if config.forced_program_pair_ids is None:
        return _build_conditional_nodes(
            estimator=estimator,
            candidate_poses=candidate_poses,
            path_lengths=path_lengths,
            coverage_norm=coverage_norm,
            coverage_raw=coverage_raw,
            revisit_penalties=revisit_penalties,
            bearing_gains=bearing_gains,
            frontier_gains=frontier_gains,
            turn_penalties=turn_penalties,
            local_orbit_gains=local_orbit_gains,
            elevation_condition_gains=elevation_condition_gains,
            coverage_floor=float(coverage_floor),
            coverage_support=str(coverage_support),
            coverage_quadrature_diagnostics=coverage_quadrature_diagnostics,
            config=config,
            rng=rng,
            joint_particles=joint_particles,
            motion_times=motion_times,
            motion_time_components=motion_time_components,
            path_length_support=str(path_length_support),
        )
    evaluation_pose_indices = np.arange(candidate_poses.shape[0], dtype=np.int64)
    raw_nodes: list[DSSPPNode] = []
    pending: list[_PendingDSSPPNode] = []
    eval_indices = [int(idx) for idx in evaluation_pose_indices]
    pose_eval_context: dict[str, object] = {
        "candidate_poses": candidate_poses,
        "path_lengths": path_lengths,
        "programs": programs,
        "config": config,
        "coverage_norm": coverage_norm,
        "coverage_raw": coverage_raw,
        "revisit_penalties": revisit_penalties,
        "bearing_gains": bearing_gains,
        "frontier_gains": frontier_gains,
        "turn_penalties": turn_penalties,
        "local_orbit_gains": local_orbit_gains,
        "elevation_condition_gains": elevation_condition_gains,
        "coverage_floor": float(coverage_floor),
        "coverage_support": coverage_support,
        "coverage_quadrature": coverage_quadrature_diagnostics,
    }
    pose_results = _materialize_pose_nodes(
        eval_indices,
        context=pose_eval_context,
    )
    for (
        _pose_index,
        _local_cheap_score,
        local_pending,
    ) in pose_results:
        if local_pending:
            pending.extend(local_pending)
    if not pending:
        return [], {
            "total_action_count": 0,
            "proxy_action_count": 0,
            "exact_action_count": 0,
            "path_length_support": path_length_support,
            "coverage_support": coverage_support,
            "coverage_quadrature": coverage_quadrature_diagnostics,
        }
    total_action_count = len(pending)
    available_pose_count = len({int(item.pose_index) for item in pending})
    proxy_wall_s = 0.0
    exact_wall_s = 0.0
    proxy_information_scores = np.zeros(
        (candidate_poses.shape[0], len(programs)),
        dtype=np.float64,
    )
    shortlist_indices = np.arange(total_action_count, dtype=np.int64)
    proxy_ranking_scores = np.asarray(
        [float(item.static_score) for item in pending],
        dtype=np.float64,
    )
    shortlist_category_counts = {
        "global": int(total_action_count),
        "coverage": 0,
        "program_diversity": 0,
        "global_pose_count": int(available_pose_count),
        "coverage_pose_count": 0,
        "shortlisted_pose_count": int(available_pose_count),
    }
    proxy_action_count = 0
    proxy_particle_count = 0
    proxy_eig_runtime_diagnostics: dict[str, object] = {}
    exact_eig_runtime_rounds: list[dict[str, object]] = []
    exact_eig_seed = int(
        rng.integers(
            0,
            np.iinfo(np.int64).max,
            endpoint=False,
            dtype=np.int64,
        )
    )
    if float(config.lambda_eig) > 0.0 and available_pose_count > int(
        config.exact_eig_pose_max
    ):
        proxy_joint_particles = estimator.planning_joint_particles(
            max_particles=int(config.proxy_planning_particles),
            method="top_weight",
        )
        proxy_particle_count = int(np.asarray(proxy_joint_particles.weights_n).size)
        print(
            "[dss] proxy-start "
            f"poses={candidate_poses.shape[0]} "
            f"programs={len(programs)} "
            f"actions={total_action_count} "
            f"particles={proxy_particle_count} "
            f"samples={config.proxy_eig_samples}",
            flush=True,
        )
        proxy_started = time.perf_counter()
        proxy_information_scores = _program_information_proxy_for_poses(
            estimator,
            candidate_poses,
            programs,
            config=config,
            joint_particles=proxy_joint_particles,
            rng=rng,
            eig_call_seed=exact_eig_seed,
            diagnostics=proxy_eig_runtime_diagnostics,
        )
        proxy_wall_s = float(time.perf_counter() - proxy_started)
        print(
            "[dss] proxy-done "
            f"elapsed_s={proxy_wall_s:.3f} "
            f"actions={total_action_count}",
            flush=True,
        )
        proxy_action_count = int(total_action_count)
        (
            shortlist_indices,
            proxy_ranking_scores,
            shortlist_category_counts,
        ) = _exact_eig_shortlist(
            pending,
            programs,
            proxy_information_scores,
            config=config,
        )
    initial_indices = (
        np.asarray(shortlist_indices, dtype=np.int64)
        if float(config.lambda_eig) > 0.0
        else np.arange(total_action_count, dtype=np.int64)
    )
    proxy_order = _stable_descending_indices(proxy_ranking_scores)
    remaining_order = proxy_order[
        ~np.isin(proxy_order, initial_indices, assume_unique=False)
    ]
    evaluation_order = np.concatenate((initial_indices, remaining_order))
    if (
        np.unique(evaluation_order).size != total_action_count
        or np.any(evaluation_order < 0)
        or np.any(evaluation_order >= total_action_count)
    ):
        raise RuntimeError("Adaptive exact-EIG ordering lost a DSS action.")

    program_information_gains = np.full(
        len(pending),
        np.nan,
        dtype=np.float64,
    )
    evaluated_pending_indices = np.zeros(0, dtype=np.int64)

    def _evaluate_exact_indices(
        new_indices: NDArray[np.int64],
    ) -> None:
        """Evaluate one adaptive action batch under a fixed common RNG stream."""
        nonlocal exact_wall_s
        if new_indices.size == 0:
            return
        pending_indices_by_pose: dict[int, list[int]] = {}
        for pending_index_raw in new_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            pending_indices_by_pose.setdefault(
                int(item.pose_index),
                [],
            ).append(pending_index)

        eig_indices = sorted(pending_indices_by_pose)
        batched_programs = [
            [
                pending[index].program
                for index in pending_indices_by_pose.get(pose_index, [])
            ]
            for pose_index in eig_indices
        ]
        round_diagnostics: dict[str, object] = {}
        print(
            "[dss] exact-round-start "
            f"round={len(exact_eig_runtime_rounds) + 1} "
            f"actions={new_indices.size}",
            flush=True,
        )
        exact_started = time.perf_counter()
        batched_gains = _program_information_gains_for_poses(
            estimator,
            candidate_poses[eig_indices],
            batched_programs,
            config=config,
            rng=rng,
            joint_particles=joint_particles,
            diagnostics=round_diagnostics,
            eig_call_seed=exact_eig_seed,
        )
        round_wall_s = float(time.perf_counter() - exact_started)
        exact_wall_s += round_wall_s
        round_diagnostics["wall_s"] = round_wall_s
        round_diagnostics["action_count"] = int(new_indices.size)
        exact_eig_runtime_rounds.append(round_diagnostics)
        print(
            "[dss] exact-round-done "
            f"round={len(exact_eig_runtime_rounds)} "
            f"elapsed_s={round_wall_s:.3f} "
            f"actions={new_indices.size}",
            flush=True,
        )
        eig_results = [
            (
                pose_index,
                pending_indices_by_pose.get(pose_index, []),
                np.asarray(values, dtype=float),
            )
            for pose_index, values in zip(eig_indices, batched_gains)
        ]
        for _pose_index, pending_indices, values in eig_results:
            if values.size != len(pending_indices):
                raise RuntimeError(
                    "Program EIG result does not match the evaluated programs."
                )
            for pending_index, value in zip(
                pending_indices,
                values,
                strict=True,
            ):
                program_information_gains[pending_index] = float(value)

    normalized_joint_weights = _normalise_weights(
        np.asarray(joint_particles.weights_n, dtype=np.float64)
    )
    positive_joint_weights = normalized_joint_weights[normalized_joint_weights > 0.0]
    particle_entropy = float(
        -np.sum(positive_joint_weights * np.log(positive_joint_weights))
    )
    finite_sample_eig_upper = _finite_sample_information_gain_upper_bound(
        normalized_joint_weights
    )
    excluded_universal_upper = float("inf")
    evaluated_objective_lower = -float("inf")
    shortlist_bound_certified = False
    adaptive_round_count = 0
    next_evaluation_offset = 0
    while next_evaluation_offset < total_action_count:
        adaptive_round_count += 1
        next_stop = (
            initial_indices.size if next_evaluation_offset == 0 else total_action_count
        )
        if next_stop <= next_evaluation_offset:
            raise RuntimeError("Adaptive exact-EIG batch made no progress.")
        new_indices = evaluation_order[next_evaluation_offset:next_stop]
        if float(config.lambda_eig) > 0.0:
            _evaluate_exact_indices(new_indices)
        else:
            program_information_gains[new_indices] = 0.0
        next_evaluation_offset = int(next_stop)
        evaluated_pending_indices = evaluation_order[:next_evaluation_offset]

        info_gains.fill(0.0)
        for pending_index_raw in evaluated_pending_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            info_gains[int(item.pose_index)] = max(
                float(info_gains[int(item.pose_index)]),
                float(program_information_gains[pending_index]),
            )
        finite_path = np.isfinite(path_lengths)
        if config.lambda_distance is None:
            evaluated_pose_mask = np.zeros(
                candidate_poses.shape[0],
                dtype=bool,
            )
            evaluated_pose_mask[
                np.asarray(
                    sorted(
                        {
                            int(pending[int(index)].pose_index)
                            for index in evaluated_pending_indices
                        }
                    ),
                    dtype=np.int64,
                )
            ] = True
            lambda_distance = estimate_lambda_cost(
                info_gains[finite_path & evaluated_pose_mask],
                path_lengths[finite_path & evaluated_pose_mask],
                method="range",
            )
        else:
            lambda_distance = float(config.lambda_distance)

        raw_nodes = []
        for pending_index_raw in evaluated_pending_indices:
            pending_index = int(pending_index_raw)
            item = pending[pending_index]
            info_gain = float(program_information_gains[pending_index])
            base_score = float(item.static_score) + float(config.lambda_eig) * info_gain
            placeholder_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=float(base_score),
                distance_weight=float(lambda_distance),
                information_gain=float(info_gain),
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(item.elevation_condition_gain),
            )
            score, _ = _compose_transition_score(
                node=placeholder_node,
                previous_pose_xyz=current_pose_xyz,
                map_api=map_api,
                config=config,
                path_length_override_m=float(
                    path_lengths[int(placeholder_node.pose_index)]
                ),
                travel_time_override_s=(
                    None
                    if motion_times is None
                    else float(motion_times[int(placeholder_node.pose_index)])
                ),
                **_motion_component_overrides(int(placeholder_node.pose_index)),
            )
            raw_nodes.append(
                DSSPPNode(
                    **{
                        **placeholder_node.__dict__,
                        "score": float(score),
                    }
                )
            )
        raw_nodes.sort(key=lambda node: node.score, reverse=True)

        excluded_mask = np.ones(total_action_count, dtype=bool)
        excluded_mask[evaluated_pending_indices] = False
        if not np.any(excluded_mask):
            excluded_universal_upper = -float("inf")
            shortlist_bound_certified = True
            break
        if config.lambda_distance is None:
            # Auto-scaled distance changes with unseen EIG values, so no safe
            # finite lower/upper objective bracket exists for the excluded
            # actions.  The exact stage is nevertheless a predeclared compute
            # budget; the full-action proxy remains the ranking contract.
            break
        evaluated_lower_scores: list[float] = []
        for index_raw in evaluated_pending_indices:
            item = pending[int(index_raw)]
            lower_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=float(item.static_score),
                distance_weight=float(lambda_distance),
                information_gain=0.0,
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(item.elevation_condition_gain),
            )
            lower_score, _ = _compose_transition_score(
                node=lower_node,
                previous_pose_xyz=current_pose_xyz,
                map_api=map_api,
                config=config,
                path_length_override_m=float(path_lengths[int(lower_node.pose_index)]),
                travel_time_override_s=(
                    None
                    if motion_times is None
                    else float(motion_times[int(lower_node.pose_index)])
                ),
                **_motion_component_overrides(int(lower_node.pose_index)),
            )
            evaluated_lower_scores.append(float(lower_score))
        evaluated_objective_lower = float(max(evaluated_lower_scores))
        excluded_upper_scores: list[float] = []
        for index_raw in np.flatnonzero(excluded_mask):
            item = pending[int(index_raw)]
            upper_node = DSSPPNode(
                pose_index=int(item.pose_index),
                pose_xyz=item.pose_xyz,
                program=item.program,
                score=0.0,
                static_score=(
                    float(item.static_score)
                    + float(config.lambda_eig) * finite_sample_eig_upper
                ),
                distance_weight=float(lambda_distance),
                information_gain=float(finite_sample_eig_upper),
                coverage_gain=float(item.coverage_gain),
                revisit_penalty=float(item.revisit_penalty),
                bearing_diversity_gain=float(item.bearing_diversity_gain),
                frontier_gain=float(item.frontier_gain),
                turn_penalty=float(item.turn_penalty),
                local_orbit_gain=float(item.local_orbit_gain),
                elevation_condition_gain=float(item.elevation_condition_gain),
            )
            upper_score, _ = _compose_transition_score(
                node=upper_node,
                previous_pose_xyz=current_pose_xyz,
                map_api=map_api,
                config=config,
                path_length_override_m=float(path_lengths[int(upper_node.pose_index)]),
                travel_time_override_s=(
                    None
                    if motion_times is None
                    else float(motion_times[int(upper_node.pose_index)])
                ),
                **_motion_component_overrides(int(upper_node.pose_index)),
            )
            excluded_upper_scores.append(float(upper_score))
        excluded_universal_upper = float(max(excluded_upper_scores))
        shortlist_bound_certified = bool(
            evaluated_objective_lower >= excluded_universal_upper - 1.0e-12
        )
        # The proxy already evaluates every action with the same generative
        # model; the exact stage re-evaluates the fixed-program pose shortlist.
        break

    best_exact_score = float(raw_nodes[0].score) if raw_nodes else -float("inf")
    winner_exceeds_universal_excluded_bound = shortlist_bound_certified
    selected_pending_index = -1
    if raw_nodes:
        for index_raw in evaluated_pending_indices:
            index = int(index_raw)
            if (
                int(pending[index].pose_index) == int(raw_nodes[0].pose_index)
                and pending[index].program == raw_nodes[0].program
            ):
                selected_pending_index = index
                break
        if selected_pending_index < 0:
            raise RuntimeError("Selected exact-EIG node lost its pending identity.")
    proxy_order = _stable_descending_indices(proxy_ranking_scores)
    proxy_rank = (
        int(np.flatnonzero(proxy_order == selected_pending_index)[0] + 1)
        if selected_pending_index >= 0
        else 0
    )
    model = estimator.authenticated_full_spectrum_model()
    sample_count = int(estimator.pf_config.planning_eig_samples)
    particle_count = int(np.asarray(joint_particles.weights_n).size)
    view_count = max((len(program.pair_ids) for program in programs), default=0)
    energy_bin_count = int(np.asarray(model.energy_axis_keV).size)
    exact_action_count = (
        int(evaluated_pending_indices.size) if float(config.lambda_eig) > 0.0 else 0
    )
    evaluated_pose_array = np.asarray(
        [int(pending[int(index)].pose_index) for index in evaluated_pending_indices],
        dtype=np.int64,
    )
    evaluated_pose_indices, evaluated_program_counts = np.unique(
        evaluated_pose_array,
        return_counts=True,
    )
    full_program_sweep = bool(
        evaluated_program_counts.size > 0
        and np.all(evaluated_program_counts == len(programs))
    )
    diagnostics: dict[str, object] = {
        "candidate_pose_count": int(available_pose_count),
        "total_action_count": int(total_action_count),
        "path_length_support": path_length_support,
        "coverage_support": coverage_support,
        "coverage_quadrature": coverage_quadrature_diagnostics,
        "proxy_action_count": int(proxy_action_count),
        "proxy_subset_evaluation_count": int(proxy_action_count),
        "proxy_particle_count": int(proxy_particle_count),
        "proxy_eig_samples": int(config.proxy_eig_samples),
        "shared_full_spectrum_detector_aperture_samples": int(
            config.detector_aperture_samples
        ),
        "exact_action_count": int(exact_action_count),
        "exact_subset_evaluation_count": int(exact_action_count),
        "exact_eig_pose_max": int(config.exact_eig_pose_max),
        "shortlisted_pose_count": int(evaluated_pose_indices.size),
        "programs_per_shortlisted_pose": int(len(programs)),
        "full_program_sweep_per_shortlisted_pose": bool(full_program_sweep),
        "pose_shortlist_contract": (
            "proxy_ranks_poses_by_best_proxy_program_then_exact_evaluates_"
            "the_forced_baseline_program_at_each_shortlisted_pose"
        ),
        "exact_eig_seed": int(exact_eig_seed),
        "adaptive_exact_eig_round_count": int(adaptive_round_count),
        "adaptive_exact_eig_exhausted_all_actions": bool(
            exact_action_count == total_action_count
        ),
        "shortlist_category_counts": dict(shortlist_category_counts),
        "proxy_wall_s": float(proxy_wall_s),
        "exact_eig_wall_s": float(exact_wall_s),
        "proxy_eig_runtime": dict(proxy_eig_runtime_diagnostics),
        "exact_eig_runtime": {
            "rounds": list(exact_eig_runtime_rounds),
        },
        "proxy_unique_action_count": int(proxy_action_count),
        "exhaustive_exact_bin_state_operations": int(
            total_action_count
            * max(sample_count, 0)
            * particle_count
            * view_count
            * energy_bin_count
        ),
        "shortlisted_exact_bin_state_operations": int(
            exact_action_count
            * max(sample_count, 0)
            * particle_count
            * view_count
            * energy_bin_count
        ),
        "proxy_full_spectrum_bin_state_operations": int(
            proxy_action_count
            * int(config.proxy_eig_samples)
            * proxy_particle_count
            * view_count
            * energy_bin_count
            if proxy_action_count
            else 0
        ),
        "shortlist_mc_winner_exceeds_universal_excluded_bound": bool(
            winner_exceeds_universal_excluded_bound
        ),
        "shortlist_best_exact_score": float(best_exact_score),
        "shortlist_evaluated_objective_lower_bound": (
            None
            if not np.isfinite(evaluated_objective_lower)
            else float(evaluated_objective_lower)
        ),
        "shortlist_max_excluded_universal_objective_upper_bound": (
            None
            if not np.isfinite(excluded_universal_upper)
            else float(excluded_universal_upper)
        ),
        "shortlist_selected_proxy_rank": int(proxy_rank),
        "proxy_contract": (
            "reduced_particle_and_sample_joint_full_spectrum_generative_eig_"
            "with_identical_background_dead_time_marks_and_likelihood"
        ),
        "posterior_entropy_true_eig_upper_bound_nats": float(particle_entropy),
        "finite_sample_mc_eig_upper_bound_nats": float(finite_sample_eig_upper),
        "universal_eig_upper_bound_nats": float(finite_sample_eig_upper),
        "shortlist_formal_recall_certificate_available": bool(
            shortlist_bound_certified
        ),
        "shortlist_certification_note": (
            "Every pose is ranked by its best shared full-spectrum proxy "
            "program. The forced baseline program is then exactly evaluated "
            "at each shortlisted pose. A formal "
            "pose-recall certificate is reported only when the evaluated set "
            "also exceeds the safe finite-sample objective bound."
        ),
        "eig_shortlist_wall_s": float(proxy_wall_s + exact_wall_s),
    }
    return raw_nodes, diagnostics


def _planning_subset_from_full_snapshot(
    particles: JointPlanningParticles,
    *,
    max_particles: int,
    method: str,
    rng: np.random.Generator,
) -> JointPlanningParticles:
    """Select planning rows from one already packed full posterior snapshot."""
    weights = np.asarray(particles.weights_n, dtype=np.float64).reshape(-1)
    particle_count = int(weights.size)
    requested = int(max_particles)
    if (
        particle_count < 1
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Full planning snapshot contains invalid weights.")
    if not 1 <= requested <= particle_count:
        raise ValueError("Planning subset size lies outside the full snapshot.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Planning subset selection requires a NumPy generator.")
    if method == "top_weight":
        indices = np.argsort(weights)[::-1][:requested].astype(
            np.int64,
            copy=False,
        )
        selected_weights = weights[indices].copy()
        selected_weights /= float(np.sum(selected_weights))
    elif method == "resample":
        indices = np.asarray(
            rng.choice(
                particle_count,
                size=requested,
                replace=True,
                p=weights,
            ),
            dtype=np.int64,
        )
        selected_weights = np.full(
            requested,
            1.0 / float(requested),
            dtype=np.float64,
        )
    else:
        raise ValueError("Planning subset method must be 'top_weight' or 'resample'.")

    def _subset_mapping(
        values: Mapping[str, NDArray[Any]],
    ) -> dict[str, NDArray[Any]]:
        """Gather selected common rows from one isotope-array mapping."""
        if set(values) != set(particles.isotope_order):
            raise ValueError("Full planning snapshot isotope mappings are incomplete.")
        result: dict[str, NDArray[Any]] = {}
        for isotope in particles.isotope_order:
            array = np.asarray(values[isotope])
            if array.shape[:1] != (particle_count,):
                raise ValueError("Full planning snapshot arrays are row-misaligned.")
            result[isotope] = np.ascontiguousarray(array[indices])
        return result

    original_indices = np.asarray(
        particles.original_particle_indices,
        dtype=np.int64,
    ).reshape(-1)
    if (
        original_indices.shape != (particle_count,)
        or np.any(original_indices < 0)
        or np.unique(original_indices).size != particle_count
    ):
        raise ValueError("Full planning snapshot original row indices are invalid.")
    return JointPlanningParticles(
        isotope_order=tuple(str(value) for value in particles.isotope_order),
        weights_n=np.ascontiguousarray(selected_weights),
        positions_nk3_by_isotope=_subset_mapping(
            particles.positions_nk3_by_isotope
        ),
        surface_chart_ids_nk_by_isotope=_subset_mapping(
            particles.surface_chart_ids_nk_by_isotope
        ),
        surface_uv_nk2_by_isotope=_subset_mapping(
            particles.surface_uv_nk2_by_isotope
        ),
        strengths_nk_by_isotope=_subset_mapping(
            particles.strengths_nk_by_isotope
        ),
        source_mask_nk_by_isotope=_subset_mapping(
            particles.source_mask_nk_by_isotope
        ),
        original_particle_indices=np.ascontiguousarray(original_indices[indices]),
    )


def _select_dss_pp_core(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    map_api: object | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    config: DSSPPConfig | None = None,
    rng: np.random.Generator | None = None,
    candidate_motion_times_s: NDArray[np.float64] | None = None,
    candidate_horizontal_travel_times_s: NDArray[np.float64] | None = None,
    candidate_mast_vertical_times_s: NDArray[np.float64] | None = None,
    candidate_settling_times_s: NDArray[np.float64] | None = None,
) -> DSSPPResult:
    """Run the shared DSS-PP implementation over validated or oracle inputs."""
    planning_started = time.perf_counter()
    cfg = config or DSSPPConfig()
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "select_dss_pp_next_station requires a persistent explicit rng; "
            "reinitializing a fixed seed per planning call is forbidden."
        )
    planning_rng = rng
    pf_max_sources = _validate_mode_capacity(estimator, cfg)
    _validate_eig_likelihood_contract(estimator, cfg)
    current_pose = np.asarray(current_pose_xyz, dtype=float)
    if current_pose.shape != (3,) or np.any(~np.isfinite(current_pose)):
        raise ValueError("current_pose_xyz must be a finite shape-(3,) vector.")
    geometry_snapshot_started = time.perf_counter()
    geometry_joint_particles = estimator.planning_joint_particles()
    geometry_snapshot_wall_s = time.perf_counter() - geometry_snapshot_started
    particle_snapshot_started = time.perf_counter()
    if cfg.planning_particles is None:
        joint_particles = geometry_joint_particles
    else:
        joint_particles = _planning_subset_from_full_snapshot(
            geometry_joint_particles,
            max_particles=cfg.planning_particles,
            method=cfg.planning_method,
            rng=planning_rng,
        )
    particle_snapshot_wall_s = time.perf_counter() - particle_snapshot_started
    signature_mode_started = time.perf_counter()
    modes = extract_signature_modes(
        estimator,
        mode_cluster_radius_m=float(cfg.mode_cluster_radius_m),
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
        rng=planning_rng,
        joint_particles=geometry_joint_particles,
    )
    signature_mode_wall_s = time.perf_counter() - signature_mode_started
    official_mode_started = time.perf_counter()
    _official_modes, official_snapshot_diagnostics = _official_signature_modes(
        estimator,
        max_modes_per_isotope=int(cfg.max_modes_per_isotope),
    )
    official_mode_wall_s = time.perf_counter() - official_mode_started
    input_candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if (
        input_candidates.ndim != 2
        or input_candidates.shape[1:] != (3,)
        or np.any(~np.isfinite(input_candidates))
    ):
        raise ValueError(
            "candidate_poses_xyz must be a finite array with shape (N, 3)."
        )
    candidates = input_candidates
    input_motion_times = None
    if candidate_motion_times_s is not None:
        input_motion_times = np.asarray(
            candidate_motion_times_s,
            dtype=np.float64,
        ).reshape(-1)
        if (
            input_motion_times.shape != (input_candidates.shape[0],)
            or np.any(~np.isfinite(input_motion_times))
            or np.any(input_motion_times < 0.0)
        ):
            raise ValueError(
                "candidate_motion_times_s must align with candidates and "
                "contain finite nonnegative values."
            )
    raw_motion_components = (
        candidate_horizontal_travel_times_s,
        candidate_mast_vertical_times_s,
        candidate_settling_times_s,
    )
    input_motion_components = None
    if any(values is not None for values in raw_motion_components):
        if not all(values is not None for values in raw_motion_components):
            raise ValueError(
                "Candidate horizontal, mast, and settling times must be supplied "
                "together."
            )
        input_motion_components = tuple(
            np.asarray(values, dtype=np.float64).reshape(-1)
            for values in raw_motion_components
        )
        if any(
            values.shape != (input_candidates.shape[0],)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            for values in input_motion_components
        ):
            raise ValueError(
                "Candidate motion-time components must align with candidates and "
                "contain finite nonnegative values."
            )
        component_totals = np.sum(np.vstack(input_motion_components), axis=0)
        if input_motion_times is None:
            raise ValueError(
                "Candidate motion-time components require exact total motion times."
            )
        if not np.array_equal(
            input_motion_times,
            component_totals,
        ):
            raise ValueError(
                "Candidate motion-time components must sum to motion times."
            )
    if (input_motion_times is None) is not (input_motion_components is None):
        raise ValueError(
            "Candidate motion totals and all three components must be supplied "
            "together."
        )

    candidates, separation_filtered = _filter_station_separation(
        candidates,
        visited_poses_xyz,
        min_separation_m=float(cfg.min_station_separation_m),
    )
    if input_motion_times is None:
        candidates, path_filtered = _filter_path_reachable_stations(
            candidates,
            current_pose_xyz=current_pose,
            map_api=map_api,
        )
    else:
        # Runtime-authored candidate snapshots are already reachability
        # filtered and carry one exact time-valued motion cost per pose.
        path_filtered = 0
    motion_times = (
        None
        if input_motion_times is None
        else _align_candidate_values(
            input_candidates,
            input_motion_times,
            candidates,
        )
    )
    motion_time_components = (
        None
        if input_motion_components is None
        else tuple(
            _align_candidate_values(
                input_candidates,
                values,
                candidates,
            )
            for values in input_motion_components
        )
    )
    if candidates.size == 0:
        raise ValueError(
            "DSS-PP received no reachable candidate after the generic 3-D "
            "station-separation contract."
        )
    if cfg.forced_program_pair_ids is not None:
        pair_count = int(estimator.num_orientations) ** 2
        if any(int(pair_id) >= pair_count for pair_id in cfg.forced_program_pair_ids):
            raise ValueError(
                "forced_program_pair_ids exceed the estimator shield-pair "
                f"support [0, {pair_count - 1}]."
            )
        programs = [
            ShieldProgram(
                name="forced_baseline_shield_program",
                pair_ids=tuple(int(pair_id) for pair_id in cfg.forced_program_pair_ids),
                kind="forced_baseline",
            )
        ]
    else:
        programs = []
    candidate_pair_ids = [
        int(pair_id) for program in programs for pair_id in program.pair_ids
    ]
    pair_occurrences = np.bincount(
        np.asarray(candidate_pair_ids, dtype=np.int64),
        minlength=int(estimator.num_orientations) ** 2,
    )
    positive_occurrences = pair_occurrences[pair_occurrences > 0]
    companion_sets = {pair_id: set() for pair_id in np.flatnonzero(pair_occurrences)}
    for program in programs:
        program_pairs = set(int(pair_id) for pair_id in program.pair_ids)
        for pair_id in program_pairs:
            companion_sets[pair_id].update(program_pairs - {pair_id})
    node_build_started = time.perf_counter()
    nodes, shortlist_diagnostics = _build_nodes(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        programs=programs,
        modes_by_isotope=modes,
        current_pose_xyz=current_pose,
        visited_poses_xyz=visited_poses_xyz,
        map_api=map_api,
        bounds_xyz=bounds_xyz,
        config=cfg,
        rng=planning_rng,
        joint_particles=joint_particles,
        candidate_motion_times_s=motion_times,
        candidate_motion_time_components_s=motion_time_components,
    )
    node_build_wall_s = time.perf_counter() - node_build_started
    if not nodes:
        raise ValueError("DSS-PP could not evaluate any station-program node.")
    nodes_by_pose: dict[int, list[DSSPPNode]] = {}
    for node in nodes:
        nodes_by_pose.setdefault(int(node.pose_index), []).append(node)
    pose_program_leaders = [
        max(
            pose_nodes,
            key=lambda node: (
                float(node.information_gain),
                float(node.score),
                str(node.program.name),
            ),
        )
        for _, pose_nodes in sorted(nodes_by_pose.items())
    ]
    first = max(
        pose_program_leaders,
        key=lambda node: (float(node.score), -int(node.pose_index)),
    )
    selected_pose_nodes = nodes_by_pose[int(first.pose_index)]
    selected_pose_eig_leader = max(
        float(node.information_gain) for node in selected_pose_nodes
    )
    selected_program_is_eig_leader = bool(
        np.isclose(
            float(first.information_gain),
            selected_pose_eig_leader,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )
    conditional_standard = cfg.forced_program_pair_ids is None
    if float(cfg.lambda_eig) > 0.0:
        if conditional_standard and len(selected_pose_nodes) != 1:
            raise RuntimeError(
                "Conditional DSS must retain exactly one EIG-maximized program "
                "per exact pose."
            )
        if not conditional_standard and len(selected_pose_nodes) != len(programs):
            raise RuntimeError(
                "The selected DSS pose was not exactly evaluated with every "
                "predeclared shield program."
            )
        if not selected_program_is_eig_leader:
            raise RuntimeError(
                "DSS shield selection must maximize exact EIG at the selected pose."
            )
    sequence = (first,)
    best_score = float(first.score)
    mode_count = sum(len(mode_list) for mode_list in modes.values())
    ranked_limit = int(cfg.diagnostic_ranked_node_limit)
    ranked_nodes = (
        sorted(nodes, key=lambda node: float(node.score), reverse=True)[:ranked_limit]
        if ranked_limit > 0
        else []
    )
    diagnostics: dict[str, Any] = {
        "candidate_count": int(candidates.shape[0]),
        "separation_filtered_candidates": int(separation_filtered),
        "path_filtered_candidates": int(path_filtered),
        "runtime_motion_times_applied": bool(motion_times is not None),
        "runtime_motion_time_components_applied": bool(
            motion_time_components is not None
        ),
        "motion_time_weights": {
            "horizontal": float(cfg.lambda_horizontal_time),
            "mast_vertical": float(cfg.lambda_mast_vertical_time),
            "settling": float(cfg.lambda_settling_time),
        },
        "program_count": int(
            int(estimator.num_orientations) ** 2
            if conditional_standard
            else len(programs)
        ),
        "forced_baseline_program_count": int(len(programs)),
        "shield_pair_count": int(estimator.num_orientations) ** 2,
        "shield_program_search_policy": (
            "conditional_greedy_all_pairs"
            if conditional_standard
            else "forced_baseline"
        ),
        "program_library_realized_count": int(len(programs)),
        "program_library_policy": (
            "forced_baseline"
            if cfg.forced_program_pair_ids is not None
            else "conditional_greedy_all_pairs"
        ),
        "program_library_global_optimality_claimed": False,
        "program_library_exact_eig_over_every_predeclared_action": bool(
            int(shortlist_diagnostics.get("exact_action_count", 0))
            == int(shortlist_diagnostics.get("total_action_count", 0))
        ),
        "program_library_exact_eig_over_every_program_at_shortlisted_pose": bool(
            shortlist_diagnostics.get(
                "full_program_sweep_per_shortlisted_pose",
                False,
            )
        ),
        "program_library_unique_pair_count": int(np.count_nonzero(pair_occurrences)),
        "program_library_pair_occurrence_min": int(
            1
            if conditional_standard
            else np.min(positive_occurrences)
            if positive_occurrences.size
            else 0
        ),
        "program_library_pair_occurrence_max": int(
            1
            if conditional_standard
            else np.max(positive_occurrences)
            if positive_occurrences.size
            else 0
        ),
        "program_library_companion_diversity_min": int(
            int(estimator.num_orientations) ** 2 - 1
            if conditional_standard
            else min(
                (len(companions) for companions in companion_sets.values()),
                default=0,
            )
        ),
        "program_library_companion_diversity_max": int(
            int(estimator.num_orientations) ** 2 - 1
            if conditional_standard
            else max(
                (len(companions) for companions in companion_sets.values()),
                default=0,
            )
        ),
        "evaluated_candidate_count": int(len({int(node.pose_index) for node in nodes})),
        "node_count": int(len(nodes)),
        "mode_count": int(mode_count),
        "planning_particle_count": int(np.asarray(joint_particles.weights_n).size),
        "max_modes_per_isotope": int(cfg.max_modes_per_isotope),
        "pf_max_sources": int(pf_max_sources),
        "planner_belief_sources": ["pf_posterior"],
        "planner_official_posterior_projection": dict(official_snapshot_diagnostics),
        "planner_geometry_mode_projection": {
            "source": "full_aligned_joint_posterior",
            "particle_count": int(np.asarray(geometry_joint_particles.weights_n).size),
            "mass_semantics": ("unconditional_particle_mass_with_k_zero_preserved"),
            "position_representative": "intrinsic_surface_weighted_medoid",
            "synthetic_xyz_centroids": False,
        },
        "planning_policy": "one_step_joint_eig",
        "pose_selection_objective": (
            "best_exact_program_eig_plus_spatial_coverage_and_robot_motion_terms"
        ),
        "program_selection_objective": (
            "maximum_same_sample_eig_among_conditional_greedy_and_one_swap"
            if conditional_standard
            else "forced_baseline_program_exact_eig"
        ),
        "shield_rotation_cost_applied": False,
        "first_program_kind": first.program.kind,
        "planning_eig_joint_program_views": True,
        "planning_eig_joint_isotope_vector": True,
        "planning_eig_aligned_joint_posterior_snapshot": True,
        "planning_eig_raw_spectrum_observations": True,
        "planning_eig_persistent_named_rng": True,
        "planning_eig_all_valid_candidates_exact": bool(
            int(shortlist_diagnostics.get("exact_action_count", 0))
            == int(shortlist_diagnostics.get("total_action_count", 0))
        ),
        "planning_eig_batched_source_line_response": True,
        "planning_eig_action_memory_budget_bytes": int(
            cfg.exact_eig_memory_budget_bytes
        ),
        "planning_eig_likelihood_model": "joint_full_spectrum_generative",
        "planning_eig_contract_hash_sha256": str(
            estimator.full_spectrum_generative_model.contract_hash_sha256
        ),
        "planning_eig_observation_semantics": (
            "same_full_spectrum_predictive_sampler_and_log_likelihood_as_pf"
        ),
        "planning_eig_shortlist": dict(shortlist_diagnostics),
        "planning_stage_wall_s": {
            "planning_particle_snapshot": float(particle_snapshot_wall_s),
            "geometry_particle_snapshot": float(geometry_snapshot_wall_s),
            "signature_mode_extraction": float(signature_mode_wall_s),
            "official_mode_projection": float(official_mode_wall_s),
            "node_build_and_eig": float(node_build_wall_s),
            "total_before_result": float(time.perf_counter() - planning_started),
        },
        "first_information_gain": float(first.information_gain),
        "selected_pose_exact_program_count": int(len(selected_pose_nodes)),
        "selected_pose_exact_information_gain_leader": float(selected_pose_eig_leader),
        "selected_program_is_exact_eig_leader_at_selected_pose": bool(
            selected_program_is_eig_leader
        ),
        "first_coverage_gain": float(first.coverage_gain),
        "coverage_support": str(
            shortlist_diagnostics.get(
                "coverage_support",
                "unavailable",
            )
        ),
        "coverage_quadrature": shortlist_diagnostics.get("coverage_quadrature"),
        "coverage_sample_count": int(
            (shortlist_diagnostics.get("coverage_quadrature") or {}).get(
                "sample_count",
                cfg.coverage_surface_quadrature_max_points or 0,
            )
        ),
        "first_revisit_penalty": float(first.revisit_penalty),
        "first_bearing_diversity_gain": float(first.bearing_diversity_gain),
        "first_frontier_gain": float(first.frontier_gain),
        "first_turn_penalty": float(first.turn_penalty),
        "first_local_orbit_gain": float(first.local_orbit_gain),
        "first_elevation_condition_gain": float(first.elevation_condition_gain),
        "diagnostic_ranked_node_limit": int(ranked_limit),
        "component_leaders": _component_leader_payloads(nodes),
        "ranked_nodes": [
            _node_diagnostic_payload(node, rank)
            for rank, node in enumerate(ranked_nodes, start=1)
        ],
    }
    return DSSPPResult(
        next_pose=first.pose_xyz.copy(),
        next_pose_index=int(first.pose_index),
        shield_program=first.program,
        score=best_score,
        sequence=tuple(sequence),
        diagnostics=diagnostics,
    )


def _select_dss_pp_test_oracle(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    map_api: object | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    config: DSSPPConfig | None = None,
    rng: np.random.Generator | None = None,
    candidate_motion_times_s: NDArray[np.float64] | None = None,
    candidate_horizontal_travel_times_s: NDArray[np.float64] | None = None,
    candidate_mast_vertical_times_s: NDArray[np.float64] | None = None,
    candidate_settling_times_s: NDArray[np.float64] | None = None,
) -> DSSPPResult:
    """Expose optional deterministic inputs only to focused unit tests."""
    return _select_dss_pp_core(
        estimator,
        candidate_poses_xyz,
        current_pose_xyz,
        visited_poses_xyz=visited_poses_xyz,
        map_api=map_api,
        bounds_xyz=bounds_xyz,
        config=config,
        rng=rng,
        candidate_motion_times_s=candidate_motion_times_s,
        candidate_horizontal_travel_times_s=(candidate_horizontal_travel_times_s),
        candidate_mast_vertical_times_s=candidate_mast_vertical_times_s,
        candidate_settling_times_s=candidate_settling_times_s,
    )


def select_dss_pp_next_station(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    visited_poses_xyz: NDArray[np.float64],
    map_api: object | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    config: DSSPPConfig,
    rng: np.random.Generator,
    candidate_motion_times_s: NDArray[np.float64],
    candidate_horizontal_travel_times_s: NDArray[np.float64],
    candidate_mast_vertical_times_s: NDArray[np.float64],
    candidate_settling_times_s: NDArray[np.float64],
) -> DSSPPResult:
    """Select one live action from a complete runtime-authored contract."""
    if not isinstance(config, DSSPPConfig):
        raise TypeError("Live DSS-PP requires a validated DSSPPConfig.")
    return _select_dss_pp_core(
        estimator,
        candidate_poses_xyz,
        current_pose_xyz,
        visited_poses_xyz=visited_poses_xyz,
        map_api=map_api,
        bounds_xyz=bounds_xyz,
        config=config,
        rng=rng,
        candidate_motion_times_s=candidate_motion_times_s,
        candidate_horizontal_travel_times_s=(candidate_horizontal_travel_times_s),
        candidate_mast_vertical_times_s=candidate_mast_vertical_times_s,
        candidate_settling_times_s=candidate_settling_times_s,
    )
