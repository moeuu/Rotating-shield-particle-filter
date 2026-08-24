"""Durable, truth-free audit records for PF action selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np
from runtime.artifacts import DurableJSONLWriter

from planning.dss_pp import DSSPPResult
from planning.program_types import ShieldProgram


SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES = (
    "particle_diversity_evidence_available_and_warning_absent",
    "smc_soft_budget_respected",
    "rejuvenation_mixing_complete",
    "structural_mixing_complete",
    "posterior_predictive_innovation_available_and_passed",
    "cardinality_not_at_upper_boundary_for_every_isotope",
    "no_newly_activated_isotope",
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Return one mapping or fail on a malformed planner diagnostic."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _leader(
    leaders: Mapping[str, object],
    name: str,
) -> dict[str, object] | None:
    """Return one JSON-safe planner component leader when available."""
    value = leaders.get(name)
    if value is None:
        return None
    return dict(_mapping(value, name=f"component_leaders.{name}"))


def _shadow_action(
    raw: Mapping[str, object],
    *,
    score_field: str,
    score_semantics: str,
    selection_reason: str,
    fallback_applied: bool = False,
    fallback_reasons: Sequence[str] = (),
    program_name: str | None = None,
    program_kind: str | None = "nested_conditional_greedy_prefix",
    require_unique_pairs: bool = True,
) -> dict[str, object]:
    """Normalize every persisted shadow action to one stable schema."""
    pose_xyz = raw.get("pose_xyz")
    pair_ids = raw.get("pair_ids")
    if not isinstance(pose_xyz, Sequence) or isinstance(pose_xyz, (str, bytes)):
        raise TypeError("Shadow action pose_xyz must be a sequence.")
    if not isinstance(pair_ids, Sequence) or isinstance(pair_ids, (str, bytes)):
        raise TypeError("Shadow action pair_ids must be a sequence.")
    selected_view_count = int(raw["selected_view_count"])
    pose_index = int(raw["pose_index"])
    resolved_pose = [float(value) for value in pose_xyz]
    resolved_pairs = [int(value) for value in pair_ids]
    if (
        pose_index < 0
        or len(resolved_pose) != 3
        or any(not math.isfinite(value) for value in resolved_pose)
    ):
        raise ValueError(
            "Shadow action pose must be a finite nonnegative-index 3-D pose."
        )
    if (
        selected_view_count < 1
        or len(resolved_pairs) != selected_view_count
        or any(value < 0 or value >= 64 for value in resolved_pairs)
        or (require_unique_pairs and len(set(resolved_pairs)) != len(resolved_pairs))
    ):
        raise ValueError("Shadow action view count and pair IDs disagree.")
    score = raw.get(score_field)
    information_gain = raw.get("information_gain_mean_nat")
    if score is not None and not math.isfinite(float(score)):
        raise ValueError("Shadow action score must be finite or None.")
    if information_gain is not None and (
        not math.isfinite(float(information_gain)) or float(information_gain) < 0.0
    ):
        raise ValueError("Shadow action information gain must be nonnegative finite.")
    return {
        "pose_index": pose_index,
        "pose_xyz": resolved_pose,
        "selected_view_count": selected_view_count,
        "pair_ids": resolved_pairs,
        "program_name": program_name,
        "program_kind": program_kind,
        "information_gain_mean_nat": (
            None if information_gain is None else float(information_gain)
        ),
        "selection_score": None if score is None else float(score),
        "score_semantics": str(score_semantics),
        "selection_reason": str(selection_reason),
        "fallback_applied": bool(fallback_applied),
        "fallback_reasons": [str(value) for value in fallback_reasons],
        "configured_measurement_time_weight": (
            None
            if raw.get("configured_measurement_time_weight") is None
            else float(raw["configured_measurement_time_weight"])
        ),
        "calibrated_for_dynamic_acquisition": (
            None
            if raw.get("calibrated_for_dynamic_acquisition") is None
            else bool(raw["calibrated_for_dynamic_acquisition"])
        ),
    }


def _executed_shadow_action(
    *,
    pose_index: int,
    pose_xyz: Sequence[float],
    program: ShieldProgram,
    information_gain: float | None,
    score: float | None,
) -> dict[str, object]:
    """Return one normalized actually executed fixed-view action."""
    return _shadow_action(
        {
            "pose_index": int(pose_index),
            "pose_xyz": list(pose_xyz),
            "selected_view_count": int(len(program.pair_ids)),
            "pair_ids": list(program.pair_ids),
            "information_gain_mean_nat": information_gain,
            "pose_score": score,
        },
        score_field="pose_score",
        score_semantics="executed_planner_score",
        selection_reason="runtime_contract_fixed_view_count",
        program_name=str(program.name),
        program_kind=str(program.kind),
        require_unique_pairs=False,
    )


def _unavailable_shadow_health(
    belief_after_station_id: int | None,
    *,
    reason: str,
) -> dict[str, object]:
    """Return one truth-free fail-closed health payload."""
    return {
        "policy_schema_version": 1,
        "hard_gate_contract": list(SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES),
        "available": False,
        "passed": False,
        "source_station_id": belief_after_station_id,
        "hard_failure_reasons": [str(reason)],
        "truth_used": False,
    }


def _validated_shadow_health(
    value: Mapping[str, object],
    *,
    belief_after_station_id: int | None,
) -> dict[str, object]:
    """Validate complete truth-free health before it can permit shortening."""
    health = dict(value)
    if health.get("source_station_id") != belief_after_station_id:
        raise ValueError("Posterior health and planner belief stations differ.")
    if health.get("truth_used") is not False:
        raise ValueError("Posterior health must explicitly be truth-free.")
    if health.get("policy_schema_version") != 1:
        raise ValueError("Posterior health policy schema must be version 1.")
    gates = health.get("hard_gate_contract")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        raise TypeError("Posterior health hard-gate contract must be a sequence.")
    if tuple(str(value) for value in gates) != SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES:
        raise ValueError("Posterior health hard-gate contract is incomplete.")
    available = health.get("available")
    passed = health.get("passed")
    if not isinstance(available, bool) or not isinstance(passed, bool):
        raise TypeError("Posterior health available/passed flags must be booleans.")
    reasons = health.get("hard_failure_reasons")
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        raise TypeError("Posterior health failure reasons must be a sequence.")
    resolved_reasons = [str(reason) for reason in reasons]
    if passed and (not available or resolved_reasons):
        raise ValueError("Passing posterior health must be available and reason-free.")
    if not passed and not resolved_reasons:
        raise ValueError("Failing posterior health must state at least one reason.")
    health["hard_failure_reasons"] = resolved_reasons
    return health


def _validated_pose_rows(
    pose_indices: object,
    pose_xyz: object,
    *,
    name: str,
) -> list[int]:
    """Validate aligned finite 3-D pose rows at the durable audit boundary."""
    if not isinstance(pose_indices, Sequence) or isinstance(
        pose_indices,
        (str, bytes),
    ):
        raise TypeError(f"{name} pose indices must be a sequence.")
    if not isinstance(pose_xyz, Sequence) or isinstance(pose_xyz, (str, bytes)):
        raise TypeError(f"{name} pose coordinates must be a sequence.")
    raw_indices = np.asarray(pose_indices)
    if (
        raw_indices.ndim != 1
        or np.issubdtype(raw_indices.dtype, np.bool_)
        or not np.issubdtype(raw_indices.dtype, np.integer)
    ):
        raise ValueError(f"{name} pose indices must be an integer vector.")
    indices = np.asarray(raw_indices, dtype=np.int64)
    poses = np.asarray(pose_xyz, dtype=np.float64)
    if (
        poses.shape != (indices.size, 3)
        or np.any(~np.isfinite(poses))
        or np.any(indices < 0)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(f"{name} pose rows must be aligned finite 3-D values.")
    return [int(value) for value in indices]


def _validate_view_count_pair_rows(
    by_view_count: Mapping[str, object],
    *,
    candidate_view_counts: tuple[int, ...],
    pose_count: int,
    name: str,
) -> None:
    """Validate every persisted shield-pair row for all view counts."""
    for view_count in candidate_view_counts:
        count_payload = _mapping(
            by_view_count.get(str(view_count)),
            name=f"{name}.by_view_count[{view_count}]",
        )
        raw_pairs = np.asarray(count_payload.get("pair_ids"))
        if (
            raw_pairs.shape != (pose_count, view_count)
            or np.issubdtype(raw_pairs.dtype, np.bool_)
            or not np.issubdtype(raw_pairs.dtype, np.integer)
        ):
            raise ValueError(f"{name} shield pair rows must align with poses.")
        pairs = np.asarray(raw_pairs, dtype=np.int64)
        duplicated = np.any(
            np.diff(np.sort(pairs, axis=1), axis=1) == 0,
            axis=1,
        )
        if np.any(pairs < 0) or np.any(pairs >= 64) or np.any(duplicated):
            raise ValueError(f"{name} shield pair row violates the 64-pair domain.")


def build_planner_audit(
    *,
    station_id: int,
    result: DSSPPResult,
    top_k: int = 10,
    mc_rank_stability: Mapping[str, object] | None = None,
    belief_after_station_id: int | None = None,
    posterior_health: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one compact audit of the action domain and selected EIG."""
    if isinstance(station_id, bool) or not isinstance(station_id, int):
        raise TypeError("station_id must be an integer.")
    if station_id < 0:
        raise ValueError("station_id must be nonnegative.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer.")
    if belief_after_station_id is not None and (
        isinstance(belief_after_station_id, bool)
        or not isinstance(belief_after_station_id, int)
        or belief_after_station_id < 0
    ):
        raise ValueError("belief_after_station_id must be nonnegative or None.")
    expected_belief_station = None if station_id == 0 else int(station_id - 1)
    if belief_after_station_id != expected_belief_station:
        raise ValueError(
            "A planned station must be audited against the immediately "
            "preceding PF belief station."
        )
    validated_health = (
        None
        if posterior_health is None
        else _validated_shadow_health(
            posterior_health,
            belief_after_station_id=belief_after_station_id,
        )
    )
    diagnostics = _mapping(result.diagnostics, name="planner diagnostics")
    shortlist = _mapping(
        diagnostics.get("planning_eig_shortlist", {}),
        name="planning_eig_shortlist",
    )
    leaders = _mapping(
        diagnostics.get("component_leaders", {}),
        name="component_leaders",
    )
    ranked = diagnostics.get("ranked_nodes", [])
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
        raise TypeError("ranked_nodes must be a sequence.")
    selected_eig = (
        None if not result.sequence else float(result.sequence[0].information_gain)
    )
    information_leader = _leader(leaders, "information_gain")
    best_exact_eig = (
        selected_eig
        if information_leader is None
        else float(information_leader["information_gain"])
    )
    stability = (
        {
            "status": "not_evaluated_in_control_loop",
            "reason": (
                "Independent-seed EIG repetition is an offline diagnostic "
                "because it doubles expensive planning work."
            ),
        }
        if mc_rank_stability is None
        else dict(mc_rank_stability)
    )
    shadow = _resolved_shield_view_count_shadow(
        raw=shortlist.get("shield_view_count_shadow"),
        result=result,
        selected_information_gain=selected_eig,
        belief_after_station_id=belief_after_station_id,
        posterior_health=validated_health,
    )
    return {
        "schema_version": 2,
        "station_id": int(station_id),
        "belief_after_station_id": belief_after_station_id,
        "selected_pose_index": int(result.next_pose_index),
        "selection_mode": str(diagnostics.get("selection_mode", "pf_dss_pp")),
        "selected_pose_xyz": [float(value) for value in result.next_pose],
        "selected_program": {
            "name": str(result.shield_program.name),
            "kind": str(result.shield_program.kind),
            "pair_ids": [int(value) for value in result.shield_program.pair_ids],
        },
        "selected_score": float(result.score),
        "selected_information_gain": selected_eig,
        "best_exact_information_gain": best_exact_eig,
        "selected_pose_best_exact_information_gain": diagnostics.get(
            "selected_pose_exact_information_gain_leader"
        ),
        "selected_program_is_exact_eig_leader_at_selected_pose": bool(
            diagnostics.get(
                "selected_program_is_exact_eig_leader_at_selected_pose",
                False,
            )
        ),
        "selected_pose_exact_program_count": int(
            diagnostics.get("selected_pose_exact_program_count", 0)
        ),
        "total_action_count": int(shortlist.get("total_action_count", 0)),
        "shortlisted_pose_count": int(shortlist.get("shortlisted_pose_count", 0)),
        "programs_per_shortlisted_pose": int(
            shortlist.get("programs_per_shortlisted_pose", 0)
        ),
        "full_program_sweep_per_shortlisted_pose": bool(
            shortlist.get("full_program_sweep_per_shortlisted_pose", False)
        ),
        "selected_proxy_rank": int(shortlist.get("shortlist_selected_proxy_rank", 0)),
        "exact_action_count": int(shortlist.get("exact_action_count", 0)),
        "proxy_action_count": int(shortlist.get("proxy_action_count", 0)),
        "planning_particle_count": int(diagnostics.get("planning_particle_count", 0)),
        "score_leader": _leader(leaders, "score"),
        "information_gain_leader": information_leader,
        "top_ranked_actions": [
            dict(_mapping(value, name="ranked node")) for value in ranked[:top_k]
        ],
        "shortlist_certificate": {
            "available": bool(
                shortlist.get(
                    "shortlist_formal_recall_certificate_available",
                    False,
                )
            ),
            "winner_exceeds_excluded_bound": bool(
                shortlist.get(
                    "shortlist_mc_winner_exceeds_universal_excluded_bound",
                    False,
                )
            ),
            "evaluated_objective_lower_bound": shortlist.get(
                "shortlist_evaluated_objective_lower_bound"
            ),
            "excluded_objective_upper_bound": shortlist.get(
                "shortlist_max_excluded_universal_objective_upper_bound"
            ),
        },
        "exact_eig_seed": shortlist.get("exact_eig_seed"),
        "mc_seed_rank_stability": stability,
        "shield_view_count_shadow": shadow,
    }


def build_bootstrap_planner_audit(
    *,
    station_id: int,
    pose_index: int,
    pose_xyz: Sequence[float],
    program: ShieldProgram,
    shadow_enabled: bool,
    candidate_view_counts: tuple[int, ...] = (2, 4, 8),
    retention_fraction: float = 0.95,
    per_comparison_confidence: float = 0.95,
) -> dict[str, object]:
    """Build a schema-aligned prior-only bootstrap planner audit."""
    if (
        isinstance(station_id, bool)
        or not isinstance(station_id, int)
        or station_id != 0
    ):
        raise ValueError("Bootstrap planner audit requires station_id=0.")
    if isinstance(pose_index, bool) or not isinstance(pose_index, int):
        raise ValueError("Bootstrap pose_index must be an integer.")
    pose = [float(value) for value in pose_xyz]
    if len(pose) != 3:
        raise ValueError("Bootstrap pose_xyz must contain three coordinates.")
    counts = tuple(int(value) for value in candidate_view_counts)
    if not counts or tuple(sorted(set(counts))) != counts:
        raise ValueError("Bootstrap shadow view counts must strictly increase.")
    if bool(shadow_enabled) and counts != (2, 4, 8):
        raise ValueError("Enabled bootstrap shadow counts must be (2, 4, 8).")
    pair_ids = tuple(int(value) for value in program.pair_ids)
    if not pair_ids:
        raise ValueError("Bootstrap shield pair IDs must be nonempty.")
    if bool(shadow_enabled) and (
        len(pair_ids) != counts[-1] or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError(
            "Enabled bootstrap shadow execution must use eight unique views."
        )
    executed = _executed_shadow_action(
        pose_index=int(pose_index),
        pose_xyz=pose,
        program=program,
        information_gain=None,
        score=None,
    )
    bootstrap_action = {
        **executed,
        "selection_reason": "bootstrap_prior_only_forced_reference_view_count",
        "fallback_applied": True,
        "fallback_reasons": ["bootstrap_prior_only"],
    }
    shadow = {
        "schema_version": 1,
        "status": "bootstrap_forced" if shadow_enabled else "not_applicable",
        "mode": (
            "audit_only_fixed_8_execution" if shadow_enabled else "not_applicable"
        ),
        "truth_used": False,
        "belief_after_station_id": None,
        "policy": {
            "candidate_view_counts": [int(value) for value in counts],
            "reference_view_count": int(counts[-1]),
            "bootstrap_forced_view_count": int(counts[-1]),
            "retention_fraction": float(retention_fraction),
            "per_comparison_one_sided_confidence": float(per_comparison_confidence),
            "global_coverage_claimed": False,
            "selection_statistic": (
                "paired_lcb_of_information_gain_short_minus_retention_"
                "times_information_gain_reference"
            ),
            "lcb_pass_condition": "strictly_greater_than_zero",
            "program_semantics": "nested_conditional_greedy_prefix",
            "measurement_time_weight_affects_selection": False,
            "configured_measurement_time_weight_audit_only": None,
        },
        "mc_contract": {
            "status": "not_evaluated_before_first_observation",
            "paired_across_view_counts": True,
            "paired_across_poses": False,
            "paired_across_proxy_and_exact": False,
            "prefix_selection_independent_of_exact_lcb_samples": None,
            "selection_bias_control": "not_evaluated_before_first_observation",
            "predictive_pairing": "not_evaluated_before_first_observation",
        },
        "health": _unavailable_shadow_health(
            None,
            reason="bootstrap_prior_only",
        ),
        "proxy": {"status": "not_evaluated_before_first_observation"},
        "exact": {"status": "not_evaluated_before_first_observation"},
        "hypothetical_actions": {
            "point_rule": dict(bootstrap_action),
            "paired_lcb_rule": dict(bootstrap_action),
            "health_gated_rule": dict(bootstrap_action),
            "configured_time_weight_counterfactual": None,
        },
        "executed_action": executed,
        "comparison": {
            "saved_view_count": 0,
            "saved_live_time_s": None,
            "saved_measurement_elapsed_time_s": None,
            "shadow_reference_relationship": None,
        },
    }
    return {
        "schema_version": 2,
        "station_id": 0,
        "belief_after_station_id": None,
        "selected_pose_index": int(pose_index),
        "selection_mode": (
            "external_control_bootstrap"
            if program.kind == "external_control"
            else "pf_prior_balanced_bootstrap"
        ),
        "selected_pose_xyz": pose,
        "selected_program": {
            "name": str(program.name),
            "kind": str(program.kind),
            "pair_ids": [int(value) for value in program.pair_ids],
        },
        "selected_score": None,
        "selected_information_gain": None,
        "best_exact_information_gain": None,
        "selected_pose_best_exact_information_gain": None,
        "selected_program_is_exact_eig_leader_at_selected_pose": False,
        "selected_pose_exact_program_count": 0,
        "total_action_count": 0,
        "shortlisted_pose_count": 0,
        "programs_per_shortlisted_pose": 0,
        "full_program_sweep_per_shortlisted_pose": False,
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
        "mc_seed_rank_stability": {"status": "not_applicable_before_first_observation"},
        "shield_view_count_shadow": shadow,
    }


def _resolved_shield_view_count_shadow(
    *,
    raw: object,
    result: DSSPPResult,
    selected_information_gain: float | None,
    belief_after_station_id: int | None,
    posterior_health: Mapping[str, object] | None,
) -> dict[str, object]:
    """Join truth-free PF health to one planner-owned shadow calculation."""
    executed = _executed_shadow_action(
        pose_index=int(result.next_pose_index),
        pose_xyz=[float(value) for value in result.next_pose],
        program=result.shield_program,
        information_gain=selected_information_gain,
        score=float(result.score),
    )
    health = (
        _unavailable_shadow_health(
            belief_after_station_id,
            reason="posterior_health_unavailable",
        )
        if posterior_health is None
        else dict(posterior_health)
    )
    if raw is None:
        return {
            "schema_version": 1,
            "status": "not_applicable",
            "mode": "not_applicable",
            "truth_used": False,
            "belief_after_station_id": belief_after_station_id,
            "policy": {
                "candidate_view_counts": None,
                "reference_view_count": None,
                "retention_fraction": None,
                "per_comparison_one_sided_confidence": None,
                "global_coverage_claimed": False,
                "selection_statistic": None,
                "lcb_pass_condition": None,
                "program_semantics": None,
                "measurement_time_weight_affects_selection": False,
                "configured_measurement_time_weight_audit_only": None,
            },
            "mc_contract": {
                "status": "not_applicable",
                "paired_across_view_counts": None,
                "paired_across_poses": None,
                "paired_across_proxy_and_exact": None,
                "prefix_selection_independent_of_exact_lcb_samples": None,
                "selection_bias_control": None,
                "predictive_pairing": None,
            },
            "health": health,
            "proxy": {"status": "not_evaluated"},
            "exact": {"status": "not_evaluated"},
            "hypothetical_actions": {
                "point_rule": None,
                "paired_lcb_rule": None,
                "health_gated_rule": dict(executed),
                "configured_time_weight_counterfactual": None,
            },
            "executed_action": executed,
            "comparison": {
                "saved_view_count": 0,
                "saved_live_time_s": None,
                "saved_measurement_elapsed_time_s": None,
                "shadow_reference_relationship": None,
            },
        }
    payload = dict(_mapping(raw, name="shield_view_count_shadow"))
    if payload.get("status") != "evaluated":
        raise ValueError("A present shield view-count shadow must be evaluated.")
    if payload.get("truth_used") is not False:
        raise ValueError("Shield view-count shadow diagnostics must be truth-free.")
    exact = _mapping(payload.get("exact"), name="shield_view_count_shadow.exact")
    policy = _mapping(payload.get("policy"), name="shadow policy")
    candidate_counts_raw = policy.get("candidate_view_counts")
    if not isinstance(candidate_counts_raw, Sequence) or isinstance(
        candidate_counts_raw,
        (str, bytes),
    ):
        raise TypeError("Shadow candidate view counts must be a sequence.")
    candidate_counts = tuple(int(value) for value in candidate_counts_raw)
    reference_count = int(policy.get("reference_view_count", 8))
    if candidate_counts != (2, 4, 8) or reference_count != 8:
        raise ValueError("Evaluated shadow audit requires the (2, 4, 8) policy.")
    if len(result.shield_program.pair_ids) != reference_count:
        raise ValueError("Evaluated shadow audit requires fixed-eight execution.")
    if len(set(result.shield_program.pair_ids)) != reference_count:
        raise ValueError("Executed fixed-eight shield pairs must be unique.")
    exact_pose_indices = _validated_pose_rows(
        exact.get("pose_indices"),
        exact.get("pose_xyz"),
        name="exact shadow",
    )
    exact_by_count = _mapping(exact.get("by_view_count"), name="exact.by_view_count")
    _validate_view_count_pair_rows(
        exact_by_count,
        candidate_view_counts=candidate_counts,
        pose_count=len(exact_pose_indices),
        name="exact shadow",
    )
    proxy = _mapping(payload.get("proxy"), name="shield_view_count_shadow.proxy")
    if proxy.get("status") == "evaluated":
        proxy_pose_indices = _validated_pose_rows(
            proxy.get("pose_indices"),
            proxy.get("pose_xyz"),
            name="proxy shadow",
        )
        proxy_by_count = _mapping(
            proxy.get("by_view_count"),
            name="proxy.by_view_count",
        )
        _validate_view_count_pair_rows(
            proxy_by_count,
            candidate_view_counts=candidate_counts,
            pose_count=len(proxy_pose_indices),
            name="proxy shadow",
        )

    raw_point_action = _mapping(
        exact.get("point_rule_action"),
        name="point_rule_action",
    )
    raw_lcb_action = _mapping(
        exact.get("paired_lcb_rule_action"),
        name="paired_lcb_rule_action",
    )
    raw_time_action = _mapping(
        exact.get("configured_time_weight_counterfactual_action"),
        name="configured_time_weight_counterfactual_action",
    )
    point_count = int(raw_point_action["selected_view_count"])
    lcb_count = int(raw_lcb_action["selected_view_count"])
    point_action = _shadow_action(
        raw_point_action,
        score_field="pose_score_without_measurement_time_penalty",
        score_semantics="eig_plus_spatial_and_motion_without_measurement_time",
        selection_reason=(
            "no_shorter_view_count_met_point_retention"
            if point_count == reference_count
            else "shortest_view_count_met_point_retention"
        ),
    )
    lcb_action = _shadow_action(
        raw_lcb_action,
        score_field="pose_score_without_measurement_time_penalty",
        score_semantics="eig_plus_spatial_and_motion_without_measurement_time",
        selection_reason=(
            "no_shorter_view_count_passed_strict_paired_lcb"
            if lcb_count == reference_count
            else "shortest_view_count_passed_strict_paired_lcb"
        ),
    )
    time_action = _shadow_action(
        raw_time_action,
        score_field="pose_score_with_configured_measurement_time_weight",
        score_semantics="uncalibrated_configured_measurement_time_counterfactual",
        selection_reason="configured_time_weight_counterfactual_only",
    )
    health_passed = bool(health.get("available")) and bool(health.get("passed"))
    failure_reasons = health.get("hard_failure_reasons", [])
    if not isinstance(failure_reasons, Sequence) or isinstance(
        failure_reasons,
        (str, bytes),
    ):
        raise TypeError("posterior health failure reasons must be a sequence.")
    if health_passed:
        health_action = {
            **lcb_action,
            "fallback_applied": False,
            "fallback_reasons": [],
        }
    else:
        health_action = {
            **executed,
            "selection_reason": "posterior_health_forced_reference_view_count",
            "fallback_applied": True,
            "fallback_reasons": [str(value) for value in failure_reasons],
        }
    selected_count = int(health_action["selected_view_count"])
    reference_payload = _mapping(
        exact_by_count.get(str(reference_count)),
        name="reference view-count exact payload",
    )
    selected_payload = _mapping(
        exact_by_count.get(str(selected_count)),
        name="selected view-count exact payload",
    )
    saved_live_time = float(reference_payload["measurement_live_time_s"]) - float(
        selected_payload["measurement_live_time_s"]
    )
    saved_elapsed_time = float(reference_payload["measurement_elapsed_time_s"]) - float(
        selected_payload["measurement_elapsed_time_s"]
    )
    reference_pairs_raw = reference_payload.get("pair_ids")
    if not isinstance(reference_pairs_raw, Sequence) or isinstance(
        reference_pairs_raw,
        (str, bytes),
    ):
        raise TypeError("Exact reference pair rows must be a sequence.")
    try:
        executed_pose_offset = exact_pose_indices.index(int(result.next_pose_index))
    except ValueError as error:
        raise ValueError(
            "Executed pose is absent from the exact shadow union."
        ) from error
    reference_pair_row = reference_pairs_raw[executed_pose_offset]
    if not isinstance(reference_pair_row, Sequence) or isinstance(
        reference_pair_row,
        (str, bytes),
    ):
        raise TypeError("Exact reference shield pairs must be a sequence.")
    reference_pairs = [int(value) for value in reference_pair_row]
    executed_pairs = [int(value) for value in result.shield_program.pair_ids]
    payload.update(
        {
            "belief_after_station_id": belief_after_station_id,
            "health": health,
            "hypothetical_actions": {
                "point_rule": point_action,
                "paired_lcb_rule": lcb_action,
                "health_gated_rule": health_action,
                "configured_time_weight_counterfactual": time_action,
            },
            "executed_action": executed,
            "comparison": {
                "saved_view_count": int(reference_count - selected_count),
                "saved_live_time_s": float(saved_live_time),
                "saved_measurement_elapsed_time_s": float(saved_elapsed_time),
                "shadow_reference_relationship": {
                    "executed_pose_present": True,
                    "executed_program_matches_greedy_prefix_reference": (
                        executed_pairs == reference_pairs
                    ),
                    "shadow_reference_program_semantics": (
                        "nested_conditional_greedy_prefix"
                    ),
                    "executed_program_kind": str(result.shield_program.kind),
                    "shadow_reference_eig_is_executed_eig": False,
                    "reason": (
                        "executed EIG may use one-swap, legacy guard, or "
                        "independent confirmation; shadow I8 uses independent "
                        "holdout samples for the fixed greedy prefix"
                    ),
                },
            },
        }
    )
    return payload


class PlannerAuditWriter:
    """Append one fsync-backed planner decision record per station."""

    def __init__(self, path: str | Path) -> None:
        """Initialize a new append-only audit file."""
        self.path = Path(path).expanduser().resolve()
        if self.path.exists():
            raise FileExistsError(f"Refusing to replace planner audit {self.path}.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = DurableJSONLWriter(self.path, mode=0o644)

    def append(self, payload: Mapping[str, object]) -> None:
        """Durably append one finite JSON object."""
        self._writer.append(dict(payload))

    def close(self) -> None:
        """Close the shared durable writer exactly once."""
        self._writer.close()

    def __enter__(self) -> "PlannerAuditWriter":
        """Return this writer for one deterministic audit lifetime."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the audit writer when leaving a managed lifetime."""
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "PlannerAuditWriter",
    "SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES",
    "build_bootstrap_planner_audit",
    "build_planner_audit",
]
