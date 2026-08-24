"""Tests for durable PF planner decision auditing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from planning.audit import (
    PlannerAuditWriter,
    SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES,
    _shadow_action,
    build_bootstrap_planner_audit,
    build_planner_audit,
)
from planning.dss_pp import DSSPPNode, DSSPPResult, ShieldProgram


def _result() -> DSSPPResult:
    """Return one deterministic DSS result with shortlist diagnostics."""
    program = ShieldProgram("program", (0, 9), "test")
    node = DSSPPNode(
        pose_index=2,
        pose_xyz=np.asarray([1.0, 2.0, 3.0]),
        program=program,
        score=4.0,
        static_score=3.0,
        distance_weight=0.1,
        information_gain=2.5,
        coverage_gain=0.5,
        revisit_penalty=0.0,
        bearing_diversity_gain=0.2,
        frontier_gain=0.3,
        turn_penalty=0.0,
        local_orbit_gain=0.1,
        elevation_condition_gain=0.4,
    )
    leader = {
        "rank": 1,
        "pose_index": 2,
        "pose_xyz": [1.0, 2.0, 3.0],
        "program_name": "program",
        "program_kind": "test",
        "pair_ids": [0, 9],
        "score": 4.0,
        "information_gain": 2.5,
    }
    return DSSPPResult(
        next_pose=node.pose_xyz,
        next_pose_index=2,
        shield_program=program,
        score=node.score,
        sequence=(node,),
        diagnostics={
            "planning_eig_shortlist": {
                "total_action_count": 128,
                "proxy_action_count": 128,
                "exact_action_count": 32,
                "shortlisted_pose_count": 2,
                "programs_per_shortlisted_pose": 16,
                "full_program_sweep_per_shortlisted_pose": True,
                "shortlist_selected_proxy_rank": 3,
                "shortlist_formal_recall_certificate_available": True,
                "shortlist_mc_winner_exceeds_universal_excluded_bound": True,
                "shortlist_evaluated_objective_lower_bound": 3.5,
                "shortlist_max_excluded_universal_objective_upper_bound": 3.2,
                "exact_eig_seed": 41,
            },
            "selected_pose_exact_information_gain_leader": 2.5,
            "selected_program_is_exact_eig_leader_at_selected_pose": True,
            "selected_pose_exact_program_count": 16,
            "component_leaders": {
                "score": leader,
                "information_gain": leader,
            },
            "ranked_nodes": [leader],
        },
    )


def test_planner_audit_captures_domain_shortlist_and_leaders() -> None:
    """The audit must preserve every requested action-selection diagnostic."""
    audit = build_planner_audit(
        station_id=4,
        belief_after_station_id=3,
        result=_result(),
        top_k=5,
    )

    assert audit["station_id"] == 4
    assert audit["schema_version"] == 2
    assert audit["selected_pose_index"] == 2
    assert audit["total_action_count"] == 128
    assert audit["selected_proxy_rank"] == 3
    assert audit["exact_action_count"] == 32
    assert audit["selected_information_gain"] == 2.5
    assert audit["best_exact_information_gain"] == 2.5
    assert audit["selected_pose_best_exact_information_gain"] == 2.5
    assert audit["selected_program_is_exact_eig_leader_at_selected_pose"] is True
    assert audit["selected_pose_exact_program_count"] == 16
    assert audit["shortlisted_pose_count"] == 2
    assert audit["programs_per_shortlisted_pose"] == 16
    assert audit["full_program_sweep_per_shortlisted_pose"] is True
    assert audit["score_leader"]["pose_index"] == 2
    assert audit["information_gain_leader"]["pose_index"] == 2
    assert len(audit["top_ranked_actions"]) == 1
    assert audit["shortlist_certificate"]["available"] is True
    assert audit["exact_eig_seed"] == 41
    assert audit["mc_seed_rank_stability"]["status"] == (
        "not_evaluated_in_control_loop"
    )
    assert audit["shield_view_count_shadow"]["status"] == "not_applicable"


def test_planner_audit_health_gates_an_evaluated_shadow_action() -> None:
    """Information-only choices must remain visible under an eight-view fallback."""
    result = _result()
    fixed_eight = ShieldProgram(
        "fixed-eight",
        (0, 1, 2, 3, 4, 5, 6, 7),
        "test",
    )
    result = replace(
        result,
        shield_program=fixed_eight,
        sequence=(replace(result.sequence[0], program=fixed_eight),),
    )
    result.diagnostics["planning_eig_shortlist"]["shield_view_count_shadow"] = {
        "schema_version": 1,
        "status": "evaluated",
        "mode": "audit_only_fixed_8_execution",
        "truth_used": False,
        "policy": {
            "reference_view_count": 8,
            "candidate_view_counts": [2, 4, 8],
            "retention_fraction": 0.95,
            "per_comparison_one_sided_confidence": 0.95,
            "global_coverage_claimed": False,
            "selection_statistic": "paired_lcb",
            "lcb_pass_condition": "strictly_greater_than_zero",
            "program_semantics": "nested_conditional_greedy_prefix",
            "measurement_time_weight_affects_selection": False,
            "configured_measurement_time_weight_audit_only": 0.02,
        },
        "mc_contract": {
            "status": "evaluated",
            "paired_across_view_counts": True,
            "paired_across_poses": False,
            "paired_across_proxy_and_exact": False,
            "prefix_selection_independent_of_exact_lcb_samples": True,
            "selection_bias_control": "independent_holdout",
            "predictive_pairing": "same_holdout_cache",
        },
        "proxy": {"status": "skipped_all_valid_poses_exact"},
        "exact": {
            "point_rule_action": {
                "pose_index": 1,
                "pose_xyz": [0.0, 1.0, 2.0],
                "selected_view_count": 2,
                "pair_ids": [1, 2],
            },
            "paired_lcb_rule_action": {
                "pose_index": 1,
                "pose_xyz": [0.0, 1.0, 2.0],
                "selected_view_count": 4,
                "pair_ids": [1, 2, 3, 4],
            },
            "configured_time_weight_counterfactual_action": {
                "pose_index": 1,
                "pose_xyz": [0.0, 1.0, 2.0],
                "selected_view_count": 2,
                "pair_ids": [1, 2],
                "pose_score_with_configured_measurement_time_weight": 0.1,
            },
            "pose_indices": [1, 2],
            "pose_xyz": [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
            "by_view_count": {
                "2": {
                    "pair_ids": [[1, 2], [0, 1]],
                    "measurement_live_time_s": 40.0,
                    "measurement_elapsed_time_s": 42.0,
                },
                "4": {
                    "pair_ids": [[1, 2, 3, 4], [0, 1, 2, 3]],
                    "measurement_live_time_s": 80.0,
                    "measurement_elapsed_time_s": 84.0,
                },
                "8": {
                    "pair_ids": [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                    ],
                    "measurement_live_time_s": 160.0,
                    "measurement_elapsed_time_s": 168.0,
                },
            },
        },
    }
    unhealthy = {
        "policy_schema_version": 1,
        "hard_gate_contract": list(SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES),
        "available": True,
        "passed": False,
        "source_station_id": 3,
        "hard_failure_reasons": ["particle_diversity_warning"],
        "truth_used": False,
    }

    audit = build_planner_audit(
        station_id=4,
        belief_after_station_id=3,
        posterior_health=unhealthy,
        result=result,
    )
    shadow = audit["shield_view_count_shadow"]

    assert shadow["hypothetical_actions"]["point_rule"]["selected_view_count"] == 2
    assert shadow["hypothetical_actions"]["paired_lcb_rule"]["selected_view_count"] == 4
    assert (
        shadow["hypothetical_actions"]["health_gated_rule"]["selected_view_count"] == 8
    )
    assert (
        shadow["hypothetical_actions"]["health_gated_rule"]["fallback_applied"] is True
    )
    comparison = shadow["comparison"]
    assert comparison["saved_view_count"] == 0
    assert comparison["saved_live_time_s"] == 0.0
    assert comparison["saved_measurement_elapsed_time_s"] == 0.0
    assert comparison["shadow_reference_relationship"] == {
        "executed_pose_present": True,
        "executed_program_matches_greedy_prefix_reference": True,
        "shadow_reference_program_semantics": ("nested_conditional_greedy_prefix"),
        "executed_program_kind": "test",
        "shadow_reference_eig_is_executed_eig": False,
        "reason": (
            "executed EIG may use one-swap, legacy guard, or independent "
            "confirmation; shadow I8 uses independent holdout samples for the "
            "fixed greedy prefix"
        ),
    }
    required_policy_keys = {
        "candidate_view_counts",
        "reference_view_count",
        "retention_fraction",
        "per_comparison_one_sided_confidence",
        "global_coverage_claimed",
        "selection_statistic",
        "lcb_pass_condition",
        "program_semantics",
        "measurement_time_weight_affects_selection",
        "configured_measurement_time_weight_audit_only",
    }
    required_mc_keys = {
        "status",
        "paired_across_view_counts",
        "paired_across_poses",
        "paired_across_proxy_and_exact",
        "prefix_selection_independent_of_exact_lcb_samples",
        "selection_bias_control",
        "predictive_pairing",
    }
    assert required_policy_keys.issubset(shadow["policy"])
    assert required_mc_keys.issubset(shadow["mc_contract"])


def test_bootstrap_audit_forces_fixed_eight_views_with_schema_parity() -> None:
    """The prior-only row must expose the same top-level audit contract."""
    result = _result()
    planned = build_planner_audit(
        station_id=1,
        belief_after_station_id=0,
        result=result,
    )
    bootstrap = build_bootstrap_planner_audit(
        station_id=0,
        pose_index=0,
        pose_xyz=[1.0, 1.0, 0.5],
        program=ShieldProgram(
            "bootstrap",
            (0, 9, 18, 27, 36, 45, 54, 63),
            "prior_balanced_bootstrap",
        ),
        shadow_enabled=True,
    )

    assert set(bootstrap) == set(planned)
    assert bootstrap["schema_version"] == 2
    shadow = bootstrap["shield_view_count_shadow"]
    assert shadow["status"] == "bootstrap_forced"
    assert shadow["executed_action"]["selected_view_count"] == 8
    assert (
        shadow["hypothetical_actions"]["health_gated_rule"]["selected_view_count"] == 8
    )
    not_applicable = planned["shield_view_count_shadow"]
    assert set(shadow["policy"]) == set(not_applicable["policy"]) | {
        "bootstrap_forced_view_count"
    }
    assert set(shadow["mc_contract"]) == set(not_applicable["mc_contract"])
    assert set(shadow["comparison"]) == set(not_applicable["comparison"])


def test_external_bootstrap_marks_shadow_not_applicable() -> None:
    """A repeated-pair baseline must not enter the conditional-greedy audit."""
    audit = build_bootstrap_planner_audit(
        station_id=0,
        pose_index=0,
        pose_xyz=[1.0, 1.0, 0.5],
        program=ShieldProgram(
            "passive-baseline",
            (0, 0, 0, 0, 0, 0, 0, 0),
            "external_control",
        ),
        shadow_enabled=False,
    )

    shadow = audit["shield_view_count_shadow"]
    assert shadow["status"] == "not_applicable"
    assert shadow["mode"] == "not_applicable"
    assert shadow["executed_action"]["pair_ids"] == [0] * 8


def test_planner_audit_rejects_misaligned_belief_or_health() -> None:
    """Planner evidence must identify the exact preceding truth-free belief."""
    with pytest.raises(ValueError, match="preceding PF belief"):
        build_planner_audit(station_id=2, result=_result())
    with pytest.raises(ValueError, match="health and planner belief"):
        build_planner_audit(
            station_id=2,
            belief_after_station_id=1,
            posterior_health={
                "source_station_id": 0,
                "truth_used": False,
            },
            result=_result(),
        )
    with pytest.raises(ValueError, match="policy schema"):
        build_planner_audit(
            station_id=2,
            belief_after_station_id=1,
            posterior_health={
                "source_station_id": 1,
                "truth_used": False,
                "available": True,
                "passed": True,
                "hard_failure_reasons": [],
            },
            result=_result(),
        )


@pytest.mark.parametrize(
    "raw",
    (
        {
            "pose_index": 1,
            "pose_xyz": [0.0, 1.0, float("nan")],
            "selected_view_count": 2,
            "pair_ids": [0, 1],
        },
        {
            "pose_index": 1,
            "pose_xyz": [0.0, 1.0, 2.0],
            "selected_view_count": 2,
            "pair_ids": [0, 0],
        },
        {
            "pose_index": 1,
            "pose_xyz": [0.0, 1.0, 2.0],
            "selected_view_count": 2,
            "pair_ids": [0, 64],
        },
    ),
)
def test_shadow_action_rejects_invalid_pose_or_pair_domain(
    raw: dict[str, object],
) -> None:
    """Malformed shadow actions must not reach durable planner audit."""
    with pytest.raises(ValueError):
        _shadow_action(
            raw,
            score_field="pose_score_without_measurement_time_penalty",
            score_semantics="test",
            selection_reason="test",
        )


def test_planner_audit_writer_is_append_only_and_durable(tmp_path: Path) -> None:
    """Each station must produce one independent JSONL audit row."""
    path = tmp_path / "planner.jsonl"
    with PlannerAuditWriter(path) as writer:
        writer.append(build_planner_audit(station_id=0, result=_result()))
        writer.append(
            build_planner_audit(
                station_id=1,
                belief_after_station_id=0,
                result=_result(),
            )
        )

    with pytest.raises(ValueError, match="closed"):
        writer.append(
            build_planner_audit(
                station_id=2,
                belief_after_station_id=1,
                result=_result(),
            )
        )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["station_id"] for row in rows] == [0, 1]

    try:
        PlannerAuditWriter(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing planner audit was not protected.")
