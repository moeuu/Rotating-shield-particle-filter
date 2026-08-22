"""Tests for durable PF planner decision auditing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planning.audit import PlannerAuditWriter, build_planner_audit
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
                "shortlist_selected_proxy_rank": 3,
                "shortlist_formal_recall_certificate_available": True,
                "shortlist_mc_winner_exceeds_universal_excluded_bound": True,
                "shortlist_evaluated_objective_lower_bound": 3.5,
                "shortlist_max_excluded_universal_objective_upper_bound": 3.2,
                "exact_eig_seed": 41,
            },
            "component_leaders": {
                "score": leader,
                "information_gain": leader,
            },
            "ranked_nodes": [leader],
        },
    )


def test_planner_audit_captures_domain_shortlist_and_leaders() -> None:
    """The audit must preserve every requested action-selection diagnostic."""
    audit = build_planner_audit(station_id=4, result=_result(), top_k=5)

    assert audit["station_id"] == 4
    assert audit["total_action_count"] == 128
    assert audit["selected_proxy_rank"] == 3
    assert audit["exact_action_count"] == 32
    assert audit["selected_information_gain"] == 2.5
    assert audit["best_exact_information_gain"] == 2.5
    assert audit["score_leader"]["pose_index"] == 2
    assert audit["information_gain_leader"]["pose_index"] == 2
    assert len(audit["top_ranked_actions"]) == 1
    assert audit["shortlist_certificate"]["available"] is True
    assert audit["exact_eig_seed"] == 41
    assert audit["mc_seed_rank_stability"]["status"] == (
        "not_evaluated_in_control_loop"
    )


def test_planner_audit_writer_is_append_only_and_durable(tmp_path: Path) -> None:
    """Each station must produce one independent JSONL audit row."""
    path = tmp_path / "planner.jsonl"
    with PlannerAuditWriter(path) as writer:
        writer.append(build_planner_audit(station_id=0, result=_result()))
        writer.append(build_planner_audit(station_id=1, result=_result()))

    with pytest.raises(ValueError, match="closed"):
        writer.append(build_planner_audit(station_id=2, result=_result()))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["station_id"] for row in rows] == [0, 1]

    try:
        PlannerAuditWriter(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing planner audit was not protected.")
