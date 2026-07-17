"""Tests for the opt-in, recommendation-only hybrid DSS-PP boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from pf.external_relocation import covered_records_sha256
from pf.hybrid_planning import (
    HybridPlanningBoundaryError,
    candidate_poses_sha256,
    planning_request_from_mapping,
    recommend_hybrid_dsspp_action,
)
from pf.replay import build_replay_estimator, replay_records
from pf.provenance import sha256_json
from planning.dss_pp import select_dss_pp_next_station
from runtime.measurement_log import load_measurement_log
from tests.pure_pf_test_support import make_measurement_log, replay_config


def _dsspp_config() -> dict[str, object]:
    """Return a small deterministic DSS-PP configuration for boundary tests."""
    return {
        "augment_candidates": False,
        "include_runtime_rescue_modes": False,
        "include_global_surface_rescue_modes": False,
        "horizon": 1,
        "beam_width": 2,
        "max_programs": 1,
        "program_length": 1,
        "forced_program_pair_ids": [0],
        "program_eval_workers": 1,
        "candidate_preselect_enable": False,
        "lambda_eig": 0.0,
        "lambda_signature": 1.0,
        "lambda_distance": 0.1,
        "same_isotope_direct_separation_guard": False,
        "max_modes_per_isotope": 8,
        "diagnostic_ranked_node_limit": 8,
    }


def _request(
    log: object,
    resolved_config_sha256: str,
    *,
    external_modes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one cutoff-bound, collision-attested planning request."""
    candidates = [
        [0.45, 0.35, 0.35],
        [1.25, 0.75, 0.8],
        [1.65, 1.55, 1.25],
    ]
    return {
        "schema_version": 1,
        "request_id": "planning-request-001",
        "source_run_id": log.run_id,  # type: ignore[attr-defined]
        "data_cutoff_step": 1,
        "data_cutoff_station": 0,
        "covered_records_sha256": covered_records_sha256(log, 2),  # type: ignore[arg-type]
        "pf_resolved_config_sha256": resolved_config_sha256,
        "current_pose_xyz": [0.25, 0.25, 0.4],
        "current_pair_id": None,
        "visited_poses_xyz": [[0.25, 0.25, 0.4]],
        "candidate_poses_xyz": candidates,
        "candidate_attestation": {
            "candidate_poses_sha256": candidate_poses_sha256(candidates),
            "workspace_sha256": "b" * 64,
            "planning_config_sha256": "c" * 64,
            "collision_checked": True,
            "reachability_filtered": True,
        },
        "dsspp_config": _dsspp_config(),
        "external_modes": [] if external_modes is None else external_modes,
        "bounds_xyz": {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 1.5]},
        "continuous_height_bounds_m": [0.2, 1.4],
    }


def _context(tmp_path: Path, *, record_count: int = 4) -> tuple[Path, object, str]:
    """Create a marked log and resolve the PF config digest used by requests."""
    path = make_measurement_log(
        tmp_path / f"measurement-log-{record_count}",
        record_count=record_count,
        station_complete_markers=True,
    )
    log = load_measurement_log(path)
    estimator = build_replay_estimator(
        log,
        replay_config(),
        profile="pf_strict",
        seed=37,
    )
    return path, log, estimator.resolved_config_hash


def test_request_requires_attested_candidates_and_disables_augmentation(
    tmp_path: Path,
) -> None:
    """Unattested or internally augmentable pose sets must fail closed."""
    _path, log, config_digest = _context(tmp_path)
    payload = _request(log, config_digest)
    parsed = planning_request_from_mapping(payload)
    assert parsed.current_pair_id is None
    assert parsed.candidate_attestation.candidate_poses_sha256 == (
        candidate_poses_sha256(payload["candidate_poses_xyz"])
    )

    unsafe = json.loads(json.dumps(payload))
    unsafe["candidate_attestation"]["collision_checked"] = False
    with pytest.raises(HybridPlanningBoundaryError, match="collision_checked=true"):
        planning_request_from_mapping(unsafe)

    augmented = json.loads(json.dumps(payload))
    augmented["dsspp_config"]["augment_candidates"] = True
    with pytest.raises(HybridPlanningBoundaryError, match="augment_candidates=false"):
        planning_request_from_mapping(augmented)

    altered = json.loads(json.dumps(payload))
    altered["candidate_poses_xyz"][0][2] = 0.36
    with pytest.raises(HybridPlanningBoundaryError, match="digest does not match"):
        planning_request_from_mapping(altered)


def test_empty_external_modes_match_normal_pure_pf_dsspp(tmp_path: Path) -> None:
    """An empty external belief must reduce to the normal pure-PF planner."""
    path, log, config_digest = _context(tmp_path)
    payload = _request(log, config_digest)
    parsed = planning_request_from_mapping(payload)
    prefix = log.prefix(2)  # type: ignore[attr-defined]
    pure = build_replay_estimator(
        prefix,
        replay_config(),
        profile="pf_strict",
        seed=37,
    )
    replay_records(prefix, pure)
    pure_state = pure.serialized_state()
    direct = select_dss_pp_next_station(
        pure,
        np.asarray(parsed.candidate_poses_xyz, dtype=float),
        np.asarray(parsed.current_pose_xyz, dtype=float),
        current_pair_id=None,
        visited_poses_xyz=np.asarray(parsed.visited_poses_xyz, dtype=float),
        map_api=None,
        bounds_xyz=(
            np.asarray(parsed.bounds_xyz[0], dtype=float),  # type: ignore[index]
            np.asarray(parsed.bounds_xyz[1], dtype=float),  # type: ignore[index]
        ),
        continuous_height_bounds_m=parsed.continuous_height_bounds_m,
        config=parsed.dsspp_config,
    )
    assert pure.serialized_state() == pure_state
    pure.planner_only_external_signature_modes = lambda: {}  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="explicit hybrid recommendation boundary"):
        select_dss_pp_next_station(
            pure,
            np.asarray(parsed.candidate_poses_xyz, dtype=float),
            np.asarray(parsed.current_pose_xyz, dtype=float),
            config=parsed.dsspp_config,
        )
    del pure.planner_only_external_signature_modes

    wrapped = recommend_hybrid_dsspp_action(
        path,
        replay_config(),
        payload,
        seed=37,
    )
    np.testing.assert_array_equal(
        wrapped["selected_action"]["pose_xyz"],
        direct.next_pose,
    )
    assert wrapped["selected_action"]["shield_program"]["pair_ids"] == list(
        direct.shield_program.pair_ids
    )
    assert wrapped["selected_action"]["score"] == direct.score
    assert wrapped["belief"]["planner_belief_sources"] == [
        "pf_posterior",
        "pf_tentative",
    ]


def test_pending_and_verified_modes_are_planner_only_and_quarantine_is_excluded(
    tmp_path: Path,
) -> None:
    """Only eligible external modes reach DSS-PP and PF bytes remain unchanged."""
    path, log, config_digest = _context(tmp_path)
    modes = [
        {
            "mode_id": "mode-pending",
            "isotope": "Cs-137",
            "position_xyz": [1.45, 1.35, 1.1],
            "strength_cps_1m": 900.0,
            "weight": 0.35,
            "spread_m": 0.2,
            "verification_state": "pending",
            "source_snapshot_id": "snapshot-001",
        },
        {
            "mode_id": "mode-verified",
            "isotope": "Cs-137",
            "position_xyz": [0.55, 1.45, 0.65],
            "strength_cps_1m": 700.0,
            "weight": 0.25,
            "spread_m": 0.1,
            "verification_state": "verified",
            "source_snapshot_id": "snapshot-001",
        },
        {
            "mode_id": "mode-quarantined",
            "isotope": "Xe-999",
            "position_xyz": [20.0, 20.0, 20.0],
            "strength_cps_1m": 1.0e6,
            "weight": 1.0,
            "spread_m": 0.0,
            "verification_state": "quarantined",
            "source_snapshot_id": "snapshot-001",
        },
    ]
    payload = _request(log, config_digest, external_modes=modes)
    payload["dsspp_config"]["program_eval_workers"] = 2  # type: ignore[index]
    result = recommend_hybrid_dsspp_action(
        path,
        replay_config(),
        payload,
        seed=37,
    )

    assert result["algorithmic_recommendation_only"] is True
    assert result["robot_actuation_authorized"] is False
    assert result["provenance"]["causal_planning_request_sha256"] == sha256_json(
        payload
    )
    assert result["selected_action"]["candidate_index"] in {0, 1, 2}
    assert result["selected_action"]["detector_height_m"] in {0.35, 0.8, 1.25}
    assert result["selected_action"]["shield_program"]["pair_ids"] == [0]
    assert result["belief"]["planner_belief_sources"] == [
        "pf_posterior",
        "pf_tentative",
        "external_mode_pending",
        "external_mode_verified",
    ]
    assert result["belief"]["included_external_mode_ids"] == [
        "mode-pending",
        "mode-verified",
    ]
    assert result["belief"]["excluded_quarantined_mode_ids"] == ["mode-quarantined"]
    assert result["diagnostics"]["planner_only_external_mode_counts"] == {"Cs-137": 2}
    integrity = result["pf_state_integrity"]
    assert (
        integrity["state_sha256_before_planning"]
        == integrity["state_sha256_after_planning"]
    )
    assert integrity["pf_particles_or_weights_mutated_by_planning"] is False
    assert integrity["external_modes_mutated_pf"] is False
    assert result["candidate_attestation"] == {
        "candidate_poses_sha256": candidate_poses_sha256(
            _request(log, config_digest)["candidate_poses_xyz"]
        ),
        "workspace_sha256": "b" * 64,
        "planning_config_sha256": "c" * 64,
        "collision_checked": True,
        "reachability_filtered": True,
    }
    assert "measurement_log_sha256" not in json.dumps(result, sort_keys=True)


def test_future_log_and_directive_suffix_do_not_change_recommendation(
    tmp_path: Path,
) -> None:
    """Only the bound record prefix and causal directive prefix affect output."""
    short_path, short_log, short_digest = _context(tmp_path / "short", record_count=4)
    long_path, long_log, long_digest = _context(tmp_path / "long", record_count=6)
    assert short_digest == long_digest
    short_request = _request(short_log, short_digest)
    long_request = _request(long_log, long_digest)
    assert (
        short_request["covered_records_sha256"]
        == long_request["covered_records_sha256"]
    )
    empty = {"schema_version": 1, "directives": []}
    future_suffix = {
        "schema_version": 1,
        "directives": [
            {
                "directive_id": "intentionally-unvalidated-future-suffix",
                "apply_after_step": 999,
            }
        ],
    }
    short_result = recommend_hybrid_dsspp_action(
        short_path,
        replay_config(),
        short_request,
        directive_schedule=empty,
        seed=37,
    )
    long_result = recommend_hybrid_dsspp_action(
        long_path,
        replay_config(),
        long_request,
        directive_schedule=future_suffix,
        seed=37,
    )
    assert short_result == long_result


def test_hybrid_planning_boundary_has_no_mle_or_orchestrator_import() -> None:
    """The generic PF planning boundary must remain estimator-neutral."""
    source = (
        Path(__file__).parents[1] / "src" / "pf" / "hybrid_planning.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("three_d_estimation" in name for name in imported)
    assert not any(name.startswith("orchestrator") for name in imported)
