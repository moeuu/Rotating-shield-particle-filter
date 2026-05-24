"""Tests for RA-L ablation baseline utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.ral_ablation.config_factory import (
    DEFAULT_ABLATION_CASES,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    build_ablation_plan,
)
from baselines.ral_ablation.path_policies import select_baseline_next_pose
from baselines.ral_ablation.shield_policies import select_baseline_shield_program


def test_fixed_shield_policy_repeats_one_pair() -> None:
    """Fixed-shield ablation should repeat the requested pair id."""
    program = select_baseline_shield_program(
        {"name": "fixed", "fixed_pair_id": 7},
        total_pairs=64,
        program_length=8,
        pose_index=3,
    )
    assert program is not None
    assert program.pair_ids == (7,) * 8


def test_round_robin_shield_policy_advances_by_pose() -> None:
    """Round-robin ablation should produce deterministic non-adaptive programs."""
    program = select_baseline_shield_program(
        {"name": "round_robin", "start_pair_id": 2, "advance_by_pose": True},
        total_pairs=8,
        program_length=4,
        pose_index=1,
    )
    assert program is not None
    assert program.pair_ids == (6, 7, 0, 1)


def test_passive_serpentine_path_policy_selects_candidate_near_waypoint() -> None:
    """Passive path baseline should select by geometry, not PF information."""
    candidates = np.asarray(
        [
            [9.0, 0.0, 0.5],
            [0.0, 10.0, 0.5],
            [8.0, 20.0, 0.5],
        ],
        dtype=float,
    )
    selection = select_baseline_next_pose(
        {"name": "passive_serpentine", "row_count": 3},
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.asarray([1.0, 1.0, 0.5], dtype=float),
        visited_poses_xyz=np.asarray([[1.0, 1.0, 0.5]], dtype=float),
        bounds_xyz=(
            np.asarray([0.0, 0.0, 0.5], dtype=float),
            np.asarray([10.0, 20.0, 0.5], dtype=float),
        ),
    )
    assert selection is not None
    assert selection.candidate_index == 1


def test_ablation_plan_generates_isolated_baseline_configs(tmp_path) -> None:
    """Ablation factory should write source/config files with baseline-only keys."""
    entries = build_ablation_plan(
        output_dir=tmp_path,
        seeds=(1234,),
        cases=DEFAULT_ABLATION_CASES[:1],
        variants=DEFAULT_ABLATION_VARIANTS,
        intensity_cps_1m=30000.0,
    )
    by_variant = {entry.variant: entry for entry in entries}
    assert "proposed" in by_variant
    assert "fixed_shield" in by_variant
    assert "baseline_passive_no_shield" in by_variant
    assert "baseline_passive_fixed_shield_single_view" in by_variant
    assert "baseline_onestep_fixed_shield" in by_variant
    assert "no_obstacle_signature" in by_variant
    assert "no_pf_obstacle_attenuation" in by_variant
    assert "volume_source_prior" in by_variant
    fixed_config = json.loads(by_variant["fixed_shield"].config_path.read_text())
    assert fixed_config["baseline_shield_policy"]["name"] == "fixed"
    assert fixed_config["cui_split_view_dir"] == DEFAULT_CUI_SPLIT_VIEW_DIR
    assert fixed_config["usd_path"].endswith("/configs/isaacsim/demo_room.usda")
    assert Path(fixed_config["usd_path"]).is_absolute()
    assert fixed_config["random_environment_base_usd_path"].endswith(
        "/configs/isaacsim/demo_room.usda"
    )
    assert Path(fixed_config["random_environment_base_usd_path"]).is_absolute()
    obstacle_off = json.loads(
        by_variant["no_obstacle_signature"].config_path.read_text()
    )
    assert obstacle_off["dss_pp"]["environment_signature_weight"] == 0.0
    assert obstacle_off["dss_pp"]["occlusion_boundary_weight"] == 0.0
    pf_obstacle_off = json.loads(
        by_variant["no_pf_obstacle_attenuation"].config_path.read_text()
    )
    assert pf_obstacle_off["pf_obstacle_attenuation"] is False
    assert pf_obstacle_off["author_obstacle_prims"] is True
    assert pf_obstacle_off["dss_pp"]["environment_signature_weight"] > 0.0
    volume_prior = json.loads(
        by_variant["volume_source_prior"].config_path.read_text()
    )
    assert volume_prior["source_surface_prior"] is False
    no_birth = json.loads(by_variant["no_residual_birth"].config_path.read_text())
    assert no_birth["birth_max_per_update"] == 0
    assert no_birth["pf_max_sources"] == DEFAULT_ABLATION_CASES[0].max_sources
    passive_no_shield = json.loads(
        by_variant["baseline_passive_no_shield"].config_path.read_text()
    )
    assert passive_no_shield["shield_transmission_target"] == 1.0
    assert passive_no_shield["shield_thickness_scale"] == 0.0
    assert passive_no_shield["baseline_path_policy"]["name"] == "passive_serpentine"
    assert passive_no_shield["baseline_shield_policy"]["name"] == "fixed"
    assert passive_no_shield["thread_count"] >= 1
    assert passive_no_shield["python_worker_count"] >= 1
    assert passive_no_shield["parallel_isotope_updates"] is True
    assert passive_no_shield["parallel_isotope_workers"] >= 1
    assert passive_no_shield["dss_pp"]["program_eval_workers"] >= 1
    single_view = json.loads(
        by_variant["baseline_passive_fixed_shield_single_view"].config_path.read_text()
    )
    assert single_view["orientation_k"] == 1
    assert single_view["min_rotations_per_pose"] == 1
    assert single_view["dss_pp"]["program_length"] == 1
    assert single_view["dss_pp"]["residual_program_length"] == 1
    assert single_view["baseline_path_policy"]["name"] == "passive_serpentine"
    assert single_view["baseline_shield_policy"]["name"] == "fixed"
    one_step_fixed = json.loads(
        by_variant["baseline_onestep_fixed_shield"].config_path.read_text()
    )
    assert one_step_fixed["path_planner"] == "one_step"
    assert one_step_fixed["baseline_shield_policy"]["name"] == "fixed"
    source_payload = json.loads(by_variant["proposed"].source_path.read_text())
    assert len(source_payload["sources"]) == DEFAULT_ABLATION_CASES[0].source_count
    assert "--full-simulation" in by_variant["proposed"].command
    assert "--rotation-overhead-s" in by_variant[
        "baseline_passive_no_shield_single_view"
    ].command
    assert "0.0" in by_variant["baseline_passive_no_shield_single_view"].command
