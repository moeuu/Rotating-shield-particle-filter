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
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.estimator import RotatingShieldPFConfig
from pf.particle_filter import PFConfig
from realtime_demo import _resolve_rotation_limit_for_active_program
from runtime_defaults import DEFAULT_MEASUREMENT_TIME_S, DEFAULT_NO_ROTATION_OVERHEAD_S
from scripts.build_cs4_feature_validation import (
    build_cs4_feature_validation_plan,
    build_feature_validation_plan,
)


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


def test_pf_max_sources_default_is_shared() -> None:
    """PF entry points should use one shared default source-count support."""
    assert RotatingShieldPFConfig().max_sources == DEFAULT_MAX_SOURCES_PER_ISOTOPE
    assert PFConfig().max_sources == DEFAULT_MAX_SOURCES_PER_ISOTOPE


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


def test_explicit_shield_program_rotation_limit_is_strict_for_baselines() -> None:
    """Baseline shield programs should not be padded by adaptive shield selection."""
    assert (
        _resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy={"name": "round_robin"},
        )
        == 2
    )
    assert (
        _resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=True,
            baseline_shield_policy=None,
        )
        == 2
    )
    assert (
        _resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy=None,
        )
        == 8
    )
    assert (
        _resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy=None,
            force_strict_program=True,
        )
        == 2
    )


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
    assert "baseline_passive_equal_time_no_shield" in by_variant
    assert "baseline_passive_fixed_shield_single_view" in by_variant
    assert "baseline_onestep_fixed_shield" in by_variant
    assert "eig_only_path" in by_variant
    assert "no_verification" in by_variant
    assert "no_obstacle_signature" in by_variant
    assert "no_pf_obstacle_attenuation" in by_variant
    assert "volume_source_prior" in by_variant
    fixed_config = json.loads(by_variant["fixed_shield"].config_path.read_text())
    proposed_config = json.loads(by_variant["proposed"].config_path.read_text())
    round_robin = json.loads(by_variant["round_robin_shield"].config_path.read_text())
    assert proposed_config["response_poisson_low_snr_suppress_count"] is False
    assert proposed_config["precision_diagnostic_birth_candidate_enable"] is False
    assert proposed_config["precision_diagnostic_birth_candidate_log_limit"] == 0
    assert proposed_config["precision_diagnostic_particle_log_limit"] == 0
    assert (
        proposed_config["precision_diagnostic_full_spectrum_response_enable"] is False
    )
    assert proposed_config["surface_observability_diagnostic_candidates"] == 0
    assert proposed_config["sparse_poisson_evidence_min_distinct_stations"] == 2
    assert round_robin["orientation_k"] == proposed_config["orientation_k"]
    assert (
        round_robin["min_rotations_per_pose"]
        == proposed_config["min_rotations_per_pose"]
    )
    assert (
        round_robin["dss_pp"]["program_length"]
        == proposed_config["dss_pp"]["program_length"]
    )
    assert (
        round_robin["dss_pp"]["residual_program_length"]
        == proposed_config["dss_pp"]["residual_program_length"]
    )
    assert round_robin["strict_planned_shield_program"] is True
    assert round_robin["dss_pp"]["adaptive_program_length_enable"] is False
    assert round_robin["baseline_shield_policy"]["name"] == "round_robin"
    assert "baseline_path_policy" not in round_robin
    assert fixed_config["baseline_shield_policy"]["name"] == "fixed"
    assert fixed_config["cui_split_view_dir"] == DEFAULT_CUI_SPLIT_VIEW_DIR
    assert fixed_config["usd_path"].endswith("/configs/isaacsim/demo_room.usda")
    assert Path(fixed_config["usd_path"]).is_absolute()
    assert fixed_config["random_environment_base_usd_path"].endswith(
        "/configs/isaacsim/demo_room.usda"
    )
    assert Path(fixed_config["random_environment_base_usd_path"]).is_absolute()
    no_shield = json.loads(by_variant["no_shield"].config_path.read_text())
    assert no_shield["shield_transmission_target"] == 1.0
    assert no_shield["shield_thickness_scale"] == 0.0
    assert no_shield["orientation_k"] == 1
    assert no_shield["min_rotations_per_pose"] == 1
    assert no_shield["dss_pp"]["program_length"] == 1
    assert no_shield["dss_pp"]["residual_program_length"] == 1
    obstacle_off = json.loads(
        by_variant["no_obstacle_signature"].config_path.read_text()
    )
    assert obstacle_off["dss_pp"]["environment_signature_weight"] == 0.0
    assert obstacle_off["dss_pp"]["occlusion_boundary_weight"] == 0.0
    assert obstacle_off["dss_pp"]["vertical_environment_signature_weight"] == 0.0
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
    assert no_birth.get("pf_max_sources") is None
    assert no_birth["birth_residual_always_try"] is False
    assert no_birth["birth_global_rescue_enable"] is False
    assert no_birth["report_mle_rescue_enable"] is False
    assert no_birth["runtime_report_rescue_enable"] is False
    assert no_birth["runtime_report_rescue_candidate_weight"] == 0.0
    assert no_birth["runtime_report_rescue_memory_enable"] is False
    passive_no_shield = json.loads(
        by_variant["baseline_passive_no_shield"].config_path.read_text()
    )
    assert passive_no_shield["shield_transmission_target"] == 1.0
    assert passive_no_shield["shield_thickness_scale"] == 0.0
    assert passive_no_shield["orientation_k"] == 1
    assert passive_no_shield["min_rotations_per_pose"] == 1
    assert passive_no_shield["dss_pp"]["program_length"] == 1
    assert passive_no_shield["dss_pp"]["residual_program_length"] == 1
    assert passive_no_shield["baseline_path_policy"]["name"] == "passive_serpentine"
    assert passive_no_shield["baseline_shield_policy"]["name"] == "fixed"
    assert passive_no_shield["thread_count"] >= 1
    assert passive_no_shield["python_worker_count"] >= 1
    assert passive_no_shield["pose_selection_workers"] >= 1
    assert passive_no_shield["parallel_isotope_updates"] is False
    assert passive_no_shield["parallel_isotope_workers"] >= 1
    assert passive_no_shield["dss_pp"]["program_eval_workers"] >= 1
    passive_equal_time = json.loads(
        by_variant["baseline_passive_equal_time_no_shield"].config_path.read_text()
    )
    assert passive_equal_time["shield_transmission_target"] == 1.0
    assert passive_equal_time["shield_thickness_scale"] == 0.0
    assert passive_equal_time["orientation_k"] == proposed_config["orientation_k"]
    assert (
        passive_equal_time["min_rotations_per_pose"]
        == proposed_config["min_rotations_per_pose"]
    )
    assert (
        passive_equal_time["dss_pp"]["program_length"]
        == proposed_config["dss_pp"]["program_length"]
    )
    assert (
        passive_equal_time["dss_pp"]["residual_program_length"]
        == proposed_config["dss_pp"]["residual_program_length"]
    )
    assert passive_equal_time["baseline_path_policy"]["name"] == "passive_serpentine"
    assert passive_equal_time["baseline_shield_policy"]["name"] == "fixed"
    eig_only = json.loads(by_variant["eig_only_path"].config_path.read_text())
    assert eig_only["dss_pp"]["signature_weight"] == 0.0
    assert eig_only["dss_pp"]["temporal_separation_weight"] == 0.0
    assert eig_only["dss_pp"]["environment_signature_weight"] == 0.0
    assert eig_only["dss_pp"]["elevation_signature_weight"] == 0.0
    assert eig_only["dss_pp"]["correlation_reduction_weight"] == 0.0
    assert eig_only["dss_pp"]["same_isotope_direct_separation_guard"] is False
    no_verification = json.loads(by_variant["no_verification"].config_path.read_text())
    assert no_verification["pseudo_source_verification_enable"] is False
    assert no_verification["source_prune_refit_after_remove"] is False
    assert no_verification["report_strength_refit_preserve_cardinality"] is True
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
    assert "one_step_pose_eval_use_gpu" not in one_step_fixed
    one_step_no_shield = json.loads(
        by_variant["baseline_onestep_no_shield"].config_path.read_text()
    )
    assert one_step_no_shield["path_planner"] == "one_step"
    assert "one_step_pose_eval_use_gpu" not in one_step_no_shield
    assert one_step_no_shield["shield_transmission_target"] == 1.0
    assert one_step_no_shield["shield_thickness_scale"] == 0.0
    assert one_step_no_shield["orientation_k"] == 1
    assert one_step_no_shield["min_rotations_per_pose"] == 1
    assert one_step_no_shield["dss_pp"]["program_length"] == 1
    assert one_step_no_shield["dss_pp"]["residual_program_length"] == 1
    one_step_path = json.loads(
        by_variant["one_step_path"].config_path.read_text()
    )
    assert one_step_path["path_planner"] == "one_step"
    assert one_step_path["strict_planned_shield_program"] is True
    assert "one_step_pose_eval_use_gpu" not in one_step_path
    assert one_step_path["orientation_k"] == proposed_config["orientation_k"]
    assert (
        one_step_path["min_rotations_per_pose"]
        == proposed_config["min_rotations_per_pose"]
    )
    assert (
        one_step_path["dss_pp"]["program_length"]
        == proposed_config["dss_pp"]["program_length"]
    )
    assert (
        one_step_path["dss_pp"]["residual_program_length"]
        == proposed_config["dss_pp"]["residual_program_length"]
    )
    assert "baseline_shield_policy" not in one_step_path
    source_payload = json.loads(by_variant["proposed"].source_path.read_text())
    assert len(source_payload["sources"]) == DEFAULT_ABLATION_CASES[0].source_count
    isotope_counts: dict[str, int] = {}
    for source in source_payload["sources"]:
        isotope_counts[source["isotope"]] = isotope_counts.get(source["isotope"], 0) + 1
    assert isotope_counts == {"Cs-137": 4, "Co-60": 3, "Eu-154": 2}
    assert source_payload["metadata"]["visibility_filter"] is True
    assert proposed_config.get("pf_max_sources") is None
    assert proposed_config.get("init_num_sources_max") is None
    assert proposed_config["measurement_log_output_dir"] == (
        "results/ral_ablation/measurement_logs/"
        "mix9_multi_isotope_cardinality_proposed_seed_1234"
    )
    assert proposed_config["measurement_log_run_id"] == (
        "mix9_multi_isotope_cardinality_proposed_seed_1234"
    )
    measurement_log_targets = {
        json.loads(entry.config_path.read_text())["measurement_log_output_dir"]
        for entry in entries
    }
    assert len(measurement_log_targets) == len(entries)
    assert "--full-simulation" in by_variant["proposed"].command
    assert "--max-sources" not in by_variant["proposed"].command
    assert "--adaptive-dwell" not in by_variant["proposed"].command
    assert "--measurement-time-s" in by_variant["proposed"].command
    measurement_time_idx = (
        by_variant["proposed"].command.index("--measurement-time-s") + 1
    )
    assert by_variant["proposed"].command[measurement_time_idx] == (
        f"{DEFAULT_MEASUREMENT_TIME_S:g}"
    )
    assert "--rotation-overhead-s" in by_variant[
        "baseline_passive_no_shield_single_view"
    ].command
    assert f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}" in by_variant[
        "baseline_passive_no_shield_single_view"
    ].command


def test_cs4_feature_validation_plan_generates_feature_toggles(tmp_path) -> None:
    """Cs4 validation plan should compare feature toggles on one source layout."""
    manifest_path, script_path = build_cs4_feature_validation_plan(
        output_dir=tmp_path,
        seeds=(2026051001,),
    )
    assert manifest_path.exists()
    assert script_path.exists()
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 6
    manifest = {
        line.split(",", maxsplit=5)[1]: line.split(",", maxsplit=5)
        for line in rows[1:]
    }
    assert set(manifest) == {
        "feature_all_on",
        "no_dynamic_particle_allocation",
        "no_condition_planning",
        "no_recovery_verification_modes",
        "no_orthogonal_birth",
    }
    all_on_config = json.loads(Path(manifest["feature_all_on"][3]).read_text())
    no_dynamic_config = json.loads(
        Path(manifest["no_dynamic_particle_allocation"][3]).read_text()
    )
    no_condition_config = json.loads(
        Path(manifest["no_condition_planning"][3]).read_text()
    )
    no_recovery_config = json.loads(
        Path(manifest["no_recovery_verification_modes"][3]).read_text()
    )
    no_orthogonal_config = json.loads(
        Path(manifest["no_orthogonal_birth"][3]).read_text()
    )
    source_payload = json.loads(Path(manifest["feature_all_on"][4]).read_text())
    assert len(source_payload["sources"]) == 4
    assert {source["isotope"] for source in source_payload["sources"]} == {"Cs-137"}
    assert all_on_config["candidate_isotopes"] == ["Cs-137"]
    assert no_dynamic_config["candidate_isotopes"] == ["Cs-137"]
    assert no_condition_config["candidate_isotopes"] == ["Cs-137"]
    assert no_recovery_config["candidate_isotopes"] == ["Cs-137"]
    assert no_orthogonal_config["candidate_isotopes"] == ["Cs-137"]
    assert all_on_config["birth_orthogonalize_residual_candidates"] is True
    assert all_on_config["mode_preserving_dynamic_cardinality_allocation"] is True
    assert no_dynamic_config["mode_preserving_dynamic_cardinality_allocation"] is False
    assert no_condition_config["dss_pp"]["station_condition_weight"] == 0.0
    assert no_condition_config["dss_pp"]["elevation_condition_weight"] == 0.0
    assert no_recovery_config["dss_pp"]["include_runtime_rescue_modes"] is False
    assert no_recovery_config["remaining_measurement_estimate"][
        "verification_weight"
    ] == 0.0
    assert no_orthogonal_config["birth_orthogonalize_residual_candidates"] is False


def test_mix9_feature_validation_plan_uses_all_candidate_isotopes(
    tmp_path,
) -> None:
    """Mix9 validation plan should keep all isotope channels as PF candidates."""
    manifest_path, script_path = build_feature_validation_plan(
        case_name="mix9",
        output_dir=tmp_path,
        seeds=(2026051001,),
    )
    assert manifest_path.exists()
    assert script_path.exists()
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 6
    manifest = {
        line.split(",", maxsplit=5)[1]: line.split(",", maxsplit=5)
        for line in rows[1:]
    }
    all_on_config = json.loads(Path(manifest["feature_all_on"][3]).read_text())
    source_payload = json.loads(Path(manifest["feature_all_on"][4]).read_text())

    assert all_on_config["candidate_isotopes"] == ["Cs-137", "Co-60", "Eu-154"]
    assert len(source_payload["sources"]) == 9
    assert {source["isotope"] for source in source_payload["sources"]} == {
        "Cs-137",
        "Co-60",
        "Eu-154",
    }
