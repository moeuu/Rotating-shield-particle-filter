"""Tests for RA-L ablation baseline utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.ral_ablation import config_factory
from baselines.ral_ablation.config_factory import (
    DEFAULT_ABLATION_CASES,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_BASE_CONFIG,
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    _load_json,
    _source_generation_options,
    _validate_ral_transport_sampling,
    _variant_config,
    build_ablation_plan,
    resolve_ablation_seeds,
    write_ablation_plan,
)
from baselines.ral_ablation.path_policies import (
    resolve_rotation_limit_for_active_program,
    select_baseline_next_pose,
)
from baselines.ral_ablation.shield_policies import select_baseline_shield_program
from measurement.source_boundary import (
    SURFACE_SOURCE_RUNTIME_KEYS,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.estimator import RotatingShieldPFConfig
from pf.particle_filter import PFConfig
from pf.runtime_defaults import (
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_NO_ROTATION_OVERHEAD_S,
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


def test_ral_source_generation_defaults_are_unconditioned() -> None:
    """Source generation should expose physical geometry options only."""
    options = _source_generation_options({})

    assert options["obstacle_height_m"] == pytest.approx(2.0)
    assert options["include_room_boundaries"] is False
    assert options["room_boundary_thickness_m"] == pytest.approx(0.1)
    assert options["structural_rj_surface_chart_max_edge_m"] == pytest.approx(
        1.0
    )


def test_fresh_ablation_seed_is_generated_when_seeds_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new comparison batch should not silently reuse a historical scene."""
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_ablation_seed",
        lambda: 987654321,
    )

    assert resolve_ablation_seeds(None) == (987654321,)


def test_explicit_ablation_seed_remains_available_for_exact_replay() -> None:
    """Explicit seeds should retain deterministic replay semantics."""
    assert resolve_ablation_seeds((1234, 5678)) == (1234, 5678)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_ablation_seeds((1234, 1234))


@pytest.mark.parametrize(
    "removed_key",
    (
        "random_source_preferred_max_z_m",
        "random_source_max_ceiling_sources",
        "random_source_visibility_filter",
        "random_source_response_observability_filter",
        "random_source_same_isotope_min_distance_m",
    ),
)
def test_ral_source_generation_rejects_removed_selection_keys(
    removed_key: str,
) -> None:
    """Legacy truth-selection knobs should fail before generating a layout."""
    with pytest.raises(ValueError, match="were removed"):
        _source_generation_options({removed_key: None})


def test_ral_transport_sampling_requires_full_unit_weight_histories() -> None:
    """RA-L generation should label only full unit-weight native histories."""
    assert _validate_ral_transport_sampling({}) == "full_unit_weight"


@pytest.mark.parametrize(
    "override",
    [
        {"backend": "analytic"},
        {"engine_mode": "in_process"},
    ],
)
def test_ral_variant_rejects_non_native_backend(
    override: dict[str, object],
) -> None:
    """Paper variants cannot switch away from native external Geant4."""
    base_config = _load_json(DEFAULT_BASE_CONFIG)
    base_config.update(override)

    with pytest.raises(ValueError, match="backend='geant4'"):
        _variant_config(
            base_config,
            base_config_path=DEFAULT_BASE_CONFIG,
            case=DEFAULT_ABLATION_CASES[0],
            variant=DEFAULT_ABLATION_VARIANTS[0],
            seed=1234,
            output_tag="invalid_backend",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"primary_sampling_fraction": 0.02}, "primary_sampling_fraction=1.0"),
        (
            {"accelerated_weighted_transport_enable": True},
            "accelerated_weighted_transport_enable=false",
        ),
        ({"target_sampled_primaries": 1_500_000}, "target_sampled_primaries=null"),
        (
            {"source_rate_model": "isotropic_emission_equivalent"},
            "source_rate_model=detector_cps_1m",
        ),
    ],
)
def test_ral_transport_sampling_rejects_every_shortcut(
    override: dict[str, object],
    message: str,
) -> None:
    """Thinning, weighting, history caps, and alternate rates are all retired."""
    with pytest.raises(ValueError, match=message):
        _validate_ral_transport_sampling(override)


def test_ral_base_config_loader_exposes_and_rejects_inherited_thinning(
    tmp_path: Path,
) -> None:
    """Inherited acceleration must remain visible to the fail-closed validator."""
    parent_path = tmp_path / "parent.json"
    child_path = tmp_path / "accelerated.json"
    parent_path.write_text(
        json.dumps(
            {
                "thread_count": 32,
                "source_rate_model": "detector_cps_1m",
                "primary_sampling_fraction": 1.0,
                "full_spectrum_generative_model_path": (
                    "results/spectrum_validation/"
                    "geometry_conditioned_full_spectrum_approved.json"
                ),
                "full_spectrum_generative_model_file_sha256": "1" * 64,
                "full_spectrum_contract_hash_sha256": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        json.dumps(
            {
                "extends": "parent.json",
                "primary_sampling_fraction": 0.02,
                "accelerated_weighted_transport_enable": True,
            }
        ),
        encoding="utf-8",
    )
    resolved = _load_json(child_path)
    assert resolved["thread_count"] == 32
    assert resolved["full_spectrum_contract_hash_sha256"] == "2" * 64
    with pytest.raises(ValueError, match="primary_sampling_fraction=1.0"):
        _validate_ral_transport_sampling(resolved)


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
        resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy={"name": "round_robin"},
        )
        == 2
    )
    assert (
        resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=True,
            baseline_shield_policy=None,
        )
        == 2
    )
    assert (
        resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy=None,
        )
        == 8
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
    """The factory should emit only the four declared paper configurations."""
    entries = build_ablation_plan(
        output_dir=tmp_path,
        seeds=(1234,),
        cases=DEFAULT_ABLATION_CASES[:1],
        variants=DEFAULT_ABLATION_VARIANTS,
        intensity_cps_1m=30000.0,
    )
    by_variant = {entry.variant: entry for entry in entries}
    assert tuple(by_variant) == (
        "proposed",
        "baseline_passive_equal_time_no_shield",
        "round_robin_shield",
        "eig_only_path",
    )
    assert len(entries) == 4

    proposed_config = json.loads(by_variant["proposed"].config_path.read_text())
    round_robin = json.loads(by_variant["round_robin_shield"].config_path.read_text())
    assert proposed_config["primary_sampling_fraction"] == 1.0
    assert proposed_config["thread_count"] > 1
    assert "response_poisson_low_snr_suppress_count" not in proposed_config
    assert "precision_diagnostic_particle_log_limit" not in proposed_config
    assert (
        "precision_diagnostic_full_spectrum_response_enable"
        not in proposed_config
    )
    assert proposed_config["surface_observability_diagnostic_candidates"] == 0
    for entry in entries:
        generated = json.loads(entry.config_path.read_text())
        assert generated["backend"] == "geant4"
        assert generated["engine_mode"] == "external"
        assert generated["pure_pf_schema_version"] == 1
        assert generated["estimator_profile"] == "pf_strict"
        assert generated["variable_cardinality"] is True
        assert generated["primary_sampling_fraction"] == pytest.approx(1.0)
        assert generated["accelerated_weighted_transport_enable"] is False
        assert generated["target_sampled_primaries"] is None
        assert generated["pf_strength_prior_min_cps_1m"] == pytest.approx(300000.0)
        assert generated["pf_strength_prior_max_cps_1m"] == pytest.approx(2000000.0)
        assert float(generated["structural_rj_surface_chart_max_edge_m"]) > 0.0
        assert float(generated["structural_rj_move_probability"]) > 0.0
        assert float(generated["structural_cardinality_prior_mean"]) > 0.0
        generated_metadata = generated["metadata"]
        assert generated_metadata["ral_transport_history_mode"] == (
            "full_unit_weight"
        )
        assert generated_metadata["ral_accelerated_transport"] is False
        assert generated_metadata["ral_primary_sampling_fraction"] == pytest.approx(
            1.0
        )
        assert generated_metadata["ral_primary_history_weight"] == pytest.approx(1.0)
        assert generated_metadata["ral_target_sampled_primaries"] is None

    assert round_robin["orientation_k"] == proposed_config["orientation_k"]
    assert (
        round_robin["min_rotations_per_pose"]
        == proposed_config["min_rotations_per_pose"]
    )
    assert (
        round_robin["dss_pp"]["program_length"]
        == proposed_config["dss_pp"]["program_length"]
    )
    assert round_robin["strict_planned_shield_program"] is True
    assert round_robin["baseline_shield_policy"]["name"] == "round_robin"
    assert "baseline_path_policy" not in round_robin

    assert proposed_config["cui_split_view_dir"] == DEFAULT_CUI_SPLIT_VIEW_DIR
    assert proposed_config["usd_path"].endswith("/configs/isaacsim/demo_room.usda")
    assert Path(proposed_config["usd_path"]).is_absolute()
    assert proposed_config["random_environment_base_usd_path"].endswith(
        "/configs/isaacsim/demo_room.usda"
    )
    assert Path(proposed_config["random_environment_base_usd_path"]).is_absolute()

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
    assert passive_equal_time["baseline_path_policy"]["name"] == "passive_serpentine"
    assert passive_equal_time["baseline_shield_policy"]["name"] == "fixed"
    assert passive_equal_time["thread_count"] >= 1
    assert passive_equal_time["python_worker_count"] >= 1
    assert passive_equal_time["pose_selection_workers"] >= 1

    eig_only = json.loads(by_variant["eig_only_path"].config_path.read_text())
    for key in (
        "coverage_weight",
        "bearing_diversity_weight",
        "frontier_weight",
        "local_orbit_weight",
        "elevation_condition_weight",
        "revisit_penalty_weight",
        "turn_smoothness_weight",
    ):
        assert eig_only["dss_pp"][key] == 0.0

    source_payload = json.loads(by_variant["proposed"].source_path.read_text())
    assert len(source_payload["sources"]) == DEFAULT_ABLATION_CASES[0].source_count
    isotope_counts: dict[str, int] = {}
    for source in source_payload["sources"]:
        isotope_counts[source["isotope"]] = isotope_counts.get(source["isotope"], 0) + 1
    assert isotope_counts == {"Cs-137": 4, "Co-60": 3, "Eu-154": 2}
    source_metadata = source_payload["metadata"]
    assert source_metadata["source_seed"] == 1251
    assert source_metadata["obstacle_seed"] == 1234
    assert source_metadata["scene_seed_policy"] == "explicit_replay"
    assert source_metadata["source_surface_sampling_schema_version"] == 3
    assert {
        "sampling": source_metadata["sampling"],
        "sampling_measure": source_metadata["sampling_measure"],
        "surface_geometry": source_metadata["surface_geometry"],
        "selection_conditioning": source_metadata["selection_conditioning"],
    } == {
        "sampling": "continuous area-uniform physical-surface placement",
        "sampling_measure": "continuous_area_uniform",
        "surface_geometry": "runtime_transport_component_union",
        "selection_conditioning": "none_physical_area_only",
    }
    assert source_metadata["obstacle_height_m"] == pytest.approx(2.0)
    assert source_metadata["include_room_boundaries"] is True
    assert (
        source_metadata["surface_emission_policy_sha256"]
        == surface_emission_policy_sha256()
    )
    assert len(source_metadata["surface_atlas_contract_sha256"]) == 64
    assert all(
        set(source) == SURFACE_SOURCE_RUNTIME_KEYS
        for source in source_payload["sources"]
    )
    assert (
        source_metadata["surface_source_runtime_contract_sha256"]
        == surface_source_runtime_contract_sha256(source_payload["sources"])
    )
    removed_truth_selection_keys = {
        "random_source_preferred_max_z_m",
        "random_source_max_ceiling_sources",
        "random_source_visibility_filter",
        "random_source_response_observability_filter",
        "random_source_same_isotope_min_distance_m",
    }
    assert removed_truth_selection_keys.isdisjoint(source_metadata)
    position_values = [
        float(value)
        for source in source_payload["sources"]
        for value in source["position"]
    ]
    assert any(
        abs(value - round(value, 6)) > 1e-12
        for value in position_values
    )

    assert proposed_config["pf_max_sources"] == 5
    assert proposed_config["structural_cardinality_prior_mean"] == pytest.approx(2.0)
    assert proposed_config["measurement_log_output_dir"] == (
        "results/ral_ablation/measurement_logs/"
        "mix9_multi_isotope_cardinality_proposed_seed_1234"
    )
    assert proposed_config["measurement_log_run_id"] == (
        "mix9_multi_isotope_cardinality_proposed_seed_1234"
    )
    assert proposed_config["metadata"]["ral_environment_seed"] == 1234
    assert "ral_truth_source_seed" not in proposed_config["metadata"]
    assert proposed_config["metadata"]["ral_scene_seed_policy"] == (
        "explicit_replay"
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
    assert (
        "--rotation-overhead-s"
        in by_variant["baseline_passive_equal_time_no_shield"].command
    )
    assert (
        f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"
        in by_variant["baseline_passive_equal_time_no_shield"].command
    )

    manifest_path, _ = write_ablation_plan(entries, output_dir=tmp_path)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "source_seed,seed_policy" in manifest_text.splitlines()[0]
    assert ",1234,1251,explicit_replay," in manifest_text


def test_ablation_plan_default_uses_one_fresh_scene_for_all_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All methods should share one new scene within a generated batch."""
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_ablation_seed",
        lambda: 246813579,
    )

    entries = build_ablation_plan(
        output_dir=tmp_path,
        seeds=None,
        cases=DEFAULT_ABLATION_CASES[:1],
        variants=DEFAULT_ABLATION_VARIANTS,
        intensity_cps_1m=30000.0,
    )

    assert {entry.seed for entry in entries} == {246813579}
    assert {entry.source_seed for entry in entries} == {246813596}
    assert {entry.seed_policy for entry in entries} == {"fresh_per_batch"}
    assert len({entry.source_path for entry in entries}) == 1
    for entry in entries:
        config = json.loads(entry.config_path.read_text(encoding="utf-8"))
        assert config["metadata"]["ral_scene_seed_policy"] == "fresh_per_batch"
