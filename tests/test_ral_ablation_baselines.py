"""Tests for RA-L policy baselines and shared-runtime plan generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sim.runtime import load_runtime_config

from baselines.ral_ablation import config_factory
from baselines.ral_ablation.config_factory import (
    DEFAULT_ABLATION_CASES,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_RUNTIME_ROOT,
    build_ablation_plan,
    resolve_ablation_seeds,
    write_ablation_plan,
)
from baselines.ral_ablation.path_policies import (
    resolve_rotation_limit_for_active_program,
    select_baseline_next_pose,
)
from baselines.ral_ablation.shield_policies import select_baseline_shield_program
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.estimator import RotatingShieldPFConfig
from pf.particle_filter import PFConfig


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
    """Round-robin ablation should produce deterministic programs."""
    program = select_baseline_shield_program(
        {"name": "round_robin", "start_pair_id": 2, "advance_by_pose": True},
        total_pairs=8,
        program_length=4,
        pose_index=1,
    )
    assert program is not None
    assert program.pair_ids == (6, 7, 0, 1)


def test_pf_max_sources_default_is_shared() -> None:
    """PF entry points should use one shared source-count support."""
    assert RotatingShieldPFConfig().max_sources == DEFAULT_MAX_SOURCES_PER_ISOTOPE
    assert PFConfig().max_sources == DEFAULT_MAX_SOURCES_PER_ISOTOPE


def test_fresh_ablation_seed_is_generated_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new comparison batch must not silently reuse a historical scene."""
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_ablation_seed",
        lambda: 987654321,
    )
    assert resolve_ablation_seeds(None) == (987654321,)


def test_explicit_ablation_seeds_reject_duplicates() -> None:
    """Explicit seeds should retain deterministic replay semantics."""
    assert resolve_ablation_seeds((1234, 5678)) == (1234, 5678)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_ablation_seeds((1234, 1234))


def test_explicit_shield_program_rotation_limit_is_strict() -> None:
    """Baseline shield programs should not be padded by adaptive selection."""
    assert (
        resolve_rotation_limit_for_active_program(
            base_rotation_limit=8,
            active_shield_program=(2, 3),
            strict_planned_shield_program=False,
            baseline_shield_policy={"name": "round_robin"},
        )
        == 2
    )


def test_passive_serpentine_selects_candidate_near_waypoint() -> None:
    """Passive path baseline should select by geometry, not PF information."""
    candidates = np.asarray(
        [[9.0, 0.0, 0.5], [0.0, 10.0, 0.5], [8.0, 20.0, 0.5]],
        dtype=float,
    )
    selection = select_baseline_next_pose(
        {"name": "passive_serpentine", "row_count": 3},
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.asarray([1.0, 1.0, 0.5]),
        visited_poses_xyz=np.asarray([[1.0, 1.0, 0.5]]),
        bounds_xyz=(np.asarray([0.0, 0.0, 0.5]), np.asarray([10.0, 20.0, 0.5])),
    )
    assert selection is not None
    assert selection.candidate_index == 1


def test_ablation_plan_separates_pf_runtime_and_private_truth(tmp_path: Path) -> None:
    """The factory should emit four causal trials without local truth files."""
    output_dir = tmp_path / "public-results"
    private_root = tmp_path / "private-runtime"
    entries = build_ablation_plan(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        output_dir=output_dir,
        private_root=private_root,
        seeds=(1234,),
    )
    assert [entry.variant for entry in entries] == [
        "proposed",
        "baseline_passive_equal_time_no_shield",
        "round_robin_shield",
        "eig_only_path",
    ]
    assert not (output_dir / "sources").exists()
    assert all(not entry.scenario_path.exists() for entry in entries)
    assert all(entry.scenario_path.is_relative_to(private_root) for entry in entries)
    assert len({entry.measurement_log_path for entry in entries}) == 4
    assert len({entry.pf_output_dir for entry in entries}) == 4

    by_variant = {entry.variant: entry for entry in entries}
    for entry in entries:
        pf_config = json.loads(entry.pf_config_path.read_text(encoding="utf-8"))
        assert pf_config["pure_pf_schema_version"] == 1
        assert pf_config["estimator_profile"] == "pf_strict"
        assert pf_config["variable_cardinality"] is True
        assert pf_config["metadata"]["ral_scene_seed"] == 1234
        assert "backend" not in pf_config
        assert "shield_thickness_scale" not in pf_config

        runtime_config = load_runtime_config(entry.runtime_config_path)
        assert runtime_config["backend"] == "geant4"
        assert runtime_config["engine_mode"] == "external"
        assert runtime_config["primary_sampling_fraction"] == pytest.approx(1.0)
        assert runtime_config["accelerated_weighted_transport_enable"] is False
        assert runtime_config["target_sampled_primaries"] is None

        assert "generate-ral-scenario" in entry.scenario_command
        assert "--scene-seed" in entry.scenario_command
        assert "rotating-shield-pf-live" in entry.pf_command
        assert "--full-simulation" not in entry.scenario_command
        assert "main.py" not in entry.pf_command

    passive_pf = json.loads(
        by_variant["baseline_passive_equal_time_no_shield"].pf_config_path.read_text(
            encoding="utf-8"
        )
    )
    assert passive_pf["baseline_path_policy"]["name"] == "passive_serpentine"
    assert passive_pf["baseline_shield_policy"]["name"] == "fixed"
    passive_runtime = load_runtime_config(
        by_variant["baseline_passive_equal_time_no_shield"].runtime_config_path
    )
    assert passive_runtime["shield_thickness_scale"] == pytest.approx(0.0)
    assert passive_runtime["shield_transmission_target"] == pytest.approx(1.0)

    round_robin = json.loads(
        by_variant["round_robin_shield"].pf_config_path.read_text(encoding="utf-8")
    )
    assert round_robin["baseline_shield_policy"]["name"] == "round_robin"
    eig_only = json.loads(
        by_variant["eig_only_path"].pf_config_path.read_text(encoding="utf-8")
    )
    assert eig_only["dss_pp"]["coverage_weight"] == pytest.approx(0.0)

    manifest_path, script_path = write_ablation_plan(entries, output_dir=output_dir)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["source_profile"] == "ral-mix9"
    script = script_path.read_text(encoding="utf-8")
    assert script.count("generate-ral-scenario") == 4
    assert script.count("rotating-shield-pf-live") == 4
    assert "--full-simulation" not in script


def test_ablation_plan_default_shares_one_new_scene_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All methods should share one new scene seed within a batch."""
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_ablation_seed",
        lambda: 246813579,
    )
    entries = build_ablation_plan(
        output_dir=tmp_path / "results",
        private_root=tmp_path / "private",
        seeds=None,
        cases=DEFAULT_ABLATION_CASES,
        variants=DEFAULT_ABLATION_VARIANTS,
    )
    assert {entry.seed for entry in entries} == {246813579}
    assert {entry.seed_policy for entry in entries} == {"fresh_per_batch"}
    assert len(entries) == 4
