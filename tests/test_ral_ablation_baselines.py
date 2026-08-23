"""Tests for RA-L policy baselines and shared-runtime plan generation."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import stat

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
    resolve_pf_seeds,
    resolve_transport_seeds,
    write_ablation_plan,
)
from baselines.ral_ablation.control_policy import load_ral_control_policy
from baselines.ral_ablation.live_controller import main as live_controller_main
from baselines.ral_ablation.session_runner import _controller_command
from baselines.ral_ablation.path_policies import (
    resolve_rotation_limit_for_active_program,
    select_baseline_next_pose,
)
from baselines.ral_ablation.shield_policies import select_baseline_shield_program
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.estimator import RotatingShieldPFConfig
from pf.particle_filter import PFConfig
from pf.profiles import enforce_pure_runtime_settings


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
    """Recorded live seeds may be repeated but must remain unique per batch."""
    assert resolve_ablation_seeds((1234, 5678)) == (1234, 5678)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_ablation_seeds((1234, 1234))


def test_pf_seeds_are_independent_from_private_scene_seeds() -> None:
    """Estimator randomness must not alias deterministic truth generation."""
    assert resolve_pf_seeds((1234,), (5678,)) == (5678,)
    with pytest.raises(ValueError, match="independent"):
        resolve_pf_seeds((1234,), (1234,))


@pytest.mark.parametrize(
    "field_name",
    ("metadata", "baseline_path_policy", "baseline_shield_policy"),
)
def test_pf_config_rejects_experiment_adapter_fields(field_name: str) -> None:
    """Experiment metadata and baseline policies cannot enter generic PF config."""
    with pytest.raises(ValueError, match="Experiment-only"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                field_name: {},
            }
        )


def test_transport_seeds_are_independent_from_scene_and_pf() -> None:
    """Logged transport replay must not reveal the private scene seed."""
    assert resolve_transport_seeds((1234,), (5678,), (8765,)) == (8765,)
    with pytest.raises(ValueError, match="independent"):
        resolve_transport_seeds((1234,), (5678,), (1234,))


def test_ral_controller_process_receives_no_private_scene_inputs(
    tmp_path: Path,
) -> None:
    """The process executing PF must be launchable from truth-free arguments."""
    command = _controller_command(
        socket_path=tmp_path / "runtime.sock",
        runtime_root=tmp_path / "runtime",
        pf_config_path=tmp_path / "pf.json",
        control_policy_path=tmp_path / "policy.json",
        pf_output_dir=tmp_path / "output",
        pf_seed=5678,
    )
    rendered = " ".join(command)

    assert "private-scenario" not in rendered
    assert "source-profile" not in rendered
    assert "scene-seed" not in rendered
    assert "truth-manifest" not in rendered
    assert "runtime.sock" in rendered


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
        pf_seeds=(5678,),
        transport_seeds=(8765,),
        batch_ids=("opaque001",),
    )
    assert [entry.variant for entry in entries] == [
        "proposed",
        "baseline_passive_equal_time_no_shield",
        "round_robin_shield",
        "eig_only_path",
    ]
    assert not (output_dir / "sources").exists()
    assert all(not entry.scenario_path.exists() for entry in entries)
    assert all(not entry.truth_manifest_path.exists() for entry in entries)
    assert all(entry.scenario_path.is_relative_to(private_root) for entry in entries)
    assert all(
        entry.truth_manifest_path.is_relative_to(private_root) for entry in entries
    )
    assert len({entry.measurement_log_path for entry in entries}) == 4
    assert len({entry.pf_output_dir for entry in entries}) == 4
    assert {entry.seed_policy for entry in entries} == {"explicit_live_repeat"}

    by_variant = {entry.variant: entry for entry in entries}
    for entry in entries:
        pf_config = json.loads(entry.pf_config_path.read_text(encoding="utf-8"))
        assert pf_config["pure_pf_schema_version"] == 1
        assert pf_config["estimator_profile"] == "pf_strict"
        assert pf_config["variable_cardinality"] is True
        assert "metadata" not in pf_config
        assert "baseline_path_policy" not in pf_config
        assert "baseline_shield_policy" not in pf_config
        serialized_pf = json.dumps(pf_config, sort_keys=True)
        assert "1234" not in serialized_pf
        assert "ral-mix9" not in serialized_pf
        assert "backend" not in pf_config
        assert "shield_thickness_scale" not in pf_config
        assert "1234" not in entry.pf_config_path.name
        assert "mix9" not in entry.pf_config_path.name
        assert entry.pf_seed == 5678

        runtime_config = load_runtime_config(entry.runtime_config_path)
        assert runtime_config["backend"] == "geant4"
        assert runtime_config["engine_mode"] == "external"
        assert runtime_config["primary_sampling_fraction"] == pytest.approx(1.0)
        assert runtime_config["accelerated_weighted_transport_enable"] is False
        assert runtime_config["target_sampled_primaries"] is None
        assert runtime_config["random_seed_base"] == 8765
        assert runtime_config["random_seed_base"] != entry.scene_seed

        assert "generate-ral-scenario" in entry.scenario_command
        assert "--scene-seed" in entry.scenario_command
        assert "--truth-manifest-output" in entry.scenario_command
        assert "baselines.ral_ablation.session_runner" in entry.session_command
        assert "--scene-seed" not in entry.session_command
        assert "--source-profile" not in entry.session_command
        assert "--private-scene-profile" not in entry.session_command
        assert entry.source_profile not in entry.session_command
        assert "--full-simulation" not in entry.scenario_command
        assert "main.py" not in entry.session_command

    passive_policy = json.loads(
        by_variant[
            "baseline_passive_equal_time_no_shield"
        ].control_policy_path.read_text(encoding="utf-8")
    )
    assert passive_policy["path_policy"]["name"] == "passive_serpentine"
    assert passive_policy["shield_policy"]["name"] == "fixed"
    load_ral_control_policy(
        by_variant["baseline_passive_equal_time_no_shield"].control_policy_path
    )
    passive_runtime = load_runtime_config(
        by_variant["baseline_passive_equal_time_no_shield"].runtime_config_path
    )
    assert passive_runtime["shield_thickness_scale"] == pytest.approx(0.0)
    assert passive_runtime["shield_transmission_target"] == pytest.approx(1.0)

    round_robin = json.loads(
        by_variant["round_robin_shield"].control_policy_path.read_text(encoding="utf-8")
    )
    assert round_robin["shield_policy"]["name"] == "round_robin"
    eig_only = json.loads(
        by_variant["eig_only_path"].pf_config_path.read_text(encoding="utf-8")
    )
    assert eig_only["dss_pp"]["coverage_weight"] == pytest.approx(0.0)

    manifest_path, script_path = write_ablation_plan(
        entries,
        private_root=private_root,
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["source_profile"] == "ral-mix9"
    assert manifest_path.is_relative_to(private_root)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(script_path.stat().st_mode) == 0o700
    script = script_path.read_text(encoding="utf-8")
    assert script.count("generate-ral-scenario") == 4
    assert script.count("baselines.ral_ablation.session_runner") == 4
    assert "--full-simulation" not in script

    controller_source = inspect.getsource(live_controller_main)
    assert "--scenario" not in controller_source
    assert "--source-profile" not in controller_source
    assert "--scene-seed" not in controller_source


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
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_pf_seed",
        lambda: 975318642,
    )
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_batch_id",
        lambda: "opaque002",
    )
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_transport_seed",
        lambda: 864297531,
    )
    entries = build_ablation_plan(
        output_dir=tmp_path / "results",
        private_root=tmp_path / "private",
        seeds=None,
        cases=DEFAULT_ABLATION_CASES,
        variants=DEFAULT_ABLATION_VARIANTS,
    )
    assert {entry.scene_seed for entry in entries} == {246813579}
    assert {entry.pf_seed for entry in entries} == {975318642}
    assert {entry.transport_seed for entry in entries} == {864297531}
    assert {entry.batch_id for entry in entries} == {"opaque002"}
    assert {entry.seed_policy for entry in entries} == {"fresh_per_batch"}
    assert len(entries) == 4
