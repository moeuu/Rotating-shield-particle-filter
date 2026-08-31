"""Tests for RA-L policy baselines and shared-runtime plan generation."""

from __future__ import annotations

import csv
from hashlib import sha256
import inspect
import json
from pathlib import Path
import stat

import pytest
from sim.runtime import load_production_runtime_config
from sim.shield_geometry import resolve_shield_thickness_config

from baselines.ral_ablation import config_factory
from baselines.ral_ablation.config_factory import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_RUNTIME_ROOT,
    RAL_EXPERIMENT_PROFILE_ID,
    RAL_SCENE_VARIANT_ID,
    build_ablation_plan,
    resolve_ablation_seeds,
    resolve_pf_seeds,
    resolve_transport_seeds,
    write_ablation_plan,
)
from baselines.ral_ablation.control_policy import (
    RALControlPolicy,
    RALControlPolicyError,
    load_ral_control_policy,
    load_ral_control_policy_document,
)
from baselines.ral_ablation.live_controller import (
    RAL_MINIMUM_FINALIZABLE_STATIONS,
    RALStationStopRequest,
    main as live_controller_main,
)
from baselines.ral_ablation.session_runner import _controller_command
from baselines.ral_ablation.shield_policies import select_baseline_shield_program
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.estimator import RotatingShieldPFConfig
from pf.profiles import enforce_pure_runtime_settings


def _production_pf_config() -> dict[str, object]:
    """Load a fresh complete production schema-v2 PF configuration."""
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "configs/pf/pf_strict_3d.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "policy",
    (
        {"name": "fixed", "fixed_pair_id": 7},
        {"name": "round_robin", "start_pair_id": 0},
        {"name": "round_robin", "start_pair_id": 0, "advance_by_pose": 1},
    ),
)
def test_shield_policies_do_not_fill_missing_or_coerced_fields(
    policy: dict[str, object],
) -> None:
    """Every discriminated shield-policy field must be explicit and exact."""
    with pytest.raises(ValueError):
        select_baseline_shield_program(
            policy,
            total_pairs=64,
            program_length=8,
            pose_index=0,
        )


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


@pytest.mark.parametrize(
    "raw_policy",
    (
        '{"schema_version":2,"schema_version":2,'
        '"variant":"proposed","shield_policy":null}',
        '{"schema_version":true,"variant":"proposed","shield_policy":null}',
        '{"schema_version":2.0,"variant":"proposed","shield_policy":null}',
        '{"schema_version":2,"variant":"proposed","shield_policy":NaN}',
        '{"schema_version":2,"variant":"retired_passive","shield_policy":null}',
        '{"schema_version":2,"variant":"proposed",'
        '"shield_policy":{"name":"round_robin","start_pair_id":0,'
        '"advance_by_pose":true}}',
        '{"schema_version":2,"variant":"round_robin_shield",'
        '"shield_policy":null}',
    ),
)
def test_control_policy_loader_rejects_ambiguous_or_incomplete_json(
    tmp_path: Path,
    raw_policy: str,
) -> None:
    """Policy loading must reject duplicates, coercions, defaults, and aliases."""
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(raw_policy, encoding="utf-8")

    with pytest.raises((RALControlPolicyError, TypeError, ValueError)):
        load_ral_control_policy_document(policy_path)


def test_control_policy_document_binds_source_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    """A loaded policy must retain exact bytes and a stable resolved identity."""
    policy_path = tmp_path / "policy.json"
    source = (
        b'{ "schema_version": 2, "variant": "round_robin_shield", '
        b'"shield_policy": {"name":"round_robin","start_pair_id":3,'
        b'"advance_by_pose":true} }\n'
    )
    policy_path.write_bytes(source)

    document = load_ral_control_policy_document(policy_path)
    detached = document.payload()
    detached["shield_policy"] = None

    assert document.source_bytes == source
    assert document.source_sha256 == sha256(source).hexdigest()
    assert document.canonical_sha256 == sha256(
        document.canonical_policy_json
    ).hexdigest()
    assert document.payload()["shield_policy"] == {
        "name": "round_robin",
        "start_pair_id": 3,
        "advance_by_pose": True,
    }
    assert document.policy().provenance.to_dict()["policy"] == document.payload()


def test_control_policy_cannot_self_attach_loader_provenance(tmp_path: Path) -> None:
    """Only the strict loader may bind canonical identity to executable policy."""
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "variant": "round_robin_shield",
                "shield_policy": {
                    "name": "round_robin",
                    "start_pair_id": 3,
                    "advance_by_pose": True,
                },
            }
        ),
        encoding="utf-8",
    )
    sealed = load_ral_control_policy(policy_path)

    with pytest.raises(RALControlPolicyError, match="loader token together"):
        RALControlPolicy(
            variant="round_robin_shield",
            shield_policy={
                "name": "round_robin",
                "start_pair_id": 3,
                "advance_by_pose": True,
            },
            _provenance=sealed.provenance,
        )


def test_expected_policy_digest_rejects_valid_policy_swap(tmp_path: Path) -> None:
    """Replacing one valid variant policy must fail before runtime connection."""
    expected_path = tmp_path / "expected.json"
    swapped_path = tmp_path / "swapped.json"
    expected_path.write_text(
        json.dumps(
            {"schema_version": 2, "variant": "proposed", "shield_policy": None}
        ),
        encoding="utf-8",
    )
    swapped_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "variant": "round_robin_shield",
                "shield_policy": {
                    "name": "round_robin",
                    "start_pair_id": 0,
                    "advance_by_pose": True,
                },
            }
        ),
        encoding="utf-8",
    )
    expected_digest = sha256(expected_path.read_bytes()).hexdigest()

    with pytest.raises(RALControlPolicyError, match="variant-policy digest"):
        load_ral_control_policy_document(
            swapped_path,
            expected_source_sha256=expected_digest,
        )


def test_live_controller_rejects_policy_swap_before_closed_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expected variant-policy identity must be checked before socket use."""
    from baselines.ral_ablation import live_controller

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {"schema_version": 2, "variant": "proposed", "shield_policy": None}
        ),
        encoding="utf-8",
    )

    def forbidden_closed_loop(*args: object, **kwargs: object) -> object:
        """Fail if control reaches the runtime-facing controller."""
        del args, kwargs
        raise AssertionError("closed loop must not start after a policy mismatch")

    monkeypatch.setattr(live_controller, "run_pf_closed_loop", forbidden_closed_loop)

    with pytest.raises(RALControlPolicyError, match="variant-policy digest"):
        live_controller_main(
            [
                "--session-socket",
                str(tmp_path / "runtime.sock"),
                "--runtime-root",
                str(tmp_path),
                "--cui-truth-overlay-socket",
                str(tmp_path / "cui-truth.sock"),
                "--config",
                str(tmp_path / "pf.json"),
                "--control-policy",
                str(policy_path),
                "--expected-control-policy-sha256",
                "0" * 64,
                "--output-dir",
                str(tmp_path / "output"),
                "--seed",
                "7",
            ]
        )


def test_pf_max_sources_default_is_shared() -> None:
    """PF entry points should use one shared source-count support."""
    assert RotatingShieldPFConfig().max_sources == DEFAULT_MAX_SOURCES_PER_ISOTOPE


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
    with pytest.raises(ValueError, match="at least 1"):
        resolve_ablation_seeds((0,))


def test_pf_seeds_are_independent_from_private_scene_seeds() -> None:
    """Estimator randomness must not alias deterministic truth generation."""
    assert resolve_pf_seeds((1234,), (5678,)) == (5678,)
    with pytest.raises(ValueError, match="independent"):
        resolve_pf_seeds((1234,), (1234,))
    with pytest.raises(ValueError, match="independent"):
        resolve_pf_seeds((1234, 5678), (8765, 1234))


def test_generated_pf_seeds_avoid_every_scene_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh PF generation must reject cross-batch scene-seed collisions."""
    candidates = iter((5678, 8765, 9876))
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_pf_seed",
        lambda: next(candidates),
    )
    assert resolve_pf_seeds((1234, 5678), None) == (8765, 9876)


@pytest.mark.parametrize(
    "field_name",
    ("metadata", "baseline_path_policy", "baseline_shield_policy"),
)
def test_pf_config_rejects_experiment_adapter_fields(field_name: str) -> None:
    """Experiment metadata and baseline policies cannot enter generic PF config."""
    payload = _production_pf_config()
    payload[field_name] = {}
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


def test_transport_seeds_are_independent_from_scene_and_pf() -> None:
    """Logged transport replay must not reveal the private scene seed."""
    assert resolve_transport_seeds((1234,), (5678,), (8765,)) == (8765,)
    with pytest.raises(ValueError, match="independent"):
        resolve_transport_seeds((1234,), (5678,), (1234,))
    with pytest.raises(ValueError, match="independent"):
        resolve_transport_seeds((1234, 5678), (8765, 9876), (2468, 1234))


def test_generated_transport_seeds_avoid_every_other_seed_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh transport generation must reject cross-batch seed collisions."""
    candidates = iter((5678, 8765, 2468, 3690))
    monkeypatch.setattr(
        config_factory,
        "generate_fresh_transport_seed",
        lambda: next(candidates),
    )
    assert resolve_transport_seeds((1234, 5678), (8765, 9876), None) == (
        2468,
        3690,
    )


@pytest.mark.parametrize(
    ("pf_seeds", "transport_seeds", "batch_ids"),
    [
        (None, (8765,), ("opaque001",)),
        ((5678,), None, ("opaque001",)),
        ((5678,), (8765,), None),
    ],
)
def test_explicit_live_repeat_requires_complete_recorded_provenance(
    tmp_path: Path,
    pf_seeds: tuple[int, ...] | None,
    transport_seeds: tuple[int, ...] | None,
    batch_ids: tuple[str, ...] | None,
) -> None:
    """A recorded scene cannot silently acquire new stochastic provenance."""
    with pytest.raises(ValueError, match="explicit live repeat"):
        build_ablation_plan(
            runtime_root=DEFAULT_RUNTIME_ROOT,
            runtime_config_path=DEFAULT_RUNTIME_CONFIG,
            output_dir=tmp_path / "results",
            private_root=tmp_path / "private",
            seeds=(1234,),
            pf_seeds=pf_seeds,
            transport_seeds=transport_seeds,
            batch_ids=batch_ids,
        )


def test_fresh_batch_rejects_partially_supplied_seed_streams(tmp_path: Path) -> None:
    """Fresh mode must generate all three independent seed streams together."""
    with pytest.raises(ValueError, match="omit every seed option"):
        build_ablation_plan(
            runtime_root=DEFAULT_RUNTIME_ROOT,
            runtime_config_path=DEFAULT_RUNTIME_CONFIG,
            output_dir=tmp_path / "results",
            private_root=tmp_path / "private",
            seeds=None,
            pf_seeds=(5678,),
        )


def test_ablation_plan_rejects_no_op_pf_variant(tmp_path: Path) -> None:
    """The EIG-only row must materially differ from the proposed PF config."""
    pf_config = _production_pf_config()
    dss_pp = pf_config["dss_pp"]
    assert isinstance(dss_pp, dict)
    for field in (
        "coverage_weight",
        "bearing_diversity_weight",
        "frontier_weight",
        "local_orbit_weight",
        "elevation_condition_weight",
        "revisit_penalty_weight",
        "turn_smoothness_weight",
    ):
        dss_pp[field] = 0.0
    dss_pp.update(
        {
            "coverage_floor_quantile": 0.0,
            "coverage_floor_weight": 0.0,
            "exact_eig_coverage_reserve": 0,
            "coverage_surface_max_hausdorff_m": None,
            "coverage_surface_quadrature_max_points": None,
            "local_orbit_ring_radii_m": [],
            "local_orbit_sigma_m": None,
            "elevation_pair_xy_scale_m": None,
            "elevation_pair_z_scale_m": None,
            "elevation_angle_threshold_deg": None,
        }
    )
    pf_path = tmp_path / "no-op-pf.json"
    pf_path.write_text(json.dumps(pf_config), encoding="utf-8")

    with pytest.raises(ValueError, match="no-op PF intervention"):
        build_ablation_plan(
            runtime_root=DEFAULT_RUNTIME_ROOT,
            runtime_config_path=DEFAULT_RUNTIME_CONFIG,
            pf_config_path=pf_path,
            output_dir=tmp_path / "results",
            private_root=tmp_path / "private",
            seeds=None,
        )


def test_ral_controller_process_receives_no_private_scene_inputs(
    tmp_path: Path,
) -> None:
    """The process executing PF must be launchable from truth-free arguments."""
    command = _controller_command(
        socket_path=tmp_path / "runtime.sock",
        runtime_root=tmp_path / "runtime",
        cui_truth_overlay_socket_path=tmp_path / "cui-truth.sock",
        pf_config_path=tmp_path / "pf.json",
        control_policy_path=tmp_path / "policy.json",
        expected_control_policy_sha256="a" * 64,
        pf_output_dir=tmp_path / "output",
        pf_seed=5678,
        station_stop_request_path=tmp_path / "station.stop",
    )
    rendered = " ".join(command)

    assert "private-scenario" not in rendered
    assert "scene-variant" not in rendered
    assert "scene-seed" not in rendered
    assert "truth-manifest" not in rendered
    assert "runtime.sock" in rendered
    assert "cui-truth.sock" in rendered
    assert "station.stop" in rendered


def test_ral_station_stop_request_waits_for_ten_complete_stations(
    tmp_path: Path,
) -> None:
    """An operator sentinel must finalize only at an eligible boundary."""
    path = tmp_path / "run.stop"
    request = RALStationStopRequest(
        path,
        minimum_stations=RAL_MINIMUM_FINALIZABLE_STATIONS,
    )

    assert request(9) is False
    path.touch()
    assert request(9) is False
    assert request(10) is True


def test_ral_station_stop_request_rejects_stale_or_nonempty_files(
    tmp_path: Path,
) -> None:
    """A stale or malformed stop control must never be silently accepted."""
    stale = tmp_path / "stale.stop"
    stale.touch()
    with pytest.raises(FileExistsError, match="absent"):
        RALStationStopRequest(stale, minimum_stations=10)

    malformed = tmp_path / "malformed.stop"
    request = RALStationStopRequest(malformed, minimum_stations=10)
    malformed.write_text("stop", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty sentinel"):
        request(10)


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
        "no_shield_native_path",
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
        assert pf_config["pure_pf_schema_version"] == 2
        assert pf_config["estimator_profile"] == "pf_strict"
        assert "variable_cardinality" not in pf_config
        assert "metadata" not in pf_config
        assert "baseline_path_policy" not in pf_config
        assert "baseline_shield_policy" not in pf_config
        serialized_pf = json.dumps(pf_config, sort_keys=True)
        assert "1234" not in serialized_pf
        assert f'"{RAL_SCENE_VARIANT_ID}"' not in serialized_pf
        assert "backend" not in pf_config
        assert "shield_thickness_scale" not in pf_config
        assert "1234" not in entry.pf_config_path.name
        assert RAL_SCENE_VARIANT_ID not in entry.pf_config_path.name
        assert entry.pf_seed == 5678
        assert entry.control_policy_sha256 == sha256(
            entry.control_policy_path.read_bytes()
        ).hexdigest()
        assert (
            entry.session_command[
                entry.session_command.index("--expected-control-policy-sha256") + 1
            ]
            == entry.control_policy_sha256
        )

        runtime_config = load_production_runtime_config(entry.runtime_config_path)
        assert runtime_config["backend"] == "geant4"
        assert runtime_config["engine_mode"] == "external"
        assert runtime_config["primary_sampling_fraction"] == pytest.approx(1.0)
        assert "extends" not in runtime_config
        assert "accelerated_weighted_transport_enable" not in runtime_config
        assert "target_sampled_primaries" not in runtime_config
        assert runtime_config["random_seed_base"] == 8765
        assert runtime_config["random_seed_base"] != entry.scene_seed

        assert "generate-scenario" in entry.scenario_command
        assert "--scene-seed" in entry.scenario_command
        assert (
            entry.scenario_command[
                entry.scenario_command.index("--experiment-profile") + 1
            ]
            == RAL_EXPERIMENT_PROFILE_ID
        )
        assert (
            entry.scenario_command[entry.scenario_command.index("--scene-variant") + 1]
            == RAL_SCENE_VARIANT_ID
        )
        assert "--truth-manifest-output" in entry.scenario_command
        assert "baselines.ral_ablation.session_runner" in entry.session_command
        assert "--scene-seed" not in entry.session_command
        assert "--scene-variant" not in entry.session_command
        assert "--private-scene-profile" not in entry.session_command
        assert "--full-simulation" not in entry.scenario_command
        assert "main.py" not in entry.session_command

    no_shield_policy = json.loads(
        by_variant["no_shield_native_path"].control_policy_path.read_text(
            encoding="utf-8"
        )
    )
    assert no_shield_policy == {
        "schema_version": 2,
        "shield_policy": None,
        "variant": "no_shield_native_path",
    }
    no_shield_document = load_ral_control_policy_document(
        by_variant["no_shield_native_path"].control_policy_path
    )
    assert no_shield_document.source_sha256 == no_shield_document.canonical_sha256
    load_ral_control_policy(
        by_variant["no_shield_native_path"].control_policy_path,
        expected_source_sha256=no_shield_document.source_sha256,
    )
    no_shield_pf = json.loads(
        by_variant["no_shield_native_path"].pf_config_path.read_text(
            encoding="utf-8"
        )
    )
    proposed_pf = json.loads(
        by_variant["proposed"].pf_config_path.read_text(encoding="utf-8")
    )
    assert no_shield_pf == proposed_pf
    assert isinstance(no_shield_pf["dss_pp"], dict)
    assert no_shield_pf["planning_eig_samples"] >= 2
    no_shield_runtime = load_production_runtime_config(
        by_variant["no_shield_native_path"].runtime_config_path
    )
    assert no_shield_runtime["shield_transmission_target"] == pytest.approx(1.0)
    no_shield_geometry = resolve_shield_thickness_config(no_shield_runtime)
    assert no_shield_geometry.thickness_scale == pytest.approx(0.0)
    assert no_shield_geometry.thickness_fe_cm == pytest.approx(0.0)
    assert no_shield_geometry.thickness_pb_cm == pytest.approx(0.0)

    round_robin = json.loads(
        by_variant["round_robin_shield"].control_policy_path.read_text(encoding="utf-8")
    )
    assert round_robin["variant"] == "round_robin_shield"
    assert round_robin["shield_policy"]["name"] == "round_robin"
    round_robin_pf = json.loads(
        by_variant["round_robin_shield"].pf_config_path.read_text(encoding="utf-8")
    )
    assert round_robin_pf["dss_pp"]["shield_view_count_shadow_enabled"] is False
    assert round_robin_pf["dss_pp"]["conditional_greedy_one_swap"] is False
    eig_only = json.loads(
        by_variant["eig_only_path"].pf_config_path.read_text(encoding="utf-8")
    )
    eig_dss = eig_only["dss_pp"]
    for field in (
        "coverage_weight",
        "coverage_floor_quantile",
        "coverage_floor_weight",
        "bearing_diversity_weight",
        "frontier_weight",
        "local_orbit_weight",
        "elevation_condition_weight",
        "revisit_penalty_weight",
        "turn_smoothness_weight",
    ):
        assert eig_dss[field] == pytest.approx(0.0)
    assert eig_dss["exact_eig_coverage_reserve"] == 0
    assert eig_dss["coverage_surface_max_hausdorff_m"] is None
    assert eig_dss["coverage_surface_quadrature_max_points"] is None
    assert eig_dss["local_orbit_ring_radii_m"] == []
    assert eig_dss["local_orbit_sigma_m"] is None
    assert eig_dss["elevation_pair_xy_scale_m"] is None
    assert eig_dss["elevation_pair_z_scale_m"] is None
    assert eig_dss["elevation_angle_threshold_deg"] is None
    assert eig_dss["horizontal_time_weight"] > 0.0
    assert eig_dss["mast_vertical_time_weight"] > 0.0
    assert eig_dss["settling_time_weight"] > 0.0

    manifest_path, script_path = write_ablation_plan(
        entries,
        private_root=private_root,
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["experiment_profile_id"] for row in rows} == {RAL_EXPERIMENT_PROFILE_ID}
    assert {row["scene_variant_id"] for row in rows} == {RAL_SCENE_VARIANT_ID}
    assert manifest_path.is_relative_to(private_root)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(script_path.stat().st_mode) == 0o700
    script = script_path.read_text(encoding="utf-8")
    assert script.count("generate-scenario") == 4
    assert script.count("baselines.ral_ablation.batch_contract") == 1
    assert script.count("baselines.ral_ablation.session_runner") == 4
    assert script.index("baselines.ral_ablation.batch_contract") > script.rindex(
        "generate-scenario"
    )
    assert script.index("baselines.ral_ablation.batch_contract") < script.index(
        "baselines.ral_ablation.session_runner"
    )
    assert "--full-simulation" not in script

    controller_source = inspect.getsource(live_controller_main)
    assert "--scenario" not in controller_source
    assert "--scene-variant" not in controller_source
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
    )
    assert {entry.scene_seed for entry in entries} == {246813579}
    assert {entry.pf_seed for entry in entries} == {975318642}
    assert {entry.transport_seed for entry in entries} == {864297531}
    assert {entry.batch_id for entry in entries} == {"opaque002"}
    assert {entry.seed_policy for entry in entries} == {"fresh_per_batch"}
    assert len(entries) == 4
