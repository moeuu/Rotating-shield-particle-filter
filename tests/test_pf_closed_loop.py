"""Tests for PF control over the estimator-neutral runtime protocol."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from measurement.shielding import generate_octant_orientations
from runtime.adaptive_client import (
    AdaptiveCandidateSnapshot,
    AdaptiveCandidatesEvent,
    AdaptivePublishedEvent,
    AdaptiveReadyEvent,
    AdaptiveRecordEvent,
    AdaptiveRefineRequest,
    AdaptiveStepRequest,
)
from runtime.experiment_profiles import AcquisitionContract

from pf.closed_loop import (
    AdaptiveStopTracker,
    PFClosedLoopResult,
    PFControlBudget,
    _bootstrap_program,
    _completion_diagnostics_extensions,
    _particle_diagnostics,
    _require_plannable_sampler_health,
    _require_refinement_seed_capacity,
    _shield_view_count_shadow_health,
    main,
    run_pf_closed_loop,
)
from pf.live_session import _require_refined_candidate_extension
from planning.configuration import dss_config_from_pf_settings
from planning.dss_types import DSSPPConfig


def _production_settings() -> dict[str, object]:
    """Load a fresh complete production schema-v2 PF configuration."""
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "configs/pf/pf_strict_3d.json").read_text(encoding="utf-8")
    )


def _candidate_snapshot_payload() -> dict[str, object]:
    """Return a two-pose typed candidate payload for refinement tests."""
    return {
        "current_pose_xyz": [0.0, 0.0, 0.5],
        "candidate_poses_xyz": [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
        "travel_costs": [0.0, 2.0],
        "allowed_pair_ids": list(range(64)),
        "current_pair_id": 3,
        "shield_angular_speed_rad_s": 1.0,
        "horizontal_travel_times_s": [0.0, 1.0],
        "mast_vertical_times_s": [0.0, 0.0],
        "settling_times_s": [0.0, 1.0],
    }


def _extended_candidate_snapshot_payload() -> dict[str, object]:
    """Return a valid refinement that adds one runtime-authored pose."""
    payload = _candidate_snapshot_payload()
    payload["candidate_poses_xyz"].append([2.0, 0.0, 0.5])
    payload["travel_costs"].append(3.0)
    payload["horizontal_travel_times_s"].append(2.0)
    payload["mast_vertical_times_s"].append(0.0)
    payload["settling_times_s"].append(1.0)
    return payload


def test_refinement_preserves_anchor_and_prior_motion_quotes() -> None:
    """A valid refinement must be a strict extension of its causal snapshot."""
    previous = AdaptiveCandidateSnapshot.from_payload(_candidate_snapshot_payload())
    refined = AdaptiveCandidateSnapshot.from_payload(
        _extended_candidate_snapshot_payload()
    )

    _require_refined_candidate_extension(previous, refined)


@pytest.mark.parametrize("mismatch", ("shield_state", "motion_quote", "no_addition"))
def test_refinement_rejects_changed_state_or_prior_quote(mismatch: str) -> None:
    """A refinement response cannot rewrite the state used by initial ranking."""
    previous = AdaptiveCandidateSnapshot.from_payload(_candidate_snapshot_payload())
    payload = _extended_candidate_snapshot_payload()
    if mismatch == "shield_state":
        payload["current_pair_id"] = 4
    elif mismatch == "motion_quote":
        payload["travel_costs"][1] = 2.5
        payload["horizontal_travel_times_s"][1] = 1.5
    else:
        payload = _candidate_snapshot_payload()
    refined = AdaptiveCandidateSnapshot.from_payload(payload)

    with pytest.raises(RuntimeError, match="refinement|Refined"):
        _require_refined_candidate_extension(previous, refined)


def test_refinement_top_k_must_fit_authenticated_candidate_snapshot() -> None:
    """An impossible refinement count must fail immediately after handshake."""
    candidates = AdaptiveCandidateSnapshot.from_payload(
        _candidate_snapshot_payload()
    )

    with pytest.raises(ValueError, match="exceeds the authenticated candidate"):
        _require_refinement_seed_capacity(
            {"runtime_candidate_refinement_top_k": 3},
            candidates,
        )


@pytest.mark.parametrize(
    "failed_gate",
    (
        "smc_rejuvenation_wall_time_respected",
        "rejuvenation_mixing_complete",
        "structural_mixing_complete",
    ),
)
def test_failed_sampler_health_aborts_before_next_planning(
    failed_gate: str,
) -> None:
    """No incomplete rejuvenation state may feed another live action."""
    health = {
        "smc_rejuvenation_wall_time_respected": True,
        "rejuvenation_mixing_complete": True,
        "structural_mixing_complete": True,
    }
    health[failed_gate] = False

    with pytest.raises(RuntimeError, match="forbids further live planning"):
        _require_plannable_sampler_health({"sampler_health": health})


def _write_one_station_production_config(
    path: Path,
    *,
    cui_enabled: bool,
) -> None:
    """Write a complete production config for the one-station fake runtime."""
    settings = _production_settings()
    settings["runtime_candidate_refinement_top_k"] = 0
    settings["planner_audit_top_k"] = 0
    settings["dss_pp"]["shield_view_count_shadow_enabled"] = False
    settings["adaptive_stop"]["assessment_start_station"] = 1
    settings["adaptive_stop"]["required_consecutive_stations"] = 1
    settings["cui_split_view"] = bool(cui_enabled)
    settings["cui_split_view_serve"] = bool(cui_enabled)
    if not cui_enabled:
        settings["cui_split_view_host"] = None
        settings["cui_split_view_port"] = None
        settings["cui_split_view_public_host"] = None
    path.write_text(json.dumps(settings), encoding="utf-8")


def _stub_cuda_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate controller tests from the host's physical CUDA availability."""
    from pf import closed_loop

    monkeypatch.setattr(
        closed_loop,
        "preflight_compute_backend",
        lambda **kwargs: "batched_torch_float64",
    )


def _minimal_live_cli_args(tmp_path: Path) -> list[str]:
    """Return required production CLI arguments other than the PF seed."""
    return [
        "--session-socket",
        str(tmp_path / "runtime.sock"),
        "--runtime-root",
        str(tmp_path),
        "--config",
        str(tmp_path / "pf.json"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_pf_closed_loop_requires_explicit_seed_keyword(tmp_path: Path) -> None:
    """The production API must not invent a deterministic default seed."""
    with pytest.raises(TypeError, match="seed"):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=tmp_path / "pf.json",
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    ("seed", "error_type"),
    (
        (True, TypeError),
        (-1, ValueError),
        (1.0, TypeError),
        ("1", TypeError),
        (np.int64(1), TypeError),
    ),
)
def test_pf_closed_loop_rejects_invalid_seed_before_runtime_setup(
    seed: object,
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid production seeds must fail before config loading or connection."""
    from pf import closed_loop

    def forbidden_config_load(*args: object, **kwargs: object) -> object:
        """Fail if seed validation does not precede production setup."""
        del args, kwargs
        raise AssertionError("Production setup must not run for an invalid seed.")

    monkeypatch.setattr(
        closed_loop,
        "load_production_live_pf_config",
        forbidden_config_load,
    )

    with pytest.raises(error_type, match="seed must be a nonnegative integer"):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=tmp_path / "pf.json",
            output_dir=tmp_path / "output",
            seed=seed,
        )


def test_closed_loop_rejects_forged_policy_with_valid_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canonical provenance cannot authorize an arbitrary behavior object."""
    from baselines.ral_ablation.control_policy import load_ral_control_policy
    from pf import closed_loop

    policy_path = tmp_path / "sealed-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "path_policy": {"name": "passive_serpentine", "row_count": 2},
                "shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            }
        ),
        encoding="utf-8",
    )
    sealed = load_ral_control_policy(policy_path)

    class ForgedPolicy:
        """Mimic the removed structural protocol while changing its behavior."""

        has_fixed_path = True
        provenance = sealed.provenance

        def select_shield_program(self, **kwargs: object) -> None:
            """Pretend to provide a shield decision unrelated to provenance."""
            del kwargs
            return None

        def select_path(self, **kwargs: object) -> None:
            """Pretend to provide a path decision unrelated to provenance."""
            del kwargs
            return None

        def validate_pf_settings(self, settings: object) -> None:
            """Pretend every PF configuration is compatible."""
            del settings

    def forbidden_config_load(*args: object, **kwargs: object) -> object:
        """Fail if forged policy validation reaches configuration loading."""
        del args, kwargs
        raise AssertionError("forged policy reached production configuration")

    monkeypatch.setattr(
        closed_loop,
        "load_production_live_pf_config",
        forbidden_config_load,
    )

    with pytest.raises(TypeError, match="exact loader-sealed RALControlPolicy"):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=tmp_path / "pf.json",
            output_dir=tmp_path / "output",
            seed=7,
            control_policy=ForgedPolicy(),
        )

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "mode",
    ("native_with_disabled_sentinels", "round_robin_with_disabled_sentinels", "passive_with_full_dss"),
)
def test_planner_mode_mismatch_fails_before_runtime_connection(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native and external planner representations must not be interchangeable."""
    from baselines.ral_ablation.control_policy import load_ral_control_policy
    from pf import closed_loop

    settings = _production_settings()
    policy = None
    if mode in {"native_with_disabled_sentinels", "round_robin_with_disabled_sentinels"}:
        settings["dss_pp"] = None
        settings["planning_eig_samples"] = None
    if mode == "round_robin_with_disabled_sentinels":
        policy_payload = {
            "schema_version": 1,
            "path_policy": None,
            "shield_policy": {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": True,
            },
        }
    elif mode == "passive_with_full_dss":
        settings["dss_pp"]["shield_view_count_shadow_enabled"] = False
        settings["dss_pp"]["conditional_greedy_one_swap"] = False
        settings["runtime_candidate_refinement_top_k"] = 0
        settings["planner_audit_top_k"] = 0
        policy_payload = {
            "schema_version": 1,
            "path_policy": {"name": "passive_serpentine", "row_count": 2},
            "shield_policy": {"name": "fixed", "fixed_pair_id": 0},
        }
    else:
        policy_payload = None
    if policy_payload is not None:
        policy_path = tmp_path / f"{mode}.policy.json"
        policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
        policy = load_ral_control_policy(policy_path)
    config_path = tmp_path / f"{mode}.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")

    def forbidden_preflight(**kwargs: object) -> None:
        """Fail if planner-mode validation reaches compute/runtime setup."""
        del kwargs
        raise AssertionError("planner-mode mismatch reached compute preflight")

    monkeypatch.setattr(closed_loop, "preflight_compute_backend", forbidden_preflight)

    with pytest.raises(ValueError, match="(Native|fixed RA-L path)"):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config_path,
            output_dir=tmp_path / "output",
            seed=7,
            control_policy=policy,
        )

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.*"))


@pytest.mark.parametrize("seed_args", ((), ("--seed", "not-an-integer")))
def test_pf_live_cli_rejects_missing_or_noninteger_seed(
    seed_args: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """CLI parsing must fail closed when its mandatory seed is unusable."""
    with pytest.raises(SystemExit) as raised:
        main([*_minimal_live_cli_args(tmp_path), *seed_args])

    assert raised.value.code == 2
    assert not (tmp_path / "output").exists()


def test_pf_live_cli_rejects_negative_seed_before_runtime_setup(
    tmp_path: Path,
) -> None:
    """A parsed negative CLI seed must fail before reading runtime inputs."""
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        main([*_minimal_live_cli_args(tmp_path), "--seed", "-1"])

    assert not (tmp_path / "output").exists()


def _context_payload() -> dict[str, object]:
    """Return one minimal truth-free adaptive runtime context."""
    return {
        "repository_commit": "a" * 40,
        "runtime_config": {
            "full_spectrum_contract_hash_sha256": "b" * 64,
        },
        "environment": {
            "size_x": 2.0,
            "size_y": 2.0,
            "size_z": 2.0,
            "detector_position": [0.5, 0.5, 0.5],
            "acquisition_contract": AcquisitionContract(
                max_stations=1,
                views_per_station=1,
                live_time_s=30.0,
                max_measurements=1,
                min_station_separation_m=3.0,
                coverage_radius_m=3.0,
            ).to_payload(),
        },
        "sim_backend": "test",
        "spectrum_count_method": "joint_full_spectrum_generative",
        "isotopes": ["Cs-137"],
        "obstacle_layout_path": None,
        "source_rate_model": "detector_cps_1m",
        "metadata": {},
        "run_id": "pf-live-test",
        "source_rate_semantics": {},
        "forward_model_manifest": {},
        "runtime_config_sha256": "c" * 64,
        "schema_version": 2,
    }


def test_standard_bootstrap_does_not_depend_on_legacy_program_library() -> None:
    """Guard-free live bootstrap must remain usable after old48 removal."""
    estimator = SimpleNamespace(normals=np.zeros((8, 3), dtype=np.float64))
    planner = DSSPPConfig(
        program_length=8,
        proxy_eig_samples=2,
    )

    program = _bootstrap_program(estimator, planner, None)

    assert program.pair_ids == (0, 9, 18, 27, 36, 45, 54, 63)
    assert program.kind == "prior_balanced_bootstrap"


def test_shield_view_count_health_is_truth_free_and_fail_closed() -> None:
    """Diversity, model mismatch, and cardinality caps must all force eight."""
    health = _shield_view_count_shadow_health(
        belief_after_station_id=7,
        particle_adequacy={
            "assessment": {
                "diversity_warning": True,
                "minimum_guided_initialization_ess_ratio": 0.5,
                "minimum_cumulative_unique_ancestor_count": 1,
            },
            "isotopes": {"Cs-137": {}, "Co-60": {}},
        },
        posterior_convergence={
            "sampler_health": {
                "smc_rejuvenation_wall_time_respected": True,
                "rejuvenation_mixing_complete": False,
                "structural_mixing_complete": True,
            },
            "innovation": {"available": True, "passed": False},
            "isotopes": {
                "Cs-137": {
                    "gates": {
                        "cardinality_not_at_upper_boundary": False,
                        "surface_path_concentration": True,
                    }
                },
                "Co-60": {
                    "gates": {
                        "cardinality_not_at_upper_boundary": True,
                        "surface_path_concentration": True,
                    }
                },
            },
            "ready": False,
        },
    )

    assert health["available"] is True
    assert health["passed"] is False
    assert health["truth_used"] is False
    assert health["source_station_id"] == 7
    assert health["hard_failure_reasons"] == [
        "particle_diversity_warning",
        "sampler_health:rejuvenation_mixing_complete",
        "posterior_predictive_innovation_failed",
        "cardinality_upper_boundary:Cs-137",
    ]


def test_shield_view_count_health_does_not_require_convergence_ready() -> None:
    """Normal early posterior uncertainty alone must not be labelled unhealthy."""
    health = _shield_view_count_shadow_health(
        belief_after_station_id=1,
        particle_adequacy={
            "assessment": {
                "diversity_warning": False,
                "minimum_guided_initialization_ess_ratio": 0.8,
                "minimum_cumulative_unique_ancestor_count": 100,
            },
            "isotopes": {"Cs-137": {}},
        },
        posterior_convergence={
            "sampler_health": {
                "smc_rejuvenation_wall_time_respected": True,
                "rejuvenation_mixing_complete": True,
                "structural_mixing_complete": True,
            },
            "innovation": {"available": True, "passed": True},
            "isotopes": {
                "Cs-137": {
                    "gates": {
                        "cardinality_not_at_upper_boundary": True,
                        "surface_path_concentration": False,
                    }
                }
            },
            "ready": False,
        },
    )

    assert health["passed"] is True
    assert health["hard_failure_reasons"] == []


def test_missing_particle_diversity_evidence_fails_closed() -> None:
    """Absent ESS and ancestry evidence must never permit view shortening."""
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        pf_config=SimpleNamespace(num_particles=4096, target_ess_ratio=0.4),
        step_diagnostics=lambda **_kwargs: {
            "Cs-137": {
                "joint_smc_wall_time_limit_exceeded": False,
                "joint_rejuvenation_mixing_incomplete": False,
                "joint_structural_mixing_incomplete": False,
            }
        },
    )

    diagnostics = _particle_diagnostics(estimator)
    assessment = diagnostics["assessment"]
    assert assessment["diversity_evidence_available"] is False
    assert assessment["diversity_warning"] is True

    health = _shield_view_count_shadow_health(
        belief_after_station_id=1,
        particle_adequacy=diagnostics,
        posterior_convergence={
            "sampler_health": {
                "smc_rejuvenation_wall_time_respected": True,
                "rejuvenation_mixing_complete": True,
                "structural_mixing_complete": True,
            },
            "innovation": {"available": True, "passed": True},
            "isotopes": {
                "Cs-137": {
                    "gates": {
                        "cardinality_not_at_upper_boundary": True,
                        "surface_path_concentration": False,
                    }
                }
            },
        },
    )
    assert health["passed"] is False
    assert health["hard_failure_reasons"] == ["particle_diversity_evidence_unavailable"]


def test_particle_diagnostics_omit_deep_rejection_payloads() -> None:
    """The durable station trace must retain only compact particle health."""
    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        pf_config=SimpleNamespace(num_particles=100, target_ess_ratio=0.4),
        step_diagnostics=lambda **_kwargs: {
            "Cs-137": {
                "particle_count": 100,
                "current_ess": 50.0,
                "current_ess_ratio": 0.5,
                "temper_resamples": 1,
                "temper_min_ess": 35.0,
                "joint_guided_initialization_ess": 60.0,
                "station_unique_ancestor_count": 50,
                "cumulative_unique_ancestor_count": 40,
                "r_probability_by_count": {"1": 0.7, "2": 0.3},
                "joint_smc_wall_time_limit_exceeded": False,
                "joint_rejuvenation_mixing_incomplete": False,
                "joint_structural_mixing_incomplete": False,
                "transition_weight_mass": {
                    "birth_attempted_weight_mass": 0.6,
                    "birth_accepted_weight_mass": 0.2,
                    "death_attempted_weight_mass": 0.4,
                    "death_accepted_weight_mass": 0.1,
                    "block_cardinality_changed_weight_mass": 0.05,
                },
                "structural_rejection_diagnostics": {"large": [1] * 100},
                "joint_cross_isotope_rejection_diagnostics": {
                    "large": [1] * 100
                },
                "joint_cross_isotope_state_rejection_diagnostics": {
                    "large": [1] * 100
                },
            }
        },
    )

    payload = _particle_diagnostics(estimator)

    isotope = payload["isotopes"]["Cs-137"]
    assert isotope["cardinality_distribution"] == {"1": 0.7, "2": 0.3}
    assert isotope["structural_transition_weight_mass"] == {
        "attempted": 1.0,
        "accepted": pytest.approx(0.3),
    }
    assert payload["sampler_health"] == {
        "smc_rejuvenation_wall_time_respected": True,
        "rejuvenation_mixing_complete": True,
        "structural_mixing_complete": True,
    }
    assert "interpretation" not in payload["assessment"]
    assert "structural_rejection_diagnostics" not in isotope
    assert "joint_cross_isotope_rejection_diagnostics" not in isotope
    assert "joint_cross_isotope_state_rejection_diagnostics" not in isotope


class _FakeRuntimeClient:
    """Return one station while capturing every PF-selected action."""

    instance: "_FakeRuntimeClient | None" = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a deterministic one-pose runtime session."""
        del args, kwargs
        type(self).instance = self
        self.closed = False
        self.aborted = False
        self.requests: list[dict[str, object]] = []
        self.refinement_requests: list[dict[str, object]] = []
        self.overlay_requests: list[bool] = []
        self.candidates = {
            "current_pose_xyz": [0.5, 0.5, 0.5],
            "candidate_poses_xyz": [[0.5, 0.5, 0.5]],
            "travel_costs": [0.0],
            "allowed_pair_ids": list(range(64)),
            "current_pair_id": 63,
            "shield_angular_speed_rad_s": 1.0,
            "horizontal_travel_times_s": [0.0],
            "mast_vertical_times_s": [0.0],
            "settling_times_s": [0.0],
        }

    @classmethod
    def connect(
        cls,
        socket_path: Path,
        **kwargs: object,
    ) -> "_FakeRuntimeClient":
        """Construct a fake client from the generic socket-only boundary."""
        del socket_path, kwargs
        return cls()

    def read_event(self) -> dict[str, object]:
        """Return a bootstrap pair that PF is not required to execute."""
        return {
            "type": "ready",
            "schema_version": 1,
            "context": _context_payload(),
            "candidates": self.candidates,
            "bootstrap": {
                "candidate_index": 0,
                "fe_orientation_index": 7,
                "pb_orientation_index": 7,
            },
        }

    def read_ready_event(self) -> AdaptiveReadyEvent:
        """Return the typed form of the fake runtime handshake."""
        return AdaptiveReadyEvent.from_payload(self.read_event())

    def handshake(self) -> AdaptiveReadyEvent:
        """Return the handshake through the concise runtime API."""
        return self.read_ready_event()

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        """Return an exact integer raw spectrum for the chosen PF action."""
        self.requests.append(dict(payload))
        request_candidates = AdaptiveCandidateSnapshot.from_payload(
            self.candidates
        )
        candidate_index = int(payload["candidate_index"])
        pair_id = int(payload["fe_orientation_index"]) * 8 + int(
            payload["pb_orientation_index"]
        )
        travel_time_s = request_candidates.travel_costs[candidate_index]
        shield_actuation_time_s = (
            request_candidates.quote_shield_program_time_s((pair_id,))
        )
        self.candidates["current_pair_id"] = pair_id
        return {
            "type": "record",
            "record": {
                "step_id": int(payload["action_id"]),
                "action_id": int(payload["action_id"]),
                "station_id": int(payload["station_id"]),
                "detector_pose_xyz": [0.5, 0.5, 0.5],
                "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "fe_orientation_index": int(payload["fe_orientation_index"]),
                "pb_orientation_index": int(payload["pb_orientation_index"]),
                "live_time_s": float(payload["dwell_time_s"]),
                "travel_time_s": travel_time_s,
                "shield_actuation_time_s": shield_actuation_time_s,
                "energy_bin_edges_keV": [0.0, 1.0, 2.0],
                "spectrum_counts": [2, 3],
                "metadata": {
                    "full_spectrum_contract_hash_sha256": "b" * 64,
                    "station_complete": bool(payload["station_complete"]),
                    "travel_waypoints_xyz": [
                        [0.25, 0.5, 0.5],
                        [0.5, 0.5, 0.5],
                    ],
                },
            },
            "candidates": self.candidates,
        }

    def request_step(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Return the typed form of one fake runtime record response."""
        return AdaptiveRecordEvent.from_payload(self.request(request.to_payload()))

    def acquire(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Acquire one record through the concise runtime API."""
        return self.request_step(request)

    def request_refinement(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Return unchanged typed candidates for a fake refinement request."""
        self.refinement_requests.append(request.to_payload())
        return AdaptiveCandidatesEvent.from_payload(
            {"type": "candidates", "candidates": self.candidates}
        )

    def refine_candidates(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Refine candidates through the concise runtime API."""
        return self.request_refinement(request)

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Fail if an estimator-owned controller requests realized truth."""
        self.overlay_requests.append(bool(include_truth))
        raise AssertionError("PF closed loop must not request a truth overlay.")

    def finalize(self) -> dict[str, object]:
        """Return the fake immutable log path."""
        return {
            "type": "published",
            "path": "/tmp/pf-live-log",
            "record_count": len(self.requests),
        }

    def finalize_event(self) -> AdaptivePublishedEvent:
        """Return the typed form of the fake publication response."""
        return AdaptivePublishedEvent.from_payload(self.finalize())

    def finalize_log(self) -> AdaptivePublishedEvent:
        """Finalize the log through the concise runtime API."""
        return self.finalize_event()

    def close(self) -> None:
        """Record deterministic client cleanup."""
        self.closed = True

    def abort(self) -> None:
        """Expose the runtime cleanup method."""
        self.aborted = True



class _FakeEstimator:
    """Expose only PF operations needed for the one-station controller test."""

    def __init__(self) -> None:
        """Initialize one pose and an empty record history."""
        self.isotopes = ("Cs-137",)
        self.normals = np.asarray(generate_octant_orientations(), dtype=float)
        self.poses = [np.asarray([0.5, 0.5, 0.5])]
        self.measurements: list[object] = []
        self.kernel_cache = None
        self.detector_aperture_samples = 121
        self.pf_config = SimpleNamespace(
            num_particles=2000,
            target_ess_ratio=0.4,
        )

    def update_spectrum_station(
        self,
        records: tuple[object, ...],
        **kwargs: object,
    ) -> None:
        """Capture the raw station inputs supplied by the controller."""
        del kwargs
        self.measurements.extend(records)

    def step_diagnostics(self, **kwargs: object) -> dict[str, object]:
        """Return minimal particle-adequacy evidence."""
        del kwargs
        return {
            "Cs-137": {
                "particle_count": 2000,
                "current_ess": 1000.0,
                "current_ess_ratio": 0.5,
                "joint_smc_wall_time_limit_exceeded": False,
                "joint_rejuvenation_mixing_incomplete": False,
                "joint_structural_mixing_incomplete": False,
            }
        }

    def posterior_convergence_diagnostics(self) -> dict[str, object]:
        """Return one explicit non-converged production stop assessment."""
        return {"ready": False}

    def posterior_point_estimate(self) -> dict[str, SimpleNamespace]:
        """Return one truth-free point estimate for the station trace."""
        return {
            "Cs-137": SimpleNamespace(
                to_dict=lambda: {
                    "map_cardinality": 1,
                    "cardinality_distribution": {"0": 0.1, "1": 0.9},
                    "selected_stratum_mass": 0.9,
                    "modes": [
                        {
                            "label_index": 0,
                            "position_medoid_xyz": [0.1, 0.2, 0.3],
                            "credible_radius_95_m": 0.5,
                            "strength_representative_cps_1m": 2.0,
                            "posterior_mass": 0.9,
                        }
                    ],
                },
            )
        }


class _FakePFLiveSession:
    """Exercise the controller against one session-owned fake estimator."""

    next_estimator: _FakeEstimator | None = None
    instance: "_FakePFLiveSession | None" = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Take ownership of the estimator selected by the current test."""
        del args, kwargs
        estimator = type(self).next_estimator
        if estimator is None:
            raise AssertionError("A fake live estimator must be configured.")
        self.estimator = estimator
        self.records: list[object] = []
        self.pending_station: list[object] = []
        self.completed_extensions: dict[str, object] | None = None
        self.bound_log: object | None = None
        type(self).instance = self

    @property
    def record_count(self) -> int:
        """Return the exact count of records accepted by the fake session."""
        return len(self.records)

    def receive_persisted_station(self, records: object) -> None:
        """Restore one complete prefix station through the canonical API."""
        rows = tuple(records)
        self.records.extend(rows)
        self.estimator.update_spectrum_station(tuple(rows))

    def receive_acquired(
        self,
        record: object,
        *,
        request: AdaptiveStepRequest,
        request_candidates: AdaptiveCandidateSnapshot,
        next_candidates: AdaptiveCandidateSnapshot,
    ) -> bool:
        """Verify controller request wiring before accepting one fake record."""
        requested_pose = request_candidates.candidate_poses_xyz[
            request.candidate_index
        ]
        assert int(record.step_id) == len(self.records)
        assert int(record.action_id) == len(self.records)
        assert int(record.station_id) == request.station_id
        assert tuple(record.detector_pose_xyz) == tuple(requested_pose)
        assert int(record.fe_orientation_index) == request.fe_orientation_index
        assert int(record.pb_orientation_index) == request.pb_orientation_index
        assert float(record.live_time_s) == request.dwell_time_s
        assert float(record.travel_time_s) == request_candidates.travel_costs[
            request.candidate_index
        ]
        requested_pair_id = (
            request.fe_orientation_index * 8 + request.pb_orientation_index
        )
        assert float(record.shield_actuation_time_s) == (
            request_candidates.quote_shield_program_time_s((requested_pair_id,))
        )
        assert (
            record.metadata.get("station_complete") is True
        ) is request.station_complete
        assert next_candidates.current_pair_id == (
            request.fe_orientation_index * 8 + request.pb_orientation_index
        )
        self.records.append(record)
        self.pending_station.append(record)
        if not request.station_complete:
            return False
        self.estimator.update_spectrum_station(tuple(self.pending_station))
        self.pending_station.clear()
        return True

    def complete_live_state(
        self,
        *,
        diagnostics_extensions: dict[str, object],
    ) -> object:
        """Record controller diagnostics at the mandatory pre-finalize seal."""
        self.completed_extensions = dict(diagnostics_extensions)
        return object()

    def bind_published_log(self, log: object) -> object:
        """Record the exact finalized log supplied by the controller."""
        if self.completed_extensions is None:
            raise AssertionError("The fake PF must be sealed before binding.")
        self.bound_log = log
        return object()

    def _publish_bound_result_into_staging(self, output_dir: Path) -> object:
        """Write minimal package artifacts into the outer atomic stage."""
        if self.bound_log is None:
            raise AssertionError("The fake PF must bind before publication.")
        for name in (
            "pf_posterior.json",
            "pf_diagnostics.json",
            "pf_state.json",
            "pf_checkpoint.json",
        ):
            (output_dir / name).write_text("{}", encoding="utf-8")
        (output_dir / "pf_particles.npz").write_bytes(b"fake-particles")
        return object()


class _FakeLog:
    """Expose finalized-log fields and the shared station-view surface."""

    path = Path("/tmp/pf-live-log")
    run_id = "pf-live-test"
    records = (SimpleNamespace(station_id=0),)

    def station_view(self) -> SimpleNamespace:
        """Return the station count used by closed-loop result publication."""
        return SimpleNamespace(station_count=1)


def test_final_diagnostics_keep_one_compact_copy_of_each_diagnostic(
) -> None:
    """Controller completion adds only stop and fixed-budget provenance."""
    budget = PFControlBudget(
        max_stations=1,
        max_measurements=1,
        views_per_station=1,
        live_time_s=1.0,
        stop_assessment_start_station=1,
        stop_required_consecutive_stations=1,
        runtime_refinement_top_k=0,
        planner_audit_top_k=0,
    )
    diagnostics = _completion_diagnostics_extensions(
        stop_reason="maximum_station_budget",
        budget=budget,
        adaptive_stop_status={
            "assessed": False,
            "instantaneous_ready": None,
            "consecutive_ready_stations": 0,
            "stop_ready": False,
            "posterior_convergence": {"duplicated": True},
        },
    )

    assert diagnostics["stop"]["adaptive"] == {
        "assessed": False,
        "consecutive_ready_stations": 0,
        "stop_ready": False,
    }
    assert diagnostics["stop"]["reason"] == "maximum_station_budget"
    assert diagnostics["control_budget"] == {
        "max_stations": 1,
        "max_measurements": 1,
        "views_per_station": 1,
        "live_time_s": 1.0,
        "stop_assessment_start_station": 1,
        "stop_required_consecutive_stations": 1,
        "runtime_refinement_top_k": 0,
        "planner_audit_top_k": 0,
    }


def test_pf_budget_requires_one_complete_estimator_station() -> None:
    """The runtime contract itself must reject a truncated station budget."""
    with pytest.raises(ValueError, match=r"max_stations \* views_per_station"):
        AcquisitionContract(
            max_stations=1,
            views_per_station=8,
            live_time_s=30.0,
            max_measurements=1,
            min_station_separation_m=3.0,
            coverage_radius_m=3.0,
        )


def test_adaptive_stop_starts_at_ten_and_first_stops_at_twelve() -> None:
    """Three ready assessments from station 10 must first stop at station 12."""
    settings = _production_settings()
    contract = AcquisitionContract(
        max_stations=16,
        views_per_station=8,
        live_time_s=20.0,
        max_measurements=128,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    )
    budget = PFControlBudget.from_runtime_contract(settings, contract)

    class _ReadyEstimator:
        """Return one ready posterior for every actual assessment."""

        def __init__(self) -> None:
            """Initialize an assessment call counter."""
            self.calls = 0

        def posterior_convergence_diagnostics(self) -> dict[str, bool]:
            """Return a model-ready diagnostic payload."""
            self.calls += 1
            return {"ready": True}

    estimator = _ReadyEstimator()
    tracker = AdaptiveStopTracker(budget)
    statuses = [
        tracker.assess(estimator, station_count=station_count)
        for station_count in range(1, 13)
    ]

    assert budget.earliest_adaptive_stop_station == 12
    assert all(not status["assessed"] for status in statuses[:9])
    assert [status["consecutive_ready_stations"] for status in statuses[9:]] == [
        1,
        2,
        3,
    ]
    assert all(not status["stop_ready"] for status in statuses[:11])
    assert statuses[11]["stop_ready"] is True
    assert estimator.calls == 3


def test_adaptive_stop_ready_streak_resets_after_one_failed_station() -> None:
    """A failed posterior generation must require three new ready stations."""
    settings = _production_settings()
    contract = AcquisitionContract(
        max_stations=16,
        views_per_station=8,
        live_time_s=20.0,
        max_measurements=128,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    )
    budget = PFControlBudget.from_runtime_contract(settings, contract)
    readiness = iter((True, False, True, True, True))
    estimator = SimpleNamespace(
        posterior_convergence_diagnostics=lambda: {"ready": next(readiness)}
    )
    tracker = AdaptiveStopTracker(budget)

    statuses = [
        tracker.assess(estimator, station_count=station_count)
        for station_count in range(1, 15)
    ]

    assert statuses[10]["consecutive_ready_stations"] == 0
    assert statuses[11]["consecutive_ready_stations"] == 1
    assert statuses[13]["stop_ready"] is True


def test_closed_loop_applies_declared_passive_path_and_fixed_shield() -> None:
    """RA-L passive policy must bypass PF EIG for both action dimensions."""
    from baselines.ral_ablation.control_policy import RALControlPolicy
    from pf import closed_loop

    estimator = SimpleNamespace(
        normals=np.asarray(generate_octant_orientations(), dtype=float)
    )
    settings = _production_settings()
    settings["dss_pp"]["shield_view_count_shadow_enabled"] = False
    policy = RALControlPolicy(
        path_policy={"name": "passive_serpentine", "row_count": 2},
        shield_policy={"name": "fixed", "fixed_pair_id": 0},
    )
    contract = AcquisitionContract(
        max_stations=1,
        views_per_station=2,
        live_time_s=30.0,
        max_measurements=2,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    )
    planner = dss_config_from_pf_settings(
        settings,
        acquisition_contract=contract,
        detector_aperture_samples=121,
    )
    candidates = AdaptiveCandidateSnapshot(
        current_pose_xyz=(1.0, 1.0, 0.5),
        candidate_poses_xyz=(
            (0.0, 0.0, 0.5),
            (2.0, 2.0, 0.5),
            (1.0, 1.0, 0.5),
        ),
        travel_costs=(1.0, 1.0, 0.0),
        allowed_pair_ids=tuple(range(64)),
        current_pair_id=63,
        shield_angular_speed_rad_s=1.0,
        horizontal_travel_times_s=(1.0, 1.0, 0.0),
        mast_vertical_times_s=(0.0, 0.0, 0.0),
        settling_times_s=(0.0, 0.0, 0.0),
    )

    result = closed_loop._plan(
        estimator,
        candidates,
        current_pose=np.asarray([1.0, 1.0, 0.5]),
        visited_poses=[],
        obstacle_grid=None,
        room_bounds=(np.asarray([0.0, 0.0, 0.5]), np.asarray([2.0, 2.0, 0.5])),
        planner=planner,
        rng=np.random.default_rng(7),
        station_index=0,
        control_policy=policy,
    )

    assert result.next_pose_index == 0
    assert result.shield_program.pair_ids == (0, 0)
    assert result.sequence == ()
    assert result.diagnostics == {
        "selection_mode": "external_control_path",
        "external_path_policy": "passive_serpentine",
        "external_shield_program_name": "fixed_shield_0",
    }


def test_passive_fixed_path_completes_two_station_live_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The passive RA-L path must audit and finalize without fake DSS evidence."""
    from baselines.ral_ablation.control_policy import load_ral_control_policy
    from pf import closed_loop

    class _TwoStationRuntimeClient(_FakeRuntimeClient):
        """Expose a two-station runtime contract through the standard fake client."""

        def read_event(self) -> dict[str, object]:
            """Return a fresh handshake with room for two one-view stations."""
            ready = super().read_event()
            context = ready["context"]
            assert isinstance(context, dict)
            environment = context["environment"]
            assert isinstance(environment, dict)
            environment["acquisition_contract"] = AcquisitionContract(
                max_stations=2,
                views_per_station=1,
                live_time_s=30.0,
                max_measurements=2,
                min_station_separation_m=3.0,
                coverage_radius_m=3.0,
            ).to_payload()
            return ready

    class _TwoStationLog(_FakeLog):
        """Expose the two records finalized by the passive fake runtime."""

        records = (
            SimpleNamespace(station_id=0),
            SimpleNamespace(station_id=1),
        )

        def station_view(self) -> SimpleNamespace:
            """Return the exact two-station count."""
            return SimpleNamespace(station_count=2)

    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=False)
    passive_settings = json.loads(config.read_text(encoding="utf-8"))
    passive_settings["dss_pp"] = None
    passive_settings["planning_eig_samples"] = None
    passive_settings["runtime_candidate_refinement_top_k"] = 0
    passive_settings["planner_audit_top_k"] = 0
    config.write_text(json.dumps(passive_settings), encoding="utf-8")
    _stub_cuda_preflight(monkeypatch)
    _FakePFLiveSession.next_estimator = _FakeEstimator()
    fake_log = _TwoStationLog()
    policy_path = tmp_path / "passive_control_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "path_policy": {"name": "passive_serpentine", "row_count": 2},
                "shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            }
        ),
        encoding="utf-8",
    )
    policy = load_ral_control_policy(policy_path)
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _TwoStationRuntimeClient,
    )
    def forbidden_dss_config(*args: object, **kwargs: object) -> object:
        """Fail if a fixed path constructs the native DSS-PP configuration."""
        del args, kwargs
        raise AssertionError("fixed path must not construct DSSPPConfig")

    monkeypatch.setattr(
        closed_loop,
        "dss_config_from_pf_settings",
        forbidden_dss_config,
    )
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)

    result = run_pf_closed_loop(
        tmp_path / "runtime.sock",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
        seed=17,
        control_policy=policy,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "planner_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.record_count == 2
    assert result.station_count == 2
    assert [row["selection_mode"] for row in rows] == [
        "external_control_bootstrap",
        "external_control_path",
    ]
    assert rows[1]["external_control_execution"] == {
        "path_policy_name": "passive_serpentine",
        "shield_program_name": "fixed_shield_0",
    }
    assert rows[1]["selected_program"]["pair_ids"] == [0]
    assert {
        "candidate_pose_count",
        "exact_pose_count",
        "proxy_subset_evaluation_count",
        "exact_subset_evaluation_count",
        "planning_particle_count",
        "exact_eig_seed",
        "selected_information_gain",
        "planning_eig_shortlist",
        "ranked_nodes",
        "component_leaders",
        "shield_view_count_shadow",
    }.isdisjoint(rows[1])
    client = _TwoStationRuntimeClient.instance
    assert client is not None
    assert len(client.requests) == 2
    assert client.closed is True


def test_pf_closed_loop_owns_budget_and_shield_program(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Runtime must execute PF choices without supplying a fixed action plan."""
    from pf import closed_loop

    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=False)
    _stub_cuda_preflight(monkeypatch)
    estimator = _FakeEstimator()
    fake_log = _FakeLog()
    _FakePFLiveSession.next_estimator = estimator

    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)

    result = run_pf_closed_loop(
        tmp_path / "runtime.sock",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
        seed=17,
    )

    client = _FakeRuntimeClient.instance
    assert isinstance(result, PFClosedLoopResult)
    assert client is not None
    assert client.closed is True
    assert len(client.requests) == 1
    live_session = _FakePFLiveSession.instance
    assert live_session is not None
    assert live_session.bound_log is fake_log
    assert len(live_session.records) == 1
    assert int(live_session.records[0].step_id) == 0
    assert live_session.completed_extensions is not None
    assert "actions" not in client.requests[0]
    assert client.requests[0]["station_complete"] is True
    assert (
        int(client.requests[0]["fe_orientation_index"]) * 8
        + int(client.requests[0]["pb_orientation_index"])
        != 63
    )
    audit = json.loads(
        (tmp_path / "output" / "planner_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert audit["selection_mode"] == "pf_prior_balanced_bootstrap"
    assert audit["schema_version"] == 3
    assert audit["candidate_pose_count"] == 0
    assert "selected_information_gain" not in audit
    assert "mc_seed_rank_stability" not in audit
    assert "shield_view_count_shadow" not in audit
    assert len(estimator.measurements) == 1
    assert result.station_count == 1
    station_trace = json.loads(
        (tmp_path / "output" / "pf_station_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert station_trace["schema_version"] == 2
    assert station_trace["pf_update_elapsed_s"] >= 0.0
    assert (
        station_trace["posterior_snapshot"]["isotopes"]["Cs-137"]["map_cardinality"]
        == 1
    )
    assert station_trace["posterior_snapshot"]["publishable"] is False
    assert set(station_trace["adaptive_stop"]) == {
        "assessed",
        "consecutive_ready_stations",
        "instantaneous_ready",
        "stop_ready",
    }
    assert "record_count" not in station_trace
    assert "pose_xyz" not in station_trace
    assert "pair_ids" not in station_trace
    assert "detected_isotope_gate" not in station_trace
    assert "shield_view_count_shadow_health" not in station_trace
    for name in (
        "pf_posterior.json",
        "pf_diagnostics.json",
        "pf_state.json",
        "pf_checkpoint.json",
        "pf_particles.npz",
        "closed_loop_result.json",
    ):
        assert (tmp_path / "output" / name).is_file()


def test_failed_acquisition_never_publishes_a_partial_result(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A mismatched response must abort without retaining a partial bundle."""
    from pf import closed_loop

    class _MismatchedRuntimeClient(_FakeRuntimeClient):
        """Return a record whose station identity differs from its request."""

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            """Corrupt the response after normal fake runtime construction."""
            response = super().request(payload)
            record = response["record"]
            assert isinstance(record, dict)
            record["station_id"] = int(payload["station_id"]) + 1
            return response

    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=False)
    _stub_cuda_preflight(monkeypatch)
    _FakePFLiveSession.next_estimator = _FakeEstimator()
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _MismatchedRuntimeClient,
    )
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)

    with pytest.raises(AssertionError):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config,
            output_dir=tmp_path / "output",
            seed=17,
        )

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.bundle-*"))
    assert not list(tmp_path.glob(".output.failed-*"))
    failure_receipts = list(tmp_path.glob(".output.failure-*.json"))
    assert len(failure_receipts) == 1
    failure = json.loads(
        failure_receipts[0].read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["output_bundle_published"] is False
    assert failure["error_type"] == "AssertionError"
    client = _MismatchedRuntimeClient.instance
    assert client is not None
    assert client.aborted is True
    assert client.closed is True


def test_abort_cleanup_failures_preserve_primary_error_and_failure_evidence(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Abort, resource, and CUI errors must not replace acquisition failure."""
    from pf import closed_loop

    class _CleanupFailRuntimeClient(_FakeRuntimeClient):
        """Corrupt acquisition identity and fail both runtime cleanup calls."""

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            """Return one record with a mismatched station identity."""
            response = super().request(payload)
            record = response["record"]
            assert isinstance(record, dict)
            record["station_id"] = int(payload["station_id"]) + 1
            return response

        def abort(self) -> None:
            """Record the abort attempt before raising a secondary error."""
            self.aborted = True
            raise RuntimeError("synthetic runtime abort failure")

        def close(self) -> None:
            """Record the close attempt before raising a secondary error."""
            self.closed = True
            raise OSError("synthetic runtime close failure")

    class _CleanupFailCUI:
        """Fail the renderer cleanup after the primary acquisition error."""

        def close(self) -> None:
            """Raise one deterministic secondary renderer error."""
            raise TimeoutError("synthetic CUI close failure")

    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=False)
    _stub_cuda_preflight(monkeypatch)
    _FakePFLiveSession.next_estimator = _FakeEstimator()
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _CleanupFailRuntimeClient,
    )
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(
        closed_loop,
        "_start_cui_split_view",
        lambda *args, **kwargs: _CleanupFailCUI(),
    )

    with pytest.raises(AssertionError) as raised:
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config,
            output_dir=tmp_path / "output",
            seed=17,
        )

    notes = tuple(getattr(raised.value, "__notes__", ()))
    assert any("runtime_abort" in note for note in notes)
    assert any("resource_close" in note for note in notes)
    assert any("cui_close" in note for note in notes)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.bundle-*"))
    assert not list(tmp_path.glob(".output.failed-*"))
    failure_receipts = list(tmp_path.glob(".output.failure-*.json"))
    assert len(failure_receipts) == 1
    failure = json.loads(
        failure_receipts[0].read_text(encoding="utf-8")
    )
    assert failure["error_type"] == "AssertionError"
    assert [
        row["operation"] for row in failure["secondary_failures"]
    ] == ["runtime_abort", "resource_close", "cui_close"]
    client = _CleanupFailRuntimeClient.instance
    assert client is not None
    assert client.aborted is True
    assert client.closed is True


def test_failure_receipt_error_preserves_primary_error_and_discards_stage(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A failed receipt write must not retain staging or replace the cause."""
    from pf import closed_loop

    class _MismatchedRuntimeClient(_FakeRuntimeClient):
        """Return a record whose station identity differs from its request."""

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            """Corrupt the response after normal fake runtime construction."""
            response = super().request(payload)
            record = response["record"]
            assert isinstance(record, dict)
            record["station_id"] = int(payload["station_id"]) + 1
            return response

    original_atomic_write = closed_loop.atomic_write_bytes

    def fail_receipt_write(path: str | Path, payload: bytes) -> Path:
        """Reject only the standalone failure-receipt publication."""
        target = Path(path)
        if target.name.startswith(".output.failure-"):
            raise OSError("synthetic failure receipt write failure")
        return original_atomic_write(target, payload)

    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=False)
    _stub_cuda_preflight(monkeypatch)
    _FakePFLiveSession.next_estimator = _FakeEstimator()
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _MismatchedRuntimeClient,
    )
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(closed_loop, "atomic_write_bytes", fail_receipt_write)

    with pytest.raises(AssertionError) as raised:
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config,
            output_dir=tmp_path / "output",
            seed=17,
        )

    assert any(
        "failure_receipt_write" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.bundle-*"))
    assert not list(tmp_path.glob(".output.failed-*"))
    assert not list(tmp_path.glob(".output.failure-*.json"))



def test_closed_loop_has_one_pf_without_isotope_gate_rebuild() -> None:
    """The production controller must not rebuild or replay an active-only PF."""
    import inspect

    source = inspect.getsource(run_pf_closed_loop)

    assert "FullSpectrumIsotopeGate" not in source
    assert "detected_isotopes_only" not in source
    assert "inference_isotopes" not in source


def test_cui_port_bind_failure_precedes_runtime_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An occupied fixed CUI port must fail before opening the runtime socket."""
    from pf import closed_loop

    class _NeverConnectedClient:
        """Reject any attempt to connect after the synthetic bind failure."""

        connect_calls = 0

        @classmethod
        def connect(cls, *args: object, **kwargs: object) -> object:
            """Record an invalid connection attempt."""
            del args, kwargs
            cls.connect_calls += 1
            raise AssertionError("Runtime connection must follow the CUI bind.")

    class _NeverConstructedSession:
        """Reject estimator construction before the CUI port is reserved."""

        constructor_calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Record an invalid session-construction attempt."""
            del args, kwargs
            type(self).constructor_calls += 1
            raise AssertionError("PF session must follow the CUI bind.")

    renderer_calls: list[bool] = []
    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=True)
    _stub_cuda_preflight(monkeypatch)
    monkeypatch.setattr(
        closed_loop,
        "start_cui_view_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.EADDRINUSE, "synthetic occupied CUI port")
        ),
    )
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _NeverConnectedClient,
    )
    monkeypatch.setattr(closed_loop, "PFLiveSession", _NeverConstructedSession)
    monkeypatch.setattr(
        closed_loop,
        "AsyncCUISplitPFVisualizer",
        lambda *args, **kwargs: renderer_calls.append(True),
    )

    with pytest.raises(OSError) as error:
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config,
            output_dir=tmp_path / "output",
            seed=17,
        )

    assert error.value.errno == errno.EADDRINUSE
    assert _NeverConnectedClient.connect_calls == 0
    assert _NeverConstructedSession.constructor_calls == 0
    assert renderer_calls == []
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.bundle-*"))


def test_cui_renderer_startup_failure_closes_renderer_and_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A post-spawn renderer failure must release every CUI-owned resource."""
    from pf import closed_loop

    class _OwnedServer:
        """Record release of the pre-bound CUI server."""

        url = "http://example.test:8877/index.html"

        def __init__(self) -> None:
            """Initialize a close counter."""
            self.close_calls = 0

        def close(self) -> None:
            """Release the fake fixed-port server."""
            self.close_calls += 1

    class _OwnedRenderer:
        """Represent a spawned renderer that must be reaped on startup failure."""

        instance: "_OwnedRenderer | None" = None

        def __init__(self, **kwargs: object) -> None:
            """Record construction after runtime context resolution."""
            output_dir = Path(str(kwargs["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = output_dir / "index.html"
            self.close_calls = 0
            type(self).instance = self

        def close(self) -> None:
            """Reap the fake renderer child."""
            self.close_calls += 1

    server = _OwnedServer()
    config = tmp_path / "pf.json"
    _write_one_station_production_config(config, cui_enabled=True)
    _stub_cuda_preflight(monkeypatch)
    _FakePFLiveSession.next_estimator = _FakeEstimator()
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(
        closed_loop,
        "start_cui_view_server",
        lambda *args, **kwargs: server,
    )
    monkeypatch.setattr(
        closed_loop,
        "AsyncCUISplitPFVisualizer",
        _OwnedRenderer,
    )

    def fail_notification(message: str) -> None:
        """Fail after the renderer exists but before startup is returned."""
        del message
        raise RuntimeError("synthetic CUI startup notification failure")

    with pytest.raises(RuntimeError, match="startup notification"):
        run_pf_closed_loop(
            tmp_path / "runtime.sock",
            runtime_root=tmp_path,
            pf_config_path=config,
            output_dir=tmp_path / "output",
            seed=17,
            output_hook=fail_notification,
        )

    renderer = _OwnedRenderer.instance
    assert renderer is not None
    assert renderer.close_calls == 1
    assert server.close_calls == 1
    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.aborted is True
    assert client.closed is True
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.bundle-*"))


def test_pf_closed_loop_starts_truth_free_cui_and_publishes_frames(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The closed-loop entry point must route CUI settings to a sidecar."""
    from pf import closed_loop

    config = tmp_path / "pf.json"
    _write_one_station_production_config(
        config,
        cui_enabled=True,
    )
    _stub_cuda_preflight(monkeypatch)
    estimator = _FakeEstimator()
    fake_log = _FakeLog()
    _FakePFLiveSession.next_estimator = estimator
    frames: list[object] = []
    truth_updates: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    output_messages: list[str] = []
    cui_constructor_kwargs: list[dict[str, object]] = []
    server_close_calls: list[bool] = []
    server_handle = SimpleNamespace(
        url="http://example.test:8877/index.html",
        close=lambda: server_close_calls.append(True),
    )

    class _FakeCUI:
        """Capture CUI frames without spawning a renderer process."""

        index_path = tmp_path / "cui" / "index.html"

        def __init__(self, **kwargs: object) -> None:
            """Record construction arguments for the CUI sidecar."""
            self.kwargs = kwargs
            cui_constructor_kwargs.append(dict(kwargs))
            output_dir = Path(str(kwargs["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            self.latest_overview_path = output_dir / "latest_experiment_overview.png"
            self.latest_robot_path = output_dir / "latest_robot_2d.png"
            self.latest_pf_path = output_dir / "latest_pf_3d.png"
            self.latest_pf_labeled_path = output_dir / "latest_pf_3d_labeled.png"
            self.latest_spectrum_path = output_dir / "latest_spectrum.png"
            for path in (
                self.latest_overview_path,
                self.latest_robot_path,
                self.latest_pf_path,
                self.latest_pf_labeled_path,
                self.latest_spectrum_path,
            ):
                path.write_bytes(path.name.encode())

        def update(self, frame: object) -> None:
            """Record one CUI frame."""
            frames.append(frame)

        def set_truth(
            self,
            true_sources: dict[str, np.ndarray],
            true_strengths: dict[str, np.ndarray],
        ) -> None:
            """Record one CUI-only truth update."""
            truth_updates.append((true_sources, true_strengths))

        def close(self) -> None:
            """Provide the production CUI lifecycle interface."""

    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "PFLiveSession", _FakePFLiveSession)
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)
    monkeypatch.setattr(closed_loop, "AsyncCUISplitPFVisualizer", _FakeCUI)
    monkeypatch.setattr(
        closed_loop,
        "build_frame_from_pf",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        closed_loop,
        "start_cui_view_server",
        lambda *args, **kwargs: server_handle,
    )

    run_pf_closed_loop(
        tmp_path / "runtime.sock",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
        seed=17,
        output_hook=output_messages.append,
    )

    assert len(frames) == 2
    assert np.asarray(frames[0].path_waypoints_xyz).shape == (2, 3)
    np.testing.assert_array_equal(
        frames[0].cui_route.measurement_visit_counts,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        frames[1].cui_route.measurement_visit_counts,
        np.asarray([1], dtype=np.int64),
    )
    assert truth_updates == []
    assert len(cui_constructor_kwargs) == 1
    assert cui_constructor_kwargs[0]["save_step_history"] is False
    assert server_close_calls == [True]
    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.overlay_requests == []
    assert (
        "CUI split visualization URL: http://example.test:8877/index.html"
    ) in output_messages
    enabled_message = next(
        message
        for message in output_messages
        if message.startswith("CUI split visualization enabled:")
    )
    for filename in (
        "latest_experiment_overview.png",
        "latest_robot_2d.png",
        "latest_pf_3d.png",
        "latest_pf_3d_labeled.png",
        "latest_spectrum.png",
    ):
        assert filename in enabled_message
    assert (tmp_path / "output" / "final_experiment_overview.png").is_file()
    assert (tmp_path / "output" / "final_robot_2d.png").is_file()
    assert (tmp_path / "output" / "final_pf_3d.png").is_file()
    assert (tmp_path / "output" / "final_pf_3d_labeled.png").is_file()
    assert (tmp_path / "output" / "final_spectrum.png").is_file()
