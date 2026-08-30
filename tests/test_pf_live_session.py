"""Tests for the live PF runtime-ingestion boundary."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from runtime.adaptive_client import AdaptiveCandidateSnapshot, AdaptiveStepRequest
from runtime.assets import simulation_runtime_root
from runtime.measurement_log import load_measurement_log
from runtime.prefix import measurement_records_digest
from runtime.provenance import DigestIdentity, strict_canonical_json_bytes

from pf.control_policy import PFControlPolicyProvenance
from pf.live_session import (
    PFExternalSurfaceGuidance,
    PFLiveSession,
    PFLiveSessionError,
    ValidatedProductionPFConfig,
    _compact_pf_diagnostics,
    _strict_live_artifact_json_bytes,
    build_live_estimator,
    load_production_live_pf_config,
    measurement_record_to_station_input,
)
from pf.estimator_structural import EstimatorStructuralProposalMixin
from pf.estimator_types import JointPlanningParticles
from pf.pure_estimator import RotatingShieldPFConfig
from pf.provenance import strict_sha256_json
from tests.pure_pf_test_support import make_measurement_log


def _production_live_config() -> dict[str, Any]:
    """Return a detached copy of the shipped complete schema-v2 config."""
    path = Path(__file__).parents[1] / "configs/pf/pf_strict_3d.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validated_test_config(
    path: Path,
    *,
    num_particles: int | None = None,
) -> ValidatedProductionPFConfig:
    """Write and load one complete provenance-bound production config."""
    payload = _production_live_config()
    if num_particles is not None:
        payload["num_particles"] = num_particles
        payload["dss_pp"]["planning_particles"] = max(
            2,
            num_particles // 2,
        )
        payload["dss_pp"]["proxy_planning_particles"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_production_live_pf_config(path, profile="pf_strict")


def test_resolved_session_hash_changes_with_only_control_policy() -> None:
    """Control-policy content must be part of the resolved live identity."""
    from hashlib import sha256

    from pf.live_session import _live_session_hash_payload

    canonical_policy = strict_canonical_json_bytes(
        {
            "schema_version": 2,
            "variant": "proposed",
            "shield_policy": None,
        }
    )
    external = PFControlPolicyProvenance(
        policy_family="ral_ablation",
        source_sha256="b" * 64,
        canonical_sha256=sha256(canonical_policy).hexdigest(),
        canonical_policy_json=canonical_policy,
    )
    common = {
        "runtime_config_sha256": "a" * 64,
        "measurement_log_sha256": "unavailable",
        "actual_pf": {"num_particles": 100},
        "random_seed": 7,
    }

    native_hash = strict_sha256_json(
        _live_session_hash_payload(
            **common,
            control_policy_provenance=PFControlPolicyProvenance.native_dss_pp(),
        )
    )
    external_hash = strict_sha256_json(
        _live_session_hash_payload(
            **common,
            control_policy_provenance=external,
        )
    )

    assert native_hash != external_hash


@pytest.mark.parametrize(
    "payload",
    [
        {"path": Path("implicit-path.json")},
        {"numpy_scalar": np.int64(7)},
        {1: "non-string-key"},
        {"arbitrary_object": object()},
    ],
    ids=("path", "numpy-scalar", "non-string-key", "arbitrary-object"),
)
def test_live_artifact_boundary_rejects_lossy_json_values(
    payload: object,
) -> None:
    """Production live artifacts must reject every implicit JSON coercion."""
    with pytest.raises(
        PFLiveSessionError,
        match="must contain only strict finite JSON values",
    ):
        _strict_live_artifact_json_bytes(
            payload,
            artifact_name="Test live artifact",
        )


def test_live_builder_uses_run_context_without_synthetic_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live construction resolves its forward model directly from context."""
    from pf import live_session

    context = SimpleNamespace(
        schema_version=2,
        runtime_config_sha256="a" * 64,
    )
    forward = object()
    estimator = object()
    calls: dict[str, object] = {}

    class _Resolver:
        """Capture direct shared-runtime context resolution."""

        @classmethod
        def from_run_context(
            cls,
            actual_context: object,
            *,
            run_root: object,
        ) -> object:
            """Return the sentinel authenticated forward context."""
            del cls
            calls["context"] = actual_context
            calls["run_root"] = run_root
            return forward

    def build_from_forward(
        actual_forward: object,
        config: object,
        **kwargs: object,
    ) -> object:
        """Capture the thin PF-specific construction adapter."""
        calls["forward"] = actual_forward
        calls["config"] = config
        calls["kwargs"] = kwargs
        return estimator

    monkeypatch.setattr(live_session, "ResolvedForwardContext", _Resolver)
    monkeypatch.setattr(
        live_session,
        "_build_live_estimator_from_forward_context",
        build_from_forward,
    )

    validated_config = _validated_test_config(tmp_path / "pf.json")
    control_provenance = PFControlPolicyProvenance.native_dss_pp()
    actual = build_live_estimator(
        context,  # type: ignore[arg-type]
        validated_config,
        profile="pf_strict",
        seed=9,
        runtime_root=tmp_path,
        control_policy_provenance=control_provenance,
    )

    assert actual is estimator
    assert calls["context"] is context
    assert calls["run_root"] == tmp_path.resolve()
    assert calls["forward"] is forward
    assert calls["kwargs"] == {
        "profile": "pf_strict",
        "seed": 9,
        "measurement_log_schema_version": 2,
        "measurement_runtime_config_sha256": "a" * 64,
        "control_policy_provenance": control_provenance,
    }


def test_live_builder_preserves_exact_runtime_observation_contract(
    tmp_path: Path,
) -> None:
    """The production PF must consume one unmodified runtime physics object."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            runtime_overrides={
                "fe_shield_thickness_cm": 1.25,
                "pb_shield_thickness_cm": 1.5,
                "buildup": {
                    "fe_coeff": 0.125,
                    "pb_coeff": 0.25,
                    "obstacle_coeff": 0.375,
                },
            },
        )
    )
    config = _validated_test_config(
        tmp_path / "pf.json",
        num_particles=16,
    )

    estimator = build_live_estimator(
        log.context,
        config,
        profile="pf_strict",
        seed=17,
        runtime_root=simulation_runtime_root(),
        control_policy_provenance=PFControlPolicyProvenance.native_dss_pp(),
    )
    kernel = estimator.continuous_kernel(use_gpu=False)

    assert estimator.shield_params.thickness_fe_cm == 1.25
    assert estimator.shield_params.thickness_pb_cm == 1.5
    assert estimator.shield_params.inner_radius_fe_cm == 3.95
    assert estimator.shield_params.inner_radius_pb_cm == 5.2
    assert estimator.shield_params.buildup_fe_coeff == 0.125
    assert estimator.shield_params.buildup_pb_coeff == 0.25
    assert estimator.obstacle_buildup_coeff == 0.375
    assert kernel.shield_params == estimator.shield_params
    assert kernel.strict_catalog_line_contract is True
    assert kernel.dry_air_total_attenuation_contract_id is not None
    assert kernel.dry_air_total_attenuation_contract_sha256 is not None
    assert estimator.joint_transport_cache_preflight is not None
    assert (
        estimator.joint_transport_cache_preflight["allocated_bytes"]
        == estimator.joint_transport_cache_preflight["cache_required_bytes"]
    )
    assert (
        estimator.joint_transport_cache_preflight["required_bytes"]
        > estimator.joint_transport_cache_preflight["allocated_bytes"]
    )


def test_context_energy_dimensions_require_exact_identity() -> None:
    """A nearly equal model endpoint must not authorize another energy axis."""
    from pf import live_session

    context = SimpleNamespace(
        runtime_config={
            "full_spectrum_generative_model": {
                "energy_bin_count": 2,
                "energy_min_keV": 0.0,
                "energy_max_keV": np.nextafter(2.0, 3.0),
                "bin_width_keV": 2.0,
            }
        }
    )

    with pytest.raises(PFLiveSessionError, match="dimensions are inconsistent"):
        live_session._context_energy_bin_edges(context)


def test_live_pf_config_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    """The public in-process config API must never discard unknown options."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(json.dumps(_production_live_config()), encoding="utf-8")

    loaded = load_production_live_pf_config(
        config_path,
        profile="pf_strict",
    )

    payload = loaded.settings()
    assert payload["num_particles"] == 4096
    assert len(loaded.source_sha256) == 64
    payload["num_particels"] = payload.pop("num_particles")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PFLiveSessionError, match="unknown_or_retired.*num_particels"):
        load_production_live_pf_config(config_path, profile="pf_strict")


def test_validated_live_config_rejects_caller_minting(tmp_path: Path) -> None:
    """Only the production loader may mint the live-config capability."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(json.dumps(_production_live_config()), encoding="utf-8")
    loaded = load_production_live_pf_config(config_path, profile="pf_strict")

    with pytest.raises(PFLiveSessionError, match="load_production_live_pf_config"):
        ValidatedProductionPFConfig(
            document=loaded.document,
            profile="pf_strict",
            _validation_token=object(),
        )


def test_live_pf_config_validates_nested_adaptive_stop_thresholds(
    tmp_path: Path,
) -> None:
    """Adaptive-stop thresholds must be validated from their single block."""
    config_path = tmp_path / "pf.json"
    payload = _production_live_config()
    payload["adaptive_stop"][
        "minimum_joint_map_cardinality_probability"
    ] = 1.1
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PFLiveSessionError, match="incompatible"):
        load_production_live_pf_config(config_path, profile="pf_strict")


def test_live_pf_config_validates_planner_values_before_runtime_connection(
    tmp_path: Path,
) -> None:
    """Malformed planner values must fail in the file loader itself."""
    config_path = tmp_path / "pf.json"
    payload = _production_live_config()
    payload["dss_pp"]["proxy_eig_samples"] = "2"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PFLiveSessionError, match="proxy_eig_samples"):
        load_production_live_pf_config(config_path, profile="pf_strict")


def test_live_pf_config_rejects_top_level_adaptive_stop_threshold(
    tmp_path: Path,
) -> None:
    """Estimator-facing stop settings must not duplicate the nested block."""
    config_path = tmp_path / "pf.json"
    payload = _production_live_config()
    payload["adaptive_stop_innovation_confidence"] = 0.99
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PFLiveSessionError, match="unknown_or_retired"):
        load_production_live_pf_config(config_path, profile="pf_strict")


def test_live_record_forwards_only_raw_spectrum_and_action_geometry(
    tmp_path: Path,
) -> None:
    """Live ingestion forwards the raw spectrum and selected shield action."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    station_input = measurement_record_to_station_input(log.records[0])

    assert len(station_input) == 4
    spectrum, fe_index, pb_index, live_time_s = station_input
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.dtype == np.int64
    np.testing.assert_array_equal(spectrum, log.records[0].spectrum_counts)
    assert (fe_index, pb_index, live_time_s) == (0, 0, 1.0)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("spectrum_counts", np.asarray([1.0, 0.5], dtype=np.float64)),
        ("spectrum_counts", np.asarray([1, 2], dtype=np.int32)),
        ("fe_orientation_index", True),
        ("pb_orientation_index", "0"),
        ("live_time_s", True),
        ("live_time_s", "1.0"),
    ),
)
def test_live_record_rejects_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """The live adapter preserves raw observation types without coercion."""
    values: dict[str, Any] = {
        "spectrum_counts": np.asarray([1, 2], dtype=np.int64),
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "live_time_s": 1.0,
    }
    values[field_name] = invalid

    with pytest.raises(PFLiveSessionError):
        measurement_record_to_station_input(SimpleNamespace(**values))


class _SpyPosterior:
    """Serialize the final identities supplied by a spy PF estimator."""

    def __init__(self, estimator: "_SpyEstimator") -> None:
        """Retain the estimator whose bound identities are reported."""
        self.estimator = estimator

    def to_dict(self) -> dict[str, object]:
        """Return a complete package-owned PF posterior contract."""
        return {
            "schema_version": 2,
            "estimator_family": "particle_filter",
            "estimator_profile": "pf_strict",
            "final_estimate_source": "pf_posterior",
            "uses_all_history_batch_fit": False,
            "uses_surface_map": False,
            "uses_batch_model_order": False,
            "record_count": len(self.estimator.measurements),
            "isotopes": {
                isotope: {
                    "map_cardinality": 1,
                    "cardinality_distribution": {"0": 0.0, "1": 1.0},
                    "modes": [
                        {
                            "position_mean_xyz": [0.1, 0.2, 0.3],
                            "position_covariance_xyz": [
                                [0.1, 0.0, 0.0],
                                [0.0, 0.1, 0.0],
                                [0.0, 0.0, 0.1],
                            ],
                            "strength_mean_cps_1m": 2.0,
                            "posterior_mass": 1.0,
                        }
                    ],
                }
                for isotope in self.estimator.isotopes
            },
            "provenance": {
                "estimator_repository": "Rotating-shield-particle-filter",
                "estimator_commit": "a" * 40,
                "measurement_log_schema_version": 2,
                "measurement_log_sha256": self.estimator.measurement_log_sha256,
                "resolved_config_sha256": self.estimator.resolved_config_hash,
                "config_sha256": "b" * 64,
                "random_seed": self.estimator.random_seed,
                "planner_belief_sources": ["pf_posterior"],
                "batch_feedback_applied": False,
            },
        }


class _SpyEstimator:
    """Record station updates without running PF science in facade tests."""

    def __init__(self, isotopes: tuple[str, ...], contract_hash: str) -> None:
        """Initialize mutable fields required by the existing live helpers."""
        self.isotopes = isotopes
        self.measurements: list[object] = []
        self.poses = [np.asarray([0.25, 0.25, 0.4], dtype=np.float64)]
        self.kernel_cache: object | None = object()
        self.pf_config = RotatingShieldPFConfig()
        self.random_seed = 17
        self.measurement_log_sha256 = "unavailable"
        self.resolved_config_hash = "unavailable"
        self.full_spectrum_generative_model = SimpleNamespace(
            contract_hash_sha256=contract_hash
        )
        self.update_calls: list[dict[str, object]] = []
        self.planning_calls = 0
        self.posterior_summary_calls = 0
        self.particles = JointPlanningParticles(
            isotope_order=isotopes,
            weights_n=np.asarray([0.75, 0.25], dtype=np.float64),
            positions_nk3_by_isotope={
                isotope: np.asarray(
                    [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]],
                    dtype=np.float64,
                )
                for isotope in isotopes
            },
            surface_chart_ids_nk_by_isotope={
                isotope: np.asarray([[1], [2]], dtype=np.int64)
                for isotope in isotopes
            },
            surface_uv_nk2_by_isotope={
                isotope: np.asarray(
                    [[[0.1, 0.2]], [[0.3, 0.4]]],
                    dtype=np.float64,
                )
                for isotope in isotopes
            },
            strengths_nk_by_isotope={
                isotope: np.asarray([[2.0], [3.0]], dtype=np.float64)
                for isotope in isotopes
            },
            source_mask_nk_by_isotope={
                isotope: np.asarray([[True], [True]], dtype=np.bool_)
                for isotope in isotopes
            },
            original_particle_indices=np.asarray([4, 7], dtype=np.int64),
        )

    def joint_isotope_order(self) -> tuple[str, ...]:
        """Return the spy's active isotope order."""
        return self.isotopes

    def add_measurement_pose(
        self,
        pose: np.ndarray,
        *,
        reset_filters: bool,
    ) -> None:
        """Append a copied detector pose without resetting state."""
        assert reset_filters is False
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    def update_spectrum_station(
        self,
        records: tuple[tuple[object, ...], ...],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> None:
        """Record one station update and expose its assimilated row count."""
        self.update_calls.append(
            {
                "records": records,
                "pose_idx": pose_idx,
                "contract": generative_contract_hash_sha256,
                "surface_guidance": getattr(
                    self,
                    "_joint_external_surface_guidance_by_isotope",
                    None,
                ),
                "surface_guidance_mass": getattr(
                    self,
                    "_joint_external_surface_guidance_mass",
                    0.0,
                ),
            }
        )
        guidance = getattr(self, "_joint_external_surface_guidance_by_isotope", None)
        if guidance is not None:
            self.last_external_surface_guidance_evaluated_isotopes = set(guidance)
        self.measurements.extend(object() for _ in records)

    def continuous_surface_atlas(self) -> SimpleNamespace:
        """Return two deterministic chart centers for guidance interpolation."""
        return SimpleNamespace(
            geometry=SimpleNamespace(
                centers_xyz=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            )
        )

    def planning_joint_particles(
        self,
        *,
        max_particles: int | None,
        method: str | None,
        rng: np.random.Generator | None,
    ) -> JointPlanningParticles:
        """Return mutable arrays so the facade must copy them."""
        del max_particles, method, rng
        self.planning_calls += 1
        return self.particles

    def posterior_point_estimate(self) -> dict[str, SimpleNamespace]:
        """Return one existing PF summary contract for the planning DTO."""
        self.posterior_summary_calls += 1
        estimates: dict[str, SimpleNamespace] = {}
        for isotope in self.isotopes:
            mode = SimpleNamespace(
                label_index=0,
                position_medoid_xyz=(0.1, 0.2, 0.3),
                strength_representative_cps_1m=2.0,
            )
            estimates[isotope] = SimpleNamespace(
                modes=(mode,),
                to_dict=lambda isotope=isotope: {
                    "map_cardinality": 1,
                    "cardinality_distribution": {"0": 0.1, "1": 0.9},
                    "selected_stratum_mass": 0.9,
                    "modes": [
                        {
                            "label_index": 0,
                            "position_medoid_xyz": [0.1, 0.2, 0.3],
                            "position_covariance_xyz": [
                                [0.1, 0.0, 0.0],
                                [0.0, 0.1, 0.0],
                                [0.0, 0.0, 0.1],
                            ],
                            "credible_radius_95_m": 0.5,
                            "strength_representative_cps_1m": 2.0,
                            "strength_mean_cps_1m": 2.1,
                            "strength_median_cps_1m": 2.0,
                            "strength_credible_interval_95_cps_1m": [1.0, 3.0],
                            "posterior_mass": 0.9,
                            "belief_source": "pf_posterior",
                        }
                    ],
                    "isotope": isotope,
                },
            )
        return estimates

    def source_response_signatures(
        self,
        isotope: str,
        positions_xyz_m: np.ndarray,
    ) -> np.ndarray:
        """Return normalized truth-free response signatures for test modes."""
        assert isotope in self.isotopes
        positions = np.asarray(positions_xyz_m, dtype=np.float64).reshape(-1, 3)
        if positions.shape[0] == 0:
            return np.zeros((2, 0), dtype=np.float64)
        signatures = np.vstack(
            (
                1.0 + positions[:, 0],
                1.0 + positions[:, 1],
            )
        )
        return signatures / np.linalg.norm(signatures, axis=0, keepdims=True)

    def serialized_state(self) -> bytes:
        """Return state bytes depending only on completed station updates."""
        return json.dumps(
            {
                "schema_version": 1,
                "measurement_count": len(self.measurements),
                "station_update_count": len(self.update_calls),
                "rng_states": {"joint": {"state": 17}},
            },
            sort_keys=True,
        ).encode("utf-8")

    def posterior_snapshot(self) -> _SpyPosterior:
        """Return a result view using identities set by log binding."""
        return _SpyPosterior(self)

    def posterior_convergence_diagnostics(self) -> dict[str, object]:
        """Return compactable truth-free convergence evidence."""
        sampler_health = {
            "smc_rejuvenation_wall_time_respected": True,
            "rejuvenation_mixing_complete": True,
            "structural_mixing_complete": True,
        }
        return {
            "ready": False,
            "sampler_health": sampler_health,
            "joint_gates": {
                **sampler_health,
                "joint_map_cardinality_probability": False,
            },
            "isotopes": {
                isotope: {
                    "cardinality_distribution": {0: 0.1, 1: 0.9},
                    "gates": {
                        "cardinality_not_at_upper_boundary": True,
                        "surface_path_concentration": False,
                    }
                }
                for isotope in self.isotopes
            },
        }

    def posterior_predictive_check(self) -> dict[str, object]:
        """Return one unavailable predictive-check result."""
        return {"available": False}


def test_compact_diagnostics_encodes_validated_cardinality_keys() -> None:
    """The live schema must explicitly encode known integer cardinalities."""
    estimator = _SpyEstimator(("Cs-137",), "a" * 64)

    payload = _compact_pf_diagnostics(estimator)

    distribution = payload["posterior_convergence"]["isotopes"]["Cs-137"][
        "cardinality_distribution"
    ]
    assert distribution == {"0": 0.1, "1": 0.9}
    _strict_live_artifact_json_bytes(
        payload,
        artifact_name="PF diagnostics",
    )


def _facade_with_spy(
    monkeypatch: pytest.MonkeyPatch,
    log: object,
    *,
    initial_candidates: AdaptiveCandidateSnapshot | None = None,
    control_policy_provenance: PFControlPolicyProvenance | None = None,
) -> tuple[PFLiveSession, _SpyEstimator]:
    """Construct a facade while replacing only the expensive PF estimator."""
    from pf import live_session

    context = log.context
    contract_hash = str(
        context.runtime_config["full_spectrum_contract_hash_sha256"]
    )
    estimator = _SpyEstimator(tuple(context.isotopes), contract_hash)

    def build_spy(
        actual_context: object,
        config: object,
        **kwargs: object,
    ) -> _SpyEstimator:
        """Verify authenticated construction inputs and return the spy."""
        assert actual_context is context
        assert isinstance(config, ValidatedProductionPFConfig)
        assert config.settings()["pure_pf_schema_version"] == 2
        assert kwargs["profile"] == "pf_strict"
        assert kwargs["seed"] == 17
        return estimator

    monkeypatch.setattr(live_session, "build_live_estimator", build_spy)
    config = _validated_test_config(
        Path(log.path).parent / f".{Path(log.path).name}.pf-test.json"
    )
    resolved_control_provenance = (
        PFControlPolicyProvenance.native_dss_pp()
        if control_policy_provenance is None
        else control_policy_provenance
    )
    if initial_candidates is None:
        initial_pose = tuple(
            float(value) for value in context.environment["detector_position"]
        )
        initial_candidates = AdaptiveCandidateSnapshot(
            current_pose_xyz=initial_pose,
            candidate_poses_xyz=(initial_pose,),
            travel_costs=(0.0,),
            allowed_pair_ids=tuple(range(64)),
            current_pair_id=0,
            shield_angular_speed_rad_s=1.0,
            horizontal_travel_times_s=(0.0,),
            mast_vertical_times_s=(0.0,),
            settling_times_s=(0.0,),
        )
    session = PFLiveSession(
        context,
        config,
        initial_candidates=initial_candidates,
        profile="pf_strict",
        seed=17,
        runtime_root=log.path,
        control_policy_provenance=resolved_control_provenance,
    )
    return session, estimator


def test_facade_assimilates_only_complete_persisted_stations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record delivery must use one canonical station update at its marker."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)

    assert session.receive_persisted(log.records[0]) is False
    assert estimator.update_calls == []
    with pytest.raises(PFLiveSessionError, match="station boundary"):
        session.planning_particle_snapshot()

    assert session.receive_persisted(log.records[1]) is True
    assert len(estimator.update_calls) == 1
    assert estimator.update_calls[0]["contract"] == (
        log.runtime_config["full_spectrum_contract_hash_sha256"]
    )
    assert session.records == log.records
    assert session.station_count == 1


def _response_candidates(
    record: object,
    *,
    current_pair_id: int | None = None,
    pose_xyz: tuple[float, float, float] | None = None,
    allowed_pair_ids: tuple[int, ...] | None = None,
    shield_angular_speed_rad_s: float = 1.0,
) -> AdaptiveCandidateSnapshot:
    """Return one strict next-candidate snapshot for an acquired record."""
    record_pose = tuple(float(value) for value in record.detector_pose_xyz)
    pose = record_pose if pose_xyz is None else pose_xyz
    pair_id = (
        int(record.fe_orientation_index) * 8 + int(record.pb_orientation_index)
        if current_pair_id is None
        else current_pair_id
    )
    return AdaptiveCandidateSnapshot(
        current_pose_xyz=pose,
        candidate_poses_xyz=(pose,),
        travel_costs=(0.0,),
        allowed_pair_ids=(
            tuple(range(64)) if allowed_pair_ids is None else allowed_pair_ids
        ),
        current_pair_id=pair_id,
        shield_angular_speed_rad_s=shield_angular_speed_rad_s,
        horizontal_travel_times_s=(0.0,),
        mast_vertical_times_s=(0.0,),
        settling_times_s=(0.0,),
    )


def _request_candidates(
    record: object,
    *,
    current_pose_xyz: tuple[float, float, float] | None = None,
) -> AdaptiveCandidateSnapshot:
    """Return the exact pre-action snapshot that selects the record pose."""
    target_pose = tuple(float(value) for value in record.detector_pose_xyz)
    current_pose = target_pose if current_pose_xyz is None else current_pose_xyz
    travel_time_s = float(record.travel_time_s)
    if current_pose == target_pose:
        alternate_pose = (target_pose[0] + 0.25, target_pose[1], target_pose[2])
        poses = (target_pose, alternate_pose)
        travel_costs = (0.0, travel_time_s)
    else:
        poses = (current_pose, target_pose)
        travel_costs = (0.0, travel_time_s)
    return AdaptiveCandidateSnapshot(
        current_pose_xyz=current_pose,
        candidate_poses_xyz=poses,
        travel_costs=travel_costs,
        allowed_pair_ids=tuple(range(64)),
        current_pair_id=0,
        shield_angular_speed_rad_s=1.0,
        horizontal_travel_times_s=travel_costs,
        mast_vertical_times_s=(0.0, 0.0),
        settling_times_s=(0.0, 0.0),
    )


def _selected_candidate_index(
    record: object,
    candidates: AdaptiveCandidateSnapshot,
) -> int:
    """Return the exact candidate index for one fixture record pose."""
    target = tuple(float(value) for value in record.detector_pose_xyz)
    return candidates.candidate_poses_xyz.index(target)


def test_acquired_response_is_validated_before_canonical_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An exact action response must enter the session and update only once."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    raw_record = log.records[0]
    request_candidates = _request_candidates(raw_record)
    session, estimator = _facade_with_spy(
        monkeypatch,
        log,
        initial_candidates=request_candidates,
    )
    requested_pair_id = (
        int(raw_record.fe_orientation_index) * 8
        + int(raw_record.pb_orientation_index)
    )
    candidate_index = _selected_candidate_index(raw_record, request_candidates)
    record = replace(
        raw_record,
        travel_time_s=request_candidates.travel_costs[candidate_index],
        shield_actuation_time_s=(
            request_candidates.quote_shield_program_time_s((requested_pair_id,))
        ),
    )
    request = AdaptiveStepRequest(
        action_id=0,
        candidate_index=candidate_index,
        fe_orientation_index=record.fe_orientation_index,
        pb_orientation_index=record.pb_orientation_index,
        dwell_time_s=record.live_time_s,
        station_id=record.station_id,
        station_complete=True,
    )

    completed = session.receive_acquired(
        record,
        request=request,
        request_candidates=request_candidates,
        next_candidates=_response_candidates(record),
    )

    assert completed is True
    assert session.records == (record,)
    assert session.station_count == 1
    assert len(estimator.update_calls) == 1


def test_session_owns_candidate_refinement_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A refinement event must extend exactly the session-owned snapshot."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    initial = _request_candidates(log.records[0])
    session, _estimator = _facade_with_spy(
        monkeypatch,
        log,
        initial_candidates=initial,
    )
    target = initial.candidate_poses_xyz[-1]
    added_pose = (target[0], target[1] + 0.25, target[2])
    refined = AdaptiveCandidateSnapshot(
        current_pose_xyz=initial.current_pose_xyz,
        candidate_poses_xyz=(*initial.candidate_poses_xyz, added_pose),
        travel_costs=(*initial.travel_costs, 0.5),
        allowed_pair_ids=initial.allowed_pair_ids,
        current_pair_id=initial.current_pair_id,
        shield_angular_speed_rad_s=initial.shield_angular_speed_rad_s,
        horizontal_travel_times_s=(*initial.horizontal_travel_times_s, 0.5),
        mast_vertical_times_s=(*initial.mast_vertical_times_s, 0.0),
        settling_times_s=(*initial.settling_times_s, 0.0),
    )

    session.receive_refined_candidates(refined)

    assert session.phase == "receiving"
    with pytest.raises(RuntimeError, match="did not add"):
        session.receive_refined_candidates(refined)
    assert session.phase == "failed"
    assert session.records == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("allowed_pairs", "all 64 shield pairs"),
        ("shield_speed", "shield speed differs"),
    ),
)
def test_handshake_candidates_match_the_runtime_motion_contract(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Initial candidates cannot rewrite shield availability or motion speed."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    candidates = _request_candidates(log.records[0])
    if mutation == "allowed_pairs":
        candidates = replace(candidates, allowed_pair_ids=tuple(reversed(range(64))))
    else:
        candidates = replace(candidates, shield_angular_speed_rad_s=2.0)

    with pytest.raises(PFLiveSessionError, match=message):
        _facade_with_spy(
            monkeypatch,
            log,
            initial_candidates=candidates,
        )


def test_acquired_response_accepts_equivalent_quaternion_sign(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The response may use either sign for the same commanded orientation."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    raw_record = log.records[0]
    request_candidates = _request_candidates(raw_record)
    session, estimator = _facade_with_spy(
        monkeypatch,
        log,
        initial_candidates=request_candidates,
    )
    requested_pair_id = (
        int(raw_record.fe_orientation_index) * 8
        + int(raw_record.pb_orientation_index)
    )
    candidate_index = _selected_candidate_index(raw_record, request_candidates)
    record = replace(
        raw_record,
        detector_quat_wxyz=(-1.0, 0.0, 0.0, 0.0),
        travel_time_s=request_candidates.travel_costs[candidate_index],
        shield_actuation_time_s=(
            request_candidates.quote_shield_program_time_s((requested_pair_id,))
        ),
    )
    request = AdaptiveStepRequest(
        action_id=0,
        candidate_index=candidate_index,
        fe_orientation_index=record.fe_orientation_index,
        pb_orientation_index=record.pb_orientation_index,
        dwell_time_s=record.live_time_s,
        station_id=record.station_id,
        station_complete=True,
    )

    completed = session.receive_acquired(
        record,
        request=request,
        request_candidates=request_candidates,
        next_candidates=_response_candidates(record),
    )

    assert completed is True
    assert session.records == (record,)
    assert len(estimator.update_calls) == 1


def test_zero_motion_response_preserves_previous_commanded_yaw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A shield-only follow-up must retain the preceding arrival yaw."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    half_sqrt_two = math.sqrt(0.5)
    yaw_quaternion = (half_sqrt_two, 0.0, 0.0, half_sqrt_two)

    raw_first = log.records[0]
    initial_pose = tuple(
        float(value) for value in log.context.environment["detector_position"]
    )
    target_pose = (initial_pose[0], initial_pose[1] + 0.25, initial_pose[2])
    first = replace(
        raw_first,
        detector_pose_xyz=target_pose,
        detector_quat_wxyz=yaw_quaternion,
    )
    first_candidates = _request_candidates(
        first,
        current_pose_xyz=initial_pose,
    )
    session, estimator = _facade_with_spy(
        monkeypatch,
        log,
        initial_candidates=first_candidates,
    )
    first_pair_id = (
        int(first.fe_orientation_index) * 8
        + int(first.pb_orientation_index)
    )
    first = replace(
        first,
        shield_actuation_time_s=(
            first_candidates.quote_shield_program_time_s((first_pair_id,))
        ),
    )
    first_request = AdaptiveStepRequest(
        action_id=0,
        candidate_index=_selected_candidate_index(first, first_candidates),
        fe_orientation_index=first.fe_orientation_index,
        pb_orientation_index=first.pb_orientation_index,
        dwell_time_s=first.live_time_s,
        station_id=first.station_id,
        station_complete=False,
    )
    second_candidates = _response_candidates(first)

    assert session.receive_acquired(
        first,
        request=first_request,
        request_candidates=first_candidates,
        next_candidates=second_candidates,
    ) is False
    assert estimator.update_calls == []

    raw_second = log.records[1]
    second_pair_id = (
        int(raw_second.fe_orientation_index) * 8
        + int(raw_second.pb_orientation_index)
    )
    second = replace(
        raw_second,
        detector_pose_xyz=target_pose,
        detector_quat_wxyz=yaw_quaternion,
        travel_time_s=0.0,
        shield_actuation_time_s=(
            second_candidates.quote_shield_program_time_s((second_pair_id,))
        ),
    )
    second_request = AdaptiveStepRequest(
        action_id=1,
        candidate_index=0,
        fe_orientation_index=second.fe_orientation_index,
        pb_orientation_index=second.pb_orientation_index,
        dwell_time_s=second.live_time_s,
        station_id=second.station_id,
        station_complete=True,
    )

    assert session.receive_acquired(
        second,
        request=second_request,
        request_candidates=second_candidates,
        next_candidates=_response_candidates(second),
    ) is True
    assert session.records == (first, second)
    assert len(estimator.update_calls) == 1


@pytest.mark.parametrize(
    "mismatch",
    (
        "request.action_id",
        "request.candidate_index",
        "request_candidates",
        "step_id",
        "action_id",
        "station_id",
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "fe_orientation_index",
        "pb_orientation_index",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "station_complete",
        "energy_bin_edges_keV",
        "full_spectrum_contract_hash_sha256",
        "candidates.current_pair_id",
        "candidates.current_pose",
        "candidates.allowed_pair_ids",
        "candidates.shield_angular_speed_rad_s",
    ),
)
def test_acquired_response_mismatch_fails_before_ingestion(
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every request, spectrum-axis, contract, and candidate mismatch must abort."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=1,
            station_complete_markers=True,
        )
    )
    raw_original = log.records[0]
    request_candidates = _request_candidates(raw_original)
    session, estimator = _facade_with_spy(
        monkeypatch,
        log,
        initial_candidates=request_candidates,
    )
    requested_pair_id = (
        int(raw_original.fe_orientation_index) * 8
        + int(raw_original.pb_orientation_index)
    )
    candidate_index = _selected_candidate_index(raw_original, request_candidates)
    original = replace(
        raw_original,
        travel_time_s=request_candidates.travel_costs[candidate_index],
        shield_actuation_time_s=(
            request_candidates.quote_shield_program_time_s((requested_pair_id,))
        ),
    )
    request = AdaptiveStepRequest(
        action_id=0,
        candidate_index=candidate_index,
        fe_orientation_index=original.fe_orientation_index,
        pb_orientation_index=original.pb_orientation_index,
        dwell_time_s=original.live_time_s,
        station_id=original.station_id,
        station_complete=True,
    )
    record = original
    candidates = _response_candidates(original)
    if mismatch == "request.action_id":
        request = replace(request, action_id=1)
    elif mismatch == "request.candidate_index":
        request = replace(request, candidate_index=2)
    elif mismatch == "request_candidates":
        request_candidates = replace(
            request_candidates,
            current_pair_id=(request_candidates.current_pair_id + 2) % 64,
        )
    elif mismatch == "step_id":
        record = replace(record, step_id=1)
    elif mismatch == "action_id":
        record = replace(record, action_id=1)
    elif mismatch == "station_id":
        record = replace(record, station_id=1)
    elif mismatch == "detector_pose_xyz":
        record = replace(record, detector_pose_xyz=(0.25, 0.5, 0.5))
    elif mismatch == "detector_quat_wxyz":
        record = replace(record, detector_quat_wxyz=(0.0, 0.0, 0.0, 1.0))
    elif mismatch == "fe_orientation_index":
        record = replace(record, fe_orientation_index=1)
    elif mismatch == "pb_orientation_index":
        record = replace(record, pb_orientation_index=1)
    elif mismatch == "live_time_s":
        record = replace(record, live_time_s=original.live_time_s + 1.0)
    elif mismatch == "travel_time_s":
        record = replace(record, travel_time_s=original.travel_time_s + 1.0)
    elif mismatch == "shield_actuation_time_s":
        record = replace(
            record,
            shield_actuation_time_s=original.shield_actuation_time_s + 1.0,
        )
    elif mismatch == "station_complete":
        record = replace(
            record,
            metadata={
                key: value
                for key, value in record.metadata.items()
                if key != "station_complete"
            },
        )
    elif mismatch == "energy_bin_edges_keV":
        record = replace(
            record,
            energy_bin_edges_keV=np.asarray(
                record.energy_bin_edges_keV,
                dtype=np.float64,
            )
            + 0.25,
        )
    elif mismatch == "full_spectrum_contract_hash_sha256":
        record = replace(
            record,
            metadata={
                **dict(record.metadata),
                "full_spectrum_contract_hash_sha256": "f" * 64,
            },
        )
    elif mismatch == "candidates.current_pair_id":
        pair_id = (
            int(original.fe_orientation_index) * 8
            + int(original.pb_orientation_index)
        )
        candidates = _response_candidates(
            original,
            current_pair_id=(pair_id + 1) % 64,
        )
    elif mismatch == "candidates.current_pose":
        candidates = _response_candidates(
            original,
            pose_xyz=(0.25, 0.5, 0.5),
        )
    elif mismatch == "candidates.allowed_pair_ids":
        candidates = _response_candidates(
            original,
            allowed_pair_ids=tuple(reversed(range(64))),
        )
    elif mismatch == "candidates.shield_angular_speed_rad_s":
        candidates = _response_candidates(
            original,
            shield_angular_speed_rad_s=2.0,
        )

    with pytest.raises(PFLiveSessionError, match=mismatch.replace(".", r"\.")):
        session.receive_acquired(
            record,
            request=request,
            request_candidates=request_candidates,
            next_candidates=candidates,
        )

    assert session.phase == "failed"
    assert session.records == ()
    assert estimator.update_calls == []


def test_mle_style_surface_guidance_adjusts_pf_proposals_without_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A causal surface grid must guide the exact-RJ proposal for that station."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    isotopes = tuple(log.context.isotopes)
    guidance = PFExternalSurfaceGuidance(
        source_run_id=log.context.run_id,
        record_count=len(log.records),
        data_cutoff_step=log.records[-1].step_id,
        data_cutoff_station=log.records[-1].station_id,
        covered_records_digest=measurement_records_digest(log.records),
        isotope_order=isotopes,
        patch_centroids_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        density_by_isotope=np.asarray(
            [[1.0, 4.0] for _ in isotopes],
            dtype=np.float64,
        ),
        proposal_mass=0.6,
        bandwidth_m=0.25,
    )

    session.stage_external_surface_guidance(guidance)
    session.receive_persisted_station(log.records)

    mapped = estimator.update_calls[0]["surface_guidance"]
    assert isinstance(mapped, dict)
    assert tuple(mapped) == isotopes
    assert estimator.update_calls[0]["surface_guidance_mass"] == 0.6
    assert estimator._joint_external_surface_guidance_by_isotope is None
    assert estimator._joint_external_surface_guidance_mass == 0.0
    receipt = session.last_surface_guidance_receipt
    assert receipt is not None
    assert receipt.guidance_sha256 == guidance.guidance_sha256
    assert receipt.source_run_id == log.context.run_id
    assert receipt.record_count == len(log.records)
    assert receipt.informative_isotopes == isotopes
    assert receipt.evaluated_isotopes == isotopes
    assert receipt.mapped_chart_count == 2
    assert receipt.target_preserving is True
    assert receipt.direct_weight_update is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_run_id", 7),
        ("source_run_id", " run-1"),
        ("isotope_order", ("Cs-137", 7)),
        ("isotope_order", ("Cs-137 ",)),
    ),
)
def test_surface_guidance_rejects_string_coercion(
    field: str,
    value: object,
) -> None:
    """External proposal lineage must preserve exact run and isotope strings."""
    values: dict[str, object] = {
        "source_run_id": "run-1",
        "record_count": 1,
        "data_cutoff_step": 0,
        "data_cutoff_station": 0,
        "covered_records_digest": DigestIdentity(
            algorithm="measurement-records-v1",
            sha256="a" * 64,
        ),
        "isotope_order": ("Cs-137",),
        "patch_centroids_xyz": np.asarray([[0.0, 0.0, 0.0]]),
        "density_by_isotope": np.asarray([[1.0]]),
        "proposal_mass": 0.5,
        "bandwidth_m": 0.5,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        PFExternalSurfaceGuidance(**values)


def test_surface_guidance_mix_is_vectorized_and_proposal_only() -> None:
    """The configured mass must mix normalized grids without a weight update."""
    owner = EstimatorStructuralProposalMixin()
    owner._joint_external_surface_guidance_by_isotope = {
        "Cs-137": np.asarray([0.0, 4.0], dtype=np.float64)
    }
    owner._joint_external_surface_guidance_mass = 0.25
    owner.last_external_surface_guidance_diagnostics = {}
    owner.last_external_surface_guidance_evaluated_isotopes = set()
    particle_weights = np.asarray([0.7, 0.3], dtype=np.float64)

    mixed, informative = owner._mix_external_surface_guidance(
        isotope="Cs-137",
        alignment=np.asarray([2.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(mixed, [0.75, 0.25], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(particle_weights, [0.7, 0.3])
    assert informative is True
    assert owner.last_external_surface_guidance_evaluated_isotopes == {"Cs-137"}
    assert owner.last_external_surface_guidance_diagnostics["Cs-137"][
        "target_preserving_proposal_only"
    ] == 1.0


def test_surface_guidance_rejects_a_different_record_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PF must fail closed when MLE guidance does not bind the incoming station."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    isotopes = tuple(log.context.isotopes)
    actual_digest = measurement_records_digest(log.records)
    guidance = PFExternalSurfaceGuidance(
        source_run_id=log.context.run_id,
        record_count=len(log.records),
        data_cutoff_step=log.records[-1].step_id,
        data_cutoff_station=log.records[-1].station_id,
        covered_records_digest=DigestIdentity(
            algorithm=actual_digest.algorithm,
            sha256="f" * 64,
        ),
        isotope_order=isotopes,
        patch_centroids_xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        density_by_isotope=np.ones((len(isotopes), 1), dtype=np.float64),
        proposal_mass=0.5,
        bandwidth_m=0.5,
    )

    session.stage_external_surface_guidance(guidance)
    with pytest.raises(PFLiveSessionError, match="exact incoming PF prefix"):
        session.receive_persisted_station(log.records)
    assert estimator.update_calls == []


def test_facade_particle_snapshot_is_an_immutable_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Planning DTO arrays and isotope mappings must not alias the estimator."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)

    snapshot = session.planning_particle_snapshot()
    estimator.particles.weights_n[0] = 0.5
    estimator.particles.positions_nk3_by_isotope[log.context.isotopes[0]][0, 0, 0] = 9.0

    np.testing.assert_array_equal(snapshot.weights_n, [0.75, 0.25])
    assert snapshot.positions_nk3_by_isotope[log.context.isotopes[0]][0, 0, 0] == 0.1
    with pytest.raises(ValueError):
        snapshot.weights_n[0] = 1.0
    with pytest.raises(ValueError):
        snapshot.weights_n.setflags(write=True)
    with pytest.raises(TypeError):
        snapshot.positions_nk3_by_isotope["new"] = np.zeros((2, 1, 3))
    summary = snapshot.posterior_summary()
    assert summary["schema_version"] == 2
    assert summary["publishable"] is False
    assert set(summary["isotopes"]) == set(log.context.isotopes)
    isotope_summary = summary["isotopes"][log.context.isotopes[0]]
    assert isotope_summary["map_cardinality"] == 1
    assert set(isotope_summary["modes"][0]) == {
        "label_index",
        "position_medoid_xyz",
        "credible_radius_95_m",
        "strength_representative_cps_1m",
        "posterior_mass",
    }
    assert "position_covariance_xyz" not in isotope_summary["modes"][0]
    assert "strength_credible_interval_95_cps_1m" not in isotope_summary["modes"][0]
    summary["isotopes"] = {}
    assert snapshot.posterior_summary()["isotopes"]
    assert estimator.posterior_summary_calls == 1


def test_bind_and_publication_never_advance_completed_pf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final binding and publication input must not assimilate observations."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    completed = session.complete_live_state()
    update_count = len(estimator.update_calls)
    planning_count = estimator.planning_calls

    with pytest.raises(PFLiveSessionError, match="cannot receive"):
        session.receive_persisted(log.records[-1])
    with pytest.raises(PFLiveSessionError, match="cannot plan"):
        session.planning_particle_snapshot()
    bound = session.bind_published_log(log)
    publication = session.publication_input()

    assert len(estimator.update_calls) == update_count == 1
    assert estimator.planning_calls == planning_count
    assert publication is bound
    assert bound.completed is completed
    assert bound.checkpoint_state == completed.checkpoint_state
    checkpoint_state = json.loads(bound.checkpoint_state)
    assert checkpoint_state["schema_version"] == 2
    assert checkpoint_state["estimator_state_schema_version"] == 1
    assert checkpoint_state["control_policy"] == (
        PFControlPolicyProvenance.native_dss_pp().to_dict()
    )
    posterior = json.loads(bound.posterior_json)
    assert posterior["provenance"]["measurement_log_sha256"] == log.log_sha256
    assert posterior["provenance"]["control_policy"] == (
        PFControlPolicyProvenance.native_dss_pp().to_dict()
    )
    assert session.phase == "bound"


def test_external_policy_provenance_matches_posterior_state_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every final live artifact must carry the same exact external policy."""
    from hashlib import sha256

    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    canonical_policy = strict_canonical_json_bytes(
        {
            "schema_version": 2,
            "variant": "round_robin_shield",
            "shield_policy": {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": True,
            },
        }
    )
    control_provenance = PFControlPolicyProvenance(
        policy_family="ral_ablation",
        source_sha256="b" * 64,
        canonical_sha256=sha256(canonical_policy).hexdigest(),
        canonical_policy_json=canonical_policy,
    )
    session, _estimator = _facade_with_spy(
        monkeypatch,
        log,
        control_policy_provenance=control_provenance,
    )
    session.receive_persisted_station(log.records)
    completed = session.complete_live_state()
    bound = session.bind_published_log(log)
    published = session.publish_bound_result(tmp_path / "pf-result")
    expected = control_provenance.to_dict()

    assert json.loads(completed.checkpoint_state)["control_policy"] == expected
    assert json.loads(bound.posterior_json)["provenance"]["control_policy"] == expected
    checkpoint = json.loads(published.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 2
    assert checkpoint["state_schema_version"] == 2
    assert checkpoint["control_policy"] == expected


def test_live_completion_rejects_failed_sampler_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final sealing must not promote an unhealthy sampler to complete."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    original_diagnostics = estimator.posterior_convergence_diagnostics

    def unhealthy_diagnostics() -> dict[str, object]:
        """Return the normal schema with one explicit failed sampler gate."""
        diagnostics = original_diagnostics()
        sampler = diagnostics["sampler_health"]
        joint = diagnostics["joint_gates"]
        assert isinstance(sampler, dict)
        assert isinstance(joint, dict)
        sampler["rejuvenation_mixing_complete"] = False
        joint["rejuvenation_mixing_complete"] = False
        return diagnostics

    monkeypatch.setattr(
        estimator,
        "posterior_convergence_diagnostics",
        unhealthy_diagnostics,
    )

    with pytest.raises(PFLiveSessionError, match="sampler-health gates"):
        session.complete_live_state()

    assert session.phase == "failed"


def test_bound_facade_publishes_package_owned_result_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PF publication writes posterior, checkpoint, and particles once."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    session.complete_live_state()
    session.bind_published_log(log)
    update_count = len(estimator.update_calls)

    published = session.publish_bound_result(tmp_path / "pf-result")

    assert published.root == (tmp_path / "pf-result").resolve()
    assert published.posterior_path.is_file()
    assert published.checkpoint_path.is_file()
    assert published.checkpoint_state_path.is_file()
    assert published.particle_snapshot_path.is_file()
    assert published.post_run_evaluation_input_path.is_file()
    assert published.diagnostics_path.is_file()
    inventory_path = published.root / "pf_artifact_inventory.json"
    assert inventory_path.is_file()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["metadata"]["artifact_family"] == "pure_pf_live_result"
    assert inventory["metadata"]["source_run_id"] == log.run_id
    diagnostics = json.loads(
        published.diagnostics_path.read_text(encoding="utf-8")
    )
    assert diagnostics["schema_version"] == 2
    assert diagnostics["posterior_predictive_check"] == {"available": False}
    assert (
        diagnostics["sampler_health"][
            "smc_rejuvenation_wall_time_respected"
        ]
        is True
    )
    assert "measurement_log_sha256" not in diagnostics
    assert "config" not in diagnostics
    evaluation_input = json.loads(
        published.post_run_evaluation_input_path.read_text(encoding="utf-8")
    )
    assert evaluation_input["schema_version"] == 1
    assert evaluation_input["source_run_id"] == log.run_id
    assert evaluation_input["measurement_log_sha256"] == log.log_sha256
    assert evaluation_input["truth_read"] is False
    assert not (published.root / "pf_trace.jsonl").exists()
    assert len(published.result_sha256) == 64
    assert len(estimator.update_calls) == update_count
    checkpoint = json.loads(published.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["prefix_measurement_log_sha256"] == log.log_sha256
    assert checkpoint["covered_step_ids"] == [0, 1]
    assert checkpoint["control_policy"] == (
        PFControlPolicyProvenance.native_dss_pp().to_dict()
    )
    with np.load(published.particle_snapshot_path, allow_pickle=False) as arrays:
        assert arrays["weights_n"].tolist() == [0.75, 0.25]
        assert arrays["isotope_names"].tolist() == list(log.context.isotopes)


def test_facade_rejects_out_of_order_and_mismatched_final_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The facade must bind only its exact run, context, and ordered records."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    out_of_order, _ = _facade_with_spy(monkeypatch, log)
    with pytest.raises(PFLiveSessionError, match="causal record order"):
        out_of_order.receive_persisted(log.records[1])

    incomplete, _ = _facade_with_spy(monkeypatch, log)
    incomplete.receive_persisted(log.records[0])
    with pytest.raises(PFLiveSessionError, match="station marker"):
        incomplete.complete_live_state()

    session, _ = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    session.complete_live_state()
    changed_record = replace(log.records[-1], live_time_s=2.0)
    changed_log = replace(log, records=(log.records[0], changed_record))
    with pytest.raises(PFLiveSessionError, match="records differ"):
        session.bind_published_log(changed_log)

    changed_context = replace(
        log,
        environment={**log.environment, "size_x": 3.0},
    )
    with pytest.raises(PFLiveSessionError, match="context differs"):
        session.bind_published_log(changed_context)
