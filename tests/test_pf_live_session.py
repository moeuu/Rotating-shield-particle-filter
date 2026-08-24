"""Tests for the live PF runtime-ingestion boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from runtime.measurement_log import load_measurement_log
from runtime.prefix import measurement_records_digest
from runtime.provenance import DigestIdentity

from pf.live_session import (
    PFExternalSurfaceGuidance,
    PFLiveSession,
    PFLiveSessionError,
    bind_published_measurement_log,
    build_live_estimator,
    load_live_pf_config,
    measurement_record_to_station_input,
)
from pf.estimator_structural import EstimatorStructuralProposalMixin
from pf.estimator_types import JointPlanningParticles
from pf.pure_estimator import RotatingShieldPFConfig
from tests.pure_pf_test_support import make_measurement_log


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

    actual = build_live_estimator(
        context,  # type: ignore[arg-type]
        {"pure_pf_schema_version": 1},
        profile="pf_strict",
        seed=9,
        runtime_root=tmp_path,
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
        "config_hash": None,
        "inference_isotopes": None,
    }


def test_live_pf_config_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    """The public in-process config API must never discard unknown options."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "num_particles": 8,
                "max_sources": 1,
                "init_num_sources": [0, 1],
                "use_gpu": False,
            }
        ),
        encoding="utf-8",
    )

    payload, source_sha256 = load_live_pf_config(
        config_path,
        profile="pf_strict",
    )

    assert payload["num_particles"] == 8
    assert len(source_sha256) == 64
    payload["num_particels"] = payload.pop("num_particles")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PFLiveSessionError, match="unknown fields.*num_particels"):
        load_live_pf_config(config_path, profile="pf_strict")


def test_live_pf_config_validates_nested_adaptive_stop_thresholds(
    tmp_path: Path,
) -> None:
    """Adaptive-stop thresholds must be validated from their single block."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "adaptive_stop": {
                    "enabled": True,
                    "assessment_start_station": 10,
                    "required_consecutive_stations": 3,
                    "minimum_joint_map_cardinality_probability": 1.1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PFLiveSessionError, match="incompatible"):
        load_live_pf_config(config_path, profile="pf_strict")


def test_live_pf_config_rejects_top_level_adaptive_stop_threshold(
    tmp_path: Path,
) -> None:
    """Estimator-facing stop settings must not duplicate the nested block."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "adaptive_stop_innovation_confidence": 0.99,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PFLiveSessionError, match="unknown fields"):
        load_live_pf_config(config_path, profile="pf_strict")


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


def test_published_log_binding_requires_the_live_record_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final provenance binding accepts only the assimilated live session."""
    from pf import live_session

    published = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    isotopes = tuple(published.run_manifest["isotopes"])
    estimator = SimpleNamespace(
        candidate_isotopes=isotopes,
        joint_isotope_order=lambda: isotopes,
        measurements=[object()],
        pf_config=RotatingShieldPFConfig(),
        random_seed=7,
        measurement_log_sha256="unavailable",
        resolved_config_hash="unavailable",
    )
    validated: list[object] = []
    monkeypatch.setattr(
        live_session,
        "_validate_published_forward_context",
        validated.append,
    )

    bind_published_measurement_log(
        estimator,
        published,
        live_records=published.records,
    )

    assert validated == [published]
    assert estimator.measurement_log_sha256 == published.log_sha256
    assert len(estimator.resolved_config_hash) == 64

    changed_record = replace(published.records[0], live_time_s=2.0)
    changed_published = replace(published, records=(changed_record,))
    with pytest.raises(PFLiveSessionError, match="ordered live records"):
        bind_published_measurement_log(
            estimator,
            changed_published,
            live_records=published.records,
        )


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
        self.candidate_isotopes = isotopes
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
        return {
            isotope: SimpleNamespace(
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
                }
            )
            for isotope in self.isotopes
        }

    def serialized_state(self) -> bytes:
        """Return state bytes depending only on completed station updates."""
        return json.dumps(
            {
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
            "smc_soft_budget_respected": True,
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
            "isotopes": {},
        }

    def posterior_predictive_check(self) -> dict[str, object]:
        """Return one unavailable predictive-check result."""
        return {"available": False}


def _facade_with_spy(
    monkeypatch: pytest.MonkeyPatch,
    log: object,
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
        assert config == {"pure_pf_schema_version": 1}
        assert kwargs["profile"] == "pf_strict"
        assert kwargs["seed"] == 17
        return estimator

    monkeypatch.setattr(live_session, "build_live_estimator", build_spy)
    session = PFLiveSession(
        context,
        {"pure_pf_schema_version": 1},
        profile="pf_strict",
        seed=17,
        runtime_root=log.path,
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


def test_facade_delegates_pose_and_shield_planning_to_pf_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The live facade must return the PF planner's complete runtime action."""
    from pf import live_session

    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, estimator = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)
    captured: dict[str, object] = {}

    class _Forward:
        """Expose only runtime-owned planning geometry."""

        bounds_xyz = (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([2.0, 2.0, 2.0]),
        )
        obstacle_grid = object()

    class _Resolver:
        """Return the authenticated planning geometry sentinel."""

        @classmethod
        def from_run_context(cls, context: object, *, run_root: Path) -> _Forward:
            """Validate the facade's context and runtime root."""
            del cls
            assert context.to_payload() == log.context.to_payload()
            assert run_root == log.path
            return _Forward()

    def select(
        estimator_arg: object,
        candidates: object,
        current: object,
        **kwargs: object,
    ) -> object:
        """Capture PF-owned planner inputs and return a complete action."""
        assert estimator_arg is estimator
        captured["candidates"] = candidates
        captured["current"] = current
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            # DSS-PP may index its filtered candidates, so the public facade
            # must remap the selected pose to the original runtime snapshot.
            next_pose_index=0,
            next_pose=np.asarray([1.0, 1.5, 0.5]),
            shield_program=SimpleNamespace(
                pair_ids=(7, 23),
                name="dynamic",
                kind="dss_pp",
            ),
            score=3.5,
            diagnostics={"selected_information_gain": 2.25},
        )

    monkeypatch.setattr(live_session, "ResolvedForwardContext", _Resolver)
    monkeypatch.setattr(live_session, "select_dss_pp_next_station", select)
    action = session.plan_next_action(
        [[0.5, 0.5, 0.5], [1.0, 1.5, 0.5]],
        candidate_motion_times_s=[0.0, 2.0],
        candidate_horizontal_travel_times_s=[0.0, 1.0],
        candidate_mast_vertical_times_s=[0.0, 0.5],
        candidate_settling_times_s=[0.0, 0.5],
        config={"augment_candidates": False, "program_length": 2},
    )

    assert action.candidate_index == 1
    assert action.detector_pose_xyz == (1.0, 1.5, 0.5)
    assert action.shield_pair_ids == (7, 23)
    assert action.diagnostics()["selected_information_gain"] == 2.25
    assert set(action.diagnostics()) == {
        "schema_version",
        "selected_information_gain",
    }
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["current_pair_id"] == (
        log.records[-1].fe_orientation_index * 8
        + log.records[-1].pb_orientation_index
    )
    assert kwargs["map_api"] is _Forward.obstacle_grid
    assert kwargs["config"].augment_candidates is False
    np.testing.assert_allclose(
        kwargs["candidate_horizontal_travel_times_s"],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        kwargs["candidate_mast_vertical_times_s"],
        [0.0, 0.5],
    )
    np.testing.assert_allclose(
        kwargs["candidate_settling_times_s"],
        [0.0, 0.5],
    )


def test_facade_rejects_unknown_pf_planning_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unknown planning settings must fail instead of being silently dropped."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=2,
            station_complete_markers=True,
        )
    )
    session, _ = _facade_with_spy(monkeypatch, log)
    session.receive_persisted_station(log.records)

    with pytest.raises(PFLiveSessionError, match="configuration is incompatible"):
        session.plan_next_action(
            [[0.5, 0.5, 0.5]],
            candidate_motion_times_s=[0.0],
            config={"augment_candidates": False, "unknown_option": 1},
        )


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
    posterior = json.loads(bound.posterior_json)
    assert posterior["provenance"]["measurement_log_sha256"] == log.log_sha256
    assert session.phase == "bound"


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
    assert published.diagnostics_path.is_file()
    diagnostics = json.loads(
        published.diagnostics_path.read_text(encoding="utf-8")
    )
    assert diagnostics["schema_version"] == 2
    assert diagnostics["posterior_predictive_check"] == {"available": False}
    assert diagnostics["sampler_health"]["smc_soft_budget_respected"] is True
    assert "measurement_log_sha256" not in diagnostics
    assert "config" not in diagnostics
    assert not (published.root / "pf_trace.jsonl").exists()
    assert len(published.result_sha256) == 64
    assert len(estimator.update_calls) == update_count
    checkpoint = json.loads(published.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["prefix_measurement_log_sha256"] == log.log_sha256
    assert checkpoint["covered_step_ids"] == [0, 1]
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
