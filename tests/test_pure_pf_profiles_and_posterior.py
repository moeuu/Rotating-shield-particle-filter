"""Scientific-variant and PF-only posterior aggregation tests."""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mission_control import resolve_mission_max_poses, resolve_mission_max_steps
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kind
from measurement.surface_patches import build_surface_patch_dictionary
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import IsotopeParticle
from pf.posterior import posterior_point_estimate_from_states
from pf.profiles import (
    PURE_PF_SCHEMA_VERSION,
    EstimatorProfile,
    apply_profile_to_config,
    enforce_pure_runtime_settings,
    resolve_estimator_profile,
    resolve_structural_transition_provenance,
)
from pf.pure_estimator import PurePFEstimator
from pf.runtime_route import canonical_runtime_likelihood_route_mapping
from pf.state import IsotopeState
from planning.dss_pp import extract_signature_modes
from sim.runtime import load_runtime_config


def test_measurement_record_requires_canonical_runtime_metadata() -> None:
    """PF history records must not expose legacy orientation fallbacks."""
    record_fields = {field.name: field for field in fields(MeasurementRecord)}

    assert "orient_idx" not in record_fields
    for field_name in (
        "fe_index",
        "pb_index",
        "detector_position_xyz_m",
        "station_sequence_id",
        "station_view_index",
        "runtime_likelihood_route_by_isotope",
    ):
        field = record_fields[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING


@pytest.mark.parametrize(
    "routes",
    [
        None,
        {},
        {"Cs-137": "count"},
        {"Cs-137": "poisson", "Co-60": "count"},
        {"Cs-137": "count", "Co-60": "count", "Eu-154": "count"},
    ],
)
def test_runtime_likelihood_route_mapping_fails_closed(
    routes: object,
) -> None:
    """Missing, aliased, and extra runtime routes must be rejected."""
    with pytest.raises(
        ValueError,
        match="runtime_likelihood|Runtime likelihood|exactly every",
    ):
        canonical_runtime_likelihood_route_mapping(
            routes,
            ("Cs-137", "Co-60"),
        )


def _exact_rj_config(**overrides: object) -> RotatingShieldPFConfig:
    """Build an exact finite-surface RJ-MH config for focused tests."""
    values: dict[str, object] = {
        "estimator_profile": "pf_strict",
        "max_sources": 5,
        "init_num_sources": (0, 5),
    }
    values.update(overrides)
    return RotatingShieldPFConfig(**values)


def _stable_fixed_k_estimator() -> RotatingShieldPFEstimator:
    """Build a fixed-K pure PF with a degenerate stable posterior."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=float,
        ),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=True,
            gpu_device="cpu",
        ),
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    particle_filter = estimator.filters["Cs-137"]
    particle_filter.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.asarray([10.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for _ in range(2)
    ]
    estimator.history_estimates = [estimator.estimates(), estimator.estimates()]
    return estimator


@pytest.mark.parametrize(
    "profile",
    ["pf_strict", EstimatorProfile.PF_STRICT],
)
def test_only_strict_profile_is_supported(
    profile: EstimatorProfile | str,
) -> None:
    """The canonical name must resolve to the single strict PF profile."""
    resolved_profile, capabilities = resolve_estimator_profile(profile)

    assert resolved_profile is EstimatorProfile.PF_STRICT
    assert capabilities.sequential_updates_only is True
    assert capabilities.posterior_reporting_only is True
    assert capabilities.likelihood_consistent_structural_evidence is True


@pytest.mark.parametrize("profile", [None, "strict", "pure_pf", "pf_only"])
def test_profile_aliases_are_not_part_of_the_runtime_schema(
    profile: object,
) -> None:
    """Only the explicit canonical profile belongs to the runtime schema."""
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        resolve_estimator_profile(profile)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "schema_version",
    [
        None,
        0,
        2,
        True,
        "1",
    ],
)
def test_runtime_requires_exact_pure_pf_schema_version(
    schema_version: object,
) -> None:
    """Runtime configuration must explicitly select pure-PF schema version 1."""
    payload = {"estimator_profile": "pf_strict"}
    if schema_version is not None:
        payload["pure_pf_schema_version"] = schema_version
    with pytest.raises(ValueError, match="pure_pf_schema_version=1"):
        enforce_pure_runtime_settings(payload)


def test_runtime_accepts_the_positive_pure_pf_schema() -> None:
    """The versioned schema must preserve the canonical strict profile."""
    payload = {
        "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
        "estimator_profile": "pf_strict",
    }
    resolved = enforce_pure_runtime_settings(payload)

    assert resolved == payload


def test_runtime_schema_requires_the_canonical_profile() -> None:
    """The version marker must be paired with the canonical estimator profile."""
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        enforce_pure_runtime_settings(
            {"pure_pf_schema_version": PURE_PF_SCHEMA_VERSION}
        )


def test_explicit_profile_cannot_override_runtime_schema() -> None:
    """A caller profile must not replace an invalid logged profile."""
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "removed-profile",
            },
            profile="pf_strict",
        )


def test_runtime_schema_rejects_unknown_pf_settings() -> None:
    """Unknown PF-prefixed settings must fail instead of becoming no-ops."""
    with pytest.raises(ValueError, match="Unsupported pure-PF runtime settings"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "pf_unknown_transition": True,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "structural_rj_mov_probability",
        "structural_cardinality_prior_probability",
    ],
)
def test_runtime_schema_rejects_unknown_structural_settings(
    field_name: str,
) -> None:
    """Typos in exact-RJ controls must fail before a runtime starts."""
    with pytest.raises(ValueError, match="structural"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                field_name: 1.0,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("variable_cardinality", "true"),
        ("pf_max_sources", None),
        ("pf_max_sources", 0),
        ("init_num_sources", [0]),
        ("init_num_sources", [0, "5"]),
    ],
)
def test_runtime_schema_rejects_malformed_cardinality_settings(
    field_name: str,
    field_value: object,
) -> None:
    """Cardinality controls must not be silently coerced or clamped."""
    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                field_name: field_value,
            }
        )


def test_runtime_schema_requires_pf_setting_objects() -> None:
    """Nested PF setting groups must keep their declared object shape."""
    with pytest.raises(ValueError, match="must be objects"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "pf_count_likelihood": "student_t",
            }
        )


@pytest.mark.parametrize(
    "retired_key",
    [
        "birth_enable",
        "candidate_verification_queue_enable",
        "conditional_strength_refit",
        "delayed_resample_update",
        "init_num_sources_min",
        "joint_observation_update",
        "report_model_order_min_bic_margin",
        "roughening_k",
        "report_mle_enable",
        "sparse_poisson_evidence_enable",
        "spectrum_likelihood_bin_chunk",
        "surface_map_enable",
        "refit_after_moves",
    ],
)
def test_runtime_schema_rejects_retired_estimator_settings(
    retired_key: str,
) -> None:
    """Deleted estimator generations must not survive as silent no-op keys."""
    with pytest.raises(ValueError, match="Retired particle-filter settings"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                retired_key: True,
            }
        )


@pytest.mark.parametrize(
    "retired_key",
    [
        "adaptive_program_length_enable",
        "global_surface_rescue_mode_weight",
        "recovery_isotope_mode_weight_multiplier",
        "residual_program_length",
        "same_isotope_direct_separation_guard",
        "typo_weight",
    ],
)
def test_runtime_schema_rejects_retired_dss_settings(
    retired_key: str,
) -> None:
    """Deleted DSS rescue and heuristic settings must fail closed."""
    with pytest.raises(ValueError, match="Unsupported dss_pp"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "dss_pp": {retired_key: True},
            }
        )


@pytest.mark.parametrize(
    "retired_key",
    [
        "dss_count_utility_weight",
        "high_surface_ambiguity_weight",
        "residual_chi2_threshold",
        "unresolved_absent_budget_weight",
    ],
)
def test_runtime_schema_rejects_retired_remaining_measurement_settings(
    retired_key: str,
) -> None:
    """Deleted residual and rescue budget settings must fail closed."""
    with pytest.raises(
        ValueError,
        match="Unsupported remaining_measurement_estimate",
    ):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "remaining_measurement_estimate": {retired_key: True},
            }
        )


@pytest.mark.parametrize(
    "unknown_key",
    [
        "observation_count_variance_includes_counting_noise",
        "direct_spectrum_likelihood_enable",
        "birth_delta_ll_threshold",
        "typo_model",
    ],
)
def test_runtime_schema_rejects_unknown_count_likelihood_settings(
    unknown_key: str,
) -> None:
    """The count-likelihood block must use only schema-v1 canonical fields."""
    with pytest.raises(ValueError, match="Unsupported pf_count_likelihood"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "pf_count_likelihood": {unknown_key: True},
            }
        )


def test_runtime_schema_validates_transport_response_model_keys() -> None:
    """Transport-response payloads must not accept historical coefficient aliases."""
    with pytest.raises(ValueError, match="tau_coefficients"):
        enforce_pure_runtime_settings(
            {
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "estimator_profile": "pf_strict",
                "pf_transport_response_model": {
                    "enabled": True,
                    "model": "log_tau_regression_v1",
                    "feature_semantics": "canonical",
                    "by_isotope": {
                        "Cs-137": {
                            "tau_coefficients": {"shield_tau": 0.1},
                        }
                    },
                },
            }
        )


def test_fixed_k_provenance_declares_target_preserving_mh_kernel() -> None:
    """Fixed-K provenance must declare exact within-cardinality MH moves."""
    fixed_config = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        init_num_sources=(3, 3),
        variable_cardinality=False,
    )
    fixed_capabilities = apply_profile_to_config(fixed_config)
    fixed = resolve_structural_transition_provenance(
        fixed_config,
        capabilities=fixed_capabilities,
    ).to_dict()

    assert fixed["posterior_semantics"] == (
        "fixed_cardinality_sequential_particle_filter_with_"
        "target_preserving_mh_rejuvenation"
    )
    assert fixed["structural_kernel_family"] == (
        "fixed_cardinality_surface_position_strength_mh"
    )
    assert fixed["structural_moves_enabled"] is True
    assert fixed["variable_cardinality"] is False
    assert fixed["birth_death_moves_enabled"] is False
    assert fixed["within_cardinality_moves_enabled"] is True
    assert fixed["within_cardinality_kernel_exact_mh"] is True
    assert fixed["structural_kernel_target_preserving"] is True
    assert fixed["structural_kernel_exact_rj"] is False
    assert fixed["reversible_jump_mcmc_used"] is False
    assert fixed["structural_evidence_uses_pf_likelihood"] is True


def test_exact_rj_provenance_declares_target_preserving_pf_kernel() -> None:
    """Exact RJ-MH mode must report its finite-surface posterior semantics."""
    config = _exact_rj_config(
        init_num_sources=(0, 5),
        variable_cardinality=True,
    )
    capabilities = apply_profile_to_config(config)
    provenance = resolve_structural_transition_provenance(
        config,
        capabilities=capabilities,
    ).to_dict()

    assert provenance["posterior_semantics"] == (
        "sequential_particle_filter_with_target_preserving_rj_mh_rejuvenation"
    )
    assert provenance["structural_kernel_family"] == (
        "area_weighted_surface_birth_death_rj_mh"
    )
    assert provenance["structural_moves_enabled"] is True
    assert provenance["variable_cardinality"] is True
    assert provenance["birth_death_moves_enabled"] is True
    assert provenance["within_cardinality_moves_enabled"] is True
    assert provenance["within_cardinality_kernel_exact_mh"] is True
    assert provenance["structural_kernel_target_preserving"] is True
    assert provenance["structural_kernel_exact_rj"] is True
    assert provenance["reversible_jump_mcmc_used"] is True
    assert provenance["structural_evidence_uses_pf_likelihood"] is True


def test_stable_pure_pf_posterior_stops_shield_rotation() -> None:
    """A stable fixed-K PF posterior should satisfy the stopping rule."""
    estimator = _stable_fixed_k_estimator()

    assert estimator.should_stop_shield_rotation(
        pose_idx=0,
        ig_threshold=1.0e-6,
        change_tol=1.0e-6,
        uncertainty_tol=1.0e-6,
        live_time_s=1.0,
    )


def test_uncertain_pure_pf_posterior_keeps_exploration_active() -> None:
    """Posterior strength uncertainty should prevent exploration stopping."""
    estimator = _stable_fixed_k_estimator()
    particle_filter = estimator.filters["Cs-137"]
    particle_filter.continuous_particles[0].state.strengths = np.asarray([1.0])
    particle_filter.continuous_particles[1].state.strengths = np.asarray([10.0])
    estimator.history_estimates = [estimator.estimates(), estimator.estimates()]

    assert not estimator.should_stop_exploration(
        ig_threshold=1.0e-6,
        change_tol=1.0e-6,
        uncertainty_tol=1.0e-3,
        live_time_s=1.0,
    )


def test_pure_pf_dwell_limit_stops_shield_rotation() -> None:
    """The declared dwell budget should stop rotation without another estimator."""
    estimator = _stable_fixed_k_estimator()
    estimator.pf_config.max_dwell_time_s = 0.5
    estimator.measurements = [
        MeasurementRecord(
            z_k={"Cs-137": 1.0},
            pose_idx=0,
            live_time_s=0.3,
            fe_index=0,
            pb_index=0,
            detector_position_xyz_m=(0.5, 0.0, 0.0),
            station_sequence_id=0,
            station_view_index=0,
            runtime_likelihood_route_by_isotope={"Cs-137": "count"},
        ),
        MeasurementRecord(
            z_k={"Cs-137": 1.0},
            pose_idx=0,
            live_time_s=0.3,
            fe_index=0,
            pb_index=0,
            detector_position_xyz_m=(0.5, 0.0, 0.0),
            station_sequence_id=1,
            station_view_index=0,
            runtime_likelihood_route_by_isotope={"Cs-137": "count"},
        ),
    ]

    assert estimator.should_stop_shield_rotation(
        pose_idx=0,
        ig_threshold=0.0,
        change_tol=0.0,
        uncertainty_tol=0.0,
        live_time_s=1.0,
    )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("structural_rj_patch_spacing_m", 0.0),
        ("structural_rj_move_probability", -0.1),
        ("structural_rj_birth_probability", 1.1),
        ("structural_rj_death_probability", float("nan")),
        ("structural_rj_position_move_probability", -1.0),
        ("structural_rj_local_position_move_probability", 1.1),
        ("structural_rj_strength_move_probability", 2.0),
    ],
)
def test_exact_rj_numeric_configuration_is_validated(
    field_name: str,
    field_value: float,
) -> None:
    """RJ-MH spacing and attempt probabilities must stay in their domains."""
    with pytest.raises(ValueError, match=field_name):
        _exact_rj_config(**{field_name: field_value})


def test_structural_cardinality_prior_is_positive_and_canonical() -> None:
    """Cardinality prior masses must be finite, positive, and tuple-normalized."""
    config = _exact_rj_config(
        max_sources=2,
        init_num_sources=(0, 2),
        structural_cardinality_prior_probs=[1.0, 2.0, 3.0]
    )
    assert config.structural_cardinality_prior_probs == pytest.approx(
        (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    )

    with pytest.raises(ValueError, match="structural_cardinality_prior_probs"):
        _exact_rj_config(
            max_sources=1,
            init_num_sources=(0, 1),
            structural_cardinality_prior_probs=[1.0, 0.0],
        )


def test_pure_estimator_initializes_the_single_strict_profile() -> None:
    """PurePFEstimator must expose the positive strict-PF capability contract."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(estimator_profile="pf_strict"),
        measurement_log_sha256="b" * 64,
    )

    assert estimator.estimator_variant == "pf_strict"
    assert estimator.profile_capabilities.posterior_reporting_only is True
    assert estimator.profile_capabilities.sequential_updates_only is True


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_pure_estimator_rejects_non_integer_schema_versions(
    schema_version: object,
) -> None:
    """PurePFEstimator must not coerce schema-version compatibility values."""
    with pytest.raises(ValueError, match="schema version 1"):
        PurePFEstimator(
            isotopes=("Cs-137",),
            candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            shield_normals=None,
            mu_by_isotope={"Cs-137": 0.0},
            pf_config=RotatingShieldPFConfig(),
            measurement_log_schema_version=schema_version,  # type: ignore[arg-type]
        )


def test_structural_model_manifest_resolves_priors_and_surface_dictionaries() -> None:
    """Structural provenance must be complete without assuming shared dictionaries."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137", "Co-60"),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=_exact_rj_config(
            max_sources=2,
            init_num_sources=(0, 2),
            structural_cardinality_prior_probs=[1.0, 2.0, 3.0],
            strength_prior_min_cps_1m=300_000.0,
            strength_prior_max_cps_1m=2_000_000.0,
            use_gpu=False,
        ),
        measurement_log_sha256="b" * 64,
    )

    before_filters = estimator.structural_model_manifest()
    assert before_filters["manifest_completeness"] == "config_only"
    cardinality_prior = before_filters["cardinality_prior"]
    assert cardinality_prior["support"] == [0, 1, 2]
    assert cardinality_prior["probabilities"] == pytest.approx(
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]
    )
    assert cardinality_prior["configuration_source"] == "explicit"
    assert cardinality_prior["applies_independently_per_isotope"] is True
    assert before_filters["strength_prior"]["units"] == "detector_cps_1m"
    assert before_filters["strength_prior"]["minimum_cps_1m"] == 300_000.0
    assert before_filters["strength_prior"]["maximum_cps_1m"] == 2_000_000.0
    surface_prior = before_filters["surface_set_prior"]
    assert surface_prior["semantics"] == "area_product_distinct_patch_sets"
    assert surface_prior["dictionary_status"] == "not_initialized"
    assert surface_prior["dictionaries_identical_across_isotopes"] is None
    assert surface_prior["missing_isotopes"] == ["Co-60", "Cs-137"]
    rj_kernel = before_filters["rj_move_kernel"]
    assert rj_kernel["position_move_attempt_probability"] == 1.0
    assert rj_kernel["local_position_move_attempt_probability"] == 1.0
    assert rj_kernel["boundary_normalization"]["at_k_zero"] == {
        "birth": 1.0,
        "death": 0.0,
    }
    assert rj_kernel["boundary_normalization"]["at_k_max"] == {
        "cardinality": 2,
        "birth": 0.0,
        "death": 1.0,
    }
    assert (
        rj_kernel["dimension_matching"]["absolute_jacobian_determinant"]
        == 1.0
    )

    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    shared_patches = build_surface_patch_dictionary(environment, None, 1.0)
    estimator.filters = {
        isotope: SimpleNamespace(
            _structural_rj_surface_patches=shared_patches,
        )
        for isotope in estimator.isotopes
    }
    shared_manifest = estimator.structural_model_manifest()
    shared_surface = shared_manifest["surface_set_prior"]
    assert shared_manifest["manifest_completeness"] == "complete"
    assert shared_surface["dictionaries_identical_across_isotopes"] is True
    assert len(shared_surface["dictionary_groups"]) == 1
    assert shared_surface["dictionary_groups"][0]["isotopes"] == [
        "Co-60",
        "Cs-137",
    ]

    different_patches = build_surface_patch_dictionary(
        EnvironmentConfig(size_x=3.0, size_y=2.0, size_z=2.0),
        None,
        1.0,
    )
    estimator.filters["Co-60"] = SimpleNamespace(
        _structural_rj_surface_patches=different_patches,
    )
    different_surface = estimator.structural_model_manifest()[
        "surface_set_prior"
    ]
    assert different_surface["dictionaries_identical_across_isotopes"] is False
    assert len(different_surface["dictionary_groups"]) == 2
    different_hashes = {
        group["ordered_centers_areas_sha256"]
        for group in different_surface["dictionary_groups"]
    }
    assert (
        shared_surface["dictionary_groups"][0][
            "ordered_centers_areas_sha256"
        ]
        in different_hashes
    )


def test_structural_history_rebuilds_recorded_view_covariance() -> None:
    """Recorded structural history must infer the covariance runtime route."""
    covariance = ((4.0, 1.5), (1.5, 9.0))
    common = {
        "z_k": {"Cs-137": 12.0},
        "pose_idx": 0,
        "live_time_s": 1.0,
        "z_variance_k": {"Cs-137": 4.0},
        "detector_position_xyz_m": (1.0, 2.0, 0.5),
        "station_sequence_id": 17,
        "runtime_likelihood_route_by_isotope": {
            "Cs-137": "count_covariance"
        },
        "station_view_covariance_by_isotope": {"Cs-137": covariance},
    }
    records = [
        MeasurementRecord(
            **common,
            fe_index=0,
            pb_index=0,
            station_view_index=0,
        ),
        MeasurementRecord(
            **{**common, "z_k": {"Cs-137": 20.0}},
            fe_index=1,
            pb_index=1,
            station_view_index=1,
        ),
    ]
    estimator = object.__new__(PurePFEstimator)
    estimator.measurements = records
    estimator.poses = [np.asarray([99.0, 99.0, 99.0], dtype=float)]
    estimator.isotopes = ("Cs-137",)

    data = estimator._measurement_data_for_iso("Cs-137", window=None)

    assert data is not None
    np.testing.assert_allclose(
        data.detector_positions,
        [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
    )
    np.testing.assert_array_equal(data.fe_indices, [0, 1])
    np.testing.assert_array_equal(data.pb_indices, [0, 1])
    np.testing.assert_array_equal(data.station_sequence_ids, [17, 17])
    np.testing.assert_allclose(data.observation_count_covariance, covariance)
    assert data.runtime_likelihood_routes.tolist() == [
        "count_covariance",
        "count_covariance",
    ]
    final_row = estimator._measurement_data_for_iso("Cs-137", window=1)
    assert final_row is not None
    np.testing.assert_allclose(final_row.observation_count_covariance, [[9.0]])
    assert final_row.runtime_likelihood_routes.tolist() == [
        "count_covariance"
    ]


def test_pure_planner_uses_only_pf_posterior() -> None:
    """DSS modes must be derived only from the weighted PF ensemble."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    state = SimpleNamespace(
        num_sources=2,
        positions=np.asarray([[0.5, 0.5, 0.4], [1.5, 1.5, 1.2]]),
        strengths=np.asarray([10.0, 5.0]),
    )

    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        profile_capabilities=capabilities,
        planner_belief_sources=("pf_posterior",),
        pf_config=SimpleNamespace(),
        planning_particles=lambda **_kwargs: {"Cs-137": ([state], np.asarray([1.0]))},
    )
    modes = extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
    )
    assert len(modes["Cs-137"]) == 2
    assert estimator.planner_belief_sources == ("pf_posterior",)
    assert PurePFEstimator.planner_belief_sources == ("pf_posterior",)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/python/experiments/pf_strict_3d.json",
        "configs/geant4/experiments/pf_strict_3d.json",
    ],
)
def test_strict_profile_keeps_fixed_budget_and_continuous_3d_planning(
    relative_path: str,
) -> None:
    """Strict configs retain mission budget, collision, and 3-D planning."""
    root = Path(__file__).resolve().parents[1]
    resolved = enforce_pure_runtime_settings(load_runtime_config(root / relative_path))
    assert resolved["adaptive_cardinality_dwell_enable"] is False
    assert resolved["adaptive_mission_stop"] is False
    assert resolved["measurement_budget_max_steps"] == 160
    assert resolved["mission_stop_max_poses"] == 20
    assert resolve_mission_max_steps(None, resolved) == 160
    assert resolve_mission_max_poses(None, resolved) == 20
    assert resolved["parallel_isotope_updates"] is False
    assert resolved["detector_height_sampling_mode"] == "continuous"
    assert resolved["measurement_pose_clearance_enabled"] is True
    assert resolved["path_planner"] == "dss_pp"
    assert resolved["spectrum_count_method"] == "response_poisson"
    expected_backend = "geant4" if "geant4" in relative_path else "python"
    assert resolved["measurement_log_output_dir"] == (
        f"logs/pure_pf/{expected_backend}_pf_strict_3d_measurement_log"
    )
    dss = resolved["dss_pp"]
    # These inherited settings prove the nested section is fully specified,
    # not accidentally replaced by a three-key shallow override.
    assert int(dss["horizon"]) >= 1
    assert int(dss["program_length"]) >= 1
    assert int(dss["max_programs"]) >= 1


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json",
        "configs/geant4/high_fidelity_external_no_isaac.json",
        "configs/python/high_fidelity_no_isaac.json",
    ],
)
def test_standard_runtime_configs_declare_strict_pf_boundary(
    relative_path: str,
) -> None:
    """Standard config files must not rely on runtime overrides for PF purity."""
    root = Path(__file__).resolve().parents[1]
    payload = load_runtime_config(root / relative_path)

    assert payload["pure_pf_schema_version"] == PURE_PF_SCHEMA_VERSION
    assert payload["estimator_profile"] == "pf_strict"
    assert float(payload["pf_strength_prior_max_cps_1m"]) > float(
        payload["pf_strength_prior_min_cps_1m"]
    )


def test_standard_geant4_config_selects_exact_surface_rj_kernel() -> None:
    """The production Geant4 config must select only exact surface RJ-MH."""
    root = Path(__file__).resolve().parents[1]
    payload = load_runtime_config(
        root
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )

    assert payload["variable_cardinality"] is True
    assert float(payload["structural_rj_patch_spacing_m"]) > 0.0
    assert float(payload["structural_rj_move_probability"]) > 0.0
    assert float(payload["structural_rj_birth_probability"]) > 0.0
    assert float(payload["structural_rj_death_probability"]) > 0.0
    assert (
        float(payload["structural_rj_local_position_move_probability"])
        == 1.0
    )
    assert len(payload["structural_cardinality_prior_probs"]) == (
        int(payload["pf_max_sources"]) + 1
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json",
        "configs/geant4/high_fidelity_external_no_isaac.json",
        "configs/python/high_fidelity_no_isaac.json",
    ],
)
def test_standard_runtime_configs_select_parallel_compute_paths(
    relative_path: str,
) -> None:
    """Standard runtimes must select parallel planning worker paths."""
    root = Path(__file__).resolve().parents[1]
    payload = load_runtime_config(root / relative_path)

    assert int(payload["python_worker_count"]) > 1
    assert int(payload["ig_workers"]) > 1
    assert int(payload["pose_selection_workers"]) > 1
    assert int(payload["dss_pp"]["program_eval_workers"]) > 1
    # Standard replay keeps isotope order deterministic; parallel work stays
    # inside batched PF kernels and independent planner evaluations.
    assert payload["parallel_isotope_updates"] is False


def test_final_estimates_are_projected_directly_from_pf_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The array view must be a direct PF-posterior projection."""
    estimator = object.__new__(PurePFEstimator)
    point_estimate = posterior_point_estimate_from_states(
        [
            SimpleNamespace(
                num_sources=1,
                positions=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
                strengths=np.asarray([25.0], dtype=float),
                background=0.0,
            )
        ],
        np.asarray([1.0], dtype=float),
        max_cardinality=1,
    )
    monkeypatch.setattr(
        estimator,
        "posterior_point_estimate",
        lambda: {"Cs-137": point_estimate},
    )

    actual = PurePFEstimator.estimates(estimator)

    np.testing.assert_array_equal(
        actual["Cs-137"][0],
        np.asarray([[1.0, 2.0, 3.0]], dtype=float),
    )
    np.testing.assert_array_equal(
        actual["Cs-137"][1],
        np.asarray([25.0], dtype=float),
    )
def test_pure_posterior_projects_surface_particle_mean_to_surface() -> None:
    """A mean of separated surface particles must remain in the source state space."""
    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[0.0, 1.0, 1.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(2.0, 2.0, 2.0),
        ),
        measurement_log_sha256="b" * 64,
    )
    estimator.add_measurement_pose(np.asarray([1.0, 1.0, 1.0], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.asarray([position], dtype=float),
                strengths=np.asarray([10.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for position in ((0.0, 1.0, 1.0), (2.0, 1.0, 1.0))
    ]

    point_estimate = estimator.posterior_point_estimate()["Cs-137"]
    reported_positions = estimator.estimates()["Cs-137"][0]

    assert len(point_estimate.modes) == 1
    assert (
        source_surface_kind(
            point_estimate.modes[0].position_mean_xyz,
            environment,
        )
        is not None
    )
    assert source_surface_kind(reported_positions[0], environment) is not None
    assert (
        estimator.filters["Cs-137"].structural_surface_patch_indices(
            reported_positions,
            strict=True,
        )[0]
        >= 0
    )


def test_exact_surface_dictionary_drives_projection_and_surface_kinds() -> None:
    """PF report projection and labels must use the exact transport-box dictionary."""
    isotope = "Cs-137"
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
    )
    estimator = PurePFEstimator(
        isotopes=(isotope,),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_patch_spacing_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
        measurement_log_sha256="b" * 64,
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    patches = filt._structural_rj_surface_patches
    assert patches is not None

    actual_kinds = filt.structural_surface_kinds(
        patches.centers_xyz,
        strict=True,
    )
    np.testing.assert_array_equal(
        actual_kinds,
        np.asarray(patches.kinds, dtype=object),
    )
    assert {
        "floor",
        "ceiling",
        "wall",
        "obstacle_side",
        "obstacle_top",
        "obstacle_bottom",
    }.issubset(set(actual_kinds.tolist()))
    bottom_indices = np.flatnonzero(actual_kinds == "obstacle_bottom")
    np.testing.assert_allclose(
        patches.normals_xyz[bottom_indices],
        np.tile(np.asarray([0.0, 0.0, -1.0]), (bottom_indices.size, 1)),
    )
    assert all(patches.face_ids[index].endswith("_z0") for index in bottom_indices)
    assert np.sum(patches.areas_m2[bottom_indices]) == pytest.approx(0.36)

    representative_indices = np.asarray(
        [
            int(np.flatnonzero(actual_kinds == kind)[0])
            for kind in sorted(set(actual_kinds.tolist()))
        ],
        dtype=np.int64,
    )
    representative_centers = patches.centers_xyz[representative_indices]
    np.testing.assert_array_equal(
        filt._project_positions_to_source_prior(patches.centers_xyz),
        patches.centers_xyz,
    )

    query = np.asarray([[1.55, 1.61, 0.83]], dtype=float)
    nearest_index = int(
        np.argmin(
            np.sum(
                (patches.centers_xyz - query[0][None, :]) ** 2,
                axis=1,
            )
        )
    )
    projected = filt._project_positions_to_source_prior(query)
    np.testing.assert_array_equal(
        projected,
        patches.centers_xyz[[nearest_index]],
    )
    assert filt.structural_surface_patch_indices(projected, strict=True)[0] == (
        nearest_index
    )
    signed_zero_floor = representative_centers[
        np.flatnonzero(actual_kinds[representative_indices] == "floor")[0]
    ].copy()
    signed_zero_floor[2] = -0.0
    assert (
        filt.structural_surface_patch_indices(
            signed_zero_floor[None, :],
            strict=True,
        )[0]
        == representative_indices[
            np.flatnonzero(actual_kinds[representative_indices] == "floor")[0]
        ]
    )
    dictionary_outside = np.asarray([[0.0, 0.3, 0.3]], dtype=float)
    assert filt.structural_surface_patch_indices(
        dictionary_outside,
        strict=False,
    )[0] == -1
    assert filt.structural_surface_kinds(
        dictionary_outside,
        strict=False,
    )[0] is None


def test_posterior_aligns_swapped_labels_and_reports_uncertainty() -> None:
    """Spatial modes must not collapse when particle source labels are swapped."""
    states = [
        SimpleNamespace(
            num_sources=2,
            positions=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            strengths=np.asarray([10.0, 20.0]),
            background=2.0,
        ),
        SimpleNamespace(
            num_sources=2,
            positions=np.asarray([[2.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            strengths=np.asarray([22.0, 12.0]),
            background=4.0,
        ),
        SimpleNamespace(
            num_sources=1,
            positions=np.asarray([[1.0, 1.0, 0.0]]),
            strengths=np.asarray([7.0]),
            background=8.0,
        ),
    ]
    estimate = posterior_point_estimate_from_states(
        states,
        np.asarray([0.45, 0.35, 0.20]),
        max_cardinality=2,
    )

    assert estimate.map_cardinality == 2
    assert estimate.cardinality_distribution == pytest.approx({0: 0.0, 1: 0.2, 2: 0.8})
    assert len(estimate.modes) == 2
    assert estimate.modes[0].position_mean_xyz[0] < 0.2
    assert estimate.modes[1].position_mean_xyz[0] > 2.0
    assert estimate.modes[0].strength_mean_cps_1m < 13.0
    assert estimate.modes[1].strength_mean_cps_1m > 19.0
    for mode in estimate.modes:
        covariance = np.asarray(mode.position_covariance_xyz)
        assert np.allclose(covariance, covariance.T)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-12
        assert mode.credible_radius_95_m >= 0.0
        lower, upper = mode.strength_credible_interval_95_cps_1m
        assert 0.0 <= lower <= upper
        assert mode.posterior_mass == pytest.approx(0.8)
    payload = estimate.to_dict()
    assert "background_rate_mean_cps" in payload
    assert "background_rate_credible_interval_95_cps" in payload
    assert "background_mean_counts" not in payload
