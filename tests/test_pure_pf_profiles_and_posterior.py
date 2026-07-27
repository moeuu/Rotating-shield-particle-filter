"""Scientific-variant and PF-only posterior aggregation tests."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mission_control import resolve_mission_max_poses, resolve_mission_max_steps
from measurement.model import EnvironmentConfig
from measurement.source_surfaces import source_surface_kind
from measurement.surface_patches import build_surface_patch_dictionary
from pf.estimator import MeasurementRecord, RotatingShieldPFConfig
from pf.particle_filter import IsotopeParticle
from pf.posterior import posterior_point_estimate_from_states
from pf.profiles import (
    EstimatorProfile,
    apply_profile_to_config,
    enforce_pure_runtime_settings,
    removed_estimator_config_keys,
    resolve_estimator_profile,
    resolve_structural_transition_provenance,
)
from pf.pure_estimator import PurePFEstimator
from pf.state import IsotopeState
from planning.dss_pp import extract_signature_modes
from sim.runtime import load_runtime_config


def _exact_rj_config(**overrides: object) -> RotatingShieldPFConfig:
    """Build an exact finite-surface RJ-MH config for focused tests."""
    values: dict[str, object] = {
        "estimator_profile": "pf_strict",
        "source_position_prior": "surface",
        "max_sources": 5,
        "init_num_sources": (0, 5),
    }
    values.update(overrides)
    return RotatingShieldPFConfig(**values)


@pytest.mark.parametrize("profile", [None, "pf_strict", "strict", "pure_pf", "pf_only"])
def test_only_strict_profile_is_supported(profile: str | None) -> None:
    """Every supported alias must resolve to the single strict PF profile."""
    resolved_profile, capabilities = resolve_estimator_profile(profile)

    assert resolved_profile is EstimatorProfile.PF_STRICT
    assert capabilities.sequential_updates_only is True
    assert capabilities.posterior_reporting_only is True
    assert capabilities.likelihood_consistent_structural_evidence is True


@pytest.mark.parametrize("profile", ["pf_profiled", "profiled", "mle", "batch"])
def test_removed_profiles_fail_fast(profile: str) -> None:
    """Removed estimator variants must be rejected instead of downgraded."""
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        resolve_estimator_profile(profile)
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        enforce_pure_runtime_settings({}, profile=profile)


@pytest.mark.parametrize(
    ("payload", "removed_key"),
    [
        ({"report_mle_rescue_enable": False}, "report_mle_rescue_enable"),
        ({"sparse_poisson_evidence_enable": False}, "sparse_poisson_evidence_enable"),
        (
            {"surface_map_reconstruction_enable": False},
            "surface_map_reconstruction_enable",
        ),
        ({"conditional_strength_refit": False}, "conditional_strength_refit"),
        ({"adaptive_strength_prior": False}, "adaptive_strength_prior"),
        (
            {"birth_residual_force_proposal_on_gate": False},
            "birth_residual_force_proposal_on_gate",
        ),
        (
            {"birth_existing_response_corr_max": 0.99},
            "birth_existing_response_corr_max",
        ),
        (
            {"birth_response_condition_max": 100.0},
            "birth_response_condition_max",
        ),
        ({"death_require_low_strength": False}, "death_require_low_strength"),
        ({"high_strength_split_enable": False}, "high_strength_split_enable"),
        (
            {"report_exclude_unverified_sources": False},
            "report_exclude_unverified_sources",
        ),
        (
            {"pseudo_source_quarantine_excludes_runtime": False},
            "pseudo_source_quarantine_excludes_runtime",
        ),
        ({"structural_kernel_mode": "rj_mh"}, "structural_kernel_mode"),
        ({"p_birth": 0.1}, "p_birth"),
        ({"p_kill": 0.1}, "p_kill"),
        ({"birth_residual_min_support": 1}, "birth_residual_min_support"),
        (
            {"pseudo_source_verification_enable": False},
            "pseudo_source_verification_enable",
        ),
        ({"split_prob": 0.0}, "split_prob"),
        ({"merge_prob": 0.0}, "merge_prob"),
        ({"mode_preserving_resample": False}, "mode_preserving_resample"),
        (
            {"cardinality_preserving_resample": False},
            "cardinality_preserving_resample",
        ),
        ({"roughening_k": 0.0}, "roughening_k"),
        ({"init_grid_repeats": 1}, "init_grid_repeats"),
        ({"init_joint_position_design": False}, "init_joint_position_design"),
        ({"source_detector_exclusion_m": 0.0}, "source_detector_exclusion_m"),
        ({"init_source_min_separation_m": 0.0}, "init_source_min_separation_m"),
        ({"surface_rejuvenation_enable": False}, "surface_rejuvenation_enable"),
        ({"structural_trial_workers": 1}, "structural_trial_workers"),
        ({"support_window": 0}, "support_window"),
        ({"birth_residual_always_try": False}, "birth_residual_always_try"),
        ({"split_residual_always_try": False}, "split_residual_always_try"),
        (
            {"mission_stop_require_model_order_ready": False},
            "mission_stop_require_model_order_ready",
        ),
        (
            {"online_absent_isotope_pruning": False},
            "online_absent_isotope_pruning",
        ),
        (
            {"adaptive_cardinality_min_bic_margin": 0.0},
            "adaptive_cardinality_min_bic_margin",
        ),
        ({"refit_after_moves": False}, "refit_after_moves"),
        ({"use_clustered_output": False}, "use_clustered_output"),
        ({"cluster_eps_m": 0.8}, "cluster_eps_m"),
        ({"converge_freeze_updates": False}, "converge_freeze_updates"),
        ({"source_position_min": [0.0, 0.0, 0.0]}, "source_position_min"),
        ({"source_position_max": [1.0, 1.0, 1.0]}, "source_position_max"),
        ({"source_z_min_m": 0.0}, "source_z_min_m"),
        ({"source_z_max_m": 1.0}, "source_z_max_m"),
        (
            {"dss_pp": {"include_runtime_rescue_modes": False}},
            "dss_pp.include_runtime_rescue_modes",
        ),
        (
            {"dss_pp": {"lambda_cardinality_discrimination": 0.0}},
            "dss_pp.lambda_cardinality_discrimination",
        ),
        (
            {"dss_lambda_cardinality_discrimination": 0.0},
            "dss_lambda_cardinality_discrimination",
        ),
        (
            {
                "remaining_measurement_estimate": {
                    "report_residual_weight": 0.0,
                }
            },
            "remaining_measurement_estimate.report_residual_weight",
        ),
        (
            {"remaining_measurement_report_residual_weight": 0.0},
            "remaining_measurement_report_residual_weight",
        ),
    ],
)
def test_removed_estimator_keys_fail_fast(
    payload: dict[str, object],
    removed_key: str,
) -> None:
    """Even false-valued removed keys must fail instead of becoming no-op flags."""
    assert removed_estimator_config_keys(payload) == (removed_key,)
    with pytest.raises(ValueError, match=removed_key.replace(".", r"\.")):
        enforce_pure_runtime_settings(payload)


def test_pf_config_physically_omits_removed_estimator_fields() -> None:
    """The PF dataclass must not retain MLE, surface-map, or refit switches."""
    config_fields = {field.name for field in fields(RotatingShieldPFConfig)}
    removed_fields = {
        "all_history_dictionary_proposal_enable",
        "birth_existing_response_corr_max",
        "birth_response_condition_max",
        "conditional_strength_refit",
        "conditional_strength_profile_before_likelihood",
        "report_mle_rescue_enable",
        "report_strength_refit",
        "runtime_report_rescue_enable",
        "sparse_poisson_evidence_enable",
        "surface_map_reconstruction_enable",
        "pseudo_source_quarantine_excludes_runtime",
        "structural_kernel_mode",
        "p_birth",
        "p_kill",
        "birth_residual_min_support",
        "pseudo_source_verification_enable",
        "split_prob",
        "merge_prob",
        "mode_preserving_resample",
        "cardinality_preserving_resample",
        "roughening_k",
        "init_grid_repeats",
        "init_joint_position_design",
        "source_detector_exclusion_m",
        "init_source_min_separation_m",
        "surface_rejuvenation_enable",
        "structural_trial_workers",
        "use_clustered_output",
        "cluster_eps_m",
        "cluster_min_samples",
        "cluster_report_max_points",
        "cluster_exact_max_points",
        "converge_freeze_updates",
        "converge_cluster_spread_max_m",
        "converge_cluster_min_support_fraction",
    }

    assert config_fields.isdisjoint(removed_fields)


def test_strict_profile_requires_environment_surface_source_support() -> None:
    """The declared pure-PF capability must reject a volume source prior."""
    assert RotatingShieldPFConfig().source_position_prior == "surface"
    with pytest.raises(
        ValueError,
        match="source_position_prior='surface'",
    ):
        apply_profile_to_config(
            RotatingShieldPFConfig(source_position_prior="volume")
        )
    with pytest.raises(ValueError, match="source_surface_prior=true"):
        enforce_pure_runtime_settings({"source_surface_prior": False})
    with pytest.raises(
        ValueError,
        match="source_position_prior='surface'",
    ):
        enforce_pure_runtime_settings({"source_position_prior": "volume"})


def test_fixed_k_provenance_declares_target_preserving_no_move_kernel() -> None:
    """Fixed-K provenance must declare a target-preserving no-move kernel."""
    fixed_config = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        init_num_sources=(3, 3),
        birth_enable=False,
    )
    fixed_capabilities = apply_profile_to_config(fixed_config)
    fixed = resolve_structural_transition_provenance(
        fixed_config,
        capabilities=fixed_capabilities,
    ).to_dict()

    assert fixed["posterior_semantics"] == (
        "fixed_cardinality_sequential_particle_filter"
    )
    assert fixed["structural_kernel_family"] == (
        "fixed_cardinality_no_structural_moves"
    )
    assert fixed["structural_moves_enabled"] is False
    assert fixed["structural_kernel_target_preserving"] is True
    assert fixed["structural_kernel_exact_rj"] is False
    assert fixed["reversible_jump_mcmc_used"] is False
    assert fixed["data_conditioned_structural_proposal"] is False
    assert fixed["structural_evidence_uses_pf_likelihood"] is True


def test_exact_rj_provenance_declares_target_preserving_pf_kernel() -> None:
    """Exact RJ-MH mode must report its finite-surface posterior semantics."""
    config = _exact_rj_config(
        init_num_sources=(0, 5),
        birth_enable=True,
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
    assert provenance["structural_kernel_target_preserving"] is True
    assert provenance["structural_kernel_exact_rj"] is True
    assert provenance["reversible_jump_mcmc_used"] is True
    assert provenance["data_conditioned_structural_proposal"] is False
    assert provenance["data_conditioned_strength_proposal"] is False
    assert provenance["structural_evidence_uses_pf_likelihood"] is True


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
            init_strength_prior="uniform",
            init_strength_min=300_000.0,
            init_strength_max=2_000_000.0,
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
        for isotope in estimator.all_isotopes
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


def test_structural_history_prefers_view_covariance_over_unrecorded_spectrum() -> None:
    """Recorded structural history must infer the covariance runtime route."""
    covariance = ((4.0, 1.5), (1.5, 9.0))
    common = {
        "z_k": {"Cs-137": 12.0},
        "pose_idx": 0,
        "orient_idx": 0,
        "live_time_s": 1.0,
        "z_variance_k": {"Cs-137": 4.0},
        "spectrum_response_templates_by_isotope": {
            "Cs-137": (0.25, 0.75)
        },
        "spectrum_background": (1.0, 2.0),
        "station_sequence_id": 17,
        "station_view_covariance_by_isotope": {"Cs-137": covariance},
    }
    records = [
        MeasurementRecord(
            **common,
            fe_index=0,
            pb_index=0,
            spectrum_counts=(4.0, 8.0),
            station_view_index=0,
        ),
        MeasurementRecord(
            **{**common, "z_k": {"Cs-137": 20.0}, "orient_idx": 1},
            fe_index=1,
            pb_index=1,
            spectrum_counts=(7.0, 13.0),
            station_view_index=1,
        ),
    ]
    estimator = object.__new__(PurePFEstimator)
    estimator.measurements = records
    estimator.poses = [np.asarray([1.0, 2.0, 0.5], dtype=float)]

    data = estimator._measurement_data_for_iso("Cs-137", window=None)

    assert data is not None
    np.testing.assert_allclose(data.observation_count_covariance, covariance)
    assert data.runtime_likelihood_routes is not None
    assert data.runtime_likelihood_routes.tolist() == [
        "count_covariance",
        "count_covariance",
    ]
    assert data.spectrum_counts is None
    assert data.spectrum_response_template is None
    assert data.spectrum_background is None

    final_row = estimator._measurement_data_for_iso("Cs-137", window=1)
    assert final_row is not None
    np.testing.assert_allclose(final_row.observation_count_covariance, [[9.0]])
    assert final_row.runtime_likelihood_routes is not None
    assert final_row.runtime_likelihood_routes.tolist() == ["direct_spectrum"]
    np.testing.assert_allclose(final_row.spectrum_counts, [[7.0, 13.0]])


def test_removed_estimator_methods_are_physically_absent() -> None:
    """Pure PF must not retain refusal stubs for deleted estimator families."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        measurement_log_sha256="b" * 64,
    )

    removed_methods = (
        "_refit_reported_strengths",
        "_surface_stratified_rescue_indices",
        "final_report_estimate",
        "fit_surface_map",
        "planning_surface_rescue_modes",
        "refresh_sparse_poisson_evidence",
        "report_model_order_diagnostics",
        "runtime_report_rescue_modes",
        "sparse_poisson_evidence_diagnostics",
    )
    for method_name in removed_methods:
        assert not hasattr(PurePFEstimator, method_name)
        assert not hasattr(estimator, method_name)
    assert not hasattr(estimator, "forbidden_batch_entry_points")
    assert not hasattr(estimator, "batch_methods_invoked")


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
    assert resolved["mission_stop_soft_extend_on_unresolved"] is False
    assert "final_absent_isotope_filter" not in resolved
    assert resolved["measurement_budget_max_steps"] == 160
    assert resolved["mission_stop_max_poses"] == 20
    assert resolve_mission_max_steps(None, resolved) == 160
    assert resolve_mission_max_poses(None, resolved) == 20
    assert resolved["parallel_isotope_updates"] is False
    assert resolved["detector_height_sampling_mode"] == "continuous"
    assert resolved["measurement_pose_clearance_enabled"] is True
    assert resolved["path_planner"] == "dss_pp"
    assert resolved["spectrum_count_method"] == "response_poisson"
    assert resolved["joint_observation_update"] is False
    assert resolved["delayed_resample_update"] is True
    expected_backend = "geant4" if "geant4" in relative_path else "python"
    assert resolved["measurement_log_output_dir"] == (
        f"logs/pure_pf/{expected_backend}_pf_strict_3d_measurement_log"
    )
    dss = resolved["dss_pp"]
    assert "include_runtime_rescue_modes" not in dss
    assert "include_global_surface_rescue_modes" not in dss
    assert dss["adaptive_program_length_enable"] is False
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

    assert payload["estimator_profile"] == "pf_strict"
    forbidden = (
        "all_history_dictionary_proposal_enable",
        "birth_global_rescue_enable",
        "conditional_strength_profile_before_likelihood",
        "conditional_strength_refit",
        "report_cluster_model_selection",
        "report_mle_rescue_enable",
        "report_strength_refit",
        "report_surface_local_refine",
        "runtime_report_rescue_enable",
        "sparse_poisson_evidence_enable",
        "source_strength_prior_mean",
        "surface_map_reconstruction_enable",
    )
    for field in forbidden:
        assert field not in payload
    dss = payload.get("dss_pp", {})
    assert dss.get("adaptive_program_length_enable", False) is False
    assert "include_runtime_rescue_modes" not in dss
    assert "include_global_surface_rescue_modes" not in dss


def test_standard_geant4_config_selects_exact_surface_rj_kernel() -> None:
    """The production Geant4 config must select only exact surface RJ-MH."""
    root = Path(__file__).resolve().parents[1]
    payload = load_runtime_config(
        root
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )

    assert payload["birth_enable"] is True
    assert float(payload["structural_rj_patch_spacing_m"]) > 0.0
    assert float(payload["structural_rj_move_probability"]) > 0.0
    assert float(payload["structural_rj_birth_probability"]) > 0.0
    assert float(payload["structural_rj_death_probability"]) > 0.0
    assert (
        float(payload["structural_rj_local_position_move_probability"])
        == 1.0
    )
    assert payload["source_surface_prior"] is True
    assert len(payload["structural_cardinality_prior_probs"]) == (
        int(payload["pf_max_sources"]) + 1
    )
    heuristic_keys = (
        "structural_kernel_mode",
        "p_birth",
        "p_kill",
        "split_prob",
        "merge_prob",
        "surface_rejuvenation_enable",
        "mode_preserving_resample",
        "cardinality_preserving_resample",
        "pseudo_source_verification_enable",
        "source_detector_exclusion_m",
        "pf_init_source_min_separation_m",
        "deferred_resample_roughening_scale",
        "disable_regularize_on_temper_resample",
        "structural_trial_workers",
        "structural_trial_parallel_min_trials",
    )
    assert set(payload).isdisjoint(heuristic_keys)


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
    # Per-isotope filters currently share NumPy's deterministic RNG stream.
    # Parallel work therefore stays inside batched PF and planner computations.
    assert payload["parallel_isotope_updates"] is False


def test_final_estimates_are_projected_directly_from_pf_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical array view must be a direct PF-posterior projection."""
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
    assert not hasattr(
        estimator,
        "final_report_estimate",
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
            birth_enable=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_min=(0.0, 0.0, 0.0),
            position_max=(2.0, 2.0, 2.0),
            source_position_prior="surface",
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
