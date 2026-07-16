"""Scientific-variant and PF-only posterior aggregation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mission_control import resolve_mission_max_poses, resolve_mission_max_steps
from pf.estimator import RotatingShieldPFConfig
from pf.posterior import posterior_point_estimate_from_states
from pf.profiles import (
    ProposalOrigin,
    apply_profile_to_config,
    enforce_pure_runtime_settings,
    resolve_estimator_profile,
)
from pf.pure_estimator import PurePFEstimator
from planning.dss_pp import extract_signature_modes
from sim.runtime import load_runtime_config


@pytest.mark.parametrize("profile", ["pf_strict", "pf_profiled"])
def test_profiles_override_hostile_batch_booleans(profile: str) -> None:
    """Legacy booleans cannot grant any all-history capability."""
    hostile = {
        "estimator_profile": profile,
        "sparse_poisson_evidence_enable": True,
        "sparse_poisson_joint_evidence_enable": True,
        "report_mle_rescue_enable": True,
        "runtime_report_rescue_enable": True,
        "all_history_dictionary_proposal_enable": True,
        "surface_map_reconstruction_enable": True,
        "report_cluster_model_selection": True,
        "report_strength_refit": True,
        "report_surface_local_refine": True,
        "joint_observation_update": True,
        "delayed_resample_update": False,
        "dss_pp": {
            "include_runtime_rescue_modes": True,
            "include_global_surface_rescue_modes": True,
        },
    }
    resolved = enforce_pure_runtime_settings(hostile)
    for key in (
        "sparse_poisson_evidence_enable",
        "sparse_poisson_joint_evidence_enable",
        "report_mle_rescue_enable",
        "runtime_report_rescue_enable",
        "all_history_dictionary_proposal_enable",
        "surface_map_reconstruction_enable",
        "report_cluster_model_selection",
        "report_strength_refit",
        "report_surface_local_refine",
    ):
        assert resolved[key] is False
    assert resolved["dss_pp"]["include_runtime_rescue_modes"] is False
    assert resolved["dss_pp"]["include_global_surface_rescue_modes"] is False
    # Station-level joint/block-sequential PF updates are not batch estimators.
    assert resolved["joint_observation_update"] is True
    assert resolved["delayed_resample_update"] is False


def test_only_profiled_variant_can_enable_conditional_strength_profile() -> None:
    """The two profiles differ by exactly the declared online strength capability."""
    strict = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        conditional_strength_refit=True,
        conditional_strength_profile_before_likelihood=True,
        sparse_poisson_evidence_enable=True,
        report_mle_rescue_enable=True,
    )
    profiled = RotatingShieldPFConfig(
        estimator_profile="pf_profiled",
        conditional_strength_refit=True,
        conditional_strength_profile_before_likelihood=True,
        sparse_poisson_evidence_enable=True,
        report_mle_rescue_enable=True,
    )
    strict_capabilities = apply_profile_to_config(strict)
    profiled_capabilities = apply_profile_to_config(profiled)

    assert strict.conditional_strength_refit is False
    assert strict.conditional_strength_profile_before_likelihood is False
    assert profiled.conditional_strength_refit is True
    assert profiled.conditional_strength_profile_before_likelihood is True
    assert strict.sparse_poisson_evidence_enable is False
    assert profiled.sparse_poisson_evidence_enable is False
    assert strict.report_mle_rescue_enable is False
    assert profiled.report_mle_rescue_enable is False
    differences = {
        key
        for key, value in strict_capabilities.to_dict().items()
        if value != profiled_capabilities.to_dict()[key]
    }
    assert differences == {"conditional_strength_profile"}


def test_pure_estimator_applies_boundary_before_legacy_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hostile batch flags are disabled before the inherited shell sees config."""
    from pf.estimator import RotatingShieldPFEstimator

    original_init = RotatingShieldPFEstimator.__init__

    def guarded_init(*args: object, **kwargs: object) -> None:
        config = kwargs.get("pf_config")
        assert isinstance(config, RotatingShieldPFConfig)
        assert config.sparse_poisson_evidence_enable is False
        assert config.report_mle_rescue_enable is False
        assert config.runtime_report_rescue_enable is False
        original_init(*args, **kwargs)

    monkeypatch.setattr(RotatingShieldPFEstimator, "__init__", guarded_init)
    hostile = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        sparse_poisson_evidence_enable=True,
        report_mle_rescue_enable=True,
        runtime_report_rescue_enable=True,
    )
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=hostile,
        measurement_log_sha256="b" * 64,
    )
    assert estimator.pf_config.sparse_poisson_evidence_enable is False


def test_pure_planner_uses_only_pf_posterior_and_tentative_origins() -> None:
    """Hostile report/surface modes cannot cross the pure planner boundary."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    state = SimpleNamespace(
        num_sources=2,
        positions=np.asarray([[0.5, 0.5, 0.4], [1.5, 1.5, 1.2]]),
        strengths=np.asarray([10.0, 5.0]),
        tentative_sources=np.asarray([False, True]),
        verification_fail_streaks=np.asarray([0, 0]),
    )

    def forbidden() -> object:
        """Fail if a batch-derived planner source is requested."""
        raise AssertionError("batch/report planner mode was requested")

    estimator = SimpleNamespace(
        isotopes=("Cs-137",),
        profile_capabilities=capabilities,
        planner_belief_sources=("pf_posterior", "pf_tentative"),
        pf_config=SimpleNamespace(
            pseudo_source_quarantine_excludes_runtime=False,
        ),
        planning_particles=lambda **_kwargs: {"Cs-137": ([state], np.asarray([1.0]))},
        runtime_report_rescue_modes=forbidden,
        planning_surface_rescue_modes=forbidden,
    )
    modes = extract_signature_modes(
        estimator,
        mode_cluster_radius_m=0.1,
        include_runtime_rescue_modes=True,
        include_global_surface_rescue_modes=True,
    )
    assert len(modes["Cs-137"]) == 2
    assert estimator.planner_belief_sources == ("pf_posterior", "pf_tentative")

    boundary = object.__new__(PurePFEstimator)
    for origin in (
        ProposalOrigin.PF_BIRTH,
        ProposalOrigin.PF_RESIDUAL,
        ProposalOrigin.PF_SPLIT,
    ):
        assert boundary.accepts_proposal_origin(origin)
    for origin in (
        ProposalOrigin.BATCH_SPARSE,
        ProposalOrigin.REPORT_MLE,
        ProposalOrigin.SURFACE_MAP,
        ProposalOrigin.EXTERNAL_MLE,
    ):
        assert not boundary.accepts_proposal_origin(origin)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/python/experiments/pf_strict_3d.json",
        "configs/python/experiments/pf_profiled_3d.json",
        "configs/geant4/experiments/pf_strict_3d.json",
        "configs/geant4/experiments/pf_profiled_3d.json",
    ],
)
def test_pure_profiles_keep_fixed_budget_and_continuous_3d_planning(
    relative_path: str,
) -> None:
    """Purity profiles retain mission budget, collision, and 3-D planner settings."""
    root = Path(__file__).resolve().parents[1]
    resolved = enforce_pure_runtime_settings(load_runtime_config(root / relative_path))
    assert resolved["adaptive_cardinality_dwell_enable"] is False
    assert resolved["adaptive_mission_stop"] is False
    assert resolved["mission_stop_soft_extend_on_unresolved"] is False
    assert resolved["final_absent_isotope_filter"] is False
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
    expected_variant = "profiled" if "profiled" in relative_path else "strict"
    expected_backend = "geant4" if "geant4" in relative_path else "python"
    assert resolved["measurement_log_output_dir"] == (
        f"logs/pure_pf/{expected_backend}_pf_{expected_variant}_3d_measurement_log"
    )
    dss = resolved["dss_pp"]
    assert dss["include_runtime_rescue_modes"] is False
    assert dss["include_global_surface_rescue_modes"] is False
    assert dss["adaptive_program_length_enable"] is False
    # These inherited settings prove the nested section is fully specified,
    # not accidentally replaced by a three-key shallow override.
    assert int(dss["horizon"]) >= 1
    assert int(dss["program_length"]) >= 1
    assert int(dss["max_programs"]) >= 1


def test_pure_final_report_ignores_legacy_best_so_far_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility report options must not replace the current PF posterior."""
    estimator = object.__new__(PurePFEstimator)
    expected = {
        "Cs-137": (
            np.asarray([[1.0, 2.0, 3.0]], dtype=float),
            np.asarray([25.0], dtype=float),
        )
    }
    monkeypatch.setattr(estimator, "estimates", lambda: expected)

    actual = estimator.final_report_estimate(use_best_so_far=True)

    np.testing.assert_array_equal(actual["Cs-137"][0], expected["Cs-137"][0])
    np.testing.assert_array_equal(actual["Cs-137"][1], expected["Cs-137"][1])


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
