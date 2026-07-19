"""Tests for Geant4 calibration helper functions."""

import inspect
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.obstacles import ObstacleGrid


def _load_calibration_script() -> object:
    """Load the calibration script as a module for helper tests."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "calibrate_geant4_net_response.py"
    spec = spec_from_file_location("calibrate_geant4_net_response", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_validation_script() -> object:
    """Load the spectrum validation script as a module for helper tests."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "validate_geant4_spectrum_decomposition.py"
    spec = spec_from_file_location(
        "validate_geant4_spectrum_decomposition", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATION_SCRIPT = _load_calibration_script()
VALIDATION_SCRIPT = _load_validation_script()


def test_relative_error_omits_zero_target_at_zero_threshold() -> None:
    """A diagnostic threshold of zero must not divide by a zero target."""
    assert VALIDATION_SCRIPT.relative_error(0.0, 0.0, 0.0) is None


def test_validation_metadata_keeps_native_fidelity_provenance() -> None:
    """Validation artifacts should prove threading and full-history semantics."""
    metadata = {
        "backend": "geant4",
        "requested_threads": 32,
        "multithreaded_run_manager": True,
        "source_rate_model": "detector_cps_1m",
        "source_bias_mode": "detector_cone",
        "intensity_cps_1m_definition": "net_detector_count_rate_at_1m",
        "line_intensities_normalized": True,
        "detector_response_applied_in_native": False,
        "secondary_transport_mode": "gamma_only",
        "gamma_only_secondary_transport": True,
        "theory_tvl_attenuation": False,
        "background_cps": 12.0,
        "poisson_background": True,
        "expected_detector_equivalent_primaries": 1234.5,
        "expected_sampled_primaries": 1234.5,
        "expected_primary_semantics": "detector_equivalent_histories",
        "primary_sampling_fraction": 1.0,
        "primary_history_weight": 1.0,
        "unrelated_internal_field": "omit",
    }

    retained = VALIDATION_SCRIPT.validation_observation_metadata(metadata)

    assert retained == {
        "backend": "geant4",
        "background_cps": 12.0,
        "detector_response_applied_in_native": False,
        "expected_detector_equivalent_primaries": 1234.5,
        "expected_primary_semantics": "detector_equivalent_histories",
        "expected_sampled_primaries": 1234.5,
        "intensity_cps_1m_definition": "net_detector_count_rate_at_1m",
        "gamma_only_secondary_transport": True,
        "line_intensities_normalized": True,
        "multithreaded_run_manager": True,
        "primary_history_weight": 1.0,
        "primary_sampling_fraction": 1.0,
        "poisson_background": True,
        "requested_threads": 32,
        "secondary_transport_mode": "gamma_only",
        "source_bias_mode": "detector_cone",
        "source_rate_model": "detector_cps_1m",
        "theory_tvl_attenuation": False,
    }


def test_validation_metadata_keeps_weighted_covariance_provenance() -> None:
    """Weighted validation artifacts must retain every acceptance field."""
    metadata = {
        "accelerated_weighted_transport_enable": True,
        "primary_sampling_budget_enabled": True,
        "primary_sampling_fraction_resolution": "target_budget_limited",
        "requested_primary_sampling_fraction": 1.0,
        "target_sampled_primaries": 1_500_000,
        "expected_unthinned_primaries": 10_000.0,
        "history_thinning_enabled": True,
        "transport_history_mode": "weighted_thinning",
        "transport_tally_weighted": True,
        "spectrum_variance_semantics": ("compound_poisson_sumw2_includes_counting"),
        "spectrum_variance_dead_time_propagation": "fixed_observed_scale",
        "dead_time_tau_s": 5.813e-9,
        "dead_time_observed_scale": 0.99,
        "pre_dead_time_total_spectrum_counts": 100.0,
        "pre_dead_time_weighted_spectrum_sumw2": 5_000.0,
        "dwell_time_s": 30.0,
        "effective_entries_per_sec": 4.0,
        "primaries_per_sec": 5.0,
        "seed": 123,
        "unrelated_internal_field": "omit",
    }

    retained = VALIDATION_SCRIPT.validation_observation_metadata(metadata)

    assert retained == {
        key: value
        for key, value in metadata.items()
        if key != "unrelated_internal_field"
    }


def test_source_tally_counts_are_history_normalized() -> None:
    """Native source-equivalent tallies should be divided by history scale."""
    metadata = {
        "source_equivalent_counts_Cs-137": 500.0,
        "source_equivalent_counts_Co-60": 100.0,
    }

    counts = CALIBRATION_SCRIPT._source_tally_counts(
        metadata,
        ["Cs-137", "Co-60", "Eu-154"],
        history_scale=5.0,
    )

    assert counts["Cs-137"] == pytest.approx(100.0)
    assert counts["Co-60"] == pytest.approx(20.0)
    assert counts["Eu-154"] == pytest.approx(0.0)


def test_scale_sources_multiplies_intensity_only() -> None:
    """History scaling should keep source identity and position unchanged."""
    source = PointSource("Cs-137", position=(1.0, 2.0, 3.0), intensity_cps_1m=10.0)

    scaled = CALIBRATION_SCRIPT._scale_sources([source], 4.0)

    assert scaled[0].isotope == source.isotope
    assert scaled[0].position == source.position
    assert scaled[0].intensity_cps_1m == pytest.approx(40.0)


def test_validation_outputs_export_runtime_transport_model(tmp_path: Path) -> None:
    """Validation output should include a runtime-loadable transport model file."""
    model = {
        "enabled": True,
        "by_isotope": {
            "Cs-137": {
                "scale": 1.05,
                "tau_coefficients": {"obstacle": 0.1},
            }
        },
    }
    summary = {
        "pf_transport_response_calibration": {
            "transport_response_model": model,
        },
    }

    VALIDATION_SCRIPT.write_outputs(
        tmp_path,
        results=[],
        spectra={},
        summary=summary,
        cases=None,
        write_detailed_results=False,
    )

    model_path = tmp_path / "pf_transport_response_model.json"
    assert model_path.exists()
    payload = VALIDATION_SCRIPT.json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["pf_transport_response_model"] == model


def test_validation_uses_runtime_covariance_projection_helper() -> None:
    """Validation projection should exactly match the runtime PF envelope."""
    projector = VALIDATION_SCRIPT.build_runtime_covariance_projector(
        {
            "observation_covariance_projection_enable": True,
            "observation_covariance_projection_weight": 1.0,
            "observation_covariance_projection_max_corr": 0.999,
        }
    )
    counts = {"Cs-137": 100.0, "Co-60": 80.0}
    variances = {"Cs-137": 25.0, "Co-60": 16.0}
    covariance = {
        "Cs-137": {"Cs-137": 25.0, "Co-60": -12.0},
        "Co-60": {"Cs-137": -12.0, "Co-60": 16.0},
    }

    projected, sanitized = VALIDATION_SCRIPT.project_runtime_observation_covariance(
        projector,
        counts,
        variances,
        covariance,
    )

    assert projected == pytest.approx({"Cs-137": 37.0, "Co-60": 28.0})
    assert sanitized is not None
    assert sanitized["Cs-137"]["Co-60"] == pytest.approx(-12.0)
    assert sanitized["Co-60"]["Cs-137"] == pytest.approx(-12.0)


def test_validation_covariance_summary_reports_stages_and_coverage() -> None:
    """Covariance summary should retain stage ratios, ceilings, and coverage."""
    stage = {
        "formal_variance": 100.0,
        "ceilinged_formal_variance": 25.0,
        "runtime_variance": 36.0,
        "projected_variance": 45.0,
        "formal_ceiling_applied": True,
    }
    mahalanobis = {
        "formal_full": 1.0,
        "formal_diagonal": 1.0,
        "runtime_full": 2.0,
        "runtime_diagonal": 2.0,
        "estimator_sanitized_full": 2.0,
        "projected_diagonal": 1.6,
    }
    result = {
        "case": {"include_in_accuracy_summary": True},
        "response_poisson_covariance": {
            "residual_diagnostics": {
                "degrees_of_freedom": 1,
                "mahalanobis_squared": mahalanobis,
            }
        },
        "per_isotope": {
            "Cs-137": {
                "transport_truth_counts": 90.0,
                "method_counts": {"response_poisson": 100.0},
                "response_poisson_formal_variance": 100.0,
                "response_poisson_ceilinged_formal_variance": 25.0,
                "response_poisson_variance": 36.0,
                "response_poisson_projected_variance": 45.0,
                "response_poisson_variance_stages": stage,
            }
        },
    }

    summary = VALIDATION_SCRIPT.summarize_response_poisson_covariance(
        [result],
        min_target=25.0,
    )

    assert summary["num_isotope_records"] == 1
    assert summary["variance_ratios"]["runtime_over_formal"]["median"] == (
        pytest.approx(0.36)
    )
    assert summary["variance_ratios"]["projected_over_runtime"]["median"] == (
        pytest.approx(1.25)
    )
    assert summary["ceiling_exceedance_counts"] == {
        "formal_ceiling_applied": 1,
        "runtime_above_ceilinged_formal": 1,
        "projected_above_ceilinged_formal": 1,
        "projected_above_runtime": 1,
    }
    projected_coverage = summary["normalized_residual_coverage_vs_transport_truth"][
        "projected"
    ]
    assert projected_coverage["num_points"] == 1
    assert projected_coverage["coverage_within_2sigma"] == pytest.approx(1.0)
    projected_mahalanobis = summary["mahalanobis_vs_transport_truth"][
        "projected_diagonal"
    ]
    assert projected_mahalanobis["num_cases"] == 1
    assert projected_mahalanobis["pooled_squared_per_degree_of_freedom"] == (
        pytest.approx(1.6)
    )


def _pair_screening_case() -> object:
    """Return a compact multi-isotope case for pair-screening tests."""
    return VALIDATION_SCRIPT.ValidationCase(
        name="pair_screening",
        description="test",
        detector_pose_xyz=(1.0, 1.0, 0.5),
        sources=(
            VALIDATION_SCRIPT.ValidationSource(
                "Cs-137",
                (2.0, 1.0, 0.5),
                1000.0,
            ),
            VALIDATION_SCRIPT.ValidationSource(
                "Co-60",
                (1.0, 2.0, 0.5),
                1500.0,
            ),
            VALIDATION_SCRIPT.ValidationSource(
                "Eu-154",
                (1.0, 1.0, 1.5),
                2000.0,
            ),
        ),
        dwell_time_s=2.0,
    )


def _empty_obstacle_grid() -> ObstacleGrid:
    """Return a single-cell free obstacle grid for selection-helper tests."""
    return ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
    )


def test_all_pair_count_batch_matches_serial_oracle() -> None:
    """Torch all-pair batching should preserve the scalar PF response exactly."""
    case = _pair_screening_case()
    runtime_config = {
        "measurement_scale_by_isotope": {"Cs-137": 1.1, "Co-60": 0.9},
        "measurement_scale_by_isotope_and_pair": {
            "Cs-137": {"7": 1.25},
        },
    }
    scalar_kernel = ContinuousKernel(use_gpu=False)
    batched_kernel = ContinuousKernel(
        use_gpu=True,
        gpu_device="cpu",
        gpu_dtype="float64",
    )
    pair_ids = (0, 7, 18, 63)

    expected = VALIDATION_SCRIPT._expected_count_matrix_over_shield_pairs_serial(
        case,
        runtime_config,
        scalar_kernel,
        pair_ids=pair_ids,
    )
    actual = VALIDATION_SCRIPT._expected_count_matrix_over_shield_pairs(
        case,
        runtime_config,
        batched_kernel,
        pair_ids=pair_ids,
    )

    assert actual == pytest.approx(expected, rel=1.0e-10, abs=1.0e-10)


def test_all_pair_screening_enables_batched_torch_backend() -> None:
    """Exact validation screening should never retain the scalar kernel path."""
    kernel = ContinuousKernel(use_gpu=False)

    selected = VALIDATION_SCRIPT._enable_batched_pair_screening(
        kernel,
        {
            "validation_pair_screening_device": "cpu",
            "validation_pair_screening_dtype": "float64",
        },
    )

    assert selected is kernel
    assert selected.use_gpu is True
    assert selected.gpu_device == "cpu"
    assert selected.gpu_dtype == "float64"


def test_exact_detector_gate_rejects_higher_ranked_proxy_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy winner below the exact all-pair target must not be selected."""
    candidates = [
        VALIDATION_SCRIPT.DetectorPoseCandidate((1.0, 0.0, 0.5), 2.0, {}),
        VALIDATION_SCRIPT.DetectorPoseCandidate((2.0, 0.0, 0.5), 1.0, {}),
    ]

    def fake_features(
        _base_case: object,
        detector_xyz: tuple[float, float, float],
        **_kwargs: object,
    ) -> dict[str, float]:
        """Return an exact minimum determined by the candidate x coordinate."""
        return {"min_isotope_target": 5000.0 if detector_xyz[0] == 1.0 else 12000.0}

    monkeypatch.setattr(
        VALIDATION_SCRIPT,
        "_detector_selection_features",
        fake_features,
    )

    selected, exact_features, attempts = (
        VALIDATION_SCRIPT._select_exact_target_qualified_detector(
            candidates,
            base_case=_pair_screening_case(),
            transport_grid=_empty_obstacle_grid(),
            runtime_config={},
            target_kernel=ContinuousKernel(use_gpu=False),
            min_target_counts=10000.0,
            all_shield_pairs=True,
        )
    )

    assert selected.pose_xyz[0] == pytest.approx(2.0)
    assert exact_features["min_isotope_target"] == pytest.approx(12000.0)
    assert attempts == 2


def test_exact_detector_gate_fails_when_no_candidate_meets_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scenario without a qualifying exact pose should fail explicitly."""
    candidate = VALIDATION_SCRIPT.DetectorPoseCandidate(
        (1.0, 0.0, 0.5),
        1.0,
        {},
    )

    def fake_features(*_args: object, **_kwargs: object) -> dict[str, float]:
        """Return a deterministic below-threshold exact response."""
        return {"min_isotope_target": 9000.0}

    monkeypatch.setattr(
        VALIDATION_SCRIPT,
        "_detector_selection_features",
        fake_features,
    )

    with pytest.raises(ValueError, match="all 64 shield pairs"):
        VALIDATION_SCRIPT._select_exact_target_qualified_detector(
            [candidate],
            base_case=_pair_screening_case(),
            transport_grid=_empty_obstacle_grid(),
            runtime_config={},
            target_kernel=ContinuousKernel(use_gpu=False),
            min_target_counts=10000.0,
            all_shield_pairs=True,
        )


def test_runtime_likelihood_specs_match_nested_standard_config() -> None:
    """Validation should resolve nested PF likelihood values isotope-wise."""
    specs = VALIDATION_SCRIPT.build_runtime_count_likelihood_specs(
        {
            "backend": "geant4",
            "spectrum_count_method": "response_poisson",
            "pf_count_likelihood": {
                "count_likelihood_model": "student_t",
                "count_likelihood_df": 7.0,
                "transport_model_rel_sigma": {"Cs-137": 0.2, "Co-60": 0.1},
                "spectrum_count_rel_sigma": 0.05,
                "transport_model_abs_sigma": 4.0,
                "spectrum_count_abs_sigma": 3.0,
                "low_count_abs_sigma": 10.0,
                "low_count_transition_counts": 50.0,
                "observation_count_variance_includes_counting_noise": True,
            },
        }
    )

    assert specs["Cs-137"].model == "student_t"
    assert specs["Cs-137"].student_t_df == pytest.approx(7.0)
    assert specs["Cs-137"].transport_model_rel_sigma == pytest.approx(0.2)
    assert specs["Co-60"].transport_model_rel_sigma == pytest.approx(0.1)
    assert specs["Eu-154"].transport_model_rel_sigma == pytest.approx(0.0)
    assert specs["Eu-154"].spectrum_count_rel_sigma == pytest.approx(0.05)
    assert specs["Cs-137"].observation_count_variance_includes_counting_noise is True


def test_pf_likelihood_summary_uses_shared_student_t_scale() -> None:
    """Final PF diagnostics should include projected variance in Student-t scale."""
    counts = {"Cs-137": 110.0, "Co-60": 70.0, "Eu-154": 55.0}
    targets = {"Cs-137": 100.0, "Co-60": 80.0, "Eu-154": 50.0}
    projected = {"Cs-137": 25.0, "Co-60": 16.0, "Eu-154": 9.0}
    specs = {
        isotope: VALIDATION_SCRIPT.CountLikelihoodSpec(
            model="student_t",
            transport_model_rel_sigma=0.1,
            transport_model_abs_sigma=5.0,
            spectrum_count_rel_sigma=0.05,
            spectrum_count_abs_sigma=5.0,
            low_count_abs_sigma=20.0,
            low_count_transition_counts=100.0,
            observation_count_variance_includes_counting_noise=True,
            student_t_df=5.0,
        )
        for isotope in VALIDATION_SCRIPT.ISOTOPES
    }

    diagnostic = VALIDATION_SCRIPT._pf_count_likelihood_diagnostics(
        counts,
        targets,
        targets,
        projected,
        specs,
        min_target=25.0,
    )
    runtime_target = diagnostic["targets"]["runtime_pf_forward"]
    expected_cs_scale = VALIDATION_SCRIPT.count_likelihood_variance(
        np.asarray([110.0]),
        np.asarray([100.0]),
        transport_model_rel_sigma=0.1,
        transport_model_abs_sigma=5.0,
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=5.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
        observation_count_variance=25.0,
        observation_count_variance_includes_counting_noise=True,
    )[0]

    assert runtime_target["likelihood_scale_squared_by_isotope"]["Cs-137"] == (
        pytest.approx(expected_cs_scale)
    )
    assert (
        diagnostic["likelihood_spec_by_isotope"]["Cs-137"][
            "observation_count_variance_includes_counting_noise"
        ]
        is True
    )
    assert "not its marginal variance" in diagnostic["scale_semantics"]

    summary = VALIDATION_SCRIPT.summarize_pf_count_likelihood_diagnostics(
        [
            {
                "case": {"include_in_accuracy_summary": True},
                "pf_count_likelihood_diagnostics": diagnostic,
            }
        ],
        min_target=25.0,
    )
    runtime_summary = summary["targets"]["runtime_pf_forward"]
    expected_distance = sum(
        float(value) ** 2
        for value in runtime_target["normalized_residual_by_isotope"].values()
    )
    assert runtime_summary["num_records"] == 3
    assert runtime_summary["diagonal_squared_distance"][
        "pooled_squared_per_degree_of_freedom"
    ] == pytest.approx(expected_distance / 3.0)


def test_validation_calibration_defaults_match_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration helper defaults should match validation CLI defaults."""
    monkeypatch.setattr(sys, "argv", ["validate_geant4_spectrum_decomposition.py"])

    args = VALIDATION_SCRIPT.parse_args()

    assert args.calibration_min_pair_points == (
        VALIDATION_SCRIPT.DEFAULT_CALIBRATION_MIN_PAIR_POINTS
    )
    assert args.calibration_pair_shrinkage_count == pytest.approx(
        VALIDATION_SCRIPT.DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT
    )
    assert args.calibration_holdout_fraction == pytest.approx(
        VALIDATION_SCRIPT.DEFAULT_CALIBRATION_HOLDOUT_FRACTION
    )
    assert args.calibration_holdout_seed == (
        VALIDATION_SCRIPT.DEFAULT_CALIBRATION_HOLDOUT_SEED
    )
    for name in (
        "summarize_response_poisson_calibration",
        "summarize_pf_transport_calibration",
        "_fit_pf_transport_response_model",
    ):
        signature = inspect.signature(getattr(VALIDATION_SCRIPT, name))
        assert signature.parameters["min_pair_fit_points"].default == (
            VALIDATION_SCRIPT.DEFAULT_CALIBRATION_MIN_PAIR_POINTS
        )
        assert signature.parameters["pair_shrinkage_count"].default == (
            VALIDATION_SCRIPT.DEFAULT_CALIBRATION_PAIR_SHRINKAGE_COUNT
        )
    for name in (
        "summarize_response_poisson_calibration",
        "summarize_pf_transport_calibration",
    ):
        signature = inspect.signature(getattr(VALIDATION_SCRIPT, name))
        assert signature.parameters["holdout_fraction"].default == (
            VALIDATION_SCRIPT.DEFAULT_CALIBRATION_HOLDOUT_FRACTION
        )
        assert signature.parameters["holdout_seed"].default == (
            VALIDATION_SCRIPT.DEFAULT_CALIBRATION_HOLDOUT_SEED
        )


def test_validation_calibration_split_keeps_scenarios_disjoint() -> None:
    """Calibration holdout should keep all shield pairs for a scenario together."""
    records = []
    for scenario_idx in range(6):
        for pair_id in range(4):
            records.append(
                {
                    "isotope": "Cs-137",
                    "scenario": f"multi_iso_{scenario_idx:04d}",
                    "shield_pair_id": pair_id,
                    "theory_counts": 100.0,
                    "net_counts": 100.0,
                    "weight": 1.0,
                }
            )

    train, holdout = VALIDATION_SCRIPT._split_calibration_records(
        records,
        holdout_fraction=0.5,
        seed=20260607,
    )

    train_scenarios = {record["scenario"] for record in train}
    holdout_scenarios = {record["scenario"] for record in holdout}
    assert train_scenarios
    assert holdout_scenarios
    assert train_scenarios.isdisjoint(holdout_scenarios)
    assert len(train) + len(holdout) == len(records)


def test_response_poisson_calibration_records_include_scenario_key() -> None:
    """Response-Poisson calibration records should carry scenario split keys."""
    result = {
        "case": {
            "name": "multi_iso_0003_pair09_fe1_pb1",
            "include_in_accuracy_summary": True,
            "fe_index": 1,
            "pb_index": 1,
        },
        "per_isotope": {
            "Cs-137": {
                "target_pf_counts": 1000.0,
                "method_counts": {"response_poisson": 1100.0},
                "response_poisson_method_name": "response_poisson",
                "response_poisson_variance": 25.0,
            }
        },
    }

    records = VALIDATION_SCRIPT._response_poisson_calibration_records(
        [result],
        min_target=100.0,
    )

    assert len(records) == 1
    assert records[0]["case"] == "multi_iso_0003_pair09_fe1_pb1"
    assert records[0]["scenario"] == "multi_iso_0003"


def test_clone_kernel_with_line_mu_preserves_obstacle_mu() -> None:
    """Geant4-mu validation clones should keep the runtime obstacle coefficients."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.02},
        use_gpu=False,
    )
    replacement = VALIDATION_SCRIPT.clone_kernel_with_line_mu(
        kernel,
        {"Cs-137": ({"weight": 1.0, "fe": 0.0, "pb": 0.0},)},
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    assert replacement.obstacle_mu_by_isotope == {"Cs-137": 0.02}
    assert replacement.kernel_value_pair("Cs-137", detector, source, 0, 0) == (
        pytest.approx(
            free_kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
            * np.exp(-2.0),
            rel=1.0e-12,
        )
    )


def test_transport_response_model_matches_runtime_kernel_terms() -> None:
    """Validation transport-response math should match a model-loaded kernel."""
    isotope = "Cs-137"
    detector = np.zeros(3, dtype=float)
    sources = (
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([1.0, 0.5, 0.25], dtype=float),
    )
    strengths = (120.0, 35.0)
    live_time_s = 2.5
    fe_index = 7
    pb_index = 0
    pair_id = fe_index * 8 + pb_index
    line_mu = {isotope: ({"weight": 1.0, "fe": 0.2, "pb": 0.0},)}
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
    )
    payload = {
        "scale": 1.0,
        "scale_by_pair": {str(pair_id): 1.15},
        "tau_coefficients": {
            "shield": 2.0,
            "fe": 0.25,
        },
        "min_log_scale": -5.0,
        "max_log_scale": 5.0,
    }
    transport_model = {
        "enabled": True,
        "by_isotope": {isotope: payload},
    }
    base_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    model_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    source_terms = []
    for source, strength in zip(sources, strengths):
        for term in base_kernel.transport_response_terms_pair(
            isotope,
            detector,
            source,
            fe_index,
            pb_index,
        ):
            source_terms.append(
                {
                    "counts": live_time_s * strength * float(term["kernel"]),
                    "shield_tau_feature": float(term["shield_tau_feature"]),
                    "fe_tau_feature": float(term["fe_tau_feature"]),
                    "pb_tau_feature": float(term["pb_tau_feature"]),
                    "obstacle_tau_feature": float(term["obstacle_tau_feature"]),
                }
            )
    group = {
        "shield_pair_id": pair_id,
        "source_terms": source_terms,
    }
    validation_expected = VALIDATION_SCRIPT._transport_response_model_expected_counts(
        group,
        payload,
    )
    runtime_expected = model_kernel.expected_counts_pair(
        isotope,
        detector,
        np.vstack(sources),
        np.asarray(strengths, dtype=float),
        fe_index=fe_index,
        pb_index=pb_index,
        live_time_s=live_time_s,
    )

    assert validation_expected == pytest.approx(runtime_expected, rel=1.0e-12)


def test_transport_response_summary_uses_runtime_log_bounds_by_default() -> None:
    """Validation residual helpers should not add non-runtime log clamps."""
    group = {
        "shield_pair_id": 0,
        "source_terms": [
            {
                "counts": 10.0,
                "shield_tau_feature": 3.0,
                "fe_tau_feature": 0.0,
                "pb_tau_feature": 0.0,
                "obstacle_tau_feature": 0.0,
            }
        ],
    }
    payload = {
        "scale": 1.0,
        "tau_coefficients": {"shield": 1.0},
    }

    expected = VALIDATION_SCRIPT._transport_response_model_expected_counts(
        group,
        payload,
    )

    assert expected == pytest.approx(10.0 * np.exp(3.0), rel=1.0e-12)


def test_transport_response_summary_uses_feature_caps() -> None:
    """Validation residual helpers should apply transport-response feature caps."""
    group = {
        "transport_truth_counts": 10.0 * np.exp(1.0),
        "source_terms": [
            {
                "counts": 10.0,
                "shield_tau_feature": 2.0,
                "fe_tau_feature": 0.0,
                "pb_tau_feature": 0.0,
                "obstacle_tau_feature": 0.0,
                "distance_feature": 10.0,
                "distance_shield_feature": 20.0,
            }
        ],
    }
    payload = {
        "scale": 1.0,
        "tau_coefficients": {"shield_squared": 1.0, "distance_shield": 1.0},
        "tau_feature_caps": {"shield": 0.5, "distance_shield": 0.75},
        "min_log_scale": -10.0,
        "max_log_scale": 10.0,
    }

    expected = VALIDATION_SCRIPT._transport_response_model_expected_counts(
        group,
        payload,
    )

    assert expected == pytest.approx(10.0 * np.exp(1.0), rel=1.0e-12)


def test_transport_response_summary_uses_distance_material_caps() -> None:
    """Validation residual helpers should cap distance-material features."""
    group = {
        "transport_truth_counts": 10.0 * np.exp(1.0),
        "source_terms": [
            {
                "counts": 10.0,
                "shield_tau_feature": 0.0,
                "fe_tau_feature": 1.0,
                "pb_tau_feature": 2.0,
                "obstacle_tau_feature": 3.0,
                "distance_feature": 10.0,
            }
        ],
    }
    payload = {
        "scale": 1.0,
        "tau_coefficients": {
            "distance_fe": 1.0,
            "distance_pb": 1.0,
            "distance_obstacle": 1.0,
        },
        "tau_feature_caps": {
            "distance_fe": 0.25,
            "distance_pb": 0.25,
            "distance_obstacle": 0.5,
        },
        "min_log_scale": -10.0,
        "max_log_scale": 10.0,
    }

    expected = VALIDATION_SCRIPT._transport_response_model_expected_counts(
        group,
        payload,
    )

    assert expected == pytest.approx(10.0 * np.exp(1.0), rel=1.0e-12)


def test_transport_response_fit_features_use_runtime_caps() -> None:
    """Transport-response fitting should see the same capped features as runtime."""
    term = {
        "shield_tau_feature": 5.0,
        "fe_tau_feature": 6.0,
        "pb_tau_feature": 7.0,
        "obstacle_tau_feature": 2.0,
        "distance_feature": 3.0,
        "distance_shield_feature": 20.0,
    }

    features = VALIDATION_SCRIPT._transport_response_feature_vector(term)

    assert features[1] == pytest.approx(3.5)
    assert features[3] == pytest.approx(3.5 * 3.5)
    assert features[6] == pytest.approx(3.5)
    assert features[7] == pytest.approx(3.5)
    assert features[10] == pytest.approx(3.5 * 3.5)
    assert features[13] == pytest.approx(3.0)
    assert features[14] == pytest.approx(8.0)
    assert features[15] == pytest.approx(8.0)
    assert features[16] == pytest.approx(8.0)
    assert features[17] == pytest.approx(6.0)


def test_expected_pf_count_diagnostics_use_base_terms_for_transport_fit() -> None:
    """Sidecar-enabled diagnostics should fit transport response from base terms."""
    isotope = "Cs-137"
    payload = {
        "scale": 2.0,
        "tau_coefficients": {},
        "min_log_scale": -5.0,
        "max_log_scale": 5.0,
    }
    runtime_config = {
        "pf_line_resolved_shield_attenuation": False,
        "pf_transport_response_model": {
            "enabled": True,
            "by_isotope": {isotope: payload},
        },
        "measurement_scale_by_isotope": {isotope: 3.0},
    }
    case = VALIDATION_SCRIPT.ValidationCase(
        name="sidecar_diagnostic_case",
        description="sidecar diagnostic case",
        detector_pose_xyz=(0.0, 0.0, 0.0),
        sources=(
            VALIDATION_SCRIPT.ValidationSource(
                isotope=isotope,
                position_xyz=(1.0, 1.0, 1.0),
                intensity_cps_1m=100.0,
            ),
        ),
        fe_index=0,
        pb_index=0,
        dwell_time_s=1.0,
    )

    rows = VALIDATION_SCRIPT.expected_pf_count_diagnostics(
        case,
        runtime_config,
        {},
    )
    terms = rows[0]["transport_response_terms"]
    source_terms = VALIDATION_SCRIPT._pf_transport_source_terms(rows)
    expected = VALIDATION_SCRIPT._transport_response_model_expected_counts(
        {"shield_pair_id": 0, "source_terms": source_terms},
        payload,
    )
    base_counts = sum(float(term["base_counts"]) for term in terms)
    adjusted_counts = sum(float(term["adjusted_counts"]) for term in terms)
    scaled_base_counts = sum(float(term["scaled_base_counts"]) for term in terms)
    scaled_adjusted_counts = sum(
        float(term["scaled_adjusted_counts"]) for term in terms
    )

    assert terms
    for term in terms:
        assert term["counts"] == pytest.approx(term["base_counts"], rel=1.0e-12)
        assert term["scaled_counts"] == pytest.approx(
            term["scaled_base_counts"],
            rel=1.0e-12,
        )
        assert term["adjusted_counts"] == pytest.approx(
            2.0 * term["base_counts"],
            rel=1.0e-12,
        )
    assert adjusted_counts == pytest.approx(2.0 * base_counts, rel=1.0e-12)
    assert rows[0]["full_target_counts"] == pytest.approx(
        adjusted_counts,
        rel=1.0e-12,
    )
    assert sum(term["counts"] for term in source_terms) == pytest.approx(
        scaled_base_counts,
        rel=1.0e-12,
    )
    assert expected == pytest.approx(
        scaled_adjusted_counts,
        rel=1.0e-12,
    )


def test_pf_transport_source_terms_reconstruct_distance_feature() -> None:
    """Transport source terms should recover distance from saved source rows."""
    diagnostics = [
        {
            "position_xyz": [3.0, 4.0, 0.0],
            "measurement_source_scale": 2.0,
            "transport_response_terms": [
                {
                    "counts": 5.0,
                    "shield_tau_feature": 0.5,
                    "fe_tau_feature": 0.5,
                    "pb_tau_feature": 0.0,
                    "obstacle_tau_feature": 0.0,
                }
            ],
        }
    ]

    terms = VALIDATION_SCRIPT._pf_transport_source_terms(
        diagnostics,
        detector_xyz=(0.0, 0.0, 0.0),
    )

    assert terms[0]["counts"] == pytest.approx(10.0)
    assert terms[0]["distance_feature"] == pytest.approx(5.0)
    assert terms[0]["distance_shield_feature"] == pytest.approx(2.5)


def test_pf_transport_feature_diagnostics_reconstruct_distance_feature() -> None:
    """CSV transport diagnostics should recover distance from saved source rows."""
    diagnostics = [
        {
            "position_xyz": [3.0, 4.0, 0.0],
            "measurement_source_scale": 2.0,
            "transport_response_terms": [
                {
                    "counts": 5.0,
                    "shield_tau_feature": 0.5,
                    "fe_tau_feature": 0.5,
                    "pb_tau_feature": 0.0,
                    "obstacle_tau_feature": 0.0,
                }
            ],
        }
    ]

    features = VALIDATION_SCRIPT._weighted_pf_transport_feature_diagnostics(
        diagnostics,
        detector_xyz=(0.0, 0.0, 0.0),
    )

    assert features["distance_feature"] == pytest.approx(5.0)
    assert features["distance_shield_feature"] == pytest.approx(2.5)
