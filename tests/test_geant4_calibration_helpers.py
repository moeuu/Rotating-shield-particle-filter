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
    spec = spec_from_file_location("validate_geant4_spectrum_decomposition", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATION_SCRIPT = _load_calibration_script()
VALIDATION_SCRIPT = _load_validation_script()


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


def test_validation_calibration_defaults_match_cli(monkeypatch: pytest.MonkeyPatch) -> None:
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
