"""Guard the standard runtime against removed simulation shortcuts."""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.profiles import enforce_pure_runtime_settings
from sim.runtime import load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARD_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)


def _runtime_config_paths() -> tuple[Path, ...]:
    """Return every checked-in simulation runtime JSON configuration."""
    config_root = PROJECT_ROOT / "configs"
    roots = (
        config_root / "geant4",
        config_root / "python",
        config_root / "isaacsim",
    )
    return tuple(
        path
        for root in roots
        for path in sorted(root.rglob("*.json"))
        if "calibration" not in path.parts
    )


def test_standard_runtime_uses_unit_weight_external_geant4() -> None:
    """The standard full simulation must retain its native fidelity contract."""
    payload = load_runtime_config(STANDARD_CONFIG)

    assert payload.get("engine_mode", "external") == "external"
    assert payload["persistent_process"] is True
    assert payload["source_rate_model"] == "detector_cps_1m"
    assert payload["primary_sampling_fraction"] == pytest.approx(1.0)
    assert payload.get("accelerated_weighted_transport_enable", False) is False
    assert int(payload.get("target_sampled_primaries", 0)) == 0
    assert payload["sample_detector_response"] is True
    assert payload["secondary_transport_mode"] == "full_transport"
    assert payload.get("theory_tvl_attenuation", False) is False
    assert str(payload.get("physics_profile", "balanced")).lower() != "theory_tvl"


def test_standard_runtime_has_one_joint_spectrum_observation_contract() -> None:
    """Removed count and duplicate-likelihood keys must not re-enter runtime."""
    payload = load_runtime_config(STANDARD_CONFIG)
    retired_exact = {
        "calibration_count_method",
        "pf_count_likelihood",
        "pf_shield_contrast_likelihood",
        "pf_shield_view_ratio_likelihood",
        "spectrum_count_method",
    }
    retired_prefixes = (
        "contrast_",
        "count_likelihood_",
        "response_poisson_",
        "shield_contrast_",
        "shield_view_ratio_",
        "view_ratio_",
    )

    assert retired_exact.isdisjoint(payload)
    assert not any(str(key).startswith(retired_prefixes) for key in payload)


@pytest.mark.parametrize("config_path", _runtime_config_paths())
def test_pure_runtime_strength_prior_matches_random_truth_contract(
    config_path: Path,
) -> None:
    """Every pure runtime must use the predeclared Uniform[300k, 2M] support."""
    payload = load_runtime_config(config_path)
    if payload.get("pure_pf_schema_version") != 1:
        return

    prior_minimum = payload["pf_strength_prior_min_cps_1m"]
    prior_maximum = payload["pf_strength_prior_max_cps_1m"]
    assert (
        isinstance(prior_minimum, (int, float))
        and not isinstance(prior_minimum, bool)
    )
    assert (
        isinstance(prior_maximum, (int, float))
        and not isinstance(prior_maximum, bool)
    )
    assert float(prior_minimum) == pytest.approx(300_000.0)
    assert float(prior_maximum) == pytest.approx(2_000_000.0)
    assert 0.0 < float(prior_minimum) < float(prior_maximum)

    truth_minimum = payload.get("random_source_intensity_min_cps_1m")
    truth_maximum = payload.get("random_source_intensity_max_cps_1m")
    if truth_minimum is None and truth_maximum is None:
        return
    assert (
        isinstance(truth_minimum, (int, float))
        and not isinstance(truth_minimum, bool)
    )
    assert (
        isinstance(truth_maximum, (int, float))
        and not isinstance(truth_maximum, bool)
    )
    assert float(truth_minimum) == pytest.approx(float(prior_minimum))
    assert float(truth_maximum) == pytest.approx(float(prior_maximum))


def test_removed_count_and_rescue_paths_stay_deleted() -> None:
    """Legacy estimator, calibration, and shortcut paths must remain absent."""
    removed_paths = (
        (
            "configs/geant4/"
            "accelerated_weighted_external_no_isaac_32threads.json"
        ),
        (
            "configs/geant4/calibration/"
            "pf_transport_response_model_combined_random64_uniform_"
            "v3_distance_material_20260609.json"
        ),
        (
            "configs/geant4/calibration/"
            "pf_transport_response_model_dominanceguard_transport_"
            "20260608.json"
        ),
        "scripts/analyze_pf_run_metrics.py",
        "scripts/analyze_weighted_transport_acceptance.py",
        "scripts/calibrate_geant4_net_response.py",
        "scripts/replay_photopeak_channel_counts.py",
        "scripts/replay_response_poisson_guard.py",
        "scripts/validate_eu154_count_model.py",
        "scripts/validate_geant4_spectrum_decomposition.py",
        "src/counts/__init__.py",
        "src/counts/isotope_sequence.py",
        "src/measurement/surface_patches.py",
        "src/pf/forward_response_conformance.py",
        "src/pf/likelihood.py",
        "src/pf/measurement.py",
        "src/pf/reporting.py",
        "src/pf/runtime_route.py",
        "src/planning/pose_selection.py",
        "src/planning/remaining_measurements.py",
        "src/planning/shield_rotation.py",
        "src/spectrum/activity_estimation.py",
        "src/spectrum/baseline.py",
        "src/spectrum/dead_time.py",
        "src/spectrum/decomposition.py",
        "src/spectrum/net_response.py",
        "src/spectrum/nnls.py",
        "src/spectrum/peak_detection.py",
        "src/spectrum/pipeline.py",
        "src/spectrum/response_truth_calibration.py",
        "src/spectrum/runtime_config.py",
        "src/spectrum/runtime_counts.py",
        "src/spectrum/smoothing.py",
        "src/spectrum/tuning.py",
    )

    assert not [
        relative_path
        for relative_path in removed_paths
        if (PROJECT_ROOT / relative_path).exists()
    ]


@pytest.mark.parametrize(
    "retired_key",
    (
        "adaptive_cardinality_min_bic_margin",
        "batch_fit_enable",
        "maximum_likelihood_enable",
        "mle_enable",
        "report_mle_enable",
        "report_refit_enable",
        "strength_refit_enable",
        "surface_map_enable",
    ),
)
def test_removed_estimation_switches_fail_closed(retired_key: str) -> None:
    """Deleted estimator switches must not survive as silently ignored keys."""
    runtime_config = {
        "pure_pf_schema_version": 1,
        "estimator_profile": "pf_strict",
        retired_key: False,
    }

    with pytest.raises(
        ValueError,
        match="Retired particle-filter settings",
    ):
        enforce_pure_runtime_settings(runtime_config)
