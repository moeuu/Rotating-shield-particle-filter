"""Tests for response-Poisson guard replay utilities."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_replay_script() -> object:
    """Load the replay script as a module for helper tests."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "replay_response_poisson_guard.py"
    spec = spec_from_file_location("replay_response_poisson_guard", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY_SCRIPT = _load_replay_script()


def test_count_semantics_quality_gates_use_agreed_thresholds() -> None:
    """Quality gates should use agreed mean/p95 targets and ignore max tails."""
    gates = REPLAY_SCRIPT._count_semantics_quality_gates(
        response_summary={"mean": 0.039, "p95": 0.119, "max": 0.90},
        target_summary={"mean": 0.031, "p95": 0.090, "max": 0.20},
        response_target_summary={"mean": 0.049, "p95": 0.149, "max": 2.0},
    )

    response_gate = gates["current_response_poisson_vs_truth"]
    target_gate = gates["target_pf_counts_vs_truth"]
    response_target_gate = gates["current_response_vs_target_pf_counts"]
    assert response_gate["mean_threshold"] == pytest.approx(0.04)
    assert response_gate["p95_threshold"] == pytest.approx(0.12)
    assert response_gate["passed"] is True
    assert response_gate["max_tail_report_only"] == pytest.approx(0.90)
    assert target_gate["mean_threshold"] == pytest.approx(0.03)
    assert target_gate["p95_threshold"] == pytest.approx(0.10)
    assert target_gate["passed"] is False
    assert response_target_gate["mean_threshold"] == pytest.approx(0.05)
    assert response_target_gate["p95_threshold"] == pytest.approx(0.15)
    assert response_target_gate["passed"] is True
    assert response_target_gate["max_tail_report_only"] == pytest.approx(2.0)


def test_replay_applies_runtime_response_poisson_variance_ceiling() -> None:
    """Replay PF z-score diagnostics should match runtime variance capping."""
    settings = REPLAY_SCRIPT._runtime_variance_ceiling_settings(
        {
            "response_poisson_count_variance_ceiling_enable": True,
            "response_poisson_count_variance_max_rel_sigma": 0.15,
            "response_poisson_count_variance_max_abs_sigma": 40.0,
        }
    )

    assert REPLAY_SCRIPT._cap_response_poisson_variance(
        1000.0,
        1.0e9,
        settings,
    ) == pytest.approx((0.15 * 1000.0) ** 2)
    assert REPLAY_SCRIPT._cap_response_poisson_variance(
        5.0,
        1.0e9,
        settings,
    ) == pytest.approx(40.0**2)


def test_replay_preserves_guard_variance_floor_across_ceiling() -> None:
    """Replay z-score diagnostics should preserve runtime guard uncertainty."""
    settings = REPLAY_SCRIPT._runtime_variance_ceiling_settings(
        {
            "response_poisson_count_variance_ceiling_enable": True,
            "response_poisson_count_variance_max_rel_sigma": 0.15,
            "response_poisson_count_variance_max_abs_sigma": 40.0,
            "response_poisson_count_variance_preserve_guard_floors": True,
        }
    )

    assert REPLAY_SCRIPT._cap_response_poisson_variance(
        100000.0,
        1.0e8,
        settings,
        diagnostic_payload={
            "reason": "combined_crosstalk_photopeak_log_blend",
            "poisson_count": 100000.0,
            "photopeak_count": 1000.0,
            "guarded_variance": 1.0e8,
        },
    ) == pytest.approx(1.0e8)


def test_replay_preserves_low_snr_photo_count_disagreement() -> None:
    """Replay diagnostics should match low-SNR runtime uncertainty semantics."""
    settings = REPLAY_SCRIPT._runtime_variance_ceiling_settings(
        {
            "response_poisson_count_variance_ceiling_enable": True,
            "response_poisson_count_variance_max_rel_sigma": 0.15,
            "response_poisson_count_variance_max_abs_sigma": 40.0,
            "response_poisson_count_variance_preserve_guard_floors": True,
        }
    )

    assert REPLAY_SCRIPT._cap_response_poisson_variance(
        1100.0,
        3.0e7,
        settings,
        diagnostic_payload={
            "reason": "missing_expected_photopeaks",
            "poisson_count": 6500.0,
            "photo_count": 1100.0,
        },
    ) == pytest.approx(3.0e7)


def test_replay_preserves_low_snr_threshold_variance() -> None:
    """Replay should not cap zero-count low-SNR threshold variance."""
    settings = REPLAY_SCRIPT._runtime_variance_ceiling_settings(
        {
            "response_poisson_count_variance_ceiling_enable": True,
            "response_poisson_count_variance_max_rel_sigma": 0.15,
            "response_poisson_count_variance_max_abs_sigma": 40.0,
            "response_poisson_count_variance_preserve_guard_floors": True,
        }
    )

    assert REPLAY_SCRIPT._cap_response_poisson_variance(
        0.0,
        50000.0,
        settings,
        diagnostic_payload={
            "reason": "zero_poisson_photopeak_fused",
            "poisson_count": 0.0,
            "photo_count": 0.0,
            "suppressed": False,
        },
    ) == pytest.approx(50000.0)


def test_replay_records_projects_current_transport_target(tmp_path: Path) -> None:
    """Replay diagnostics should use current transport-response targets."""
    scaled_base_counts = 200.0
    truth = scaled_base_counts * float(np.exp(0.25))
    records_path = tmp_path / "records.csv"
    records_path.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "case",
                        "isotope",
                        "method",
                        "transport_truth_counts",
                        "estimated_counts",
                        "target_pf_counts",
                        "estimated_variance",
                        "response_poisson_raw_coefficient",
                        "response_poisson_photopeak_count",
                        "response_poisson_photopeak_variance",
                        "response_poisson_reduced_chi2",
                    ]
                ),
                ",".join(
                    [
                        "case_0000_pair00_fe0_pb0",
                        "Cs-137",
                        "response_poisson",
                        f"{truth:.12f}",
                        f"{truth:.12f}",
                        "100.0",
                        "1.0",
                        f"{truth:.12f}",
                        f"{truth:.12f}",
                        "1.0",
                        "1.0",
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "pf_transport_response_model": {
                    "enabled": True,
                    "by_isotope": {
                        "Cs-137": {
                            "scale": 1.0,
                            "tau_coefficients": {"shield_squared": 1.0},
                            "tau_feature_caps": {"shield": 0.5},
                            "min_log_scale": -10.0,
                            "max_log_scale": 10.0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "case": {
                        "name": "case_0000_pair00_fe0_pb0",
                        "include_in_accuracy_summary": True,
                        "fe_index": 0,
                        "pb_index": 0,
                    },
                    "per_isotope": {
                        "Cs-137": {
                            "target_pf_count_diagnostics": [
                                {
                                    "transport_response_terms": [
                                        {
                                            "counts": 100.0,
                                            "scaled_base_counts": scaled_base_counts,
                                            "shield_tau_feature": 2.0,
                                            "fe_tau_feature": 0.0,
                                            "pb_tau_feature": 0.0,
                                            "obstacle_tau_feature": 0.0,
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    replay = REPLAY_SCRIPT.replay_records(
        records_path=records_path,
        config_path=config_path,
        results_json_path=results_path,
        truth_min=1.0,
        checkpoints=[1],
        top=1,
    )

    assert replay["target_source"] == "current_runtime_transport_response_model"
    assert replay["target_pf_counts"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert replay["recorded_target_pf_counts"]["max"] > 0.2
    assert replay["projected_response_vs_target_pf_counts"]["max"] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert replay["recorded_response_vs_recorded_target_pf_counts"]["max"] > 0.2
    assert replay["worst_triad_rows"][0]["projected_response_target_mismatch"] == (
        pytest.approx(0.0, abs=1.0e-12)
    )
    assert replay["worst_projected_rows"][0]["target_count"] == pytest.approx(
        truth,
        rel=1.0e-12,
    )
    assert replay["worst_projected_rows"][0]["pf_likelihood_variance"] > 0.0
    assert replay["worst_projected_rows"][0]["projected_pf_z_vs_truth"] == (
        pytest.approx(0.0, abs=1.0e-12)
    )
    assert replay["worst_projected_rows"][0]["projected_pf_z_vs_target"] == (
        pytest.approx(0.0, abs=1.0e-12)
    )
    assert replay["worst_projected_rows"][0]["target_pf_z_vs_truth"] == (
        pytest.approx(0.0, abs=1.0e-12)
    )


def test_replay_transport_target_uses_runtime_log_bounds(tmp_path: Path) -> None:
    """Replay target projection should match runtime transport-response defaults."""
    scaled_base_counts = 200.0
    truth = scaled_base_counts * float(np.exp(2.5))
    records_path = tmp_path / "records.csv"
    records_path.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "case",
                        "isotope",
                        "method",
                        "transport_truth_counts",
                        "estimated_counts",
                        "target_pf_counts",
                        "estimated_variance",
                        "response_poisson_raw_coefficient",
                        "response_poisson_photopeak_count",
                        "response_poisson_photopeak_variance",
                        "response_poisson_reduced_chi2",
                    ]
                ),
                ",".join(
                    [
                        "case_0000_pair00_fe0_pb0",
                        "Cs-137",
                        "response_poisson",
                        f"{truth:.12f}",
                        f"{truth:.12f}",
                        "100.0",
                        "1.0",
                        f"{truth:.12f}",
                        f"{truth:.12f}",
                        "1.0",
                        "1.0",
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "pf_transport_response_model": {
                    "enabled": True,
                    "by_isotope": {
                        "Cs-137": {
                            "scale": 1.0,
                            "tau_coefficients": {"shield": 3.0, "distance": -0.1},
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "case": {
                        "name": "case_0000_pair00_fe0_pb0",
                        "include_in_accuracy_summary": True,
                        "detector_pose_xyz": [0.0, 0.0, 0.0],
                        "fe_index": 0,
                        "pb_index": 0,
                    },
                    "per_isotope": {
                        "Cs-137": {
                            "target_pf_count_diagnostics": [
                                {
                                    "position_xyz": [3.0, 4.0, 0.0],
                                    "transport_response_terms": [
                                        {
                                            "counts": scaled_base_counts,
                                            "shield_tau_feature": 1.0,
                                            "fe_tau_feature": 0.0,
                                            "pb_tau_feature": 0.0,
                                            "obstacle_tau_feature": 0.0,
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    replay = REPLAY_SCRIPT.replay_records(
        records_path=records_path,
        config_path=config_path,
        results_json_path=results_path,
        truth_min=1.0,
        checkpoints=[1],
        top=1,
    )

    assert replay["target_source"] == "current_runtime_transport_response_model"
    assert replay["target_pf_counts"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert replay["recorded_target_pf_counts"]["max"] > 0.9
    assert replay["worst_projected_rows"][0]["target_count"] == pytest.approx(
        truth,
        rel=1.0e-12,
    )


def test_recompute_spectrum_records_uses_saved_spectra(tmp_path: Path) -> None:
    """Spectrum recompute diagnostics should refit saved spectra with current config."""
    truth = 5000.0
    config_payload = {
        "response_poisson_line_resolved_fit": False,
        "response_poisson_crosstalk_count_guard_enable": False,
        "response_poisson_low_snr_photopeak_anchor": False,
    }
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    config = REPLAY_SCRIPT.spectrum_config_from_runtime_config(config_payload)
    decomposer = REPLAY_SCRIPT.SpectralDecomposer(config)
    isotope_index = decomposer.isotope_names.index("Cs-137")
    spectrum = decomposer._count_response_matrix()[:, isotope_index] * truth
    spectra_path = tmp_path / "spectra.npz"
    np.savez(spectra_path, case_0000_pair00_fe0_pb0=spectrum)
    records_path = tmp_path / "records.csv"
    records_path.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "case",
                        "isotope",
                        "method",
                        "transport_truth_counts",
                        "estimated_counts",
                        "target_pf_counts",
                        "estimated_variance",
                        "dwell_time_s",
                    ]
                ),
                ",".join(
                    [
                        "case_0000_pair00_fe0_pb0",
                        "Cs-137",
                        "response_poisson",
                        f"{truth:.12f}",
                        "2500.0",
                        f"{truth:.12f}",
                        "1.0",
                        "30.0",
                    ]
                ),
                ",".join(
                    [
                        "case_0000_pair00_fe0_pb0",
                        "Co-60",
                        "response_poisson",
                        "0.0",
                        "1000.0",
                        "0.0",
                        "1.0",
                        "30.0",
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )

    recomputed = REPLAY_SCRIPT.recompute_spectrum_records(
        records_path=records_path,
        config_path=config_path,
        spectra_path=spectra_path,
        truth_min=1.0,
        checkpoints=[1],
        top=1,
    )

    assert recomputed["recorded_response_poisson"]["max"] > 0.4
    assert recomputed["current_response_poisson"]["max"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert recomputed["current_raw_response_poisson"]["max"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert recomputed["current_photopeak_nnls"]["n"] == pytest.approx(0.0)
    assert recomputed["current_best_count_source_counts"] == {"final": 1}
    assert recomputed["current_absent_by_truth_counts"]["n"] == pytest.approx(1.0)
    assert recomputed["current_absent_by_truth_counts"]["max"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert recomputed["current_absent_by_target_counts"]["n"] == pytest.approx(1.0)
    assert recomputed["worst_absent_by_truth_rows"][0]["isotope"] == "Co-60"
    assert recomputed["worst_absent_by_target_rows"][0]["isotope"] == "Co-60"
    assert recomputed["current_by_isotope"][0]["isotope"] == "Cs-137"
    assert recomputed["current_by_isotope"][0]["response_vs_truth"]["max"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["current_by_shield_pair"][0]["pair_id"] == 0
    assert recomputed["current_by_shield_pair_target_rank"][0]["pair_id"] == 0
    assert recomputed["current_by_shield_pair_isotope"][0]["pair_id"] == 0
    assert recomputed["current_by_shield_pair_isotope"][0]["isotope"] == "Cs-137"
    assert recomputed["current_by_shield_pair_isotope_target_rank"][0]["pair_id"] == 0
    assert recomputed["current_by_shield_pair_isotope_target_rank"][0]["isotope"] == (
        "Cs-137"
    )
    assert recomputed["current_by_guard_reason"][0]["guard_reason"] == "none"
    assert recomputed["current_by_guard_reason"][0]["response_pf_z_vs_truth"]["max"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["current_by_guard_reason"][0]["response_pf_z_vs_target"]["max"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["current_by_guard_reason"][0]["target_pf_z_vs_truth"]["max"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["current_by_guard_reason"][0]["response_poisson_variance"]["n"] == (
        pytest.approx(1.0)
    )
    assert recomputed["current_by_guard_reason"][0]["pf_likelihood_variance"]["n"] == (
        pytest.approx(1.0)
    )
    assert recomputed["current_by_method"][0]["method"] == (
        "response_poisson_source_equivalent"
    )
    assert recomputed["current_by_best_count_source"][0]["best_count_source"] == (
        "final"
    )
    assert recomputed["current_by_line_bic_bucket"][0]["line_bic_bucket"] == (
        "line_disabled"
    )
    assert recomputed["current_response_vs_target_pf_counts"]["max"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert recomputed["recorded_response_vs_recorded_target_pf_counts"]["max"] > 0.4
    assert recomputed["worst_triad_rows"][0]["current_triad_spread"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert recomputed["worst_current_rows"][0]["current_count"] == pytest.approx(
        truth,
        rel=1.0e-6,
    )
    assert recomputed["worst_target_rows"][0]["target_count"] == pytest.approx(
        truth,
        rel=1.0e-6,
    )
    assert recomputed["worst_current_rows"][0]["current_raw_count"] == pytest.approx(
        truth,
        rel=1.0e-6,
    )
    assert recomputed["worst_current_rows"][0]["best_count_source"] == "final"
    assert recomputed["worst_current_rows"][0]["line_bic_bucket"] == "line_disabled"
    assert recomputed["worst_current_rows"][0]["response_poisson_variance"] > 0.0
    assert recomputed["worst_current_rows"][0]["pf_likelihood_variance"] > 0.0
    assert recomputed["worst_current_rows"][0]["response_pf_z_vs_truth"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["worst_current_rows"][0]["response_pf_z_vs_target"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
    assert recomputed["worst_current_rows"][0]["target_pf_z_vs_truth"] == (
        pytest.approx(0.0, abs=1.0e-6)
    )
