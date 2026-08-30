"""Explicit synthetic detector-Green fixtures for PF contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spectrum.detector_green_construction import (
    DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256,
    build_detector_green_operator,
)
from spectrum.detector_green_operator import DetectorGreenOperator
from spectrum.detector_green_validation import (
    DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256,
    DETECTOR_GREEN_VALIDATION_ID,
    DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT,
    DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
    detector_green_holdout_energies_keV,
)


def synthetic_detector_green_operator() -> DetectorGreenOperator:
    """Return a provenance-complete isotope-free unit-test operator."""
    nodes = np.asarray((0.0, 400.0, 900.0, 1700.0), dtype=np.float64)
    edges = np.sqrt(np.linspace(0.0, 1.0, 9, dtype=np.float64))
    raw = np.zeros((4, 8, 851), dtype=np.float64)
    for node_index, energy in enumerate(nodes[1:], start=1):
        raw_bin = int(energy // 2.0)
        phase_index = np.arange(8, dtype=np.float64)
        raw[node_index, :, raw_bin] = 7_000.0 - 375.0 * phase_index
        raw[node_index, :, max(raw_bin // 3, 1)] = 1_000.0 + 375.0 * phase_index
    construction = {
        "method": "native_geant4_monoenergetic_full_detector",
        "raw_corpus_sha256": "1" * 64,
        "native_executable_sha256": "2" * 64,
        "native_execution_environment_sha256": "3" * 64,
        "detector_implementation_bundle_sha256": "4" * 64,
        "detector_model_sha256": "5" * 64,
        "geant4_physics_contract_sha256": "6" * 64,
        "energy_resolution_contract_sha256": (
            DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256
        ),
        "construction_seed": 991_337,
        "histories_per_energy": 100_000,
        "energy_node_design": (
            "catalog_independent_deterministic_continuous_domain_v1"
        ),
        "phase_strata": 8,
        "detector_target_radius_m": 0.0395,
        "completed": True,
    }
    provisional = build_detector_green_operator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=edges,
        raw_deposit_histograms_ncb=raw,
        sampled_histories_nc=np.full((4, 8), 10_000.0),
        construction=construction,
    )
    return DetectorGreenOperator(
        energy_nodes_keV=provisional.energy_nodes_keV,
        impact_parameter_edges_fraction=(provisional.impact_parameter_edges_fraction),
        conditional_response_ncb=provisional.conditional_response_ncb,
        effective_histories_nc=provisional.effective_histories_nc,
        pulse_detection_probability_nc=(provisional.pulse_detection_probability_nc),
        output_energy_min_keV=provisional.output_energy_min_keV,
        output_bin_width_keV=provisional.output_bin_width_keV,
        construction=construction,
        binary_sha256="a" * 64,
        contract_hash_sha256=provisional.contract_hash_sha256,
    )


def write_synthetic_detector_green_artifact(directory: Path) -> Path:
    """Write and return one authenticated synthetic operator manifest."""
    operator = synthetic_detector_green_operator()
    publishable = DetectorGreenOperator(
        energy_nodes_keV=operator.energy_nodes_keV,
        impact_parameter_edges_fraction=(operator.impact_parameter_edges_fraction),
        conditional_response_ncb=operator.conditional_response_ncb,
        effective_histories_nc=operator.effective_histories_nc,
        pulse_detection_probability_nc=(operator.pulse_detection_probability_nc),
        output_energy_min_keV=operator.output_energy_min_keV,
        output_bin_width_keV=operator.output_bin_width_keV,
        construction=operator.construction,
    )
    return publishable.write_artifact(directory)


def synthetic_detector_green_validation_manifest(
    operator: DetectorGreenOperator,
    *,
    runtime_config_sha256: str = "7" * 64,
    native_executable_sha256: str = "2" * 64,
    native_execution_environment_sha256: str = "3" * 64,
    detector_implementation_bundle_sha256: str = "4" * 64,
) -> dict[str, object]:
    """Return explicit passing unit evidence for one synthetic operator."""
    design_seed = 1_337_991
    transport_seed = 1_337_992
    energies = detector_green_holdout_energies_keV(
        design_seed,
        operator=operator,
    )
    phase_detection = operator.phase_detection_probability_for_axis(energies)
    phase_response, _ = operator.phase_response_for_axis(energies)
    output_axis = np.arange(operator.output_bin_count, dtype=np.float64) * (
        operator.output_bin_width_keV
    )
    conditional_mean = np.einsum(
        "cbe,b->ce",
        phase_response,
        output_axis,
        optimize=True,
    )
    cells: list[dict[str, object]] = []
    for energy_index, energy in enumerate(energies):
        for phase_index in range(phase_response.shape[0]):
            mean = float(conditional_mean[phase_index, energy_index])
            detection = float(phase_detection[phase_index, energy_index])
            cells.append(
                {
                    "energy_index": int(energy_index),
                    "impact_bin_index": int(phase_index),
                    "energy_keV": float(energy),
                    "grouped_total_variation": 0.0,
                    "cdf_distance": 0.0,
                    "conditional_mean_scaled_error": 0.0,
                    "detection_probability_absolute_error": 0.0,
                    "observed_conditional_mean_keV": mean,
                    "candidate_conditional_mean_keV": mean,
                    "observed_detection_probability": detection,
                    "candidate_detection_probability": detection,
                    "passed": True,
                }
            )
    metrics = {
        name: {
            "value": 0.0,
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": True,
        }
        for name, (comparison, threshold) in (
            DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT.items()
        )
    }
    return {
        "schema_version": DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
        "validation": DETECTOR_GREEN_VALIDATION_ID,
        "validation_contract_sha256": (DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256),
        "operator_contract_sha256": operator.contract_hash_sha256,
        "operator_binary_sha256": operator.binary_sha256,
        "raw_corpus_sha256": "8" * 64,
        "design_seed": design_seed,
        "transport_seed": transport_seed,
        "holdout_energies_keV": energies.tolist(),
        "impact_parameter_edges_fraction": (
            operator.impact_parameter_edges_fraction.tolist()
        ),
        "histories_per_energy": 100_000,
        "runtime_config_sha256": runtime_config_sha256,
        "native_executable_sha256": native_executable_sha256,
        "native_execution_environment_sha256": (native_execution_environment_sha256),
        "detector_implementation_bundle_sha256": (
            detector_implementation_bundle_sha256
        ),
        "cell_results": cells,
        "metrics": metrics,
        "all_passed": True,
    }


__all__ = [
    "synthetic_detector_green_operator",
    "synthetic_detector_green_validation_manifest",
    "write_synthetic_detector_green_artifact",
]
