"""Local schema-v3 full-spectrum fixtures for pure-PF contract tests."""

from __future__ import annotations

import copy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from measurement.source_boundary import surface_emission_policy_sha256
from measurement.shielding import SHIELD_POSE_CONTRACT_SHA256
from pf.full_spectrum import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from pf.provenance import canonical_json_bytes
from runtime.measurement_log import (
    MeasurementLogRecord,
    build_forward_model_manifest,
    write_measurement_log,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)
from spectrum.physics_contracts import (
    OBSTACLE_MATERIAL_CONTRACT_SHA256,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256,
)
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID,
    AdditiveNoncollidedTransportResponse,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    GeometryConditionedSpectralModel,
    MARK_CONCENTRATION_GRID,
    RATE_SCALE_HALF_WIDTH_GRID,
    VALIDATION_SCENARIO_IDS,
    rate_scale_mixture_for_half_width,
)


TEST_COMMIT = "a" * 40
TEST_ISOTOPES = ("Co-60", "Cs-137", "Eu-154")


def _synthetic_training_manifest() -> dict[str, object]:
    """Return a structurally valid test-only training provenance manifest."""
    width = RATE_SCALE_HALF_WIDTH_GRID[0]
    concentration = MARK_CONCENTRATION_GRID[-1]
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "pair_ids_by_scene": {
            str(seed): list(range(64))
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        },
        "artifact_sha256_by_scene": {
            str(seed): sha256(
                f"test-training-scene-{seed}".encode("utf-8")
            ).hexdigest()
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        },
        "rate_scale_family": (
            "station_shared_three_node_symmetric_mean_one"
        ),
        "mark_family": "source_fraction_dirichlet_multinomial",
        "selection_objective": (
            "maximum_joint_training_log_predictive_density"
        ),
        "selected_rate_scale_half_width": width,
        "selected_mark_concentration_source": concentration,
        "candidate_count": (
            len(RATE_SCALE_HALF_WIDTH_GRID)
            * len(MARK_CONCENTRATION_GRID)
        ),
        "selected_training_log_predictive_density": -1.0,
        "selection_artifact_sha256": sha256(
            b"test-only-global-discrepancy-selection"
        ).hexdigest(),
        "selection_completed": True,
    }


def _synthetic_validation_manifest(
    model_contract_hash: str,
    additive_scatter_contract_hash: str,
) -> dict[str, object]:
    """Return a structurally valid test-only independent-gate manifest."""
    all_seeds = (
        DESIGNATED_TRAINING_SCENE_SEEDS
        + DESIGNATED_HOLDOUT_SCENE_SEEDS
    )
    metrics: dict[str, dict[str, object]] = {}
    for metric_id, (comparison, threshold) in (
        ACCEPTANCE_METRIC_CONTRACT.items()
    ):
        metrics[metric_id] = {
            "value": float(threshold),
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": True,
        }
    return {
        "schema_version": 1,
        "validation_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "approved_model_contract_sha256": str(model_contract_hash),
        "native_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "additive_scatter_contract_sha256": str(
            additive_scatter_contract_hash
        ),
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "holdout_scene_seeds": list(DESIGNATED_HOLDOUT_SCENE_SEEDS),
        "training_selection_scene_seeds": list(
            DESIGNATED_TRAINING_SCENE_SEEDS
        ),
        "metric_scene_seeds": list(DESIGNATED_HOLDOUT_SCENE_SEEDS),
        "metric_split": "holdout_only",
        "metric_aggregation": "holdout_scene_conservative_worst_case",
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "pair_ids_by_scene": {
            str(seed): list(range(64)) for seed in all_seeds
        },
        "artifact_sha256_by_scene": {
            str(seed): sha256(
                f"test-validation-scene-{seed}".encode("utf-8")
            ).hexdigest()
            for seed in all_seeds
        },
        "scene_hash_by_scene_and_scenario": {
            str(seed): {
                scenario: sha256(
                    (
                        "test-validation-scene-geometry-"
                        f"{seed}-{scenario}"
                    ).encode("utf-8")
                ).hexdigest()
                for scenario in VALIDATION_SCENARIO_IDS
            }
            for seed in all_seeds
        },
        "surface_source_contract_sha256_by_scene_and_scenario": {
            str(seed): {
                scenario: sha256(
                    (
                        "test-validation-source-contract-"
                        f"{seed}-{scenario}"
                    ).encode("utf-8")
                ).hexdigest()
                for scenario in VALIDATION_SCENARIO_IDS
            }
            for seed in all_seeds
        },
        "metrics": metrics,
        "all_passed": True,
    }


def _synthetic_additive_scatter_response(
) -> AdditiveNoncollidedTransportResponse:
    """Return a nonzero authenticated test-only additive scatter response."""
    artifact_contracts = {
        str(seed): sha256(
            f"test-additive-contract-{seed}".encode("utf-8")
        ).hexdigest()
        for seed in DESIGNATED_TRAINING_SCENE_SEEDS
    }
    return AdditiveNoncollidedTransportResponse(
        coefficients=(0.8, 0.5, 0.4, 0.2, 0.1, 0.05, 0.025),
        ridge_lambda=0.1,
        training_manifest={
            "schema_version": 2,
            "acceptance_contract_sha256": (
                FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            ),
            "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
            "scenario_ids": list(VALIDATION_SCENARIO_IDS),
            "pair_ids_by_scene": {
                str(seed): list(range(64))
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            },
            "artifact_sha256_by_scene": {
                str(seed): sha256(
                    f"test-additive-scatter-{seed}".encode("utf-8")
                ).hexdigest()
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            },
            "artifact_contract_sha256_by_scene": artifact_contracts,
            "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
            "detector_response_contract_sha256": (
                NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            ),
            "obstacle_material_contract_sha256": (
                OBSTACLE_MATERIAL_CONTRACT_SHA256
            ),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
            "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
            "selection_objective": (
                "leave_one_training_scene_out_weighted_log1p_mse"
            ),
            "fit_sample_count": 256,
            "loso_scene_ids": [
                str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            ],
            "candidate_validation_scores": {
                format(value, ".12g"): (
                    0.5 if value == 0.1 else 1.0 + float(value)
                )
                for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            },
            "selected_validation_score": 0.5,
            "selected_ridge_lambda": 0.1,
            "selection_completed": True,
        },
    )


@lru_cache(maxsize=1)
def approved_full_spectrum_model() -> GeometryConditionedSpectralModel:
    """Return an approved model using explicit synthetic test provenance."""
    training = _synthetic_training_manifest()
    width = float(training["selected_rate_scale_half_width"])
    concentration = float(training["selected_mark_concentration_source"])
    nodes, weights = rate_scale_mixture_for_half_width(width)
    additive_scatter = _synthetic_additive_scatter_response()
    unvalidated = GeometryConditionedSpectralModel.standard_native(
        TEST_ISOTOPES,
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=concentration,
        discrepancy_training_manifest=training,
        additive_scatter_response=additive_scatter,
    )
    validation = _synthetic_validation_manifest(
        unvalidated.contract_hash_sha256,
        additive_scatter.contract_hash_sha256,
    )
    model = GeometryConditionedSpectralModel.standard_native(
        TEST_ISOTOPES,
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=concentration,
        discrepancy_training_manifest=training,
        validation_manifest=validation,
        additive_scatter_response=additive_scatter,
    )
    assert model.production_ready
    return model


@lru_cache(maxsize=1)
def _runtime_config_template() -> dict[str, object]:
    """Build one immutable schema-v3 runtime fixture template."""
    model = approved_full_spectrum_model()
    payload = model.manifest_payload()
    return {
        "simulation_runtime_schema_version": 1,
        "sim_backend": "analytic_test_fixture",
        "source_rate_model": "detector_cps_1m",
        "candidate_isotopes": list(TEST_ISOTOPES),
        "line_resolved_shield_attenuation": True,
        "detector_model_id": "test-detector.v2",
        "shield_model_id": "test-shield.v2",
        "transport_model_id": "test-transport.v2",
        "spectrum_model_id": "test-full-spectrum.v2",
        "detector_count_radius_m": 0.025,
        "detector_aperture_radius_m": 0.0,
        "detector_aperture_samples": 1,
        "obstacle_attenuation_enabled": True,
        "obstacle_height_m": 1.0,
        "energy_min_keV": 0.0,
        "energy_max_keV": 1700.0,
        "bin_width_keV": 2.0,
        "energy_bin_count": 851,
        "dead_time_tau_s": 5.813e-9,
        "background_rate_cps": 5.0,
        "full_spectrum_generative_model": payload,
        "full_spectrum_contract_hash_sha256": (
            model.contract_hash_sha256
        ),
    }


def runtime_config() -> dict[str, object]:
    """Return a fresh resolved schema-v3 physical test configuration."""
    return copy.deepcopy(_runtime_config_template())


def environment() -> dict[str, object]:
    """Return a small physical room without embedded source truth."""
    return {
        "environment_model_id": "test-room.v2",
        "obstacle_model_id": "test-obstacle-empty.v2",
        "size_x": 2.0,
        "size_y": 2.0,
        "size_z": 1.5,
        "detector_position": [0.25, 0.25, 0.4],
        "obstacle_grid": None,
    }


def records(
    record_count: int = 4,
    *,
    station_complete_markers: bool = False,
) -> tuple[MeasurementLogRecord, ...]:
    """Return ordered raw integer spectra with optional station boundaries."""
    contract_hash = approved_full_spectrum_model().contract_hash_sha256
    edges = np.arange(0.0, 1702.0 + 2.0, 2.0, dtype=np.float64)
    result: list[MeasurementLogRecord] = []
    for index in range(int(record_count)):
        station = index // 2
        pose = (0.25 + 0.5 * station, 0.25 + 0.25 * station, 0.4)
        spectrum_counts = np.zeros(851, dtype=np.int64)
        spectrum_counts[:4] = np.asarray(
            [15 + index, 10, 8, 4],
            dtype=np.int64,
        )
        station_end = (
            index + 1 == int(record_count)
            or (index + 1) // 2 != station
        )
        metadata: dict[str, object] = {
            "fixture_record": index,
            FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY: contract_hash,
        }
        if station_complete_markers and station_end:
            metadata["station_complete"] = True
        result.append(
            MeasurementLogRecord(
                step_id=index,
                action_id=index,
                station_id=station,
                detector_pose_xyz=pose,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=index % 8,
                pb_orientation_index=(index * 3) % 8,
                live_time_s=1.0,
                travel_time_s=0.0 if index % 2 else 0.25,
                shield_actuation_time_s=0.05,
                energy_bin_edges_keV=edges,
                spectrum_counts=spectrum_counts,
                metadata=metadata,
            )
        )
    return tuple(result)


def make_measurement_log(
    root: Path,
    *,
    record_count: int = 4,
    runtime_overrides: dict[str, object] | None = None,
    station_complete_markers: bool = False,
) -> Path:
    """Write one complete local schema-v2 MeasurementLog."""
    config = runtime_config()
    if runtime_overrides:
        config.update(runtime_overrides)
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    write_measurement_log(
        root,
        run_id="pure-pf-schema-v2-local-fixture",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
        records=records(
            record_count,
            station_complete_markers=station_complete_markers,
        ),
    )
    return root


def replay_config() -> dict[str, object]:
    """Return a small deterministic strict-PF replay configuration."""
    return {
        "pure_pf_schema_version": 1,
        "estimator_profile": "pf_strict",
        "num_particles": 12,
        "max_sources": 2,
        "init_num_sources": [1, 1],
        "variable_cardinality": False,
        "strength_prior_min_cps_1m": 1.0,
        "strength_prior_max_cps_1m": 2_000_000.0,
        "use_gpu": False,
        "gpu_dtype": "float64",
    }
