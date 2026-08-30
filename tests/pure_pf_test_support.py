"""Local schema-v6 physics-only fixtures for pure-PF contract tests."""

from __future__ import annotations

import copy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import tempfile

import numpy as np

from green_test_support import (
    synthetic_detector_green_validation_manifest,
    write_synthetic_detector_green_artifact,
)
from measurement.detector_geometry import DetectorObservationGeometry
from measurement.kernels import ShieldParams
from measurement.observation_model import RuntimeObservationModel
from measurement.source_boundary import surface_emission_policy_sha256
from pf.full_spectrum import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from pf.provenance import strict_canonical_json_bytes
from runtime.measurement_log import (
    MeasurementLogRecord,
    build_forward_model_manifest,
    write_measurement_log,
)
from runtime.forward_model_manifest import production_line_mu_by_isotope
from spectrum.air_attenuation import (
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
)
from spectrum.detector_green_operator import DetectorGreenOperator
from spectrum.detector_green_validation import (
    detector_green_validation_manifest_sha256,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)


TEST_COMMIT = "a" * 40
TEST_ISOTOPES = ("Co-60", "Cs-137", "Eu-154")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_REPOSITORY_ROOT = Path(
    __import__("spectrum").__file__ or ""
).resolve().parents[2]


@lru_cache(maxsize=1)
def _test_operator_manifest_path() -> Path:
    """Publish one process-local immutable synthetic Green artifact."""
    results_root = _RUNTIME_REPOSITORY_ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pf-test-green-", dir=results_root))
    return write_synthetic_detector_green_artifact(root / "operator")


@lru_cache(maxsize=1)
def _test_operator() -> DetectorGreenOperator:
    """Load the process-local file-backed synthetic Green operator."""
    return DetectorGreenOperator.from_artifact(_test_operator_manifest_path())


def _synthetic_validation_manifest(
    *,
    model_contract_hash: str,
    additive_scatter_contract_hash: str,
) -> dict[str, object]:
    """Return a strict schema-v6 validation-only approval manifest."""
    operator = _test_operator()
    green_validation = synthetic_detector_green_validation_manifest(
        operator,
        runtime_config_sha256="7" * 64,
        native_executable_sha256="2" * 64,
        native_execution_environment_sha256="3" * 64,
        detector_implementation_bundle_sha256="4" * 64,
    )
    metrics = {
        metric_id: {
            "value": float(threshold),
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": True,
        }
        for metric_id, (comparison, threshold) in (ACCEPTANCE_METRIC_CONTRACT.items())
    }
    seeds = DESIGNATED_VALIDATION_SCENE_SEEDS
    return {
        "schema_version": 6,
        "validation_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "approved_model_contract_sha256": model_contract_hash,
        "acceptance_run_contract_sha256": "6" * 64,
        "runtime_config_sha256": "7" * 64,
        "native_executable_sha256": "2" * 64,
        "native_execution_environment_sha256": "3" * 64,
        "implementation_bundle_sha256": "4" * 64,
        "detector_green_operator_contract_sha256": (operator.contract_hash_sha256),
        "detector_green_operator_binary_sha256": operator.binary_sha256,
        "detector_green_validation": green_validation,
        "detector_green_validation_manifest_sha256": (
            detector_green_validation_manifest_sha256(
                green_validation,
                operator=operator,
            )
        ),
        "additive_scatter_contract_sha256": additive_scatter_contract_hash,
        "surface_emission_policy_sha256": surface_emission_policy_sha256(),
        "validation_scene_seeds": list(seeds),
        "candidate_selection": "none_predeclared_physics_only",
        "scene_calibration_count": 0,
        "metric_scene_seeds": list(seeds),
        "metric_split": "independent_validation_only",
        "metric_aggregation": "validation_scene_conservative_worst_case",
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "pair_ids_by_scene": {str(seed): list(range(64)) for seed in seeds},
        "artifact_sha256_by_scene": {
            str(seed): sha256(
                f"test-validation-scene-{seed}".encode("utf-8")
            ).hexdigest()
            for seed in seeds
        },
        "scene_hash_by_scene_and_scenario": {
            str(seed): {
                scenario: sha256(
                    f"test-scene-{seed}-{scenario}".encode("utf-8")
                ).hexdigest()
                for scenario in VALIDATION_SCENARIO_IDS
            }
            for seed in seeds
        },
        "surface_source_contract_sha256_by_scene_and_scenario": {
            str(seed): {
                scenario: sha256(
                    f"test-source-{seed}-{scenario}".encode("utf-8")
                ).hexdigest()
                for scenario in VALIDATION_SCENARIO_IDS
            }
            for seed in seeds
        },
        "metrics": metrics,
        "all_passed": True,
    }


@lru_cache(maxsize=None)
def approved_full_spectrum_model(
    isotopes: tuple[str, ...] = TEST_ISOTOPES,
) -> GeometryConditionedSpectralModel:
    """Return an approved physics-only model with synthetic unit evidence."""
    operator = _test_operator()
    unvalidated = GeometryConditionedSpectralModel.physics_only_native(
        isotopes,
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        detector_green_operator=operator,
    )
    response = unvalidated.additive_scatter_response
    assert response is not None
    validation = _synthetic_validation_manifest(
        model_contract_hash=unvalidated.contract_hash_sha256,
        additive_scatter_contract_hash=response.contract_hash_sha256,
    )
    model = GeometryConditionedSpectralModel.physics_only_native(
        isotopes,
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        validation_manifest=validation,
        detector_green_operator=operator,
    )
    assert model.production_ready
    return model


@lru_cache(maxsize=None)
def _runtime_observation_model_cached(
    isotopes: tuple[str, ...],
) -> RuntimeObservationModel:
    """Build one exact runtime observation contract for PF boundary tests."""
    line_table = production_line_mu_by_isotope(isotopes)
    scalar_mu = {
        isotope: {
            material: float(
                sum(
                    float(row["weight"]) * float(row[material])
                    for row in line_table[isotope]
                )
            )
            for material in ("fe", "pb")
        }
        for isotope in isotopes
    }
    return RuntimeObservationModel(
        detector_geometry=DetectorObservationGeometry(
            count_radius_m=0.0,
            aperture_radius_m=0.0395,
            aperture_samples=33,
            aperture_sampling="solid_angle_cone",
        ),
        shield_params=ShieldParams(),
        mu_by_isotope=scalar_mu,
        line_mu_by_isotope=line_table,
        additive_scatter_response=(
            approved_full_spectrum_model(isotopes).additive_scatter_response
        ),
        obstacle_mu_by_isotope=None,
        obstacle_height_m=2.0,
        obstacle_buildup_coeff=0.0,
        source_extent_radius_m=0.0,
        source_extent_samples=1,
        dry_air_total_attenuation_contract_id=(
            NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID
        ),
        dry_air_total_attenuation_contract_sha256=(
            NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256
        ),
    )


def runtime_observation_model(
    isotopes: tuple[str, ...],
) -> RuntimeObservationModel:
    """Return a shared exact observation contract for the requested isotopes."""
    return _runtime_observation_model_cached(tuple(isotopes))


@lru_cache(maxsize=1)
def _runtime_config_template() -> dict[str, object]:
    """Build one immutable schema-v6 runtime fixture template."""
    model = approved_full_spectrum_model()
    payload = model.manifest_payload()
    return {
        "simulation_runtime_schema_version": 1,
        "sim_backend": "analytic_test_fixture",
        "source_rate_model": "detector_cps_1m",
        "detector_green_operator_manifest": (
            _test_operator_manifest_path()
            .relative_to(_RUNTIME_REPOSITORY_ROOT)
            .as_posix()
        ),
        "candidate_isotopes": list(TEST_ISOTOPES),
        "line_resolved_shield_attenuation": True,
        "detector_count_radius_m": 0.025,
        "detector_aperture_radius_m": 0.0395,
        "detector_aperture_samples": 33,
        "obstacle_attenuation_enabled": True,
        "obstacle_height_m": 1.0,
        "energy_min_keV": 0.0,
        "energy_max_keV": 1700.0,
        "bin_width_keV": 2.0,
        "energy_bin_count": 851,
        "dead_time_tau_s": 5.813e-9,
        "background_cps": 5.0,
        "full_spectrum_generative_model": payload,
        "full_spectrum_contract_hash_sha256": model.contract_hash_sha256,
    }


def runtime_config() -> dict[str, object]:
    """Return a fresh resolved schema-v6 physical test configuration."""
    return copy.deepcopy(_runtime_config_template())


def environment() -> dict[str, object]:
    """Return a small physical room without embedded source truth."""
    return {
        "environment_model_id": "test-room",
        "size_x": 2.0,
        "size_y": 2.0,
        "size_z": 1.5,
        "detector_position": [0.25, 0.25, 0.4],
        "obstacle_grid": None,
        "adaptive_measurement": {"shield_angular_speed_rad_s": 1.0},
        "acquisition_contract": {
            "schema_version": 1,
            "max_stations": 16,
            "views_per_station": 2,
            "live_time_s": 1.0,
            "max_measurements": 32,
            "min_station_separation_m": 0.1,
            "coverage_radius_m": 1.0,
        },
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
        station_end = index + 1 == int(record_count) or (index + 1) // 2 != station
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
    """Write one complete local MeasurementLog."""
    config = runtime_config()
    if runtime_overrides:
        config.update(runtime_overrides)
    env = environment()
    config_hash = sha256(strict_canonical_json_bytes(config)).hexdigest()
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
        run_id="pure-pf-local-fixture",
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
