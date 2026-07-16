"""Local truth-free fixtures for pure-PF contract tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from pf.provenance import canonical_json_bytes
from runtime.measurement_log import (
    MeasurementLogRecord,
    build_forward_model_manifest,
    write_measurement_log,
)


TEST_COMMIT = "a" * 40
TEST_ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


def runtime_config() -> dict[str, object]:
    """Return a small resolved physical configuration for replay tests."""
    return {
        "sim_backend": "analytic",
        "spectrum_count_method": "response_poisson",
        "source_rate_model": "detector_cps_1m",
        "pf_line_resolved_shield_attenuation": True,
        "detector_model_id": "test-detector.v1",
        "shield_model_id": "test-shield.v1",
        "transport_model_id": "test-transport.v1",
        "spectrum_model_id": "test-spectrum.v1",
        "pf_detector_count_radius_m": 0.025,
        "pf_detector_aperture_radius_m": 0.0,
        "pf_detector_aperture_samples": 1,
        "obstacle_height_m": 1.0,
    }


def environment() -> dict[str, object]:
    """Return a small physical room without embedded truth."""
    return {
        "environment_model_id": "test-room.v1",
        "obstacle_model_id": "test-obstacle-empty.v1",
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
    """Return an ordered multi-station observation sequence."""
    edges = np.asarray([0.0, 500.0, 1000.0, 1500.0, 2000.0], dtype=np.float64)
    result: list[MeasurementLogRecord] = []
    for index in range(int(record_count)):
        station = index // 2
        pose = (0.25 + 0.5 * station, 0.25 + 0.25 * station, 0.4)
        counts = {
            "Cs-137": 20.0 + index,
            "Co-60": 12.0 + 0.5 * index,
            "Eu-154": 8.0 + 0.25 * index,
        }
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
                spectrum_counts=np.asarray(
                    [15.0 + index, 10.0, 8.0, 4.0],
                    dtype=np.float64,
                ),
                spectrum_variance=np.asarray(
                    [15.0 + index, 10.0, 8.0, 4.0],
                    dtype=np.float64,
                ),
                isotope_counts=counts,
                isotope_count_covariance=np.diag(
                    [counts["Cs-137"], counts["Co-60"], counts["Eu-154"]]
                ).astype(np.float64),
                metadata={
                    "fixture_record": index,
                    **(
                        {"station_complete": True}
                        if station_complete_markers
                        and (
                            index + 1 == int(record_count)
                            or (index + 1) // 2 != station
                        )
                        else {}
                    ),
                },
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
    """Write one complete local MeasurementLog and return its root."""
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


def replay_config() -> dict[str, object]:
    """Return a small deterministic strict-PF replay configuration."""
    return {
        "estimator_profile": "pf_strict",
        "num_particles": 12,
        "max_sources": 2,
        "init_num_sources": [1, 1],
        "birth_enable": False,
        "use_gpu": False,
        "replay_candidate_sources_xyz": [
            [0.25, 0.25, 0.0],
            [1.0, 1.0, 0.75],
            [1.75, 1.75, 1.5],
        ],
    }
