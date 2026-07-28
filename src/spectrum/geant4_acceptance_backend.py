"""Real external-Geant4 backend for the fixed all-64 acceptance corpus.

The backend owns only data acquisition and known-geometry tensor construction.
It never fits a response, reads a holdout result while training, or substitutes
an analytic observation for native transport.  Every acquired observation is
the post-response, post-dead-time integer spectrum returned by the configured
external Geant4 executable.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacle_assets import obstacle_instances_to_dicts
from measurement.obstacles import ObstacleGrid, build_obstacle_grid
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    canonical_surface_source_runtime_payload,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from measurement.source_surfaces import generate_surface_sources
from measurement.surface_charts import build_surface_chart_geometry
from pf.randomness import named_random_generator
from pf.surface_atlas import ContinuousSurfaceAtlas
from runtime_environment import attach_random_manchester_transport_geometry
from sim.geant4_app.app import Geant4AppConfig, Geant4Application
from sim.geant4_app.engine import Geant4StepRequest
from sim.geant4_app.io_format import (
    write_request_file,
    write_scene_file,
)
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_TARGET_SEMANTICS,
    physical_scatter_basis_numpy,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_DWELL_TIME_S,
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_PAIR_IDS,
    ACCEPTANCE_SCENARIO_SOURCE_SPEC,
    AcceptanceScenarioSession,
    AcceptanceTransportBackend,
    NATIVE_ACCEPTANCE_FIDELITY,
    acceptance_transport_seed,
    canonical_json_sha256,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import (
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    TRANSPORT_FEATURE_ORDER,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)


ACCEPTANCE_ROOM_SIZE_XYZ = (10.0, 20.0, 10.0)
ACCEPTANCE_DETECTOR_POSE_XYZ = (1.0, 1.0, 0.5)
ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION = 0.4
ACCEPTANCE_PASSAGE_WIDTH_M = 1.0
ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M = 1.0
_SOURCE_RNG_DOMAIN = "full_spectrum_acceptance_surface_truth_v1"
_BOUNDARY_RNG_DOMAIN = "full_spectrum_acceptance_boundary_probe_v1"
_BOUNDARY_ERROR_MARKER = "surface-anchor epsilon contract"
_FULL_SPECTRUM_RUNTIME_KEYS = frozenset(
    {
        "full_spectrum_generative_model",
        "full_spectrum_generative_model_path",
        "full_spectrum_generative_model_file_sha256",
        "full_spectrum_contract_hash_sha256",
    }
)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_metadata_bool(
    metadata: Mapping[str, object],
    key: str,
) -> bool:
    """Return one exact native boolean after response-file parsing."""
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"Native Geant4 metadata {key} must be boolean.")
    return value


def _strict_metadata_number(
    metadata: Mapping[str, object],
    key: str,
) -> float:
    """Return one finite native number excluding booleans."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Native Geant4 metadata {key} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"Native Geant4 metadata {key} must be finite.")
    return parsed


def _integer_spectrum(
    value: object,
    *,
    field_name: str,
) -> NDArray[np.int64]:
    """Return one exact native unit-weight 851-bin integer spectrum."""
    raw = np.asarray(value)
    if (
        raw.shape != (NATIVE_GEANT4_BIN_COUNT,)
        or raw.dtype == np.bool_
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise RuntimeError(
            f"{field_name} must be a numeric 851-bin native spectrum."
        )
    numeric = np.asarray(raw, dtype=np.float64)
    rounded = np.rint(numeric)
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or np.any(np.abs(numeric - rounded) > 1.0e-9)
        or np.any(rounded > np.iinfo(np.int64).max)
    ):
        raise RuntimeError(
            f"{field_name} must contain unit-weight nonnegative integers."
        )
    return rounded.astype(np.int64)


def _metadata_spectrum(
    metadata: Mapping[str, object],
    key: str,
) -> NDArray[np.int64]:
    """Parse one comma-separated native validation-only spectrum."""
    raw = metadata.get(key)
    if not isinstance(raw, str):
        raise RuntimeError(f"Native Geant4 metadata {key} is missing.")
    parts = raw.split(",")
    if len(parts) != NATIVE_GEANT4_BIN_COUNT:
        raise RuntimeError(
            f"Native Geant4 metadata {key} must contain 851 bins."
        )
    try:
        values = np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise RuntimeError(
            f"Native Geant4 metadata {key} is not numeric."
        ) from exc
    return _integer_spectrum(values, field_name=key)


def _source_payload(source: PointSource) -> dict[str, object]:
    """Return one canonical truth-anchor/native-transport source payload."""
    return canonical_surface_source_runtime_payload(
        [
            {
                "isotope": source.isotope,
                "position": list(source.position),
                "transport_position": list(
                    source.transport_position_array()
                ),
                "intensity_cps_1m": source.intensity_cps_1m,
                "surface_chart_id": source.surface_chart_id,
                "surface_uv": list(source.surface_uv),
                "surface_normal": list(source.surface_normal),
                "surface_emission_policy_sha256": (
                    source.surface_emission_policy_sha256
                ),
            }
        ]
    )[0]


def _line_token(
    *,
    source_index: int,
    isotope: str,
    energy_keV: float,
) -> str:
    """Return the exact native source-line validation token."""
    return (
        f"src{source_index}_{isotope.replace('-', '_')}_"
        f"e{energy_keV:.1f}"
    ).replace(".", "p")


def _validation_labels(
    metadata: Mapping[str, object],
    *,
    sources: Sequence[PointSource],
    model: GeometryConditionedSpectralModel,
) -> dict[str, object]:
    """Extract source-resolved pre-dead-time labels for training only."""
    expected_tokens = {
        _line_token(
            source_index=source_index,
            isotope=source.isotope,
            energy_keV=float(line["energy_keV"]),
        )
        for source_index, source in enumerate(sources)
        for line in model.line_identity
        if line["isotope"] == source.isotope
    }
    class_names = (
        "uncollided_primary",
        "interacted_primary",
        "secondary",
    )
    native_prefix = "validation_only_entry_spectrum_"
    expected_metadata_keys = {
        native_prefix + token + "_" + class_name
        for token in expected_tokens
        for class_name in class_names
    }
    unexpected = {
        key
        for key in metadata
        if key.startswith(native_prefix) and key not in expected_metadata_keys
    }
    if unexpected:
        raise RuntimeError(
            "Native Geant4 emitted unexpected source-line labels: "
            f"{sorted(unexpected)}."
        )
    zero = np.zeros(NATIVE_GEANT4_BIN_COUNT, dtype=np.int64)
    totals: dict[str, dict[str, int]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for token in sorted(expected_tokens):
        totals[token] = {}
        hashes[token] = {}
        for class_name in class_names:
            key = native_prefix + token + "_" + class_name
            spectrum = (
                _metadata_spectrum(metadata, key)
                if key in metadata
                else zero
            )
            totals[token][class_name] = int(np.sum(spectrum, dtype=np.int64))
            hashes[token][class_name] = canonical_json_sha256(
                spectrum.tolist()
            )
    background = _metadata_spectrum(
        metadata,
        "validation_only_background_analysis_spectrum",
    )
    return {
        "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
        "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
        "entry_class_totals_by_source_line": totals,
        "entry_spectrum_sha256_by_source_line_class": hashes,
        "background_entry_total": int(
            np.sum(background, dtype=np.int64)
        ),
        "background_entry_spectrum_sha256": canonical_json_sha256(
            background.tolist()
        ),
    }


def _native_fidelity(
    metadata: Mapping[str, object],
    *,
    config: Geant4AppConfig,
    source_count: int,
) -> dict[str, object]:
    """Authenticate native unit-history postconditions and return the contract."""
    exact_strings = {
        "backend": "geant4",
        "engine_mode": "external",
        "physics_profile": "balanced",
        "source_rate_model": "detector_cps_1m",
        "detector_scoring_mode": "incident_gamma_energy",
        "secondary_transport_mode": "full_transport",
        "transport_history_mode": "full_unit_weight",
        "validation_entry_spectrum_space": (
            "pre_dead_time_raw_incident_gamma"
        ),
        "validation_entry_spectrum_grouping": (
            "source_token_initial_gamma_line_entry_class"
        ),
    }
    for key, expected in exact_strings.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Native Geant4 metadata {key} != {expected!r}."
            )
    if (
        config.engine_mode != "external"
        or config.physics_profile != "balanced"
        or config.thread_count != 32
        or config.source_rate_model != "detector_cps_1m"
        or config.detector_scoring_mode != "incident_gamma_energy"
        or config.secondary_transport_mode != "full_transport"
        or config.primary_sampling_fraction != 1.0
        or config.target_sampled_primaries is not None
        or config.accelerated_weighted_transport_enable
        or not config.sample_detector_response
        or not config.validation_entry_class_spectra
    ):
        raise RuntimeError(
            "Acceptance backend configuration is not native full fidelity."
        )
    numeric_expected = {
        "requested_threads": 32.0,
        "primary_sampling_fraction": 1.0,
        "primary_history_weight": 1.0,
        "target_sampled_primaries": 0.0,
        "spectrum_bin_count": float(NATIVE_GEANT4_BIN_COUNT),
    }
    for key, expected in numeric_expected.items():
        if not np.isclose(
            _strict_metadata_number(metadata, key),
            expected,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise RuntimeError(f"Native Geant4 numeric fidelity failed: {key}.")
    bool_expected = {
        "multithreaded_run_manager": True,
        "primary_sampling_budget_enabled": False,
        "history_thinning_enabled": False,
        "transport_tally_weighted": False,
        "weighted_transport": False,
        "theory_tvl_attenuation": False,
        "detector_response_applied_in_native": True,
        "validation_entry_class_spectra": True,
        "source_bias_weighted_transport": False,
    }
    for key, expected in bool_expected.items():
        if _strict_metadata_bool(metadata, key) is not expected:
            raise RuntimeError(f"Native Geant4 boolean fidelity failed: {key}.")
    if (
        metadata.get("detector_response_sampling_contract_sha256")
        != NATIVE_ACCEPTANCE_FIDELITY[
            "detector_response_sampling_contract_sha256"
        ]
        or metadata.get("detector_response_sampling_mode")
        != "multinomial_marking_with_nonparalyzable_event_time"
    ):
        raise RuntimeError("Native detector-response marking is incompatible.")
    surface_bound = _strict_metadata_bool(
        metadata,
        "all_sources_surface_bound",
    )
    if source_count == 0:
        if (
            surface_bound
            or metadata.get("surface_emission_policy_sha256") != ""
            or _strict_metadata_number(
                metadata,
                "surface_emission_epsilon_m",
            )
            != 0.0
        ):
            raise RuntimeError(
                "Background-only native surface provenance is inconsistent."
            )
    elif (
        not surface_bound
        or metadata.get("surface_emission_policy_sha256")
        != surface_emission_policy_sha256()
        or not np.isclose(
            _strict_metadata_number(
                metadata,
                "surface_emission_epsilon_m",
            ),
            SURFACE_EMISSION_EPSILON_M,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise RuntimeError("Native surface-emission provenance is invalid.")
    return dict(NATIVE_ACCEPTANCE_FIDELITY)


def _scene_payload(
    *,
    grid: ObstacleGrid,
    instances: Sequence[object],
    sources: Sequence[PointSource],
    author_room_boundaries: bool,
    obstacle_material: str,
) -> dict[str, object]:
    """Return one strict sidecar scene payload for a fixed environment."""
    return {
        "usd_path": None,
        "room_size_xyz": list(ACCEPTANCE_ROOM_SIZE_XYZ),
        "sources": [_source_payload(source) for source in sources],
        "obstacle_origin_xy": list(grid.origin),
        "obstacle_cell_size_m": float(grid.cell_size),
        "obstacle_material": obstacle_material,
        "obstacle_grid_shape": list(grid.grid_shape),
        "obstacle_cells": [list(cell) for cell in grid.blocked_cells],
        "collision_boxes_m": [
            list(box) for box in grid.collision_boxes_m
        ],
        "transport_boxes_m": [
            list(box) for box in grid.transport_boxes_m
        ],
        "transport_mu_by_isotope": {
            str(isotope): [float(value) for value in values]
            for isotope, values in grid.transport_mu_by_isotope.items()
        },
        "transport_line_mu_by_isotope": {
            str(isotope): [
                [float(value) for value in row] for row in rows
            ]
            for isotope, rows in (
                grid.transport_line_mu_by_isotope.items()
            )
        },
        "transport_line_compton_mu_by_isotope": {
            str(isotope): [
                [float(value) for value in row] for row in rows
            ]
            for isotope, rows in (
                grid.transport_line_compton_mu_by_isotope.items()
            )
        },
        "obstacle_instances": obstacle_instances_to_dicts(instances),
        "author_obstacle_prims": True,
        "author_room_boundary_prims": author_room_boundaries,
        "use_config_usd_fallback": False,
    }


def _build_environment(
    *,
    scene_seed: int,
    obstacle_height_m: float,
    author_room_boundaries: bool,
    room_boundary_thickness_m: float,
) -> tuple[EnvironmentConfig, ObstacleGrid, tuple[object, ...]]:
    """Build the deterministic random Manchester environment for one seed."""
    environment = EnvironmentConfig(
        size_x=ACCEPTANCE_ROOM_SIZE_XYZ[0],
        size_y=ACCEPTANCE_ROOM_SIZE_XYZ[1],
        size_z=ACCEPTANCE_ROOM_SIZE_XYZ[2],
        detector_position=ACCEPTANCE_DETECTOR_POSE_XYZ,
    )
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=environment.size_x,
        size_y=environment.size_y,
        cell_size=1.0,
        blocked_fraction=ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION,
        rng_seed=scene_seed,
        keep_free_points=(ACCEPTANCE_DETECTOR_POSE_XYZ[:2],),
        passage_width_m=ACCEPTANCE_PASSAGE_WIDTH_M,
    )
    grid, instances = attach_random_manchester_transport_geometry(
        grid,
        room_size_xyz=ACCEPTANCE_ROOM_SIZE_XYZ,
        obstacle_height_m=obstacle_height_m,
        rng_seed=scene_seed,
        include_room_boundaries=author_room_boundaries,
        room_boundary_thickness_m=room_boundary_thickness_m,
    )
    return environment, grid, tuple(instances)


def _generate_sources(
    *,
    environment: EnvironmentConfig,
    grid: ObstacleGrid,
    scene_seed: int,
    scenario_id: str,
    obstacle_height_m: float,
) -> tuple[PointSource, ...]:
    """Draw continuous area-uniform truth for one fixed scenario."""
    specification = ACCEPTANCE_SCENARIO_SOURCE_SPEC[scenario_id]
    if not specification:
        return ()
    isotopes = tuple(isotope for isotope, _ in specification)
    intensities = {
        isotope: float(intensity) for isotope, intensity in specification
    }
    sources = generate_surface_sources(
        env=environment,
        obstacle_grid=grid,
        isotopes=isotopes,
        intensity_cps_1m=intensities,
        rng=named_random_generator(
            scene_seed,
            _SOURCE_RNG_DOMAIN,
            scenario_id,
        ),
        count=len(specification),
        obstacle_height_m=obstacle_height_m,
        chart_max_edge_m=ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
    )
    return tuple(sources)


def _perturbed_sources(
    *,
    environment: EnvironmentConfig,
    grid: ObstacleGrid,
    sources: Sequence[PointSource],
    obstacle_height_m: float,
) -> tuple[PointSource, ...]:
    """Return a fixed tangent displacement without response-based selection."""
    if len(sources) != 1:
        raise ValueError("Perturbation scenario requires exactly one source.")
    source = sources[0]
    atlas = ContinuousSurfaceAtlas(
        build_surface_chart_geometry(
            environment,
            grid,
            max_edge_m=ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
            obstacle_height_m=obstacle_height_m,
        )
    )
    chart_ids = np.full(8, int(source.surface_chart_id), dtype=np.int64)
    uv = np.broadcast_to(
        np.asarray(source.surface_uv, dtype=np.float64),
        (8, 2),
    ).copy()
    displacements = np.asarray(
        (
            (0.25, 0.0),
            (-0.25, 0.0),
            (0.0, 0.25),
            (0.0, -0.25),
            (0.18, 0.18),
            (-0.18, 0.18),
            (0.18, -0.18),
            (-0.18, -0.18),
        ),
        dtype=np.float64,
    )
    proposed_ids, proposed_uv, _, valid, _ = (
        atlas.trace_tangent_displacements(
            chart_ids,
            uv,
            displacements,
        )
    )
    proposed_positions = atlas.positions_xyz(proposed_ids, proposed_uv)
    anchor = np.asarray(source.position, dtype=np.float64)
    usable = valid & (
        np.linalg.norm(proposed_positions - anchor[None, :], axis=1) >= 0.20
    )
    if not np.any(usable):
        raise RuntimeError(
            "Fixed continuous-surface perturbation could not be traced."
        )
    index = int(np.flatnonzero(usable)[0])
    chart_id = int(proposed_ids[index])
    position = proposed_positions[index]
    normal = atlas.air_facing_normals_xyz(
        np.asarray([chart_id], dtype=np.int64)
    )[0]
    transport = position + SURFACE_EMISSION_EPSILON_M * normal
    return (
        PointSource(
            isotope=source.isotope,
            position=tuple(float(value) for value in position),
            intensity_cps_1m=float(source.intensity_cps_1m),
            surface_chart_id=chart_id,
            surface_uv=tuple(float(value) for value in proposed_uv[index]),
            surface_normal=tuple(float(value) for value in normal),
            transport_position=tuple(float(value) for value in transport),
            surface_emission_policy_sha256=(
                surface_emission_policy_sha256()
            ),
        ),
    )


@dataclass(frozen=True)
class _GeometryBatch:
    """Store all-64 source-line geometry for one scenario."""

    unattenuated_vsl: NDArray[np.float64]
    uncollided_vsl: NDArray[np.float64]
    features_vslf: NDArray[np.float64]
    scatter_basis_vslf: NDArray[np.float64]


def _geometry_batch(
    *,
    kernel: object,
    model: GeometryConditionedSpectralModel,
    detector_pose_xyz: tuple[float, float, float],
    sources: Sequence[PointSource],
) -> _GeometryBatch:
    """Evaluate all source slots and all 64 shield pairs in batched kernels."""
    view_count = len(ACCEPTANCE_PAIR_IDS)
    source_count = len(sources)
    line_rows = model.line_identity
    line_count = len(line_rows)
    unattenuated = np.zeros(
        (view_count, source_count, line_count),
        dtype=np.float64,
    )
    uncollided = np.zeros_like(unattenuated)
    features = np.zeros(
        unattenuated.shape + (len(TRANSPORT_FEATURE_ORDER),),
        dtype=np.float64,
    )
    scatter = np.zeros(
        unattenuated.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),),
        dtype=np.float64,
    )
    if source_count == 0:
        return _GeometryBatch(unattenuated, uncollided, features, scatter)
    detectors = np.broadcast_to(
        np.asarray(detector_pose_xyz, dtype=np.float64),
        (view_count, 3),
    ).copy()
    fe_indices = np.asarray(
        [pair_id // 8 for pair_id in ACCEPTANCE_PAIR_IDS],
        dtype=np.int64,
    )
    pb_indices = np.asarray(
        [pair_id % 8 for pair_id in ACCEPTANCE_PAIR_IDS],
        dtype=np.int64,
    )
    for isotope in ACCEPTANCE_ISOTOPES:
        source_indices = np.asarray(
            [
                index
                for index, source in enumerate(sources)
                if source.isotope == isotope
            ],
            dtype=np.int64,
        )
        if source_indices.size == 0:
            continue
        global_indices = np.asarray(
            [
                index
                for index, line in enumerate(line_rows)
                if line["isotope"] == isotope
            ],
            dtype=np.int64,
        )
        local_indices = np.asarray(
            [
                int(line_rows[index]["transport_line_index"])
                for index in global_indices
            ],
            dtype=np.int64,
        )
        source_positions = np.asarray(
            [
                sources[index].transport_position_array()
                for index in source_indices
            ],
            dtype=np.float64,
        )
        components = (
            kernel.line_transport_components_selected_pairs_for_detectors(
                isotope,
                detectors,
                source_positions,
                fe_indices,
                pb_indices,
                local_indices,
            )
        )
        branching = kernel.line_branching_weights(
            isotope,
            local_indices,
        )
        strengths = np.asarray(
            [
                sources[index].intensity_cps_1m
                for index in source_indices
            ],
            dtype=np.float64,
        )
        rate_scale = strengths[None, :, None] * branching[None, None, :]
        selection = np.ix_(
            np.arange(view_count, dtype=np.int64),
            source_indices,
            global_indices,
        )
        unattenuated[selection] = (
            components.unattenuated_kernel * rate_scale
        )
        uncollided[selection] = components.uncollided_kernel * rate_scale
        feature_values = np.stack(
            (
                components.tau_fe,
                components.tau_pb,
                components.tau_obstacle,
                components.distance_m,
            ),
            axis=-1,
        )
        scatter_values = physical_scatter_basis_numpy(
            tau_fe=components.tau_fe,
            tau_pb=components.tau_pb,
            tau_obstacle=components.tau_obstacle,
            tau_obstacle_compton=components.tau_obstacle_compton,
            distance_m=components.distance_m,
            energy_keV=np.asarray(
                [line_rows[index]["energy_keV"] for index in global_indices],
                dtype=np.float64,
            )[None, None, :],
            mu_fe_cm_inv=np.asarray(
                [line_rows[index]["mu_fe_cm_inv"] for index in global_indices],
                dtype=np.float64,
            )[None, None, :],
            mu_pb_cm_inv=np.asarray(
                [line_rows[index]["mu_pb_cm_inv"] for index in global_indices],
                dtype=np.float64,
            )[None, None, :],
        )
        features[selection] = feature_values
        scatter[selection] = scatter_values
    if (
        np.any(~np.isfinite(unattenuated))
        or np.any(unattenuated < 0.0)
        or np.any(~np.isfinite(uncollided))
        or np.any(uncollided < 0.0)
        or np.any(uncollided > unattenuated * (1.0 + 1.0e-12))
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
        or np.any(~np.isfinite(scatter))
        or np.any(scatter < 0.0)
    ):
        raise RuntimeError("Acceptance geometry batch is physically invalid.")
    return _GeometryBatch(unattenuated, uncollided, features, scatter)


def _command_for_pair(pair_id: int) -> SimulationCommand:
    """Return the fixed detector pose and one of all 64 shield programs."""
    return SimulationCommand(
        step_id=int(pair_id),
        target_pose_xyz=ACCEPTANCE_DETECTOR_POSE_XYZ,
        target_base_yaw_rad=0.0,
        fe_orientation_index=int(pair_id // 8),
        pb_orientation_index=int(pair_id % 8),
        dwell_time_s=ACCEPTANCE_DWELL_TIME_S,
    )


def _request_for_command(
    app: Geant4Application,
    command: SimulationCommand,
    *,
    seed: int,
    dwell_time_s: float,
) -> Geant4StepRequest:
    """Apply one command and return the exact native request poses."""
    app.robot_controller.apply_command(command)
    detector = app.robot_controller.detector_world_pose()
    fe_pose = app._stage_backend.get_world_pose(  # noqa: SLF001
        app.scene.prim_paths.fe_shield_path
    )
    pb_pose = app._stage_backend.get_world_pose(  # noqa: SLF001
        app.scene.prim_paths.pb_shield_path
    )
    return Geant4StepRequest(
        step_id=int(command.step_id),
        dwell_time_s=float(dwell_time_s),
        seed=int(seed),
        detector_pose_xyz=detector.translation_xyz,
        detector_quat_wxyz=detector.orientation_wxyz,
        fe_shield_pose_xyz=fe_pose.translation_xyz,
        fe_shield_quat_wxyz=fe_pose.orientation_wxyz,
        pb_shield_pose_xyz=pb_pose.translation_xyz,
        pb_shield_quat_wxyz=pb_pose.orientation_wxyz,
    )


def _mutated_surface_scene(
    raw_scene: str,
    *,
    variant: str,
) -> str:
    """Move the first native source to one signed boundary variant."""
    lines = raw_scene.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("SOURCE "):
            continue
        fields = line.split()
        values = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in fields[1:]
        }
        anchor = np.asarray(
            [
                float(values["anchor_x"]),
                float(values["anchor_y"]),
                float(values["anchor_z"]),
            ],
            dtype=np.float64,
        )
        normal = np.asarray(
            [
                float(values["surface_normal_x"]),
                float(values["surface_normal_y"]),
                float(values["surface_normal_z"]),
            ],
            dtype=np.float64,
        )
        if variant == "air_plus_epsilon":
            position = anchor + SURFACE_EMISSION_EPSILON_M * normal
        elif variant == "exact_surface_anchor":
            position = anchor
        elif variant == "solid_minus_epsilon":
            position = anchor - SURFACE_EMISSION_EPSILON_M * normal
        else:
            raise ValueError(f"Unknown boundary variant: {variant}.")
        replacements = dict(
            zip(("x", "y", "z"), (format(value, ".17g") for value in position))
        )
        lines[index] = " ".join(
            (
                fields[0],
                *(
                    (
                        token.split("=", 1)[0]
                        + "="
                        + replacements[token.split("=", 1)[0]]
                    )
                    if token.split("=", 1)[0] in replacements
                    else token
                    for token in fields[1:]
                ),
            )
        )
        return "\n".join(lines) + "\n"
    raise RuntimeError("Exported probe scene contains no SOURCE record.")


def _surface_boundary_gate(
    *,
    app: Geant4Application,
    scene_seed: int,
) -> dict[str, object]:
    """Execute actual native air/exact/solid signed-epsilon probes."""
    exported = getattr(app.engine, "scene", None)
    if exported is None or not exported.sources:
        raise RuntimeError("Boundary probe requires one exported surface source.")
    command = _command_for_pair(0)
    request = _request_for_command(
        app,
        command,
        seed=acceptance_transport_seed(
            scene_seed=scene_seed,
            scenario_id="single_line_source_resolved",
            shield_pair_id=0,
        ),
        dwell_time_s=1.0e-6,
    )
    executable = Path(str(app.config.executable_path)).resolve()
    variants = (
        "exact_surface_anchor",
        "air_plus_epsilon",
        "solid_minus_epsilon",
    )
    results: dict[str, subprocess.CompletedProcess[str]] = {}
    evidence: dict[str, str] = {}
    with tempfile.TemporaryDirectory(
        prefix="full_spectrum_boundary_gate_"
    ) as temporary:
        root = Path(temporary)
        base_scene_path = root / "base_scene.txt"
        request_path = root / "request.txt"
        write_scene_file(exported, base_scene_path)
        write_request_file(request, request_path)
        base_scene = base_scene_path.read_text(encoding="utf-8")
        for variant in variants:
            scene_path = root / f"scene_{variant}.txt"
            response_path = root / f"response_{variant}.txt"
            scene_text = _mutated_surface_scene(
                base_scene,
                variant=variant,
            )
            scene_path.write_text(scene_text, encoding="utf-8")
            command_line = [
                executable.as_posix(),
                "--scene",
                scene_path.as_posix(),
                "--request",
                request_path.as_posix(),
                "--response",
                response_path.as_posix(),
                "--physics-profile",
                app.config.physics_profile,
                "--threads",
                str(app.config.thread_count),
                "--dead-time-tau-s",
                str(app.config.dead_time_tau_s),
                "--source-rate-model",
                app.config.source_rate_model,
                "--source-bias-cone-half-angle-deg",
                str(app.config.source_bias_cone_half_angle_deg),
                "--detector-scoring-mode",
                app.config.detector_scoring_mode,
                "--secondary-transport-mode",
                app.config.secondary_transport_mode,
                "--primary-sampling-fraction",
                str(app.config.primary_sampling_fraction),
                "--background-cps",
                str(app.config.background_cps),
                "--sample-detector-response",
                "--validation-entry-class-spectra",
                *app.config.executable_args,
            ]
            result = subprocess.run(
                command_line,
                text=True,
                capture_output=True,
                timeout=min(float(app.config.timeout_s), 600.0),
                check=False,
            )
            results[variant] = result
            evidence[variant] = canonical_json_sha256(
                {
                    "variant": variant,
                    "scene_sha256": hashlib.sha256(
                        scene_text.encode("utf-8")
                    ).hexdigest(),
                    "request_sha256": _file_sha256(request_path),
                    "returncode": int(result.returncode),
                    "stdout_sha256": hashlib.sha256(
                        result.stdout.encode("utf-8")
                    ).hexdigest(),
                    "stderr_sha256": hashlib.sha256(
                        result.stderr.encode("utf-8")
                    ).hexdigest(),
                    "response_sha256": (
                        _file_sha256(response_path)
                        if response_path.exists()
                        else None
                    ),
                }
            )
    air_ok = results["air_plus_epsilon"].returncode == 0
    exact_error = (
        results["exact_surface_anchor"].returncode != 0
        and _BOUNDARY_ERROR_MARKER
        in (
            results["exact_surface_anchor"].stdout
            + results["exact_surface_anchor"].stderr
        )
    )
    solid_error = (
        results["solid_minus_epsilon"].returncode != 0
        and _BOUNDARY_ERROR_MARKER
        in (
            results["solid_minus_epsilon"].stdout
            + results["solid_minus_epsilon"].stderr
        )
    )
    passed = air_ok and exact_error and solid_error
    if not passed:
        raise RuntimeError(
            "Native signed-epsilon surface-boundary probes failed."
        )
    return {
        "schema_version": 1,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "native_position_variants": list(variants),
        "evidence_sha256_by_variant": evidence,
        "exact_anchor_vs_air_gate_passed": air_ok and exact_error,
        "solid_minus_air_gate_passed": air_ok and solid_error,
        "passed": passed,
    }


class _NativeScenarioSession(AcceptanceScenarioSession):
    """Acquire all pair observations from one cached exported native scene."""

    def __init__(
        self,
        *,
        app: Geant4Application,
        scene_seed: int,
        split: str,
        scenario_id: str,
        sources: tuple[PointSource, ...],
        model: GeometryConditionedSpectralModel,
        geometry: _GeometryBatch,
        perturbed_geometry: _GeometryBatch | None,
        boundary_gate: Mapping[str, object],
    ) -> None:
        """Store immutable acquisition state for one scenario."""
        self.app = app
        self.scene_seed = scene_seed
        self.split = split
        self.scenario_id = scenario_id
        self.sources = sources
        self.model = model
        self.geometry = geometry
        self.perturbed_geometry = perturbed_geometry
        self.boundary_gate = json.loads(
            json.dumps(dict(boundary_gate), allow_nan=False)
        )
        exported = getattr(app.engine, "scene", None)
        if exported is None:
            raise RuntimeError("Acceptance native scene was not exported.")
        self.scene_hash = str(exported.scene_hash)
        self.source_payloads = tuple(
            _source_payload(source) for source in sources
        )
        self.source_hash = surface_source_runtime_contract_sha256(
            self.source_payloads
        )

    def acquire_pair(self, shield_pair_id: int) -> Mapping[str, object]:
        """Run one exact native pair and return its authenticated artifact."""
        if (
            isinstance(shield_pair_id, bool)
            or not isinstance(shield_pair_id, int)
            or shield_pair_id not in ACCEPTANCE_PAIR_IDS
        ):
            raise ValueError("shield_pair_id must belong to the exact all-64.")
        command = _command_for_pair(shield_pair_id)
        seed = acceptance_transport_seed(
            scene_seed=self.scene_seed,
            scenario_id=self.scenario_id,
            shield_pair_id=shield_pair_id,
        )
        request = _request_for_command(
            self.app,
            command,
            seed=seed,
            dwell_time_s=ACCEPTANCE_DWELL_TIME_S,
        )
        spectrum, raw_metadata = self.app.engine.simulate(request)
        metadata = dict(raw_metadata)
        observed = _integer_spectrum(
            spectrum,
            field_name="observed_spectrum_counts",
        )
        fidelity = _native_fidelity(
            metadata,
            config=self.app.config,
            source_count=len(self.sources),
        )
        labels = _validation_labels(
            metadata,
            sources=self.sources,
            model=self.model,
        )
        index = int(shield_pair_id)
        if self.sources:
            geometry: dict[str, object] = {
                "unattenuated_source_line_rate_vsl": (
                    self.geometry.unattenuated_vsl[index : index + 1].tolist()
                ),
                "uncollided_source_line_rate_vsl": (
                    self.geometry.uncollided_vsl[index : index + 1].tolist()
                ),
                "transport_features_vslf": (
                    self.geometry.features_vslf[index : index + 1].tolist()
                ),
                "additive_scatter_basis_vslf": (
                    self.geometry.scatter_basis_vslf[
                        index : index + 1
                    ].tolist()
                ),
            }
        else:
            geometry = {
                "unattenuated_source_line_rate_vsl": None,
                "uncollided_source_line_rate_vsl": None,
                "transport_features_vslf": None,
                "additive_scatter_basis_vslf": None,
            }
        if self.perturbed_geometry is None:
            geometry.update(
                {
                    "perturbed_unattenuated_source_line_rate_vsl": None,
                    "perturbed_uncollided_source_line_rate_vsl": None,
                    "perturbed_transport_features_vslf": None,
                    "perturbed_additive_scatter_basis_vslf": None,
                }
            )
        else:
            geometry.update(
                {
                    "perturbed_unattenuated_source_line_rate_vsl": (
                        self.perturbed_geometry.unattenuated_vsl[
                            index : index + 1
                        ].tolist()
                    ),
                    "perturbed_uncollided_source_line_rate_vsl": (
                        self.perturbed_geometry.uncollided_vsl[
                            index : index + 1
                        ].tolist()
                    ),
                    "perturbed_transport_features_vslf": (
                        self.perturbed_geometry.features_vslf[
                            index : index + 1
                        ].tolist()
                    ),
                    "perturbed_additive_scatter_basis_vslf": (
                        self.perturbed_geometry.scatter_basis_vslf[
                            index : index + 1
                        ].tolist()
                    ),
                }
            )
        return {
            "schema_version": 1,
            "acceptance_contract_sha256": (
                FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            ),
            "scene_seed": self.scene_seed,
            "split": self.split,
            "scenario_id": self.scenario_id,
            "shield_pair_id": shield_pair_id,
            "transport_seed": seed,
            "dwell_time_s": ACCEPTANCE_DWELL_TIME_S,
            "scene_hash": self.scene_hash,
            "surface_source_contract_sha256": self.source_hash,
            "surface_boundary_gate": dict(self.boundary_gate),
            "detector_pose_xyz": list(request.detector_pose_xyz),
            "sources": list(self.source_payloads),
            "line_identity_contract_sha256": canonical_json_sha256(
                [dict(row) for row in self.model.line_identity]
            ),
            "observed_spectrum_counts": observed.tolist(),
            "geometry": geometry,
            "validation_labels": labels,
            "native_fidelity": fidelity,
        }

    def close(self) -> None:
        """Release the persistent native process and mock stage."""
        self.app.close()


class _ScenarioContext(AbstractContextManager[AcceptanceScenarioSession]):
    """Close one native scenario even when acquisition fails."""

    def __init__(self, session: _NativeScenarioSession) -> None:
        """Store the owned scenario session."""
        self.session = session

    def __enter__(self) -> AcceptanceScenarioSession:
        """Return the live native scenario."""
        return self.session

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the live native scenario."""
        del exc_type, exc, traceback
        self.session.close()


class ExternalGeant4AcceptanceBackend(AcceptanceTransportBackend):
    """Build deterministic random scenes and run real external Geant4."""

    backend_id = "external_geant4_all64_v1"

    def __init__(
        self,
        *,
        runtime_config_path: str | Path,
        repository_root: str | Path,
    ) -> None:
        """Load and authenticate the standard native runtime configuration."""
        self.repository_root = Path(repository_root).resolve()
        self.runtime_config_path = Path(runtime_config_path).resolve()
        self.runtime_config = load_runtime_config(self.runtime_config_path)
        app_payload = dict(self.runtime_config)
        app_payload["validation_entry_class_spectra"] = True
        executable_raw = app_payload.get(
            "executable_path",
            "build/geant4_sidecar",
        )
        if not isinstance(executable_raw, str) or not executable_raw:
            raise ValueError("Native executable_path must be a nonempty string.")
        executable = Path(executable_raw)
        if not executable.is_absolute():
            executable = (self.repository_root / executable).resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"Native Geant4 executable is missing: {executable}."
            )
        app_payload["executable_path"] = executable.as_posix()
        self.app_payload = app_payload
        self.app_config = Geant4AppConfig.from_dict(app_payload)
        if (
            self.app_config.engine_mode != "external"
            or self.app_config.physics_profile != "balanced"
            or self.app_config.thread_count != 32
            or self.app_config.primary_sampling_fraction != 1.0
            or self.app_config.target_sampled_primaries is not None
            or self.app_config.accelerated_weighted_transport_enable
            or self.app_config.source_rate_model != "detector_cps_1m"
            or self.app_config.detector_scoring_mode
            != "incident_gamma_energy"
            or self.app_config.secondary_transport_mode != "full_transport"
            or not self.app_config.sample_detector_response
            or not self.app_config.validation_entry_class_spectra
            or self.app_config.author_obstacle_prims is not True
            or self.app_config.author_room_boundary_prims is not True
        ):
            raise ValueError(
                "Acceptance requires external balanced Geant4, 32 threads, "
                "full unit-weight histories, full secondary transport, and "
                "native detector-response sampling."
            )
        self.runtime_config_sha256 = _file_sha256(self.runtime_config_path)
        self.native_executable_sha256 = _file_sha256(executable)
        self._boundary_gate_by_seed: dict[int, Mapping[str, object]] = {}

    def _room_boundary_thickness_m(self) -> float:
        """Return the strict configured room-boundary thickness."""
        raw = self.runtime_config.get("room_boundary_thickness_m", 0.1)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("room_boundary_thickness_m must be a JSON number.")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "room_boundary_thickness_m must be finite and positive."
            )
        return value

    def _obstacle_material(self) -> str:
        """Return the strict material used by the authored obstacle prims."""
        raw = self.runtime_config.get("obstacle_material", "concrete")
        if not isinstance(raw, str) or not raw:
            raise TypeError("obstacle_material must be a nonempty string.")
        return raw

    def _kernel(
        self,
        grid: ObstacleGrid,
    ) -> object:
        """Build the shared batched PF geometry kernel without a fitted model."""
        observation_payload = {
            key: value
            for key, value in self.runtime_config.items()
            if key not in _FULL_SPECTRUM_RUNTIME_KEYS
        }
        use_gpu = observation_payload.get("use_gpu", False)
        if not isinstance(use_gpu, bool):
            raise TypeError("Acceptance use_gpu must be a JSON boolean.")
        observation = build_runtime_observation_model(
            observation_payload,
            isotopes=ACCEPTANCE_ISOTOPES,
        )
        if observation.additive_scatter_response is not None:
            raise RuntimeError(
                "Acceptance geometry must precede fitted additive scatter."
            )
        kernel = continuous_kernel_from_observation_model(
            observation,
            obstacle_grid=grid,
            use_gpu=use_gpu,
        )
        gpu_device = observation_payload.get("gpu_device", "cuda")
        gpu_dtype = observation_payload.get("gpu_dtype", "float64")
        if (
            not isinstance(gpu_device, str)
            or not gpu_device
            or not isinstance(gpu_dtype, str)
            or not gpu_dtype
        ):
            raise ValueError("Acceptance GPU device/dtype must be strings.")
        kernel.gpu_device = gpu_device
        kernel.gpu_dtype = gpu_dtype
        return kernel

    def _app_for_scene(
        self,
        *,
        grid: ObstacleGrid,
        instances: Sequence[object],
        sources: Sequence[PointSource],
    ) -> Geant4Application:
        """Create one native app and load one exact generated scene."""
        app = Geant4Application(app_config=dict(self.app_payload))
        app.reset(
            build_scene_description(
                _scene_payload(
                    grid=grid,
                    instances=instances,
                    sources=sources,
                    author_room_boundaries=bool(
                        self.app_config.author_room_boundary_prims
                    ),
                    obstacle_material=self._obstacle_material(),
                )
            )
        )
        return app

    def _boundary_gate(
        self,
        *,
        scene_seed: int,
        environment: EnvironmentConfig,
        grid: ObstacleGrid,
        instances: Sequence[object],
    ) -> Mapping[str, object]:
        """Return the one cached actual-native gate for a scene seed."""
        cached = self._boundary_gate_by_seed.get(scene_seed)
        if cached is not None:
            return cached
        probe = generate_surface_sources(
            env=environment,
            obstacle_grid=grid,
            isotopes=("Cs-137",),
            intensity_cps_1m=300_000.0,
            rng=named_random_generator(
                scene_seed,
                _BOUNDARY_RNG_DOMAIN,
            ),
            count=1,
            obstacle_height_m=float(self.app_config.obstacle_height_m),
            chart_max_edge_m=ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
        )
        app = self._app_for_scene(
            grid=grid,
            instances=instances,
            sources=probe,
        )
        try:
            gate = _surface_boundary_gate(app=app, scene_seed=scene_seed)
        finally:
            app.close()
        self._boundary_gate_by_seed[scene_seed] = gate
        return gate

    def open_scenario(
        self,
        *,
        scene_seed: int,
        split: str,
        scenario_id: str,
        line_identity_sha256: str,
    ) -> AbstractContextManager[AcceptanceScenarioSession]:
        """Open one deterministic cached-scene native all-64 session."""
        if scenario_id not in VALIDATION_SCENARIO_IDS:
            raise ValueError("Unknown fixed acceptance scenario.")
        environment, grid, instances = _build_environment(
            scene_seed=scene_seed,
            obstacle_height_m=float(self.app_config.obstacle_height_m),
            author_room_boundaries=bool(
                self.app_config.author_room_boundary_prims
            ),
            room_boundary_thickness_m=float(
                self._room_boundary_thickness_m()
            ),
        )
        sources = _generate_sources(
            environment=environment,
            grid=grid,
            scene_seed=scene_seed,
            scenario_id=scenario_id,
            obstacle_height_m=float(self.app_config.obstacle_height_m),
        )
        model = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=float(self.app_config.dead_time_tau_s),
            background_rate_cps=float(self.app_config.background_cps),
        )
        actual_line_hash = canonical_json_sha256(
            [dict(row) for row in model.line_identity]
        )
        if actual_line_hash != line_identity_sha256:
            raise ValueError("Acceptance line identity differs from the runner.")
        kernel = self._kernel(grid)
        geometry = _geometry_batch(
            kernel=kernel,
            model=model,
            detector_pose_xyz=ACCEPTANCE_DETECTOR_POSE_XYZ,
            sources=sources,
        )
        perturbed_geometry = None
        if scenario_id == "continuous_surface_perturbation_ranking":
            perturbation = _perturbed_sources(
                environment=environment,
                grid=grid,
                sources=sources,
                obstacle_height_m=float(self.app_config.obstacle_height_m),
            )
            perturbed_geometry = _geometry_batch(
                kernel=kernel,
                model=model,
                detector_pose_xyz=ACCEPTANCE_DETECTOR_POSE_XYZ,
                sources=perturbation,
            )
        gate = self._boundary_gate(
            scene_seed=scene_seed,
            environment=environment,
            grid=grid,
            instances=instances,
        )
        app = self._app_for_scene(
            grid=grid,
            instances=instances,
            sources=sources,
        )
        return _ScenarioContext(
            _NativeScenarioSession(
                app=app,
                scene_seed=scene_seed,
                split=split,
                scenario_id=scenario_id,
                sources=sources,
                model=model,
                geometry=geometry,
                perturbed_geometry=perturbed_geometry,
                boundary_gate=gate,
            )
        )


__all__ = [
    "ACCEPTANCE_DETECTOR_POSE_XYZ",
    "ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION",
    "ACCEPTANCE_PASSAGE_WIDTH_M",
    "ACCEPTANCE_ROOM_SIZE_XYZ",
    "ExternalGeant4AcceptanceBackend",
]
