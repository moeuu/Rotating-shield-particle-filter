"""Validate Geant4 spectrum decomposition across multi-source cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.obstacles import ObstacleGrid
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from sim.shield_geometry import resolve_shield_thickness_config
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig


ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


@dataclass(frozen=True)
class ValidationSource:
    """Describe a source used by a validation case."""

    isotope: str
    position_xyz: tuple[float, float, float]
    intensity_cps_1m: float

    def to_point_source(self) -> PointSource:
        """Convert this source to the measurement-model representation."""
        return PointSource(
            isotope=self.isotope,
            position=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )

    def to_scene_source(self) -> SourceDescription:
        """Convert this source to the Geant4 scene representation."""
        return SourceDescription(
            isotope=self.isotope,
            position_xyz=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )


@dataclass(frozen=True)
class ValidationCase:
    """Describe one Geant4 spectrum-decomposition validation case."""

    name: str
    description: str
    detector_pose_xyz: tuple[float, float, float]
    sources: tuple[ValidationSource, ...]
    fe_index: int = 0
    pb_index: int = 0
    dwell_time_s: float = 30.0
    obstacle_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    include_in_accuracy_summary: bool = True


def _clamp_room_position(
    position: np.ndarray,
    *,
    room_size_xyz: tuple[float, float, float],
    margin_xy_m: float = 0.75,
) -> tuple[float, float, float]:
    """Clamp a candidate source position inside the validation room."""
    room = np.asarray(room_size_xyz, dtype=float)
    pos = np.asarray(position, dtype=float).copy()
    pos[0] = float(np.clip(pos[0], margin_xy_m, room[0] - margin_xy_m))
    pos[1] = float(np.clip(pos[1], margin_xy_m, room[1] - margin_xy_m))
    pos[2] = float(np.clip(pos[2], 0.5, min(room[2] - 0.5, 2.0)))
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _line_obstacle_cells(
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
    *,
    grid_shape: tuple[int, int] = (10, 20),
    max_cells: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Return obstacle cells sampled along a source-detector line segment."""
    source = np.asarray(source_xyz, dtype=float)
    detector = np.asarray(detector_xyz, dtype=float)
    cells: list[tuple[int, int]] = []
    endpoint_cells = {
        (int(np.floor(source[0])), int(np.floor(source[1]))),
        (int(np.floor(detector[0])), int(np.floor(detector[1]))),
    }
    for fraction in np.linspace(0.25, 0.75, 5):
        point = source * (1.0 - float(fraction)) + detector * float(fraction)
        cell = (int(np.floor(point[0])), int(np.floor(point[1])))
        if cell in endpoint_cells or cell in cells:
            continue
        if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]:
            cells.append(cell)
        if len(cells) >= int(max_cells):
            break
    return tuple(cells)


def _source_isotope_program(case_index: int, source_count: int) -> list[str]:
    """Return a deterministic isotope assignment with single and mixed cases."""
    programs = (
        ("Cs-137",),
        ("Co-60",),
        ("Eu-154",),
        ("Cs-137", "Co-60"),
        ("Cs-137", "Eu-154"),
        ("Co-60", "Eu-154"),
        ("Cs-137", "Co-60", "Eu-154"),
        ("Cs-137", "Cs-137", "Co-60"),
        ("Co-60", "Co-60", "Eu-154"),
        ("Cs-137", "Eu-154", "Eu-154"),
    )
    base = list(programs[int(case_index) % len(programs)])
    while len(base) < int(source_count):
        base.append(ISOTOPES[(int(case_index) + len(base)) % len(ISOTOPES)])
    return base[: int(source_count)]


def generated_cases(
    *,
    num_cases: int = 50,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = 30000.0,
    intensity_max_cps_1m: float = 90000.0,
) -> list[ValidationCase]:
    """Return deterministic Geant4 validation cases spanning sources, shielding, and obstacles."""
    rng = np.random.default_rng(int(seed))
    room_size = (10.0, 20.0, 10.0)
    detector_positions = (
        (1.5, 1.5, 0.5),
        (2.5, 4.5, 0.5),
        (5.0, 10.0, 0.5),
        (7.5, 15.5, 0.5),
        (8.5, 5.5, 0.5),
    )
    directions = np.asarray(
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, -1.0, 0.0),
            (1.0, 1.0, 0.7),
            (-1.0, 1.0, 0.7),
            (1.0, -1.0, 0.7),
            (-1.0, -1.0, 0.7),
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = (1.0, 1.5, 2.2, 3.0, 4.0, 5.2, 6.5)
    cases: list[ValidationCase] = []
    for case_index in range(max(0, int(num_cases))):
        detector = np.asarray(detector_positions[case_index % len(detector_positions)], dtype=float)
        source_count = 1 + (case_index % 5)
        isotope_program = _source_isotope_program(case_index, source_count)
        sources: list[ValidationSource] = []
        for source_index in range(source_count):
            direction = directions[(case_index * 3 + source_index * 5) % len(directions)]
            distance = float(distances[(case_index + source_index * 2) % len(distances)])
            jitter = rng.normal(0.0, 0.18, size=3)
            jitter[2] *= 0.35
            position = _clamp_room_position(
                detector + distance * direction + jitter,
                room_size_xyz=room_size,
            )
            intensity = float(
                rng.uniform(float(intensity_min_cps_1m), float(intensity_max_cps_1m))
            )
            sources.append(
                ValidationSource(
                    isotope=isotope_program[source_index],
                    position_xyz=position,
                    intensity_cps_1m=intensity,
                )
            )
        obstacle_cells: tuple[tuple[int, int], ...] = ()
        obstacle_mode = case_index % 4
        if obstacle_mode in (1, 2, 3):
            line_cells: list[tuple[int, int]] = []
            for source in sources[: 1 + (obstacle_mode == 3)]:
                line_cells.extend(
                    _line_obstacle_cells(
                        source.position_xyz,
                        tuple(float(v) for v in detector),
                        max_cells=1 + obstacle_mode,
                    )
                )
            obstacle_cells = tuple(sorted(set(line_cells)))
        cases.append(
            ValidationCase(
                name=f"g4_case_{case_index:02d}",
                description=(
                    f"{source_count} source(s), {len(set(isotope_program))} isotope(s), "
                    f"obstacle_cells={len(obstacle_cells)}"
                ),
                detector_pose_xyz=tuple(float(v) for v in detector),
                sources=tuple(sources),
                fe_index=(case_index * 3) % 8,
                pb_index=(case_index * 5 + 1) % 8,
                dwell_time_s=float(dwell_time_s),
                obstacle_cells=obstacle_cells,
                include_in_accuracy_summary=True,
            )
        )
    return cases


def default_cases() -> list[ValidationCase]:
    """Return a small hand-authored compatibility set of Geant4 validation cases."""
    detector = (1.0, 1.0, 0.5)
    near_x = (2.0, 1.0, 0.5)
    near_y = (1.0, 2.0, 0.5)
    near_z = (1.0, 1.0, 1.5)
    far_y = (1.0, 3.0, 0.5)
    blocked_octant = (2.0, 2.0, 1.5)
    free_other_octant = (0.0, 2.0, 1.5)
    return [
        ValidationCase(
            name="single_cs_free",
            description="Single Cs-137 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Cs-137", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_co_free",
            description="Single Co-60 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Co-60", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_eu_free",
            description="Single Eu-154 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Eu-154", near_x, 100.0),),
        ),
        ValidationCase(
            name="two_cs_free",
            description="Two Cs-137 sources at different distances.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 70.0),
                ValidationSource("Cs-137", far_y, 120.0),
            ),
        ),
        ValidationCase(
            name="cs_co_free",
            description="Two-isotope mixture with Cs-137 and Co-60.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Co-60", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_eu_free",
            description="Two-isotope mixture with Cs-137 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="co_eu_free",
            description="Two-isotope mixture with Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Co-60", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_balanced_free",
            description="Balanced three-isotope mixture at three directions.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 60.0),
                ValidationSource("Co-60", near_y, 60.0),
                ValidationSource("Eu-154", near_z, 60.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_imbalanced_free",
            description="Three-isotope mixture with weak Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 120.0),
                ValidationSource("Co-60", near_y, 35.0),
                ValidationSource("Eu-154", near_z, 25.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_fe_blocked",
            description="Three-isotope mixture through the iron shield octant.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Co-60", blocked_octant, 80.0),
                ValidationSource("Eu-154", blocked_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_two_sources_one_fe_blocked",
            description="Two Cs-137 sources with one direction blocked by Fe.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Cs-137", free_other_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_obstacle_stress",
                description="Three-isotope mixture with a concrete obstacle.",
            detector_pose_xyz=(1.0, 1.5, 0.5),
            dwell_time_s=20.0,
            obstacle_cells=((2, 1),),
            include_in_accuracy_summary=True,
            sources=(
                ValidationSource("Cs-137", (4.0, 1.5, 0.5), 80.0),
                ValidationSource("Co-60", (4.0, 1.5, 1.5), 80.0),
                ValidationSource("Eu-154", (4.0, 1.5, 0.8), 80.0),
            ),
        ),
    ]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def build_scene(case: ValidationCase, usd_path: str | None) -> SceneDescription:
    """Build a Geant4 scene for one validation case."""
    return SceneDescription(
        room_size_xyz=(10.0, 20.0, 10.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[tuple(cell) for cell in case.obstacle_cells],
        author_obstacle_prims=True,
        sources=[source.to_scene_source() for source in case.sources],
        usd_path=usd_path,
    )


def spectrum_config_from_runtime_config(runtime_config: dict[str, Any]) -> SpectrumConfig:
    """Build the spectrum configuration used by the runtime count extractor."""
    config = SpectrumConfig()
    scoring_mode = str(runtime_config.get("detector_scoring_mode", "")).strip().lower()
    source_rate_model = str(runtime_config.get("source_rate_model", "")).strip().lower()
    if scoring_mode == "incident_gamma_energy" and "response_efficiency_model" not in runtime_config:
        config.response_efficiency_model = "unit"
    if scoring_mode == "incident_gamma_energy":
        config.use_incident_gamma_response_matrix = True
    if source_rate_model == "detector_cps_1m":
        config.normalize_line_intensities = True
    field_names = set(SpectrumConfig.__dataclass_fields__.keys())
    for key, value in runtime_config.items():
        if key not in field_names or value is None:
            continue
        current = getattr(config, key)
        if isinstance(current, bool):
            setattr(config, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(config, key, int(value))
        elif isinstance(current, float):
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)
    config.__post_init__()
    return config


def analysis_spectrum_from_observation(
    raw_spectrum: np.ndarray,
    metadata: dict[str, Any],
    decomposer: SpectralDecomposer,
) -> np.ndarray:
    """Return the pulse-height spectrum used by the runtime spectrum decomposer."""
    spectrum = np.asarray(raw_spectrum, dtype=float)
    scoring_mode = str(metadata.get("detector_scoring_mode", "")).strip().lower()
    fast_scoring = str(metadata.get("detector_fast_scoring", "")).strip().lower()
    should_fold = (
        bool(decomposer.config.apply_incident_gamma_detector_response)
        and (scoring_mode == "incident_gamma_energy" or fast_scoring == "true")
    )
    if should_fold:
        return decomposer.fold_incident_gamma_spectrum(spectrum)
    return spectrum


def obstacle_grid_for_case(case: ValidationCase) -> ObstacleGrid | None:
    """Return the obstacle grid used by the analytic target for a case."""
    if not case.obstacle_cells:
        return None
    return ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=tuple(case.obstacle_cells),
    )


def shield_params_from_runtime_config(runtime_config: dict[str, Any]) -> ShieldParams:
    """Return PF-side shield parameters matching the Geant4 runtime config."""
    shield_thickness = resolve_shield_thickness_config(runtime_config)
    buildup = runtime_config.get("pf_buildup", {})
    if not isinstance(buildup, dict):
        buildup = {}
    return ShieldParams(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
        buildup_fe_coeff=float(
            buildup.get(
                "fe_coeff",
                runtime_config.get("pf_buildup_fe_coeff", 0.0),
            )
        ),
        buildup_pb_coeff=float(
            buildup.get(
                "pb_coeff",
                runtime_config.get("pf_buildup_pb_coeff", 0.0),
            )
        ),
    )


def obstacle_buildup_coeff_from_runtime_config(runtime_config: dict[str, Any]) -> float:
    """Return the PF-side obstacle broad-beam build-up coefficient."""
    buildup = runtime_config.get("pf_buildup", {})
    if not isinstance(buildup, dict):
        buildup = {}
    return float(
        buildup.get(
            "obstacle_coeff",
            runtime_config.get("pf_obstacle_buildup_coeff", 0.0),
        )
    )


def kernel_for_case(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
) -> ContinuousKernel:
    """Build the inverse-square plus shield/obstacle attenuation target kernel."""
    detector_model = runtime_config.get("detector_model", {})
    if not isinstance(detector_model, dict):
        detector_model = {}
    detector_radius_m = float(detector_model.get("crystal_radius_m", 0.0)) + float(
        detector_model.get("housing_thickness_m", 0.0)
    )
    return ContinuousKernel(
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params_from_runtime_config(runtime_config),
        obstacle_grid=obstacle_grid_for_case(case),
        obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
        obstacle_buildup_coeff=obstacle_buildup_coeff_from_runtime_config(runtime_config),
        detector_radius_m=detector_radius_m,
        detector_aperture_samples=int(runtime_config.get("pf_detector_aperture_samples", 121)),
        use_gpu=False,
    )


def expected_pf_counts(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
) -> dict[str, float]:
    """Compute inverse-square plus shield/obstacle attenuation target counts."""
    kernel = kernel_for_case(case, runtime_config, mu_by_isotope)
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    counts = {isotope: 0.0 for isotope in ISOTOPES}
    for source in case.sources:
        point_source = source.to_point_source()
        source_pos = point_source.position_array()
        counts[source.isotope] += (
            float(case.dwell_time_s)
            * float(source.intensity_cps_1m)
            * kernel.kernel_value_pair(
                source.isotope,
                detector,
                source_pos,
                int(case.fe_index),
                int(case.pb_index),
            )
        )
    return counts


def source_tally_counts(metadata: dict[str, Any]) -> dict[str, float]:
    """Read native Geant4 source-equivalent tally counts from metadata."""
    return {
        isotope: float(metadata.get(f"source_equivalent_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }


def transport_truth_counts(metadata: dict[str, Any]) -> dict[str, float]:
    """Read native Geant4 isotope-labeled counts after transport to the detector."""
    return {
        isotope: float(metadata.get(f"transport_detected_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }


def relative_error(value: float, target: float, min_target: float) -> float | None:
    """Return relative error when the target is large enough."""
    if abs(float(target)) < float(min_target):
        return None
    return (float(value) - float(target)) / float(target)


def case_to_dict(case: ValidationCase) -> dict[str, Any]:
    """Return a JSON-compatible representation of a validation case."""
    return {
        "name": case.name,
        "description": case.description,
        "detector_pose_xyz": list(case.detector_pose_xyz),
        "fe_index": int(case.fe_index),
        "pb_index": int(case.pb_index),
        "dwell_time_s": float(case.dwell_time_s),
        "obstacle_cells": [list(cell) for cell in case.obstacle_cells],
        "include_in_accuracy_summary": bool(case.include_in_accuracy_summary),
        "sources": [
            {
                "isotope": source.isotope,
                "position_xyz": list(source.position_xyz),
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in case.sources
        ],
    }


def run_case(
    app: Geant4Application,
    decomposer: SpectralDecomposer,
    case: ValidationCase,
    step_id: int,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
    min_target: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one Geant4 validation case and return metrics plus spectrum."""
    scene = build_scene(case, usd_path=runtime_config.get("usd_path"))
    app.reset(scene)
    start = time.time()
    observation = app.step(
        SimulationCommand(
            step_id=step_id,
            target_pose_xyz=case.detector_pose_xyz,
            target_base_yaw_rad=0.0,
            fe_orientation_index=int(case.fe_index),
            pb_orientation_index=int(case.pb_index),
            dwell_time_s=float(case.dwell_time_s),
        )
    )
    runtime_s = time.time() - start
    raw_spectrum = np.asarray(observation.spectrum_counts, dtype=float)
    spectrum = analysis_spectrum_from_observation(
        raw_spectrum,
        dict(observation.metadata),
        decomposer,
    )
    response_poisson_counts = decomposer.compute_response_poisson_counts(
        spectrum,
        isotopes=ISOTOPES,
        include_background=True,
        live_time_s=float(case.dwell_time_s),
    )
    response_poisson_variances = {
        isotope: float(decomposer.last_count_variances.get(isotope, 1.0))
        for isotope in ISOTOPES
    }
    photopeak_counts = decomposer.compute_photopeak_nnls_counts(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    response_counts = decomposer.compute_response_model_counts(
        spectrum,
        isotopes=ISOTOPES,
    )
    peak_window_counts = decomposer.compute_isotope_counts_thesis(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    target_counts = expected_pf_counts(case, runtime_config, mu_by_isotope)
    tally_counts = source_tally_counts(dict(observation.metadata))
    truth_counts = transport_truth_counts(dict(observation.metadata))
    methods = {
        "response_poisson": response_poisson_counts,
        "photopeak_nnls": photopeak_counts,
        "response_matrix": response_counts,
        "peak_window": peak_window_counts,
    }
    per_isotope: dict[str, dict[str, Any]] = {}
    for isotope in ISOTOPES:
        target = float(target_counts.get(isotope, 0.0))
        per_isotope[isotope] = {
            "target_pf_counts": target,
            "source_tally_counts": float(tally_counts.get(isotope, 0.0)),
            "transport_truth_counts": float(truth_counts.get(isotope, 0.0)),
            "method_counts": {
                method: float(values.get(isotope, 0.0))
                for method, values in methods.items()
            },
            "response_poisson_variance": float(response_poisson_variances.get(isotope, 1.0)),
            "relative_errors": {
                method: relative_error(
                    float(values.get(isotope, 0.0)),
                    target,
                    min_target,
                )
                for method, values in methods.items()
            },
            "relative_errors_vs_transport_truth": {
                method: relative_error(
                    float(values.get(isotope, 0.0)),
                    float(truth_counts.get(isotope, 0.0)),
                    min_target,
                )
                for method, values in methods.items()
            },
            "pf_target_relative_error_vs_transport_truth": relative_error(
                target,
                float(truth_counts.get(isotope, 0.0)),
                min_target,
            ),
        }
    result = {
        "case": case_to_dict(case),
        "runtime_s": float(runtime_s),
        "raw_total_spectrum_counts": float(np.sum(raw_spectrum)),
        "total_spectrum_counts": float(np.sum(spectrum)),
        "num_primaries": float(observation.metadata.get("num_primaries", 0.0)),
        "metadata": {
            key: observation.metadata[key]
            for key in sorted(observation.metadata)
            if str(key).startswith(
                    (
                        "backend",
                        "engine_mode",
                        "emission_model",
                        "physics_profile",
                        "source_equivalent",
                        "transport_detected",
                        "num_primaries",
                        "primary_sampling",
                        "weighted",
                        "detector_scoring",
                    "detector_fast",
                    "runtime_s",
                    "run_time_s",
                        "thread_count",
                        "total_track_steps",
                        "detector_hit_events",
                        "process_count",
                        "secondary_count",
                        "total_spectrum_counts",
                        "volume_step_counts",
                )
            )
        },
        "per_isotope": per_isotope,
    }
    return result, spectrum


def flatten_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested validation results into CSV records."""
    rows: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        for isotope, item in result["per_isotope"].items():
            for method, value in item["method_counts"].items():
                rel_err = item["relative_errors"][method]
                rows.append(
                    {
                        "case": case["name"],
                        "description": case["description"],
                        "include_in_accuracy_summary": case["include_in_accuracy_summary"],
                        "isotope": isotope,
                        "method": method,
                        "target_pf_counts": item["target_pf_counts"],
                        "source_tally_counts": item["source_tally_counts"],
                        "transport_truth_counts": item["transport_truth_counts"],
                        "estimated_counts": value,
                        "relative_error": "" if rel_err is None else rel_err,
                        "abs_relative_error": "" if rel_err is None else abs(rel_err),
                        "relative_error_vs_transport_truth": ""
                        if item["relative_errors_vs_transport_truth"][method] is None
                        else item["relative_errors_vs_transport_truth"][method],
                        "abs_relative_error_vs_transport_truth": ""
                        if item["relative_errors_vs_transport_truth"][method] is None
                        else abs(item["relative_errors_vs_transport_truth"][method]),
                        "pf_target_relative_error_vs_transport_truth": ""
                        if item["pf_target_relative_error_vs_transport_truth"] is None
                        else item["pf_target_relative_error_vs_transport_truth"],
                        "total_spectrum_counts": result["total_spectrum_counts"],
                        "num_primaries": result["num_primaries"],
                        "runtime_s": result["runtime_s"],
                        "fe_index": case["fe_index"],
                        "pb_index": case["pb_index"],
                        "dwell_time_s": case["dwell_time_s"],
                    }
                )
    return rows


def summarize_accuracy(
    results: list[dict[str, Any]],
    min_target: float,
) -> dict[str, dict[str, float]]:
    """Summarize accuracy metrics by decomposition method."""
    values_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    false_positive_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    values_vs_truth_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    false_positive_vs_truth_by_method: dict[str, list[float]] = {
        "response_poisson": [],
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    pf_target_vs_truth: list[float] = []
    for result in results:
        if not bool(result["case"]["include_in_accuracy_summary"]):
            continue
        for item in result["per_isotope"].values():
            target = float(item["target_pf_counts"])
            truth = float(item["transport_truth_counts"])
            target_truth_err = item.get("pf_target_relative_error_vs_transport_truth")
            if target_truth_err is not None:
                pf_target_vs_truth.append(abs(float(target_truth_err)))
            for method, estimate in item["method_counts"].items():
                if target >= float(min_target):
                    err = relative_error(float(estimate), target, min_target)
                    if err is not None:
                        values_by_method[method].append(abs(err))
                else:
                    false_positive_by_method[method].append(float(estimate))
                truth_err = item["relative_errors_vs_transport_truth"][method]
                if truth >= float(min_target):
                    if truth_err is not None:
                        values_vs_truth_by_method[method].append(abs(float(truth_err)))
                else:
                    false_positive_vs_truth_by_method[method].append(float(estimate))

    summary: dict[str, dict[str, float]] = {}
    for method, values in values_by_method.items():
        arr = np.asarray(values, dtype=float)
        fp = np.asarray(false_positive_by_method[method], dtype=float)
        truth_arr = np.asarray(values_vs_truth_by_method[method], dtype=float)
        truth_fp = np.asarray(false_positive_vs_truth_by_method[method], dtype=float)
        summary[method] = {
            "num_accuracy_points": float(arr.size),
            "mean_abs_relative_error": float(np.mean(arr)) if arr.size else float("nan"),
            "median_abs_relative_error": float(np.median(arr)) if arr.size else float("nan"),
            "max_abs_relative_error": float(np.max(arr)) if arr.size else float("nan"),
            "num_absent_isotope_points": float(fp.size),
            "max_absent_isotope_counts": float(np.max(fp)) if fp.size else 0.0,
            "mean_absent_isotope_counts": float(np.mean(fp)) if fp.size else 0.0,
            "num_accuracy_points_vs_transport_truth": float(truth_arr.size),
            "mean_abs_relative_error_vs_transport_truth": (
                float(np.mean(truth_arr)) if truth_arr.size else float("nan")
            ),
            "median_abs_relative_error_vs_transport_truth": (
                float(np.median(truth_arr)) if truth_arr.size else float("nan")
            ),
            "max_abs_relative_error_vs_transport_truth": (
                float(np.max(truth_arr)) if truth_arr.size else float("nan")
            ),
            "num_absent_truth_isotope_points": float(truth_fp.size),
            "max_absent_truth_isotope_counts": float(np.max(truth_fp)) if truth_fp.size else 0.0,
            "mean_absent_truth_isotope_counts": float(np.mean(truth_fp)) if truth_fp.size else 0.0,
        }
    pf_arr = np.asarray(pf_target_vs_truth, dtype=float)
    summary["pf_theory_target_vs_transport_truth"] = {
        "num_accuracy_points": float(pf_arr.size),
        "mean_abs_relative_error": float(np.mean(pf_arr)) if pf_arr.size else float("nan"),
        "median_abs_relative_error": float(np.median(pf_arr)) if pf_arr.size else float("nan"),
        "max_abs_relative_error": float(np.max(pf_arr)) if pf_arr.size else float("nan"),
    }
    return summary


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    spectra: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    """Write validation outputs to JSON, CSV, and NPZ files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = flatten_records(results)
    csv_path = output_dir / "records.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(output_dir / "spectra.npz", **spectra)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/geant4/variance_reduction_external_no_isaac_32threads.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case", action="append", default=None, help="Run only the named case; repeatable.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--min-target-counts", type=float, default=25.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--thread-count", type=int, default=None)
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--case-seed", type=int, default=20260430)
    parser.add_argument("--dwell-time-s", type=float, default=30.0)
    parser.add_argument("--intensity-min-cps-1m", type=float, default=30000.0)
    parser.add_argument("--intensity-max-cps-1m", type=float, default=90000.0)
    parser.add_argument(
        "--hand-authored-cases",
        action="store_true",
        help="Use the legacy small hand-authored case set instead of generated 50-case validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Geant4 spectrum-decomposition validation sweep."""
    args = parse_args()
    config_path = resolve_path(args.config)
    runtime_config = load_runtime_config(config_path.as_posix())
    executable_path = runtime_config.get("executable_path", "build/geant4_sidecar")
    runtime_config["executable_path"] = resolve_path(str(executable_path)).as_posix()
    runtime_config["timeout_s"] = float(args.timeout_s)
    if args.thread_count is not None:
        runtime_config["thread_count"] = int(args.thread_count)
    runtime_config["engine_mode"] = "external"
    runtime_config["physics_profile"] = "balanced"

    if args.hand_authored_cases:
        all_cases = default_cases()
    else:
        all_cases = generated_cases(
            num_cases=int(args.num_cases),
            seed=int(args.case_seed),
            dwell_time_s=float(args.dwell_time_s),
            intensity_min_cps_1m=float(args.intensity_min_cps_1m),
            intensity_max_cps_1m=float(args.intensity_max_cps_1m),
        )
    if args.case:
        selected = set(args.case)
        cases = [case for case in all_cases if case.name in selected]
        missing = selected.difference({case.name for case in cases})
        if missing:
            raise ValueError(f"Unknown case names: {sorted(missing)}")
    else:
        cases = all_cases
    if args.max_cases is not None:
        cases = cases[: max(int(args.max_cases), 0)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "spectrum_validation" / f"geant4_photopeak_nnls_sweep_{timestamp}"
    )

    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=ISOTOPES)
    decomposer = SpectralDecomposer(spectrum_config_from_runtime_config(runtime_config))
    results: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    sweep_start = time.time()
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    try:
        for step_id, case in enumerate(cases):
            print(f"[{step_id + 1}/{len(cases)}] running {case.name}: {case.description}", flush=True)
            result, spectrum = run_case(
                app,
                decomposer,
                case,
                step_id,
                runtime_config,
                mu_by_isotope,
                float(args.min_target_counts),
            )
            results.append(result)
            spectra[case.name] = spectrum
            response = {
                isotope: result["per_isotope"][isotope]["method_counts"]["response_poisson"]
                for isotope in ISOTOPES
            }
            target = {
                isotope: result["per_isotope"][isotope]["target_pf_counts"]
                for isotope in ISOTOPES
            }
            truth = {
                isotope: result["per_isotope"][isotope]["transport_truth_counts"]
                for isotope in ISOTOPES
            }
            rel = {
                isotope: result["per_isotope"][isotope]["relative_errors"]["response_poisson"]
                for isotope in ISOTOPES
            }
            rel_truth = {
                isotope: result["per_isotope"][isotope]["relative_errors_vs_transport_truth"]["response_poisson"]
                for isotope in ISOTOPES
            }
            print(
                f"  primaries={result['num_primaries']:.0f} "
                f"runtime={result['runtime_s']:.1f}s target={target} truth={truth} "
                f"response_poisson={response} rel_err={rel} rel_truth={rel_truth}",
                flush=True,
            )
    finally:
        app.close()

    summary = {
        "config": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_cases": len(results),
        "elapsed_s": float(time.time() - sweep_start),
        "min_target_counts": float(args.min_target_counts),
        "accuracy_summary": summarize_accuracy(results, float(args.min_target_counts)),
        "cases": [case_to_dict(case) for case in cases],
    }
    write_outputs(output_dir, results, spectra, summary)
    print(json.dumps(summary["accuracy_summary"], indent=2, sort_keys=True))
    print(f"Wrote validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
