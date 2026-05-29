"""Validate Geant4 spectrum decomposition across multi-source cases."""
# ruff: noqa: E402

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
from measurement.continuous_kernels import finite_sphere_geometric_term
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.obstacle_assets import (
    KnownObstacleInstance,
    generate_manchester_obstacle_instances,
    known_obstacle_transport_model,
    obstacle_instances_to_dicts,
)
from measurement.obstacles import ObstacleGrid
from measurement.obstacles import build_obstacle_grid
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from sim.shield_geometry import nested_shield_inner_radii_cm
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
    obstacle_instances: tuple[KnownObstacleInstance, ...] = field(default_factory=tuple)
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


def _attach_known_obstacle_transport(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
) -> ObstacleGrid:
    """Attach known material-specific transport boxes to an obstacle grid."""
    boxes_m, mu_by_isotope = known_obstacle_transport_model(instances, isotopes=ISOTOPES)
    return grid.with_transport_model(boxes_m=boxes_m, mu_by_isotope=mu_by_isotope)


def _free_cell_centers(grid: ObstacleGrid) -> list[tuple[float, float]]:
    """Return centers of free grid cells."""
    centers: list[tuple[float, float]] = []
    for ix in range(int(grid.grid_shape[0])):
        for iy in range(int(grid.grid_shape[1])):
            if not grid.is_cell_free((ix, iy)):
                continue
            centers.append(
                (
                    float(grid.origin[0] + (ix + 0.5) * grid.cell_size),
                    float(grid.origin[1] + (iy + 0.5) * grid.cell_size),
                )
            )
    return centers


def _jittered_free_pose(
    grid: ObstacleGrid,
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    z_m: float = 0.5,
) -> tuple[float, float, float]:
    """Sample a detector pose inside a free cell."""
    centers = _free_cell_centers(grid)
    if not centers:
        raise ValueError("Generated obstacle grid has no free cells.")
    center_xy = centers[int(rng.integers(0, len(centers)))]
    jitter = 0.32 * float(grid.cell_size)
    x = float(center_xy[0] + rng.uniform(-jitter, jitter))
    y = float(center_xy[1] + rng.uniform(-jitter, jitter))
    x = float(np.clip(x, 0.4, room_size_xyz[0] - 0.4))
    y = float(np.clip(y, 0.4, room_size_xyz[1] - 0.4))
    return (x, y, float(z_m))


def _component_top_z(instance: KnownObstacleInstance) -> float:
    """Return the highest z coordinate of a known obstacle instance."""
    if not instance.components:
        return 0.0
    return max(
        float(component.center_xyz[2]) + 0.5 * float(component.size_xyz[2])
        for component in instance.components
    )


def _surface_source_position(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    source_index: int,
) -> tuple[float, float, float]:
    """Sample a source position on free floor, obstacle side, or obstacle top."""
    mode = int(source_index) % 4
    if instances and mode in {1, 2}:
        instance = instances[int(rng.integers(0, len(instances)))]
        x0, x1, y0, y1 = instance.footprint_xy
        if mode == 1:
            side = int(rng.integers(0, 4))
            if side == 0:
                x, y = x0 - 0.08, float(rng.uniform(y0, y1))
            elif side == 1:
                x, y = x1 + 0.08, float(rng.uniform(y0, y1))
            elif side == 2:
                x, y = float(rng.uniform(x0, x1)), y0 - 0.08
            else:
                x, y = float(rng.uniform(x0, x1)), y1 + 0.08
            z = float(rng.uniform(0.45, min(_component_top_z(instance), room_size_xyz[2] - 0.2)))
        else:
            x = float(rng.uniform(x0 + 0.12, x1 - 0.12))
            y = float(rng.uniform(y0 + 0.12, y1 - 0.12))
            z = min(_component_top_z(instance) + 0.08, room_size_xyz[2] - 0.2)
        return _clamp_room_position(
            np.asarray((x, y, z), dtype=float),
            room_size_xyz=room_size_xyz,
            margin_xy_m=0.25,
        )

    return _jittered_free_pose(
        grid,
        rng,
        room_size_xyz=room_size_xyz,
        z_m=float(rng.uniform(0.35, 1.8)),
    )


def _multi_isotope_surface_source_position(
    grid: ObstacleGrid,
    instances: tuple[KnownObstacleInstance, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    source_index: int,
) -> tuple[float, float, float]:
    """Sample a surface source over room and obstacle surfaces."""
    mode = int(source_index) % 6
    eps = 0.08
    if mode == 0:
        pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz, z_m=eps)
        return (pose[0], pose[1], eps)
    if mode == 1:
        pose = _jittered_free_pose(
            grid,
            rng,
            room_size_xyz=room_size_xyz,
            z_m=room_size_xyz[2] - eps,
        )
        return (pose[0], pose[1], room_size_xyz[2] - eps)
    if mode == 2:
        wall = int(rng.integers(0, 4))
        z = float(rng.uniform(0.35, room_size_xyz[2] - 0.35))
        if wall == 0:
            return (eps, float(rng.uniform(0.35, room_size_xyz[1] - 0.35)), z)
        if wall == 1:
            return (
                room_size_xyz[0] - eps,
                float(rng.uniform(0.35, room_size_xyz[1] - 0.35)),
                z,
            )
        if wall == 2:
            return (float(rng.uniform(0.35, room_size_xyz[0] - 0.35)), eps, z)
        return (
            float(rng.uniform(0.35, room_size_xyz[0] - 0.35)),
            room_size_xyz[1] - eps,
            z,
        )
    if instances and mode in {3, 4}:
        return _surface_source_position(
            grid,
            instances,
            rng,
            room_size_xyz=room_size_xyz,
            source_index=1 if mode == 3 else 2,
        )
    return _jittered_free_pose(
        grid,
        rng,
        room_size_xyz=room_size_xyz,
        z_m=float(rng.uniform(0.35, room_size_xyz[2] - 0.35)),
    )


def _obstacle_tau_for_sources(
    grid: ObstacleGrid,
    detector_xyz: tuple[float, float, float],
    sources: tuple[ValidationSource, ...],
) -> float:
    """Return the maximum obstacle optical depth among source-detector rays."""
    kernel = ContinuousKernel(obstacle_grid=grid, use_gpu=False)
    detector = np.asarray(detector_xyz, dtype=float)
    max_tau = 0.0
    for source in sources:
        tau = kernel.obstacle_optical_depth_pair(
            source.isotope,
            np.asarray(source.position_xyz, dtype=float),
            detector,
        )
        max_tau = max(max_tau, float(tau))
    return max_tau


def _sample_detector_with_tau_mix(
    grid: ObstacleGrid,
    sources: tuple[ValidationSource, ...],
    rng: np.random.Generator,
    *,
    room_size_xyz: tuple[float, float, float],
    prefer_obstacle_crossing: bool,
) -> tuple[float, float, float]:
    """Sample detector poses that alternate direct and obstacle-crossing rays."""
    best_pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz)
    best_tau = _obstacle_tau_for_sources(grid, best_pose, sources)
    for _ in range(80):
        pose = _jittered_free_pose(grid, rng, room_size_xyz=room_size_xyz)
        tau = _obstacle_tau_for_sources(grid, pose, sources)
        if prefer_obstacle_crossing and tau >= 0.1:
            return pose
        if not prefer_obstacle_crossing and tau <= 0.02:
            return pose
        if prefer_obstacle_crossing and tau > best_tau:
            best_pose, best_tau = pose, tau
        if not prefer_obstacle_crossing and tau < best_tau:
            best_pose, best_tau = pose, tau
    return best_pose


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


def generated_environment_sweep_cases(
    *,
    num_environments: int = 10,
    measurement_points_per_environment: int = 24,
    rotations_per_point: int = 8,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = 30000.0,
    intensity_max_cps_1m: float = 90000.0,
    blocked_fraction: float = 0.35,
    passage_width_m: float = 2.0,
) -> list[ValidationCase]:
    """Return random known-obstacle environments with point and shield sweeps."""
    rng = np.random.default_rng(int(seed))
    room_size = (10.0, 20.0, 10.0)
    cases: list[ValidationCase] = []
    for env_index in range(max(0, int(num_environments))):
        env_seed = int(rng.integers(1, 2**31 - 1))
        keep_free = (
            (0.5, 0.5),
            (room_size[0] - 0.5, room_size[1] - 0.5),
            (0.5, room_size[1] - 0.5),
            (room_size[0] - 0.5, 0.5),
            (0.5 * room_size[0], 0.5 * room_size[1]),
        )
        grid = build_obstacle_grid(
            mode="random",
            path=None,
            size_x=room_size[0],
            size_y=room_size[1],
            cell_size=1.0,
            blocked_fraction=float(blocked_fraction),
            rng_seed=env_seed,
            keep_free_points=keep_free,
            passage_width_m=float(passage_width_m),
        )
        instances = generate_manchester_obstacle_instances(
            grid,
            room_size_xyz=room_size,
            obstacle_height_m=2.0,
            rng_seed=env_seed + 17,
        )
        transport_grid = _attach_known_obstacle_transport(grid, instances)
        source_count = 1 + (env_index % 5)
        if env_index in {2, 6, 9}:
            source_count = 3 + (env_index % 3)
        isotope_program = _source_isotope_program(env_index, source_count)
        sources: list[ValidationSource] = []
        for source_index, isotope in enumerate(isotope_program):
            source_pos = _surface_source_position(
                transport_grid,
                instances,
                rng,
                room_size_xyz=room_size,
                source_index=source_index,
            )
            sources.append(
                ValidationSource(
                    isotope=str(isotope),
                    position_xyz=source_pos,
                    intensity_cps_1m=float(
                        rng.uniform(
                            float(intensity_min_cps_1m),
                            float(intensity_max_cps_1m),
                        )
                    ),
                )
            )
        source_tuple = tuple(sources)
        template_counts: dict[str, int] = {}
        for instance in instances:
            template_counts[instance.template] = template_counts.get(instance.template, 0) + 1
        template_summary = ",".join(
            f"{template}:{count}" for template, count in sorted(template_counts.items())
        )
        measurement_points: list[tuple[float, float, float]] = []
        for point_index in range(max(0, int(measurement_points_per_environment))):
            measurement_points.append(
                _sample_detector_with_tau_mix(
                    transport_grid,
                    source_tuple,
                    rng,
                    room_size_xyz=room_size,
                    prefer_obstacle_crossing=(point_index % 3 != 0),
                )
            )
        for point_index, detector in enumerate(measurement_points):
            max_tau = _obstacle_tau_for_sources(transport_grid, detector, source_tuple)
            for rotation_index in range(max(1, int(rotations_per_point))):
                cases.append(
                    ValidationCase(
                        name=f"env{env_index:02d}_pt{point_index:02d}_rot{rotation_index:02d}",
                        description=(
                            "Random known-material obstacle environment; "
                            f"templates={template_summary}; "
                            f"max_obstacle_tau={max_tau:.3f}"
                        ),
                        detector_pose_xyz=detector,
                        sources=source_tuple,
                        fe_index=int((rotation_index + 2 * point_index + env_index) % 8),
                        pb_index=int((3 * rotation_index + point_index + 2 * env_index) % 8),
                        dwell_time_s=float(dwell_time_s),
                        obstacle_cells=tuple(grid.blocked_cells),
                        obstacle_instances=instances,
                        include_in_accuracy_summary=True,
                    )
                )
    return cases


def generated_multi_isotope_source_cases(
    *,
    num_cases: int = 1000,
    seed: int = 20260430,
    dwell_time_s: float = 30.0,
    intensity_min_cps_1m: float = 30000.0,
    intensity_max_cps_1m: float = 90000.0,
    sources_per_isotope: int = 2,
    blocked_fraction: float = 0.35,
    passage_width_m: float = 2.0,
    min_isotope_target_counts: float = 0.0,
    detector_attempts: int = 24,
    runtime_config: dict[str, Any] | None = None,
    mu_by_isotope: dict[str, object] | None = None,
) -> list[ValidationCase]:
    """Return cases where Cs-137, Co-60, and Eu-154 each have multiple sources."""
    rng = np.random.default_rng(int(seed))
    room_size = (10.0, 20.0, 10.0)
    cases: list[ValidationCase] = []
    source_replicates = max(2, int(sources_per_isotope))
    for case_index in range(max(0, int(num_cases))):
        env_seed = int(rng.integers(1, 2**31 - 1))
        keep_free = (
            (0.5, 0.5),
            (room_size[0] - 0.5, room_size[1] - 0.5),
            (0.5, room_size[1] - 0.5),
            (room_size[0] - 0.5, 0.5),
            (0.5 * room_size[0], 0.5 * room_size[1]),
        )
        grid = build_obstacle_grid(
            mode="random",
            path=None,
            size_x=room_size[0],
            size_y=room_size[1],
            cell_size=1.0,
            blocked_fraction=float(blocked_fraction),
            rng_seed=env_seed,
            keep_free_points=keep_free,
            passage_width_m=float(passage_width_m),
        )
        instances = generate_manchester_obstacle_instances(
            grid,
            room_size_xyz=room_size,
            obstacle_height_m=2.0,
            rng_seed=env_seed + 17,
        )
        transport_grid = _attach_known_obstacle_transport(grid, instances)
        sources: list[ValidationSource] = []
        source_index = 0
        isotope_order = list(ISOTOPES)
        rng.shuffle(isotope_order)
        for isotope in isotope_order:
            for _ in range(source_replicates):
                source_pos = _multi_isotope_surface_source_position(
                    transport_grid,
                    instances,
                    rng,
                    room_size_xyz=room_size,
                    source_index=source_index,
                )
                sources.append(
                    ValidationSource(
                        isotope=str(isotope),
                        position_xyz=source_pos,
                        intensity_cps_1m=float(
                            rng.uniform(
                                float(intensity_min_cps_1m),
                                float(intensity_max_cps_1m),
                            )
                        ),
                    )
                )
                source_index += 1
        source_tuple = tuple(sources)
        fe_index = int((case_index * 3 + 1) % 8)
        pb_index = int((case_index * 5 + 2) % 8)
        require_target = (
            float(min_isotope_target_counts) > 0.0
            and runtime_config is not None
            and mu_by_isotope is not None
        )
        best_detector: tuple[float, float, float] | None = None
        best_min_target = -float("inf")
        max_detector_attempts = max(1, int(detector_attempts)) if require_target else 1
        for detector_attempt in range(max_detector_attempts):
            detector_candidate = _sample_detector_with_tau_mix(
                transport_grid,
                source_tuple,
                rng,
                room_size_xyz=room_size,
                prefer_obstacle_crossing=((case_index + detector_attempt) % 3 != 0),
            )
            if not require_target:
                best_detector = detector_candidate
                break
            candidate_case = ValidationCase(
                name=f"multi_iso_{case_index:04d}",
                description="candidate",
                detector_pose_xyz=detector_candidate,
                sources=source_tuple,
                fe_index=fe_index,
                pb_index=pb_index,
                dwell_time_s=float(dwell_time_s),
                obstacle_cells=tuple(grid.blocked_cells),
                obstacle_instances=instances,
                include_in_accuracy_summary=True,
            )
            counts = expected_pf_counts(candidate_case, runtime_config, mu_by_isotope)
            min_count = min(float(counts[isotope]) for isotope in ISOTOPES)
            if min_count > best_min_target:
                best_detector = detector_candidate
                best_min_target = min_count
            if min_count >= float(min_isotope_target_counts):
                break
        if best_detector is None:
            best_detector = _jittered_free_pose(
                transport_grid,
                rng,
                room_size_xyz=room_size,
            )
        template_counts: dict[str, int] = {}
        for instance in instances:
            template_counts[instance.template] = template_counts.get(instance.template, 0) + 1
        template_summary = ",".join(
            f"{template}:{count}" for template, count in sorted(template_counts.items())
        )
        max_tau = _obstacle_tau_for_sources(transport_grid, best_detector, source_tuple)
        min_target_text = (
            f"; min_isotope_target={best_min_target:.1f}"
            if require_target
            else ""
        )
        cases.append(
            ValidationCase(
                name=f"multi_iso_{case_index:04d}",
                description=(
                    f"{source_replicates} sources per isotope; "
                    f"templates={template_summary}; max_obstacle_tau={max_tau:.3f}"
                    f"{min_target_text}"
                ),
                detector_pose_xyz=best_detector,
                sources=source_tuple,
                fe_index=fe_index,
                pb_index=pb_index,
                dwell_time_s=float(dwell_time_s),
                obstacle_cells=tuple(grid.blocked_cells),
                obstacle_instances=instances,
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
        obstacle_instances=case.obstacle_instances,
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
        sources=[source.to_scene_source() for source in case.sources],
        usd_path=usd_path,
        use_config_usd_fallback=usd_path is not None,
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
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=tuple(case.obstacle_cells),
    )
    if case.obstacle_instances:
        return _attach_known_obstacle_transport(grid, case.obstacle_instances)
    return grid


def shield_params_from_runtime_config(runtime_config: dict[str, Any]) -> ShieldParams:
    """Return PF-side shield parameters matching the Geant4 runtime config."""
    shield_thickness = resolve_shield_thickness_config(runtime_config)
    detector_model = runtime_config.get("detector_model", {})
    if not isinstance(detector_model, dict):
        detector_model = {}
    detector_outer_radius_cm = 100.0 * (
        float(detector_model.get("crystal_radius_m", 0.038))
        + float(detector_model.get("housing_thickness_m", 0.0015))
    )
    inner_radius_fe_cm, inner_radius_pb_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        detector_outer_radius_cm=detector_outer_radius_cm,
    )
    buildup = runtime_config.get("pf_buildup", {})
    if not isinstance(buildup, dict):
        buildup = {}
    return ShieldParams(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
        inner_radius_fe_cm=float(inner_radius_fe_cm),
        inner_radius_pb_cm=float(inner_radius_pb_cm),
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


def expected_pf_count_diagnostics(
    case: ValidationCase,
    runtime_config: dict[str, Any],
    mu_by_isotope: dict[str, object],
) -> list[dict[str, Any]]:
    """Return per-source PF target components for mismatch diagnosis."""
    kernel = kernel_for_case(case, runtime_config, mu_by_isotope)
    shield_only_kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope,
        shield_params=shield_params_from_runtime_config(runtime_config),
        obstacle_grid=None,
        obstacle_height_m=float(runtime_config.get("obstacle_height_m", 2.0)),
        obstacle_buildup_coeff=0.0,
        detector_radius_m=kernel.detector_radius_m,
        detector_aperture_samples=kernel.detector_aperture_samples,
        use_gpu=False,
    )
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(case.sources):
        source_pos = np.asarray(source.position_xyz, dtype=float)
        geom = finite_sphere_geometric_term(
            detector,
            source_pos,
            kernel.detector_radius_m,
        )
        shield_att = shield_only_kernel.attenuation_factor_pair(
            source.isotope,
            source_pos,
            detector,
            int(case.fe_index),
            int(case.pb_index),
        )
        obstacle_tau = 0.0
        obstacle_att = 1.0
        if kernel.obstacle_grid is not None:
            obstacle_tau = kernel.obstacle_optical_depth_pair(
                source.isotope,
                source_pos,
                detector,
            )
            obstacle_att = float(np.exp(-float(obstacle_tau)))
        full_kernel = kernel.kernel_value_pair(
            source.isotope,
            detector,
            source_pos,
            int(case.fe_index),
            int(case.pb_index),
        )
        rows.append(
            {
                "source_index": int(source_index),
                "isotope": source.isotope,
                "position_xyz": [float(value) for value in source.position_xyz],
                "intensity_cps_1m": float(source.intensity_cps_1m),
                "geometric_factor": float(geom),
                "shield_attenuation": float(shield_att),
                "obstacle_tau_center_ray": float(obstacle_tau),
                "obstacle_attenuation_center_ray": float(obstacle_att),
                "full_kernel": float(full_kernel),
                "geometric_counts": float(case.dwell_time_s * source.intensity_cps_1m * geom),
                "shield_only_counts": float(
                    case.dwell_time_s * source.intensity_cps_1m * geom * shield_att
                ),
                "full_target_counts": float(
                    case.dwell_time_s * source.intensity_cps_1m * full_kernel
                ),
            }
        )
    return rows


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
        "obstacle_instances": obstacle_instances_to_dicts(case.obstacle_instances),
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
    scene = build_scene(case, usd_path=runtime_config.get("validation_usd_path"))
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
    target_diagnostics = expected_pf_count_diagnostics(case, runtime_config, mu_by_isotope)
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
            "target_pf_count_diagnostics": [
                row for row in target_diagnostics if row["isotope"] == isotope
            ],
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
        "target_diagnostics": target_diagnostics,
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


def build_summary(
    *,
    config_path: Path,
    output_dir: Path,
    cases: list[ValidationCase],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    sweep_start: float,
    interrupted: bool,
) -> dict[str, Any]:
    """Build a JSON-compatible summary for complete or partial validation output."""
    return {
        "config": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_cases": len(results),
        "num_requested_cases": len(cases),
        "interrupted": bool(interrupted),
        "environment_sweep": bool(args.environment_sweep),
        "num_environments": int(args.num_environments) if args.environment_sweep else None,
        "measurement_points_per_environment": (
            int(args.measurement_points_per_environment) if args.environment_sweep else None
        ),
        "rotations_per_point": int(args.rotations_per_point) if args.environment_sweep else None,
        "elapsed_s": float(time.time() - sweep_start),
        "min_target_counts": float(args.min_target_counts),
        "accuracy_summary": summarize_accuracy(results, float(args.min_target_counts)),
        "cases": [case_to_dict(case) for case in cases],
    }


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
    parser.add_argument(
        "--usd-path",
        default=None,
        help=(
            "Optional external USD to import into Geant4. By default validation "
            "uses only the generated scene geometry so the PF target and Geant4 "
            "transport share the same obstacle set."
        ),
    )
    parser.add_argument(
        "--primary-sampling-fraction",
        type=float,
        default=None,
        help="Override Geant4 primary_sampling_fraction for high-statistics validation.",
    )
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--case-seed", type=int, default=20260430)
    parser.add_argument("--dwell-time-s", type=float, default=30.0)
    parser.add_argument("--intensity-min-cps-1m", type=float, default=30000.0)
    parser.add_argument("--intensity-max-cps-1m", type=float, default=90000.0)
    parser.add_argument(
        "--environment-sweep",
        action="store_true",
        help=(
            "Generate random known-material obstacle environments, multiple "
            "measurement points, and multiple shield rotations per point."
        ),
    )
    parser.add_argument(
        "--multi-isotope-multi-source-cases",
        action="store_true",
        help=(
            "Generate cases where Cs-137, Co-60, and Eu-154 each have multiple "
            "sources on room or obstacle surfaces."
        ),
    )
    parser.add_argument(
        "--sources-per-isotope",
        type=int,
        default=2,
        help="Number of sources for each isotope in multi-isotope cases.",
    )
    parser.add_argument(
        "--multi-source-min-isotope-target-counts",
        type=float,
        default=0.0,
        help=(
            "Minimum PF-theory counts required for every isotope when sampling "
            "multi-isotope multi-source detector poses."
        ),
    )
    parser.add_argument(
        "--multi-source-detector-attempts",
        type=int,
        default=24,
        help="Detector-pose resampling attempts for each multi-isotope case.",
    )
    parser.add_argument("--num-environments", type=int, default=10)
    parser.add_argument("--measurement-points-per-environment", type=int, default=24)
    parser.add_argument("--rotations-per-point", type=int, default=8)
    parser.add_argument("--blocked-fraction", type=float, default=0.35)
    parser.add_argument("--passage-width-m", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="Write partial validation outputs every N completed cases; set 0 to disable.",
    )
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
    if args.primary_sampling_fraction is not None:
        runtime_config["primary_sampling_fraction"] = float(args.primary_sampling_fraction)
    runtime_config["validation_usd_path"] = (
        resolve_path(str(args.usd_path)).as_posix() if args.usd_path else None
    )
    runtime_config["engine_mode"] = "external"
    runtime_config["physics_profile"] = "balanced"

    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=ISOTOPES)

    if args.multi_isotope_multi_source_cases:
        all_cases = generated_multi_isotope_source_cases(
            num_cases=int(args.num_cases),
            seed=int(args.case_seed),
            dwell_time_s=float(args.dwell_time_s),
            intensity_min_cps_1m=float(args.intensity_min_cps_1m),
            intensity_max_cps_1m=float(args.intensity_max_cps_1m),
            sources_per_isotope=int(args.sources_per_isotope),
            blocked_fraction=float(args.blocked_fraction),
            passage_width_m=float(args.passage_width_m),
            min_isotope_target_counts=float(
                args.multi_source_min_isotope_target_counts
            ),
            detector_attempts=int(args.multi_source_detector_attempts),
            runtime_config=runtime_config,
            mu_by_isotope=mu_by_isotope,
        )
    elif args.environment_sweep:
        all_cases = generated_environment_sweep_cases(
            num_environments=int(args.num_environments),
            measurement_points_per_environment=int(args.measurement_points_per_environment),
            rotations_per_point=int(args.rotations_per_point),
            seed=int(args.case_seed),
            dwell_time_s=float(args.dwell_time_s),
            intensity_min_cps_1m=float(args.intensity_min_cps_1m),
            intensity_max_cps_1m=float(args.intensity_max_cps_1m),
            blocked_fraction=float(args.blocked_fraction),
            passage_width_m=float(args.passage_width_m),
        )
    elif args.hand_authored_cases:
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

    decomposer = SpectralDecomposer(spectrum_config_from_runtime_config(runtime_config))
    results: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    sweep_start = time.time()
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    interrupted = False
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
            checkpoint_every = max(int(args.checkpoint_every), 0)
            if checkpoint_every and len(results) % checkpoint_every == 0:
                partial_summary = build_summary(
                    config_path=config_path,
                    output_dir=output_dir,
                    cases=cases,
                    results=results,
                    args=args,
                    sweep_start=sweep_start,
                    interrupted=False,
                )
                write_outputs(output_dir, results, spectra, partial_summary)
                print(f"  checkpoint wrote partial outputs to: {output_dir}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; writing partial validation outputs.", flush=True)
    finally:
        app.close()

    summary = build_summary(
        config_path=config_path,
        output_dir=output_dir,
        cases=cases,
        results=results,
        args=args,
        sweep_start=sweep_start,
        interrupted=interrupted,
    )
    write_outputs(output_dir, results, spectra, summary)
    print(json.dumps(summary["accuracy_summary"], indent=2, sort_keys=True))
    print(f"Wrote validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
