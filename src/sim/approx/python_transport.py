"""Approximate Python transport for diagnostics and non-Geant4 debug modes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from measurement.continuous_kernels import segment_box_intersection_length_m
from measurement.continuous_kernels import validate_orientation_pair_indices
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    path_length_cm,
    resolve_mu_values,
    rotated_positive_octant_blocks_direction,
    spherical_shell_path_length_cm,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.transport import (
    SourceTransportResult,
    TransportMaterial,
    TransportSegment,
    build_source_transport_result,
    make_transport_segment,
)
from spectrum.library import default_library
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    backscatter_energy,
    build_native_geant4_detector_response_matrix,
    compton_continuum_shape,
    default_resolution,
    gaussian_peak,
    native_geant4_background_shape,
)
from spectrum.transport_spectral import sample_nonparalyzable_counts_numpy

StageSegmentsProvider = Callable[
    [PointSource, tuple[float, float, float]],
    Sequence[TransportSegment],
]
ShieldPathProvider = Callable[
    [PointSource, tuple[float, float, float], SimulationCommand],
    tuple[float, float],
]


@dataclass
class PythonTransportScene:
    """Store scene inputs used by the shared Python transport model."""

    sources: list[PointSource] = field(default_factory=list)
    obstacle_grid: ObstacleGrid | None = None
    obstacle_material: str = "concrete"


def energy_bin_edges_keV() -> np.ndarray:
    """Return the fixed raw-spectrum bin edges used by the debug generator."""
    return np.arange(
        NATIVE_GEANT4_BIN_COUNT + 1,
        dtype=np.float64,
    ) * float(NATIVE_GEANT4_BIN_WIDTH_KEV)


def point_sources_from_payload(payload: Mapping[str, Any]) -> list[PointSource]:
    """Parse point sources from a simulator reset payload."""
    sources_payload = payload.get("sources", [])
    if not isinstance(sources_payload, list):
        raise ValueError("sources must be a list.")
    sources: list[PointSource] = []
    for index, entry in enumerate(sources_payload):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Source entry {index} must be an object.")
        position_payload = entry.get("position", (0.0, 0.0, 0.0))
        if (
            not isinstance(position_payload, (list, tuple))
            or len(position_payload) != 3
        ):
            raise ValueError(f"Source entry {index} position must have three values.")
        transport_payload = entry.get(
            "transport_position",
            position_payload,
        )
        if (
            not isinstance(transport_payload, (list, tuple))
            or len(transport_payload) != 3
        ):
            raise ValueError(
                f"Source entry {index} transport_position must have three values."
            )
        surface_uv_payload = entry.get("surface_uv")
        surface_normal_payload = entry.get("surface_normal")
        sources.append(
            PointSource(
                isotope=str(entry.get("isotope", f"source_{index}")),
                position=(
                    float(position_payload[0]),
                    float(position_payload[1]),
                    float(position_payload[2]),
                ),
                intensity_cps_1m=float(entry.get("intensity_cps_1m", 0.0)),
                surface_chart_id=entry.get("surface_chart_id"),
                surface_uv=(
                    None
                    if surface_uv_payload is None
                    else (
                        float(surface_uv_payload[0]),
                        float(surface_uv_payload[1]),
                    )
                ),
                surface_normal=(
                    None
                    if surface_normal_payload is None
                    else (
                        float(surface_normal_payload[0]),
                        float(surface_normal_payload[1]),
                        float(surface_normal_payload[2]),
                    )
                ),
                transport_position=(
                    None
                    if entry.get("surface_chart_id") is None
                    else (
                        float(transport_payload[0]),
                        float(transport_payload[1]),
                        float(transport_payload[2]),
                    )
                ),
                surface_emission_policy_sha256=entry.get(
                    "surface_emission_policy_sha256"
                ),
            )
        )
    return sources


def obstacle_grid_from_payload(payload: Mapping[str, Any]) -> ObstacleGrid | None:
    """Parse an obstacle grid from a simulator reset payload."""
    shape_payload = payload.get("obstacle_grid_shape", (0, 0))
    if not isinstance(shape_payload, (list, tuple)) or len(shape_payload) != 2:
        raise ValueError("obstacle_grid_shape must have two values.")
    grid_shape = shape_payload
    collision_boxes_payload = payload.get("collision_boxes_m", [])
    transport_boxes_payload = payload.get("transport_boxes_m", [])
    if not isinstance(collision_boxes_payload, list):
        raise ValueError("collision_boxes_m must be a list.")
    if not isinstance(transport_boxes_payload, list):
        raise ValueError("transport_boxes_m must be a list.")
    has_grid = grid_shape[0] > 0 and grid_shape[1] > 0
    if not has_grid and not collision_boxes_payload and not transport_boxes_payload:
        return None
    origin_payload = payload.get("obstacle_origin_xy", (0.0, 0.0))
    if not isinstance(origin_payload, (list, tuple)) or len(origin_payload) != 2:
        raise ValueError("obstacle_origin_xy must have two values.")
    cells_payload = payload.get("obstacle_cells", [])
    if not isinstance(cells_payload, list):
        raise ValueError("obstacle_cells must be a list.")
    return ObstacleGrid(
        origin=origin_payload,
        cell_size=payload.get("obstacle_cell_size_m", 1.0),
        grid_shape=grid_shape,
        blocked_cells=cells_payload,
        collision_boxes_m=collision_boxes_payload,
        transport_boxes_m=transport_boxes_payload,
        transport_mu_by_isotope=payload.get(
            "transport_mu_by_isotope",
            {},
        ),
        transport_line_mu_by_isotope=payload.get(
            "transport_line_mu_by_isotope",
            {},
        ),
    )


class PythonTransportSpectrumModel:
    """Generate spectra with shared Python geometry and detector-response logic."""

    def __init__(
        self,
        *,
        sources: Iterable[PointSource] = (),
        mu_by_isotope: Mapping[str, object] | None = None,
        shield_params: ShieldParams | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_material: str = "concrete",
        scatter_gain: float = 0.03,
        rng_seed: int = 123,
        dead_time_s: float = 0.0,
        background_rate_cps: float = 0.0,
        detector_model: Mapping[str, Any] | None = None,
    ) -> None:
        """Store model configuration and scene state."""
        self.library = default_library()
        self.energy_axis_keV = (
            np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
            * float(NATIVE_GEANT4_BIN_WIDTH_KEV)
        )
        self.bin_width_keV = float(NATIVE_GEANT4_BIN_WIDTH_KEV)
        self.resolution_fn = default_resolution()
        self._native_response_lb = build_native_geant4_detector_response_matrix(
            self.energy_axis_keV,
            self.bin_width_keV,
        )
        self.background_shape_b = native_geant4_background_shape(
            self.energy_axis_keV,
            self.bin_width_keV,
        )
        self.mu_by_isotope = dict(mu_by_isotope or {})
        self.shield_params = shield_params or ShieldParams()
        self.obstacle_height_m = float(obstacle_height_m)
        self.scatter_gain = float(scatter_gain)
        self.rng_seed = int(rng_seed)
        self.dead_time_s = float(dead_time_s)
        self.background_rate_cps = float(background_rate_cps)
        if (
            not np.isfinite(self.background_rate_cps)
            or self.background_rate_cps < 0.0
        ):
            raise ValueError("background_rate_cps must be finite and nonnegative.")
        self.detector_model = dict(detector_model or {})
        self.octant_shield = OctantShield()
        self.orientations = generate_octant_orientations()
        self.scene = PythonTransportScene(
            sources=list(sources),
            obstacle_grid=obstacle_grid,
            obstacle_material=str(obstacle_material),
        )
        self._line_response_cache: dict[float, np.ndarray] = {}
        self._scatter_response_cache: dict[float, np.ndarray] = {}

    def reset_from_payload(self, payload: Mapping[str, Any] | None) -> None:
        """Reset sources, obstacle geometry, and optional detector metadata."""
        if payload is None:
            return
        self.scene = PythonTransportScene(
            sources=point_sources_from_payload(payload),
            obstacle_grid=obstacle_grid_from_payload(payload),
            obstacle_material=str(
                payload.get("obstacle_material", self.scene.obstacle_material)
            ),
        )
        detector_model = payload.get("detector_model")
        if detector_model is not None:
            if not isinstance(detector_model, Mapping):
                raise ValueError("detector_model must be an object.")
            self.detector_model = dict(detector_model)

    def reset_scene(
        self,
        *,
        sources: Iterable[PointSource],
        obstacle_grid: ObstacleGrid | None,
        obstacle_material: str,
    ) -> None:
        """Reset the active scene from already parsed scene objects."""
        self.scene = PythonTransportScene(
            sources=list(sources),
            obstacle_grid=obstacle_grid,
            obstacle_material=str(obstacle_material),
        )

    def observe(
        self,
        command: SimulationCommand,
        *,
        detector_pose_xyz: tuple[float, float, float],
        detector_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        backend_label: str,
        sources: Iterable[PointSource] | None = None,
        stage_segments_provider: StageSegmentsProvider | None = None,
        shield_path_provider: ShieldPathProvider | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> SimulationObservation:
        """Generate one sampled spectrum observation."""
        transport_results = self.source_transport_results(
            command,
            detector_pose_xyz=detector_pose_xyz,
            sources=self.scene.sources if sources is None else sources,
            stage_segments_provider=stage_segments_provider,
            shield_path_provider=shield_path_provider,
        )
        expected = self.expected_spectrum_from_transport_results(
            transport_results, command.dwell_time_s
        )
        spectrum = self.sample_spectrum(
            expected,
            command.step_id,
            live_time_s=float(command.dwell_time_s),
        )
        metadata = self.metadata_from_transport_results(
            transport_results,
            backend_label=backend_label,
            expected_spectrum=expected,
            extra_metadata=extra_metadata,
        )
        num_orientations = max(1, int(len(self.orientations)))
        metadata.setdefault("fe_orientation_index", int(command.fe_orientation_index))
        metadata.setdefault("pb_orientation_index", int(command.pb_orientation_index))
        metadata.setdefault("shield_num_orientations", int(num_orientations))
        metadata.setdefault(
            "shield_pair_id",
            int(command.fe_orientation_index) * num_orientations
            + int(command.pb_orientation_index),
        )
        metadata.setdefault(
            "shield_thickness_fe_cm",
            float(self.shield_params.thickness_fe_cm),
        )
        metadata.setdefault(
            "shield_thickness_pb_cm",
            float(self.shield_params.thickness_pb_cm),
        )
        metadata.setdefault(
            "shield_thickness_scale",
            0.0
            if (
                float(self.shield_params.thickness_fe_cm) <= 0.0
                and float(self.shield_params.thickness_pb_cm) <= 0.0
            )
            else 1.0,
        )
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=tuple(float(value) for value in detector_pose_xyz),
            detector_quat_wxyz=tuple(float(value) for value in detector_quat_wxyz),
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=np.int64).tolist(),
            energy_bin_edges_keV=energy_bin_edges_keV().tolist(),
            metadata=metadata,
        )

    def source_transport_results(
        self,
        command: SimulationCommand,
        *,
        detector_pose_xyz: tuple[float, float, float],
        sources: Iterable[PointSource],
        stage_segments_provider: StageSegmentsProvider | None = None,
        shield_path_provider: ShieldPathProvider | None = None,
    ) -> tuple[SourceTransportResult, ...]:
        """Build transport results for all source-detector pairs."""
        detector_position = tuple(float(value) for value in detector_pose_xyz)
        results: list[SourceTransportResult] = []
        for source in sources:
            if source.isotope not in self.library:
                continue
            if stage_segments_provider is None:
                stage_segments = self.obstacle_stage_segments(source, detector_position)
            else:
                stage_segments = tuple(
                    stage_segments_provider(source, detector_position)
                )
            if shield_path_provider is None:
                fe_path_cm, pb_path_cm = self.shield_path_lengths_cm(
                    source,
                    detector_position,
                    command.fe_orientation_index,
                    command.pb_orientation_index,
                )
            else:
                fe_path_cm, pb_path_cm = shield_path_provider(
                    source, detector_position, command
                )
            nuclide = self.library[source.isotope]
            nuclide_lines = tuple(
                (float(line.energy_keV), float(line.intensity))
                for line in nuclide.lines
            )
            results.append(
                build_source_transport_result(
                    source=source,
                    detector_position_xyz=detector_position,
                    dwell_time_s=float(command.dwell_time_s),
                    stage_segments=stage_segments,
                    fe_segment=self._shield_segment(source.isotope, "fe", fe_path_cm),
                    pb_segment=self._shield_segment(source.isotope, "pb", pb_path_cm),
                    nuclide_lines=nuclide_lines,
                    scatter_gain=self.scatter_gain,
                )
            )
        return tuple(results)

    def obstacle_stage_segments(
        self,
        source: PointSource,
        detector_pose_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return material segments through exclusive transport or fallback boxes."""
        obstacle_grid = self.scene.obstacle_grid
        if obstacle_grid is None:
            return ()
        source_position = source.transport_position_array()
        detector_position = np.asarray(detector_pose_xyz, dtype=float)
        boxes = obstacle_grid.attenuation_boxes(
            z_min=0.0,
            z_max=self.obstacle_height_m,
        )
        isotope_mu_values = (
            obstacle_grid.transport_mu_values(source.isotope)
            if obstacle_grid.has_transport_model
            else None
        )
        line_mu_values = (
            obstacle_grid.transport_line_mu_values(source.isotope)
            if obstacle_grid.has_transport_model
            else None
        )
        gamma_lines = tuple(
            line
            for line in self.library[source.isotope].lines
            if max(float(line.intensity), 0.0) > 0.0
        )
        segments: list[TransportSegment] = []
        for box_index, box in enumerate(boxes):
            path_length_cm = 100.0 * segment_box_intersection_length_m(
                source_position,
                detector_position,
                np.asarray(box, dtype=float),
            )
            if path_length_cm <= 0.0:
                continue
            mu_by_isotope = {}
            if isotope_mu_values is not None:
                mu_by_isotope[source.isotope] = float(isotope_mu_values[box_index])
            mu_by_energy_keV = {}
            if line_mu_values is not None and len(line_mu_values) == len(gamma_lines):
                mu_by_energy_keV = {
                    float(line.energy_keV): float(line_mu_values[line_index][box_index])
                    for line_index, line in enumerate(gamma_lines)
                }
            material = TransportMaterial(
                name=self.scene.obstacle_material,
                mu_by_isotope=mu_by_isotope,
                mu_by_energy_keV=mu_by_energy_keV,
            )
            segments.append(
                make_transport_segment(
                    material,
                    path_length_cm,
                    is_obstacle=True,
                )
            )
        return tuple(segments)

    def shield_path_lengths_cm(
        self,
        source: PointSource,
        detector_pose_xyz: tuple[float, float, float],
        fe_orientation_index: int,
        pb_orientation_index: int,
    ) -> tuple[float, float]:
        """Return Fe and Pb spherical-octant shell path lengths."""
        source_pos = source.position_array()
        detector_pos = np.asarray(detector_pose_xyz, dtype=float)
        direction = detector_pos - source_pos
        fe_indices, pb_indices = validate_orientation_pair_indices(
            np.asarray([fe_orientation_index]),
            np.asarray([pb_orientation_index]),
            orientation_count=len(self.orientations),
            expected_count=1,
        )
        fe_index = int(fe_indices[0])
        pb_index = int(pb_indices[0])
        detector_to_source = source_pos - detector_pos
        fe_blocked = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -self.orientations[fe_index],
        )
        pb_blocked = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -self.orientations[pb_index],
        )
        fe_path = self._shield_path_length_cm(
            direction_m=direction,
            normal=self.orientations[fe_index],
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=fe_blocked,
        )
        pb_path = self._shield_path_length_cm(
            direction_m=direction,
            normal=self.orientations[pb_index],
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=pb_blocked,
        )
        return float(fe_path), float(pb_path)

    def expected_spectrum_from_transport_results(
        self,
        transport_results: Iterable[SourceTransportResult],
        dwell_time_s: float,
    ) -> np.ndarray:
        """Return the expected detector spectrum from transport results."""
        expected = np.zeros_like(self.energy_axis_keV, dtype=float)
        for transport_result in transport_results:
            expected += self.source_expected_spectrum(transport_result)
        if self.background_rate_cps > 0.0:
            expected += (
                self.background_shape_b
                * self.background_rate_cps
                * float(dwell_time_s)
            )
        return np.clip(expected, a_min=0.0, a_max=None)

    def source_expected_spectrum(
        self,
        transport_result: SourceTransportResult,
    ) -> np.ndarray:
        """Return the expected spectrum contribution for one transported source."""
        if not transport_result.lines or transport_result.base_source_counts <= 0.0:
            return np.zeros_like(self.energy_axis_keV, dtype=float)
        expected = np.zeros_like(self.energy_axis_keV, dtype=float)
        for line in transport_result.lines:
            expected += float(line.primary_counts) * self.line_response_template(
                line.energy_keV
            )
            if line.scatter_counts > 0.0:
                expected += float(line.scatter_counts) * self.scatter_response_template(
                    line.energy_keV
                )
        return expected

    def sample_spectrum(
        self,
        expected_spectrum: np.ndarray,
        step_id: int,
        *,
        live_time_s: float,
    ) -> np.ndarray:
        """Draw an integer renewal total and conditional raw-bin marks."""
        rng = np.random.default_rng(self.rng_seed + int(step_id))
        expected = np.clip(
            np.asarray(expected_spectrum, dtype=np.float64),
            a_min=0.0,
            a_max=None,
        )
        total_mean = float(np.sum(expected))
        live_time = float(live_time_s)
        if not np.isfinite(live_time) or live_time <= 0.0:
            raise ValueError("live_time_s must be finite and positive.")
        if total_mean <= 0.0:
            return np.zeros(expected.shape, dtype=np.int64)
        total_count = int(
            sample_nonparalyzable_counts_numpy(
                np.asarray(total_mean / live_time, dtype=np.float64),
                np.asarray(live_time, dtype=np.float64),
                dead_time_tau_s=float(self.dead_time_s),
                rng=rng,
            )
        )
        if total_count <= 0:
            return np.zeros(expected.shape, dtype=np.int64)
        probabilities = expected / total_mean
        return np.asarray(
            rng.multinomial(total_count, probabilities),
            dtype=np.int64,
        )

    def metadata_from_transport_results(
        self,
        transport_results: Sequence[SourceTransportResult],
        *,
        backend_label: str,
        expected_spectrum: np.ndarray,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build common observation metadata for Python transport observations."""
        metadata: dict[str, Any] = {
            "backend": str(backend_label),
            "transport_backend": "python",
            "python_transport_model": (
                "approximate_line_transport_native_mark_axis_v2"
            ),
            "approximate_debug_backend": True,
            "num_sources": int(len(transport_results)),
            "total_obstacle_path_cm": float(
                sum(result.total_obstacle_path_cm for result in transport_results)
            ),
            "total_stage_path_cm": float(
                sum(result.total_stage_path_cm for result in transport_results)
            ),
            "total_fe_path_cm": float(
                sum(result.total_fe_path_cm for result in transport_results)
            ),
            "total_pb_path_cm": float(
                sum(result.total_pb_path_cm for result in transport_results)
            ),
            "expected_total_counts": float(np.sum(expected_spectrum)),
            "scatter_gain": float(self.scatter_gain),
            "dead_time_s": float(self.dead_time_s),
            "background_rate_cps": float(self.background_rate_cps),
            "raw_integer_spectrum": True,
        }
        if self.detector_model:
            metadata["detector_model"] = dict(self.detector_model)
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        return metadata

    def line_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return the detector response template for a unit-intensity gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._line_response_cache.get(cache_key)
        if cached is not None:
            return cached
        incident_index = int(
            np.clip(
                np.rint(cache_key / self.bin_width_keV),
                0,
                self.energy_axis_keV.size - 1,
            )
        )
        response = np.asarray(
            self._native_response_lb[:, incident_index],
            dtype=np.float64,
        ).copy()
        self._line_response_cache[cache_key] = response
        return response

    def scatter_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return a scatter-dominated low-energy response for one gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._scatter_response_cache.get(cache_key)
        if cached is not None:
            return cached
        energy_axis = np.asarray(self.energy_axis_keV, dtype=float)
        response = compton_continuum_shape(energy_axis, cache_key, shape="exponential")
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        if cache_key > 200.0:
            e_back = backscatter_energy(cache_key)
            sigma_back = float(self.resolution_fn(e_back))
            response += 0.25 * gaussian_peak(
                energy_axis, center=e_back, sigma=sigma_back
            )
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        self._scatter_response_cache[cache_key] = response
        return response

    def _shield_path_length_cm(
        self,
        *,
        direction_m: np.ndarray,
        normal: np.ndarray,
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return the configured shield-geometry path length."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return spherical_shell_path_length_cm(
                direction_m=direction_m,
                inner_radius_cm=float(inner_radius_cm),
                outer_radius_cm=float(inner_radius_cm) + float(thickness_cm),
                blocked=blocked,
            )
        return path_length_cm(
            direction_m,
            normal,
            float(thickness_cm),
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )

    def _shield_segment(
        self,
        isotope: str,
        material_name: str,
        path_length_cm: float,
    ) -> TransportSegment:
        """Build a shield segment using the PF TVL coefficients for this isotope."""
        mu_fe, mu_pb = resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )
        mu = mu_fe if material_name == "fe" else mu_pb
        material = TransportMaterial(
            name=material_name,
            mu_by_isotope={str(isotope): float(mu)},
        )
        return make_transport_segment(material, float(path_length_cm))
