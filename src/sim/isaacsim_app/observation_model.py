"""Observation generation for the sidecar bridge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from measurement.model import EnvironmentConfig
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from sim.isaacsim_app.geometry import (
    OrientedBox,
    Sphere,
    TriangleMesh,
    quaternion_wxyz_to_matrix,
    segment_path_length_through_box,
    segment_path_length_through_mesh,
    segment_path_length_through_sphere,
)
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.scene_builder import SceneDescription
from sim.isaacsim_app.stage_backend import StageMaterialInfo
from sim.protocol import SimulationCommand, SimulationObservation
from sim.transport import (
    SourceTransportResult,
    TransportSegment,
    build_source_transport_result,
    make_transport_segment,
)
from sim.shield_geometry import (
    FE_SHIELD_INNER_RADIUS_M,
    FE_SHIELD_THICKNESS_CM,
    PB_SHIELD_INNER_RADIUS_M,
    PB_SHIELD_THICKNESS_CM,
    spherical_octant_path_length_cm,
)
from spectrum.pipeline import BACKGROUND_COUNTS_PER_SECOND, BACKGROUND_RATE_CPS, SpectralDecomposer
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    backscatter_energy,
    compton_continuum_shape,
    gaussian_peak,
)


@dataclass(frozen=True)
class IsaacAssetGeometry:
    """Collect simple geometric parameters for the bridge-authored assets."""

    detector_height_m: float = 0.5
    obstacle_height_m: float = 2.0
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)


@dataclass(frozen=True)
class StageMaterialRule:
    """Map a stage path prefix to a named attenuation material."""

    path_prefix: str
    material: str


class ObservationModel(ABC):
    """Define the observation interface used by the bridge app."""

    @abstractmethod
    def reset(self, scene: SceneDescription) -> None:
        """Update any internal state after a new scene is loaded."""

    @abstractmethod
    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return an observation for the requested command."""


class MockObservationModel(ObservationModel):
    """Generate bridge observations without requiring Isaac Sim installation."""

    def __init__(self) -> None:
        """Create the analytic spectrum simulator used in mock mode."""
        self.decomposer = SpectralDecomposer()
        self.scene = SceneDescription()
        self._mu_by_isotope = mu_by_isotope_from_tvl_mm(
            HVL_TVL_TABLE_MM,
            isotopes=list(self.decomposer.isotope_names),
        )

    def reset(self, scene: SceneDescription) -> None:
        """Store the active scene for the mock backend."""
        self.scene = scene

    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return an analytic observation anchored to the commanded detector pose."""
        detector_pose = tuple(float(v) for v in command.target_pose_xyz)
        spectrum, _ = self.decomposer.simulate_spectrum(
            sources=self.scene.to_point_sources(),
            environment=EnvironmentConfig(detector_position=detector_pose),
            acquisition_time=command.dwell_time_s,
            rng=np.random.default_rng(123 + int(command.step_id)),
            mu_by_isotope=self._mu_by_isotope,
        )
        energy = np.asarray(self.decomposer.energy_axis, dtype=float)
        step = float(np.median(np.diff(energy))) if energy.size > 1 else 1.0
        metadata = {"backend": "isaacsim-mock"}
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=detector_pose,
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=float).tolist(),
            energy_bin_edges_keV=np.concatenate([energy, [energy[-1] + step]]).tolist(),
            metadata=metadata,
        )


class IsaacSimObservationModel(ObservationModel):
    """Generate spectra using stage geometry for obstacle and shield attenuation."""

    def __init__(
        self,
        robot_controller: RobotController,
        *,
        usd_path: str | None,
        asset_geometry: IsaacAssetGeometry,
        stage_material_rules: tuple[StageMaterialRule, ...] = (),
        rng_seed: int = 123,
        scatter_gain: float = 0.03,
    ) -> None:
        """Store stage-backed handles and spectrum simulation helpers."""
        self.robot_controller = robot_controller
        self.usd_path = usd_path
        self.asset_geometry = asset_geometry
        self.stage_material_rules = tuple(stage_material_rules)
        self.rng_seed = int(rng_seed)
        self.scatter_gain = float(scatter_gain)
        self.decomposer = SpectralDecomposer()
        self.scene = SceneDescription()
        self._line_response_cache: dict[float, np.ndarray] = {}
        self._scatter_response_cache: dict[float, np.ndarray] = {}

    def reset(self, scene: SceneDescription) -> None:
        """Store the loaded scene description for subsequent observations."""
        self.scene = scene

    def observe(self, command: SimulationCommand) -> SimulationObservation:
        """Return a spectrum attenuated by stage-authored obstacles and shields."""
        detector_pose = self.robot_controller.detector_world_pose()
        total_obstacle_path_cm = 0.0
        total_stage_path_cm = 0.0
        total_fe_path_cm = 0.0
        total_pb_path_cm = 0.0
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        num_sources = 0

        for transport_result in self._build_source_transport_results(command):
            total_obstacle_path_cm += transport_result.total_obstacle_path_cm
            total_stage_path_cm += transport_result.total_stage_path_cm
            total_fe_path_cm += transport_result.total_fe_path_cm
            total_pb_path_cm += transport_result.total_pb_path_cm
            expected += self._source_expected_spectrum(transport_result)
            num_sources += 1

        background_rate = BACKGROUND_RATE_CPS
        if BACKGROUND_COUNTS_PER_SECOND != BACKGROUND_RATE_CPS:
            background_rate = BACKGROUND_COUNTS_PER_SECOND
        if background_rate > 0.0:
            expected += self.decomposer._background_shape * float(background_rate) * float(command.dwell_time_s)
        rng = np.random.default_rng(self.rng_seed + int(command.step_id))
        spectrum = rng.poisson(expected)
        energy = np.asarray(self.decomposer.energy_axis, dtype=float)
        step = float(np.median(np.diff(energy))) if energy.size > 1 else 1.0
        metadata: dict[str, float | int | str] = {
            "backend": "isaacsim",
            "transport_backend": "python",
            "num_sources": num_sources,
            "total_obstacle_path_cm": float(total_obstacle_path_cm),
            "total_stage_path_cm": float(total_stage_path_cm),
            "total_fe_path_cm": float(total_fe_path_cm),
            "total_pb_path_cm": float(total_pb_path_cm),
        }
        if self.usd_path:
            metadata["usd_path"] = self.usd_path
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=detector_pose.translation_xyz,
            detector_quat_wxyz=detector_pose.orientation_wxyz,
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=float).tolist(),
            energy_bin_edges_keV=np.concatenate([energy, [energy[-1] + step]]).tolist(),
            metadata=metadata,
        )

    def _build_source_transport_results(
        self,
        command: SimulationCommand,
    ) -> tuple[SourceTransportResult, ...]:
        """Build shared pre-spectrum transport results for all scene sources."""
        detector_pose = self.robot_controller.detector_world_pose()
        results: list[SourceTransportResult] = []
        for source in self.scene.to_point_sources():
            stage_segments = self._stage_geometry_segments(source.position, detector_pose.translation_xyz)
            fe_path_cm, pb_path_cm = self._shield_path_lengths_cm(source.position)
            nuclide = self.decomposer.library.get(source.isotope)
            nuclide_lines = ()
            if nuclide is not None:
                nuclide_lines = tuple((float(line.energy_keV), float(line.intensity)) for line in nuclide.lines)
            results.append(
                build_source_transport_result(
                    source=source,
                    detector_position_xyz=detector_pose.translation_xyz,
                    dwell_time_s=float(command.dwell_time_s),
                    stage_segments=stage_segments,
                    fe_segment=make_transport_segment(StageMaterialInfo(name="fe"), fe_path_cm),
                    pb_segment=make_transport_segment(StageMaterialInfo(name="pb"), pb_path_cm),
                    nuclide_lines=nuclide_lines,
                    scatter_gain=self.scatter_gain,
                )
            )
        return tuple(results)

    def _stage_geometry_segments(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return shared transport segments for crossed static materials."""
        segments: list[TransportSegment] = []
        prefixes = tuple(
            sorted(
                {
                    *{rule.path_prefix for rule in self.stage_material_rules},
                    self.robot_controller.prim_paths.obstacles_root,
                    "/World/Environment",
                }
            )
        )
        for solid_prim in self.robot_controller.stage_backend.list_solid_prims(path_prefixes=prefixes):
            material_info = self._material_for_prim(solid_prim.path, solid_prim.material_info)
            if material_info is None:
                continue
            if solid_prim.path in {
                self.robot_controller.prim_paths.fe_shield_path,
                self.robot_controller.prim_paths.pb_shield_path,
                self.robot_controller.prim_paths.detector_path,
            }:
                continue
            if solid_prim.path.startswith(self.robot_controller.prim_paths.sources_root):
                continue
            path_length_cm = 100.0 * self._solid_path_length_m(source_xyz, detector_xyz, solid_prim)
            if path_length_cm <= 0.0:
                continue
            segments.append(
                make_transport_segment(
                    material_info,
                    float(path_length_cm),
                    is_obstacle=material_info.name.lower() == self.scene.obstacle_material.lower(),
                )
            )
        return tuple(segments)

    def _shield_path_lengths_cm(
        self,
        source_xyz: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Return path lengths through the Fe and Pb spherical octant shells."""
        fe_pose = self.robot_controller.stage_backend.get_world_pose(
            self.robot_controller.prim_paths.fe_shield_path
        )
        pb_pose = self.robot_controller.stage_backend.get_world_pose(
            self.robot_controller.prim_paths.pb_shield_path
        )
        detector_pose = self.robot_controller.detector_world_pose()
        fe_length_cm = spherical_octant_path_length_cm(
            source_xyz,
            detector_pose.translation_xyz,
            fe_pose.orientation_wxyz,
            thickness_cm=FE_SHIELD_THICKNESS_CM,
            inner_radius_cm=FE_SHIELD_INNER_RADIUS_M * 100.0,
        )
        pb_length_cm = spherical_octant_path_length_cm(
            source_xyz,
            detector_pose.translation_xyz,
            pb_pose.orientation_wxyz,
            thickness_cm=PB_SHIELD_THICKNESS_CM,
            inner_radius_cm=PB_SHIELD_INNER_RADIUS_M * 100.0,
        )
        return float(fe_length_cm), float(pb_length_cm)

    def _source_expected_spectrum(
        self,
        transport_result: SourceTransportResult,
    ) -> np.ndarray:
        """Return the expected spectrum contribution for one transported source."""
        if not transport_result.lines or transport_result.base_source_counts <= 0.0:
            return np.zeros_like(self.decomposer.energy_axis, dtype=float)
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        for line in transport_result.lines:
            expected += float(line.primary_counts) * self._line_response_template(line.energy_keV)
            if line.scatter_counts > 0.0:
                expected += float(line.scatter_counts) * self._scatter_response_template(line.energy_keV)
        return expected

    def _line_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return the detector response template for a unit-intensity gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._line_response_cache.get(cache_key)
        if cached is not None:
            return cached
        energy_axis = np.asarray(self.decomposer.energy_axis, dtype=float)
        bin_width_keV = float(self.decomposer.config.bin_width_keV)
        sigma = float(self.decomposer.resolution_fn(cache_key))
        peak = gaussian_peak(energy_axis, center=cache_key, sigma=sigma)
        peak_area = float(peak.sum() * bin_width_keV)
        efficiency = (
            float(self.decomposer.efficiency_fn(cache_key))
            if self.decomposer.efficiency_fn is not None
            else 1.0
        )
        response = peak * efficiency * bin_width_keV
        cont_shape = compton_continuum_shape(energy_axis, cache_key, shape="exponential")
        if cont_shape.sum() > 0.0:
            cont_shape = cont_shape / float(cont_shape.sum())
            response += COMPTON_CONTINUUM_TO_PEAK * peak_area * cont_shape * efficiency
        if cache_key > 200.0:
            e_back = backscatter_energy(cache_key)
            sigma_back = float(self.decomposer.resolution_fn(e_back))
            back = gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
            back_norm = float(back.sum() * bin_width_keV)
            if back_norm > 0.0:
                area_back = BACKSCATTER_FRACTION * peak_area
                response += back * (area_back / back_norm) * float(self.decomposer.efficiency_fn(e_back))
        self._line_response_cache[cache_key] = response
        return response

    def _scatter_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return a scatter-dominated low-energy response for one gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._scatter_response_cache.get(cache_key)
        if cached is not None:
            return cached
        energy_axis = np.asarray(self.decomposer.energy_axis, dtype=float)
        response = compton_continuum_shape(energy_axis, cache_key, shape="exponential")
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        if cache_key > 200.0:
            e_back = backscatter_energy(cache_key)
            sigma_back = float(self.decomposer.resolution_fn(e_back))
            response += 0.25 * gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        self._scatter_response_cache[cache_key] = response
        return response

    def _material_for_prim(
        self,
        path: str,
        prim_material: StageMaterialInfo | None,
    ) -> StageMaterialInfo | None:
        """Resolve a material for a prim using authored metadata first."""
        if prim_material is not None:
            return prim_material
        matched_material: str | None = None
        matched_len = -1
        for rule in self.stage_material_rules:
            if path.startswith(rule.path_prefix) and len(rule.path_prefix) > matched_len:
                matched_material = rule.material
                matched_len = len(rule.path_prefix)
        if matched_material is None and path.startswith(self.robot_controller.prim_paths.obstacles_root):
            return StageMaterialInfo(name=self.scene.obstacle_material)
        if matched_material is None and path.startswith("/World/Environment"):
            return StageMaterialInfo(name="concrete")
        if matched_material is None:
            return None
        return StageMaterialInfo(name=matched_material)

    def _solid_path_length_m(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
        solid_prim: object,
    ) -> float:
        """Return the path length through a supported solid prim."""
        shape = getattr(solid_prim, "shape")
        pose = getattr(solid_prim, "pose")
        if shape == "box":
            size_xyz = getattr(solid_prim, "size_xyz")
            if size_xyz is None:
                return 0.0
            box = OrientedBox(
                center_xyz=pose.translation_xyz,
                size_xyz=size_xyz,
                rotation_matrix=quaternion_wxyz_to_matrix(pose.orientation_wxyz),
            )
            return float(segment_path_length_through_box(source_xyz, detector_xyz, box))
        if shape == "sphere":
            radius_m = getattr(solid_prim, "radius_m")
            if radius_m is None:
                return 0.0
            sphere = Sphere(center_xyz=pose.translation_xyz, radius_m=float(radius_m))
            return float(segment_path_length_through_sphere(source_xyz, detector_xyz, sphere))
        if shape == "mesh":
            triangles_xyz = getattr(solid_prim, "triangles_xyz")
            if not triangles_xyz:
                return 0.0
            mesh = TriangleMesh(triangles_xyz=triangles_xyz)
            return float(segment_path_length_through_mesh(source_xyz, detector_xyz, mesh))
        return 0.0
