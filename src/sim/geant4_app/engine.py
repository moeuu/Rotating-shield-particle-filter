"""Geant4-side observation engines and surrogate transport helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from measurement.model import PointSource
from sim.geant4_app.io_format import read_response_file, write_request_file, write_scene_file
from sim.geant4_app.scene_export import (
    ExportedDetectorModel,
    ExportedGeant4Scene,
    ExportedGeant4Volume,
    ExportedShieldModel,
)
from sim.radiation_visualization import (
    RadiationVisualizationConfig,
    build_visualization_metadata_from_scene,
    build_visualization_metadata_from_transport,
)
from sim.isaacsim_app.geometry import (
    OrientedBox,
    Sphere,
    TriangleMesh,
    quaternion_wxyz_to_matrix,
    segment_path_length_through_box,
    segment_path_length_through_mesh,
    segment_path_length_through_sphere,
)
from sim.transport import (
    SourceTransportResult,
    TransportSegment,
    build_source_transport_result,
    make_transport_segment,
)
from sim.shield_geometry import spherical_octant_path_length_cm
from spectrum.pipeline import BACKGROUND_COUNTS_PER_SECOND, BACKGROUND_RATE_CPS, SpectralDecomposer
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    backscatter_energy,
    compton_continuum_shape,
    gaussian_peak,
)


@dataclass(frozen=True)
class Geant4StepRequest:
    """Describe a single Geant4 step request."""

    step_id: int
    dwell_time_s: float
    seed: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_shield_pose_xyz: tuple[float, float, float]
    fe_shield_quat_wxyz: tuple[float, float, float, float]
    pb_shield_pose_xyz: tuple[float, float, float]
    pb_shield_quat_wxyz: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable request payload."""
        return {
            "step_id": int(self.step_id),
            "dwell_time_s": float(self.dwell_time_s),
            "seed": int(self.seed),
            "detector_pose_xyz": list(self.detector_pose_xyz),
            "detector_quat_wxyz": list(self.detector_quat_wxyz),
            "fe_shield_pose_xyz": list(self.fe_shield_pose_xyz),
            "fe_shield_quat_wxyz": list(self.fe_shield_quat_wxyz),
            "pb_shield_pose_xyz": list(self.pb_shield_pose_xyz),
            "pb_shield_quat_wxyz": list(self.pb_shield_quat_wxyz),
        }


@dataclass(frozen=True)
class Geant4EngineConfig:
    """Collect surrogate or external Geant4 engine settings."""

    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.03
    executable_path: str | None = None
    executable_args: tuple[str, ...] = ()
    timeout_s: float = 120.0
    radiation_visualization: RadiationVisualizationConfig = field(default_factory=RadiationVisualizationConfig)


class Geant4Engine(ABC):
    """Define the Geant4 engine interface used by the sidecar app."""

    @abstractmethod
    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Load a scene and return whether a cached world was reused."""

    @abstractmethod
    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one transport step and return a spectrum plus metadata."""

    @abstractmethod
    def close(self) -> None:
        """Release engine-owned resources."""


class SurrogateGeant4Engine(Geant4Engine):
    """Approximate Geant4 transport while preserving the final backend contract."""

    def __init__(self, config: Geant4EngineConfig) -> None:
        """Store engine configuration and initialize reusable helpers."""
        self.config = config
        self.scene: ExportedGeant4Scene | None = None
        self.decomposer = SpectralDecomposer()
        self._line_response_cache: dict[float, np.ndarray] = {}
        self._scatter_response_cache: dict[float, np.ndarray] = {}
        self._scene_hash: str | None = None
        self._last_cache_hit = False

    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Store the exported scene and reuse the cached world when hashes match."""
        cache_hit = scene.scene_hash == self._scene_hash
        self.scene = scene
        self._scene_hash = scene.scene_hash
        self._last_cache_hit = cache_hit
        return cache_hit

    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a surrogate Geant4 spectrum for the requested pose."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        started_at = time.perf_counter()
        rng = np.random.default_rng(int(request.seed))
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        total_obstacle_path_cm = 0.0
        total_stage_path_cm = 0.0
        total_scatter_counts = 0.0
        total_primary_mean = 0.0
        detector_scale = self._detector_scale(self.scene.detector_model)
        transport_results = self._build_source_transport_results(request)
        for transport_result in transport_results:
            total_obstacle_path_cm += transport_result.total_obstacle_path_cm
            total_stage_path_cm += transport_result.total_stage_path_cm
            total_primary_mean += float(sum(line.emission_counts for line in transport_result.lines))
            total_scatter_counts += float(sum(line.scatter_counts for line in transport_result.lines)) * detector_scale
            expected += self._source_expected_spectrum(transport_result, detector_scale=detector_scale)
        background_rate = BACKGROUND_RATE_CPS
        if BACKGROUND_COUNTS_PER_SECOND != BACKGROUND_RATE_CPS:
            background_rate = BACKGROUND_COUNTS_PER_SECOND
        if background_rate > 0.0:
            expected += self.decomposer._background_shape * float(background_rate) * float(request.dwell_time_s)
        expected *= self._dead_time_observed_scale(expected, request.dwell_time_s)
        spectrum = rng.poisson(np.clip(expected, 0.0, None))
        runtime_s = float(time.perf_counter() - started_at)
        metadata: dict[str, Any] = {
            "backend": "geant4",
            "engine_mode": "surrogate",
            "physics_profile": self.config.physics_profile,
            "scene_hash": self.scene.scene_hash,
            "cache_hit": self._last_cache_hit,
            "seed": int(request.seed),
            "run_time_s": runtime_s,
            "transport_backend": "geant4-surrogate",
            "num_primaries": int(np.round(total_primary_mean)),
            "mean_primaries": float(total_primary_mean),
            "total_obstacle_path_cm": float(total_obstacle_path_cm),
            "total_stage_path_cm": float(total_stage_path_cm),
            "total_scatter_counts": float(total_scatter_counts),
            "detector_scale": float(detector_scale),
        }
        metadata.update(
            build_visualization_metadata_from_transport(
                transport_results,
                seed=int(request.seed),
                config=self.config.radiation_visualization,
                mode="geant4-surrogate",
            )
        )
        if self.scene.usd_path:
            metadata["usd_path"] = self.scene.usd_path
        return np.asarray(spectrum, dtype=float), metadata

    def close(self) -> None:
        """Release surrogate-engine resources."""
        self.scene = None

    def _build_source_transport_results(
        self,
        request: Geant4StepRequest,
    ) -> tuple[SourceTransportResult, ...]:
        """Build shared pre-spectrum transport results for all exported sources."""
        if self.scene is None:
            return ()
        results: list[SourceTransportResult] = []
        for source in self.scene.sources:
            point_source = PointSource(
                isotope=source.isotope,
                position=source.position_xyz,
                intensity_cps_1m=source.intensity_cps_1m,
            )
            stage_segments = self._static_path_segments(source.position_xyz, request.detector_pose_xyz)
            fe_path_cm = self._shield_path_length_cm(
                source.position_xyz,
                request.detector_pose_xyz,
                self.scene.fe_shield,
                request.fe_shield_pose_xyz,
                request.fe_shield_quat_wxyz,
            )
            pb_path_cm = self._shield_path_length_cm(
                source.position_xyz,
                request.detector_pose_xyz,
                self.scene.pb_shield,
                request.pb_shield_pose_xyz,
                request.pb_shield_quat_wxyz,
            )
            nuclide = self.decomposer.library.get(source.isotope)
            nuclide_lines = ()
            if nuclide is not None:
                nuclide_lines = tuple((float(line.energy_keV), float(line.intensity)) for line in nuclide.lines)
            results.append(
                build_source_transport_result(
                    source=point_source,
                    detector_position_xyz=request.detector_pose_xyz,
                    dwell_time_s=float(request.dwell_time_s),
                    stage_segments=stage_segments,
                    fe_segment=make_transport_segment(self.scene.fe_shield.material, fe_path_cm),
                    pb_segment=make_transport_segment(self.scene.pb_shield.material, pb_path_cm),
                    nuclide_lines=nuclide_lines,
                    scatter_gain=self.config.scatter_gain,
                )
            )
        return tuple(results)

    def _static_path_segments(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return crossed static materials and their path lengths."""
        if self.scene is None:
            return ()
        segments: list[TransportSegment] = []
        for volume in self.scene.static_volumes:
            if volume.material is None:
                continue
            path_length_cm = 100.0 * self._static_volume_path_length_m(source_xyz, detector_xyz, volume)
            if path_length_cm <= 0.0:
                continue
            segments.append(
                make_transport_segment(
                    volume.material,
                    float(path_length_cm),
                    is_obstacle=self._static_volume_is_obstacle(volume),
                )
            )
        return tuple(segments)

    def _static_volume_is_obstacle(self, volume: ExportedGeant4Volume) -> bool:
        """Return whether a static stage volume should be reported as an obstacle."""
        if self.scene is None:
            return False
        path = str(volume.path)
        if path.startswith(self.scene.prim_paths.obstacles_root):
            return True
        if not path.startswith("/World/Environment"):
            return False
        material = volume.material
        material_name = "" if material is None else str(material.name).strip().lower()
        return material_name not in {"", "air", "vacuum"}

    def _static_volume_path_length_m(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
        volume: ExportedGeant4Volume,
    ) -> float:
        """Return the source-detector chord length through a static volume."""
        if volume.shape == "box" and volume.size_xyz is not None:
            box = OrientedBox(
                center_xyz=volume.translation_xyz,
                size_xyz=volume.size_xyz,
                rotation_matrix=quaternion_wxyz_to_matrix(volume.orientation_wxyz),
            )
            return float(segment_path_length_through_box(source_xyz, detector_xyz, box))
        if volume.shape == "sphere" and volume.radius_m is not None:
            sphere = Sphere(center_xyz=volume.translation_xyz, radius_m=float(volume.radius_m))
            return float(segment_path_length_through_sphere(source_xyz, detector_xyz, sphere))
        if volume.shape == "mesh" and volume.triangles_xyz is not None:
            mesh = TriangleMesh(triangles_xyz=volume.triangles_xyz)
            return float(segment_path_length_through_mesh(source_xyz, detector_xyz, mesh))
        return 0.0

    def _shield_path_length_cm(
        self,
        source_xyz: tuple[float, float, float],
        detector_xyz: tuple[float, float, float],
        shield: ExportedShieldModel,
        pose_xyz: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> float:
        """Return the Python reference path length through a movable shield."""
        if shield.shape != "spherical_octant_shell" and shield.size_xyz is not None:
            box = OrientedBox(
                center_xyz=pose_xyz,
                size_xyz=shield.size_xyz,
                rotation_matrix=quaternion_wxyz_to_matrix(quat_wxyz),
            )
            return float(100.0 * segment_path_length_through_box(source_xyz, detector_xyz, box))
        return spherical_octant_path_length_cm(
            source_xyz,
            detector_xyz,
            quat_wxyz,
            thickness_cm=shield.thickness_cm,
            use_angle_attenuation=shield.use_angle_attenuation,
        )

    def _detector_scale(self, detector_model: ExportedDetectorModel) -> float:
        """Return a detector-efficiency scale based on active crystal volume."""
        reference_model = ExportedDetectorModel()
        reference_volume = max(reference_model.active_volume_m3, 1e-12)
        active_volume = max(detector_model.active_volume_m3, 0.0)
        return float(np.clip(active_volume / reference_volume, 0.1, 8.0))

    def _source_expected_spectrum(
        self,
        transport_result: SourceTransportResult,
        *,
        detector_scale: float,
    ) -> np.ndarray:
        """Return the expected spectrum contribution for one transported source."""
        if not transport_result.lines or transport_result.base_source_counts <= 0.0:
            return np.zeros_like(self.decomposer.energy_axis, dtype=float)
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        for line in transport_result.lines:
            expected += float(line.primary_counts) * float(detector_scale) * self._line_response_template(line.energy_keV)
            if line.scatter_counts > 0.0:
                expected += float(line.scatter_counts) * float(detector_scale) * self._scatter_response_template(
                    line.energy_keV
                )
        return expected

    def _dead_time_observed_scale(self, expected: np.ndarray, dwell_time_s: float) -> float:
        """Return the observed-count scale under a non-paralyzable dead-time model."""
        if dwell_time_s <= 0.0:
            return 1.0
        true_rate = float(np.sum(expected)) / float(dwell_time_s)
        tau = max(0.0, float(self.config.dead_time_tau_s))
        if true_rate <= 0.0 or tau <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + true_rate * tau))

    def _line_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return the detector response for a transmitted full-energy line."""
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
                response += back * ((BACKSCATTER_FRACTION * peak_area) / back_norm) * float(
                    self.decomposer.efficiency_fn(e_back)
                )
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


class ExternalCommandGeant4Engine(Geant4Engine):
    """Delegate transport to an external executable through stdin/stdout JSON."""

    def __init__(self, config: Geant4EngineConfig) -> None:
        """Store external-engine launch configuration."""
        if config.executable_path in (None, ""):
            raise ValueError("executable_path is required for the external Geant4 engine.")
        self.config = config
        self.scene: ExportedGeant4Scene | None = None
        self._last_cache_hit = False
        self.decomposer = SpectralDecomposer()

    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Store the latest scene for the next external simulation call."""
        cache_hit = self.scene is not None and self.scene.scene_hash == scene.scene_hash
        self.scene = scene
        self._last_cache_hit = cache_hit
        return cache_hit

    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Call the configured external executable and parse its response."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        with tempfile.TemporaryDirectory(prefix="geant4_sidecar_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_path = tmp_path / "scene.txt"
            request_path = tmp_path / "request.txt"
            response_path = tmp_path / "response.txt"
            write_scene_file(self.scene, scene_path)
            write_request_file(request, request_path)
            command = [
                str(self.config.executable_path),
                "--scene",
                scene_path.as_posix(),
                "--request",
                request_path.as_posix(),
                "--response",
                response_path.as_posix(),
                "--physics-profile",
                self.config.physics_profile,
                "--threads",
                str(self.config.thread_count),
                "--dead-time-tau-s",
                str(self.config.dead_time_tau_s),
                *self.config.executable_args,
            ]
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "External Geant4 executable failed: "
                    f"returncode={result.returncode} stderr={result.stderr.strip()}"
                )
            spectrum, metadata = read_response_file(response_path)
        metadata.setdefault("backend", "geant4")
        metadata.setdefault("engine_mode", "external")
        metadata.setdefault("scene_hash", self.scene.scene_hash)
        metadata.setdefault("cache_hit", self._last_cache_hit)
        metadata.setdefault("seed", int(request.seed))
        metadata.update(
            build_visualization_metadata_from_scene(
                self.scene,
                request,
                seed=int(request.seed),
                config=self.config.radiation_visualization,
                library=self.decomposer.library,
                mode="geant4-external-representative",
                scatter_gain=self.config.scatter_gain,
            )
        )
        return spectrum, metadata

    def close(self) -> None:
        """Release the cached scene reference."""
        self.scene = None


def build_geant4_engine(config: Geant4EngineConfig, *, engine_mode: str) -> Geant4Engine:
    """Instantiate the requested Geant4 engine implementation."""
    normalized = engine_mode.strip().lower()
    if normalized == "surrogate":
        return SurrogateGeant4Engine(config)
    if normalized == "external":
        return ExternalCommandGeant4Engine(config)
    raise ValueError(f"Unsupported Geant4 engine mode: {engine_mode}")
