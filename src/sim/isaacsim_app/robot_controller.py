"""Robot state and prim control helpers for the Isaac Sim sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, ceil
import time

import numpy as np

from measurement.shielding import generate_octant_orientations
from sim.isaacsim_app.scene_builder import StagePrimPaths
from sim.isaacsim_app.stage_backend import (
    PrimPose,
    StageBackend,
    yaw_to_quaternion_wxyz,
)
from sim.protocol import SimulationCommand


ROBOT_MAST_SIZE_XY_M = (0.08, 0.08)
MIN_VISUAL_MAST_HEIGHT_M = 1.0e-6


def _normalize_vector(
    vector_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalize a 3D vector and fall back to +X for degenerate inputs."""
    vector = np.asarray(vector_xyz, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return (1.0, 0.0, 0.0)
    vector /= norm
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _rotation_between_vectors_wxyz(
    source_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a quaternion rotating one direction onto another direction."""
    source = np.asarray(_normalize_vector(source_xyz), dtype=float)
    target = np.asarray(_normalize_vector(target_xyz), dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        fallback = np.asarray((0.0, 0.0, 1.0), dtype=float)
        if abs(float(np.dot(source, fallback))) > 0.9:
            fallback = np.asarray((0.0, 1.0, 0.0), dtype=float)
        axis = np.cross(source, fallback)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return (0.0, float(axis[0]), float(axis[1]), float(axis[2]))
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    axis = cross / cross_norm
    half_angle = 0.5 * acos(dot)
    sin_half = np.sin(half_angle)
    return (
        float(np.cos(half_angle)),
        float(axis[0] * sin_half),
        float(axis[1] * sin_half),
        float(axis[2] * sin_half),
    )


def _movement_yaw_rad(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    fallback_yaw_rad: float,
) -> float:
    """Return a planar heading yaw from start to target, or a fallback for tiny moves."""
    delta = np.asarray(target_xyz, dtype=float) - np.asarray(start_xyz, dtype=float)
    if float(np.linalg.norm(delta[:2])) <= 1e-9:
        return float(fallback_yaw_rad)
    return float(atan2(delta[1], delta[0]))


def _detector_waypoint_path(
    start_detector_pose: np.ndarray,
    target_detector_pose: np.ndarray,
    travel_waypoints_xyz: tuple[tuple[float, float, float], ...] | None,
) -> np.ndarray:
    """Return a deduplicated detector path that starts and ends at command poses."""
    path_points = [np.asarray(start_detector_pose, dtype=float).reshape(3)]
    if travel_waypoints_xyz:
        path_points.extend(
            np.asarray(waypoint, dtype=float).reshape(3)
            for waypoint in travel_waypoints_xyz
        )
    path_points.append(np.asarray(target_detector_pose, dtype=float).reshape(3))
    deduplicated = [path_points[0]]
    for point in path_points[1:]:
        if float(np.linalg.norm(point - deduplicated[-1])) > 1.0e-9:
            deduplicated.append(point)
    return np.vstack(deduplicated).astype(float)


def _allocate_segment_steps(
    segment_lengths_m: np.ndarray,
    requested_steps: int,
) -> np.ndarray:
    """Allocate interpolation steps by segment length while retaining each waypoint."""
    lengths = np.asarray(segment_lengths_m, dtype=float).reshape(-1)
    if lengths.size == 0:
        return np.zeros(0, dtype=np.int64)
    step_budget = max(int(requested_steps), int(lengths.size))
    total_length = float(np.sum(lengths))
    if total_length <= 1.0e-12:
        return np.ones(lengths.size, dtype=np.int64)
    ideal = lengths * float(step_budget) / total_length
    allocated = np.maximum(np.floor(ideal).astype(np.int64), 1)
    while int(np.sum(allocated)) < step_budget:
        residual = ideal - allocated
        allocated[int(np.argmax(residual))] += 1
    while int(np.sum(allocated)) > step_budget:
        reducible = allocated > 1
        if not np.any(reducible):
            break
        excess = allocated - ideal
        excess[~reducible] = -np.inf
        allocated[int(np.argmax(excess))] -= 1
    return allocated


def _interpolate_detector_path(
    detector_path_xyz: np.ndarray,
    requested_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return detector samples paired with their active path-segment starts."""
    path = np.asarray(detector_path_xyz, dtype=float).reshape(-1, 3)
    if path.shape[0] <= 1:
        return [(path[0], path[0])]
    segment_vectors = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    steps_by_segment = _allocate_segment_steps(segment_lengths, requested_steps)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for segment_index, segment_steps in enumerate(steps_by_segment):
        segment_start = path[segment_index]
        segment_end = path[segment_index + 1]
        for step_index in range(1, int(segment_steps) + 1):
            fraction = float(step_index) / float(segment_steps)
            sample = segment_start + fraction * (segment_end - segment_start)
            samples.append((sample, segment_start))
    return samples


@dataclass
class RobotState:
    """Track the latest robot and shield command state."""

    pose_xyz: tuple[float, float, float] = (1.0, 1.0, 0.0)
    detector_world_z_m: float = 0.5
    base_yaw_rad: float = 0.0
    fe_orientation_index: int = 0
    pb_orientation_index: int = 0

    @property
    def detector_pose_xyz(self) -> tuple[float, float, float]:
        """Return the detector world position represented by this state."""
        return (
            float(self.pose_xyz[0]),
            float(self.pose_xyz[1]),
            float(self.detector_world_z_m),
        )

    def apply_command(self, command: SimulationCommand, *, ground_z_m: float) -> None:
        """Update the grounded base and detector height from a command."""
        self.pose_xyz = (
            float(command.target_pose_xyz[0]),
            float(command.target_pose_xyz[1]),
            float(ground_z_m),
        )
        self.detector_world_z_m = float(command.target_pose_xyz[2])
        self.base_yaw_rad = float(command.target_base_yaw_rad)
        self.fe_orientation_index = int(command.fe_orientation_index)
        self.pb_orientation_index = int(command.pb_orientation_index)


class RobotController:
    """Apply simulator commands onto robot helper prims."""

    def __init__(
        self,
        stage_backend: StageBackend,
        prim_paths: StagePrimPaths,
        *,
        detector_height_m: float = 0.5,
        fe_offset_xyz: tuple[float, float, float] = (0.25, 0.0, 0.5),
        pb_offset_xyz: tuple[float, float, float] = (-0.25, 0.0, 0.5),
        ground_z_m: float = 0.0,
        motion_speed_m_s: float = 0.5,
        animation_dt_s: float = 0.2,
        animation_time_scale: float = 0.0,
        max_animation_steps: int = 200,
    ) -> None:
        """Store stage handles and shield orientation references."""
        self.stage_backend = stage_backend
        self.prim_paths = prim_paths
        self.detector_height_m = float(detector_height_m)
        self.fe_offset_xyz = tuple(float(v) for v in fe_offset_xyz)
        self.pb_offset_xyz = tuple(float(v) for v in pb_offset_xyz)
        self.ground_z_m = float(ground_z_m)
        self.motion_speed_m_s = max(float(motion_speed_m_s), 1e-9)
        self.animation_dt_s = max(float(animation_dt_s), 1e-3)
        self.animation_time_scale = max(float(animation_time_scale), 0.0)
        self.max_animation_steps = max(int(max_animation_steps), 1)
        self.state = RobotState(
            pose_xyz=(1.0, 1.0, self.ground_z_m),
            detector_world_z_m=self.ground_z_m + self.detector_height_m,
        )
        self._shield_normals = [
            tuple(float(v) for v in normal) for normal in generate_octant_orientations()
        ]

    def reset(self) -> None:
        """Reset the cached robot state and publish the initial robot pose."""
        self.state = RobotState(
            pose_xyz=(1.0, 1.0, self.ground_z_m),
            detector_world_z_m=self.ground_z_m + self.detector_height_m,
        )
        self._apply_pose_and_shields(
            base_pose_xyz=self.state.pose_xyz,
            detector_world_z_m=self.state.detector_world_z_m,
            base_yaw_rad=self.state.base_yaw_rad,
            fe_orientation_index=self.state.fe_orientation_index,
            pb_orientation_index=self.state.pb_orientation_index,
        )

    def apply_command(self, command: SimulationCommand) -> None:
        """Move the grounded base and detector mast to a commanded detector pose."""
        start_detector_pose = np.asarray(self.state.detector_pose_xyz, dtype=float)
        target_detector_pose = np.asarray(command.target_pose_xyz, dtype=float)
        detector_path = _detector_waypoint_path(
            start_detector_pose,
            target_detector_pose,
            command.travel_waypoints_xyz,
        )
        distance_m = float(
            np.sum(np.linalg.norm(np.diff(detector_path, axis=0), axis=1))
        )
        travel_time_s = float(command.travel_time_s)
        if travel_time_s <= 0.0 and distance_m > 0.0:
            travel_time_s = distance_m / self.motion_speed_m_s
        steps = 1
        if travel_time_s > 0.0 and distance_m > 0.0:
            steps = min(
                max(int(ceil(travel_time_s / self.animation_dt_s)), 1),
                self.max_animation_steps,
            )
        detector_samples = _interpolate_detector_path(detector_path, steps)
        for detector_sample, segment_start in detector_samples:
            interp_base_pose = (
                float(detector_sample[0]),
                float(detector_sample[1]),
                self.ground_z_m,
            )
            motion_yaw = _movement_yaw_rad(
                segment_start,
                detector_sample,
                fallback_yaw_rad=float(command.target_base_yaw_rad),
            )
            self._apply_pose_and_shields(
                base_pose_xyz=interp_base_pose,
                detector_world_z_m=float(detector_sample[2]),
                base_yaw_rad=motion_yaw,
                fe_orientation_index=int(command.fe_orientation_index),
                pb_orientation_index=int(command.pb_orientation_index),
            )
            if self.animation_time_scale > 0.0 and len(detector_samples) > 1:
                time.sleep(self.animation_dt_s * self.animation_time_scale)
        self.stage_backend.set_local_pose(
            self.prim_paths.robot_root,
            orientation_wxyz=yaw_to_quaternion_wxyz(float(command.target_base_yaw_rad)),
        )
        self.stage_backend.step()
        self.state.apply_command(command, ground_z_m=self.ground_z_m)

    def _apply_pose_and_shields(
        self,
        *,
        base_pose_xyz: tuple[float, float, float],
        detector_world_z_m: float,
        base_yaw_rad: float,
        fe_orientation_index: int,
        pb_orientation_index: int,
    ) -> None:
        """Apply one grounded base pose and mast-height sample to the stage."""
        detector_local_z_m = float(detector_world_z_m) - float(base_pose_xyz[2])
        fe_translation = (
            float(self.fe_offset_xyz[0]),
            float(self.fe_offset_xyz[1]),
            detector_local_z_m + float(self.fe_offset_xyz[2]) - self.detector_height_m,
        )
        pb_translation = (
            float(self.pb_offset_xyz[0]),
            float(self.pb_offset_xyz[1]),
            detector_local_z_m + float(self.pb_offset_xyz[2]) - self.detector_height_m,
        )
        self.stage_backend.set_local_pose(
            self.prim_paths.robot_root,
            translation_xyz=base_pose_xyz,
            orientation_wxyz=yaw_to_quaternion_wxyz(base_yaw_rad),
        )
        mast_height_m = max(detector_local_z_m, MIN_VISUAL_MAST_HEIGHT_M)
        self.stage_backend.set_local_pose(
            self.prim_paths.robot_mast_path,
            translation_xyz=(0.0, 0.0, 0.5 * mast_height_m),
            scale_xyz=(
                float(ROBOT_MAST_SIZE_XY_M[0]),
                float(ROBOT_MAST_SIZE_XY_M[1]),
                mast_height_m,
            ),
        )
        self.stage_backend.set_local_pose(
            self.prim_paths.detector_path,
            translation_xyz=(0.0, 0.0, detector_local_z_m),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        # PF octant indices describe the incoming source-to-detector direction.
        # The physical shell occupies the opposite detector-to-source side.
        fe_normal = -np.asarray(
            self._shield_normals[fe_orientation_index % len(self._shield_normals)],
            dtype=float,
        )
        pb_normal = -np.asarray(
            self._shield_normals[pb_orientation_index % len(self._shield_normals)],
            dtype=float,
        )
        local_octant_center = _normalize_vector((1.0, 1.0, 1.0))
        self.stage_backend.set_local_pose(
            self.prim_paths.fe_shield_path,
            translation_xyz=fe_translation,
            orientation_wxyz=_rotation_between_vectors_wxyz(
                local_octant_center, fe_normal
            ),
        )
        self.stage_backend.set_local_pose(
            self.prim_paths.pb_shield_path,
            translation_xyz=pb_translation,
            orientation_wxyz=_rotation_between_vectors_wxyz(
                local_octant_center, pb_normal
            ),
        )
        self.stage_backend.step()

    def detector_world_pose(self) -> PrimPose:
        """Return the world pose of the detector prim."""
        return self.stage_backend.get_world_pose(self.prim_paths.detector_path)
