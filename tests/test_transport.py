"""Tests for the shared pre-spectrum transport layer."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.model import PointSource
from measurement.shielding import OCTANT_NORMALS, path_length_cm
from sim.geant4_app.app import Geant4Application
from sim.geant4_app.engine import Geant4StepRequest, SurrogateGeant4Engine
from sim.isaacsim_app.app import IsaacSimApplication
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend, StageMaterialInfo
from sim.protocol import SimulationCommand
from sim.shield_geometry import FE_SHIELD_THICKNESS_CM, shield_normal_from_quaternion_wxyz
from sim.transport import build_source_transport_result, make_transport_segment


def test_build_source_transport_result_tracks_obstacle_and_scatter() -> None:
    """The shared transport result should expose path totals and line statistics."""
    source = PointSource(isotope="Cs-137", position=(4.0, 4.0, 1.0), intensity_cps_1m=1000.0)
    stage_segments = (
        make_transport_segment(StageMaterialInfo(name="concrete"), 15.0, is_obstacle=True),
    )
    result = build_source_transport_result(
        source=source,
        detector_position_xyz=(4.0, 1.0, 1.0),
        dwell_time_s=10.0,
        stage_segments=stage_segments,
        fe_segment=make_transport_segment(StageMaterialInfo(name="fe"), 5.0),
        pb_segment=make_transport_segment(StageMaterialInfo(name="pb"), 2.0),
        nuclide_lines=((662.0, 0.85),),
        scatter_gain=0.12,
    )

    assert result.total_obstacle_path_cm == pytest.approx(15.0)
    assert result.total_stage_path_cm == pytest.approx(15.0)
    assert result.total_fe_path_cm == pytest.approx(5.0)
    assert result.total_pb_path_cm == pytest.approx(2.0)
    assert len(result.lines) == 1
    assert result.lines[0].total_transmission <= 1.0
    assert result.lines[0].scatter_counts >= 0.0


def test_python_and_geant4_surrogate_share_pre_spectrum_transport() -> None:
    """Isaac and Geant4 surrogate backends should agree on the shared transport result."""
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[(4, 2), (4, 3)],
        sources=[
            SourceDescription(
                isotope="Cs-137",
                position_xyz=(4.5, 4.5, 1.0),
                intensity_cps_1m=5.0e5,
            )
        ],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=7,
        target_pose_xyz=(4.5, 1.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=1,
        pb_orientation_index=2,
        dwell_time_s=10.0,
    )

    isaac_app = IsaacSimApplication(
        use_mock=False,
        app_config={"usd_path": "demo_room.usda", "detector_height_m": 0.5},
        stage_backend=FakeStageBackend(),
    )
    isaac_app.reset(scene)
    assert isaac_app.robot_controller is not None
    isaac_app.robot_controller.apply_command(command)
    python_results = isaac_app.observation_model._build_source_transport_results(command)  # type: ignore[attr-defined]
    isaac_app.close()

    geant4_app = Geant4Application(
        app_config={"usd_path": "demo_room.usda", "use_mock_stage": True},
        stage_backend=FakeStageBackend(),
    )
    geant4_app.reset(scene)
    geant4_app.robot_controller.apply_command(command)
    assert isinstance(geant4_app.engine, SurrogateGeant4Engine)
    detector_pose = geant4_app.robot_controller.detector_world_pose()
    fe_pose = geant4_app._stage_backend.get_world_pose(scene.prim_paths.fe_shield_path)
    pb_pose = geant4_app._stage_backend.get_world_pose(scene.prim_paths.pb_shield_path)
    request = Geant4StepRequest(
        step_id=command.step_id,
        dwell_time_s=float(command.dwell_time_s),
        seed=123 + int(command.step_id),
        detector_pose_xyz=detector_pose.translation_xyz,
        detector_quat_wxyz=detector_pose.orientation_wxyz,
        fe_shield_pose_xyz=fe_pose.translation_xyz,
        fe_shield_quat_wxyz=fe_pose.orientation_wxyz,
        pb_shield_pose_xyz=pb_pose.translation_xyz,
        pb_shield_quat_wxyz=pb_pose.orientation_wxyz,
    )
    geant4_results = geant4_app.engine._build_source_transport_results(request)
    geant4_app.close()

    assert len(python_results) == 1
    assert len(geant4_results) == 1
    py_result = python_results[0]
    g4_result = geant4_results[0]

    assert py_result.total_obstacle_path_cm == pytest.approx(g4_result.total_obstacle_path_cm)
    assert py_result.total_stage_path_cm == pytest.approx(g4_result.total_stage_path_cm)
    assert py_result.total_fe_path_cm == pytest.approx(g4_result.total_fe_path_cm)
    assert py_result.total_pb_path_cm == pytest.approx(g4_result.total_pb_path_cm)
    assert len(py_result.lines) == len(g4_result.lines)
    assert py_result.lines[0].energy_keV == pytest.approx(g4_result.lines[0].energy_keV)
    assert py_result.lines[0].stage_transmission == pytest.approx(g4_result.lines[0].stage_transmission)
    assert py_result.lines[0].shield_transmission == pytest.approx(g4_result.lines[0].shield_transmission)
    assert py_result.lines[0].total_transmission == pytest.approx(g4_result.lines[0].total_transmission)


def test_geant4_surrogate_shield_path_matches_python_octant_model() -> None:
    """Geant4 surrogate shield lengths should match the reference Python model."""
    source = SourceDescription(
        isotope="Cs-137",
        position_xyz=(5.0, 5.0, 1.0),
        intensity_cps_1m=1.0e5,
    )
    scene = SceneDescription(
        room_size_xyz=(10.0, 20.0, 3.0),
        sources=[source],
        usd_path="demo_room.usda",
    )
    command = SimulationCommand(
        step_id=8,
        target_pose_xyz=(4.0, 4.0, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=7,
        pb_orientation_index=0,
        dwell_time_s=1.0,
    )
    geant4_app = Geant4Application(
        app_config={"usd_path": "demo_room.usda", "use_mock_stage": True},
        stage_backend=FakeStageBackend(),
    )
    geant4_app.reset(scene)
    geant4_app.robot_controller.apply_command(command)
    assert isinstance(geant4_app.engine, SurrogateGeant4Engine)
    assert geant4_app.engine.scene is not None
    detector_pose = geant4_app.robot_controller.detector_world_pose()
    fe_pose = geant4_app._stage_backend.get_world_pose(scene.prim_paths.fe_shield_path)

    measured_cm = geant4_app.engine._shield_path_length_cm(
        source.position_xyz,
        detector_pose.translation_xyz,
        geant4_app.engine.scene.fe_shield,
        fe_pose.translation_xyz,
        fe_pose.orientation_wxyz,
    )
    normal = np.asarray(shield_normal_from_quaternion_wxyz(fe_pose.orientation_wxyz), dtype=float)
    direction = np.asarray(source.position_xyz, dtype=float) - np.asarray(
        detector_pose.translation_xyz,
        dtype=float,
    )
    expected_cm = path_length_cm(
        direction,
        normal,
        FE_SHIELD_THICKNESS_CM,
    )
    geant4_app.close()

    assert normal == pytest.approx(-OCTANT_NORMALS[7])
    assert measured_cm == pytest.approx(expected_cm)
    assert measured_cm == pytest.approx(FE_SHIELD_THICKNESS_CM)
