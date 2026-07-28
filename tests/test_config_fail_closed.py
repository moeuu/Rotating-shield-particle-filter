"""Regression tests for fail-closed runtime and RA-L configuration parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.ral_ablation.config_factory import (
    DEFAULT_ABLATION_CASES,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_BASE_CONFIG,
    _load_json,
    _parallel_runtime_overrides,
    _variant_config,
)
from baselines.ral_ablation.path_policies import select_baseline_next_pose
from baselines.ral_ablation.shield_policies import (
    select_baseline_shield_program,
)
from planning.traversability import (
    TraversabilityMap,
    build_traversability_map_from_obstacle_grid,
    build_traversability_map_from_stage_solids,
)
from measurement.obstacles import ObstacleGrid
from sim.geant4_app.app import Geant4AppConfig
from sim.geant4_app.io_format import write_scene_file
from sim.geant4_app.scene_export import (
    ExportedDetectorModel,
    ExportedGeant4Material,
    ExportedShieldModel,
    export_scene_for_geant4,
)
from sim.isaacsim_app.app import IsaacSimAppConfig, IsaacSimApplication
from sim.isaacsim_app.observation_model import IsaacAssetGeometry
from sim.isaacsim_app.scene_builder import (
    SceneDescription,
    build_scene_description,
)
from sim.isaacsim_app.stage_backend import (
    FakeStageBackend,
    PrimPose,
    StageSolidPrim,
)
from sim.shield_geometry import (
    SHIELD_SHAPE_SPHERICAL_OCTANT,
    ShieldThicknessConfig,
    require_no_angle_attenuation,
    spherical_octant_path_length_cm,
)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("thread_count", "32"),
        ("thread_count", 32.0),
        ("random_seed_base", True),
        ("timeout_s", "120"),
        ("dead_time_tau_s", False),
        ("detector_height_m", "0.5"),
        ("background_cps", "12"),
    ),
)
def test_geant4_config_rejects_scalar_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """Physics settings must not change meaning through Python coercion."""
    with pytest.raises(ValueError):
        Geant4AppConfig.from_dict({field_name: invalid})


@pytest.mark.parametrize(
    "payload",
    (
        {"engine_mode": "EXTERNAL"},
        {"physics_profile": "theory_tvl"},
        {"detector_model": {"crystal_shape": "cylinder"}},
        {"fe_shield_size_xyz": [0.25, "0.08", 0.25]},
        {"executable_args": [123]},
        {"absorbing_transport_groups": [1]},
        {"scatter_gain": 0.03},
        {"executable_path": None},
        {"source_bias_cone_half_angle_deg": 181.0},
    ),
)
def test_geant4_config_rejects_aliases_and_physics_fallbacks(
    payload: dict[str, object],
) -> None:
    """Unsupported geometry and aliases must fail before native transport."""
    with pytest.raises(ValueError):
        Geant4AppConfig.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"headless": "false"},
        {"author_obstacle_prims": 1},
        {"detector_height_m": "0.5"},
        {"robot_max_animation_steps": 10.0},
        {"detector_model": None},
        {"detector_model": {"crystal_shape": "cylinder"}},
        {"usd_path": ""},
    ),
)
def test_isaac_config_rejects_truthiness_and_numeric_coercion(
    payload: dict[str, object],
) -> None:
    """Isaac geometry settings must preserve exact JSON semantics."""
    with pytest.raises((TypeError, ValueError)):
        IsaacSimAppConfig.from_dict(payload)


def test_isaac_reset_preserves_configured_shield_geometry() -> None:
    """Resetting a real-mode app must not restore default shield thickness."""
    application = IsaacSimApplication(
        use_mock=False,
        app_config={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
        },
        stage_backend=FakeStageBackend(),
    )

    application.reset(SceneDescription())

    assert application.observation_model.shield_thickness.thickness_fe_cm == 0.0
    assert application.observation_model.shield_thickness.thickness_pb_cm == 0.0


@pytest.mark.parametrize("value", (1, 0, "false", None, True))
def test_production_shield_rejects_angle_attenuation_coercion(
    value: object,
) -> None:
    """Native and PF shield geometry must share exact false semantics."""
    with pytest.raises((TypeError, ValueError)):
        require_no_angle_attenuation(value)
    with pytest.raises((TypeError, ValueError)):
        spherical_octant_path_length_cm(
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            thickness_cm=1.0,
            use_angle_attenuation=value,
        )


@pytest.mark.parametrize("value", (1, "false", True))
def test_exported_shield_rejects_nonproduction_angle_contract(
    value: object,
) -> None:
    """Scene export must not serialize a shield option ignored by Geant4."""
    with pytest.raises((TypeError, ValueError)):
        ExportedShieldModel(
            path="/World/Shield",
            shape=SHIELD_SHAPE_SPHERICAL_OCTANT,
            inner_radius_m=0.05,
            outer_radius_m=0.06,
            thickness_cm=1.0,
            size_xyz=None,
            material=ExportedGeant4Material(name="lead"),
            use_angle_attenuation=value,
        )


def test_zero_thickness_shields_are_absent_from_native_scene(
    tmp_path: Path,
) -> None:
    """A no-shield baseline must export no material shell of either kind."""
    scene = export_scene_for_geant4(
        SceneDescription(),
        stage_backend=FakeStageBackend(),
        asset_geometry=IsaacAssetGeometry(),
        detector_model=ExportedDetectorModel(),
        shield_thickness=ShieldThicknessConfig(
            thickness_fe_cm=0.0,
            thickness_pb_cm=0.0,
            thickness_scale=0.0,
            transmission_target=1.0,
        ),
    )

    assert scene.fe_shield is None
    assert scene.pb_shield is None
    output_path = tmp_path / "no_shield.scene"
    write_scene_file(scene, output_path)
    assert "\nSHIELD " not in output_path.read_text(encoding="utf-8")


def test_native_shield_export_is_individually_nullable() -> None:
    """Fe and Pb volumes must be omitted independently at zero thickness."""
    scene = export_scene_for_geant4(
        SceneDescription(),
        stage_backend=FakeStageBackend(),
        asset_geometry=IsaacAssetGeometry(),
        detector_model=ExportedDetectorModel(),
        shield_thickness=ShieldThicknessConfig(
            thickness_fe_cm=0.0,
            thickness_pb_cm=1.0,
            thickness_scale=1.0,
            transmission_target=None,
        ),
    )

    assert scene.fe_shield is None
    assert scene.pb_shield is not None
    assert scene.pb_shield.outer_radius_m == (
        scene.pb_shield.inner_radius_m
        + scene.pb_shield.thickness_cm / 100.0
    )


def test_exported_shield_rejects_radius_thickness_disagreement() -> None:
    """The parser must never repair a physically different outer radius."""
    with pytest.raises(ValueError, match="outer_radius_m"):
        ExportedShieldModel(
            path="/World/Shield",
            shape=SHIELD_SHAPE_SPHERICAL_OCTANT,
            inner_radius_m=0.05,
            outer_radius_m=0.06,
            thickness_cm=1.0,
            size_xyz=None,
            material=ExportedGeant4Material(name="lead"),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"author_obstacle_prims": "false"},
        {"obstacle_grid_shape": [2.5, 3]},
        {"obstacle_cell_size_m": "1.0"},
        {"collision_boxes_m": [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]},
        {
            "obstacle_grid_shape": [2, 2],
            "obstacle_cells": [[2, 0]],
        },
        {"usd_path": ""},
    ),
)
def test_scene_reset_rejects_geometry_coercion(
    payload: dict[str, object],
) -> None:
    """A malformed reset must not create a different obstacle scene."""
    with pytest.raises((TypeError, ValueError)):
        build_scene_description(payload)


@pytest.mark.parametrize(
    "policy",
    (
        "fixed",
        {"name": "fixed_shield", "fixed_pair_id": 0},
        {"name": "round-robin", "start_pair_id": 0},
    ),
)
def test_shield_policy_rejects_retired_aliases(policy: object) -> None:
    """Only canonical object-form shield policies are accepted."""
    with pytest.raises((TypeError, ValueError)):
        select_baseline_shield_program(
            policy,
            total_pairs=64,
            program_length=8,
            pose_index=0,
        )


def test_shield_policy_rejects_pair_wrapping_and_truthy_flags() -> None:
    """Out-of-range pairs and truthy strings must never select another program."""
    with pytest.raises(ValueError, match="fixed_pair_id"):
        select_baseline_shield_program(
            {"name": "fixed", "fixed_pair_id": 64},
            total_pairs=64,
            program_length=8,
            pose_index=0,
        )
    with pytest.raises(ValueError, match="JSON boolean"):
        select_baseline_shield_program(
            {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": "false",
            },
            total_pairs=64,
            program_length=8,
            pose_index=0,
        )


def test_passive_serpentine_cycles_instead_of_sticking_to_last_row() -> None:
    """Long passive missions must not remain absorbed at the final row."""
    candidates = np.asarray(
        (
            (10.0, 5.0, 0.5),
            (10.0, 10.0, 0.5),
        ),
        dtype=float,
    )
    visited = np.zeros((4, 3), dtype=float)

    selection = select_baseline_next_pose(
        {"name": "passive_serpentine", "row_count": 3},
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.asarray((0.0, 0.0, 0.5)),
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.asarray((0.0, 0.0, 0.5)),
            np.asarray((10.0, 10.0, 0.5)),
        ),
    )

    assert selection is not None
    assert selection.candidate_index == 0


@pytest.mark.parametrize(
    "base_config",
    (
        {"python_worker_count": "32"},
        {"python_worker_count": 0},
        {"python_worker_count": 32, "thread_count": 0},
        {"python_worker_count": 32, "pose_selection_workers": 1.5},
    ),
)
def test_ral_factory_rejects_worker_clamping(
    base_config: dict[str, object],
) -> None:
    """Invalid worker settings must not silently become one worker."""
    with pytest.raises(ValueError):
        _parallel_runtime_overrides(base_config)


def test_ral_factory_rejects_non_object_metadata() -> None:
    """Paper provenance must not disappear when metadata has the wrong type."""
    base = _load_json(DEFAULT_BASE_CONFIG)
    base["metadata"] = []

    with pytest.raises(ValueError, match="metadata"):
        _variant_config(
            base,
            base_config_path=DEFAULT_BASE_CONFIG,
            case=DEFAULT_ABLATION_CASES[0],
            variant=DEFAULT_ABLATION_VARIANTS[0],
            seed=123,
            output_tag="metadata_contract",
        )


def test_traversability_json_requires_complete_exact_schema() -> None:
    """Missing map geometry must not fall back to another coordinate system."""
    payload = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        traversable_cells=((0, 0),),
        robot_radius_m=0.35,
    ).to_dict()
    payload.pop("cell_size")
    with pytest.raises(ValueError, match="schema mismatch"):
        TraversabilityMap.from_dict(payload)

    invalid = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        traversable_cells=((0, 0),),
        robot_radius_m=0.35,
    ).to_dict()
    invalid["grid_shape"] = ["2", 2]
    with pytest.raises(ValueError, match="JSON integer"):
        TraversabilityMap.from_dict(invalid)


def test_traversability_rejects_blocked_reachable_origin() -> None:
    """A blocked robot start must fail instead of producing an empty map."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0),),
    )

    with pytest.raises(ValueError, match="reachable_from"):
        build_traversability_map_from_obstacle_grid(
            grid,
            robot_radius_m=0.0,
            reachable_from=(0.5, 0.5),
        )


@pytest.mark.parametrize(
    "solid",
    (
        StageSolidPrim(
            path="/World/Unknown",
            shape="capsule",
            pose=PrimPose(),
        ),
        StageSolidPrim(
            path="/World/Box",
            shape="box",
            pose=PrimPose(),
            size_xyz=None,
        ),
        StageSolidPrim(
            path="/World/Sphere",
            shape="sphere",
            pose=PrimPose(),
            radius_m=0.0,
        ),
        StageSolidPrim(
            path="/World/Mesh",
            shape="mesh",
            pose=PrimPose(),
            triangles_xyz=(),
        ),
    ),
)
def test_traversability_rejects_unrepresentable_stage_solids(
    solid: StageSolidPrim,
) -> None:
    """Malformed solids must not disappear from the planner obstacle map."""
    with pytest.raises(ValueError):
        build_traversability_map_from_stage_solids(
            [solid],
            origin=(0.0, 0.0),
            cell_size=1.0,
            grid_shape=(2, 2),
            robot_radius_m=0.0,
        )


@pytest.mark.parametrize(
    "blocking_range",
    (
        (2.0, 0.05),
        (0.05, float("nan")),
        ("0.05", 2.0),
    ),
)
def test_traversability_rejects_invalid_blocking_height_range(
    blocking_range: tuple[object, object],
) -> None:
    """Invalid height filters must not silently remove all obstacles."""
    solid = StageSolidPrim(
        path="/World/Box",
        shape="box",
        pose=PrimPose(translation_xyz=(1.0, 1.0, 1.0)),
        size_xyz=(1.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError):
        build_traversability_map_from_stage_solids(
            [solid],
            origin=(0.0, 0.0),
            cell_size=1.0,
            grid_shape=(2, 2),
            robot_radius_m=0.0,
            blocking_z_range_m=blocking_range,
        )
