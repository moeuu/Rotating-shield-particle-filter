"""Contracts for exact surface anchors and air-side transport positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    expected_air_facing_normal,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
    validate_air_facing_surface_normals,
)
from measurement.source_surfaces import generate_surface_sources
from measurement.surface_charts import build_surface_chart_geometry
from pf.surface_atlas import ContinuousSurfaceAtlas
from sim.geant4_app.engine import validate_native_scene_identity
from sim.geant4_app.io_format import write_scene_file
from sim.geant4_app.scene_export import (
    ExportedDetectorModel,
    ExportedGeant4Material,
    ExportedGeant4Scene,
    ExportedGeant4Source,
    ExportedShieldModel,
)
from sim.isaacsim_app.scene_builder import (
    StagePrimPaths,
    build_scene_description,
)


def _exported_surface_scene() -> ExportedGeant4Scene:
    """Return one exact surface-bound native scene test fixture."""
    material = ExportedGeant4Material(name="air")
    shield = ExportedShieldModel(
        path="/World/Shield",
        shape="spherical_octant_shell",
        inner_radius_m=0.05,
        outer_radius_m=0.05 + 1.0 / 100.0,
        thickness_cm=1.0,
        size_xyz=None,
        material=material,
    )
    return ExportedGeant4Scene(
        scene_hash="a" * 64,
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=(
            ExportedGeant4Source(
                isotope="Cs-137",
                position_xyz=(SURFACE_EMISSION_EPSILON_M, 0.5, 0.5),
                anchor_position_xyz=(0.0, 0.5, 0.5),
                intensity_cps_1m=300_000.0,
                surface_chart_id=0,
                surface_uv=(0.5, 0.5),
                surface_normal_xyz=(1.0, 0.0, 0.0),
                surface_emission_policy_sha256=(
                    surface_emission_policy_sha256()
                ),
            ),
        ),
        detector_model=ExportedDetectorModel(),
        fe_shield=shield,
        pb_shield=shield,
        prim_paths=StagePrimPaths(),
    )


def _native_scene_identity_metadata(
    scene: ExportedGeant4Scene,
) -> dict[str, object]:
    """Return the identity fields emitted after native scene parsing."""
    source = scene.sources[0]
    source_payload = {
        "isotope": source.isotope,
        "position": list(source.anchor_position_xyz),
        "transport_position": list(source.position_xyz),
        "intensity_cps_1m": source.intensity_cps_1m,
        "surface_chart_id": source.surface_chart_id,
        "surface_uv": list(source.surface_uv),
        "surface_normal": list(source.surface_normal_xyz),
        "surface_emission_policy_sha256": (
            source.surface_emission_policy_sha256
        ),
    }
    prefix = "native_surface_source_0_"
    return {
        "backend": "geant4",
        "engine_mode": "external",
        "scene_hash": scene.scene_hash,
        "surface_source_contract_sha256": (
            surface_source_runtime_contract_sha256([source_payload])
        ),
        "native_surface_source_count": 1,
        prefix + "isotope": source.isotope,
        prefix + "intensity_cps_1m": source.intensity_cps_1m,
        prefix + "surface_chart_id": source.surface_chart_id,
        prefix + "surface_emission_policy_sha256": (
            source.surface_emission_policy_sha256
        ),
        prefix + "anchor_x": source.anchor_position_xyz[0],
        prefix + "anchor_y": source.anchor_position_xyz[1],
        prefix + "anchor_z": source.anchor_position_xyz[2],
        prefix + "transport_x": source.position_xyz[0],
        prefix + "transport_y": source.position_xyz[1],
        prefix + "transport_z": source.position_xyz[2],
        prefix + "surface_u": source.surface_uv[0],
        prefix + "surface_v": source.surface_uv[1],
        prefix + "surface_normal_x": source.surface_normal_xyz[0],
        prefix + "surface_normal_y": source.surface_normal_xyz[1],
        prefix + "surface_normal_z": source.surface_normal_xyz[2],
    }


def test_surface_atlas_normals_follow_semantic_air_side() -> None:
    """Room normals point inward and every floating solid normal points outward."""
    environment = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 1.0, 2.0, 2.0, 2.0),),
    )
    geometry = build_surface_chart_geometry(
        environment,
        obstacle_grid,
        max_edge_m=0.75,
    )

    validate_air_facing_surface_normals(geometry)

    for kind, face_id, normal in zip(
        geometry.kinds,
        geometry.face_ids,
        geometry.normals_xyz,
    ):
        assert tuple(float(value) for value in normal) == (
            expected_air_facing_normal(kind=kind, face_id=face_id)
        )
    assert {
        "transport_component_0_x0",
        "transport_component_0_x1",
        "transport_component_0_y0",
        "transport_component_0_y1",
        "transport_component_0_z0",
        "transport_component_0_z1",
    }.issubset(set(geometry.face_ids))


def test_generated_truth_retains_exact_chart_uv_and_transport_offset() -> None:
    """Truth anchors must map exactly while physics uses one deterministic epsilon."""
    environment = EnvironmentConfig(size_x=2.0, size_y=3.0, size_z=2.5)
    chart_edge_m = 0.4
    sources = generate_surface_sources(
        env=environment,
        obstacle_grid=None,
        isotopes=("Cs-137", "Co-60"),
        intensity_cps_1m=300_000.0,
        rng=np.random.default_rng(2026072801),
        count=32,
        chart_max_edge_m=chart_edge_m,
    )
    atlas = ContinuousSurfaceAtlas(
        build_surface_chart_geometry(
            environment,
            None,
            max_edge_m=chart_edge_m,
        )
    )

    for source in sources:
        chart_id = np.asarray([source.surface_chart_id], dtype=np.int64)
        surface_uv = np.asarray([source.surface_uv], dtype=np.float64)
        anchor = source.position_array()
        normal = atlas.air_facing_normals_xyz(chart_id)[0]
        mapped = atlas.positions_xyz(chart_id, surface_uv)[0]

        np.testing.assert_allclose(mapped, anchor, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            source.transport_position_array() - anchor,
            SURFACE_EMISSION_EPSILON_M * normal,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert (
            source.surface_emission_policy_sha256
            == surface_emission_policy_sha256()
        )


@pytest.mark.parametrize("invalid_chart_id", (True, 1.0, "1"))
def test_point_source_rejects_coerced_surface_chart_id(
    invalid_chart_id: object,
) -> None:
    """Chart identity must never be truncated or converted from another type."""
    with pytest.raises(ValueError, match="nonnegative integer"):
        PointSource(
            isotope="Cs-137",
            position=(0.0, 0.5, 0.5),
            intensity_cps_1m=300_000.0,
            surface_chart_id=invalid_chart_id,  # type: ignore[arg-type]
            surface_uv=(0.5, 0.5),
            surface_normal=(1.0, 0.0, 0.0),
            transport_position=(SURFACE_EMISSION_EPSILON_M, 0.5, 0.5),
            surface_emission_policy_sha256=surface_emission_policy_sha256(),
        )


def test_reset_scene_rejects_wrong_signed_surface_epsilon() -> None:
    """A source shifted into a solid must fail before native transport starts."""
    payload = {
        "sources": [
            {
                "isotope": "Cs-137",
                "position": [0.0, 0.5, 0.5],
                "transport_position": [
                    -SURFACE_EMISSION_EPSILON_M,
                    0.5,
                    0.5,
                ],
                "intensity_cps_1m": 300_000.0,
                "surface_chart_id": 0,
                "surface_uv": [0.5, 0.5],
                "surface_normal": [1.0, 0.0, 0.0],
                "surface_emission_policy_sha256": (
                    surface_emission_policy_sha256()
                ),
            }
        ]
    }

    with pytest.raises(ValueError, match="surface-emission position"):
        build_scene_description(payload)


def test_native_scene_file_uses_emission_xyz_and_preserves_anchor(
    tmp_path: Path,
) -> None:
    """The sidecar source line must distinguish native XYZ from truth XYZ."""
    policy_hash = surface_emission_policy_sha256()
    scene = _exported_surface_scene()
    output = tmp_path / "scene.txt"

    write_scene_file(scene, output)

    source_line = next(
        line
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("SOURCE ")
    )
    assert f"x={SURFACE_EMISSION_EPSILON_M}" in source_line
    assert "anchor_x=0.0" in source_line
    assert "surface_normal_x=1.0" in source_line
    assert f"surface_emission_policy_sha256={policy_hash}" in source_line


def test_native_parsed_source_payload_is_rehashed_before_ingestion() -> None:
    """An echoed header hash cannot hide a changed native source strength."""
    scene = _exported_surface_scene()
    metadata = _native_scene_identity_metadata(scene)

    validate_native_scene_identity(metadata, scene)

    assert not any(
        key.startswith("native_surface_source_") for key in metadata
    )
    tampered = _native_scene_identity_metadata(scene)
    tampered["native_surface_source_0_intensity_cps_1m"] = 1.0
    with pytest.raises(RuntimeError, match="strengths or transport positions"):
        validate_native_scene_identity(tampered, scene)
    extra = _native_scene_identity_metadata(scene)
    extra["native_surface_source_1_isotope"] = "Eu-154"
    with pytest.raises(RuntimeError, match="unexpected fields"):
        validate_native_scene_identity(extra, scene)
