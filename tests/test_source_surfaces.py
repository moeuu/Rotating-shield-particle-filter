"""Tests for surface-constrained random source placement."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    _build_source_surface_atlas,
    generate_surface_sources,
    is_allowed_source_surface_position,
    source_surface_kind_counts,
    source_surface_kind,
    source_surface_kinds,
    transport_interior_mask,
    validate_area_uniform_source_config,
)
from measurement.surface_charts import (
    build_surface_chart_geometry,
    sample_continuous_surface_positions,
)
from pf.surface_atlas import ContinuousSurfaceAtlas


def test_generate_surface_sources_never_places_sources_in_air_or_obstacles() -> None:
    """Random source generation should only place sources on allowed surfaces."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4), (4, 4), (3, 5)),
        transport_boxes_m=(
            (3.1, 4.1, 0.0, 3.8, 4.9, 1.6),
            (3.2, 5.2, 0.0, 3.9, 5.8, 1.2),
            (4.1, 4.2, 0.0, 4.8, 4.8, 1.8),
        ),
    )
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(4),
        count=200,
        obstacle_height_m=2.0,
    )

    assert len(sources) == 200
    positions = np.asarray([source.position for source in sources], dtype=float)
    kinds = source_surface_kinds(positions, env, grid)
    assert not np.any(np.equal(kinds, None))
    assert np.any(np.char.startswith(kinds.astype(str), "obstacle_"))
    for source in sources:
        assert is_allowed_source_surface_position(
            source.position,
            env,
            grid,
            obstacle_height_m=2.0,
        )


def test_generate_surface_sources_samples_random_intensity_range() -> None:
    """Random source generation should support randomized source strengths."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137",),
        intensity_cps_1m=(100000.0, 200000.0),
        rng=np.random.default_rng(123),
        count=4,
    )
    strengths = [source.intensity_cps_1m for source in sources]
    assert all(100000.0 <= value <= 200000.0 for value in strengths)
    assert len({round(value, 6) for value in strengths}) > 1


@pytest.mark.parametrize("count", (0, -1, True, 1.5))
def test_generate_surface_sources_rejects_invalid_count(count: object) -> None:
    """Invalid truth cardinality must not silently become one source."""
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        generate_surface_sources(
            env=EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0),
            obstacle_grid=None,
            isotopes=("Cs-137",),
            intensity_cps_1m=30000.0,
            rng=np.random.default_rng(123),
            count=count,
        )


def test_generate_surface_sources_rejects_synthetic_obstacle_surfaces() -> None:
    """Truth generation should fail when physical obstacle faces are unknown."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
    )

    with pytest.raises(ValueError, match="requires transport_boxes_m"):
        generate_surface_sources(
            env=env,
            obstacle_grid=grid,
            isotopes=("Cs-137",),
            intensity_cps_1m=30000.0,
            rng=np.random.default_rng(8),
        )


@pytest.mark.parametrize(
    "removed_key",
    (
        "random_source_preferred_max_z_m",
        "random_source_max_ceiling_sources",
        "random_source_visibility_filter",
        "random_source_response_observability_filter",
        "random_source_same_isotope_min_distance_m",
    ),
)
def test_area_uniform_source_config_rejects_truth_selection(
    removed_key: str,
) -> None:
    """Every legacy source-selection key should fail even at a neutral value."""
    with pytest.raises(ValueError, match="were removed"):
        validate_area_uniform_source_config({removed_key: None})


def test_area_uniform_source_config_has_one_valid_measure() -> None:
    """The source-position measure should be fixed rather than tunable."""
    assert validate_area_uniform_source_config({}) == "continuous_area_uniform"
    for invalid in (
        "chart_uniform",
        " Continuous_Area_Uniform ",
        "CONTINUOUS_AREA_UNIFORM",
    ):
        with pytest.raises(ValueError, match="continuous_area_uniform"):
            validate_area_uniform_source_config(
                {"random_source_surface_sampling_measure": invalid}
            )


@pytest.mark.parametrize(
    "isotopes",
    (
        "Cs-137",
        (137,),
        ("Cs-137", ""),
    ),
)
def test_generate_surface_sources_rejects_coerced_isotopes(
    isotopes: object,
) -> None:
    """Truth isotope identities must never be inferred by string conversion."""
    with pytest.raises(TypeError, match="isotopes"):
        generate_surface_sources(
            env=EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0),
            obstacle_grid=None,
            isotopes=isotopes,  # type: ignore[arg-type]
            intensity_cps_1m=30_000.0,
            rng=np.random.default_rng(123),
            count=1,
        )


@pytest.mark.parametrize(
    "intensity",
    ("30000", True, ("10000", 20_000.0)),
)
def test_generate_surface_sources_rejects_coerced_strengths(
    intensity: object,
) -> None:
    """Truth strengths must retain exact detector-cps@1m numeric semantics."""
    with pytest.raises(TypeError, match="real number"):
        generate_surface_sources(
            env=EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0),
            obstacle_grid=None,
            isotopes=("Cs-137",),
            intensity_cps_1m=intensity,  # type: ignore[arg-type]
            rng=np.random.default_rng(123),
            count=1,
        )


def test_generate_surface_sources_matches_physical_surface_area_ratios() -> None:
    """Truth surface-kind frequencies should follow physical area alone."""
    env = EnvironmentConfig(size_x=4.0, size_y=5.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 5),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 0.0, 2.0, 2.0, 1.5),),
    )
    sample_count = 40_000
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(2026072701),
        count=sample_count,
        obstacle_height_m=1.5,
    )
    positions = np.asarray([source.position for source in sources], dtype=float)
    kinds = source_surface_kinds(
        positions,
        env,
        grid,
        obstacle_height_m=1.5,
    )
    atlas = _build_source_surface_atlas(
        env,
        grid,
        obstacle_height_m=1.5,
    )
    atlas_kinds = np.asarray(atlas.kinds, dtype=object)
    total_area = float(np.sum(atlas.areas_m2))
    for kind in sorted(set(atlas.kinds)):
        expected = float(np.sum(atlas.areas_m2[atlas_kinds == kind])) / total_area
        observed = float(np.mean(kinds == kind))
        assert observed == pytest.approx(expected, abs=0.012)
    assert np.any(kinds == "ceiling")
    assert np.any(kinds == "obstacle_top")
    assert np.any(kinds == "obstacle_side")


def test_generate_surface_sources_returns_continuous_independent_coordinates() -> None:
    """Truth positions should not be quantized to atlas chart centers."""
    env = EnvironmentConfig(size_x=4.0, size_y=5.0, size_z=3.0)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=None,
        isotopes=("Cs-137",),
        intensity_cps_1m=30000.0,
        rng=np.random.default_rng(2026072702),
        count=200,
    )
    positions = np.asarray([source.position for source in sources], dtype=float)
    atlas = _build_source_surface_atlas(env, None, obstacle_height_m=2.0)
    deltas = positions[:, None, :] - atlas.centers_xyz[None, :, :]
    nearest_center_distance = np.min(np.linalg.norm(deltas, axis=2), axis=1)

    assert np.all(nearest_center_distance > 1.0e-10)
    assert np.unique(np.round(positions, decimals=10), axis=0).shape[0] == 200


def test_truth_and_pf_atlases_have_identical_continuous_surface_support() -> None:
    """Chart max_edge_m must change topology only, never PF surface support."""
    env = EnvironmentConfig(size_x=5.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 4),
        blocked_cells=((1, 1), (3, 2)),
        transport_boxes_m=(
            (1.0, 1.0, 0.0, 2.0, 2.0, 1.5),
            (3.1, 2.1, 0.4, 3.8, 2.8, 2.2),
        ),
    )
    truth_rectangles = _build_source_surface_atlas(
        env,
        grid,
        obstacle_height_m=2.0,
    )
    pf_rectangles = build_surface_chart_geometry(
        env,
        grid,
        max_edge_m=0.73,
        obstacle_height_m=2.0,
    )
    truth_kinds = np.asarray(truth_rectangles.kinds, dtype=object)
    pf_kinds = np.asarray(pf_rectangles.kinds, dtype=object)
    for kind in sorted(set(truth_rectangles.kinds) | set(pf_rectangles.kinds)):
        truth_area = float(
            np.sum(truth_rectangles.areas_m2[truth_kinds == kind])
        )
        pf_area = float(np.sum(pf_rectangles.areas_m2[pf_kinds == kind]))
        assert pf_area == pytest.approx(truth_area, rel=1.0e-12, abs=1.0e-12)

    truth_positions, _ = sample_continuous_surface_positions(
        truth_rectangles,
        2_000,
        np.random.default_rng(2026072703),
    )
    pf_atlas = ContinuousSurfaceAtlas(pf_rectangles)
    chart_ids, uv = pf_atlas.locate_positions(truth_positions)
    reconstructed = pf_atlas.positions_xyz(chart_ids, uv)

    assert np.allclose(
        reconstructed,
        truth_positions,
        rtol=0.0,
        atol=1.0e-9,
    )


def test_source_surface_kind_rejects_air_and_obstacle_interior() -> None:
    """Surface classification should reject unsupported 3D positions."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=((3, 4),),
        transport_boxes_m=((3.0, 4.0, 0.0, 4.0, 5.0, 2.0),),
    )

    assert source_surface_kind((5.0, 5.0, 5.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 1.0), env, grid) is None
    assert source_surface_kind((3.5, 4.5, 2.0), env, grid) == "obstacle_top"
    assert source_surface_kind((3.0, 4.5, 1.0), env, grid) == "obstacle_side"
    assert source_surface_kind((1.5, 1.5, 0.0), env, grid) == "floor"
    assert source_surface_kind((5.0, 20.0, 4.0), env, grid) == "wall"


def test_blocked_cells_do_not_invent_source_surfaces_without_geometry() -> None:
    """Navigation occupancy must not create synthetic source-support faces."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
    )

    kinds = source_surface_kinds(
        np.asarray(
            [
                (2.5, 2.5, 1.0),
                (2.0, 2.5, 0.5),
                (2.5, 2.5, 0.0),
            ],
            dtype=float,
        ),
        env,
        grid,
    )

    assert kinds.tolist() == [None, None, "floor"]


def test_transport_component_interior_is_not_allowed_source_support() -> None:
    """Known transport-box interiors should not become source support."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
    ).with_transport_model(
        boxes_m=((1.2, 1.2, 0.2, 1.8, 1.8, 1.4),),
        mu_by_isotope={"Cs-137": (0.1,)},
    )
    points = np.asarray(
        [
            (1.5, 1.5, 0.8),
            (1.0, 1.5, 0.8),
            (0.5, 0.5, 0.0),
        ],
        dtype=float,
    )

    mask = transport_interior_mask(points, grid)

    assert mask.tolist() == [True, False, False]
    assert not is_allowed_source_surface_position((1.5, 1.5, 0.8), env, grid)
    assert not is_allowed_source_surface_position((1.0, 1.5, 0.8), env, grid)
    assert is_allowed_source_surface_position((1.2, 1.5, 0.8), env, grid)


def test_transport_solids_hide_room_interfaces_from_source_support() -> None:
    """A solid-room interface must not be classified as an exposed surface."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((0, 1), (2, 2)),
    ).with_transport_model(
        boxes_m=(
            (0.0, 1.0, 0.8, 0.9, 2.0, 1.8),
            (2.0, 2.0, 2.2, 3.0, 3.0, 3.0),
        ),
        mu_by_isotope={"Cs-137": (0.1, 0.1)},
    )
    points = np.asarray(
        [
            (0.0, 1.5, 1.3),
            (2.5, 2.5, 3.0),
            (0.9, 1.5, 1.3),
            (2.5, 2.5, 2.2),
        ],
        dtype=float,
    )

    kinds = source_surface_kinds(points, env, grid)

    assert kinds.tolist() == [
        None,
        None,
        "obstacle_side",
        "obstacle_bottom",
    ]
    assert not is_allowed_source_surface_position(points[0], env, grid)
    assert not is_allowed_source_surface_position(points[1], env, grid)


def test_source_surface_kinds_matches_scalar_classification() -> None:
    """Vectorized surface classification should match the scalar oracle."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((2, 2),),
        transport_boxes_m=((2.0, 2.0, 0.0, 3.0, 3.0, 1.0),),
    )
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 2.0],
            [2.5, 2.5, 1.0],
            [2.0, 2.5, 0.5],
            [2.5, 2.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    vectorized = source_surface_kinds(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )
    scalar = np.array(
        [
            source_surface_kind(point, env, grid, obstacle_height_m=1.0)
            for point in points
        ],
        dtype=object,
    )
    counts = source_surface_kind_counts(
        points,
        env,
        grid,
        obstacle_height_m=1.0,
    )

    assert vectorized.tolist() == scalar.tolist()
    assert counts["floor"] == 1
    assert counts["wall"] == 1
    assert counts["obstacle_top"] == 1
    assert counts["obstacle_side"] == 1
    assert counts["off_surface"] == 2


@pytest.mark.parametrize(
    "tolerance",
    (-1.0, float("nan"), float("inf"), True, "1e-6"),
)
def test_source_surface_tolerance_fails_instead_of_clamping(
    tolerance: object,
) -> None:
    """Invalid surface tolerances must not silently change source support."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)

    with pytest.raises((TypeError, ValueError), match="tolerance_m"):
        source_surface_kinds(
            np.asarray([[0.0, 1.0, 1.0]], dtype=float),
            env,
            tolerance_m=tolerance,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", (True, 1, None))
def test_area_uniform_measure_requires_an_exact_string(value: object) -> None:
    """Truth sampling must not stringify a malformed selection measure."""
    with pytest.raises(TypeError, match="JSON string"):
        validate_area_uniform_source_config(
            {"random_source_surface_sampling_measure": value}
        )
