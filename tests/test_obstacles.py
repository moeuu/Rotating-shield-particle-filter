"""Tests for obstacle grid generation and serialization."""

from pathlib import Path

import numpy as np
import pytest

from measurement.obstacles import (
    ObstacleGrid,
    build_obstacle_grid,
    generate_obstacle_grid,
    load_or_generate_obstacle_grid,
)


def test_obstacle_grid_roundtrip_and_is_free(tmp_path: Path) -> None:
    """Obstacle grids should round-trip and block expected cells."""
    rng = np.random.default_rng(0)
    grid = generate_obstacle_grid(
        size_x=4.0,
        size_y=4.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng=rng,
    )
    path = tmp_path / "layout.json"
    grid.save(path)
    loaded = ObstacleGrid.load(path)
    assert loaded == grid
    assert loaded.blocked_cells
    ix, iy = loaded.blocked_cells[0]
    x = loaded.origin[0] + ix * loaded.cell_size + 0.1
    y = loaded.origin[1] + iy * loaded.cell_size + 0.1
    assert loaded.is_free((x, y, 0.0)) is False
    assert loaded.is_free((-1.0, -1.0, 0.0)) is True


def test_obstacle_grid_batch_free_space_matches_scalar() -> None:
    """Batched obstacle lookup should preserve scalar outside-grid semantics."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 2),
        blocked_cells=((0, 1), (2, 0)),
    )
    points = np.asarray(
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 1.5, 2.0],
            [2.5, 0.5, 0.0],
            [3.5, 0.5, 0.0],
        ],
        dtype=float,
    )

    batch = grid.is_free_batch(points)
    scalar = np.asarray([grid.is_free(point) for point in points], dtype=bool)

    assert np.array_equal(batch, scalar)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("origin", ("0.0", 0.0)),
        ("origin", (float("nan"), 0.0)),
        ("cell_size", "1.0"),
        ("cell_size", True),
        ("cell_size", 0.0),
        ("grid_shape", (2.0, 2)),
        ("grid_shape", (True, 2)),
        ("blocked_cells", (("0", 0),)),
        ("blocked_cells", ((0.0, 0),)),
        ("blocked_cells", ((True, 0),)),
        ("blocked_cells", ((0, 0), (0, 0))),
    ),
)
def test_obstacle_grid_rejects_implicit_scalar_coercion_and_duplicates(
    field_name: str,
    invalid_value: object,
) -> None:
    """Grid geometry must not silently reinterpret malformed JSON values."""
    values: dict[str, object] = {
        "origin": (0.0, 0.0),
        "cell_size": 1.0,
        "grid_shape": (2, 2),
        "blocked_cells": ((0, 0),),
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        ObstacleGrid(**values)


@pytest.mark.parametrize(
    "box",
    (
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, -1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, "1.0", 1.0, 1.0),
        (0.0, 0.0, 0.0, True, 1.0, 1.0),
        (0.0, 0.0, 0.0, float("nan"), 1.0, 1.0),
    ),
)
def test_obstacle_grid_rejects_invalid_physical_boxes(
    box: tuple[object, ...],
) -> None:
    """Every collision and transport solid must have finite positive volume."""
    base = {
        "origin": (0.0, 0.0),
        "cell_size": 1.0,
        "grid_shape": (2, 2),
        "blocked_cells": (),
    }
    with pytest.raises(ValueError):
        ObstacleGrid(**base, collision_boxes_m=(box,))
    with pytest.raises(ValueError):
        ObstacleGrid(**base, transport_boxes_m=(box,))


@pytest.mark.parametrize(
    "field_name",
    ("collision_boxes_m", "transport_boxes_m"),
)
def test_obstacle_grid_rejects_positive_volume_box_overlap(
    field_name: str,
) -> None:
    """Overlapping physical solids must not double-count attenuation or geometry."""
    boxes = (
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5, 1.5, 1.5, 1.5),
    )
    values = {
        "origin": (0.0, 0.0),
        "cell_size": 1.0,
        "grid_shape": (2, 2),
        "blocked_cells": (),
        field_name: boxes,
    }

    with pytest.raises(ValueError, match="positive-volume overlap"):
        ObstacleGrid(**values)


@pytest.mark.parametrize(
    "gap_m",
    (0.0, -5.0e-13),
)
def test_obstacle_grid_allows_face_contact_and_roundoff(
    gap_m: float,
) -> None:
    """Face contact within serialization roundoff is not a volume overlap."""
    first = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    second = (
        1.0 + gap_m,
        0.0,
        0.0,
        2.0 + gap_m,
        1.0,
        1.0,
    )

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=(),
        transport_boxes_m=(first, second),
    )

    assert grid.transport_boxes_m == (first, second)


@pytest.mark.parametrize(
    "invalid_table",
    (
        {"Cs-137": ("0.1",)},
        {"Cs-137": (True,)},
        {"Cs-137": (float("nan"),)},
        {"Cs-137": (-0.1,)},
        {1: (0.1,)},
        {"": (0.1,)},
        {"Cs-137": (0.1,), "CS137": (0.1,)},
    ),
)
def test_obstacle_grid_rejects_invalid_attenuation_tables(
    invalid_table: dict[object, tuple[object, ...]],
) -> None:
    """Attenuation metadata must match one unambiguous physical box model."""
    with pytest.raises(ValueError):
        ObstacleGrid(
            origin=(0.0, 0.0),
            cell_size=1.0,
            grid_shape=(1, 1),
            blocked_cells=(),
            transport_boxes_m=((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),),
            transport_mu_by_isotope=invalid_table,
        )


def test_obstacle_grid_json_requires_complete_consistent_schema() -> None:
    """Serialized layouts must not fall back to invented grid geometry."""
    payload = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0),),
    ).to_dict()

    missing = dict(payload)
    missing.pop("origin")
    with pytest.raises(ValueError, match="schema mismatch"):
        ObstacleGrid.from_dict(missing)

    unknown = dict(payload)
    unknown["cells"] = []
    with pytest.raises(ValueError, match="schema mismatch"):
        ObstacleGrid.from_dict(unknown)

    wrong_fraction = dict(payload)
    wrong_fraction["blocked_fraction"] = 0.0
    with pytest.raises(ValueError, match="does not match"):
        ObstacleGrid.from_dict(wrong_fraction)

    wrong_version = dict(payload)
    wrong_version["version"] = "1"
    with pytest.raises(ValueError, match="JSON integer"):
        ObstacleGrid.from_dict(wrong_version)


def test_blocked_boxes_rejects_invalid_vertical_extent() -> None:
    """Grid extrusion must not swap or create a zero-thickness solid."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )

    with pytest.raises(ValueError, match="greater"):
        grid.blocked_boxes(z_min=2.0, z_max=1.0)
    with pytest.raises(ValueError, match="greater"):
        grid.blocked_boxes(z_min=1.0, z_max=1.0)
    with pytest.raises(ValueError, match="real number"):
        grid.blocked_boxes(z_min="0.0", z_max=1.0)


def test_collision_geometry_roundtrips_and_survives_transport_attachment(
    tmp_path: Path,
) -> None:
    """Physical collision boxes must remain separate from attenuation boxes."""
    collision_box = (0.1, 0.2, 0.3, 0.8, 0.9, 1.4)
    transport_box = (1.1, 1.2, 0.0, 1.8, 1.9, 2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0),),
    ).with_collision_model(boxes_m=(collision_box,))
    grid = grid.with_transport_model(
        boxes_m=(transport_box,),
        mu_by_isotope={"Cs-137": (0.1,)},
    )

    path = tmp_path / "collision_layout.json"
    grid.save(path)
    loaded = ObstacleGrid.load(path)

    assert loaded.collision_boxes_m == (collision_box,)
    assert loaded.transport_boxes_m == (transport_box,)
    assert loaded.transport_mu_by_isotope == {"Cs-137": (0.1,)}


def test_generate_obstacle_grid_respects_keep_free_points() -> None:
    """Keep-free points should never be blocked."""
    rng = np.random.default_rng(1)
    grid = generate_obstacle_grid(
        size_x=3.0,
        size_y=3.0,
        cell_size=1.0,
        blocked_fraction=0.6,
        rng=rng,
        keep_free_points=[(0.2, 0.2)],
    )
    assert (0, 0) not in grid.blocked_cells


def test_generate_obstacle_grid_reserves_passable_corridor() -> None:
    """Passage waypoints should remain connected even in a fully blocked layout."""
    rng = np.random.default_rng(2)
    grid = generate_obstacle_grid(
        size_x=6.0,
        size_y=6.0,
        cell_size=1.0,
        blocked_fraction=1.0,
        rng=rng,
        passage_points=[(0.5, 0.5), (5.5, 0.5)],
        passage_width_m=2.0,
    )

    assert grid.has_free_path((0.5, 0.5), (5.5, 0.5))
    for ix in range(6):
        assert (ix, 0) not in grid.blocked_cells
        assert (ix, 1) not in grid.blocked_cells


def test_generate_obstacle_grid_reserves_exploration_backbone_by_default() -> None:
    """Generated layouts should keep a sparse whole-room exploration backbone."""
    rng = np.random.default_rng(3)
    grid = generate_obstacle_grid(
        size_x=10.0,
        size_y=20.0,
        cell_size=1.0,
        blocked_fraction=1.0,
        rng=rng,
        keep_free_points=[(1.5, 1.5)],
    )

    anchors = [
        (0.5, 0.5),
        (9.5, 0.5),
        (0.5, 19.5),
        (9.5, 19.5),
        (5.5, 10.5),
    ]
    for anchor in anchors:
        assert grid.has_free_path((1.5, 1.5), anchor)


def test_load_or_generate_obstacle_grid_creates_file(tmp_path: Path) -> None:
    """Missing obstacle layouts should be generated and saved."""
    path = tmp_path / "generated.json"
    grid = load_or_generate_obstacle_grid(
        path,
        size_x=2.0,
        size_y=2.0,
        cell_size=1.0,
        blocked_fraction=0.5,
        rng_seed=0,
    )
    assert path.exists()
    loaded = ObstacleGrid.load(path)
    assert loaded == grid


def test_build_obstacle_grid_fixed_uses_json_layout(tmp_path: Path) -> None:
    """Fixed mode should load the JSON-backed obstacle layout."""
    path = tmp_path / "fixed_layout.json"
    original = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((0, 1), (2, 2)),
    )
    original.save(path)

    loaded = build_obstacle_grid(
        mode="fixed",
        path=path,
        size_x=3.0,
        size_y=3.0,
        rng_seed=123,
    )

    assert loaded == original
    assert ObstacleGrid.load(path) == original


def test_build_obstacle_grid_random_is_ephemeral_and_seeded(tmp_path: Path) -> None:
    """Random mode should create an in-memory layout without writing a file."""
    path = tmp_path / "random_layout.json"

    grid_one = build_obstacle_grid(
        mode="random",
        path=path,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=0.35,
        rng_seed=7,
    )
    grid_two = build_obstacle_grid(
        mode="random",
        path=path,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=0.35,
        rng_seed=7,
    )

    assert not path.exists()
    assert grid_one == grid_two
    assert grid_one.blocked_cells


def test_build_obstacle_grid_random_has_default_passage() -> None:
    """Random mode should reserve a passage from the start to a far corner."""
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=6.0,
        size_y=6.0,
        blocked_fraction=1.0,
        rng_seed=9,
        keep_free_points=[(1.5, 1.5)],
        passage_width_m=1.0,
    )

    assert grid.has_free_path((1.5, 1.5), (5.5, 5.5))
