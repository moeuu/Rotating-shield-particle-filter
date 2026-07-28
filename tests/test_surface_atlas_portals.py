"""Tests for reversible continuous-surface portal proposals."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.surface_charts import (
    SurfaceChartGeometry,
    build_surface_chart_geometry,
)
from pf.surface_atlas import ContinuousSurfaceAtlas


def _room_atlas() -> ContinuousSurfaceAtlas:
    """Return a small room atlas containing coplanar and folded portals."""
    charts = build_surface_chart_geometry(
        EnvironmentConfig(size_x=2.0, size_y=1.0, size_z=1.0),
        None,
        max_edge_m=1.0,
    )
    return ContinuousSurfaceAtlas(charts)


def _global_to_local_displacement(
    atlas: ContinuousSurfaceAtlas,
    chart_id: int,
    displacement_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Express a Cartesian tangent displacement in one chart's metre frame."""
    vertices = atlas.geometry.vertices_xyz[int(chart_id)]
    u_edge = vertices[1] - vertices[0]
    v_edge = vertices[3] - vertices[0]
    u_axis = u_edge / np.linalg.norm(u_edge)
    v_axis = v_edge / np.linalg.norm(v_edge)
    displacement = np.asarray(displacement_xyz, dtype=np.float64)
    return np.asarray(
        [np.dot(displacement, u_axis), np.dot(displacement, v_axis)],
        dtype=np.float64,
    )


def _physical_chart_coordinates(
    atlas: ContinuousSurfaceAtlas,
    chart_ids: NDArray[np.int64],
    uv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert normalized UV values to physical local chart coordinates."""
    ids = np.asarray(chart_ids, dtype=np.int64).reshape(-1)
    vertices = atlas.geometry.vertices_xyz[ids]
    lengths = np.column_stack(
        (
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 3] - vertices[:, 0], axis=1),
        )
    )
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2) * lengths


def _non_manifold_atlas() -> ContinuousSurfaceAtlas:
    """Return three rectangles whose same boundary has two destinations."""
    vertices = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
        ],
        dtype=np.float64,
    )
    charts = SurfaceChartGeometry(
        centers_xyz=np.mean(vertices, axis=1),
        areas_m2=np.ones(3, dtype=np.float64),
        kinds=("floor", "wall", "wall"),
        face_ids=("floor", "wall_positive", "wall_negative"),
        normals_xyz=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        local_uv_m=np.zeros((3, 2), dtype=np.float64),
        vertices_xyz=vertices,
        adjacency_edges=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        shared_edge_lengths_m=np.ones(2, dtype=np.float64),
    )
    return ContinuousSurfaceAtlas(charts)


def _unequal_area_atlas() -> ContinuousSurfaceAtlas:
    """Return two folded charts with physical areas two and three."""
    vertices = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
                [0.0, 0.0, 3.0],
            ],
        ],
        dtype=np.float64,
    )
    charts = SurfaceChartGeometry(
        centers_xyz=np.mean(vertices, axis=1),
        areas_m2=np.asarray([2.0, 3.0], dtype=np.float64),
        kinds=("floor", "wall"),
        face_ids=("floor", "wall"),
        normals_xyz=np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        local_uv_m=np.zeros((2, 2), dtype=np.float64),
        vertices_xyz=vertices,
        adjacency_edges=np.asarray([[0, 1]], dtype=np.int64),
        shared_edge_lengths_m=np.ones(1, dtype=np.float64),
    )
    return ContinuousSurfaceAtlas(charts)


def _crowded_parallel_surface_atlas() -> ContinuousSurfaceAtlas:
    """Return one large chart hidden behind more than 32 nearer chart centers."""
    target = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    distractor_centers = np.column_stack(
        (
            np.full(40, 0.01, dtype=np.float64),
            np.full(40, 0.01, dtype=np.float64),
            np.linspace(0.01, 0.40, 40, dtype=np.float64),
        )
    )
    offset = np.asarray(
        [
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ],
        dtype=np.float64,
    )
    vertices = np.concatenate(
        (
            target[None, :, :],
            distractor_centers[:, None, :] + offset[None, :, :],
        ),
        axis=0,
    )
    chart_count = int(vertices.shape[0])
    return ContinuousSurfaceAtlas(
        SurfaceChartGeometry(
            centers_xyz=np.mean(vertices, axis=1),
            areas_m2=np.concatenate(
                (
                    np.asarray([4.0], dtype=np.float64),
                    np.full(40, 0.01, dtype=np.float64),
                )
            ),
            kinds=tuple("floor" for _ in range(chart_count)),
            face_ids=tuple(
                f"parallel_face_{index}" for index in range(chart_count)
            ),
            normals_xyz=np.repeat(
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
                chart_count,
                axis=0,
            ),
            local_uv_m=np.zeros((chart_count, 2), dtype=np.float64),
            vertices_xyz=vertices,
            adjacency_edges=np.zeros((0, 2), dtype=np.int64),
            shared_edge_lengths_m=np.zeros(0, dtype=np.float64),
        )
    )


def test_position_locator_does_not_assume_only_32_nearby_surfaces() -> None:
    """Containing charts must remain discoverable in dense 3-D component unions."""
    atlas = _crowded_parallel_surface_atlas()
    point = np.asarray([[0.01, 0.01, 0.0]], dtype=np.float64)

    chart_ids, uv = atlas.locate_positions(point)

    assert chart_ids.tolist() == [0]
    np.testing.assert_allclose(
        uv,
        np.asarray([[0.005, 0.005]], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        atlas.positions_xyz(chart_ids, uv),
        point,
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "tolerance_m",
    (True, "1e-8", -1.0, np.inf, np.nan),
)
def test_position_locator_rejects_coerced_or_invalid_tolerance(
    tolerance_m: object,
) -> None:
    """Surface tolerance must remain an explicit finite nonnegative real."""
    atlas = _room_atlas()

    with pytest.raises(ValueError, match="tolerance_m"):
        atlas.locate_positions(
            np.asarray([[0.25, 0.5, 0.0]], dtype=np.float64),
            tolerance_m=tolerance_m,
        )


@pytest.mark.parametrize(
    "positions",
    (
        np.asarray([[False, False, False]], dtype=np.bool_),
        np.asarray([["0.25", "0.5", "0.0"]]),
        np.asarray([[0.25, 0.5, 0.0]], dtype=object),
        np.asarray([[0.25 + 0.0j, 0.5, 0.0]], dtype=np.complex128),
    ),
)
def test_position_locator_rejects_implicitly_coerced_coordinate_arrays(
    positions: NDArray[object],
) -> None:
    """Truth coordinates must not enter the atlas through dtype coercion."""
    atlas = _room_atlas()

    with pytest.raises(ValueError, match="real numeric values"):
        atlas.locate_positions(positions)


def test_portal_trace_crosses_coplanar_chart_without_quantization() -> None:
    """A tangent step should cross a chart seam at its continuous endpoint."""
    atlas = _room_atlas()
    start_xyz = np.asarray([[0.75, 0.5, 0.0]], dtype=np.float64)
    chart_ids, uv = atlas.locate_positions(start_xyz)
    displacement = _global_to_local_displacement(
        atlas,
        int(chart_ids[0]),
        np.asarray([0.5, 0.0, 0.0], dtype=np.float64),
    )[None, :]

    proposed_ids, proposed_uv, reverse, valid, crossings = (
        atlas.trace_tangent_displacements(chart_ids, uv, displacement)
    )

    assert valid.tolist() == [True]
    assert crossings.tolist() == [1]
    assert proposed_ids[0] != chart_ids[0]
    assert np.allclose(
        atlas.positions_xyz(proposed_ids, proposed_uv),
        np.asarray([[1.25, 0.5, 0.0]], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.linalg.norm(reverse[0]) == pytest.approx(0.5, abs=1.0e-12)


def test_surface_path_metric_uses_shared_portals_instead_of_free_space() -> None:
    """A folded floor-wall distance must follow their shared physical edge."""
    atlas = _room_atlas()
    floor = np.asarray([[0.25, 0.5, 0.0]], dtype=np.float64)
    wall = np.asarray([[0.0, 0.5, 0.25]], dtype=np.float64)

    distance = atlas.surface_path_distance_upper_bound_m(floor, wall)

    assert distance.shape == (1,)
    assert distance[0] == pytest.approx(0.5, abs=1.0e-10)
    assert distance[0] > np.linalg.norm(floor[0] - wall[0])


def test_surface_path_metric_is_exact_across_one_coplanar_portal() -> None:
    """Adjacent coplanar charts must not create a tessellation distance jump."""
    atlas = _room_atlas()
    first = np.asarray([[0.75, 0.5, 0.0]], dtype=np.float64)
    second = np.asarray([[1.25, 0.5, 0.0]], dtype=np.float64)

    distance = atlas.surface_path_distance_upper_bound_m(first, second)

    assert distance[0] == pytest.approx(0.5, abs=1.0e-10)


def test_surface_path_metric_marks_disconnected_components() -> None:
    """Ambiguous non-manifold charts must not be joined through free space."""
    connected = _non_manifold_atlas()
    rectangles = connected.geometry
    atlas = ContinuousSurfaceAtlas(
        SurfaceChartGeometry(
            centers_xyz=rectangles.centers_xyz,
            areas_m2=rectangles.areas_m2,
            kinds=rectangles.kinds,
            face_ids=rectangles.face_ids,
            normals_xyz=rectangles.normals_xyz,
            local_uv_m=rectangles.local_uv_m,
            vertices_xyz=rectangles.vertices_xyz,
            adjacency_edges=np.zeros((0, 2), dtype=np.int64),
            shared_edge_lengths_m=np.zeros(0, dtype=np.float64),
        )
    )
    floor = np.asarray([[0.5, 0.5, 0.0]], dtype=np.float64)
    wall = np.asarray([[0.0, 0.5, 0.5]], dtype=np.float64)

    distance = atlas.surface_path_distance_upper_bound_m(floor, wall)
    floor_chart, floor_uv = atlas.locate_positions(floor)
    wall_chart, wall_uv = atlas.locate_positions(wall)
    coordinate_distance = (
        atlas.surface_coordinate_path_distance_upper_bound_m(
            floor_chart,
            floor_uv,
            wall_chart,
            wall_uv,
        )
    )

    assert np.isposinf(distance[0])
    assert np.isposinf(coordinate_distance[0])


def test_folded_portal_trace_has_exact_reverse_move() -> None:
    """A floor-to-wall move and returned reverse should form an involution."""
    atlas = _room_atlas()
    start_xyz = np.asarray([[0.25, 0.5, 0.0]], dtype=np.float64)
    chart_ids, uv = atlas.locate_positions(start_xyz)
    displacement = _global_to_local_displacement(
        atlas,
        int(chart_ids[0]),
        np.asarray([-0.5, 0.0, 0.0], dtype=np.float64),
    )[None, :]

    proposed_ids, proposed_uv, reverse, valid, crossings = (
        atlas.trace_tangent_displacements(chart_ids, uv, displacement)
    )
    returned_ids, returned_uv, _, returned_valid, returned_crossings = (
        atlas.trace_tangent_displacements(
            proposed_ids,
            proposed_uv,
            reverse,
        )
    )

    assert valid.tolist() == [True]
    assert returned_valid.tolist() == [True]
    assert crossings.tolist() == [1]
    assert returned_crossings.tolist() == [1]
    assert np.allclose(
        atlas.positions_xyz(proposed_ids, proposed_uv),
        np.asarray([[0.0, 0.5, 0.25]], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.array_equal(returned_ids, chart_ids)
    assert returned_uv == pytest.approx(uv, abs=1.0e-12)


def test_vertex_hit_is_an_explicit_self_transition() -> None:
    """A draw aimed at a chart vertex should reject without resampling."""
    atlas = _room_atlas()
    start_xyz = np.asarray([[0.5, 0.5, 0.0]], dtype=np.float64)
    chart_ids, uv = atlas.locate_positions(start_xyz)
    displacement = _global_to_local_displacement(
        atlas,
        int(chart_ids[0]),
        np.asarray([-0.75, -0.75, 0.0], dtype=np.float64),
    )[None, :]

    proposed_ids, proposed_uv, _, valid, crossings = (
        atlas.trace_tangent_displacements(chart_ids, uv, displacement)
    )

    assert valid.tolist() == [False]
    assert crossings.tolist() == [0]
    assert np.array_equal(proposed_ids, chart_ids)
    assert np.array_equal(proposed_uv, uv)


def test_non_manifold_portal_is_an_explicit_self_transition() -> None:
    """Two destinations for one physical crossing should be rejected."""
    atlas = _non_manifold_atlas()
    chart_ids = np.asarray([0], dtype=np.int64)
    uv = np.asarray([[0.25, 0.5]], dtype=np.float64)
    displacement = np.asarray([[-0.5, 0.0]], dtype=np.float64)

    proposed_ids, proposed_uv, _, valid, crossings = (
        atlas.trace_tangent_displacements(chart_ids, uv, displacement)
    )

    assert valid.tolist() == [False]
    assert crossings.tolist() == [0]
    assert np.array_equal(proposed_ids, chart_ids)
    assert np.array_equal(proposed_uv, uv)


def test_folded_portal_map_has_unit_physical_area_jacobian() -> None:
    """The finite-difference physical-coordinate Jacobian should have abs det 1."""
    atlas = _room_atlas()
    start_xyz = np.asarray([[0.25, 0.5, 0.0]], dtype=np.float64)
    chart_ids, uv = atlas.locate_positions(start_xyz)
    base_displacement = _global_to_local_displacement(
        atlas,
        int(chart_ids[0]),
        np.asarray([-0.5, 0.1, 0.0], dtype=np.float64),
    )
    vertices = atlas.geometry.vertices_xyz[int(chart_ids[0])]
    start_lengths = np.asarray(
        [
            np.linalg.norm(vertices[1] - vertices[0]),
            np.linalg.norm(vertices[3] - vertices[0]),
        ],
        dtype=np.float64,
    )
    epsilon_m = 1.0e-6
    columns: list[NDArray[np.float64]] = []
    destination_id: int | None = None
    for axis in range(2):
        offset = np.zeros(2, dtype=np.float64)
        offset[axis] = epsilon_m
        perturbed_uv = np.vstack(
            (
                uv[0] + offset / start_lengths,
                uv[0] - offset / start_lengths,
            )
        )
        perturbed_ids = np.repeat(chart_ids, 2)
        displacements = np.repeat(base_displacement[None, :], 2, axis=0)
        result_ids, result_uv, _, valid, _ = (
            atlas.trace_tangent_displacements(
                perturbed_ids,
                perturbed_uv,
                displacements,
            )
        )
        assert np.all(valid)
        assert result_ids[0] == result_ids[1]
        if destination_id is None:
            destination_id = int(result_ids[0])
        assert np.all(result_ids == destination_id)
        result_local_m = _physical_chart_coordinates(
            atlas,
            result_ids,
            result_uv,
        )
        columns.append(
            (result_local_m[0] - result_local_m[1])
            / (2.0 * epsilon_m)
        )
    jacobian = np.column_stack(columns)

    assert abs(np.linalg.det(jacobian)) == pytest.approx(
        1.0,
        abs=2.0e-8,
    )


def test_batched_random_valid_traces_are_forward_reverse_symmetric() -> None:
    """Every valid random endpoint should trace exactly back with its reverse."""
    atlas = _room_atlas()
    rng = np.random.default_rng(93_510)
    chart_ids, uv, _ = atlas.sample(512, rng=rng)
    displacements = rng.normal(0.0, 0.45, size=(chart_ids.size, 2))

    proposed_ids, proposed_uv, reverse, valid, crossings = (
        atlas.trace_tangent_displacements(
            chart_ids,
            uv,
            displacements,
        )
    )
    assert np.count_nonzero(valid) >= 500
    returned_ids, returned_uv, _, returned_valid, returned_crossings = (
        atlas.trace_tangent_displacements(
            proposed_ids[valid],
            proposed_uv[valid],
            reverse[valid],
        )
    )

    assert np.all(returned_valid)
    assert np.array_equal(returned_ids, chart_ids[valid])
    assert returned_uv == pytest.approx(uv[valid], abs=2.0e-10)
    assert np.array_equal(returned_crossings, crossings[valid])


def test_batched_trace_matches_single_row_oracle() -> None:
    """The vectorized trace should equal independent one-row evaluations."""
    atlas = _room_atlas()
    rng = np.random.default_rng(70_431)
    chart_ids, uv, _ = atlas.sample(24, rng=rng)
    displacements = rng.normal(0.0, 0.6, size=(chart_ids.size, 2))

    batched = atlas.trace_tangent_displacements(
        chart_ids,
        uv,
        displacements,
    )
    scalar_results = [
        atlas.trace_tangent_displacements(
            chart_ids[index : index + 1],
            uv[index : index + 1],
            displacements[index : index + 1],
        )
        for index in range(chart_ids.size)
    ]
    scalar_ids = np.concatenate([result[0] for result in scalar_results])
    scalar_uv = np.concatenate([result[1] for result in scalar_results])
    scalar_reverse = np.concatenate([result[2] for result in scalar_results])
    scalar_valid = np.concatenate([result[3] for result in scalar_results])
    scalar_crossings = np.concatenate([result[4] for result in scalar_results])

    assert np.array_equal(batched[0], scalar_ids)
    assert batched[1] == pytest.approx(scalar_uv, abs=1.0e-12)
    assert batched[2] == pytest.approx(scalar_reverse, abs=1.0e-12)
    assert np.array_equal(batched[3], scalar_valid)
    assert np.array_equal(batched[4], scalar_crossings)


def test_portal_proposal_draws_once_without_retrying_rejects() -> None:
    """The proposal should consume one Gaussian batch and never retry rejects."""
    atlas = _room_atlas()
    chart_ids = np.zeros(8, dtype=np.int64)
    uv = np.repeat([[0.5, 0.5]], chart_ids.size, axis=0)
    seed = 14_728
    proposal_rng = np.random.default_rng(seed)

    _, _, log_reverse_over_forward = (
        atlas.tangent_geodesic_portal_proposal(
            chart_ids,
            uv,
            sigma_m=2.0,
            rng=proposal_rng,
        )
    )
    following_draw = proposal_rng.random()
    reference_rng = np.random.default_rng(seed)
    reference_rng.normal(0.0, 2.0, size=(chart_ids.size, 2))
    expected_following_draw = reference_rng.random()

    expected_log_ratio = (
        atlas.log_chart_probabilities[chart_ids]
        - atlas.log_chart_probabilities[
            atlas.tangent_geodesic_portal_proposal(
                chart_ids,
                uv,
                sigma_m=2.0,
                rng=np.random.default_rng(seed),
            )[0]
        ]
    )
    assert log_reverse_over_forward == pytest.approx(expected_log_ratio)
    assert following_draw == expected_following_draw


def test_unequal_chart_area_proposal_ratio_cancels_surface_prior() -> None:
    """Normalized-UV proposal and area-prior ratios should exactly cancel."""
    atlas = _unequal_area_atlas()
    chart_ids = np.asarray([0], dtype=np.int64)
    uv = np.asarray([[0.25, 0.5]], dtype=np.float64)
    seed = 4

    proposed_ids, _, log_reverse_over_forward = (
        atlas.tangent_geodesic_portal_proposal(
            chart_ids,
            uv,
            sigma_m=1.0,
            rng=np.random.default_rng(seed),
        )
    )

    assert proposed_ids.tolist() == [1]
    log_prior_ratio = (
        atlas.log_chart_probabilities[proposed_ids]
        - atlas.log_chart_probabilities[chart_ids]
    )
    assert log_reverse_over_forward[0] == pytest.approx(
        np.log(2.0 / 3.0),
        abs=1.0e-12,
    )
    assert log_prior_ratio[0] + log_reverse_over_forward[0] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
