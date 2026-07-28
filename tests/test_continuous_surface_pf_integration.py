"""Small integration tests for the continuous-surface exact PF kernel."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.continuous_kernels import LineTransportComponents
from measurement.source_boundary import surface_transport_positions
from pf.particle_filter import (
    IsotopeParticleFilter,
    PFConfig,
    StructuralGeometryBatch,
    _extended_log_target_ratio,
)


def _fixed_one_source_filter() -> IsotopeParticleFilter:
    """Return a compact fixed-K filter with exact prior independence proposals."""
    filt = IsotopeParticleFilter(
        "Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=12,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            position_max=(2.0, 2.0, 2.0),
            strength_prior_min_cps_1m=1.0,
            strength_prior_max_cps_1m=3.0,
            structural_rj_position_move_probability=1.0,
            structural_rj_position_proposal_prior_weight=1.0,
            structural_rj_strength_proposal_prior_weight=1.0,
            use_gpu=False,
        ),
        random_seed=20260727,
    )
    filt.set_joint_target_evaluator(
        lambda *, positions_pks, **_: np.zeros(
            int(np.asarray(positions_pks).shape[0]),
            dtype=np.float64,
        )
    )
    filt.set_joint_proposal_evaluator(
        lambda *, chart_centers_xyz, **_: (
            np.zeros(
                int(np.asarray(chart_centers_xyz).shape[0]),
                dtype=np.float64,
            ),
            np.full(
                int(np.asarray(chart_centers_xyz).shape[0]),
                2.0,
                dtype=np.float64,
            ),
            False,
        )
    )
    return filt


def _one_row_geometry() -> StructuralGeometryBatch:
    """Return one valid geometry row for callback-driven integration tests."""
    return StructuralGeometryBatch(
        detector_positions=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times=np.asarray([1.0], dtype=np.float64),
        station_sequence_ids=np.asarray([0], dtype=np.int64),
    )


def _multi_station_geometry() -> StructuralGeometryBatch:
    """Return two completed rows followed by three newest-station rows."""
    return StructuralGeometryBatch(
        detector_positions=np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5],
                [1.5, 1.5, 1.5],
                [1.5, 1.5, 1.5],
            ],
            dtype=np.float64,
        ),
        fe_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        pb_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        live_times=np.ones(5, dtype=np.float64),
        station_sequence_ids=np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
    )


def test_structural_moves_forward_and_reset_tempering_station_boundary() -> None:
    """Every internal MH target call must use the active newest-station boundary."""
    filt = _fixed_one_source_filter()
    observed_boundaries: list[int | None] = []

    def _target(**kwargs: object) -> np.ndarray:
        """Record the station boundary forwarded by the real move path."""
        observed_boundaries.append(kwargs["tempering_start_row"])
        positions = np.asarray(kwargs["positions_pks"], dtype=np.float64)
        return np.zeros(int(positions.shape[0]), dtype=np.float64)

    filt.set_joint_target_evaluator(_target)
    for geometry, boundary in (
        (_one_row_geometry(), 0),
        (_multi_station_geometry(), 2),
        (_one_row_geometry(), 0),
    ):
        call_start = len(observed_boundaries)
        filt.apply_structural_moves(
            geometry,
            target_beta=0.5,
            tempering_start_row=boundary,
        )
        current_calls = observed_boundaries[call_start:]
        assert current_calls
        assert set(current_calls) == {boundary}
        assert filt._structural_rj_tempering_start_row is None


def test_split_merge_skips_cardinality_with_no_reversible_direction() -> None:
    """K=Kmax=1 has neither split nor merge and must be a self-transition."""
    filt = _fixed_one_source_filter()
    geometry = _one_row_geometry()
    proposal = filt._build_continuous_rj_position_proposal(
        geometry,
        target_beta=1.0,
    )
    filt._structural_rj_position_proposal = proposal

    split_count, merge_count = filt._apply_continuous_rj_split_merge(
        geometry,
        target_beta=1.0,
    )

    assert split_count == 0
    assert merge_count == 0
    assert filt._structural_rj_move_counts["split_attempted"] == 0
    assert filt._structural_rj_move_counts["merge_attempted"] == 0


def test_extended_log_target_ratio_preserves_zero_mass_semantics() -> None:
    """Zero-mass proposals must reject without hiding invalid target values."""
    ratio = _extended_log_target_ratio(
        np.asarray([3.0, -np.inf, 4.0, -np.inf], dtype=np.float64),
        np.asarray([1.0, 2.0, -np.inf, -np.inf], dtype=np.float64),
    )

    assert ratio[0] == pytest.approx(2.0)
    assert np.isneginf(ratio[1])
    assert np.isposinf(ratio[2])
    assert np.isneginf(ratio[3])
    assert not bool(np.log(0.5) < np.minimum(ratio[3], 0.0))
    for invalid in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite or negative infinity"):
            _extended_log_target_ratio(
                np.asarray([invalid], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="finite or negative infinity"):
            _extended_log_target_ratio(
                np.asarray([0.0], dtype=np.float64),
                np.asarray([invalid], dtype=np.float64),
            )


def test_joint_target_batch_keeps_individual_zero_mass_candidates() -> None:
    """One minus-infinity candidate must not discard other valid MH rows."""
    filt = _fixed_one_source_filter()
    particle_indices = np.asarray([0, 1], dtype=np.int64)
    chart_ids, _, positions, strengths = filt._continuous_rj_group_arrays(
        particle_indices,
        1,
    )
    filt.set_joint_target_evaluator(
        lambda **_: np.asarray([0.0, -np.inf], dtype=np.float64)
    )

    result = filt._continuous_rj_group_log_likelihood(
        _one_row_geometry(),
        positions,
        strengths,
        chart_ids=chart_ids,
        particle_indices=particle_indices,
    )

    assert result[0] == 0.0
    assert np.isneginf(result[1])
    filt.set_joint_target_evaluator(
        lambda **_: np.asarray([0.0, np.nan], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="finite or negative-infinity"):
        filt._continuous_rj_group_log_likelihood(
            _one_row_geometry(),
            positions,
            strengths,
            chart_ids=chart_ids,
            particle_indices=particle_indices,
        )


def test_global_kernel_jointly_moves_position_and_strength() -> None:
    """The irreducible global kernel must cross position-strength correlation."""
    filt = _fixed_one_source_filter()
    geometry = _one_row_geometry()
    proposal = filt._build_continuous_rj_position_proposal(
        geometry,
        target_beta=1.0,
    )
    filt._structural_rj_position_proposal = proposal
    before_strengths = np.asarray(
        [
            particle.state.strengths[0]
            for particle in filt.continuous_particles
        ],
        dtype=np.float64,
    )
    before_coordinates = [
        (
            int(particle.state.surface_chart_ids[0]),
            tuple(float(value) for value in particle.state.surface_uv[0]),
        )
        for particle in filt.continuous_particles
    ]

    accepted = filt._apply_continuous_rj_global_position_moves(
        geometry,
        target_beta=1.0,
    )

    after_strengths = np.asarray(
        [
            particle.state.strengths[0]
            for particle in filt.continuous_particles
        ],
        dtype=np.float64,
    )
    after_coordinates = [
        (
            int(particle.state.surface_chart_ids[0]),
            tuple(float(value) for value in particle.state.surface_uv[0]),
        )
        for particle in filt.continuous_particles
    ]
    assert accepted == len(filt.continuous_particles)
    assert np.all(after_strengths != before_strengths)
    assert after_coordinates != before_coordinates


def test_transient_xyz_cannot_override_authoritative_chart_coordinates() -> None:
    """A target or commit must reject XYZ that differs from chart/UV geometry."""
    filt = _fixed_one_source_filter()
    particle = filt.continuous_particles[0]
    chart_ids = particle.state.surface_chart_ids.reshape(1, 1)
    surface_uv = particle.state.surface_uv.reshape(1, 1, 2)
    strengths = particle.state.strengths.reshape(1, 1)
    derived = filt._structural_rj_surface_atlas.positions_xyz(
        chart_ids,
        surface_uv,
    )

    with pytest.raises(ValueError, match="authoritative chart/UV"):
        filt._continuous_rj_canonicalize_rows(
            chart_ids,
            surface_uv,
            derived + np.asarray([[[0.1, 0.0, 0.0]]]),
            strengths,
        )


def test_shared_edge_transport_uses_authoritative_chart_normal() -> None:
    """A shared XYZ edge must retain its state chart's air-side epsilon."""
    filt = _fixed_one_source_filter()
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    normals = np.asarray(atlas.geometry.normals_xyz, dtype=np.float64)
    source_chart = -1
    destination_chart = -1
    portal_slot = -1
    for chart_id in range(atlas.chart_count):
        for slot, neighbor_id in enumerate(
            atlas._portal_neighbor_ids[chart_id].tolist()
        ):
            if neighbor_id >= 0 and not np.array_equal(
                normals[chart_id],
                normals[int(neighbor_id)],
            ):
                source_chart = chart_id
                destination_chart = int(neighbor_id)
                portal_slot = slot
                break
        if source_chart >= 0:
            break
    assert source_chart >= 0
    anchor = 0.5 * (
        atlas._portal_starts_xyz[source_chart, portal_slot]
        + atlas._portal_ends_xyz[source_chart, portal_slot]
    )

    transported = filt._surface_transport_positions(
        anchor.reshape(1, 3),
        chart_ids=np.asarray([source_chart], dtype=np.int64),
    )
    expected = surface_transport_positions(
        anchor.reshape(1, 3),
        normals[source_chart].reshape(1, 3),
    )
    other_face = surface_transport_positions(
        anchor.reshape(1, 3),
        normals[destination_chart].reshape(1, 3),
    )

    np.testing.assert_array_equal(transported, expected)
    assert not np.array_equal(transported, other_face)


def test_rj_likelihood_evaluates_air_side_transport_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-RJ likelihood must never evaluate the solid-side anchor."""
    filt = _fixed_one_source_filter()
    state = filt.continuous_particles[0].state
    chart_ids = np.asarray(state.surface_chart_ids, dtype=np.int64)
    anchors = filt.continuous_state_positions(state)
    expected_transport = filt._surface_transport_positions(
        anchors,
        chart_ids=chart_ids,
    )
    captured: dict[str, np.ndarray] = {}

    def _capture_components(**kwargs: object) -> LineTransportComponents:
        """Capture physical source coordinates and return valid unit columns."""
        captured["sources"] = np.asarray(kwargs["sources"], dtype=np.float64)
        shape = (1, anchors.shape[0], 1)
        ones = np.ones(shape, dtype=np.float64)
        zeros = np.zeros(shape, dtype=np.float64)
        return LineTransportComponents(
            total_kernel=ones,
            unattenuated_kernel=ones,
            uncollided_kernel=ones,
            tau_fe=zeros,
            tau_pb=zeros,
            tau_obstacle=zeros,
            tau_obstacle_compton=zeros,
            distance_m=ones,
        )

    monkeypatch.setattr(
        filt.continuous_kernel,
        "line_transport_components_selected_pairs_for_detectors",
        _capture_components,
    )
    filt._continuous_rj_line_transport_component_columns(
        _one_row_geometry(),
        anchors,
        np.asarray([0], dtype=np.int64),
        chart_ids=chart_ids,
    )

    np.testing.assert_array_equal(captured["sources"], expected_transport)
    assert not np.array_equal(captured["sources"], anchors)
