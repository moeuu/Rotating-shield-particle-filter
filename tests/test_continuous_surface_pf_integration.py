"""Small integration tests for the continuous-surface exact PF kernel."""

from __future__ import annotations

import copy
import math

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
from pf.state import IsotopeState


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


def _split_merge_filter(
    *,
    cardinality: int,
    max_sources: int = 2,
) -> IsotopeParticleFilter:
    """Return a compact variable-K filter for split/merge support tests."""
    if cardinality > max_sources:
        raise ValueError("cardinality must not exceed max_sources.")
    filt = IsotopeParticleFilter(
        "Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=12,
            max_sources=max_sources,
            variable_cardinality=True,
            init_num_sources=(0, max_sources),
            position_max=(2.0, 2.0, 2.0),
            strength_prior_min_cps_1m=1.0,
            strength_prior_max_cps_1m=3.0,
            structural_rj_split_merge_probability=1.0,
            structural_rj_position_proposal_prior_weight=1.0,
            structural_rj_strength_proposal_prior_weight=1.0,
            use_gpu=False,
        ),
        random_seed=20260729 + cardinality,
    )
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_ids, surface_uv, _ = atlas.sample(
        len(filt.continuous_particles) * cardinality,
        rng=np.random.default_rng(20260731 + cardinality),
    )
    chart_ids = chart_ids.reshape(len(filt.continuous_particles), cardinality)
    surface_uv = surface_uv.reshape(
        len(filt.continuous_particles),
        cardinality,
        2,
    )
    positions = atlas.positions_xyz(chart_ids, surface_uv)
    chart_ids, surface_uv, _, strengths = (
        filt._continuous_rj_canonicalize_rows(
            chart_ids,
            surface_uv,
            positions,
            np.full(
                (len(filt.continuous_particles), cardinality),
                2.0,
                dtype=np.float64,
            ),
        )
    )
    for row, particle in enumerate(filt.continuous_particles):
        particle.state = IsotopeState(
            num_sources=cardinality,
            surface_chart_ids=chart_ids[row],
            surface_uv=surface_uv[row],
            strengths=strengths[row],
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


def test_structural_sweep_reuses_supplied_current_target_values() -> None:
    """A batched sweep must evaluate proposals without recomputing its base."""
    filt = _fixed_one_source_filter()
    filt.config.structural_rj_move_probability = 0.0
    filt.config.structural_rj_position_move_probability = 1.0
    filt.config.structural_rj_local_position_move_probability = 0.0
    filt.config.structural_rj_strength_move_probability = 0.0
    filt.config.structural_rj_split_merge_probability = 0.0
    evaluated_row_counts: list[int] = []

    def _target(**kwargs: object) -> np.ndarray:
        """Record only candidate target evaluations."""
        positions = np.asarray(kwargs["positions_pks"], dtype=np.float64)
        evaluated_row_counts.append(int(positions.shape[0]))
        return np.zeros(int(positions.shape[0]), dtype=np.float64)

    filt.set_joint_target_evaluator(_target)
    supplied = np.zeros(len(filt.continuous_particles), dtype=np.float64)
    filt.apply_structural_moves(
        _one_row_geometry(),
        target_beta=0.5,
        tempering_start_row=0,
        current_target_log_likelihood=supplied,
    )

    assert evaluated_row_counts == [len(filt.continuous_particles)]
    assert filt.last_structural_target_log_likelihood is not None
    np.testing.assert_array_equal(
        filt.last_structural_target_log_likelihood,
        supplied,
    )


def test_conditional_strength_proposal_uses_fixed_geometry_grid_target() -> None:
    """The standard joint proposal must select the batched grid evaluator."""
    filt = _split_merge_filter(cardinality=1)
    geometry = _one_row_geometry()
    filt._structural_rj_position_proposal = (
        filt._build_continuous_rj_position_proposal(
            geometry,
            target_beta=0.5,
        )
    )
    rows = np.arange(len(filt.continuous_particles), dtype=np.int64)
    charts, _, positions, _ = filt._continuous_rj_group_arrays(rows, 1)
    observed_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _expanded_target(**_: object) -> np.ndarray:
        """Fail if the old row-times-grid evaluator remains selected."""
        raise AssertionError("Expanded scalar target must not be called.")

    def _grid_target(**kwargs: object) -> np.ndarray:
        """Record fixed geometry and return one score per strength candidate."""
        fixed_positions = np.asarray(
            kwargs["positions_pks"],
            dtype=np.float64,
        )
        strength_grid = np.asarray(
            kwargs["strengths_pgk"],
            dtype=np.float64,
        )
        observed_shapes.append((fixed_positions.shape, strength_grid.shape))
        return np.zeros(strength_grid.shape[:2], dtype=np.float64)

    filt.set_joint_target_evaluator(_expanded_target)
    filt.set_joint_strength_grid_target_evaluator(_grid_target)
    proposal = filt._continuous_rj_conditional_block_strength_proposal(
        geometry,
        chart_ids=charts,
        positions=positions,
        particle_indices=rows,
        target_beta=0.5,
    )

    grid_size = int(filt.config.structural_rj_strength_proposal_grid_size)
    assert observed_shapes == [
        ((rows.size, 1, 3), (rows.size, grid_size, 1))
    ]
    assert proposal.data_locations.shape == (rows.size, 1)


def test_current_strength_center_cache_invalidates_only_changed_rows() -> None:
    """Accepted state changes must preserve cached centers for other rows."""
    filt = _split_merge_filter(cardinality=1)
    geometry = _one_row_geometry()
    filt._structural_rj_position_proposal = (
        filt._build_continuous_rj_position_proposal(
            geometry,
            target_beta=0.5,
        )
    )
    particle_count = len(filt.continuous_particles)
    filt._structural_rj_current_block_strength_centers = np.full(
        (particle_count, filt.config.hard_max_sources),
        float("nan"),
        dtype=np.float64,
    )
    filt._structural_rj_current_block_strength_cardinalities = np.full(
        particle_count,
        -1,
        dtype=np.int64,
    )
    evaluated_indices: list[np.ndarray] = []

    def _grid_target(**kwargs: object) -> np.ndarray:
        """Record cache misses and return deterministic exact-target scores."""
        indices = np.asarray(kwargs["particle_indices"], dtype=np.int64)
        strength_grid = np.asarray(
            kwargs["strengths_pgk"],
            dtype=np.float64,
        )
        evaluated_indices.append(indices.copy())
        return np.sum(strength_grid, axis=2)

    filt.set_joint_strength_grid_target_evaluator(_grid_target)
    rows = np.arange(particle_count, dtype=np.int64)
    charts, uv, positions, strengths = filt._continuous_rj_group_arrays(
        rows,
        1,
    )
    first = filt._continuous_rj_conditional_block_strength_proposal(
        geometry,
        chart_ids=charts,
        positions=positions,
        particle_indices=rows,
        target_beta=0.5,
        cache_current_state=True,
    )
    second = filt._continuous_rj_conditional_block_strength_proposal(
        geometry,
        chart_ids=charts,
        positions=positions,
        particle_indices=rows,
        target_beta=0.5,
        cache_current_state=True,
    )
    np.testing.assert_array_equal(first.data_locations, second.data_locations)
    assert len(evaluated_indices) == 1
    np.testing.assert_array_equal(evaluated_indices[0], rows)

    changed_strengths = strengths.copy()
    changed_strengths[0, 0] = 2.5
    accepted = np.zeros(particle_count, dtype=np.bool_)
    accepted[0] = True
    filt._commit_continuous_rj_states(
        rows,
        accepted,
        charts,
        uv,
        positions,
        changed_strengths,
    )
    charts, _, positions, _ = filt._continuous_rj_group_arrays(rows, 1)
    filt._continuous_rj_conditional_block_strength_proposal(
        geometry,
        chart_ids=charts,
        positions=positions,
        particle_indices=rows,
        target_beta=0.5,
        cache_current_state=True,
    )
    assert len(evaluated_indices) == 2
    np.testing.assert_array_equal(evaluated_indices[1], np.asarray([0]))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_torch_mh_and_fixed_capacity_state_scatter_match_numpy(
    device_name: str,
) -> None:
    """Torch MH decisions and state scatters must equal their NumPy oracle."""
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    filt = _split_merge_filter(cardinality=1)
    rows = np.arange(len(filt.continuous_particles), dtype=np.int64)
    charts, uv, positions, strengths = filt._continuous_rj_group_arrays(rows, 1)
    reference = torch.zeros(
        1,
        device=torch.device(device_name),
        dtype=torch.float64,
    )
    initial_packed = filt._packed_continuous_surface_state_arrays()
    assert filt._initialize_continuous_rj_device_state(reference)
    rng_oracle = np.random.default_rng()
    rng_oracle.bit_generator.state = copy.deepcopy(
        filt._random_generator.bit_generator.state
    )
    log_ratio = np.linspace(-4.0, 2.0, rows.size, dtype=np.float64)
    support = (rows % 3) != 0
    with np.errstate(divide="ignore"):
        expected = (
            np.log(rng_oracle.random(rows.size))
            < np.minimum(log_ratio, 0.0)
        ) & support
    actual = filt._continuous_rj_mh_acceptance_mask(
        log_ratio,
        support=support,
    )
    np.testing.assert_array_equal(actual, expected)

    accepted = (rows % 2) == 0
    proposed_strengths = strengths.copy()
    proposed_strengths[accepted, 0] += 0.25
    filt._commit_continuous_rj_states(
        rows,
        accepted,
        charts,
        uv,
        positions,
        proposed_strengths,
    )
    packed = filt._packed_continuous_surface_state_arrays()
    device_state = filt._structural_rj_device_state
    assert device_state is not None
    for name, expected_values in zip(
        ("positions", "strengths", "mask", "chart_ids", "surface_uv"),
        packed,
        strict=True,
    ):
        np.testing.assert_array_equal(
            device_state[name].detach().cpu().numpy(),
            expected_values,
        )
    for name, expected_values in zip(
        (
            "cache_positions",
            "cache_strengths",
            "cache_mask",
            "cache_chart_ids",
            "cache_surface_uv",
        ),
        initial_packed,
        strict=True,
    ):
        np.testing.assert_array_equal(
            device_state[name].detach().cpu().numpy(),
            expected_values,
        )
    diagnostics = filt.last_structural_device_diagnostics
    assert diagnostics["mh_acceptance_calls"] == 1
    assert diagnostics["state_scatter_rows"] == int(np.count_nonzero(accepted))

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


def test_block_independence_crosses_multiple_cardinalities_exactly() -> None:
    """A prior block proposal must cross directly from K=5 to K<=3."""
    filt = _split_merge_filter(cardinality=5, max_sources=5)
    filt.config.structural_rj_block_independence_probability = 1.0
    filt.config.structural_rj_position_proposal_prior_weight = 1.0
    filt.config.structural_rj_strength_proposal_prior_weight = 1.0
    filt.set_joint_target_evaluator(
        lambda *, positions_pks, **_: np.zeros(
            int(np.asarray(positions_pks).shape[0]),
            dtype=np.float64,
        )
    )
    geometry = _one_row_geometry()
    filt._structural_rj_position_proposal = (
        filt._build_continuous_rj_position_proposal(
            geometry,
            target_beta=1.0,
        )
    )

    accepted, cardinality_changed = (
        filt._apply_continuous_rj_block_independence(
            geometry,
            target_beta=1.0,
        )
    )

    assert accepted == len(filt.continuous_particles)
    assert cardinality_changed > 0
    assert any(
        particle.state.num_sources <= 3
        for particle in filt.continuous_particles
    )
    assert filt._structural_rj_move_counts["block_attempted"] == 12
    assert filt._structural_rj_move_counts["block_accepted"] == 12


def test_block_density_uses_canonical_unordered_state_measure() -> None:
    """Canonical K-source density must include the K-factorial multiplier."""
    filt = _split_merge_filter(cardinality=3, max_sources=5)
    geometry = _one_row_geometry()
    filt._structural_rj_position_proposal = (
        filt._build_continuous_rj_position_proposal(
            geometry,
            target_beta=1.0,
        )
    )
    particle_indices = np.asarray([0], dtype=np.int64)
    charts, _, _, strengths = filt._continuous_rj_group_arrays(
        particle_indices,
        3,
    )
    prior, proposal = filt._continuous_rj_block_log_densities(
        charts,
        strengths,
    )
    atlas = filt._structural_rj_surface_atlas
    cardinality_prior = filt._structural_rj_cardinality_prior
    assert atlas is not None
    assert cardinality_prior is not None
    expected = (
        float(cardinality_prior.log_prob(3))
        + math.lgamma(4.0)
        + float(np.sum(atlas.log_chart_probabilities[charts[0]]))
        + float(np.sum(filt._strength_prior.log_prob(strengths[0])))
    )

    assert float(prior[0]) == pytest.approx(expected)
    assert float(proposal[0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("cardinality", "strength", "unexpected_candidate_cardinality"),
    [
        (1, 1.5, 2),
        (2, 2.0, 1),
    ],
)
def test_split_merge_does_not_evaluate_out_of_support_candidates(
    cardinality: int,
    strength: float,
    unexpected_candidate_cardinality: int,
) -> None:
    """Infeasible split/merge rows must reject before likelihood evaluation."""
    filt = _split_merge_filter(cardinality=cardinality)
    for particle in filt.continuous_particles:
        state = particle.state
        particle.state = IsotopeState(
            num_sources=cardinality,
            surface_chart_ids=state.surface_chart_ids,
            surface_uv=state.surface_uv,
            strengths=np.full(cardinality, strength, dtype=np.float64),
        )
    evaluated_cardinalities: list[int] = []

    def _target(**kwargs: object) -> np.ndarray:
        """Reject if an infeasible candidate reaches the target evaluator."""
        strengths = np.asarray(kwargs["strengths_pk"], dtype=np.float64)
        evaluated_cardinalities.append(int(strengths.shape[1]))
        assert int(strengths.shape[1]) != unexpected_candidate_cardinality
        assert np.all(filt._strength_prior.in_support(strengths))
        return np.zeros(int(strengths.shape[0]), dtype=np.float64)

    filt.set_joint_target_evaluator(_target)
    geometry = _one_row_geometry()
    filt._structural_rj_position_proposal = (
        filt._build_continuous_rj_position_proposal(
            geometry,
            target_beta=1.0,
        )
    )

    split_count, merge_count = filt._apply_continuous_rj_split_merge(
        geometry,
        target_beta=1.0,
    )

    assert split_count == 0
    assert merge_count == 0
    assert evaluated_cardinalities == [cardinality]
    assert all(
        particle.state.num_sources == cardinality
        for particle in filt.continuous_particles
    )


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


def test_rj_gpu_components_deduplicate_pose_and_select_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production GPU path must reuse geometry across shield views."""
    filt = _fixed_one_source_filter()
    state = filt.continuous_particles[0].state
    chart_ids = np.asarray(state.surface_chart_ids, dtype=np.int64)
    anchors = filt.continuous_state_positions(state)
    geometry = StructuralGeometryBatch(
        detector_positions=np.asarray(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        ),
        fe_indices=np.asarray([0, 1], dtype=np.int64),
        pb_indices=np.asarray([2, 3], dtype=np.int64),
        live_times=np.ones(2, dtype=np.float64),
        station_sequence_ids=np.zeros(2, dtype=np.int64),
    )
    captured: dict[str, np.ndarray] = {}

    def _pair_program_components(**kwargs: object) -> LineTransportComponents:
        """Return pair-coded components for one deduplicated detector pose."""
        detectors = np.asarray(
            kwargs["detector_positions"],
            dtype=np.float64,
        )
        sources = np.asarray(kwargs["sources"], dtype=np.float64)
        fe_program = np.asarray(kwargs["fe_indices"], dtype=np.int64)
        pb_program = np.asarray(kwargs["pb_indices"], dtype=np.int64)
        captured["detectors"] = detectors
        captured["sources"] = sources
        pair_ids = (
            fe_program * len(filt.continuous_kernel.orientations)
            + pb_program
        )
        shape = (
            detectors.shape[0],
            fe_program.shape[1],
            sources.shape[0],
            1,
        )
        values = np.broadcast_to(
            pair_ids[:, :, None, None],
            shape,
        ).astype(np.float64, copy=True)
        return LineTransportComponents(
            total_kernel=values,
            unattenuated_kernel=values + 100.0,
            uncollided_kernel=values + 200.0,
            tau_fe=values + 300.0,
            tau_pb=values + 400.0,
            tau_obstacle=values + 500.0,
            tau_obstacle_compton=values + 600.0,
            distance_m=values + 700.0,
        )

    monkeypatch.setattr(filt, "_can_use_gpu", lambda: True)
    monkeypatch.setattr(
        filt.continuous_kernel,
        "line_transport_components_pair_program_for_detectors",
        _pair_program_components,
    )
    monkeypatch.setattr(
        filt.continuous_kernel,
        "line_transport_components_selected_pairs_for_detectors",
        lambda **_: pytest.fail("GPU RJ must use the all-pair batched path."),
    )

    components = filt._continuous_rj_line_transport_component_columns(
        geometry,
        anchors,
        np.asarray([0], dtype=np.int64),
        chart_ids=chart_ids,
    )

    assert captured["detectors"].shape == (1, 3)
    expected_transport = filt._surface_transport_positions(
        anchors,
        chart_ids=chart_ids,
    )
    np.testing.assert_array_equal(captured["sources"], expected_transport)
    np.testing.assert_array_equal(
        components.total_kernel[:, 0, 0],
        np.asarray([2.0, 11.0]),
    )
    np.testing.assert_array_equal(
        components.distance_m[:, 0, 0],
        np.asarray([702.0, 711.0]),
    )


def test_multi_merge_group_selection_uses_physical_response_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response-equivalent components must outrank a distant spectral shape."""
    filt = _split_merge_filter(cardinality=4, max_sources=5)
    filt.config.structural_rj_merge_distance_sigma_m = 1.0e9
    filt.config.structural_rj_merge_response_sigma = 0.05
    filt.config.structural_rj_merge_uniform_pair_probability = 0.1
    charts = np.zeros((1, 4), dtype=np.int64)
    uv = np.asarray(
        [[[0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2]]],
        dtype=np.float64,
    )
    positions = filt._structural_rj_surface_atlas.positions_xyz(charts, uv)
    geometry = StructuralGeometryBatch(
        detector_positions=np.asarray(
            [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
            dtype=np.float64,
        ),
        fe_indices=np.asarray([0, 0], dtype=np.int64),
        pb_indices=np.asarray([0, 0], dtype=np.int64),
        live_times=np.ones(2, dtype=np.float64),
        station_sequence_ids=np.asarray([0, 1], dtype=np.int64),
    )

    def _physical_components(
        *args: object,
        **kwargs: object,
    ) -> LineTransportComponents:
        """Return three aligned columns and one incompatible response."""
        del kwargs
        source_count = int(np.asarray(args[1]).shape[0])
        assert source_count == 4
        total = np.asarray(
            [[[1.0], [1.0], [1.0], [0.01]],
             [[0.01], [0.01], [0.01], [1.0]]],
            dtype=np.float64,
        )
        ones = np.ones_like(total)
        zeros = np.zeros_like(total)
        return LineTransportComponents(
            total_kernel=total,
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
        "positive_line_indices",
        lambda _: np.asarray([0], dtype=np.int64),
    )
    monkeypatch.setattr(
        filt.continuous_kernel,
        "line_branching_weights",
        lambda *_: np.asarray([1.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        filt,
        "_continuous_rj_line_transport_component_columns",
        _physical_components,
    )

    groups, probabilities = filt._continuous_rj_multi_group_probabilities(
        geometry,
        charts,
        uv,
        positions,
        group_size=3,
    )
    aligned_column = int(
        np.flatnonzero(np.all(groups == np.asarray([0, 1, 2]), axis=1))[0]
    )

    assert probabilities.shape == (1, 4)
    assert float(np.sum(probabilities[0])) == pytest.approx(1.0)
    assert int(np.argmax(probabilities[0])) == aligned_column
    assert probabilities[0, aligned_column] > 0.55


def test_multi_component_direction_support_respects_cardinality_boundaries(
) -> None:
    """Multi-component RJ must not split K=0 or split above the K limit."""
    empty_filter = _split_merge_filter(cardinality=0, max_sources=5)
    assert empty_filter._continuous_rj_multi_direction_support(0) == (
        (),
        (),
        0.0,
        0.0,
    )
    assert empty_filter._apply_continuous_rj_multi_component(
        _one_row_geometry()
    ) == (0, 0)

    full_filter = _split_merge_filter(cardinality=5, max_sources=5)
    split_sizes, merge_sizes, split_probability, merge_probability = (
        full_filter._continuous_rj_multi_direction_support(5)
    )
    assert split_sizes == ()
    assert merge_sizes == (3, 4)
    assert split_probability == 0.0
    assert merge_probability == 1.0
