"""Tests for exact finite-surface structural RJ-MH probability bookkeeping."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from pf.structural_rj import (
    BIRTH_DEATH_LOG_ABS_JACOBIAN,
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    SurfaceAdjacency,
    SurfaceSetPrior,
    add_surface_indices,
    birth_log_acceptance_ratio,
    conditional_birth_surface_log_probability,
    death_log_acceptance_ratio,
    local_position_log_acceptance_ratio,
    log_acceptance_probability,
    log_elementary_symmetric_normalizers,
    remove_surface_columns,
    uniform_death_index_log_probability,
)


def _enumerated_sets(
    dictionary_size: int,
    cardinality: int,
) -> np.ndarray:
    """Enumerate all canonical sets for one deliberately tiny dictionary."""
    if cardinality == 0:
        return np.empty((1, 0), dtype=np.int64)
    return np.asarray(
        list(combinations(range(dictionary_size), cardinality)),
        dtype=np.int64,
    )


def test_elementary_symmetric_normalizers_match_full_enumeration() -> None:
    """Dynamic normalizers and set masses must match a small brute-force oracle."""
    areas = np.array([1.0, 2.0, 4.0, 3.0])
    normalized = areas / np.sum(areas)
    prior = SurfaceSetPrior(areas)
    observed = log_elementary_symmetric_normalizers(areas)

    expected = []
    for cardinality in range(areas.size + 1):
        sets = _enumerated_sets(areas.size, cardinality)
        products = (
            np.ones(1)
            if cardinality == 0
            else np.prod(normalized[sets], axis=1)
        )
        expected.append(np.log(np.sum(products)))
        assert np.sum(np.exp(prior.log_prob(sets))) == pytest.approx(
            1.0,
            abs=1.0e-13,
        )

    np.testing.assert_allclose(observed, expected, atol=1.0e-14, rtol=0.0)
    scaled = SurfaceSetPrior(areas * 7.3)
    for cardinality in range(areas.size + 1):
        sets = _enumerated_sets(areas.size, cardinality)
        np.testing.assert_allclose(
            prior.log_prob(sets),
            scaled.log_prob(sets),
            atol=1.0e-14,
            rtol=0.0,
        )


def test_batched_rejection_sampler_matches_exact_set_histogram() -> None:
    """The iid-then-reject sampler must reproduce area-product set masses."""
    areas = np.array([1.0, 2.0, 3.0, 4.0])
    prior = SurfaceSetPrior(areas, max_cardinality=2)
    samples = prior.sample_rejection(
        2,
        60_000,
        rng=np.random.default_rng(20260727),
        proposal_batch_size=4096,
    )
    sets = _enumerated_sets(areas.size, 2)
    expected = np.exp(prior.log_prob(sets))

    unique, counts = np.unique(samples, axis=0, return_counts=True)
    np.testing.assert_array_equal(unique, sets)
    observed = counts / np.sum(counts)
    np.testing.assert_allclose(observed, expected, atol=0.006, rtol=0.0)
    assert np.all(np.diff(samples, axis=1) > 0)


def test_explicit_cardinality_prior_and_boundary_move_probabilities() -> None:
    """Configured K mass and boundary renormalization must remain explicit."""
    cardinality_prior = CardinalityPrior([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        np.exp(cardinality_prior.log_prob([0, 1, 2, 3])),
        [0.1, 0.2, 0.3, 0.4],
    )
    assert np.isneginf(cardinality_prior.log_prob(-1))
    assert np.isneginf(cardinality_prior.log_prob(4))

    moves = BirthDeathMoveProbabilities(
        max_cardinality=3,
        birth_weight=2.0,
        death_weight=1.0,
    )
    birth, death = moves.probabilities(np.arange(4))
    np.testing.assert_allclose(birth, [1.0, 2.0 / 3.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(death, [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0])
    assert np.isneginf(moves.log_probability("death", 0))
    assert np.isneginf(moves.log_probability("birth", 3))

    immobile = BirthDeathMoveProbabilities(max_cardinality=2, min_cardinality=2)
    immobile_birth, immobile_death = immobile.probabilities(2)
    assert immobile_birth == 0.0
    assert immobile_death == 0.0


def test_birth_and_reverse_death_log_ratios_are_antisymmetric() -> None:
    """Paired birth/death bookkeeping must produce exactly opposite log ratios."""
    surface_prior = SurfaceSetPrior(
        [1.0, 2.0, 3.0, 4.0],
        max_cardinality=3,
    )
    cardinality_prior = CardinalityPrior([0.15, 0.35, 0.30, 0.20])
    moves = BirthDeathMoveProbabilities(
        max_cardinality=3,
        birth_weight=1.7,
        death_weight=0.8,
    )
    current = np.array([[0], [1], [2]], dtype=np.int64)
    birth_indices = np.array([2, 3, 0], dtype=np.int64)
    proposed = add_surface_indices(
        current,
        birth_indices,
        dictionary_size=surface_prior.dictionary_size,
    )
    death_columns = np.argmax(
        proposed == birth_indices[:, None],
        axis=1,
    )

    likelihood_ratio = np.array([0.4, -0.7, 1.2])
    strength_prior = np.array([-0.2, -0.3, -0.5])
    strength_proposal = np.array([-0.7, -0.1, -0.4])
    position_proposal = conditional_birth_surface_log_probability(
        surface_prior,
        current,
        birth_indices,
    )
    death_index = uniform_death_index_log_probability(2)

    birth_ratio = birth_log_acceptance_ratio(
        current_surface_sets=current,
        birth_surface_indices=birth_indices,
        log_likelihood_ratio=likelihood_ratio,
        surface_prior=surface_prior,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_strength_prior_density=strength_prior,
        log_forward_position_proposal=position_proposal,
        log_forward_strength_proposal=strength_proposal,
        log_reverse_death_index_probability=death_index,
    )
    death_ratio = death_log_acceptance_ratio(
        current_surface_sets=proposed,
        death_columns=death_columns,
        log_likelihood_ratio=-likelihood_ratio,
        surface_prior=surface_prior,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_removed_strength_prior_density=strength_prior,
        log_forward_death_index_probability=death_index,
        log_reverse_position_proposal=position_proposal,
        log_reverse_strength_proposal=strength_proposal,
    )

    np.testing.assert_allclose(birth_ratio, -death_ratio, atol=1.0e-14)
    assert BIRTH_DEATH_LOG_ABS_JACOBIAN == 0.0
    np.testing.assert_allclose(
        log_acceptance_probability(birth_ratio),
        np.minimum(birth_ratio, 0.0),
    )


def test_prior_only_birth_death_kernel_satisfies_detailed_balance() -> None:
    """Every adjacent tiny-dictionary state pair must balance prior-only flux."""
    areas = np.array([1.0, 2.0, 5.0, 3.0])
    maximum = 3
    surface_prior = SurfaceSetPrior(areas, max_cardinality=maximum)
    cardinality_prior = CardinalityPrior([0.20, 0.35, 0.30, 0.15])
    moves = BirthDeathMoveProbabilities(
        max_cardinality=maximum,
        birth_weight=1.4,
        death_weight=0.9,
    )
    log_strength_density = -np.log(2.0)

    for cardinality in range(maximum):
        for current_tuple in combinations(range(areas.size), cardinality):
            current = np.asarray(
                [current_tuple],
                dtype=np.int64,
            ).reshape(1, cardinality)
            unused = sorted(set(range(areas.size)) - set(current_tuple))
            for birth_index in unused:
                birth_indices = np.array([birth_index], dtype=np.int64)
                proposed = add_surface_indices(
                    current,
                    birth_indices,
                    dictionary_size=areas.size,
                )
                death_column = np.flatnonzero(
                    proposed[0] == birth_index
                ).astype(np.int64)
                log_position = conditional_birth_surface_log_probability(
                    surface_prior,
                    current,
                    birth_indices,
                )
                log_death_index = uniform_death_index_log_probability(
                    cardinality + 1
                )

                log_birth_ratio = float(
                    birth_log_acceptance_ratio(
                        current_surface_sets=current,
                        birth_surface_indices=birth_indices,
                        log_likelihood_ratio=0.0,
                        surface_prior=surface_prior,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=moves,
                        log_strength_prior_density=log_strength_density,
                        log_forward_position_proposal=log_position,
                        log_forward_strength_proposal=log_strength_density,
                        log_reverse_death_index_probability=log_death_index,
                    )[0]
                )
                log_death_ratio = float(
                    death_log_acceptance_ratio(
                        current_surface_sets=proposed,
                        death_columns=death_column,
                        log_likelihood_ratio=0.0,
                        surface_prior=surface_prior,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=moves,
                        log_removed_strength_prior_density=(
                            log_strength_density
                        ),
                        log_forward_death_index_probability=log_death_index,
                        log_reverse_position_proposal=log_position,
                        log_reverse_strength_proposal=log_strength_density,
                    )[0]
                )

                log_target_current = (
                    float(cardinality_prior.log_prob(cardinality))
                    + float(surface_prior.log_prob(current)[0])
                    + cardinality * log_strength_density
                )
                log_target_proposed = (
                    float(cardinality_prior.log_prob(cardinality + 1))
                    + float(surface_prior.log_prob(proposed)[0])
                    + (cardinality + 1) * log_strength_density
                )
                log_forward_transition = (
                    float(moves.log_probability("birth", cardinality))
                    + float(log_position[0])
                    + log_strength_density
                    + min(0.0, log_birth_ratio)
                )
                log_reverse_transition = (
                    float(
                        moves.log_probability(
                            "death",
                            cardinality + 1,
                        )
                    )
                    + float(log_death_index)
                    + min(0.0, log_death_ratio)
                )
                assert (
                    log_target_current + log_forward_transition
                ) == pytest.approx(
                    log_target_proposed + log_reverse_transition,
                    abs=2.0e-13,
                )


def test_area_weighted_within_k_position_proposal_satisfies_detailed_balance() -> None:
    """Runtime relocation proposal must cancel the area-weighted set prior."""
    areas = np.asarray([1.0, 2.0, 5.0, 3.0], dtype=float)
    cardinality = 2
    prior = SurfaceSetPrior(areas, max_cardinality=cardinality)
    sets = _enumerated_sets(areas.size, cardinality)
    log_targets = {
        tuple(surface_set): float(prior.log_prob(surface_set[None, :])[0])
        for surface_set in sets
    }

    for current_set in sets:
        current = current_set[None, :]
        for source_column in range(cardinality):
            reduced = remove_surface_columns(
                current,
                np.asarray([source_column], dtype=np.int64),
                dictionary_size=areas.size,
            )
            old_patch = int(current_set[source_column])
            remaining = sorted(set(range(areas.size)) - set(reduced[0]))
            for new_patch in remaining:
                proposed = add_surface_indices(
                    reduced,
                    np.asarray([new_patch], dtype=np.int64),
                    dictionary_size=areas.size,
                )
                new_column = int(
                    np.flatnonzero(proposed[0] == new_patch)[0]
                )
                reverse_reduced = remove_surface_columns(
                    proposed,
                    np.asarray([new_column], dtype=np.int64),
                    dictionary_size=areas.size,
                )
                np.testing.assert_array_equal(reverse_reduced, reduced)
                log_forward = (
                    -np.log(cardinality)
                    + float(
                        conditional_birth_surface_log_probability(
                            prior,
                            reduced,
                            np.asarray([new_patch], dtype=np.int64),
                        )[0]
                    )
                )
                log_reverse = (
                    -np.log(cardinality)
                    + float(
                        conditional_birth_surface_log_probability(
                            prior,
                            reduced,
                            np.asarray([old_patch], dtype=np.int64),
                        )[0]
                    )
                )

                assert (
                    log_targets[tuple(current_set)] + log_forward
                ) == pytest.approx(
                    log_targets[tuple(proposed[0])] + log_reverse,
                    abs=1.0e-13,
                )


def test_invalid_duplicate_set_and_exhausted_birth_fail_fast() -> None:
    """Duplicate canonical states and births from full sets must be rejected."""
    prior = SurfaceSetPrior([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="duplicate"):
        prior.log_prob(np.array([[0, 0]], dtype=np.int64))
    with pytest.raises(ValueError, match="every surface index"):
        conditional_birth_surface_log_probability(
            prior,
            np.array([[0, 1, 2]], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )


def _tiny_surface_adjacency() -> SurfaceAdjacency:
    """Return a small graph with unequal degrees and one isolated patch."""
    return SurfaceAdjacency(
        dictionary_size=5,
        edges=np.asarray(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [2, 3],
            ],
            dtype=np.int64,
        ),
    )


def test_local_position_sampler_is_batched_uniform_and_avoids_occupancy() -> None:
    """Batched local draws must be uniform over only unoccupied neighbors."""
    adjacency = _tiny_surface_adjacency()
    repetitions = 40_000
    surface_sets = np.repeat(
        np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        repetitions,
        axis=0,
    )
    source_columns = np.zeros(surface_sets.shape[0], dtype=np.int64)

    proposed, degrees, movable = adjacency.sample_unoccupied_neighbors(
        surface_sets,
        source_columns,
        rng=np.random.default_rng(20260727),
    )

    first = proposed[:repetitions]
    second = proposed[repetitions:]
    assert np.all(degrees == 2)
    assert np.all(movable)
    assert set(np.unique(first).tolist()) == {2, 3}
    assert set(np.unique(second).tolist()) == {1, 3}
    assert np.count_nonzero(first == 2) / repetitions == pytest.approx(
        0.5,
        abs=0.008,
    )
    assert np.count_nonzero(second == 1) / repetitions == pytest.approx(
        0.5,
        abs=0.008,
    )


def test_local_position_degrees_match_small_serial_oracle() -> None:
    """Vectorized available degrees must equal direct set-based counting."""
    adjacency = _tiny_surface_adjacency()
    occupied = np.asarray(
        [[1], [2], [0], [1], [0]],
        dtype=np.int64,
    )
    centers = np.arange(5, dtype=np.int64)

    observed = adjacency.available_neighbor_degrees(centers, occupied)
    expected = np.asarray(
        [
            sum(
                int(neighbor not in set(occupied[row].tolist()))
                for neighbor in adjacency.neighbors[center]
                if neighbor >= 0
            )
            for row, center in enumerate(centers.tolist())
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(observed, expected)


def test_local_position_kernel_satisfies_prior_only_detailed_balance() -> None:
    """Every admissible graph edge must balance the area-weighted set prior."""
    areas = np.asarray([1.0, 2.0, 5.0, 3.0, 7.0], dtype=float)
    cardinality = 2
    prior = SurfaceSetPrior(areas, max_cardinality=cardinality)
    adjacency = _tiny_surface_adjacency()

    for current_tuple in combinations(range(areas.size), cardinality):
        current = np.asarray([current_tuple], dtype=np.int64)
        for source_column in range(cardinality):
            old_patch = int(current[0, source_column])
            reduced = remove_surface_columns(
                current,
                np.asarray([source_column], dtype=np.int64),
                dictionary_size=areas.size,
            )
            occupied = set(reduced[0].tolist())
            available = [
                int(neighbor)
                for neighbor in adjacency.neighbors[old_patch]
                if neighbor >= 0 and int(neighbor) not in occupied
            ]
            for new_patch in available:
                proposed = add_surface_indices(
                    reduced,
                    np.asarray([new_patch], dtype=np.int64),
                    dictionary_size=areas.size,
                )
                forward_degree = len(available)
                reverse_degree = int(
                    adjacency.available_neighbor_degrees(
                        np.asarray([new_patch], dtype=np.int64),
                        reduced,
                    )[0]
                )
                forward_ratio = float(
                    local_position_log_acceptance_ratio(
                        old_surface_indices=old_patch,
                        new_surface_indices=new_patch,
                        forward_available_degrees=forward_degree,
                        reverse_available_degrees=reverse_degree,
                        log_likelihood_ratio=0.0,
                        surface_prior=prior,
                    )[0]
                )
                reverse_ratio = float(
                    local_position_log_acceptance_ratio(
                        old_surface_indices=new_patch,
                        new_surface_indices=old_patch,
                        forward_available_degrees=reverse_degree,
                        reverse_available_degrees=forward_degree,
                        log_likelihood_ratio=0.0,
                        surface_prior=prior,
                    )[0]
                )
                assert reverse_ratio == pytest.approx(
                    -forward_ratio,
                    abs=1.0e-14,
                )
                log_forward_flux = (
                    float(prior.log_prob(current)[0])
                    - np.log(cardinality)
                    - np.log(forward_degree)
                    + min(0.0, forward_ratio)
                )
                log_reverse_flux = (
                    float(prior.log_prob(proposed)[0])
                    - np.log(cardinality)
                    - np.log(reverse_degree)
                    + min(0.0, reverse_ratio)
                )
                assert log_forward_flux == pytest.approx(
                    log_reverse_flux,
                    abs=2.0e-13,
                )


def test_local_position_sampler_stays_when_no_neighbor_is_available() -> None:
    """Isolated and fully occupied adjacency rows must produce safe self moves."""
    adjacency = _tiny_surface_adjacency()

    isolated, isolated_degree, isolated_movable = (
        adjacency.sample_unoccupied_neighbors(
            np.asarray([[1, 4]], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            rng=np.random.default_rng(1),
        )
    )
    occupied, occupied_degree, occupied_movable = (
        adjacency.sample_unoccupied_neighbors(
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            rng=np.random.default_rng(2),
        )
    )

    np.testing.assert_array_equal(isolated, [4])
    np.testing.assert_array_equal(isolated_degree, [0])
    np.testing.assert_array_equal(isolated_movable, [False])
    np.testing.assert_array_equal(occupied, [0])
    np.testing.assert_array_equal(occupied_degree, [0])
    np.testing.assert_array_equal(occupied_movable, [False])
