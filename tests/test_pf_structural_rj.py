"""Probability-contract tests for continuous-surface exact RJ/MH moves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pf.structural_rj import (
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    ContinuousStrengthProposal,
    ContinuousSurfacePositionProposal,
    SplitMergeMoveProbabilities,
    bounded_simplex_probability,
    bounded_uniform_simplex_log_density,
    continuous_birth_log_acceptance_ratio,
    continuous_death_log_acceptance_ratio,
    continuous_joint_position_strength_log_acceptance_ratio,
    continuous_merge_log_acceptance_ratio,
    continuous_group_merge_log_acceptance_ratio,
    continuous_group_split_log_acceptance_ratio,
    continuous_position_log_acceptance_ratio,
    continuous_relocated_merge_log_acceptance_ratio,
    continuous_relocated_split_log_acceptance_ratio,
    continuous_split_log_acceptance_ratio,
    distance_weighted_ordered_pair_probabilities,
    log_acceptance_probability,
    split_fraction_bounds,
)


def test_continuous_strength_proposal_density_is_normalized_per_chart() -> None:
    """Every chart-conditional prior/data mixture must integrate to one."""
    proposal = ContinuousStrengthProposal(
        minimum=1.0,
        maximum=9.0,
        data_locations_by_chart=np.asarray([1.1, 5.0, 8.9]),
        data_sigma=0.7,
        prior_component_probability=0.2,
    )
    grid = np.linspace(1.0, 9.0, 100_001, dtype=np.float64)

    for chart_id in range(3):
        density = np.exp(
            proposal.log_density(
                np.full(grid.size, chart_id, dtype=np.int64),
                grid,
            )
        )
        assert np.trapezoid(density, grid) == pytest.approx(
            1.0,
            abs=2.0e-8,
        )
        assert np.min(density) >= 0.2 / 8.0 - 1.0e-15


def test_continuous_strength_proposal_sampling_has_full_prior_support() -> None:
    """The positive prior component must keep every physical strength reachable."""
    proposal = ContinuousStrengthProposal(
        minimum=10.0,
        maximum=20.0,
        data_locations_by_chart=np.asarray([10.2, 19.8]),
        data_sigma=0.25,
        prior_component_probability=0.3,
    )
    chart_ids = np.tile(np.asarray([0, 1], dtype=np.int64), 50_000)
    samples = proposal.sample(
        chart_ids,
        rng=np.random.default_rng(20260727),
    )

    assert np.all((samples >= 10.0) & (samples <= 20.0))
    assert np.any(samples[chart_ids == 0] > 19.0)
    assert np.any(samples[chart_ids == 1] < 11.0)
    assert np.all(
        np.isfinite(proposal.log_density(chart_ids, samples))
    )


def test_surface_position_proposal_mixture_preserves_full_support() -> None:
    """The explicit area-prior component must protect every surface chart."""
    proposal = ContinuousSurfacePositionProposal(
        area_prior_probabilities=np.asarray([0.1, 0.2, 0.7]),
        alignment_scores=np.asarray([0.0, 5.0, 0.0]),
        prior_component_probability=0.25,
    )

    np.testing.assert_allclose(
        proposal.chart_probabilities,
        [0.025, 0.8, 0.175],
        atol=1.0e-15,
        rtol=0.0,
    )
    assert np.all(
        proposal.chart_probabilities
        >= 0.25 * proposal.area_prior_probabilities
    )
    assert proposal.data_informative is True
    np.testing.assert_allclose(
        proposal.log_density(np.asarray([0, 1, 2])),
        np.log(proposal.chart_probabilities),
    )


@pytest.mark.parametrize(
    ("alignment", "prior_weight"),
    [
        (np.zeros(3), 0.25),
        (np.asarray([1.0, 8.0, 3.0]), 1.0),
    ],
)
def test_surface_position_proposal_reverts_to_area_prior(
    alignment: np.ndarray,
    prior_weight: float,
) -> None:
    """No signal or an explicit prior-only component must recover the prior."""
    prior = np.asarray([0.15, 0.25, 0.60])
    proposal = ContinuousSurfacePositionProposal(
        area_prior_probabilities=prior,
        alignment_scores=alignment,
        prior_component_probability=prior_weight,
    )

    np.testing.assert_allclose(
        proposal.chart_probabilities,
        prior,
        atol=1.0e-15,
        rtol=0.0,
    )


def test_fixed_mixture_density_is_reciprocal_across_exact_rj_moves() -> None:
    """One frozen mixture must give matching forward/reverse MH-RJ densities."""
    prior = np.asarray([0.2, 0.3, 0.5])
    proposal = ContinuousSurfacePositionProposal(
        area_prior_probabilities=prior,
        alignment_scores=np.asarray([0.0, 1.0, 4.0]),
        prior_component_probability=0.4,
    )
    old_chart = 0
    new_chart = 2
    old_prior = float(np.log(prior[old_chart]))
    new_prior = float(np.log(prior[new_chart]))
    old_proposal = float(proposal.log_density(old_chart))
    new_proposal = float(proposal.log_density(new_chart))
    cardinality_prior = CardinalityPrior([0.2, 0.5, 0.3])
    birth_death = BirthDeathMoveProbabilities(
        max_cardinality=2,
        birth_weight=0.7,
        death_weight=0.3,
    )
    birth = continuous_birth_log_acceptance_ratio(
        current_cardinality=1,
        log_likelihood_ratio=0.8,
        cardinality_prior=cardinality_prior,
        move_probabilities=birth_death,
        log_position_prior_density=new_prior,
        log_strength_prior_density=-0.6,
        log_forward_position_proposal=new_proposal,
        log_forward_strength_proposal=-0.6,
    )
    death = continuous_death_log_acceptance_ratio(
        current_cardinality=2,
        log_likelihood_ratio=-0.8,
        cardinality_prior=cardinality_prior,
        move_probabilities=birth_death,
        log_removed_position_prior_density=new_prior,
        log_removed_strength_prior_density=-0.6,
        log_reverse_position_proposal=new_proposal,
        log_reverse_strength_proposal=-0.6,
    )
    np.testing.assert_allclose(birth, -death, atol=1.0e-14)

    forward_position = continuous_position_log_acceptance_ratio(
        log_likelihood_ratio=0.35,
        log_old_position_prior_density=old_prior,
        log_new_position_prior_density=new_prior,
        log_reverse_proposal_density=old_proposal,
        log_forward_proposal_density=new_proposal,
    )
    reverse_position = continuous_position_log_acceptance_ratio(
        log_likelihood_ratio=-0.35,
        log_old_position_prior_density=new_prior,
        log_new_position_prior_density=old_prior,
        log_reverse_proposal_density=new_proposal,
        log_forward_proposal_density=old_proposal,
    )
    np.testing.assert_allclose(
        forward_position,
        -reverse_position,
        atol=1.0e-14,
    )

    split_merge = SplitMergeMoveProbabilities(
        max_cardinality=2,
        split_weight=0.6,
        merge_weight=0.4,
    )
    split = continuous_split_log_acceptance_ratio(
        current_cardinality=1,
        total_strength=3.0,
        log_likelihood_ratio=0.45,
        cardinality_prior=cardinality_prior,
        move_probabilities=split_merge,
        log_new_position_prior_density=new_prior,
        log_old_strength_prior_density=-0.8,
        log_retained_strength_prior_density=-0.5,
        log_new_strength_prior_density=-0.7,
        log_forward_position_proposal=new_proposal,
        log_forward_fraction_proposal=0.2,
    )
    merge = continuous_merge_log_acceptance_ratio(
        current_cardinality=2,
        merged_strength=3.0,
        log_likelihood_ratio=-0.45,
        cardinality_prior=cardinality_prior,
        move_probabilities=split_merge,
        log_deleted_position_prior_density=new_prior,
        log_deleted_strength_prior_density=-0.7,
        log_retained_strength_prior_density=-0.5,
        log_merged_strength_prior_density=-0.8,
        log_reverse_position_proposal=new_proposal,
        log_reverse_fraction_proposal=0.2,
    )
    np.testing.assert_allclose(split, -merge, atol=1.0e-14)


def test_relocated_split_merge_acceptance_ratios_are_reciprocal() -> None:
    """The two-child position map must satisfy exact RJ detailed balance."""
    cardinality_prior = CardinalityPrior([1.0, 2.0, 3.0, 4.0])
    move_probabilities = SplitMergeMoveProbabilities(
        max_cardinality=3,
        split_weight=0.6,
        merge_weight=0.4,
    )
    split = continuous_relocated_split_log_acceptance_ratio(
        current_cardinality=2,
        total_strength=7.0,
        log_likelihood_ratio=0.25,
        cardinality_prior=cardinality_prior,
        move_probabilities=move_probabilities,
        log_parent_position_prior_density=-1.1,
        log_first_child_position_prior_density=-1.3,
        log_second_child_position_prior_density=-1.5,
        log_parent_strength_prior_density=-0.7,
        log_first_child_strength_prior_density=-0.8,
        log_second_child_strength_prior_density=-0.9,
        log_forward_first_position_proposal=-1.7,
        log_forward_second_position_proposal=-1.9,
        log_forward_fraction_proposal=0.4,
        log_reverse_merged_position_proposal=-2.1,
        log_forward_parent_selection=-math.log(2.0),
        log_reverse_pair_selection=-2.3,
    )
    merge = continuous_relocated_merge_log_acceptance_ratio(
        current_cardinality=3,
        merged_strength=7.0,
        log_likelihood_ratio=-0.25,
        cardinality_prior=cardinality_prior,
        move_probabilities=move_probabilities,
        log_first_child_position_prior_density=-1.3,
        log_second_child_position_prior_density=-1.5,
        log_merged_position_prior_density=-1.1,
        log_first_child_strength_prior_density=-0.8,
        log_second_child_strength_prior_density=-0.9,
        log_merged_strength_prior_density=-0.7,
        log_forward_pair_selection=-2.3,
        log_forward_merged_position_proposal=-2.1,
        log_reverse_parent_selection=-math.log(2.0),
        log_reverse_first_position_proposal=-1.7,
        log_reverse_second_position_proposal=-1.9,
        log_reverse_fraction_proposal=0.4,
    )

    np.testing.assert_allclose(split, -merge, atol=1.0e-14)


def test_cardinality_prior_and_boundary_move_probabilities_are_explicit() -> None:
    """K masses and boundary-renormalized proposal probabilities must be exact."""
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


@pytest.mark.parametrize("current_cardinality", [0, 1, 2])
def test_continuous_birth_and_reverse_death_are_reciprocal(
    current_cardinality: int,
) -> None:
    """A paired birth/death proposal must have opposite raw MH log ratios."""
    cardinality_prior = CardinalityPrior([0.19, 0.31, 0.29, 0.21])
    moves = BirthDeathMoveProbabilities(
        max_cardinality=3,
        birth_weight=1.7,
        death_weight=0.8,
    )
    likelihood_ratio = np.asarray([0.4, -0.7, 1.2])
    position_prior = np.asarray([-2.1, -1.7, -3.2])
    strength_prior = np.asarray([-0.2, -0.3, -0.5])
    position_proposal = np.asarray([-2.4, -1.4, -3.0])
    strength_proposal = np.asarray([-0.7, -0.1, -0.4])
    log_jacobian = np.asarray([0.0, 0.3, -0.2])

    birth_ratio = continuous_birth_log_acceptance_ratio(
        current_cardinality=current_cardinality,
        log_likelihood_ratio=likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_position_prior_density=position_prior,
        log_strength_prior_density=strength_prior,
        log_forward_position_proposal=position_proposal,
        log_forward_strength_proposal=strength_proposal,
        log_abs_jacobian=log_jacobian,
    )
    death_ratio = continuous_death_log_acceptance_ratio(
        current_cardinality=current_cardinality + 1,
        log_likelihood_ratio=-likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_removed_position_prior_density=position_prior,
        log_removed_strength_prior_density=strength_prior,
        log_reverse_position_proposal=position_proposal,
        log_reverse_strength_proposal=strength_proposal,
        log_abs_reverse_jacobian=-log_jacobian,
    )

    np.testing.assert_allclose(birth_ratio, -death_ratio, atol=1.0e-14)
    np.testing.assert_allclose(
        log_acceptance_probability(birth_ratio),
        np.minimum(birth_ratio, 0.0),
    )


@pytest.mark.parametrize("current_cardinality", [0, 1, 2])
def test_continuous_birth_death_flux_includes_canonical_factorial_density(
    current_cardinality: int,
) -> None:
    """Detailed balance must hold on the canonical unordered-state base measure."""
    proposed_cardinality = current_cardinality + 1
    cardinality_prior = CardinalityPrior([0.19, 0.31, 0.29, 0.21])
    moves = BirthDeathMoveProbabilities(
        max_cardinality=3,
        birth_weight=1.7,
        death_weight=0.8,
    )
    log_position_prior = -1.83
    log_strength_prior = -0.47
    log_position_proposal = -2.15
    log_strength_proposal = -0.26
    log_likelihood_current = -4.2
    log_likelihood_proposed = -3.7
    likelihood_ratio = log_likelihood_proposed - log_likelihood_current

    birth_ratio = float(
        continuous_birth_log_acceptance_ratio(
            current_cardinality=current_cardinality,
            log_likelihood_ratio=likelihood_ratio,
            cardinality_prior=cardinality_prior,
            move_probabilities=moves,
            log_position_prior_density=log_position_prior,
            log_strength_prior_density=log_strength_prior,
            log_forward_position_proposal=log_position_proposal,
            log_forward_strength_proposal=log_strength_proposal,
        )[0]
    )
    death_ratio = float(
        continuous_death_log_acceptance_ratio(
            current_cardinality=proposed_cardinality,
            log_likelihood_ratio=-likelihood_ratio,
            cardinality_prior=cardinality_prior,
            move_probabilities=moves,
            log_removed_position_prior_density=log_position_prior,
            log_removed_strength_prior_density=log_strength_prior,
            log_reverse_position_proposal=log_position_proposal,
            log_reverse_strength_proposal=log_strength_proposal,
        )[0]
    )

    # Canonically ordering K iid labeled sources changes the density by K!.
    log_target_current = (
        log_likelihood_current
        + float(cardinality_prior.log_prob(current_cardinality))
        + math.lgamma(current_cardinality + 1.0)
        + current_cardinality * (log_position_prior + log_strength_prior)
    )
    log_target_proposed = (
        log_likelihood_proposed
        + float(cardinality_prior.log_prob(proposed_cardinality))
        + math.lgamma(proposed_cardinality + 1.0)
        + proposed_cardinality * (log_position_prior + log_strength_prior)
    )
    log_forward = (
        float(moves.log_probability("birth", current_cardinality))
        + log_position_proposal
        + log_strength_proposal
        + min(0.0, birth_ratio)
    )
    log_reverse = (
        float(moves.log_probability("death", proposed_cardinality))
        - math.log(proposed_cardinality)
        + min(0.0, death_ratio)
    )
    assert log_target_current + log_forward == pytest.approx(
        log_target_proposed + log_reverse,
        abs=2.0e-13,
    )


def test_continuous_global_position_ratio_is_reciprocal() -> None:
    """Swapping old/new target and proposal densities must negate the MH ratio."""
    likelihood_ratio = np.asarray([0.7, -0.4, 0.2])
    old_prior = np.asarray([-3.1, -2.2, -1.4])
    new_prior = np.asarray([-1.8, -3.0, -2.3])
    forward = np.asarray([-2.7, -1.9, -2.1])
    reverse = np.asarray([-2.0, -2.5, -1.7])
    jacobian = np.asarray([0.0, 0.2, -0.1])

    observed = continuous_position_log_acceptance_ratio(
        log_likelihood_ratio=likelihood_ratio,
        log_old_position_prior_density=old_prior,
        log_new_position_prior_density=new_prior,
        log_reverse_proposal_density=reverse,
        log_forward_proposal_density=forward,
        log_abs_jacobian=jacobian,
    )
    reversed_observed = continuous_position_log_acceptance_ratio(
        log_likelihood_ratio=-likelihood_ratio,
        log_old_position_prior_density=new_prior,
        log_new_position_prior_density=old_prior,
        log_reverse_proposal_density=forward,
        log_forward_proposal_density=reverse,
        log_abs_jacobian=-jacobian,
    )

    np.testing.assert_allclose(observed, -reversed_observed, atol=1.0e-14)


def test_continuous_joint_position_strength_ratio_is_reciprocal() -> None:
    """Joint global moves must include and reverse both proposal densities."""
    terms = {
        "log_likelihood_ratio": np.asarray([0.8, -0.3, 0.1]),
        "log_old_position_prior_density": np.asarray([-3.0, -2.4, -1.5]),
        "log_new_position_prior_density": np.asarray([-1.7, -2.9, -2.2]),
        "log_old_strength_prior_density": np.asarray([-6.2, -5.9, -6.0]),
        "log_new_strength_prior_density": np.asarray([-5.8, -6.4, -5.7]),
        "log_reverse_position_proposal_density": np.asarray(
            [-2.1, -2.5, -1.8]
        ),
        "log_forward_position_proposal_density": np.asarray(
            [-2.8, -1.9, -2.0]
        ),
        "log_reverse_strength_proposal_density": np.asarray(
            [-6.0, -6.3, -5.6]
        ),
        "log_forward_strength_proposal_density": np.asarray(
            [-5.7, -6.1, -6.4]
        ),
        "log_abs_jacobian": np.asarray([0.0, 0.2, -0.1]),
    }
    observed = continuous_joint_position_strength_log_acceptance_ratio(
        **terms
    )
    reversed_observed = (
        continuous_joint_position_strength_log_acceptance_ratio(
            log_likelihood_ratio=-terms["log_likelihood_ratio"],
            log_old_position_prior_density=terms[
                "log_new_position_prior_density"
            ],
            log_new_position_prior_density=terms[
                "log_old_position_prior_density"
            ],
            log_old_strength_prior_density=terms[
                "log_new_strength_prior_density"
            ],
            log_new_strength_prior_density=terms[
                "log_old_strength_prior_density"
            ],
            log_reverse_position_proposal_density=terms[
                "log_forward_position_proposal_density"
            ],
            log_forward_position_proposal_density=terms[
                "log_reverse_position_proposal_density"
            ],
            log_reverse_strength_proposal_density=terms[
                "log_forward_strength_proposal_density"
            ],
            log_forward_strength_proposal_density=terms[
                "log_reverse_strength_proposal_density"
            ],
            log_abs_jacobian=-terms["log_abs_jacobian"],
        )
    )

    np.testing.assert_allclose(observed, -reversed_observed, atol=1.0e-14)


def test_symmetric_local_position_ratio_reduces_to_likelihood_ratio() -> None:
    """A same-chart symmetric tangent proposal must add no artificial evidence."""
    likelihood_ratio = np.asarray([0.7, -0.4, 0.0])
    chart_log_density = np.asarray([-3.1, -2.2, -1.4])
    zeros = np.zeros(3)

    observed = continuous_position_log_acceptance_ratio(
        log_likelihood_ratio=likelihood_ratio,
        log_old_position_prior_density=chart_log_density,
        log_new_position_prior_density=chart_log_density,
        log_reverse_proposal_density=zeros,
        log_forward_proposal_density=zeros,
    )

    np.testing.assert_allclose(
        observed,
        likelihood_ratio,
        atol=4.0e-16,
        rtol=0.0,
    )


def test_split_fraction_bounds_enforce_both_strength_supports() -> None:
    """Feasible split fractions must keep both child strengths inside support."""
    totals = np.asarray([1.5, 2.0, 3.0, 5.0, 7.0])
    lower, upper, feasible = split_fraction_bounds(
        totals,
        minimum_strength=1.0,
        maximum_strength=3.0,
    )

    np.testing.assert_array_equal(feasible, [False, False, True, True, False])
    for index in np.flatnonzero(feasible):
        fractions = np.asarray([lower[index], upper[index]])
        child = totals[index] * fractions
        sibling = totals[index] - child
        assert np.all(child >= 1.0 - 1.0e-14)
        assert np.all(child <= 3.0 + 1.0e-14)
        assert np.all(sibling >= 1.0 - 1.0e-14)
        assert np.all(sibling <= 3.0 + 1.0e-14)


@pytest.mark.parametrize("current_cardinality", [1, 2])
def test_continuous_split_and_reverse_merge_are_reciprocal(
    current_cardinality: int,
) -> None:
    """Matching split/merge bookkeeping must cancel including its Jacobian."""
    cardinality_prior = CardinalityPrior([0.12, 0.27, 0.34, 0.27])
    moves = SplitMergeMoveProbabilities(
        max_cardinality=3,
        split_weight=1.4,
        merge_weight=0.6,
    )
    total_strength = np.asarray([2.7, 3.4, 4.1])
    likelihood_ratio = np.asarray([0.5, -0.3, 1.1])
    position_prior = np.asarray([-2.2, -1.7, -3.1])
    old_strength_prior = np.asarray([-0.8, -0.9, -1.0])
    retained_strength_prior = np.asarray([-0.4, -0.6, -0.7])
    new_strength_prior = np.asarray([-0.5, -0.7, -0.8])
    position_proposal = np.asarray([-2.0, -2.4, -2.7])
    fraction_proposal = np.asarray([0.2, -0.1, 0.4])

    split_ratio = continuous_split_log_acceptance_ratio(
        current_cardinality=current_cardinality,
        total_strength=total_strength,
        log_likelihood_ratio=likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_new_position_prior_density=position_prior,
        log_old_strength_prior_density=old_strength_prior,
        log_retained_strength_prior_density=retained_strength_prior,
        log_new_strength_prior_density=new_strength_prior,
        log_forward_position_proposal=position_proposal,
        log_forward_fraction_proposal=fraction_proposal,
    )
    merge_ratio = continuous_merge_log_acceptance_ratio(
        current_cardinality=current_cardinality + 1,
        merged_strength=total_strength,
        log_likelihood_ratio=-likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_deleted_position_prior_density=position_prior,
        log_deleted_strength_prior_density=new_strength_prior,
        log_retained_strength_prior_density=retained_strength_prior,
        log_merged_strength_prior_density=old_strength_prior,
        log_reverse_position_proposal=position_proposal,
        log_reverse_fraction_proposal=fraction_proposal,
    )

    np.testing.assert_allclose(split_ratio, -merge_ratio, atol=1.0e-14)


def test_nonuniform_local_split_merge_selection_is_reciprocal() -> None:
    """State-dependent parent/pair densities must cancel in reverse RJ moves."""
    cardinality_prior = CardinalityPrior([0.12, 0.27, 0.34, 0.27])
    moves = SplitMergeMoveProbabilities(max_cardinality=3)
    likelihood_ratio = np.asarray([0.4, -0.8], dtype=np.float64)
    total_strength = np.asarray([3.1, 4.2], dtype=np.float64)
    forward_parent = np.log(np.asarray([0.7, 0.3], dtype=np.float64))
    reverse_pair = np.log(np.asarray([0.55, 0.08], dtype=np.float64))

    split_ratio = continuous_split_log_acceptance_ratio(
        current_cardinality=2,
        total_strength=total_strength,
        log_likelihood_ratio=likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_new_position_prior_density=-1.7,
        log_old_strength_prior_density=-0.9,
        log_retained_strength_prior_density=-0.6,
        log_new_strength_prior_density=-0.7,
        log_forward_position_proposal=-1.2,
        log_forward_fraction_proposal=0.1,
        log_forward_parent_selection=forward_parent,
        log_reverse_pair_selection=reverse_pair,
    )
    merge_ratio = continuous_merge_log_acceptance_ratio(
        current_cardinality=3,
        merged_strength=total_strength,
        log_likelihood_ratio=-likelihood_ratio,
        cardinality_prior=cardinality_prior,
        move_probabilities=moves,
        log_deleted_position_prior_density=-1.7,
        log_deleted_strength_prior_density=-0.7,
        log_retained_strength_prior_density=-0.6,
        log_merged_strength_prior_density=-0.9,
        log_reverse_position_proposal=-1.2,
        log_reverse_fraction_proposal=0.1,
        log_forward_pair_selection=reverse_pair,
        log_reverse_parent_selection=forward_parent,
    )

    np.testing.assert_allclose(split_ratio, -merge_ratio, atol=1.0e-14)


def test_distance_weighted_pair_proposal_is_batched_and_full_support() -> None:
    """Nearby pairs must be preferred without making distant pairs unreachable."""
    distances = np.asarray(
        [
            [0.1, 1.0, np.inf],
            [2.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )

    probabilities = distance_weighted_ordered_pair_probabilities(
        distances,
        sigma_m=0.5,
        uniform_component_probability=0.1,
    )

    np.testing.assert_allclose(
        np.sum(probabilities, axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.all(probabilities > 0.0)
    assert probabilities[0, 0] > probabilities[0, 1] > probabilities[0, 2]
    assert probabilities[1, 1] > probabilities[1, 2] > probabilities[1, 0]
    scalar_oracle = []
    for row in distances:
        local = np.exp(-0.5 * np.square(row / 0.5))
        local /= np.sum(local)
        scalar_oracle.append(0.1 / row.size + 0.9 * local)
    np.testing.assert_allclose(
        probabilities,
        np.asarray(scalar_oracle, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_split_merge_flux_includes_selection_density_and_jacobian() -> None:
    """Canonical target, ordered merge choices, fraction density, and q Jacobian balance."""
    current_cardinality = 2
    proposed_cardinality = 3
    cardinality_prior = CardinalityPrior([0.12, 0.27, 0.34, 0.27])
    moves = SplitMergeMoveProbabilities(
        max_cardinality=3,
        split_weight=1.4,
        merge_weight=0.6,
    )
    total_strength = 3.4
    log_position_prior = -1.7
    log_old_strength_prior = -0.9
    log_retained_strength_prior = -0.6
    log_new_strength_prior = -0.7
    log_position_proposal = -2.4
    log_fraction_proposal = -0.1
    current_likelihood = -4.8
    proposed_likelihood = -3.9
    likelihood_ratio = proposed_likelihood - current_likelihood

    split_ratio = float(
        continuous_split_log_acceptance_ratio(
            current_cardinality=current_cardinality,
            total_strength=total_strength,
            log_likelihood_ratio=likelihood_ratio,
            cardinality_prior=cardinality_prior,
            move_probabilities=moves,
            log_new_position_prior_density=log_position_prior,
            log_old_strength_prior_density=log_old_strength_prior,
            log_retained_strength_prior_density=log_retained_strength_prior,
            log_new_strength_prior_density=log_new_strength_prior,
            log_forward_position_proposal=log_position_proposal,
            log_forward_fraction_proposal=log_fraction_proposal,
        )[0]
    )
    merge_ratio = float(
        continuous_merge_log_acceptance_ratio(
            current_cardinality=proposed_cardinality,
            merged_strength=total_strength,
            log_likelihood_ratio=-likelihood_ratio,
            cardinality_prior=cardinality_prior,
            move_probabilities=moves,
            log_deleted_position_prior_density=log_position_prior,
            log_deleted_strength_prior_density=log_new_strength_prior,
            log_retained_strength_prior_density=log_retained_strength_prior,
            log_merged_strength_prior_density=log_old_strength_prior,
            log_reverse_position_proposal=log_position_proposal,
            log_reverse_fraction_proposal=log_fraction_proposal,
        )[0]
    )

    common_position_strength = -2.0
    current_target = (
        current_likelihood
        + float(cardinality_prior.log_prob(current_cardinality))
        + math.lgamma(current_cardinality + 1.0)
        + common_position_strength
        + log_old_strength_prior
    )
    proposed_target = (
        proposed_likelihood
        + float(cardinality_prior.log_prob(proposed_cardinality))
        + math.lgamma(proposed_cardinality + 1.0)
        + common_position_strength
        + log_position_prior
        + log_retained_strength_prior
        + log_new_strength_prior
    )
    split_forward = (
        float(moves.log_probability("split", current_cardinality))
        - math.log(current_cardinality)
        + log_position_proposal
        + log_fraction_proposal
        + min(0.0, split_ratio)
    )
    merge_reverse = (
        float(moves.log_probability("merge", proposed_cardinality))
        - math.log(proposed_cardinality * current_cardinality)
        + min(0.0, merge_ratio)
        + math.log(total_strength)
    )
    # Expressing reverse flux in the split auxiliary coordinate introduces the
    # forward strength-map Jacobian d(q1, q2) / d(q, fraction) = q.
    assert current_target + split_forward == pytest.approx(
        proposed_target + merge_reverse,
        abs=2.0e-13,
    )


def test_invalid_continuous_boundary_moves_fail_fast() -> None:
    """Unavailable directions must raise instead of producing a bogus MH ratio."""
    prior = CardinalityPrior([0.2, 0.3, 0.5])
    moves = BirthDeathMoveProbabilities(max_cardinality=2)
    with pytest.raises(ValueError, match="birth exceeds"):
        continuous_birth_log_acceptance_ratio(
            current_cardinality=2,
            log_likelihood_ratio=0.0,
            cardinality_prior=prior,
            move_probabilities=moves,
            log_position_prior_density=0.0,
            log_strength_prior_density=0.0,
            log_forward_position_proposal=0.0,
            log_forward_strength_proposal=0.0,
        )
    with pytest.raises((TypeError, ValueError), match="positive"):
        continuous_death_log_acceptance_ratio(
            current_cardinality=0,
            log_likelihood_ratio=0.0,
            cardinality_prior=prior,
            move_probabilities=moves,
            log_removed_position_prior_density=0.0,
            log_removed_strength_prior_density=0.0,
            log_reverse_position_proposal=0.0,
            log_reverse_strength_proposal=0.0,
        )


def test_bounded_multi_split_density_has_exact_truncation_normalizer() -> None:
    """The multi-split fraction law must use its bounded-simplex mass."""
    totals = np.asarray([2.5, 4.5, 7.5, 13.0], dtype=np.float64)
    probability = bounded_simplex_probability(
        totals,
        group_size=3,
        minimum_strength=1.0,
        maximum_strength=4.0,
    )

    assert probability[0] == 0.0
    assert 0.0 < probability[1] <= 1.0
    assert 0.0 < probability[2] <= 1.0
    assert probability[3] == 0.0
    fractions = np.asarray([[0.25, 0.35, 0.40]], dtype=np.float64)
    density = bounded_uniform_simplex_log_density(
        fractions,
        total_strength=np.asarray([4.0]),
        minimum_strength=1.0,
        maximum_strength=4.0,
    )
    expected = math.lgamma(3.0) - math.log(
        float(
            bounded_simplex_probability(
                np.asarray([4.0]),
                group_size=3,
                minimum_strength=1.0,
                maximum_strength=4.0,
            )[0]
        )
    )
    assert float(density[0]) == pytest.approx(expected)


@pytest.mark.parametrize("group_size", [3, 4])
def test_multi_component_split_merge_ratios_are_reciprocal(
    group_size: int,
) -> None:
    """Matching multi-component proposal terms must reverse exactly."""
    target_ratio = np.asarray([1.7, -0.4], dtype=np.float64)
    forward = np.asarray([-8.2, -7.1], dtype=np.float64)
    reverse = np.asarray([-5.4, -6.6], dtype=np.float64)
    totals = np.asarray([4.5, 7.25], dtype=np.float64)

    split = continuous_group_split_log_acceptance_ratio(
        log_target_ratio=target_ratio,
        log_forward_proposal=forward,
        log_reverse_proposal=reverse,
        total_strength=totals,
        group_size=group_size,
    )
    merge = continuous_group_merge_log_acceptance_ratio(
        log_target_ratio=-target_ratio,
        log_forward_proposal=reverse,
        log_reverse_proposal=forward,
        merged_strength=totals,
        group_size=group_size,
    )

    np.testing.assert_allclose(split, -merge, atol=1.0e-14)
