"""Batched reversible split/merge algorithms for exact-RJ PF."""

from __future__ import annotations

import math

import numpy as np

from pf.particle_filter_math import (
    extended_log_target_ratio as _extended_log_target_ratio,
)
from pf.particle_types import StructuralGeometryBatch
from pf.structural_rj import (
    continuous_relocated_merge_log_acceptance_ratio,
    continuous_relocated_split_log_acceptance_ratio,
    split_fraction_bounds,
)

class StructuralRJSplitMergeMixin:
    """Provide the batched exact-RJ relocated split/merge move."""

    def _apply_continuous_rj_split_merge(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact strength-transfer split/merge RJ proposals.

        A split keeps one source position, draws a second position from the
        full-support local surface mixture, and partitions the original
        strength.  Its reverse merge selects nearby ordered donor/recipient
        pairs preferentially and transfers donor strength to the recipient.
        Both state-dependent selection probabilities, the strength-map
        Jacobian, and the truncated split-fraction density are included in the
        RJ ratio.
        """
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_split_merge_probabilities
        if atlas is None or cardinality_prior is None or move_probabilities is None:
            raise RuntimeError("Continuous split/merge priors are unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        split_probabilities, merge_probabilities = move_probabilities.probabilities(
            cardinalities
        )
        direction_available = (
            np.asarray(split_probabilities, dtype=np.float64)
            + np.asarray(merge_probabilities, dtype=np.float64)
        ) > 0.0
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_split_merge_probability)
        ) & direction_available
        attempted_splits = 0
        accepted_splits = 0
        attempted_merges = 0
        accepted_merges = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            split_probability, _ = move_probabilities.probabilities(int(cardinality))
            split_move = self._random_generator.random(group_indices.size) < float(
                split_probability
            )
            for is_split in (True, False):
                selected_rows = np.flatnonzero(split_move == is_split)
                if selected_rows.size == 0:
                    continue
                particle_indices = group_indices[selected_rows]
                (
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_arrays(
                    particle_indices,
                    int(cardinality),
                )
                base_ll = self._continuous_rj_current_log_likelihood(
                    data,
                    positions,
                    strengths,
                    chart_ids=chart_ids,
                    particle_indices=particle_indices,
                    target_beta=target_beta,
                )
                rows = np.arange(particle_indices.size, dtype=np.int64)
                if is_split:
                    attempted_splits += int(particle_indices.size)
                    self._continuous_rj_transition_mass(
                        "split_attempted",
                        particle_indices,
                    )
                    source_columns = self._random_generator.integers(
                        0,
                        int(cardinality),
                        size=particle_indices.size,
                        dtype=np.int64,
                    )
                    total_strength = strengths[rows, source_columns]
                    lower, upper, feasible = split_fraction_bounds(
                        total_strength,
                        minimum_strength=self._strength_prior.minimum,
                        maximum_strength=self._strength_prior.support_maximum,
                    )
                    width = upper - lower
                    safe_width = np.where(feasible, width, 1.0)
                    fraction = lower + safe_width * self._random_generator.random(
                        particle_indices.size
                    )
                    retained_strength = (1.0 - fraction) * total_strength
                    new_strength = fraction * total_strength
                    parent_chart_ids = chart_ids[rows, source_columns]
                    (
                        first_child_chart_ids,
                        first_child_uv,
                        first_child_positions,
                        log_forward_first_position_proposal,
                    ) = atlas.sample_local_chart_mixture(
                        parent_chart_ids,
                        global_component_probability=float(
                            self.config.structural_rj_split_global_position_probability
                        ),
                        rng=self._random_generator,
                    )
                    (
                        second_child_chart_ids,
                        second_child_uv,
                        second_child_positions,
                        log_forward_second_position_proposal,
                    ) = atlas.sample_local_chart_mixture(
                        parent_chart_ids,
                        global_component_probability=float(
                            self.config.structural_rj_split_global_position_probability
                        ),
                        rng=self._random_generator,
                    )
                    keep = (
                        np.arange(int(cardinality))[None, :] != source_columns[:, None]
                    )
                    retained_chart_ids = chart_ids[keep].reshape(
                        particle_indices.size,
                        int(cardinality) - 1,
                    )
                    retained_uv = surface_uv[keep].reshape(
                        particle_indices.size,
                        int(cardinality) - 1,
                        2,
                    )
                    retained_positions = positions[keep].reshape(
                        particle_indices.size,
                        int(cardinality) - 1,
                        3,
                    )
                    retained_strength_matrix = strengths[keep].reshape(
                        particle_indices.size,
                        int(cardinality) - 1,
                    )
                    proposed_chart_ids = np.concatenate(
                        (
                            retained_chart_ids,
                            first_child_chart_ids[:, None],
                            second_child_chart_ids[:, None],
                        ),
                        axis=1,
                    )
                    proposed_uv = np.concatenate(
                        (
                            retained_uv,
                            first_child_uv[:, None, :],
                            second_child_uv[:, None, :],
                        ),
                        axis=1,
                    )
                    proposed_positions = np.concatenate(
                        (
                            retained_positions,
                            first_child_positions[:, None, :],
                            second_child_positions[:, None, :],
                        ),
                        axis=1,
                    )
                    proposed_strengths = np.concatenate(
                        (
                            retained_strength_matrix,
                            retained_strength[:, None],
                            new_strength[:, None],
                        ),
                        axis=1,
                    )
                    (
                        reverse_donor_columns,
                        reverse_receiver_columns,
                        reverse_pair_probabilities,
                    ) = self._continuous_rj_ordered_pair_probabilities(
                        proposed_chart_ids,
                        proposed_uv,
                    )
                    reverse_pair_matches = (
                        reverse_donor_columns[None, :] == int(cardinality)
                    ) & (reverse_receiver_columns[None, :] == int(cardinality) - 1)
                    if not np.all(np.sum(reverse_pair_matches, axis=1) == 1):
                        raise RuntimeError(
                            "Reverse merge pair was not uniquely represented."
                        )
                    reverse_pair_columns = np.argmax(
                        reverse_pair_matches,
                        axis=1,
                    )
                    log_reverse_pair_selection = np.log(
                        reverse_pair_probabilities[
                            rows,
                            reverse_pair_columns,
                        ]
                    )
                    global_probability = float(
                        self.config.structural_rj_split_global_position_probability
                    )
                    log_reverse_merged_position_proposal = np.logaddexp(
                        atlas.local_chart_mixture_log_density(
                            first_child_chart_ids,
                            parent_chart_ids,
                            global_component_probability=(global_probability),
                        ),
                        atlas.local_chart_mixture_log_density(
                            second_child_chart_ids,
                            parent_chart_ids,
                            global_component_probability=(global_probability),
                        ),
                    ) - math.log(2.0)
                    (
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    ) = self._continuous_rj_canonicalize_rows(
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    log_ratio = np.full(
                        particle_indices.size,
                        float("-inf"),
                        dtype=np.float64,
                    )
                    proposed_ll = np.full(
                        particle_indices.size,
                        float("-inf"),
                        dtype=np.float64,
                    )
                    valid_rows = np.flatnonzero(feasible)
                    if valid_rows.size:
                        proposed_ll[valid_rows] = (
                            self._continuous_rj_group_log_likelihood(
                                data,
                                proposed_positions[valid_rows],
                                proposed_strengths[valid_rows],
                                chart_ids=proposed_chart_ids[valid_rows],
                                particle_indices=particle_indices[valid_rows],
                                target_beta=target_beta,
                            )
                        )
                        log_ratio[valid_rows] = (
                            continuous_relocated_split_log_acceptance_ratio(
                                current_cardinality=int(cardinality),
                                total_strength=total_strength[valid_rows],
                                log_likelihood_ratio=(
                                    _extended_log_target_ratio(
                                        proposed_ll[valid_rows],
                                        base_ll[valid_rows],
                                    )
                                ),
                                cardinality_prior=cardinality_prior,
                                move_probabilities=move_probabilities,
                                log_parent_position_prior_density=(
                                    atlas.log_chart_probabilities[
                                        parent_chart_ids[valid_rows]
                                    ]
                                ),
                                log_first_child_position_prior_density=(
                                    atlas.log_chart_probabilities[
                                        first_child_chart_ids[valid_rows]
                                    ]
                                ),
                                log_second_child_position_prior_density=(
                                    atlas.log_chart_probabilities[
                                        second_child_chart_ids[valid_rows]
                                    ]
                                ),
                                log_parent_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        total_strength[valid_rows]
                                    )
                                ),
                                log_first_child_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        retained_strength[valid_rows]
                                    )
                                ),
                                log_second_child_strength_prior_density=(
                                    self._strength_prior.log_prob(
                                        new_strength[valid_rows]
                                    )
                                ),
                                log_forward_first_position_proposal=(
                                    log_forward_first_position_proposal[valid_rows]
                                ),
                                log_forward_second_position_proposal=(
                                    log_forward_second_position_proposal[valid_rows]
                                ),
                                log_forward_fraction_proposal=(
                                    -np.log(width[valid_rows])
                                ),
                                log_forward_parent_selection=(
                                    np.full(
                                        valid_rows.size,
                                        -math.log(float(cardinality)),
                                        dtype=np.float64,
                                    )
                                ),
                                log_reverse_pair_selection=(
                                    log_reverse_pair_selection[valid_rows]
                                ),
                                log_reverse_merged_position_proposal=(
                                    log_reverse_merged_position_proposal[valid_rows]
                                ),
                            )
                        )
                    accepted = self._continuous_rj_mh_acceptance_mask(
                        log_ratio,
                        support=feasible,
                    )
                    split_delta_likelihood = _extended_log_target_ratio(
                        proposed_ll,
                        base_ll,
                    )
                    split_delta_prior = np.full(
                        particle_indices.size,
                        float("nan"),
                        dtype=np.float64,
                    )
                    split_proposal_ratio = np.full_like(
                        split_delta_prior,
                        float("nan"),
                    )
                    split_log_jacobian = np.full_like(
                        split_delta_prior,
                        float("nan"),
                    )
                    split_geometry_support = np.ones(
                        particle_indices.size,
                        dtype=np.bool_,
                    )
                    if valid_rows.size:
                        split_delta_prior[valid_rows] = (
                            float(cardinality_prior.log_prob(int(cardinality) + 1))
                            - float(cardinality_prior.log_prob(int(cardinality)))
                            + math.log(float(int(cardinality) + 1))
                            + atlas.log_chart_probabilities[
                                first_child_chart_ids[valid_rows]
                            ]
                            + atlas.log_chart_probabilities[
                                second_child_chart_ids[valid_rows]
                            ]
                            - atlas.log_chart_probabilities[
                                parent_chart_ids[valid_rows]
                            ]
                            + self._strength_prior.log_prob(
                                retained_strength[valid_rows]
                            )
                            + self._strength_prior.log_prob(new_strength[valid_rows])
                            - self._strength_prior.log_prob(total_strength[valid_rows])
                        )
                        split_log_jacobian[valid_rows] = np.log(
                            total_strength[valid_rows]
                        )
                        split_proposal_ratio[valid_rows] = (
                            log_ratio[valid_rows]
                            - split_delta_likelihood[valid_rows]
                            - split_delta_prior[valid_rows]
                            - split_log_jacobian[valid_rows]
                        )
                        split_geometry_support[valid_rows] = np.isfinite(
                            split_proposal_ratio[valid_rows]
                        )
                    self._record_structural_mh_components(
                        "split",
                        delta_log_likelihood=split_delta_likelihood,
                        delta_log_prior=split_delta_prior,
                        log_reverse_minus_forward=split_proposal_ratio,
                        log_jacobian=split_log_jacobian,
                        support_feasible=feasible,
                        accepted=accepted,
                        current_cardinality=int(cardinality),
                        proposed_cardinality=int(cardinality) + 1,
                        geometry_support_feasible=split_geometry_support,
                        strength_support_feasible=feasible,
                        log_acceptance_ratio=log_ratio,
                    )
                    accepted_splits += self._commit_continuous_rj_states(
                        particle_indices,
                        accepted,
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    self._update_continuous_rj_current_log_likelihood(
                        particle_indices,
                        accepted,
                        proposed_ll,
                    )
                    self._continuous_rj_transition_mass(
                        "split_accepted",
                        particle_indices,
                        accepted,
                    )
                    continue

                attempted_merges += int(particle_indices.size)
                self._continuous_rj_transition_mass(
                    "merge_attempted",
                    particle_indices,
                )
                (
                    donor_candidates,
                    receiver_candidates,
                    pair_probabilities,
                ) = self._continuous_rj_ordered_pair_probabilities(
                    chart_ids,
                    surface_uv,
                )
                pair_cdf = np.cumsum(pair_probabilities, axis=1)
                pair_cdf[:, -1] = 1.0
                pair_draws = self._random_generator.random(particle_indices.size)
                pair_columns = np.sum(
                    pair_draws[:, None] > pair_cdf,
                    axis=1,
                    dtype=np.int64,
                )
                delete_columns = donor_candidates[pair_columns]
                receiver_columns = receiver_candidates[pair_columns]
                log_forward_pair_selection = np.log(
                    pair_probabilities[rows, pair_columns]
                )
                second_child_chart_ids = chart_ids[rows, delete_columns]
                first_child_chart_ids = chart_ids[rows, receiver_columns]
                second_child_strengths = strengths[rows, delete_columns]
                first_child_strengths = strengths[rows, receiver_columns]
                merged_strength = second_child_strengths + first_child_strengths
                lower, upper, reverse_feasible = split_fraction_bounds(
                    merged_strength,
                    minimum_strength=self._strength_prior.minimum,
                    maximum_strength=self._strength_prior.support_maximum,
                )
                reverse_fraction = second_child_strengths / np.maximum(
                    merged_strength,
                    np.finfo(np.float64).tiny,
                )
                feasible = (
                    np.asarray(
                        self._strength_prior.in_support(merged_strength),
                        dtype=bool,
                    )
                    & reverse_feasible
                    & (reverse_fraction >= lower)
                    & (reverse_fraction <= upper)
                )
                global_probability = float(
                    self.config.structural_rj_split_global_position_probability
                )
                use_first_anchor = (
                    self._random_generator.random(particle_indices.size) < 0.5
                )
                merge_anchor_chart_ids = np.where(
                    use_first_anchor,
                    first_child_chart_ids,
                    second_child_chart_ids,
                )
                (
                    merged_chart_ids,
                    merged_uv,
                    merged_positions,
                    _sampled_merged_position_log_density,
                ) = atlas.sample_local_chart_mixture(
                    merge_anchor_chart_ids,
                    global_component_probability=global_probability,
                    rng=self._random_generator,
                )
                log_forward_merged_position_proposal = np.logaddexp(
                    atlas.local_chart_mixture_log_density(
                        first_child_chart_ids,
                        merged_chart_ids,
                        global_component_probability=(global_probability),
                    ),
                    atlas.local_chart_mixture_log_density(
                        second_child_chart_ids,
                        merged_chart_ids,
                        global_component_probability=(global_probability),
                    ),
                ) - math.log(2.0)
                keep = (
                    np.arange(int(cardinality))[None, :] != delete_columns[:, None]
                ) & (np.arange(int(cardinality))[None, :] != receiver_columns[:, None])
                retained_chart_ids = chart_ids[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 2,
                )
                retained_uv = surface_uv[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 2,
                    2,
                )
                retained_positions = positions[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 2,
                    3,
                )
                retained_strengths = strengths[keep].reshape(
                    particle_indices.size,
                    int(cardinality) - 2,
                )
                proposed_chart_ids = np.concatenate(
                    (
                        retained_chart_ids,
                        merged_chart_ids[:, None],
                    ),
                    axis=1,
                )
                proposed_uv = np.concatenate(
                    (
                        retained_uv,
                        merged_uv[:, None, :],
                    ),
                    axis=1,
                )
                proposed_positions = np.concatenate(
                    (
                        retained_positions,
                        merged_positions[:, None, :],
                    ),
                    axis=1,
                )
                proposed_strengths = np.concatenate(
                    (
                        retained_strengths,
                        merged_strength[:, None],
                    ),
                    axis=1,
                )
                (
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                ) = self._continuous_rj_canonicalize_rows(
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                log_ratio = np.full(
                    particle_indices.size,
                    float("-inf"),
                    dtype=np.float64,
                )
                proposed_ll = np.full(
                    particle_indices.size,
                    float("-inf"),
                    dtype=np.float64,
                )
                valid_rows = np.flatnonzero(feasible)
                if valid_rows.size:
                    proposed_ll[valid_rows] = self._continuous_rj_group_log_likelihood(
                        data,
                        proposed_positions[valid_rows],
                        proposed_strengths[valid_rows],
                        chart_ids=proposed_chart_ids[valid_rows],
                        particle_indices=particle_indices[valid_rows],
                        target_beta=target_beta,
                    )
                    width = upper[valid_rows] - lower[valid_rows]
                    log_ratio[valid_rows] = (
                        continuous_relocated_merge_log_acceptance_ratio(
                            current_cardinality=int(cardinality),
                            merged_strength=merged_strength[valid_rows],
                            log_likelihood_ratio=(
                                _extended_log_target_ratio(
                                    proposed_ll[valid_rows],
                                    base_ll[valid_rows],
                                )
                            ),
                            cardinality_prior=cardinality_prior,
                            move_probabilities=move_probabilities,
                            log_first_child_position_prior_density=(
                                atlas.log_chart_probabilities[
                                    first_child_chart_ids[valid_rows]
                                ]
                            ),
                            log_second_child_position_prior_density=(
                                atlas.log_chart_probabilities[
                                    second_child_chart_ids[valid_rows]
                                ]
                            ),
                            log_merged_position_prior_density=(
                                atlas.log_chart_probabilities[
                                    merged_chart_ids[valid_rows]
                                ]
                            ),
                            log_first_child_strength_prior_density=(
                                self._strength_prior.log_prob(
                                    first_child_strengths[valid_rows]
                                )
                            ),
                            log_second_child_strength_prior_density=(
                                self._strength_prior.log_prob(
                                    second_child_strengths[valid_rows]
                                )
                            ),
                            log_merged_strength_prior_density=(
                                self._strength_prior.log_prob(
                                    merged_strength[valid_rows]
                                )
                            ),
                            log_forward_merged_position_proposal=(
                                log_forward_merged_position_proposal[valid_rows]
                            ),
                            log_reverse_first_position_proposal=(
                                atlas.local_chart_mixture_log_density(
                                    merged_chart_ids[valid_rows],
                                    first_child_chart_ids[valid_rows],
                                    global_component_probability=(global_probability),
                                )
                            ),
                            log_reverse_second_position_proposal=(
                                atlas.local_chart_mixture_log_density(
                                    merged_chart_ids[valid_rows],
                                    second_child_chart_ids[valid_rows],
                                    global_component_probability=(global_probability),
                                )
                            ),
                            log_reverse_fraction_proposal=-np.log(width),
                            log_forward_pair_selection=(
                                log_forward_pair_selection[valid_rows]
                            ),
                            log_reverse_parent_selection=np.full(
                                valid_rows.size,
                                -math.log(float(cardinality - 1)),
                                dtype=np.float64,
                            ),
                        )
                    )
                accepted = self._continuous_rj_mh_acceptance_mask(
                    log_ratio,
                    support=feasible,
                )
                merge_delta_likelihood = _extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                )
                merge_delta_prior = np.full(
                    particle_indices.size,
                    float("nan"),
                    dtype=np.float64,
                )
                merge_proposal_ratio = np.full_like(
                    merge_delta_prior,
                    float("nan"),
                )
                merge_log_jacobian = np.full_like(
                    merge_delta_prior,
                    float("nan"),
                )
                merge_geometry_support = np.ones(
                    particle_indices.size,
                    dtype=np.bool_,
                )
                if valid_rows.size:
                    merge_delta_prior[valid_rows] = (
                        float(cardinality_prior.log_prob(int(cardinality) - 1))
                        - float(cardinality_prior.log_prob(int(cardinality)))
                        - math.log(float(int(cardinality)))
                        + atlas.log_chart_probabilities[merged_chart_ids[valid_rows]]
                        - atlas.log_chart_probabilities[
                            first_child_chart_ids[valid_rows]
                        ]
                        - atlas.log_chart_probabilities[
                            second_child_chart_ids[valid_rows]
                        ]
                        + self._strength_prior.log_prob(merged_strength[valid_rows])
                        - self._strength_prior.log_prob(
                            first_child_strengths[valid_rows]
                        )
                        - self._strength_prior.log_prob(
                            second_child_strengths[valid_rows]
                        )
                    )
                    merge_log_jacobian[valid_rows] = -np.log(
                        merged_strength[valid_rows]
                    )
                    merge_proposal_ratio[valid_rows] = (
                        log_ratio[valid_rows]
                        - merge_delta_likelihood[valid_rows]
                        - merge_delta_prior[valid_rows]
                        - merge_log_jacobian[valid_rows]
                    )
                    merge_geometry_support[valid_rows] = np.isfinite(
                        merge_proposal_ratio[valid_rows]
                    )
                self._record_structural_mh_components(
                    "merge",
                    delta_log_likelihood=merge_delta_likelihood,
                    delta_log_prior=merge_delta_prior,
                    log_reverse_minus_forward=merge_proposal_ratio,
                    log_jacobian=merge_log_jacobian,
                    support_feasible=feasible,
                    accepted=accepted,
                    current_cardinality=int(cardinality),
                    proposed_cardinality=int(cardinality) - 1,
                    geometry_support_feasible=merge_geometry_support,
                    strength_support_feasible=feasible,
                    log_acceptance_ratio=log_ratio,
                )
                accepted_merges += self._commit_continuous_rj_states(
                    particle_indices,
                    accepted,
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                self._update_continuous_rj_current_log_likelihood(
                    particle_indices,
                    accepted,
                    proposed_ll,
                )
                self._continuous_rj_transition_mass(
                    "merge_accepted",
                    particle_indices,
                    accepted,
                )
        self._structural_rj_move_counts.update(
            {
                "split_attempted": attempted_splits,
                "split_accepted": accepted_splits,
                "merge_attempted": attempted_merges,
                "merge_accepted": accepted_merges,
            }
        )
        return accepted_splits, accepted_merges


__all__ = ["StructuralRJSplitMergeMixin"]
