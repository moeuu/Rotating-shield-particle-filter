"""Batched block-independence rejuvenation for exact-RJ PF."""

from __future__ import annotations

import math

import numpy as np

from pf.particle_filter_math import (
    extended_log_target_ratio as _extended_log_target_ratio,
)
from pf.particle_types import StructuralGeometryBatch

class StructuralRJBlockIndependenceMixin:
    """Provide the batched exact-RJ block-independence move."""

    def _apply_continuous_rj_block_independence(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply an exact full-isotope trans-dimensional independence move.

        The proposal replaces every source component of one isotope at once.
        It therefore crosses the low-likelihood intermediate states that make
        sequential deletion of several ghosts or merging several split
        components ineffective.  Cardinality, chart coordinates, and strength
        densities all have full prior support and are evaluated in both
        directions, so this is ordinary Metropolis-Hastings on the disjoint
        union of the continuous ``K``-source spaces and requires no implicit
        dimension-matching Jacobian.
        """
        probability = float(self.config.structural_rj_block_independence_probability)
        if probability <= 0.0:
            return 0, 0
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        if atlas is None or cardinality_prior is None:
            raise RuntimeError("Continuous block-RJ priors are unavailable.")
        particle_count = len(self.continuous_particles)
        attempted_indices = np.flatnonzero(
            self._random_generator.random(particle_count) < probability
        ).astype(np.int64, copy=False)
        if attempted_indices.size == 0:
            return 0, 0
        self._continuous_rj_transition_mass(
            "block_attempted",
            attempted_indices,
        )
        current_cardinalities = np.asarray(
            [
                self.continuous_particles[int(index)].state.num_sources
                for index in attempted_indices
            ],
            dtype=np.int64,
        )
        proposed_cardinalities = self._random_generator.choice(
            cardinality_prior.probabilities.size,
            size=attempted_indices.size,
            replace=True,
            p=cardinality_prior.probabilities,
        ).astype(np.int64, copy=False)
        base_ll = np.full(
            attempted_indices.size,
            float("-inf"),
            dtype=np.float64,
        )
        current_log_prior = np.full_like(base_ll, float("-inf"))
        current_log_proposal = np.full_like(base_ll, float("-inf"))
        for cardinality in np.unique(current_cardinalities).tolist():
            rows = np.flatnonzero(current_cardinalities == int(cardinality))
            particle_indices = attempted_indices[rows]
            charts, _, positions, strengths = self._continuous_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll[rows] = self._continuous_rj_current_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=charts,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            prior, proposal = self._continuous_rj_block_log_densities(
                charts,
                strengths,
            )
            current_log_prior[rows] = prior
            if int(cardinality) == 0:
                current_log_proposal[rows] = proposal
            else:
                block_strength_proposal = (
                    self._continuous_rj_conditional_block_strength_proposal(
                        data,
                        chart_ids=charts,
                        positions=positions,
                        particle_indices=particle_indices,
                        target_beta=target_beta,
                        cache_current_state=True,
                    )
                )
                current_log_proposal[rows] = (
                    float(cardinality_prior.log_prob(int(cardinality)))
                    + math.lgamma(float(cardinality) + 1.0)
                    + np.sum(
                        self._active_continuous_rj_position_proposal().log_density(
                            charts
                        ),
                        axis=1,
                    )
                    + block_strength_proposal.log_density(strengths)
                )
        accepted_count = 0
        cardinality_change_count = 0
        for cardinality in np.unique(proposed_cardinalities).tolist():
            rows = np.flatnonzero(proposed_cardinalities == int(cardinality))
            particle_indices = attempted_indices[rows]
            row_count = int(rows.size)
            source_count = row_count * int(cardinality)
            charts_flat, uv_flat, positions_flat = atlas.sample(
                source_count,
                rng=self._random_generator,
                chart_probabilities=(
                    self._active_continuous_rj_position_proposal().chart_probabilities
                ),
            )
            charts = charts_flat.reshape(row_count, int(cardinality))
            uv = uv_flat.reshape(row_count, int(cardinality), 2)
            positions = positions_flat.reshape(
                row_count,
                int(cardinality),
                3,
            )
            if int(cardinality) == 0:
                strengths = np.zeros((row_count, 0), dtype=np.float64)
                proposed_strength_log_proposal = np.zeros(
                    row_count,
                    dtype=np.float64,
                )
            else:
                block_strength_proposal = (
                    self._continuous_rj_conditional_block_strength_proposal(
                        data,
                        chart_ids=charts,
                        positions=positions,
                        particle_indices=particle_indices,
                        target_beta=target_beta,
                    )
                )
                strengths = block_strength_proposal.sample(
                    rng=self._random_generator,
                )
                proposed_strength_log_proposal = block_strength_proposal.log_density(
                    strengths
                )
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=charts,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            proposed_log_prior, _ = self._continuous_rj_block_log_densities(
                charts,
                strengths,
            )
            proposed_log_proposal = np.full(
                row_count,
                float(cardinality_prior.log_prob(int(cardinality)))
                + math.lgamma(float(cardinality) + 1.0),
                dtype=np.float64,
            )
            if int(cardinality):
                proposed_log_proposal += np.sum(
                    self._active_continuous_rj_position_proposal().log_density(charts),
                    axis=1,
                )
                proposed_log_proposal += proposed_strength_log_proposal
            log_ratio = (
                _extended_log_target_ratio(proposed_ll, base_ll[rows])
                + proposed_log_prior
                - current_log_prior[rows]
                + current_log_proposal[rows]
                - proposed_log_proposal
            )
            accepted = self._continuous_rj_mh_acceptance_mask(log_ratio)
            block_support = (
                np.isfinite(proposed_log_prior)
                & np.isfinite(proposed_log_proposal)
                & np.isfinite(current_log_prior[rows])
                & np.isfinite(current_log_proposal[rows])
            )
            self._record_structural_mh_components(
                "block_independence",
                delta_log_likelihood=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll[rows],
                ),
                delta_log_prior=(proposed_log_prior - current_log_prior[rows]),
                log_reverse_minus_forward=(
                    current_log_proposal[rows] - proposed_log_proposal
                ),
                log_jacobian=np.zeros(row_count, dtype=np.float64),
                support_feasible=block_support,
                accepted=accepted,
                current_cardinality=current_cardinalities[rows],
                proposed_cardinality=int(cardinality),
                geometry_support_feasible=(
                    np.isfinite(proposed_log_proposal)
                    & np.isfinite(current_log_proposal[rows])
                ),
                strength_support_feasible=np.all(
                    self._strength_prior.in_support(strengths),
                    axis=1,
                )
                if int(cardinality)
                else np.ones(row_count, dtype=np.bool_),
                log_acceptance_ratio=log_ratio,
            )
            accepted_count += self._commit_continuous_rj_states(
                particle_indices,
                accepted,
                charts,
                uv,
                positions,
                strengths,
            )
            changed = accepted & (current_cardinalities[rows] != int(cardinality))
            cardinality_change_count += int(np.sum(changed))
            self._update_continuous_rj_current_log_likelihood(
                particle_indices,
                accepted,
                proposed_ll,
            )
            self._continuous_rj_transition_mass(
                "block_accepted",
                particle_indices,
                accepted,
            )
            self._continuous_rj_transition_mass(
                "block_cardinality_changed",
                particle_indices,
                changed,
            )
        self._structural_rj_move_counts.update(
            {
                "block_attempted": int(attempted_indices.size),
                "block_accepted": int(accepted_count),
                "block_cardinality_changed": int(cardinality_change_count),
            }
        )
        return accepted_count, cardinality_change_count


__all__ = ["StructuralRJBlockIndependenceMixin"]
