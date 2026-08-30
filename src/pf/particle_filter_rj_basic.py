"""Batched birth, death, position, and strength moves for exact-RJ PF."""

from __future__ import annotations

import math

import numpy as np

from pf.particle_filter_math import (
    extended_log_target_ratio as _extended_log_target_ratio,
)
from pf.particle_types import StructuralGeometryBatch
from pf.structural_rj import (
    continuous_birth_log_acceptance_ratio,
    continuous_death_log_acceptance_ratio,
    continuous_joint_position_strength_log_acceptance_ratio,
)


class StructuralRJBasicMoveMixin:
    """Provide batched elementary exact-RJ rejuvenation moves."""

    def _continuous_rj_integrated_unit_response(
        self,
        data: StructuralGeometryBatch,
        positions: np.ndarray,
        chart_ids: np.ndarray,
    ) -> np.ndarray:
        """Return all-history expected source counts per unit strength.

        Positions from every particle row are evaluated in one transport batch.
        This response is used only to define an invertible proposal map; the
        full-spectrum likelihood remains the MH target.
        """
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        charts = np.asarray(chart_ids, dtype=np.int64).reshape(-1)
        if points.shape[0] != charts.size:
            raise ValueError("Unit-response positions and charts must align.")
        line_indices = self.continuous_kernel.positive_line_indices(self.isotope)
        branching = np.asarray(
            self.continuous_kernel.line_branching_weights(
                self.isotope,
                line_indices,
            ),
            dtype=np.float64,
        )
        components = self._continuous_rj_line_transport_component_columns(
            data,
            points,
            line_indices,
            chart_ids=charts,
        )
        unit_rates = np.asarray(components.total_kernel, dtype=np.float64)
        expected_shape = (data.row_count, points.shape[0], line_indices.size)
        if unit_rates.shape != expected_shape:
            raise RuntimeError("Coupled-move unit response has an invalid shape.")
        live_times = np.asarray(data.live_times, dtype=np.float64).reshape(-1)
        response = np.einsum(
            "vsl,l,v->s",
            unit_rates,
            branching,
            live_times,
            optimize=True,
        )
        if response.shape != (points.shape[0],) or np.any(~np.isfinite(response)):
            raise RuntimeError("Coupled-move unit response is numerically invalid.")
        return np.asarray(response, dtype=np.float64)

    def _apply_continuous_rj_birth_death(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply one exact continuous-surface birth/death attempt per particle."""
        if self._continuous_rj_torch_enabled():
            return self._apply_continuous_rj_birth_death_torch(
                data,
                target_beta=target_beta,
            )
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_move_probabilities
        if atlas is None or cardinality_prior is None or move_probabilities is None:
            raise RuntimeError("Continuous RJ priors are unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        particle_count = len(self.continuous_particles)
        cardinalities = self._continuous_rj_cardinalities_numpy()
        attempt = self._random_generator.random(particle_count) < float(
            self.config.structural_rj_move_probability
        )
        accepted_births = 0
        accepted_deaths = 0
        attempted_births = 0
        attempted_deaths = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            if group_indices.size == 0:
                continue
            birth_probability, _ = move_probabilities.probabilities(int(cardinality))
            birth_move = self._random_generator.random(group_indices.size) < float(
                birth_probability
            )
            for is_birth in (True, False):
                selected_rows = np.flatnonzero(birth_move == is_birth)
                if selected_rows.size == 0:
                    continue
                selected_indices = group_indices[selected_rows]
                (
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_arrays(
                    selected_indices,
                    int(cardinality),
                )
                base_ll = self._continuous_rj_current_log_likelihood(
                    data,
                    positions,
                    strengths,
                    chart_ids=chart_ids,
                    particle_indices=selected_indices,
                    target_beta=target_beta,
                )
                if is_birth:
                    attempted_births += int(selected_indices.size)
                    self._continuous_rj_transition_mass(
                        "birth_attempted",
                        selected_indices,
                    )
                    new_chart_ids, new_uv, new_positions = atlas.sample(
                        selected_indices.size,
                        rng=self._random_generator,
                        chart_probabilities=(position_proposal.chart_probabilities),
                    )
                    new_strengths = np.asarray(
                        strength_proposal.sample(
                            new_chart_ids,
                            rng=self._random_generator,
                        ),
                        dtype=np.float64,
                    )
                    proposed_chart_ids = np.concatenate(
                        (chart_ids, new_chart_ids[:, None]),
                        axis=1,
                    )
                    proposed_uv = np.concatenate(
                        (surface_uv, new_uv[:, None, :]),
                        axis=1,
                    )
                    proposed_positions = np.concatenate(
                        (positions, new_positions[:, None, :]),
                        axis=1,
                    )
                    proposed_strengths = np.concatenate(
                        (strengths, new_strengths[:, None]),
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
                    proposed_ll = self._continuous_rj_group_log_likelihood(
                        data,
                        proposed_positions,
                        proposed_strengths,
                        chart_ids=proposed_chart_ids,
                        particle_indices=selected_indices,
                        target_beta=target_beta,
                    )
                    log_position_density = atlas.log_chart_probabilities[new_chart_ids]
                    log_position_proposal = position_proposal.log_density(new_chart_ids)
                    log_strength_prior_density = np.asarray(
                        self._strength_prior.log_prob(new_strengths),
                        dtype=np.float64,
                    )
                    log_strength_proposal_density = strength_proposal.log_density(
                        new_chart_ids,
                        new_strengths,
                    )
                    log_target_ratio = _extended_log_target_ratio(
                        proposed_ll,
                        base_ll,
                    )
                    log_ratio = continuous_birth_log_acceptance_ratio(
                        current_cardinality=int(cardinality),
                        log_likelihood_ratio=log_target_ratio,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=move_probabilities,
                        log_position_prior_density=log_position_density,
                        log_strength_prior_density=(log_strength_prior_density),
                        log_forward_position_proposal=(log_position_proposal),
                        log_forward_strength_proposal=(log_strength_proposal_density),
                        log_abs_jacobian=0.0,
                    )
                    accepted = self._continuous_rj_mh_acceptance_mask(log_ratio)
                    birth_prior_ratio = (
                        float(cardinality_prior.log_prob(int(cardinality) + 1))
                        - float(cardinality_prior.log_prob(int(cardinality)))
                        + math.log(float(int(cardinality) + 1))
                        + log_position_density
                        + log_strength_prior_density
                    )
                    birth_proposal_ratio = (
                        float(
                            move_probabilities.log_probability(
                                "death",
                                int(cardinality) + 1,
                            )
                        )
                        - math.log(float(int(cardinality) + 1))
                        - float(
                            move_probabilities.log_probability(
                                "birth",
                                int(cardinality),
                            )
                        )
                        - log_position_proposal
                        - log_strength_proposal_density
                    )
                    birth_support = np.isfinite(birth_prior_ratio) & np.isfinite(
                        birth_proposal_ratio
                    )
                    birth_geometry_support = np.isfinite(
                        log_position_density
                    ) & np.isfinite(log_position_proposal)
                    birth_strength_support = np.isfinite(
                        log_strength_prior_density
                    ) & np.isfinite(log_strength_proposal_density)
                    self._record_structural_mh_components(
                        "birth",
                        particle_indices=selected_indices,
                        delta_log_likelihood=log_target_ratio,
                        delta_log_prior=birth_prior_ratio,
                        log_reverse_minus_forward=birth_proposal_ratio,
                        log_jacobian=np.zeros(
                            selected_indices.size,
                            dtype=np.float64,
                        ),
                        support_feasible=birth_support,
                        accepted=accepted,
                        current_cardinality=int(cardinality),
                        proposed_cardinality=int(cardinality) + 1,
                        geometry_support_feasible=birth_geometry_support,
                        strength_support_feasible=birth_strength_support,
                        log_acceptance_ratio=log_ratio,
                    )
                    accepted_births += self._commit_continuous_rj_states(
                        selected_indices,
                        accepted,
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    self._update_continuous_rj_current_log_likelihood(
                        selected_indices,
                        accepted,
                        proposed_ll,
                    )
                    self._continuous_rj_transition_mass(
                        "birth_accepted",
                        selected_indices,
                        accepted,
                    )
                    for row in np.flatnonzero(accepted).tolist():
                        state = self.continuous_particles[
                            int(selected_indices[row])
                        ].state
                        matches = (
                            np.asarray(state.surface_chart_ids)
                            == int(new_chart_ids[row])
                        ) & np.all(
                            np.asarray(state.surface_uv) == new_uv[row],
                            axis=1,
                        )
                        source_column = int(np.flatnonzero(matches)[0])
                        self._record_source_event(
                            "source_birth_accepted",
                            state,
                            source_column,
                            reason="continuous_rj_mh_birth",
                            extra={
                                "delta_ll": float(log_target_ratio[row]),
                                "log_acceptance_ratio": float(log_ratio[row]),
                                "surface_chart_id": int(new_chart_ids[row]),
                                "surface_uv": new_uv[row].tolist(),
                            },
                        )
                    continue

                attempted_deaths += int(selected_indices.size)
                self._continuous_rj_transition_mass(
                    "death_attempted",
                    selected_indices,
                )
                death_columns = self._random_generator.integers(
                    0,
                    int(cardinality),
                    size=selected_indices.size,
                    dtype=np.int64,
                )
                rows = np.arange(selected_indices.size, dtype=np.int64)
                removed_chart_ids = chart_ids[rows, death_columns]
                removed_uv = surface_uv[rows, death_columns]
                removed_strengths = strengths[rows, death_columns]
                keep = np.arange(int(cardinality))[None, :] != death_columns[:, None]
                proposed_chart_ids = chart_ids[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                )
                proposed_uv = surface_uv[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                    2,
                )
                proposed_positions = positions[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                    3,
                )
                proposed_strengths = strengths[keep].reshape(
                    selected_indices.size,
                    int(cardinality) - 1,
                )
                proposed_ll = self._continuous_rj_group_log_likelihood(
                    data,
                    proposed_positions,
                    proposed_strengths,
                    chart_ids=proposed_chart_ids,
                    particle_indices=selected_indices,
                    target_beta=target_beta,
                )
                log_position_density = atlas.log_chart_probabilities[removed_chart_ids]
                log_reverse_position_proposal = position_proposal.log_density(
                    removed_chart_ids
                )
                log_strength_prior_density = np.asarray(
                    self._strength_prior.log_prob(removed_strengths),
                    dtype=np.float64,
                )
                log_reverse_strength_proposal = strength_proposal.log_density(
                    removed_chart_ids,
                    removed_strengths,
                )
                log_target_ratio = _extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                )
                log_ratio = continuous_death_log_acceptance_ratio(
                    current_cardinality=int(cardinality),
                    log_likelihood_ratio=log_target_ratio,
                    cardinality_prior=cardinality_prior,
                    move_probabilities=move_probabilities,
                    log_removed_position_prior_density=log_position_density,
                    log_removed_strength_prior_density=(log_strength_prior_density),
                    log_reverse_position_proposal=(log_reverse_position_proposal),
                    log_reverse_strength_proposal=(log_reverse_strength_proposal),
                    log_abs_reverse_jacobian=0.0,
                )
                accepted = self._continuous_rj_mh_acceptance_mask(log_ratio)
                death_prior_ratio = (
                    float(cardinality_prior.log_prob(int(cardinality) - 1))
                    - float(cardinality_prior.log_prob(int(cardinality)))
                    - math.log(float(cardinality))
                    - log_position_density
                    - log_strength_prior_density
                )
                death_proposal_ratio = (
                    float(
                        move_probabilities.log_probability(
                            "birth",
                            int(cardinality) - 1,
                        )
                    )
                    + math.log(float(cardinality))
                    + log_reverse_position_proposal
                    + log_reverse_strength_proposal
                    - float(
                        move_probabilities.log_probability(
                            "death",
                            int(cardinality),
                        )
                    )
                )
                death_support = np.isfinite(death_prior_ratio) & np.isfinite(
                    death_proposal_ratio
                )
                death_geometry_support = np.isfinite(
                    log_position_density
                ) & np.isfinite(log_reverse_position_proposal)
                death_strength_support = np.isfinite(
                    log_strength_prior_density
                ) & np.isfinite(log_reverse_strength_proposal)
                self._record_structural_mh_components(
                    "death",
                    particle_indices=selected_indices,
                    delta_log_likelihood=log_target_ratio,
                    delta_log_prior=death_prior_ratio,
                    log_reverse_minus_forward=death_proposal_ratio,
                    log_jacobian=np.zeros(
                        selected_indices.size,
                        dtype=np.float64,
                    ),
                    support_feasible=death_support,
                    accepted=accepted,
                    current_cardinality=int(cardinality),
                    proposed_cardinality=int(cardinality) - 1,
                    geometry_support_feasible=death_geometry_support,
                    strength_support_feasible=death_strength_support,
                    log_acceptance_ratio=log_ratio,
                )
                for row in np.flatnonzero(accepted).tolist():
                    old_state = self.continuous_particles[
                        int(selected_indices[row])
                    ].state
                    self._record_source_event(
                        "source_removed",
                        old_state,
                        int(death_columns[row]),
                        reason="continuous_rj_mh_death",
                        extra={
                            "delta_ll": float(log_target_ratio[row]),
                            "log_acceptance_ratio": float(log_ratio[row]),
                            "surface_chart_id": int(removed_chart_ids[row]),
                            "surface_uv": removed_uv[row].tolist(),
                        },
                    )
                accepted_deaths += self._commit_continuous_rj_states(
                    selected_indices,
                    accepted,
                    proposed_chart_ids,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                self._update_continuous_rj_current_log_likelihood(
                    selected_indices,
                    accepted,
                    proposed_ll,
                )
                self._continuous_rj_transition_mass(
                    "death_accepted",
                    selected_indices,
                    accepted,
                )
        self._structural_rj_move_counts.update(
            {
                "birth_attempted": int(attempted_births),
                "birth_accepted": int(accepted_births),
                "death_attempted": int(attempted_deaths),
                "death_accepted": int(accepted_deaths),
            }
        )
        return accepted_births, accepted_deaths

    def _apply_continuous_rj_global_position_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply an exact joint global position-and-strength independence move.

        Full-spectrum location and activity are strongly correlated.  Drawing
        both from the same sweep-frozen, state-independent proposal lets the
        kernel cross that correlation without weakening the target: the exact
        reverse position and chart-conditional strength densities remain in
        the MH ratio.
        """
        if self._continuous_rj_torch_enabled():
            return self._apply_continuous_rj_global_position_moves_torch(
                data,
                target_beta=target_beta,
            )
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        particle_count = len(self.continuous_particles)
        cardinalities = self._continuous_rj_cardinalities_numpy()
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_position_move_probability)
        ) & (cardinalities > 0)
        attempted_indices = np.flatnonzero(attempt).astype(
            np.int64,
            copy=False,
        )
        self._continuous_rj_transition_mass(
            "global_position_attempted",
            attempted_indices,
        )
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
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
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            old_chart_ids = chart_ids[rows, source_columns].copy()
            old_strengths = strengths[rows, source_columns].copy()
            new_chart_ids, new_uv, new_positions = atlas.sample(
                particle_indices.size,
                rng=self._random_generator,
                chart_probabilities=position_proposal.chart_probabilities,
            )
            new_strengths = strength_proposal.sample(
                new_chart_ids,
                rng=self._random_generator,
            )
            proposed_chart_ids = chart_ids.copy()
            proposed_uv = surface_uv.copy()
            proposed_positions = positions.copy()
            proposed_strengths = strengths.copy()
            proposed_chart_ids[rows, source_columns] = new_chart_ids
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            proposed_strengths[rows, source_columns] = new_strengths
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
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions,
                proposed_strengths,
                chart_ids=proposed_chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            old_log_density = atlas.log_chart_probabilities[old_chart_ids]
            new_log_density = atlas.log_chart_probabilities[new_chart_ids]
            old_log_proposal = position_proposal.log_density(old_chart_ids)
            new_log_proposal = position_proposal.log_density(new_chart_ids)
            old_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(old_strengths),
                dtype=np.float64,
            )
            new_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(new_strengths),
                dtype=np.float64,
            )
            old_strength_log_proposal = strength_proposal.log_density(
                old_chart_ids,
                old_strengths,
            )
            new_strength_log_proposal = strength_proposal.log_density(
                new_chart_ids,
                new_strengths,
            )
            log_ratio = continuous_joint_position_strength_log_acceptance_ratio(
                log_likelihood_ratio=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                ),
                log_old_position_prior_density=old_log_density,
                log_new_position_prior_density=new_log_density,
                log_old_strength_prior_density=old_strength_log_prior,
                log_new_strength_prior_density=new_strength_log_prior,
                log_reverse_position_proposal_density=old_log_proposal,
                log_forward_position_proposal_density=new_log_proposal,
                log_reverse_strength_proposal_density=(old_strength_log_proposal),
                log_forward_strength_proposal_density=(new_strength_log_proposal),
                log_abs_jacobian=0.0,
            )
            accepted = self._continuous_rj_mh_acceptance_mask(log_ratio)
            self._continuous_rj_transition_mass(
                "global_position_accepted",
                particle_indices,
                accepted,
            )
            accepted_count += self._commit_continuous_rj_states(
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
        self._structural_rj_move_counts.update(
            {
                "global_position_attempted": attempted_count,
                "global_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_local_position_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply one exact coupled multi-scale position-strength kernel.

        A uniformly selected fixed scale defines an individually reversible MH
        kernel, so their mixture is also invariant.  Position proposals remain
        batched by scale (a fixed, tiny configuration dimension).  Strength is
        mapped deterministically to preserve the source's all-history
        integrated response, with the exact map Jacobian in the MH ratio.
        """
        if self._continuous_rj_torch_enabled():
            return self._apply_continuous_rj_local_position_moves_torch(
                data,
                target_beta=target_beta,
            )
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = self._continuous_rj_cardinalities_numpy()
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_local_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        movable_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
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
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            selected_chart_ids = chart_ids[rows, source_columns]
            old_uv = surface_uv[rows, source_columns]
            scales = np.asarray(
                self.config.structural_rj_local_position_scales_m,
                dtype=np.float64,
            )
            scale_indices = self._random_generator.integers(
                0,
                scales.size,
                size=particle_indices.size,
                dtype=np.int64,
            )
            new_chart_ids = selected_chart_ids.copy()
            new_uv = old_uv.copy()
            log_reverse_over_forward = np.zeros(
                particle_indices.size,
                dtype=np.float64,
            )
            for scale_index, sigma_m in enumerate(scales.tolist()):
                scale_rows = np.flatnonzero(scale_indices == scale_index)
                if scale_rows.size == 0:
                    continue
                proposed = atlas.tangent_geodesic_portal_proposal(
                    selected_chart_ids[scale_rows],
                    old_uv[scale_rows],
                    sigma_m=float(sigma_m),
                    rng=self._random_generator,
                )
                new_chart_ids[scale_rows] = proposed[0]
                new_uv[scale_rows] = proposed[1]
                log_reverse_over_forward[scale_rows] = proposed[2]
            new_positions = atlas.positions_xyz(new_chart_ids, new_uv)
            old_strengths = strengths[rows, source_columns].copy()
            combined_response = self._continuous_rj_integrated_unit_response(
                data,
                np.concatenate(
                    (positions[rows, source_columns], new_positions),
                    axis=0,
                ),
                np.concatenate((selected_chart_ids, new_chart_ids), axis=0),
            )
            old_response, new_response = np.split(combined_response, 2)
            positive_response = (old_response > 0.0) & (new_response > 0.0)
            strength_scale = np.ones(particle_indices.size, dtype=np.float64)
            strength_scale[positive_response] = (
                old_response[positive_response] / new_response[positive_response]
            )
            new_strengths = old_strengths * strength_scale
            proposed_chart_ids = chart_ids.copy()
            proposed_uv = surface_uv.copy()
            proposed_positions = positions.copy()
            proposed_strengths = strengths.copy()
            proposed_chart_ids[rows, source_columns] = new_chart_ids
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            proposed_strengths[rows, source_columns] = new_strengths
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
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions,
                proposed_strengths,
                chart_ids=proposed_chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            old_chart_log_density = atlas.log_chart_probabilities[selected_chart_ids]
            new_chart_log_density = atlas.log_chart_probabilities[new_chart_ids]
            zeros = np.zeros(particle_indices.size, dtype=np.float64)
            old_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(old_strengths),
                dtype=np.float64,
            )
            new_strength_log_prior = np.asarray(
                self._strength_prior.log_prob(new_strengths),
                dtype=np.float64,
            )
            log_ratio = continuous_joint_position_strength_log_acceptance_ratio(
                log_likelihood_ratio=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                ),
                log_old_position_prior_density=old_chart_log_density,
                log_new_position_prior_density=new_chart_log_density,
                log_old_strength_prior_density=old_strength_log_prior,
                log_new_strength_prior_density=new_strength_log_prior,
                log_reverse_position_proposal_density=(
                    log_reverse_over_forward
                ),
                log_forward_position_proposal_density=zeros,
                log_reverse_strength_proposal_density=zeros,
                log_forward_strength_proposal_density=zeros,
                log_abs_jacobian=np.log(strength_scale),
            )
            moved = (new_chart_ids != selected_chart_ids) | np.any(
                new_uv != old_uv,
                axis=1,
            )
            movable_count += int(np.count_nonzero(moved))
            support = (
                moved
                & positive_response
                & np.asarray(
                    self._strength_prior.in_support(new_strengths),
                    dtype=np.bool_,
                )
            )
            accepted = self._continuous_rj_mh_acceptance_mask(
                log_ratio,
                support=support,
            )
            self._record_structural_mh_components(
                "local_position_strength",
                particle_indices=particle_indices,
                delta_log_likelihood=_extended_log_target_ratio(
                    proposed_ll,
                    base_ll,
                ),
                delta_log_prior=(
                    new_chart_log_density
                    - old_chart_log_density
                    + new_strength_log_prior
                    - old_strength_log_prior
                ),
                log_reverse_minus_forward=log_reverse_over_forward,
                log_jacobian=np.log(strength_scale),
                support_feasible=support,
                accepted=accepted,
                current_cardinality=int(cardinality),
                proposed_cardinality=int(cardinality),
                geometry_support_feasible=moved & positive_response,
                strength_support_feasible=np.asarray(
                    self._strength_prior.in_support(new_strengths),
                    dtype=np.bool_,
                ),
                log_acceptance_ratio=log_ratio,
            )
            accepted_count += self._commit_continuous_rj_states(
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
        self._structural_rj_move_counts.update(
            {
                "local_position_attempted": attempted_count,
                "local_position_movable": movable_count,
                "local_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_strength_moves(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply exact prior-independence strength proposals in one batch per K."""
        if self._continuous_rj_torch_enabled():
            return self._apply_continuous_rj_strength_moves_torch(
                data,
                target_beta=target_beta,
            )
        particle_count = len(self.continuous_particles)
        cardinalities = self._continuous_rj_cardinalities_numpy()
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_strength_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
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
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            proposed_strengths = strengths.copy()
            proposed_strengths[
                np.arange(particle_indices.size),
                source_columns,
            ] = np.asarray(
                self._strength_prior.sample(
                    particle_indices.size,
                    rng=self._random_generator,
                ),
                dtype=np.float64,
            )
            proposed_ll = self._continuous_rj_group_log_likelihood(
                data,
                positions,
                proposed_strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
            log_target_ratio = _extended_log_target_ratio(
                proposed_ll,
                base_ll,
            )
            accepted = self._continuous_rj_mh_acceptance_mask(log_target_ratio)
            accepted_count += self._commit_continuous_rj_states(
                particle_indices,
                accepted,
                chart_ids,
                surface_uv,
                positions,
                proposed_strengths,
            )
            self._update_continuous_rj_current_log_likelihood(
                particle_indices,
                accepted,
                proposed_ll,
            )
        self._structural_rj_move_counts.update(
            {
                "strength_attempted": attempted_count,
                "strength_accepted": accepted_count,
            }
        )
        return accepted_count


__all__ = ["StructuralRJBasicMoveMixin"]
