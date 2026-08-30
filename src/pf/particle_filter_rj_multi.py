"""Batched multi-component proposal algorithms for exact-RJ PF."""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import NDArray

from pf.particle_filter_math import (
    extended_log_target_ratio as _extended_log_target_ratio,
    ordered_source_pair_columns as _ordered_source_pair_columns,
)
from pf.particle_types import StructuralGeometryBatch
from pf.structural_rj import (
    bounded_simplex_probability,
    independence_refresh_log_acceptance_ratio,
)


class StructuralRJMultiComponentMixin:
    """Provide batched multi-source exact-RJ proposal and move algorithms."""

    def _continuous_rj_ordered_pair_probabilities(
        self,
        data: StructuralGeometryBatch,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        """Return exact response-aware weak-donor merge probabilities.

        Intrinsic distance and normalized all-history line response select a
        physically compatible receiver, while a prior-scale decay favors a
        donor close to the active-strength floor.  A positive uniform mixture
        gives every ordered pair full support, and callers include this exact
        state-dependent probability in both directions of the RJ ratio.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        charts = np.asarray(chart_ids, dtype=np.int64)
        uv = np.asarray(surface_uv, dtype=np.float64)
        points = np.asarray(positions, dtype=np.float64)
        values = np.asarray(strengths, dtype=np.float64)
        if (
            charts.ndim != 2
            or charts.shape[1] < 2
            or uv.shape != charts.shape + (2,)
            or points.shape != charts.shape + (3,)
            or values.shape != charts.shape
        ):
            raise ValueError(
                "Merge-pair probabilities require aligned state arrays."
            )
        donor_columns, receiver_columns = _ordered_source_pair_columns(
            int(charts.shape[1])
        )
        distances = atlas.local_surface_coordinate_path_distance_m(
            charts[:, donor_columns],
            uv[:, donor_columns, :],
            charts[:, receiver_columns],
            uv[:, receiver_columns, :],
        )
        signatures = self._continuous_rj_source_response_signatures(
            data,
            chart_ids=charts,
            positions=points,
        )
        donor_signatures = signatures[:, donor_columns, :]
        receiver_signatures = signatures[:, receiver_columns, :]
        response_cosine = np.clip(
            np.sum(donor_signatures * receiver_signatures, axis=-1),
            -1.0,
            1.0,
        )
        response_distance = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * response_cosine)
        )
        strength_scale = (
            float(self._strength_prior.gamma_scale)
            if self._strength_prior.family == "shifted_gamma"
            else float(self._strength_prior.maximum - self._strength_prior.minimum)
        )
        donor_excess = np.maximum(
            values[:, donor_columns] - float(self._strength_prior.minimum),
            0.0,
        )
        weak_donor_weight = np.exp(
            -np.minimum(donor_excess / strength_scale, 745.0)
        )
        scores = weak_donor_weight * np.exp(
            -0.5
            * np.square(
                distances / float(self.config.structural_rj_merge_distance_sigma_m)
            )
            -0.5
            * np.square(
                response_distance
                / float(self.config.structural_rj_merge_response_sigma)
            )
        )
        score_sums = np.sum(scores, axis=1, keepdims=True)
        informed = np.divide(
            scores,
            score_sums,
            out=np.full_like(scores, 1.0 / float(scores.shape[1])),
            where=score_sums > 0.0,
        )
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (
            (1.0 - uniform) * informed + uniform / float(scores.shape[1])
        )
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        expected_shape = (charts.shape[0], donor_columns.size)
        if probabilities.shape != expected_shape:
            raise RuntimeError(
                "Distance-weighted merge proposal returned an invalid shape."
            )
        return donor_columns, receiver_columns, probabilities

    def _continuous_rj_source_response_signatures(
        self,
        data: StructuralGeometryBatch,
        *,
        chart_ids: NDArray[np.int64],
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return normalized all-history line-response rows for each source."""
        charts = np.asarray(chart_ids, dtype=np.int64)
        points = np.asarray(positions, dtype=np.float64)
        if charts.ndim != 2 or points.shape != charts.shape + (3,):
            raise ValueError("Response signatures require aligned P x K states.")
        row_count, cardinality = charts.shape
        line_indices = self.continuous_kernel.positive_line_indices(self.isotope)
        branching_weights = np.asarray(
            self.continuous_kernel.line_branching_weights(
                self.isotope,
                line_indices,
            ),
            dtype=np.float64,
        )
        components = self._continuous_rj_line_transport_component_columns(
            data,
            points.reshape(row_count * cardinality, 3),
            line_indices,
            chart_ids=charts.reshape(row_count * cardinality),
        )
        physical_response = np.asarray(
            components.total_kernel,
            dtype=np.float64,
        ).reshape(
            int(data.row_count),
            row_count,
            cardinality,
            int(line_indices.size),
        )
        physical_response *= branching_weights.reshape(1, 1, 1, -1)
        signatures = np.transpose(
            physical_response,
            (1, 2, 0, 3),
        ).reshape(row_count, cardinality, -1)
        norms = np.linalg.norm(signatures, axis=-1, keepdims=True)
        return np.divide(
            signatures,
            norms,
            out=np.zeros_like(signatures),
            where=norms > 0.0,
        )

    def _continuous_rj_multi_group_probabilities(
        self,
        data: StructuralGeometryBatch,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        group_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return distance/physical-response weighted group probabilities."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        cardinality = int(chart_ids.shape[1])
        groups = np.asarray(
            tuple(itertools.combinations(range(cardinality), int(group_size))),
            dtype=np.int64,
        )
        if groups.ndim != 2 or groups.shape[0] == 0:
            raise ValueError("No multi-component group exists for this state.")
        row_count = int(chart_ids.shape[0])
        maximum_surface_distance = np.zeros(
            (row_count, groups.shape[0]),
            dtype=np.float64,
        )
        minimum_response_cosine = np.ones_like(maximum_surface_distance)
        signatures = self._continuous_rj_source_response_signatures(
            data,
            chart_ids=chart_ids,
            positions=positions,
        )
        selected_signatures = signatures[:, groups, :]
        for first, second in itertools.combinations(range(int(group_size)), 2):
            distances = atlas.local_surface_coordinate_path_distance_m(
                chart_ids[:, groups[:, first]],
                surface_uv[:, groups[:, first], :],
                chart_ids[:, groups[:, second]],
                surface_uv[:, groups[:, second], :],
            )
            maximum_surface_distance = np.maximum(
                maximum_surface_distance,
                distances,
            )
            cosine = np.sum(
                selected_signatures[:, :, first, :]
                * selected_signatures[:, :, second, :],
                axis=-1,
            )
            minimum_response_cosine = np.minimum(
                minimum_response_cosine,
                np.clip(cosine, -1.0, 1.0),
            )
        response_distance = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * minimum_response_cosine)
        )
        distance_sigma = float(self.config.structural_rj_merge_distance_sigma_m)
        response_sigma = float(self.config.structural_rj_merge_response_sigma)
        scores = np.exp(
            -0.5 * np.square(maximum_surface_distance / distance_sigma)
            - 0.5 * np.square(response_distance / response_sigma)
        )
        score_sums = np.sum(scores, axis=1, keepdims=True)
        cohesive_probabilities = np.divide(
            scores,
            score_sums,
            out=np.full_like(scores, 1.0 / float(groups.shape[0])),
            where=score_sums > 0.0,
        )
        position_proposal = self._structural_rj_position_proposal
        if position_proposal is None:
            cleanup_scores = np.ones_like(scores)
        else:
            evidence_ratio = np.exp(
                position_proposal.log_density(chart_ids)
                - atlas.log_chart_probabilities[chart_ids]
            )
            group_evidence = evidence_ratio[:, groups]
            receiver_evidence = np.max(group_evidence, axis=2)
            donor_evidence = (
                np.sum(group_evidence, axis=2) - receiver_evidence
            ) / float(int(group_size) - 1)
            cleanup_scores = receiver_evidence / np.maximum(
                donor_evidence,
                np.finfo(np.float64).tiny,
            )
        cleanup_sums = np.sum(cleanup_scores, axis=1, keepdims=True)
        cleanup_probabilities = np.divide(
            cleanup_scores,
            cleanup_sums,
            out=np.full_like(scores, 1.0 / float(groups.shape[0])),
            where=cleanup_sums > 0.0,
        )
        normalized = 0.5 * (cohesive_probabilities + cleanup_probabilities)
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (1.0 - uniform) * normalized + uniform / float(groups.shape[0])
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return groups, probabilities

    def _continuous_rj_multi_direction_support(
        self,
        cardinality: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], float, float]:
        """Return supported group sizes and normalized direction probabilities."""
        maximum_group = min(
            int(self.config.structural_rj_multi_component_max_group_size),
            int(self.config.hard_max_sources),
        )
        split_sizes = tuple(
            size
            for size in range(3, maximum_group + 1)
            if (
                int(cardinality) >= 1
                and int(cardinality) + size - 1 <= int(self.config.hard_max_sources)
            )
        )
        merge_sizes = tuple(
            size for size in range(3, maximum_group + 1) if size <= int(cardinality)
        )
        if split_sizes and merge_sizes:
            return split_sizes, merge_sizes, 0.5, 0.5
        if split_sizes:
            return split_sizes, merge_sizes, 1.0, 0.0
        if merge_sizes:
            return split_sizes, merge_sizes, 0.0, 1.0
        return split_sizes, merge_sizes, 0.0, 0.0

    def _continuous_rj_merge_anchor_probabilities(
        self,
        chart_ids: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return exact evidence-weighted anchor probabilities per group.

        A positive uniform component protects full support.  The data term is
        frozen for the whole structural sweep, so using it changes proposal
        efficiency without changing the posterior target.
        """
        atlas = self._structural_rj_surface_atlas
        position_proposal = self._structural_rj_position_proposal
        charts = np.asarray(chart_ids, dtype=np.int64)
        if atlas is None or charts.ndim != 2 or charts.shape[1] < 2:
            raise ValueError("Merge anchors require a nonempty surface group.")
        if position_proposal is None:
            evidence = np.ones(charts.shape, dtype=np.float64)
        else:
            log_evidence = (
                position_proposal.log_density(charts)
                - atlas.log_chart_probabilities[charts]
            )
            row_maximum = np.max(log_evidence, axis=1, keepdims=True)
            evidence = np.exp(np.clip(log_evidence - row_maximum, -745.0, 0.0))
        evidence /= np.sum(evidence, axis=1, keepdims=True)
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (1.0 - uniform) * evidence + uniform / float(charts.shape[1])
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities

    def _sample_bounded_simplex(
        self,
        totals: NDArray[np.float64],
        *,
        group_size: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Sample a bounded uniform simplex with vectorized exact rejection."""
        values = np.asarray(totals, dtype=np.float64).reshape(-1)
        probability = bounded_simplex_probability(
            values,
            group_size=int(group_size),
            minimum_strength=self._strength_prior.minimum,
            maximum_strength=self._strength_prior.support_maximum,
        )
        accepted = probability > 0.0
        pending = accepted.copy()
        fractions = np.zeros((values.size, int(group_size)), dtype=np.float64)
        for _ in range(16_384):
            rows = np.flatnonzero(pending)
            if rows.size == 0:
                break
            draws = self._random_generator.dirichlet(
                np.ones(int(group_size), dtype=np.float64),
                size=rows.size,
            )
            child_strengths = draws * values[rows, None]
            valid = np.all(
                child_strengths >= self._strength_prior.minimum, axis=1
            ) & np.all(
                child_strengths <= self._strength_prior.support_maximum,
                axis=1,
            )
            fractions[rows[valid]] = draws[valid]
            pending[rows[valid]] = False
        accepted &= ~pending
        return fractions, accepted

    def _continuous_rj_block_log_densities(
        self,
        chart_ids: NDArray[np.int64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return physical-prior and full-support block-proposal log densities."""
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        if atlas is None or cardinality_prior is None:
            raise RuntimeError("Continuous block-RJ priors are unavailable.")
        charts = np.asarray(chart_ids, dtype=np.int64)
        values = np.asarray(strengths, dtype=np.float64)
        if charts.ndim != 2 or values.shape != charts.shape:
            raise ValueError("Block-RJ states must share particle/source axes.")
        cardinality = int(charts.shape[1])
        row_count = int(charts.shape[0])
        log_prior = np.full(
            row_count,
            float(cardinality_prior.log_prob(cardinality))
            + math.lgamma(float(cardinality) + 1.0),
            dtype=np.float64,
        )
        log_proposal = log_prior.copy()
        if cardinality == 0:
            return log_prior, log_proposal
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        log_prior += np.sum(
            atlas.log_chart_probabilities[charts]
            + np.asarray(
                self._strength_prior.log_prob(values),
                dtype=np.float64,
            ),
            axis=1,
            dtype=np.float64,
        )
        log_proposal += np.sum(
            position_proposal.log_density(charts)
            + strength_proposal.log_density(charts, values),
            axis=1,
            dtype=np.float64,
        )
        return log_prior, log_proposal

    def _apply_continuous_rj_multi_component(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact 3--4 component split/merge proposals in one jump.

        Candidate groups mix two normalized selectors: intrinsic-distance plus
        line-resolved response cohesion for split components, and a frozen
        data-informed evidence contrast for deleting several weak ghosts into
        one supported receiver. The mixture retains an explicit uniform term.
        The full-support uniform component, state-dependent group selection,
        bounded-simplex density, position density, and strength Jacobian are
        all included in the forward/reverse ratio.
        """
        if self._continuous_rj_torch_enabled():
            return self._apply_continuous_rj_multi_component_torch(
                data,
                target_beta=target_beta,
            )
        probability = float(self.config.structural_rj_multi_component_probability)
        if probability <= 0.0:
            return 0, 0
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = self._continuous_rj_cardinalities_numpy()
        available = np.asarray(
            [
                sum(self._continuous_rj_multi_direction_support(int(value))[2:]) > 0.0
                for value in cardinalities
            ],
            dtype=bool,
        )
        attempted = (
            self._random_generator.random(particle_count) < probability
        ) & available
        accepted_splits = 0
        accepted_merges = 0
        attempted_splits = 0
        attempted_merges = 0
        global_probability = float(
            self.config.structural_rj_split_global_position_probability
        )
        for cardinality in np.unique(cardinalities[attempted]).tolist():
            particle_indices = np.flatnonzero(
                attempted & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            (
                split_sizes,
                merge_sizes,
                split_direction_probability,
                _,
            ) = self._continuous_rj_multi_direction_support(int(cardinality))
            split_rows = (
                self._random_generator.random(particle_indices.size)
                < split_direction_probability
            )
            for is_split in (True, False):
                direction_rows = np.flatnonzero(split_rows == is_split)
                if direction_rows.size == 0:
                    continue
                sizes = split_sizes if is_split else merge_sizes
                if not sizes:
                    continue
                chosen_sizes = self._random_generator.choice(
                    np.asarray(sizes, dtype=np.int64),
                    size=direction_rows.size,
                    replace=True,
                )
                for group_size in np.unique(chosen_sizes).tolist():
                    local_rows = direction_rows[chosen_sizes == int(group_size)]
                    indices = particle_indices[local_rows]
                    (
                        chart_ids,
                        surface_uv,
                        positions,
                        strengths,
                    ) = self._continuous_rj_group_arrays(
                        indices,
                        int(cardinality),
                    )
                    base_ll = self._continuous_rj_current_log_likelihood(
                        data,
                        positions,
                        strengths,
                        chart_ids=chart_ids,
                        particle_indices=indices,
                        target_beta=target_beta,
                    )
                    current_prior, _ = self._continuous_rj_block_log_densities(
                        chart_ids,
                        strengths,
                    )
                    if is_split:
                        attempted_splits += int(indices.size)
                        self._continuous_rj_transition_mass(
                            "multi_split_attempted",
                            indices,
                        )
                        accepted, proposed_ll = self._continuous_rj_multi_split(
                            data,
                            particle_indices=indices,
                            cardinality=int(cardinality),
                            group_size=int(group_size),
                            chart_ids=chart_ids,
                            surface_uv=surface_uv,
                            positions=positions,
                            strengths=strengths,
                            base_ll=base_ll,
                            current_prior=current_prior,
                            split_sizes=split_sizes,
                            split_direction_probability=(split_direction_probability),
                            target_beta=target_beta,
                            global_probability=global_probability,
                        )
                        accepted_splits += int(np.sum(accepted))
                        self._update_continuous_rj_current_log_likelihood(
                            indices,
                            accepted,
                            proposed_ll,
                        )
                        self._continuous_rj_transition_mass(
                            "multi_split_accepted",
                            indices,
                            accepted,
                        )
                    else:
                        attempted_merges += int(indices.size)
                        self._continuous_rj_transition_mass(
                            "multi_merge_attempted",
                            indices,
                        )
                        accepted, proposed_ll = self._continuous_rj_multi_merge(
                            data,
                            particle_indices=indices,
                            cardinality=int(cardinality),
                            group_size=int(group_size),
                            chart_ids=chart_ids,
                            surface_uv=surface_uv,
                            positions=positions,
                            strengths=strengths,
                            base_ll=base_ll,
                            current_prior=current_prior,
                            merge_sizes=merge_sizes,
                            target_beta=target_beta,
                            global_probability=global_probability,
                        )
                        accepted_merges += int(np.sum(accepted))
                        self._update_continuous_rj_current_log_likelihood(
                            indices,
                            accepted,
                            proposed_ll,
                        )
                        self._continuous_rj_transition_mass(
                            "multi_merge_accepted",
                            indices,
                            accepted,
                        )
        self._structural_rj_move_counts.update(
            {
                "multi_split_attempted": attempted_splits,
                "multi_split_accepted": accepted_splits,
                "multi_merge_attempted": attempted_merges,
                "multi_merge_accepted": accepted_merges,
            }
        )
        return accepted_splits, accepted_merges

    def _continuous_rj_multi_split(
        self,
        data: StructuralGeometryBatch,
        *,
        particle_indices: NDArray[np.int64],
        cardinality: int,
        group_size: int,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        base_ll: NDArray[np.float64],
        current_prior: NDArray[np.float64],
        split_sizes: tuple[int, ...],
        split_direction_probability: float,
        target_beta: float,
        global_probability: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Propose an exact split with conditional block strengths."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        row_count = int(particle_indices.size)
        rows = np.arange(row_count, dtype=np.int64)
        parent_columns = self._random_generator.integers(
            0,
            int(cardinality),
            size=row_count,
            dtype=np.int64,
        )
        parent_charts = chart_ids[rows, parent_columns]
        child_charts: list[NDArray[np.int64]] = []
        child_uv: list[NDArray[np.float64]] = []
        child_positions: list[NDArray[np.float64]] = []
        child_log_density: list[NDArray[np.float64]] = []
        for _ in range(group_size):
            sampled = atlas.sample_local_chart_mixture(
                parent_charts,
                global_component_probability=global_probability,
                rng=self._random_generator,
            )
            child_charts.append(sampled[0])
            child_uv.append(sampled[1])
            child_positions.append(sampled[2])
            child_log_density.append(sampled[3])
        keep = np.arange(int(cardinality))[None, :] != parent_columns[:, None]
        retained_count = int(cardinality) - 1
        proposed_charts = np.concatenate(
            (
                chart_ids[keep].reshape(row_count, retained_count),
                np.stack(child_charts, axis=1),
            ),
            axis=1,
        )
        proposed_uv = np.concatenate(
            (
                surface_uv[keep].reshape(row_count, retained_count, 2),
                np.stack(child_uv, axis=1),
            ),
            axis=1,
        )
        proposed_positions = np.concatenate(
            (
                positions[keep].reshape(row_count, retained_count, 3),
                np.stack(child_positions, axis=1),
            ),
            axis=1,
        )
        proposed_cardinality = int(cardinality) + int(group_size) - 1
        proposed_strength_proposal = (
            self._continuous_rj_conditional_block_strength_proposal(
                data,
                chart_ids=proposed_charts,
                positions=proposed_positions,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
        )
        proposed_strengths = proposed_strength_proposal.sample(
            rng=self._random_generator,
        )
        current_strength_proposal = (
            self._continuous_rj_conditional_block_strength_proposal(
                data,
                chart_ids=chart_ids,
                positions=positions,
                particle_indices=particle_indices,
                target_beta=target_beta,
                cache_current_state=True,
            )
        )
        current_strength_log_proposal = current_strength_proposal.log_density(strengths)
        proposed_strength_log_proposal = proposed_strength_proposal.log_density(
            proposed_strengths
        )
        reverse_groups, reverse_probabilities = (
            self._continuous_rj_multi_group_probabilities(
                data,
                proposed_charts,
                proposed_uv,
                proposed_positions,
                group_size=group_size,
            )
        )
        child_columns = np.arange(
            retained_count,
            proposed_cardinality,
            dtype=np.int64,
        )
        reverse_group_column = int(
            np.flatnonzero(np.all(reverse_groups == child_columns[None, :], axis=1))[0]
        )
        reverse_group_log_probability = np.log(
            reverse_probabilities[:, reverse_group_column]
        )
        reverse_anchor_probabilities = self._continuous_rj_merge_anchor_probabilities(
            np.stack(child_charts, axis=1)
        )
        reverse_merged_log_density = np.logaddexp.reduce(
            np.stack(
                [
                    np.log(reverse_anchor_probabilities[:, index])
                    + atlas.local_chart_mixture_log_density(
                        child_charts[index],
                        parent_charts,
                        global_component_probability=global_probability,
                    )
                    for index in range(group_size)
                ],
                axis=1,
            ),
            axis=1,
        )
        _, reverse_merge_sizes, _, reverse_merge_probability = (
            self._continuous_rj_multi_direction_support(proposed_cardinality)
        )
        log_forward = (
            math.log(split_direction_probability)
            - math.log(float(len(split_sizes)))
            - math.log(float(cardinality))
            + math.lgamma(float(group_size) + 1.0)
            + np.sum(np.stack(child_log_density, axis=1), axis=1)
            + proposed_strength_log_proposal
        )
        log_reverse = (
            math.log(reverse_merge_probability)
            - math.log(float(len(reverse_merge_sizes)))
            + reverse_group_log_probability
            + reverse_merged_log_density
            + current_strength_log_proposal
        )
        canonical = self._continuous_rj_canonicalize_rows(
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        proposed_charts, proposed_uv, proposed_positions, proposed_strengths = canonical
        proposed_ll = np.full(row_count, float("-inf"), dtype=np.float64)
        log_ratio = np.full(row_count, float("-inf"), dtype=np.float64)
        delta_prior = np.full(row_count, np.nan, dtype=np.float64)
        feasible = (
            np.isfinite(log_forward)
            & np.isfinite(log_reverse)
            & np.all(
                self._strength_prior.in_support(proposed_strengths),
                axis=1,
            )
        )
        valid_rows = np.flatnonzero(feasible)
        if valid_rows.size:
            proposed_ll[valid_rows] = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions[valid_rows],
                proposed_strengths[valid_rows],
                chart_ids=proposed_charts[valid_rows],
                particle_indices=particle_indices[valid_rows],
                target_beta=target_beta,
            )
            proposed_prior, _ = self._continuous_rj_block_log_densities(
                proposed_charts[valid_rows],
                proposed_strengths[valid_rows],
            )
            delta_prior[valid_rows] = proposed_prior - current_prior[valid_rows]
            target_ratio = (
                _extended_log_target_ratio(
                    proposed_ll[valid_rows],
                    base_ll[valid_rows],
                )
                + proposed_prior
                - current_prior[valid_rows]
            )
            log_ratio[valid_rows] = independence_refresh_log_acceptance_ratio(
                log_target_ratio=target_ratio,
                log_forward_proposal=log_forward[valid_rows],
                log_reverse_proposal=log_reverse[valid_rows],
            )
        accepted = self._continuous_rj_mh_acceptance_mask(
            log_ratio,
            support=feasible,
        )
        self._record_structural_mh_components(
            "multi_split",
            particle_indices=particle_indices,
            delta_log_likelihood=_extended_log_target_ratio(
                proposed_ll,
                base_ll,
            ),
            delta_log_prior=delta_prior,
            log_reverse_minus_forward=(log_reverse - log_forward),
            log_jacobian=np.zeros(row_count, dtype=np.float64),
            support_feasible=feasible,
            accepted=accepted,
            current_cardinality=int(cardinality),
            proposed_cardinality=int(proposed_cardinality),
            geometry_support_feasible=(
                np.isfinite(log_forward) & np.isfinite(log_reverse)
            ),
            strength_support_feasible=np.all(
                self._strength_prior.in_support(proposed_strengths),
                axis=1,
            ),
            log_acceptance_ratio=log_ratio,
        )
        self._commit_continuous_rj_states(
            particle_indices,
            accepted,
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        return accepted, proposed_ll

    def _continuous_rj_multi_merge(
        self,
        data: StructuralGeometryBatch,
        *,
        particle_indices: NDArray[np.int64],
        cardinality: int,
        group_size: int,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        base_ll: NDArray[np.float64],
        current_prior: NDArray[np.float64],
        merge_sizes: tuple[int, ...],
        target_beta: float,
        global_probability: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Propose an exact multi-merge with conditional block strengths."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        row_count = int(particle_indices.size)
        rows = np.arange(row_count, dtype=np.int64)
        groups, probabilities = self._continuous_rj_multi_group_probabilities(
            data,
            chart_ids,
            surface_uv,
            positions,
            group_size=group_size,
        )
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        group_columns = np.sum(
            self._random_generator.random(row_count)[:, None] > cumulative,
            axis=1,
            dtype=np.int64,
        )
        selected_columns = groups[group_columns]
        selected_charts = chart_ids[rows[:, None], selected_columns]
        anchor_probabilities = self._continuous_rj_merge_anchor_probabilities(
            selected_charts
        )
        anchor_cumulative = np.cumsum(anchor_probabilities, axis=1)
        anchor_cumulative[:, -1] = 1.0
        anchor_offsets = np.sum(
            self._random_generator.random(row_count)[:, None] > anchor_cumulative,
            axis=1,
            dtype=np.int64,
        )
        anchor_charts = selected_charts[rows, anchor_offsets]
        (
            merged_charts,
            merged_uv,
            merged_positions,
            _,
        ) = atlas.sample_local_chart_mixture(
            anchor_charts,
            global_component_probability=global_probability,
            rng=self._random_generator,
        )
        forward_merged_log_density = np.logaddexp.reduce(
            np.stack(
                [
                    np.log(anchor_probabilities[:, index])
                    + atlas.local_chart_mixture_log_density(
                        selected_charts[:, index],
                        merged_charts,
                        global_component_probability=global_probability,
                    )
                    for index in range(group_size)
                ],
                axis=1,
            ),
            axis=1,
        )
        keep = np.ones((row_count, cardinality), dtype=bool)
        keep[rows[:, None], selected_columns] = False
        retained_count = cardinality - group_size
        proposed_charts = np.concatenate(
            (
                chart_ids[keep].reshape(row_count, retained_count),
                merged_charts[:, None],
            ),
            axis=1,
        )
        proposed_uv = np.concatenate(
            (
                surface_uv[keep].reshape(row_count, retained_count, 2),
                merged_uv[:, None, :],
            ),
            axis=1,
        )
        proposed_positions = np.concatenate(
            (
                positions[keep].reshape(row_count, retained_count, 3),
                merged_positions[:, None, :],
            ),
            axis=1,
        )
        proposed_cardinality = cardinality - group_size + 1
        proposed_strength_proposal = (
            self._continuous_rj_conditional_block_strength_proposal(
                data,
                chart_ids=proposed_charts,
                positions=proposed_positions,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
        )
        proposed_strengths = proposed_strength_proposal.sample(
            rng=self._random_generator,
        )
        current_strength_proposal = (
            self._continuous_rj_conditional_block_strength_proposal(
                data,
                chart_ids=chart_ids,
                positions=positions,
                particle_indices=particle_indices,
                target_beta=target_beta,
                cache_current_state=True,
            )
        )
        current_strength_log_proposal = current_strength_proposal.log_density(strengths)
        proposed_strength_log_proposal = proposed_strength_proposal.log_density(
            proposed_strengths
        )
        split_sizes, _, reverse_split_probability, _ = (
            self._continuous_rj_multi_direction_support(proposed_cardinality)
        )
        reverse_child_position_log_density = np.sum(
            np.stack(
                [
                    atlas.local_chart_mixture_log_density(
                        merged_charts,
                        selected_charts[:, index],
                        global_component_probability=global_probability,
                    )
                    for index in range(group_size)
                ],
                axis=1,
            ),
            axis=1,
        )
        _, _, _, merge_direction_probability = (
            self._continuous_rj_multi_direction_support(cardinality)
        )
        log_forward = (
            math.log(merge_direction_probability)
            - math.log(float(len(merge_sizes)))
            + np.log(probabilities[rows, group_columns])
            + forward_merged_log_density
            + proposed_strength_log_proposal
        )
        log_reverse = (
            math.log(reverse_split_probability)
            - math.log(float(len(split_sizes)))
            - math.log(float(proposed_cardinality))
            + math.lgamma(float(group_size) + 1.0)
            + reverse_child_position_log_density
            + current_strength_log_proposal
        )
        feasible = (
            np.isfinite(log_forward)
            & np.isfinite(log_reverse)
            & np.all(
                self._strength_prior.in_support(proposed_strengths),
                axis=1,
            )
        )
        canonical = self._continuous_rj_canonicalize_rows(
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        proposed_charts, proposed_uv, proposed_positions, proposed_strengths = canonical
        proposed_ll = np.full(row_count, float("-inf"), dtype=np.float64)
        log_ratio = np.full(row_count, float("-inf"), dtype=np.float64)
        delta_prior = np.full(row_count, np.nan, dtype=np.float64)
        valid_rows = np.flatnonzero(feasible)
        if valid_rows.size:
            proposed_ll[valid_rows] = self._continuous_rj_group_log_likelihood(
                data,
                proposed_positions[valid_rows],
                proposed_strengths[valid_rows],
                chart_ids=proposed_charts[valid_rows],
                particle_indices=particle_indices[valid_rows],
                target_beta=target_beta,
            )
            proposed_prior, _ = self._continuous_rj_block_log_densities(
                proposed_charts[valid_rows],
                proposed_strengths[valid_rows],
            )
            delta_prior[valid_rows] = proposed_prior - current_prior[valid_rows]
            target_ratio = (
                _extended_log_target_ratio(
                    proposed_ll[valid_rows],
                    base_ll[valid_rows],
                )
                + proposed_prior
                - current_prior[valid_rows]
            )
            log_ratio[valid_rows] = independence_refresh_log_acceptance_ratio(
                log_target_ratio=target_ratio,
                log_forward_proposal=log_forward[valid_rows],
                log_reverse_proposal=log_reverse[valid_rows],
            )
        accepted = self._continuous_rj_mh_acceptance_mask(
            log_ratio,
            support=feasible,
        )
        self._record_structural_mh_components(
            "multi_merge",
            particle_indices=particle_indices,
            delta_log_likelihood=_extended_log_target_ratio(
                proposed_ll,
                base_ll,
            ),
            delta_log_prior=delta_prior,
            log_reverse_minus_forward=(log_reverse - log_forward),
            log_jacobian=np.zeros(row_count, dtype=np.float64),
            support_feasible=feasible,
            accepted=accepted,
            current_cardinality=int(cardinality),
            proposed_cardinality=int(proposed_cardinality),
            geometry_support_feasible=(
                np.isfinite(log_forward) & np.isfinite(log_reverse)
            ),
            strength_support_feasible=np.all(
                self._strength_prior.in_support(proposed_strengths),
                axis=1,
            ),
            log_acceptance_ratio=log_ratio,
        )
        self._commit_continuous_rj_states(
            particle_indices,
            accepted,
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        return accepted, proposed_ll


__all__ = ["StructuralRJMultiComponentMixin"]
