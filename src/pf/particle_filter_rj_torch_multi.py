"""Device-resident multi-component exact-RJ kernels for production CUDA PF."""

from __future__ import annotations

import itertools
import math

import torch

from pf.particle_types import StructuralGeometryBatch


class StructuralRJTorchMultiComponentMixin:
    """Provide Torch-native exact 3--4 component split and merge moves."""

    def _continuous_rj_multi_group_probabilities_torch(
        self,
        data: StructuralGeometryBatch,
        chart_ids: torch.Tensor,
        surface_uv: torch.Tensor,
        positions: torch.Tensor,
        *,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        """Return exact response-aware group probabilities on CUDA."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        cardinality = int(chart_ids.shape[1])
        groups = torch.tensor(
            tuple(itertools.combinations(range(cardinality), int(group_size))),
            device=chart_ids.device,
            dtype=torch.long,
        )
        if groups.ndim != 2 or int(groups.shape[0]) == 0:
            raise ValueError("No CUDA multi-component group exists.")
        row_count = int(chart_ids.shape[0])
        group_count = int(groups.shape[0])
        maximum_distance = torch.zeros(
            (row_count, group_count),
            device=positions.device,
            dtype=positions.dtype,
        )
        minimum_cosine = torch.ones_like(maximum_distance)
        signatures = self._continuous_rj_source_response_signatures_torch(
            data,
            chart_ids=chart_ids,
            positions=positions,
        )
        selected_signatures = signatures[:, groups, :]
        # Group size is bounded by four, so this fixed six-pair control loop is
        # smaller than launching a padded all-pairs kernel.
        for first, second in itertools.combinations(range(int(group_size)), 2):
            distance = atlas.local_surface_coordinate_path_distance_m_torch(
                chart_ids[:, groups[:, first]],
                surface_uv[:, groups[:, first], :],
                chart_ids[:, groups[:, second]],
                surface_uv[:, groups[:, second], :],
            )
            maximum_distance = torch.maximum(maximum_distance, distance)
            cosine = torch.sum(
                selected_signatures[:, :, first, :]
                * selected_signatures[:, :, second, :],
                dim=-1,
            )
            minimum_cosine = torch.minimum(
                minimum_cosine,
                torch.clamp(cosine, min=-1.0, max=1.0),
            )
        response_distance = torch.sqrt(
            torch.clamp(2.0 - 2.0 * minimum_cosine, min=0.0)
        )
        scores = torch.exp(
            -0.5
            * (
                maximum_distance
                / float(self.config.structural_rj_merge_distance_sigma_m)
            ).square()
            -0.5
            * (
                response_distance
                / float(self.config.structural_rj_merge_response_sigma)
            ).square()
        )
        score_sums = torch.sum(scores, dim=1, keepdim=True)
        cohesive = torch.where(
            score_sums > 0.0,
            scores
            / torch.where(
                score_sums > 0.0,
                score_sums,
                torch.ones_like(score_sums),
            ),
            torch.full_like(scores, 1.0 / float(group_count)),
        )
        if self._structural_rj_position_proposal is None:
            cleanup_scores = torch.ones_like(scores)
        else:
            constants = self._continuous_rj_atlas_tensors()
            log_evidence = (
                self._continuous_rj_position_proposal_log_density_torch(
                    chart_ids
                )
                - constants["log_chart_probabilities"][chart_ids]
            )
            evidence = torch.exp(log_evidence)
            group_evidence = evidence[:, groups]
            receiver = torch.amax(group_evidence, dim=2)
            donor = (
                torch.sum(group_evidence, dim=2) - receiver
            ) / float(int(group_size) - 1)
            cleanup_scores = receiver / torch.clamp(
                donor,
                min=torch.finfo(donor.dtype).tiny,
            )
        cleanup_sums = torch.sum(cleanup_scores, dim=1, keepdim=True)
        cleanup = torch.where(
            cleanup_sums > 0.0,
            cleanup_scores
            / torch.where(
                cleanup_sums > 0.0,
                cleanup_sums,
                torch.ones_like(cleanup_sums),
            ),
            torch.full_like(cleanup_scores, 1.0 / float(group_count)),
        )
        normalized = 0.5 * (cohesive + cleanup)
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (
            (1.0 - uniform) * normalized + uniform / float(group_count)
        )
        probabilities = probabilities / torch.sum(
            probabilities,
            dim=1,
            keepdim=True,
        )
        return groups, probabilities

    def _continuous_rj_merge_anchor_probabilities_torch(
        self,
        chart_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact evidence-weighted anchor probabilities on CUDA."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None or chart_ids.ndim != 2 or int(chart_ids.shape[1]) < 2:
            raise ValueError("CUDA merge anchors require a nonempty group.")
        if self._structural_rj_position_proposal is None:
            evidence = torch.ones(
                chart_ids.shape,
                device=chart_ids.device,
                dtype=self._structural_rj_device_state["strengths"].dtype,
            )
        else:
            constants = self._continuous_rj_atlas_tensors()
            log_evidence = (
                self._continuous_rj_position_proposal_log_density_torch(
                    chart_ids
                )
                - constants["log_chart_probabilities"][chart_ids]
            )
            log_evidence = log_evidence - torch.amax(
                log_evidence,
                dim=1,
                keepdim=True,
            )
            evidence = torch.exp(torch.clamp(log_evidence, min=-745.0, max=0.0))
        evidence = evidence / torch.sum(evidence, dim=1, keepdim=True)
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (
            (1.0 - uniform) * evidence
            + uniform / float(chart_ids.shape[1])
        )
        return probabilities / torch.sum(
            probabilities,
            dim=1,
            keepdim=True,
        )

    def _continuous_rj_sample_rows_torch(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Sample one categorical column per row without leaving CUDA."""
        generator = self._continuous_rj_torch_generator_required()
        cumulative = torch.cumsum(probabilities, dim=1)
        cumulative[:, -1] = 1.0
        draws = torch.rand(
            (int(probabilities.shape[0]), 1),
            device=probabilities.device,
            dtype=probabilities.dtype,
            generator=generator,
        )
        return torch.sum(draws > cumulative, dim=1, dtype=torch.long)

    def _continuous_rj_multi_split_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        particle_indices: torch.Tensor,
        cardinality: int,
        group_size: int,
        chart_ids: torch.Tensor,
        surface_uv: torch.Tensor,
        positions: torch.Tensor,
        strengths: torch.Tensor,
        base_ll: torch.Tensor,
        current_prior: torch.Tensor,
        split_sizes: tuple[int, ...],
        split_direction_probability: float,
        target_beta: float,
        global_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose one exact multi-split entirely on CUDA."""
        atlas = self._structural_rj_surface_atlas
        state = self._structural_rj_device_state
        if atlas is None or state is None:
            raise RuntimeError("CUDA multi-split surface state is unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        row_count = int(particle_indices.numel())
        rows = torch.arange(row_count, device=chart_ids.device)
        parent_columns = torch.randint(
            int(cardinality),
            (row_count,),
            device=chart_ids.device,
            generator=generator,
        )
        parent_charts = chart_ids[rows, parent_columns]
        children = [
            atlas.sample_local_chart_mixture_torch(
                parent_charts,
                global_component_probability=global_probability,
                generator=generator,
                reference=state["strengths"],
            )
            for _ in range(int(group_size))
        ]
        keep = (
            torch.arange(int(cardinality), device=chart_ids.device)[None, :]
            != parent_columns[:, None]
        )
        retained_count = int(cardinality) - 1
        child_charts = torch.stack([child[0] for child in children], dim=1)
        child_uv = torch.stack([child[1] for child in children], dim=1)
        child_positions = torch.stack([child[2] for child in children], dim=1)
        child_log_density = torch.stack(
            [child[3] for child in children],
            dim=1,
        )
        proposed_charts = torch.cat(
            (
                chart_ids[keep].reshape(row_count, retained_count),
                child_charts,
            ),
            dim=1,
        )
        proposed_uv = torch.cat(
            (
                surface_uv[keep].reshape(row_count, retained_count, 2),
                child_uv,
            ),
            dim=1,
        )
        proposed_positions = torch.cat(
            (
                positions[keep].reshape(row_count, retained_count, 3),
                child_positions,
            ),
            dim=1,
        )
        proposed_cardinality = int(cardinality) + int(group_size) - 1
        proposed_centers = self._continuous_rj_block_strength_centers_torch(
            data,
            chart_ids=proposed_charts,
            positions=proposed_positions,
            particle_indices=particle_indices,
            target_beta=target_beta,
        )
        proposed_strengths = self._continuous_rj_sample_block_strength_torch(
            proposed_centers
        )
        current_centers = self._continuous_rj_block_strength_centers_torch(
            data,
            chart_ids=chart_ids,
            positions=positions,
            particle_indices=particle_indices,
            target_beta=target_beta,
        )
        current_strength_log_proposal = (
            self._continuous_rj_block_strength_log_density_torch(
                strengths,
                current_centers,
            )
        )
        proposed_strength_log_proposal = (
            self._continuous_rj_block_strength_log_density_torch(
                proposed_strengths,
                proposed_centers,
            )
        )
        reverse_groups, reverse_probabilities = (
            self._continuous_rj_multi_group_probabilities_torch(
                data,
                proposed_charts,
                proposed_uv,
                proposed_positions,
                group_size=group_size,
            )
        )
        child_columns = torch.arange(
            retained_count,
            proposed_cardinality,
            device=chart_ids.device,
        )
        reverse_matches = torch.all(
            reverse_groups == child_columns[None, :],
            dim=1,
        )
        if int(torch.count_nonzero(reverse_matches).item()) != 1:
            raise RuntimeError("Reverse CUDA multi-merge group is not unique.")
        reverse_column = torch.nonzero(
            reverse_matches,
            as_tuple=False,
        ).reshape(-1)[0]
        reverse_group_log_probability = torch.log(
            reverse_probabilities[:, reverse_column]
        )
        reverse_anchor = self._continuous_rj_merge_anchor_probabilities_torch(
            child_charts
        )
        reverse_merged_log_density = torch.logsumexp(
            torch.stack(
                [
                    torch.log(reverse_anchor[:, index])
                    + atlas.local_chart_mixture_log_density_torch(
                        child_charts[:, index],
                        parent_charts,
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    )
                    for index in range(int(group_size))
                ],
                dim=1,
            ),
            dim=1,
        )
        _, reverse_merge_sizes, _, reverse_merge_probability = (
            self._continuous_rj_multi_direction_support(proposed_cardinality)
        )
        log_forward = (
            math.log(split_direction_probability)
            - math.log(float(len(split_sizes)))
            - math.log(float(cardinality))
            + math.lgamma(float(group_size) + 1.0)
            + torch.sum(child_log_density, dim=1)
            + proposed_strength_log_proposal
        )
        log_reverse = (
            math.log(reverse_merge_probability)
            - math.log(float(len(reverse_merge_sizes)))
            + reverse_group_log_probability
            + reverse_merged_log_density
            + current_strength_log_proposal
        )
        proposed = self._continuous_rj_canonicalize_tensors(
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        proposed_charts, proposed_uv, proposed_positions, proposed_strengths = (
            proposed
        )
        strength_support = torch.all(
            self._continuous_rj_strength_support_torch(proposed_strengths),
            dim=1,
        )
        geometry_support = torch.isfinite(log_forward) & torch.isfinite(log_reverse)
        feasible = geometry_support & strength_support
        proposed_prior = torch.full_like(base_ll, float("-inf"))
        valid_rows = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if int(valid_rows.numel()):
            proposed_prior[valid_rows] = self._continuous_rj_block_log_prior_torch(
                proposed_charts[valid_rows],
                proposed_strengths[valid_rows],
            )
        delta_prior = proposed_prior - current_prior
        reverse_minus_forward = log_reverse - log_forward
        decision = self._continuous_rj_exact_decision_torch(
            data,
            proposed_positions,
            proposed_strengths,
            proposed_chart_ids=proposed_charts,
            particle_indices=particle_indices,
            base_log_likelihood=base_ll,
            log_non_likelihood_ratio=delta_prior + reverse_minus_forward,
            support=feasible,
            target_beta=target_beta,
            move_family="multi_split",
        )
        proposed_ll = decision.proposed_target_log_likelihood
        delta_ll = decision.diagnostic_delta_log_likelihood
        log_ratio = decision.diagnostic_log_acceptance_ratio
        accepted = decision.accepted
        self._record_structural_mh_components_torch(
            "multi_split",
            particle_indices=particle_indices,
            delta_log_likelihood=delta_ll,
            delta_log_prior=delta_prior,
            log_reverse_minus_forward=reverse_minus_forward,
            log_jacobian=0.0,
            support_feasible=feasible,
            accepted=accepted,
            current_cardinality=cardinality,
            proposed_cardinality=proposed_cardinality,
            geometry_support_feasible=geometry_support,
            strength_support_feasible=strength_support,
            log_acceptance_ratio=log_ratio,
        )
        self._commit_continuous_rj_state_tensors(
            particle_indices,
            accepted,
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        return accepted, proposed_ll, decision.proposed_station_log_likelihood

    def _continuous_rj_multi_merge_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        particle_indices: torch.Tensor,
        cardinality: int,
        group_size: int,
        chart_ids: torch.Tensor,
        surface_uv: torch.Tensor,
        positions: torch.Tensor,
        strengths: torch.Tensor,
        base_ll: torch.Tensor,
        current_prior: torch.Tensor,
        merge_sizes: tuple[int, ...],
        target_beta: float,
        global_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        """Propose one exact multi-merge entirely on CUDA."""
        atlas = self._structural_rj_surface_atlas
        state = self._structural_rj_device_state
        if atlas is None or state is None:
            raise RuntimeError("CUDA multi-merge surface state is unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        row_count = int(particle_indices.numel())
        rows = torch.arange(row_count, device=chart_ids.device)
        groups, probabilities = (
            self._continuous_rj_multi_group_probabilities_torch(
                data,
                chart_ids,
                surface_uv,
                positions,
                group_size=group_size,
            )
        )
        group_columns = self._continuous_rj_sample_rows_torch(probabilities)
        selected_columns = groups[group_columns]
        selected_charts = chart_ids[rows[:, None], selected_columns]
        anchor_probabilities = (
            self._continuous_rj_merge_anchor_probabilities_torch(
                selected_charts
            )
        )
        anchor_offsets = self._continuous_rj_sample_rows_torch(
            anchor_probabilities
        )
        anchor_charts = selected_charts[rows, anchor_offsets]
        merged = atlas.sample_local_chart_mixture_torch(
            anchor_charts,
            global_component_probability=global_probability,
            generator=generator,
            reference=state["strengths"],
        )
        merged_charts, merged_uv, merged_positions, _ = merged
        forward_merged_log_density = torch.logsumexp(
            torch.stack(
                [
                    torch.log(anchor_probabilities[:, index])
                    + atlas.local_chart_mixture_log_density_torch(
                        selected_charts[:, index],
                        merged_charts,
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    )
                    for index in range(int(group_size))
                ],
                dim=1,
            ),
            dim=1,
        )
        keep = torch.ones(
            (row_count, int(cardinality)),
            device=chart_ids.device,
            dtype=torch.bool,
        )
        keep[rows[:, None], selected_columns] = False
        retained_count = int(cardinality) - int(group_size)
        proposed_charts = torch.cat(
            (
                chart_ids[keep].reshape(row_count, retained_count),
                merged_charts[:, None],
            ),
            dim=1,
        )
        proposed_uv = torch.cat(
            (
                surface_uv[keep].reshape(row_count, retained_count, 2),
                merged_uv[:, None, :],
            ),
            dim=1,
        )
        proposed_positions = torch.cat(
            (
                positions[keep].reshape(row_count, retained_count, 3),
                merged_positions[:, None, :],
            ),
            dim=1,
        )
        proposed_cardinality = int(cardinality) - int(group_size) + 1
        proposed_centers = self._continuous_rj_block_strength_centers_torch(
            data,
            chart_ids=proposed_charts,
            positions=proposed_positions,
            particle_indices=particle_indices,
            target_beta=target_beta,
        )
        proposed_strengths = self._continuous_rj_sample_block_strength_torch(
            proposed_centers
        )
        current_centers = self._continuous_rj_block_strength_centers_torch(
            data,
            chart_ids=chart_ids,
            positions=positions,
            particle_indices=particle_indices,
            target_beta=target_beta,
        )
        current_strength_log_proposal = (
            self._continuous_rj_block_strength_log_density_torch(
                strengths,
                current_centers,
            )
        )
        proposed_strength_log_proposal = (
            self._continuous_rj_block_strength_log_density_torch(
                proposed_strengths,
                proposed_centers,
            )
        )
        split_sizes, _, reverse_split_probability, _ = (
            self._continuous_rj_multi_direction_support(proposed_cardinality)
        )
        reverse_child_position_log_density = torch.sum(
            torch.stack(
                [
                    atlas.local_chart_mixture_log_density_torch(
                        merged_charts,
                        selected_charts[:, index],
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    )
                    for index in range(int(group_size))
                ],
                dim=1,
            ),
            dim=1,
        )
        _, _, _, merge_direction_probability = (
            self._continuous_rj_multi_direction_support(cardinality)
        )
        log_forward = (
            math.log(merge_direction_probability)
            - math.log(float(len(merge_sizes)))
            + torch.log(probabilities[rows, group_columns])
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
        proposed = self._continuous_rj_canonicalize_tensors(
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        proposed_charts, proposed_uv, proposed_positions, proposed_strengths = (
            proposed
        )
        strength_support = torch.all(
            self._continuous_rj_strength_support_torch(proposed_strengths),
            dim=1,
        )
        geometry_support = torch.isfinite(log_forward) & torch.isfinite(log_reverse)
        feasible = geometry_support & strength_support
        proposed_prior = torch.full_like(base_ll, float("-inf"))
        valid_rows = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if int(valid_rows.numel()):
            proposed_prior[valid_rows] = self._continuous_rj_block_log_prior_torch(
                proposed_charts[valid_rows],
                proposed_strengths[valid_rows],
            )
        delta_prior = proposed_prior - current_prior
        reverse_minus_forward = log_reverse - log_forward
        decision = self._continuous_rj_exact_decision_torch(
            data,
            proposed_positions,
            proposed_strengths,
            proposed_chart_ids=proposed_charts,
            particle_indices=particle_indices,
            base_log_likelihood=base_ll,
            log_non_likelihood_ratio=delta_prior + reverse_minus_forward,
            support=feasible,
            target_beta=target_beta,
            move_family="multi_merge",
        )
        proposed_ll = decision.proposed_target_log_likelihood
        delta_ll = decision.diagnostic_delta_log_likelihood
        log_ratio = decision.diagnostic_log_acceptance_ratio
        accepted = decision.accepted
        self._record_structural_mh_components_torch(
            "multi_merge",
            particle_indices=particle_indices,
            delta_log_likelihood=delta_ll,
            delta_log_prior=delta_prior,
            log_reverse_minus_forward=reverse_minus_forward,
            log_jacobian=0.0,
            support_feasible=feasible,
            accepted=accepted,
            current_cardinality=cardinality,
            proposed_cardinality=proposed_cardinality,
            geometry_support_feasible=geometry_support,
            strength_support_feasible=strength_support,
            log_acceptance_ratio=log_ratio,
        )
        self._commit_continuous_rj_state_tensors(
            particle_indices,
            accepted,
            proposed_charts,
            proposed_uv,
            proposed_positions,
            proposed_strengths,
        )
        return accepted, proposed_ll, decision.proposed_station_log_likelihood

    def _apply_continuous_rj_multi_component_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact multi-component proposals without host particle state."""
        probability = float(self.config.structural_rj_multi_component_probability)
        if probability <= 0.0:
            return 0, 0
        state = self._structural_rj_device_state
        if state is None or self._structural_rj_surface_atlas is None:
            raise RuntimeError("CUDA multi-component state is unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        cardinalities = state["cardinalities"]
        dtype = state["strengths"].dtype
        device = state["strengths"].device
        maximum = int(self.config.hard_max_sources or 0)
        available_table = torch.tensor(
            [
                sum(self._continuous_rj_multi_direction_support(value)[2:])
                > 0.0
                for value in range(maximum + 1)
            ],
            device=device,
            dtype=torch.bool,
        )
        attempted = (
            torch.rand(
                cardinalities.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            < probability
        ) & available_table[cardinalities]
        accepted_splits = 0
        accepted_merges = 0
        attempted_splits = 0
        attempted_merges = 0
        global_probability = float(
            self.config.structural_rj_split_global_position_probability
        )
        for value in torch.unique(cardinalities[attempted]).tolist():
            cardinality = int(value)
            particle_indices = torch.nonzero(
                attempted & (cardinalities == cardinality),
                as_tuple=False,
            ).reshape(-1)
            (
                split_sizes,
                merge_sizes,
                split_direction_probability,
                _,
            ) = self._continuous_rj_multi_direction_support(cardinality)
            split_rows = torch.rand(
                particle_indices.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            ) < split_direction_probability
            for is_split in (True, False):
                direction_rows = torch.nonzero(
                    split_rows == is_split,
                    as_tuple=False,
                ).reshape(-1)
                if int(direction_rows.numel()) == 0:
                    continue
                sizes = split_sizes if is_split else merge_sizes
                if not sizes:
                    continue
                size_table = torch.tensor(sizes, device=device, dtype=torch.long)
                size_columns = torch.randint(
                    len(sizes),
                    direction_rows.shape,
                    device=device,
                    generator=generator,
                )
                chosen_sizes = size_table[size_columns]
                for size_value in torch.unique(chosen_sizes).tolist():
                    group_size = int(size_value)
                    local_rows = direction_rows[chosen_sizes == group_size]
                    indices = particle_indices[local_rows]
                    (
                        indices,
                        chart_ids,
                        surface_uv,
                        positions,
                        strengths,
                    ) = self._continuous_rj_group_tensors(
                        indices,
                        cardinality,
                    )
                    base_ll = self._continuous_rj_current_log_likelihood_torch(
                        data,
                        positions,
                        strengths,
                        chart_ids=chart_ids,
                        particle_indices=indices,
                        target_beta=target_beta,
                    )
                    current_prior = self._continuous_rj_block_log_prior_torch(
                        chart_ids,
                        strengths,
                    )
                    row_count = int(indices.numel())
                    if is_split:
                        attempted_splits += row_count
                        self._continuous_rj_transition_mass_torch(
                            "multi_split_attempted",
                            indices,
                        )
                        accepted, proposed_ll, proposed_station_ll = (
                            self._continuous_rj_multi_split_torch(
                                data,
                                particle_indices=indices,
                                cardinality=cardinality,
                                group_size=group_size,
                                chart_ids=chart_ids,
                                surface_uv=surface_uv,
                                positions=positions,
                                strengths=strengths,
                                base_ll=base_ll,
                                current_prior=current_prior,
                                split_sizes=split_sizes,
                                split_direction_probability=(
                                    split_direction_probability
                                ),
                                target_beta=target_beta,
                                global_probability=global_probability,
                            )
                        )
                        accepted_splits += int(
                            torch.count_nonzero(accepted).item()
                        )
                        self._update_continuous_rj_current_log_likelihood_torch(
                            indices,
                            accepted,
                            proposed_ll,
                            proposed_station_ll,
                        )
                        self._continuous_rj_transition_mass_torch(
                            "multi_split_accepted",
                            indices,
                            accepted,
                        )
                    else:
                        attempted_merges += row_count
                        self._continuous_rj_transition_mass_torch(
                            "multi_merge_attempted",
                            indices,
                        )
                        accepted, proposed_ll, proposed_station_ll = (
                            self._continuous_rj_multi_merge_torch(
                                data,
                                particle_indices=indices,
                                cardinality=cardinality,
                                group_size=group_size,
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
                        )
                        accepted_merges += int(
                            torch.count_nonzero(accepted).item()
                        )
                        self._update_continuous_rj_current_log_likelihood_torch(
                            indices,
                            accepted,
                            proposed_ll,
                            proposed_station_ll,
                        )
                        self._continuous_rj_transition_mass_torch(
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


__all__ = ["StructuralRJTorchMultiComponentMixin"]
