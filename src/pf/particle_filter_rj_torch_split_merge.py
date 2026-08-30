"""Device-resident relocated split/merge kernels for exact-RJ PF."""

from __future__ import annotations

import math

from pf.particle_types import StructuralGeometryBatch


class StructuralRJTorchSplitMergeMixin:
    """Provide Torch-native exact relocated split and merge moves."""

    def _continuous_rj_split_fraction_bounds_torch(
        self,
        total_strength: object,
    ) -> tuple[object, object, object]:
        """Return the exact feasible split-fraction interval on Torch."""
        import torch

        if not torch.is_tensor(total_strength):
            raise TypeError("CUDA split strengths must be a tensor.")
        total = total_strength
        minimum = float(self._strength_prior.minimum)
        maximum = float(self._strength_prior.support_maximum)
        safe = torch.clamp(total, min=torch.finfo(total.dtype).tiny)
        lower = torch.maximum(
            torch.full_like(total, minimum) / safe,
            1.0 - torch.full_like(total, maximum) / safe,
        )
        upper = torch.minimum(
            torch.full_like(total, maximum) / safe,
            1.0 - torch.full_like(total, minimum) / safe,
        )
        feasible = (
            torch.isfinite(total)
            & (total > 0.0)
            & (upper > lower)
            & (lower >= 0.0)
            & (upper <= 1.0)
        )
        return lower, upper, feasible

    def _apply_continuous_rj_split_merge_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact relocated split/merge proposals entirely on CUDA."""
        import torch

        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_split_merge_probabilities
        state = self._structural_rj_device_state
        if (
            atlas is None
            or cardinality_prior is None
            or move_probabilities is None
            or state is None
        ):
            raise RuntimeError("Continuous split/merge priors are unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        dtype = state["strengths"].dtype
        device = state["strengths"].device
        maximum = int(self.config.hard_max_sources or 0)
        split_table = torch.tensor(
            [
                float(move_probabilities.probabilities(k)[0])
                for k in range(maximum + 1)
            ],
            device=device,
            dtype=dtype,
        )
        merge_table = torch.tensor(
            [
                float(move_probabilities.probabilities(k)[1])
                for k in range(maximum + 1)
            ],
            device=device,
            dtype=dtype,
        )
        cardinalities = state["cardinalities"]
        direction_available = (
            split_table[cardinalities] + merge_table[cardinalities]
        ) > 0.0
        attempt = (
            torch.rand(
                cardinalities.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            < float(self.config.structural_rj_split_merge_probability)
        ) & direction_available
        constants = self._continuous_rj_atlas_tensors()
        cardinality_log_prior = torch.tensor(
            cardinality_prior.log_probabilities,
            device=device,
            dtype=dtype,
        )
        attempted_splits = 0
        accepted_splits = 0
        attempted_merges = 0
        accepted_merges = 0
        for cardinality_value in torch.unique(cardinalities[attempt]).tolist():
            cardinality = int(cardinality_value)
            group_indices = torch.nonzero(
                attempt & (cardinalities == cardinality),
                as_tuple=False,
            ).reshape(-1)
            split_move = torch.rand(
                group_indices.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            ) < split_table[cardinality]
            for is_split in (True, False):
                particle_indices = group_indices[split_move == is_split]
                if int(particle_indices.numel()) == 0:
                    continue
                (
                    indices,
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_tensors(
                    particle_indices,
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
                row_count = int(indices.numel())
                rows = torch.arange(row_count, device=device)
                global_probability = float(
                    self.config.structural_rj_split_global_position_probability
                )
                if is_split:
                    attempted_splits += row_count
                    self._continuous_rj_transition_mass_torch(
                        "split_attempted",
                        indices,
                    )
                    source_columns = torch.randint(
                        cardinality,
                        (row_count,),
                        device=device,
                        generator=generator,
                    )
                    total_strength = strengths[rows, source_columns]
                    lower, upper, feasible = (
                        self._continuous_rj_split_fraction_bounds_torch(
                            total_strength
                        )
                    )
                    width = upper - lower
                    safe_width = torch.where(
                        feasible,
                        width,
                        torch.ones_like(width),
                    )
                    fraction = lower + safe_width * torch.rand(
                        (row_count,),
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    first_strength = (1.0 - fraction) * total_strength
                    second_strength = fraction * total_strength
                    parent_charts = chart_ids[rows, source_columns]
                    first = atlas.sample_local_chart_mixture_torch(
                        parent_charts,
                        global_component_probability=global_probability,
                        generator=generator,
                        reference=state["strengths"],
                    )
                    second = atlas.sample_local_chart_mixture_torch(
                        parent_charts,
                        global_component_probability=global_probability,
                        generator=generator,
                        reference=state["strengths"],
                    )
                    keep = (
                        torch.arange(cardinality, device=device)[None, :]
                        != source_columns[:, None]
                    )
                    retained_charts = chart_ids[keep].reshape(
                        row_count,
                        cardinality - 1,
                    )
                    retained_uv = surface_uv[keep].reshape(
                        row_count,
                        cardinality - 1,
                        2,
                    )
                    retained_positions = positions[keep].reshape(
                        row_count,
                        cardinality - 1,
                        3,
                    )
                    retained_strengths = strengths[keep].reshape(
                        row_count,
                        cardinality - 1,
                    )
                    proposed_charts = torch.cat(
                        (retained_charts, first[0][:, None], second[0][:, None]),
                        dim=1,
                    )
                    proposed_uv = torch.cat(
                        (retained_uv, first[1][:, None, :], second[1][:, None, :]),
                        dim=1,
                    )
                    proposed_positions = torch.cat(
                        (
                            retained_positions,
                            first[2][:, None, :],
                            second[2][:, None, :],
                        ),
                        dim=1,
                    )
                    proposed_strengths = torch.cat(
                        (
                            retained_strengths,
                            first_strength[:, None],
                            second_strength[:, None],
                        ),
                        dim=1,
                    )
                    reverse_donors, reverse_receivers, reverse_probabilities = (
                        self._continuous_rj_ordered_pair_probabilities_torch(
                            data,
                            proposed_charts,
                            proposed_uv,
                            proposed_positions,
                            proposed_strengths,
                        )
                    )
                    reverse_matches = (
                        (reverse_donors == cardinality)
                        & (reverse_receivers == cardinality - 1)
                    )
                    if int(torch.count_nonzero(reverse_matches).item()) != 1:
                        raise RuntimeError(
                            "Reverse CUDA merge pair is not uniquely represented."
                        )
                    reverse_column = torch.nonzero(
                        reverse_matches,
                        as_tuple=False,
                    ).reshape(-1)[0]
                    log_reverse_pair = torch.log(
                        reverse_probabilities[:, reverse_column]
                    )
                    log_reverse_position = torch.logaddexp(
                        atlas.local_chart_mixture_log_density_torch(
                            first[0],
                            parent_charts,
                            global_component_probability=global_probability,
                            reference=state["strengths"],
                        ),
                        atlas.local_chart_mixture_log_density_torch(
                            second[0],
                            parent_charts,
                            global_component_probability=global_probability,
                            reference=state["strengths"],
                        ),
                    ) - math.log(2.0)
                    (
                        proposed_charts,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    ) = self._continuous_rj_canonicalize_tensors(
                        proposed_charts,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    delta_prior = (
                        cardinality_log_prior[cardinality + 1]
                        - cardinality_log_prior[cardinality]
                        + math.log(float(cardinality + 1))
                        + constants["log_chart_probabilities"][first[0]]
                        + constants["log_chart_probabilities"][second[0]]
                        - constants["log_chart_probabilities"][parent_charts]
                        + self._continuous_rj_strength_log_prior_torch(
                            first_strength
                        )
                        + self._continuous_rj_strength_log_prior_torch(
                            second_strength
                        )
                        - self._continuous_rj_strength_log_prior_torch(
                            total_strength
                        )
                    )
                    proposal_delta = (
                        float(
                            move_probabilities.log_probability(
                                "merge",
                                cardinality + 1,
                            )
                        )
                        + log_reverse_pair
                        + log_reverse_position
                        - float(
                            move_probabilities.log_probability(
                                "split",
                                cardinality,
                            )
                        )
                        + math.log(float(cardinality))
                        - first[3]
                        - second[3]
                        + torch.log(width)
                    )
                    log_jacobian = torch.log(total_strength)
                    decision = self._continuous_rj_history_tree_decision_torch(
                        data,
                        proposed_positions,
                        proposed_strengths,
                        proposed_chart_ids=proposed_charts,
                        particle_indices=indices,
                        base_log_likelihood=base_ll,
                        log_non_likelihood_ratio=(
                            delta_prior + proposal_delta + log_jacobian
                        ),
                        support=feasible,
                        target_beta=target_beta,
                        move_family="split",
                    )
                    proposed_ll = decision.proposed_target_log_likelihood
                    delta_ll = decision.diagnostic_delta_log_likelihood
                    log_ratio = decision.diagnostic_log_acceptance_ratio
                    accepted = decision.accepted
                    diagnostic_nan = torch.full_like(log_ratio, float("nan"))
                    self._record_structural_mh_components_torch(
                        "split",
                        particle_indices=indices,
                        delta_log_likelihood=delta_ll,
                        delta_log_prior=torch.where(
                            feasible,
                            delta_prior,
                            diagnostic_nan,
                        ),
                        log_reverse_minus_forward=torch.where(
                            feasible,
                            proposal_delta,
                            diagnostic_nan,
                        ),
                        log_jacobian=torch.where(
                            feasible,
                            log_jacobian,
                            diagnostic_nan,
                        ),
                        support_feasible=feasible,
                        accepted=accepted,
                        current_cardinality=cardinality,
                        proposed_cardinality=cardinality + 1,
                        geometry_support_feasible=torch.where(
                            feasible,
                            torch.isfinite(proposal_delta),
                            torch.ones_like(feasible),
                        ),
                        strength_support_feasible=feasible,
                        log_acceptance_ratio=log_ratio,
                        likelihood_exact=decision.likelihood_exact,
                    )
                    accepted_splits += self._commit_continuous_rj_state_tensors(
                        indices,
                        accepted,
                        proposed_charts,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    self._update_continuous_rj_current_log_likelihood_torch(
                        indices,
                        accepted,
                        proposed_ll,
                        decision.proposed_station_log_likelihood,
                    )
                    self._continuous_rj_transition_mass_torch(
                        "split_accepted",
                        indices,
                        accepted,
                    )
                    continue

                attempted_merges += row_count
                self._continuous_rj_transition_mass_torch(
                    "merge_attempted",
                    indices,
                )
                donors, receivers, probabilities = (
                    self._continuous_rj_ordered_pair_probabilities_torch(
                        data,
                        chart_ids,
                        surface_uv,
                        positions,
                        strengths,
                    )
                )
                cdf = torch.cumsum(probabilities, dim=1)
                draws = torch.rand(
                    (row_count,),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                pair_columns = torch.sum(
                    draws[:, None] > cdf,
                    dim=1,
                    dtype=torch.long,
                )
                delete_columns = donors[pair_columns]
                receiver_columns = receivers[pair_columns]
                log_forward_pair = torch.log(probabilities[rows, pair_columns])
                second_charts = chart_ids[rows, delete_columns]
                first_charts = chart_ids[rows, receiver_columns]
                second_strengths = strengths[rows, delete_columns]
                first_strengths = strengths[rows, receiver_columns]
                merged_strength = second_strengths + first_strengths
                lower, upper, reverse_feasible = (
                    self._continuous_rj_split_fraction_bounds_torch(
                        merged_strength
                    )
                )
                reverse_fraction = second_strengths / torch.clamp(
                    merged_strength,
                    min=torch.finfo(dtype).tiny,
                )
                feasible = (
                    self._continuous_rj_strength_support_torch(merged_strength)
                    & reverse_feasible
                    & (reverse_fraction >= lower)
                    & (reverse_fraction <= upper)
                )
                use_first_anchor = torch.rand(
                    (row_count,),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ) < 0.5
                anchors = torch.where(
                    use_first_anchor,
                    first_charts,
                    second_charts,
                )
                merged = atlas.sample_local_chart_mixture_torch(
                    anchors,
                    global_component_probability=global_probability,
                    generator=generator,
                    reference=state["strengths"],
                )
                log_forward_position = torch.logaddexp(
                    atlas.local_chart_mixture_log_density_torch(
                        first_charts,
                        merged[0],
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    ),
                    atlas.local_chart_mixture_log_density_torch(
                        second_charts,
                        merged[0],
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    ),
                ) - math.log(2.0)
                source_columns = torch.arange(cardinality, device=device)[None, :]
                keep = (source_columns != delete_columns[:, None]) & (
                    source_columns != receiver_columns[:, None]
                )
                retained_charts = chart_ids[keep].reshape(
                    row_count,
                    cardinality - 2,
                )
                retained_uv = surface_uv[keep].reshape(
                    row_count,
                    cardinality - 2,
                    2,
                )
                retained_positions = positions[keep].reshape(
                    row_count,
                    cardinality - 2,
                    3,
                )
                retained_strengths = strengths[keep].reshape(
                    row_count,
                    cardinality - 2,
                )
                proposed_charts = torch.cat(
                    (retained_charts, merged[0][:, None]),
                    dim=1,
                )
                proposed_uv = torch.cat(
                    (retained_uv, merged[1][:, None, :]),
                    dim=1,
                )
                proposed_positions = torch.cat(
                    (retained_positions, merged[2][:, None, :]),
                    dim=1,
                )
                proposed_strengths = torch.cat(
                    (retained_strengths, merged_strength[:, None]),
                    dim=1,
                )
                (
                    proposed_charts,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                ) = self._continuous_rj_canonicalize_tensors(
                    proposed_charts,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                delta_prior = (
                    cardinality_log_prior[cardinality - 1]
                    - cardinality_log_prior[cardinality]
                    - math.log(float(cardinality))
                    + constants["log_chart_probabilities"][merged[0]]
                    - constants["log_chart_probabilities"][first_charts]
                    - constants["log_chart_probabilities"][second_charts]
                    + self._continuous_rj_strength_log_prior_torch(
                        merged_strength
                    )
                    - self._continuous_rj_strength_log_prior_torch(first_strengths)
                    - self._continuous_rj_strength_log_prior_torch(second_strengths)
                )
                width = upper - lower
                proposal_delta = (
                    float(
                        move_probabilities.log_probability(
                            "split",
                            cardinality - 1,
                        )
                    )
                    - math.log(float(cardinality - 1))
                    + atlas.local_chart_mixture_log_density_torch(
                        merged[0],
                        first_charts,
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    )
                    + atlas.local_chart_mixture_log_density_torch(
                        merged[0],
                        second_charts,
                        global_component_probability=global_probability,
                        reference=state["strengths"],
                    )
                    - torch.log(width)
                    - float(
                        move_probabilities.log_probability(
                            "merge",
                            cardinality,
                        )
                    )
                    - log_forward_pair
                    - log_forward_position
                )
                log_jacobian = -torch.log(merged_strength)
                decision = self._continuous_rj_history_tree_decision_torch(
                    data,
                    proposed_positions,
                    proposed_strengths,
                    proposed_chart_ids=proposed_charts,
                    particle_indices=indices,
                    base_log_likelihood=base_ll,
                    log_non_likelihood_ratio=(
                        delta_prior + proposal_delta + log_jacobian
                    ),
                    support=feasible,
                    target_beta=target_beta,
                    move_family="merge",
                )
                proposed_ll = decision.proposed_target_log_likelihood
                delta_ll = decision.diagnostic_delta_log_likelihood
                log_ratio = decision.diagnostic_log_acceptance_ratio
                accepted = decision.accepted
                diagnostic_nan = torch.full_like(log_ratio, float("nan"))
                self._record_structural_mh_components_torch(
                    "merge",
                    particle_indices=indices,
                    delta_log_likelihood=delta_ll,
                    delta_log_prior=torch.where(
                        feasible,
                        delta_prior,
                        diagnostic_nan,
                    ),
                    log_reverse_minus_forward=torch.where(
                        feasible,
                        proposal_delta,
                        diagnostic_nan,
                    ),
                    log_jacobian=torch.where(
                        feasible,
                        log_jacobian,
                        diagnostic_nan,
                    ),
                    support_feasible=feasible,
                    accepted=accepted,
                    current_cardinality=cardinality,
                    proposed_cardinality=cardinality - 1,
                    geometry_support_feasible=torch.where(
                        feasible,
                        torch.isfinite(proposal_delta),
                        torch.ones_like(feasible),
                    ),
                    strength_support_feasible=feasible,
                    log_acceptance_ratio=log_ratio,
                    likelihood_exact=decision.likelihood_exact,
                )
                accepted_merges += self._commit_continuous_rj_state_tensors(
                    indices,
                    accepted,
                    proposed_charts,
                    proposed_uv,
                    proposed_positions,
                    proposed_strengths,
                )
                self._update_continuous_rj_current_log_likelihood_torch(
                    indices,
                    accepted,
                    proposed_ll,
                    decision.proposed_station_log_likelihood,
                )
                self._continuous_rj_transition_mass_torch(
                    "merge_accepted",
                    indices,
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


__all__ = ["StructuralRJTorchSplitMergeMixin"]
