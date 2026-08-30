"""Device-resident elementary exact-RJ kernels for production CUDA PF."""

from __future__ import annotations

import math

import numpy as np

from pf.particle_types import StructuralGeometryBatch


class StructuralRJTorchBasicMoveMixin:
    """Provide Torch-native birth, death, position, and strength moves."""

    def _continuous_rj_integrated_unit_response_torch(
        self,
        data: StructuralGeometryBatch,
        positions: object,
        chart_ids: object,
    ) -> object:
        """Return all-history unit-strength counts on the active CUDA device."""
        import torch

        if not torch.is_tensor(positions) or not torch.is_tensor(chart_ids):
            raise TypeError("CUDA integrated-response inputs must be tensors.")
        flat_positions = positions.reshape(-1, 3)
        flat_charts = chart_ids.reshape(-1)
        if int(flat_positions.shape[0]) != int(flat_charts.numel()):
            raise ValueError("CUDA unit-response positions and charts must align.")
        line_indices = self.continuous_kernel.positive_line_indices(self.isotope)
        branching = torch.tensor(
            self.continuous_kernel.line_branching_weights(
                self.isotope,
                line_indices,
            ),
            device=flat_positions.device,
            dtype=flat_positions.dtype,
        )
        host_payload = torch.cat(
            (
                flat_positions,
                flat_charts[:, None].to(dtype=flat_positions.dtype),
            ),
            dim=1,
        ).detach().cpu().numpy()
        components = self._continuous_rj_line_transport_component_columns(
            data,
            host_payload[:, :3],
            line_indices,
            chart_ids=host_payload[:, 3].astype(np.int64),
            device_resident=True,
        )
        if not torch.is_tensor(components.total_kernel):
            raise RuntimeError("CUDA coupled proposal received host transport.")
        live_times = torch.tensor(
            data.live_times,
            device=flat_positions.device,
            dtype=flat_positions.dtype,
        )
        response = torch.einsum(
            "vsl,l,v->s",
            components.total_kernel,
            branching,
            live_times,
        )
        if tuple(response.shape) != (int(flat_positions.shape[0]),) or bool(
            torch.any(~torch.isfinite(response)).item()
        ):
            raise RuntimeError("CUDA coupled-move unit response is invalid.")
        return response

    def _continuous_rj_source_response_signatures_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        chart_ids: object,
        positions: object,
    ) -> object:
        """Return normalized line-response signatures on CUDA."""
        import torch

        if not torch.is_tensor(chart_ids) or not torch.is_tensor(positions):
            raise TypeError("CUDA response signatures require tensor state.")
        row_count, cardinality = chart_ids.shape
        line_indices = self.continuous_kernel.positive_line_indices(self.isotope)
        branching = torch.tensor(
            self.continuous_kernel.line_branching_weights(
                self.isotope,
                line_indices,
            ),
            device=positions.device,
            dtype=positions.dtype,
        )
        flat_positions = positions.reshape(-1, 3)
        flat_charts = chart_ids.reshape(-1)
        host_payload = torch.cat(
            (
                flat_positions,
                flat_charts[:, None].to(dtype=positions.dtype),
            ),
            dim=1,
        ).detach().cpu().numpy()
        components = self._continuous_rj_line_transport_component_columns(
            data,
            host_payload[:, :3],
            line_indices,
            chart_ids=host_payload[:, 3].astype(np.int64),
            device_resident=True,
        )
        if not torch.is_tensor(components.total_kernel):
            raise RuntimeError("CUDA response signature received host transport.")
        physical = components.total_kernel.reshape(
            int(data.row_count),
            int(row_count),
            int(cardinality),
            int(line_indices.size),
        ) * branching[None, None, None, :]
        signatures = physical.permute(1, 2, 0, 3).reshape(
            int(row_count),
            int(cardinality),
            -1,
        )
        norms = torch.linalg.vector_norm(signatures, dim=-1, keepdim=True)
        return torch.where(
            norms > 0.0,
            signatures
            / torch.where(norms > 0.0, norms, torch.ones_like(norms)),
            torch.zeros_like(signatures),
        )

    def _continuous_rj_ordered_pair_probabilities_torch(
        self,
        data: StructuralGeometryBatch,
        chart_ids: object,
        surface_uv: object,
        positions: object,
        strengths: object,
    ) -> tuple[object, object, object]:
        """Return exact response-aware ordered merge probabilities on CUDA."""
        import torch

        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        cardinality = int(chart_ids.shape[1])
        if cardinality < 2:
            raise ValueError("Ordered pair probabilities require two sources.")
        columns = torch.arange(cardinality, device=chart_ids.device)
        donors = columns[:, None].expand(cardinality, cardinality)
        receivers = columns[None, :].expand(cardinality, cardinality)
        distinct = donors != receivers
        donor_columns = donors[distinct]
        receiver_columns = receivers[distinct]
        distances = atlas.local_surface_coordinate_path_distance_m_torch(
            chart_ids[:, donor_columns],
            surface_uv[:, donor_columns, :],
            chart_ids[:, receiver_columns],
            surface_uv[:, receiver_columns, :],
        )
        signatures = self._continuous_rj_source_response_signatures_torch(
            data,
            chart_ids=chart_ids,
            positions=positions,
        )
        response_cosine = torch.clamp(
            torch.sum(
                signatures[:, donor_columns, :]
                * signatures[:, receiver_columns, :],
                dim=-1,
            ),
            min=-1.0,
            max=1.0,
        )
        response_distance = torch.sqrt(
            torch.clamp(2.0 - 2.0 * response_cosine, min=0.0)
        )
        strength_scale = (
            float(self._strength_prior.gamma_scale)
            if self._strength_prior.family == "shifted_gamma"
            else float(
                self._strength_prior.maximum - self._strength_prior.minimum
            )
        )
        donor_excess = torch.clamp(
            strengths[:, donor_columns] - self._strength_prior.minimum,
            min=0.0,
        )
        weak_weight = torch.exp(
            -torch.clamp(donor_excess / strength_scale, max=745.0)
        )
        scores = weak_weight * torch.exp(
            -0.5
            * (
                distances
                / float(self.config.structural_rj_merge_distance_sigma_m)
            ).square()
            -0.5
            * (
                response_distance
                / float(self.config.structural_rj_merge_response_sigma)
            ).square()
        )
        score_sums = torch.sum(scores, dim=1, keepdim=True)
        informed = torch.where(
            score_sums > 0.0,
            scores
            / torch.where(
                score_sums > 0.0,
                score_sums,
                torch.ones_like(score_sums),
            ),
            torch.full_like(scores, 1.0 / float(scores.shape[1])),
        )
        uniform = float(self.config.structural_rj_merge_uniform_pair_probability)
        probabilities = (
            (1.0 - uniform) * informed + uniform / float(scores.shape[1])
        )
        probabilities = probabilities / torch.sum(
            probabilities,
            dim=1,
            keepdim=True,
        )
        return donor_columns, receiver_columns, probabilities

    def _continuous_rj_attempt_groups_torch(
        self,
        probability: float,
        *,
        require_nonempty: bool = False,
    ) -> tuple[object, list[tuple[int, object]]]:
        """Draw one attempt mask and return device groups by cardinality."""
        import torch

        state = self._structural_rj_device_state
        if state is None:
            raise RuntimeError("CUDA RJ attempt grouping requires device state.")
        generator = self._continuous_rj_torch_generator_required()
        attempt = torch.rand(
            state["cardinalities"].shape,
            device=state["cardinalities"].device,
            dtype=state["strengths"].dtype,
            generator=generator,
        ) < float(probability)
        if require_nonempty:
            attempt &= state["cardinalities"] > 0
        selected_cardinalities = state["cardinalities"][attempt]
        groups: list[tuple[int, object]] = []
        for value in torch.unique(selected_cardinalities).tolist():
            cardinality = int(value)
            indices = torch.nonzero(
                attempt & (state["cardinalities"] == cardinality),
                as_tuple=False,
            ).reshape(-1)
            groups.append((cardinality, indices))
        return attempt, groups

    def _apply_continuous_rj_birth_death_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply exact birth/death proposals and MH decisions on CUDA."""
        import torch

        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_move_probabilities
        if atlas is None or cardinality_prior is None or move_probabilities is None:
            raise RuntimeError("Continuous RJ priors are unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        generator = self._continuous_rj_torch_generator_required()
        _, groups = self._continuous_rj_attempt_groups_torch(
            self.config.structural_rj_move_probability
        )
        accepted_births = 0
        accepted_deaths = 0
        attempted_births = 0
        attempted_deaths = 0
        constants = self._continuous_rj_atlas_tensors()
        cardinality_log_prior = torch.tensor(
            cardinality_prior.log_probabilities,
            device=constants["chart_probabilities"].device,
            dtype=constants["chart_probabilities"].dtype,
        )
        proposal_probabilities = torch.tensor(
            position_proposal.chart_probabilities,
            device=constants["chart_probabilities"].device,
            dtype=constants["chart_probabilities"].dtype,
        )
        for cardinality, group_indices in groups:
            birth_probability, _ = move_probabilities.probabilities(cardinality)
            birth_move = torch.rand(
                group_indices.shape,
                device=group_indices.device,
                dtype=constants["chart_probabilities"].dtype,
                generator=generator,
            ) < float(birth_probability)
            for is_birth in (True, False):
                selected_indices = group_indices[birth_move == is_birth]
                if int(selected_indices.numel()) == 0:
                    continue
                (
                    indices,
                    chart_ids,
                    surface_uv,
                    positions,
                    strengths,
                ) = self._continuous_rj_group_tensors(
                    selected_indices,
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
                if is_birth:
                    attempted_births += row_count
                    self._continuous_rj_transition_mass_torch(
                        "birth_attempted",
                        indices,
                    )
                    new_chart_ids, new_uv, new_positions = (
                        self._continuous_rj_sample_surface_torch(
                            row_count,
                            chart_probabilities=proposal_probabilities,
                        )
                    )
                    new_strengths = (
                        self._continuous_rj_sample_strength_proposal_torch(
                            new_chart_ids
                        )
                    )
                    proposed_chart_ids = torch.cat(
                        (chart_ids, new_chart_ids[:, None]),
                        dim=1,
                    )
                    proposed_uv = torch.cat(
                        (surface_uv, new_uv[:, None, :]),
                        dim=1,
                    )
                    proposed_positions = torch.cat(
                        (positions, new_positions[:, None, :]),
                        dim=1,
                    )
                    proposed_strengths = torch.cat(
                        (strengths, new_strengths[:, None]),
                        dim=1,
                    )
                    (
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    ) = self._continuous_rj_canonicalize_tensors(
                        proposed_chart_ids,
                        proposed_uv,
                        proposed_positions,
                        proposed_strengths,
                    )
                    log_position_prior = constants["log_chart_probabilities"][
                        new_chart_ids
                    ]
                    log_position_proposal = (
                        self._continuous_rj_position_proposal_log_density_torch(
                            new_chart_ids
                        )
                    )
                    log_strength_prior = (
                        self._continuous_rj_strength_log_prior_torch(new_strengths)
                    )
                    log_strength_proposal = (
                        self._continuous_rj_strength_proposal_log_density_torch(
                            new_chart_ids,
                            new_strengths,
                        )
                    )
                    proposed_cardinality = cardinality + 1
                    prior_delta = (
                        cardinality_log_prior[proposed_cardinality]
                        - cardinality_log_prior[cardinality]
                        + math.log(float(proposed_cardinality))
                        + log_position_prior
                        + log_strength_prior
                    )
                    proposal_delta = (
                        float(
                            move_probabilities.log_probability(
                                "death",
                                proposed_cardinality,
                            )
                        )
                        - math.log(float(proposed_cardinality))
                        - float(
                            move_probabilities.log_probability(
                                "birth",
                                cardinality,
                            )
                        )
                        - log_position_proposal
                        - log_strength_proposal
                    )
                    support = torch.isfinite(prior_delta) & torch.isfinite(
                        proposal_delta
                    )
                    decision = self._continuous_rj_history_tree_decision_torch(
                        data,
                        proposed_positions,
                        proposed_strengths,
                        proposed_chart_ids=proposed_chart_ids,
                        particle_indices=indices,
                        base_log_likelihood=base_ll,
                        log_non_likelihood_ratio=prior_delta + proposal_delta,
                        support=support,
                        target_beta=target_beta,
                        move_family="birth",
                    )
                    proposed_ll = decision.proposed_target_log_likelihood
                    delta_ll = decision.diagnostic_delta_log_likelihood
                    log_ratio = decision.diagnostic_log_acceptance_ratio
                    accepted = decision.accepted
                    self._record_structural_mh_components_torch(
                        "birth",
                        particle_indices=indices,
                        delta_log_likelihood=delta_ll,
                        delta_log_prior=prior_delta,
                        log_reverse_minus_forward=proposal_delta,
                        log_jacobian=0.0,
                        support_feasible=support,
                        accepted=accepted,
                        current_cardinality=cardinality,
                        proposed_cardinality=proposed_cardinality,
                        geometry_support_feasible=torch.isfinite(
                            log_position_prior
                        )
                        & torch.isfinite(log_position_proposal),
                        strength_support_feasible=torch.isfinite(
                            log_strength_prior
                        )
                        & torch.isfinite(log_strength_proposal),
                        log_acceptance_ratio=log_ratio,
                        likelihood_exact=decision.likelihood_exact,
                    )
                    new_matches = (
                        (proposed_chart_ids == new_chart_ids[:, None])
                        & torch.all(proposed_uv == new_uv[:, None, :], dim=2)
                    )
                    new_columns = torch.argmax(new_matches.to(torch.int8), dim=1)
                    self._record_source_events_torch(
                        "source_birth_accepted",
                        positions=proposed_positions,
                        strengths=proposed_strengths,
                        source_columns=new_columns,
                        accepted=accepted,
                        reason="continuous_rj_mh_birth",
                        extras={
                            "delta_ll": delta_ll,
                            "log_acceptance_ratio": log_ratio,
                            "surface_chart_id": new_chart_ids,
                            "surface_uv": new_uv,
                        },
                    )
                    accepted_births += self._commit_continuous_rj_state_tensors(
                        indices,
                        accepted,
                        proposed_chart_ids,
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
                        "birth_accepted",
                        indices,
                        accepted,
                    )
                    continue

                attempted_deaths += row_count
                self._continuous_rj_transition_mass_torch(
                    "death_attempted",
                    indices,
                )
                death_columns = torch.randint(
                    cardinality,
                    (row_count,),
                    device=indices.device,
                    generator=generator,
                )
                rows = torch.arange(row_count, device=indices.device)
                removed_chart_ids = chart_ids[rows, death_columns]
                removed_strengths = strengths[rows, death_columns]
                keep = (
                    torch.arange(cardinality, device=indices.device)[None, :]
                    != death_columns[:, None]
                )
                proposed_chart_ids = chart_ids[keep].reshape(
                    row_count,
                    cardinality - 1,
                )
                proposed_uv = surface_uv[keep].reshape(
                    row_count,
                    cardinality - 1,
                    2,
                )
                proposed_positions = positions[keep].reshape(
                    row_count,
                    cardinality - 1,
                    3,
                )
                proposed_strengths = strengths[keep].reshape(
                    row_count,
                    cardinality - 1,
                )
                log_position_prior = constants["log_chart_probabilities"][
                    removed_chart_ids
                ]
                log_reverse_position = (
                    self._continuous_rj_position_proposal_log_density_torch(
                        removed_chart_ids
                    )
                )
                log_strength_prior = (
                    self._continuous_rj_strength_log_prior_torch(removed_strengths)
                )
                log_reverse_strength = (
                    self._continuous_rj_strength_proposal_log_density_torch(
                        removed_chart_ids,
                        removed_strengths,
                    )
                )
                proposed_cardinality = cardinality - 1
                prior_delta = (
                    cardinality_log_prior[proposed_cardinality]
                    - cardinality_log_prior[cardinality]
                    - math.log(float(cardinality))
                    - log_position_prior
                    - log_strength_prior
                )
                proposal_delta = (
                    float(
                        move_probabilities.log_probability(
                            "birth",
                            proposed_cardinality,
                        )
                    )
                    + math.log(float(cardinality))
                    + log_reverse_position
                    + log_reverse_strength
                    - float(
                        move_probabilities.log_probability(
                            "death",
                            cardinality,
                        )
                    )
                )
                support = torch.isfinite(prior_delta) & torch.isfinite(
                    proposal_delta
                )
                decision = self._continuous_rj_history_tree_decision_torch(
                    data,
                    proposed_positions,
                    proposed_strengths,
                    proposed_chart_ids=proposed_chart_ids,
                    particle_indices=indices,
                    base_log_likelihood=base_ll,
                    log_non_likelihood_ratio=prior_delta + proposal_delta,
                    support=support,
                    target_beta=target_beta,
                    move_family="death",
                )
                proposed_ll = decision.proposed_target_log_likelihood
                delta_ll = decision.diagnostic_delta_log_likelihood
                log_ratio = decision.diagnostic_log_acceptance_ratio
                accepted = decision.accepted
                self._record_structural_mh_components_torch(
                    "death",
                    particle_indices=indices,
                    delta_log_likelihood=delta_ll,
                    delta_log_prior=prior_delta,
                    log_reverse_minus_forward=proposal_delta,
                    log_jacobian=0.0,
                    support_feasible=support,
                    accepted=accepted,
                    current_cardinality=cardinality,
                    proposed_cardinality=proposed_cardinality,
                    geometry_support_feasible=torch.isfinite(
                        log_position_prior
                    )
                    & torch.isfinite(log_reverse_position),
                    strength_support_feasible=torch.isfinite(
                        log_strength_prior
                    )
                    & torch.isfinite(log_reverse_strength),
                    log_acceptance_ratio=log_ratio,
                    likelihood_exact=decision.likelihood_exact,
                )
                self._record_source_events_torch(
                    "source_removed",
                    positions=positions,
                    strengths=strengths,
                    source_columns=death_columns,
                    accepted=accepted,
                    reason="continuous_rj_mh_death",
                    extras={
                        "delta_ll": delta_ll,
                        "log_acceptance_ratio": log_ratio,
                        "surface_chart_id": removed_chart_ids,
                    },
                )
                accepted_deaths += self._commit_continuous_rj_state_tensors(
                    indices,
                    accepted,
                    proposed_chart_ids,
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
                    "death_accepted",
                    indices,
                    accepted,
                )
        self._structural_rj_move_counts.update(
            {
                "birth_attempted": attempted_births,
                "birth_accepted": accepted_births,
                "death_attempted": attempted_deaths,
                "death_accepted": accepted_deaths,
            }
        )
        return accepted_births, accepted_deaths

    def _apply_continuous_rj_global_position_moves_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply exact global position-strength independence moves on CUDA."""
        import torch

        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        position_proposal = self._active_continuous_rj_position_proposal()
        generator = self._continuous_rj_torch_generator_required()
        attempt, groups = self._continuous_rj_attempt_groups_torch(
            self.config.structural_rj_position_move_probability,
            require_nonempty=True,
        )
        attempted_indices = torch.nonzero(attempt, as_tuple=False).reshape(-1)
        self._continuous_rj_transition_mass_torch(
            "global_position_attempted",
            attempted_indices,
        )
        accepted_count = 0
        constants = self._continuous_rj_atlas_tensors()
        proposal_probabilities = torch.tensor(
            position_proposal.chart_probabilities,
            device=constants["chart_probabilities"].device,
            dtype=constants["chart_probabilities"].dtype,
        )
        for cardinality, particle_indices in groups:
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
            source_columns = torch.randint(
                cardinality,
                (row_count,),
                device=indices.device,
                generator=generator,
            )
            rows = torch.arange(row_count, device=indices.device)
            old_chart_ids = chart_ids[rows, source_columns]
            old_strengths = strengths[rows, source_columns]
            new_chart_ids, new_uv, new_positions = (
                self._continuous_rj_sample_surface_torch(
                    row_count,
                    chart_probabilities=proposal_probabilities,
                )
            )
            new_strengths = self._continuous_rj_sample_strength_proposal_torch(
                new_chart_ids
            )
            proposed_chart_ids = chart_ids.clone()
            proposed_uv = surface_uv.clone()
            proposed_positions = positions.clone()
            proposed_strengths = strengths.clone()
            proposed_chart_ids[rows, source_columns] = new_chart_ids
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            proposed_strengths[rows, source_columns] = new_strengths
            (
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            ) = self._continuous_rj_canonicalize_tensors(
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            )
            delta_prior = (
                constants["log_chart_probabilities"][new_chart_ids]
                - constants["log_chart_probabilities"][old_chart_ids]
                + self._continuous_rj_strength_log_prior_torch(new_strengths)
                - self._continuous_rj_strength_log_prior_torch(old_strengths)
            )
            proposal_delta = (
                self._continuous_rj_position_proposal_log_density_torch(
                    old_chart_ids
                )
                - self._continuous_rj_position_proposal_log_density_torch(
                    new_chart_ids
                )
                + self._continuous_rj_strength_proposal_log_density_torch(
                    old_chart_ids,
                    old_strengths,
                )
                - self._continuous_rj_strength_proposal_log_density_torch(
                    new_chart_ids,
                    new_strengths,
                )
            )
            support = torch.isfinite(delta_prior) & torch.isfinite(proposal_delta)
            decision = self._continuous_rj_history_tree_decision_torch(
                data,
                proposed_positions,
                proposed_strengths,
                proposed_chart_ids=proposed_chart_ids,
                particle_indices=indices,
                base_log_likelihood=base_ll,
                log_non_likelihood_ratio=delta_prior + proposal_delta,
                support=support,
                target_beta=target_beta,
                move_family="global_position_strength",
            )
            proposed_ll = decision.proposed_target_log_likelihood
            delta_ll = decision.diagnostic_delta_log_likelihood
            log_ratio = decision.diagnostic_log_acceptance_ratio
            accepted = decision.accepted
            self._record_structural_mh_components_torch(
                "global_position_strength",
                particle_indices=indices,
                delta_log_likelihood=delta_ll,
                delta_log_prior=delta_prior,
                log_reverse_minus_forward=proposal_delta,
                log_jacobian=0.0,
                support_feasible=support,
                accepted=accepted,
                current_cardinality=cardinality,
                proposed_cardinality=cardinality,
                log_acceptance_ratio=log_ratio,
                likelihood_exact=decision.likelihood_exact,
            )
            self._continuous_rj_transition_mass_torch(
                "global_position_accepted",
                indices,
                accepted,
            )
            accepted_count += self._commit_continuous_rj_state_tensors(
                indices,
                accepted,
                proposed_chart_ids,
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
        self._structural_rj_move_counts.update(
            {
                "global_position_attempted": int(attempted_indices.numel()),
                "global_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_local_position_moves_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply coupled multi-scale portal position-strength moves on CUDA."""
        import torch

        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        attempt, groups = self._continuous_rj_attempt_groups_torch(
            self.config.structural_rj_local_position_move_probability,
            require_nonempty=True,
        )
        scales = tuple(
            float(value)
            for value in self.config.structural_rj_local_position_scales_m
        )
        attempted_count = int(torch.count_nonzero(attempt).item())
        accepted_count = 0
        movable_count = 0
        constants = self._continuous_rj_atlas_tensors()
        for cardinality, particle_indices in groups:
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
            rows = torch.arange(row_count, device=indices.device)
            source_columns = torch.randint(
                cardinality,
                (row_count,),
                device=indices.device,
                generator=generator,
            )
            selected_charts = chart_ids[rows, source_columns]
            old_uv = surface_uv[rows, source_columns]
            scale_indices = torch.randint(
                len(scales),
                (row_count,),
                device=indices.device,
                generator=generator,
            )
            new_charts = selected_charts.clone()
            new_uv = old_uv.clone()
            reverse_over_forward = torch.zeros_like(strengths[:, 0])
            for scale_index, sigma_m in enumerate(scales):
                scale_rows = torch.nonzero(
                    scale_indices == scale_index,
                    as_tuple=False,
                ).reshape(-1)
                if int(scale_rows.numel()) == 0:
                    continue
                proposed = atlas.tangent_geodesic_portal_proposal_torch(
                    selected_charts[scale_rows],
                    old_uv[scale_rows],
                    sigma_m=sigma_m,
                    generator=generator,
                )
                new_charts[scale_rows] = proposed[0]
                new_uv[scale_rows] = proposed[1]
                reverse_over_forward[scale_rows] = proposed[2]
            new_positions = self._continuous_rj_positions_torch(
                new_charts,
                new_uv,
            )
            old_strengths = strengths[rows, source_columns]
            combined_response = self._continuous_rj_integrated_unit_response_torch(
                data,
                torch.cat(
                    (positions[rows, source_columns], new_positions),
                    dim=0,
                ),
                torch.cat((selected_charts, new_charts), dim=0),
            )
            old_response, new_response = torch.chunk(combined_response, 2)
            positive_response = (old_response > 0.0) & (new_response > 0.0)
            strength_scale = torch.where(
                positive_response,
                old_response / torch.where(
                    positive_response,
                    new_response,
                    torch.ones_like(new_response),
                ),
                torch.ones_like(old_response),
            )
            new_strengths = old_strengths * strength_scale
            proposed_chart_ids = chart_ids.clone()
            proposed_uv = surface_uv.clone()
            proposed_positions = positions.clone()
            proposed_strengths = strengths.clone()
            proposed_chart_ids[rows, source_columns] = new_charts
            proposed_uv[rows, source_columns] = new_uv
            proposed_positions[rows, source_columns] = new_positions
            proposed_strengths[rows, source_columns] = new_strengths
            (
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            ) = self._continuous_rj_canonicalize_tensors(
                proposed_chart_ids,
                proposed_uv,
                proposed_positions,
                proposed_strengths,
            )
            delta_prior = (
                constants["log_chart_probabilities"][new_charts]
                - constants["log_chart_probabilities"][selected_charts]
                + self._continuous_rj_strength_log_prior_torch(new_strengths)
                - self._continuous_rj_strength_log_prior_torch(old_strengths)
            )
            log_jacobian = torch.log(strength_scale)
            moved = (new_charts != selected_charts) | torch.any(
                new_uv != old_uv,
                dim=1,
            )
            strength_support = self._continuous_rj_strength_support_torch(
                new_strengths
            )
            support = moved & positive_response & strength_support
            movable_count += int(torch.count_nonzero(moved).item())
            decision = self._continuous_rj_history_tree_decision_torch(
                data,
                proposed_positions,
                proposed_strengths,
                proposed_chart_ids=proposed_chart_ids,
                particle_indices=indices,
                base_log_likelihood=base_ll,
                log_non_likelihood_ratio=(
                    delta_prior + reverse_over_forward + log_jacobian
                ),
                support=support,
                target_beta=target_beta,
                move_family="local_position_strength",
            )
            proposed_ll = decision.proposed_target_log_likelihood
            delta_ll = decision.diagnostic_delta_log_likelihood
            log_ratio = decision.diagnostic_log_acceptance_ratio
            accepted = decision.accepted
            self._record_structural_mh_components_torch(
                "local_position_strength",
                particle_indices=indices,
                delta_log_likelihood=delta_ll,
                delta_log_prior=delta_prior,
                log_reverse_minus_forward=reverse_over_forward,
                log_jacobian=log_jacobian,
                support_feasible=support,
                accepted=accepted,
                current_cardinality=cardinality,
                proposed_cardinality=cardinality,
                geometry_support_feasible=moved & positive_response,
                strength_support_feasible=strength_support,
                log_acceptance_ratio=log_ratio,
                likelihood_exact=decision.likelihood_exact,
            )
            accepted_count += self._commit_continuous_rj_state_tensors(
                indices,
                accepted,
                proposed_chart_ids,
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
        self._structural_rj_move_counts.update(
            {
                "local_position_attempted": attempted_count,
                "local_position_movable": movable_count,
                "local_position_accepted": accepted_count,
            }
        )
        return accepted_count

    def _apply_continuous_rj_strength_moves_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> int:
        """Apply exact prior-independence source-strength moves on CUDA."""
        import torch

        attempt, groups = self._continuous_rj_attempt_groups_torch(
            self.config.structural_rj_strength_move_probability,
            require_nonempty=True,
        )
        generator = self._continuous_rj_torch_generator_required()
        accepted_count = 0
        for cardinality, particle_indices in groups:
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
            source_columns = torch.randint(
                cardinality,
                (row_count,),
                device=indices.device,
                generator=generator,
            )
            proposed_strengths = strengths.clone()
            proposed_strengths[
                torch.arange(row_count, device=indices.device),
                source_columns,
            ] = self._continuous_rj_sample_strength_prior_torch((row_count,))
            decision = self._continuous_rj_history_tree_decision_torch(
                data,
                positions,
                proposed_strengths,
                proposed_chart_ids=chart_ids,
                particle_indices=indices,
                base_log_likelihood=base_ll,
                log_non_likelihood_ratio=torch.zeros_like(base_ll),
                support=torch.ones_like(base_ll, dtype=torch.bool),
                target_beta=target_beta,
                move_family="strength",
            )
            proposed_ll = decision.proposed_target_log_likelihood
            accepted = decision.accepted
            accepted_count += self._commit_continuous_rj_state_tensors(
                indices,
                accepted,
                chart_ids,
                surface_uv,
                positions,
                proposed_strengths,
            )
            self._update_continuous_rj_current_log_likelihood_torch(
                indices,
                accepted,
                proposed_ll,
                decision.proposed_station_log_likelihood,
            )
        self._structural_rj_move_counts.update(
            {
                "strength_attempted": int(torch.count_nonzero(attempt).item()),
                "strength_accepted": accepted_count,
            }
        )
        return accepted_count


__all__ = ["StructuralRJTorchBasicMoveMixin"]
