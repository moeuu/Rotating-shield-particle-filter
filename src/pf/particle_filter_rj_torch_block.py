"""Device-resident block-independence kernel for exact-RJ PF."""

from __future__ import annotations

import math

import numpy as np

from pf.particle_types import StructuralGeometryBatch


class StructuralRJTorchBlockIndependenceMixin:
    """Provide a Torch-native full-isotope block-independence move."""

    def _continuous_rj_block_strength_centers_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        chart_ids: object,
        positions: object,
        particle_indices: object,
        target_beta: float,
    ) -> object:
        """Build exact conditional block-strength centers on CUDA."""
        import torch

        if not all(
            torch.is_tensor(value)
            for value in (chart_ids, positions, particle_indices)
        ):
            raise TypeError("CUDA block-strength centers require tensors.")
        row_count, cardinality = chart_ids.shape
        if int(cardinality) < 1:
            raise ValueError("CUDA block-strength centers require nonempty states.")
        scalar_proposal = self._active_continuous_rj_strength_proposal()
        base = torch.tensor(
            scalar_proposal.data_locations_by_chart,
            device=positions.device,
            dtype=positions.dtype,
        )[chart_ids]
        minimum = float(self._strength_prior.minimum)
        excess_floor = np.finfo(np.float64).eps * max(
            1.0,
            float(self._strength_prior.mean),
        )
        excess = torch.clamp(base - minimum, min=excess_floor)
        relative = excess / torch.sum(excess, dim=1, keepdim=True)
        grid_size = int(self.config.structural_rj_strength_proposal_grid_size)
        probabilities = np.linspace(0.005, 0.995, grid_size, dtype=np.float64)
        if self._strength_prior.family == "shifted_gamma":
            from scipy.special import gammaincinv

            total_excess_values = (
                float(self._strength_prior.gamma_scale)
                * gammaincinv(
                    float(cardinality) * float(self._strength_prior.gamma_shape),
                    probabilities,
                )
            )
        else:
            total_excess_values = (
                probabilities
                * float(cardinality)
                * (
                    float(self._strength_prior.maximum)
                    - float(self._strength_prior.minimum)
                )
            )
        total_excess = torch.tensor(
            total_excess_values,
            device=positions.device,
            dtype=positions.dtype,
        )
        candidate_strengths = (
            minimum + relative[:, None, :] * total_excess[None, :, None]
        )
        if self._strength_prior.family == "bounded_uniform":
            candidate_strengths = torch.clamp(
                candidate_strengths,
                max=float(self._strength_prior.maximum),
            )
        candidate_count = int(row_count) * grid_size
        expanded_positions = positions[:, None, :, :].expand(
            -1,
            grid_size,
            -1,
            -1,
        ).reshape(candidate_count, int(cardinality), 3)
        expanded_charts = chart_ids[:, None, :].expand(
            -1,
            grid_size,
            -1,
        ).reshape(candidate_count, int(cardinality))
        expanded_indices = particle_indices[:, None].expand(
            -1,
            grid_size,
        ).reshape(-1)
        conditional_target = (
            self._continuous_rj_recent_proposal_log_likelihood_torch(
                data,
                expanded_positions,
                candidate_strengths.reshape(candidate_count, int(cardinality)),
                chart_ids=expanded_charts,
                particle_indices=expanded_indices,
                target_beta=target_beta,
            )
        ).reshape(int(row_count), grid_size)
        conditional_target = conditional_target + torch.sum(
            self._continuous_rj_strength_log_prior_torch(candidate_strengths),
            dim=2,
        )
        finite = torch.isfinite(conditional_target)
        best_columns = torch.argmax(
            torch.where(
                finite,
                conditional_target,
                torch.full_like(conditional_target, float("-inf")),
            ),
            dim=1,
        )
        centers = candidate_strengths[
            torch.arange(int(row_count), device=positions.device),
            best_columns,
        ]
        valid_rows = torch.any(finite, dim=1)
        return torch.where(valid_rows[:, None], centers, base)

    def _continuous_rj_block_strength_log_density_torch(
        self,
        strengths: object,
        centers: object,
    ) -> object:
        """Evaluate one exact row-level block-mixture density on CUDA."""
        import torch

        if not torch.is_tensor(strengths) or not torch.is_tensor(centers):
            raise TypeError("CUDA block density requires tensor inputs.")
        scalar_proposal = self._active_continuous_rj_strength_proposal()
        row_support = torch.all(
            self._continuous_rj_strength_support_torch(strengths),
            dim=1,
        )
        prior_log = torch.sum(
            self._continuous_rj_strength_log_prior_torch(strengths),
            dim=1,
        )
        probability = float(scalar_proposal.prior_component_probability)
        if probability >= 1.0:
            return torch.where(
                row_support,
                prior_log,
                torch.full_like(prior_log, float("-inf")),
            )
        sigma = float(scalar_proposal.data_sigma)
        lower_cdf = torch.special.ndtr((self._strength_prior.minimum - centers) / sigma)
        if self._strength_prior.family == "bounded_uniform":
            upper_cdf = torch.special.ndtr(
                (self._strength_prior.maximum - centers) / sigma
            )
        else:
            upper_cdf = torch.ones_like(centers)
        standardized = (strengths - centers) / sigma
        data_log = torch.sum(
            -0.5 * standardized.square()
            - np.log(np.sqrt(2.0 * np.pi) * sigma)
            - torch.log(upper_cdf - lower_cdf),
            dim=1,
        )
        mixture = torch.logaddexp(
            math.log(probability) + prior_log,
            math.log1p(-probability) + data_log,
        )
        return torch.where(
            row_support,
            mixture,
            torch.full_like(mixture, float("-inf")),
        )

    def _continuous_rj_sample_block_strength_torch(
        self,
        centers: object,
        *,
        generator: object | None = None,
    ) -> object:
        """Draw one complete block-strength vector per row on CUDA."""
        import torch

        if not torch.is_tensor(centers):
            raise TypeError("CUDA block-strength centers must be a tensor.")
        active_generator = (
            self._continuous_rj_torch_generator_required()
            if generator is None
            else generator
        )
        if not isinstance(active_generator, torch.Generator):
            raise TypeError("CUDA block sampling requires a Torch generator.")
        scalar_proposal = self._active_continuous_rj_strength_proposal()
        result = self._continuous_rj_sample_strength_prior_torch(
            tuple(centers.shape),
            generator=active_generator,
        )
        probability = float(scalar_proposal.prior_component_probability)
        if probability >= 1.0:
            return result
        use_data = torch.rand(
            (int(centers.shape[0]),),
            device=centers.device,
            dtype=centers.dtype,
            generator=active_generator,
        ) >= probability
        sigma = float(scalar_proposal.data_sigma)
        lower_cdf = torch.special.ndtr((self._strength_prior.minimum - centers) / sigma)
        if self._strength_prior.family == "bounded_uniform":
            upper_cdf = torch.special.ndtr(
                (self._strength_prior.maximum - centers) / sigma
            )
        else:
            upper_cdf = torch.ones_like(centers)
        uniforms = lower_cdf + torch.rand(
            centers.shape,
            device=centers.device,
            dtype=centers.dtype,
            generator=active_generator,
        ) * (upper_cdf - lower_cdf)
        eps = torch.finfo(centers.dtype).eps
        sampled = centers + sigma * torch.special.ndtri(
            torch.clamp(uniforms, min=eps, max=1.0 - eps)
        )
        sampled = torch.clamp(sampled, min=self._strength_prior.minimum)
        if self._strength_prior.family == "bounded_uniform":
            sampled = torch.clamp(sampled, max=self._strength_prior.maximum)
        return torch.where(use_data[:, None], sampled, result)

    def _continuous_rj_block_log_prior_torch(
        self,
        chart_ids: object,
        strengths: object,
    ) -> object:
        """Return the ordered-state physical prior density on CUDA."""
        import torch

        cardinality_prior = self._structural_rj_cardinality_prior
        if cardinality_prior is None:
            raise RuntimeError("CUDA block prior requires a cardinality prior.")
        cardinality = int(chart_ids.shape[1])
        constants = self._continuous_rj_atlas_tensors()
        result = torch.full(
            (int(chart_ids.shape[0]),),
            float(cardinality_prior.log_prob(cardinality))
            + math.lgamma(float(cardinality) + 1.0),
            device=strengths.device,
            dtype=strengths.dtype,
        )
        if cardinality:
            result += torch.sum(
                constants["log_chart_probabilities"][chart_ids]
                + self._continuous_rj_strength_log_prior_torch(strengths),
                dim=1,
            )
        return result

    def _apply_continuous_rj_block_independence_torch(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> tuple[int, int]:
        """Apply the exact trans-dimensional block move entirely on CUDA."""
        import torch

        probability = float(self.config.structural_rj_block_independence_probability)
        if probability <= 0.0:
            return 0, 0
        atlas = self._structural_rj_surface_atlas
        cardinality_prior = self._structural_rj_cardinality_prior
        state = self._structural_rj_device_state
        if atlas is None or cardinality_prior is None or state is None:
            raise RuntimeError("CUDA block-RJ priors are unavailable.")
        generator = self._continuous_rj_torch_generator_required()
        attempted_indices = torch.nonzero(
            torch.rand(
                state["cardinalities"].shape,
                device=state["strengths"].device,
                dtype=state["strengths"].dtype,
                generator=generator,
            )
            < probability,
            as_tuple=False,
        ).reshape(-1)
        if int(attempted_indices.numel()) == 0:
            return 0, 0
        self._continuous_rj_transition_mass_torch(
            "block_attempted",
            attempted_indices,
        )
        current_cardinalities = state["cardinalities"][attempted_indices]
        cardinality_probabilities = torch.tensor(
            cardinality_prior.probabilities,
            device=state["strengths"].device,
            dtype=state["strengths"].dtype,
        )
        proposed_cardinalities = torch.multinomial(
            cardinality_probabilities,
            int(attempted_indices.numel()),
            replacement=True,
            generator=generator,
        )
        row_count = int(attempted_indices.numel())
        base_ll = torch.full(
            (row_count,),
            float("-inf"),
            device=state["strengths"].device,
            dtype=state["strengths"].dtype,
        )
        current_log_prior = torch.full_like(base_ll, float("-inf"))
        current_log_proposal = torch.full_like(base_ll, float("-inf"))
        position_proposal = self._active_continuous_rj_position_proposal()
        position_probabilities = torch.tensor(
            position_proposal.chart_probabilities,
            device=state["strengths"].device,
            dtype=state["strengths"].dtype,
        )
        for value in torch.unique(current_cardinalities).tolist():
            cardinality = int(value)
            rows = torch.nonzero(
                current_cardinalities == cardinality,
                as_tuple=False,
            ).reshape(-1)
            particle_indices = attempted_indices[rows]
            indices, charts, _, positions, strengths = (
                self._continuous_rj_group_tensors(
                    particle_indices,
                    cardinality,
                )
            )
            base_ll[rows] = self._continuous_rj_current_log_likelihood_torch(
                data,
                positions,
                strengths,
                chart_ids=charts,
                particle_indices=indices,
                target_beta=target_beta,
            )
            current_log_prior[rows] = self._continuous_rj_block_log_prior_torch(
                charts,
                strengths,
            )
            current_log_proposal[rows] = (
                float(cardinality_prior.log_prob(cardinality))
                + math.lgamma(float(cardinality) + 1.0)
            )
            if cardinality:
                centers = self._continuous_rj_block_strength_centers_torch(
                    data,
                    chart_ids=charts,
                    positions=positions,
                    particle_indices=indices,
                    target_beta=target_beta,
                )
                current_log_proposal[rows] += torch.sum(
                    self._continuous_rj_position_proposal_log_density_torch(
                        charts
                    ),
                    dim=1,
                ) + self._continuous_rj_block_strength_log_density_torch(
                    strengths,
                    centers,
                )

        accepted_count = 0
        cardinality_change_count = 0
        for value in torch.unique(proposed_cardinalities).tolist():
            cardinality = int(value)
            rows = torch.nonzero(
                proposed_cardinalities == cardinality,
                as_tuple=False,
            ).reshape(-1)
            particle_indices = attempted_indices[rows]
            selected_count = int(rows.numel())
            source_count = selected_count * cardinality
            charts_flat, uv_flat, positions_flat = (
                self._continuous_rj_sample_surface_torch(
                    source_count,
                    chart_probabilities=position_probabilities,
                )
            )
            charts = charts_flat.reshape(selected_count, cardinality)
            uv = uv_flat.reshape(selected_count, cardinality, 2)
            positions = positions_flat.reshape(selected_count, cardinality, 3)
            if cardinality:
                centers = self._continuous_rj_block_strength_centers_torch(
                    data,
                    chart_ids=charts,
                    positions=positions,
                    particle_indices=particle_indices,
                    target_beta=target_beta,
                )
                strengths = self._continuous_rj_sample_block_strength_torch(centers)
                strength_log_proposal = (
                    self._continuous_rj_block_strength_log_density_torch(
                        strengths,
                        centers,
                    )
                )
            else:
                strengths = torch.zeros(
                    (selected_count, 0),
                    device=state["strengths"].device,
                    dtype=state["strengths"].dtype,
                )
                strength_log_proposal = torch.zeros(
                    selected_count,
                    device=state["strengths"].device,
                    dtype=state["strengths"].dtype,
                )
            proposed_log_prior = self._continuous_rj_block_log_prior_torch(
                charts,
                strengths,
            )
            proposed_log_proposal = torch.full(
                (selected_count,),
                float(cardinality_prior.log_prob(cardinality))
                + math.lgamma(float(cardinality) + 1.0),
                device=state["strengths"].device,
                dtype=state["strengths"].dtype,
            )
            if cardinality:
                proposed_log_proposal += torch.sum(
                    self._continuous_rj_position_proposal_log_density_torch(
                        charts
                    ),
                    dim=1,
                ) + strength_log_proposal
            delta_prior = proposed_log_prior - current_log_prior[rows]
            proposal_delta = current_log_proposal[rows] - proposed_log_proposal
            support = (
                torch.isfinite(proposed_log_prior)
                & torch.isfinite(proposed_log_proposal)
                & torch.isfinite(current_log_prior[rows])
                & torch.isfinite(current_log_proposal[rows])
            )
            decision = self._continuous_rj_history_tree_decision_torch(
                data,
                positions,
                strengths,
                proposed_chart_ids=charts,
                particle_indices=particle_indices,
                base_log_likelihood=base_ll[rows],
                log_non_likelihood_ratio=delta_prior + proposal_delta,
                support=support,
                target_beta=target_beta,
                move_family="block_independence",
            )
            proposed_ll = decision.proposed_target_log_likelihood
            delta_ll = decision.diagnostic_delta_log_likelihood
            log_ratio = decision.diagnostic_log_acceptance_ratio
            accepted = decision.accepted
            self._record_structural_mh_components_torch(
                "block_independence",
                particle_indices=particle_indices,
                delta_log_likelihood=delta_ll,
                delta_log_prior=delta_prior,
                log_reverse_minus_forward=proposal_delta,
                log_jacobian=0.0,
                support_feasible=support,
                accepted=accepted,
                current_cardinality=current_cardinalities[rows],
                proposed_cardinality=cardinality,
                geometry_support_feasible=(
                    torch.isfinite(proposed_log_proposal)
                    & torch.isfinite(current_log_proposal[rows])
                ),
                strength_support_feasible=(
                    torch.all(
                        self._continuous_rj_strength_support_torch(strengths),
                        dim=1,
                    )
                    if cardinality
                    else torch.ones_like(accepted)
                ),
                log_acceptance_ratio=log_ratio,
                likelihood_exact=decision.likelihood_exact,
            )
            accepted_count += self._commit_continuous_rj_state_tensors(
                particle_indices,
                accepted,
                charts,
                uv,
                positions,
                strengths,
            )
            changed = accepted & (current_cardinalities[rows] != cardinality)
            cardinality_change_count += int(torch.count_nonzero(changed).item())
            self._update_continuous_rj_current_log_likelihood_torch(
                particle_indices,
                accepted,
                proposed_ll,
                decision.proposed_station_log_likelihood,
            )
            self._continuous_rj_transition_mass_torch(
                "block_accepted",
                particle_indices,
                accepted,
            )
            self._continuous_rj_transition_mass_torch(
                "block_cardinality_changed",
                particle_indices,
                changed,
            )
        self._structural_rj_move_counts.update(
            {
                "block_attempted": int(attempted_indices.numel()),
                "block_accepted": accepted_count,
                "block_cardinality_changed": cardinality_change_count,
            }
        )
        return accepted_count, cardinality_change_count


__all__ = ["StructuralRJTorchBlockIndependenceMixin"]
