"""Exact-RJ sweep orchestration for the per-isotope particle filter."""

from __future__ import annotations

import time

import numpy as np

from pf.particle_types import StructuralGeometryBatch


class StructuralRJSweepMixin:
    """Coordinate one batched exact-RJ rejuvenation sweep."""

    def _apply_exact_structural_rj_moves(
        self,
        evidence_data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
        current_target_log_likelihood: object | None = None,
        current_station_log_likelihood: object | None = None,
    ) -> None:
        """Apply continuous RJ/MH and always clear the tempered-target context."""
        for name in (
            "mh_acceptance_calls",
            "mh_acceptance_rows",
            "state_scatter_calls",
            "state_scatter_rows",
            "group_gather_calls",
            "tpht_proposal_rows",
            "tpht_refinement_bound_rejected_rows",
            "tpht_first_stage_station_evaluations",
            "tpht_first_stage_rejected_rows",
            "tpht_full_history_rows",
            "tpht_exact_rejected_rows",
            "tpht_exact_station_evaluations",
            "tpht_block_evaluation_calls",
            "tpht_refinement_rounds",
            "tpht_staged_replay_rows",
            "tpht_maximum_block_level",
            "tpht_debug_oracle_rows",
        ):
            self.last_structural_device_diagnostics[name] = 0
        self._structural_rj_tempering_start_row = tempering_start_row
        self._structural_rj_current_block_strength_centers = np.full(
            (
                len(self.continuous_particles),
                self.config.hard_max_sources,
            ),
            float("nan"),
            dtype=np.float64,
        )
        self._structural_rj_current_block_strength_cardinalities = np.full(
            len(self.continuous_particles),
            -1,
            dtype=np.int64,
        )
        self._structural_rj_current_target_log_likelihood = None
        self._structural_rj_current_target_log_likelihood_device = None
        self._structural_rj_current_station_log_likelihood_device = None
        device_input = hasattr(current_target_log_likelihood, "detach")
        if current_target_log_likelihood is not None and not device_input:
            current = np.asarray(
                current_target_log_likelihood,
                dtype=np.float64,
            ).reshape(-1)
            if (
                current.shape != (len(self.continuous_particles),)
                or np.any(np.isnan(current))
                or np.any(np.isposinf(current))
            ):
                raise ValueError(
                    "Current structural target must align with every particle."
                )
            self._structural_rj_current_target_log_likelihood = current.copy()
        self._structural_rj_torch_generator = None
        self._structural_rj_tpht_generator = None
        if self._continuous_rj_torch_enabled():
            import torch

            state = self._structural_rj_device_state
            if state is None:
                raise RuntimeError("Authoritative CUDA RJ state is missing.")
            seed = int(
                self._random_generator.integers(
                    0,
                    np.iinfo(np.int64).max,
                    dtype=np.int64,
                )
            )
            generator = torch.Generator(device=state["strengths"].device)
            generator.manual_seed(seed)
            self._structural_rj_torch_generator = generator
            tpht_generator = torch.Generator(device=state["strengths"].device)
            tpht_generator.manual_seed(seed ^ 0x2F4B6D183A5C7091)
            self._structural_rj_tpht_generator = tpht_generator
            if current_target_log_likelihood is not None:
                if device_input:
                    current_device = current_target_log_likelihood
                    if (
                        not torch.is_tensor(current_device)
                        or current_device.device != state["strengths"].device
                        or current_device.dtype != state["strengths"].dtype
                    ):
                        raise ValueError(
                            "CUDA structural target must already use the "
                            "authoritative state device and dtype."
                        )
                    current_device = current_device.reshape(-1)
                else:
                    current_device = torch.as_tensor(
                        current_target_log_likelihood,
                        device=state["strengths"].device,
                        dtype=state["strengths"].dtype,
                    ).reshape(-1)
                if (
                    tuple(current_device.shape)
                    != (len(self.continuous_particles),)
                    or bool(
                        torch.any(
                            torch.isnan(current_device)
                            | torch.isposinf(current_device)
                        ).item()
                    )
                ):
                    raise ValueError(
                        "CUDA structural target must align with every particle."
                    )
                self._structural_rj_current_target_log_likelihood_device = (
                    current_device.clone()
                )
            if current_station_log_likelihood is not None:
                station_device = torch.as_tensor(current_station_log_likelihood)
                if (
                    not device_input
                    or not torch.is_tensor(station_device)
                    or station_device.device != state["strengths"].device
                    or station_device.dtype != state["strengths"].dtype
                    or station_device.ndim != 2
                    or int(station_device.shape[0])
                    != len(self.continuous_particles)
                    or bool(torch.any(~torch.isfinite(station_device)).item())
                ):
                    raise ValueError(
                        "CUDA TPHT station targets must align with the "
                        "authoritative particle state."
                    )
                self._structural_rj_current_station_log_likelihood_device = (
                    station_device.clone()
                )
            elif self._joint_history_tree_evaluator is not None:
                raise RuntimeError(
                    "Joint production rejuvenation requires TPHT station targets."
                )
            self.last_structural_device_diagnostics["proposal_backend"] = "torch"
        elif device_input:
            raise ValueError(
                "A Torch structural target requires station-authoritative CUDA state."
            )
        try:
            self._apply_exact_structural_rj_moves_impl(
                evidence_data,
                target_beta=target_beta,
            )
        finally:
            current_device = (
                self._structural_rj_current_target_log_likelihood_device
            )
            if current_device is not None:
                self.last_structural_target_log_likelihood_device = (
                    current_device.detach().clone()
                )
                self.last_structural_target_log_likelihood = None
            else:
                current = self._structural_rj_current_target_log_likelihood
                self.last_structural_target_log_likelihood_device = None
                self.last_structural_target_log_likelihood = (
                    None if current is None else current.copy()
                )
            current_station = (
                self._structural_rj_current_station_log_likelihood_device
            )
            self.last_structural_station_log_likelihood_device = (
                None
                if current_station is None
                else current_station.detach().clone()
            )
            self._structural_rj_position_proposal = None
            self._structural_rj_strength_proposal = None
            self._structural_rj_tempering_start_row = None
            self._structural_rj_current_target_log_likelihood = None
            self._structural_rj_current_target_log_likelihood_device = None
            self._structural_rj_current_station_log_likelihood_device = None
            self._structural_rj_torch_generator = None
            self._structural_rj_tpht_generator = None
            self._structural_rj_current_block_strength_centers = None
            self._structural_rj_current_block_strength_cardinalities = None
            self._clear_continuous_rj_device_state()

    def _apply_exact_structural_rj_moves_impl(
        self,
        evidence_data: StructuralGeometryBatch,
        *,
        target_beta: float = 1.0,
    ) -> None:
        """Apply continuous-surface RJ/MH at the requested tempered target."""
        structural_start = time.perf_counter()
        original_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        self._structural_rj_move_counts = {
            "birth_attempted": 0,
            "birth_accepted": 0,
            "death_attempted": 0,
            "death_accepted": 0,
            "global_position_attempted": 0,
            "global_position_accepted": 0,
            "local_position_attempted": 0,
            "local_position_movable": 0,
            "local_position_accepted": 0,
            "strength_attempted": 0,
            "strength_accepted": 0,
            "split_attempted": 0,
            "split_accepted": 0,
            "merge_attempted": 0,
            "merge_accepted": 0,
            "multi_split_attempted": 0,
            "multi_split_accepted": 0,
            "multi_merge_attempted": 0,
            "multi_merge_accepted": 0,
            "block_attempted": 0,
            "block_accepted": 0,
            "block_cardinality_changed": 0,
        }
        self.last_structural_full_support_accepted_mask = np.zeros(
            len(self.continuous_particles),
            dtype=np.bool_,
        )
        self._structural_mh_component_samples = {}
        response_start = time.perf_counter()
        self._structural_rj_position_proposal = (
            self._build_continuous_rj_position_proposal(
                evidence_data,
                target_beta=target_beta,
            )
        )
        response_elapsed = time.perf_counter() - response_start
        birth_count = 0
        death_count = 0
        birth_death_elapsed = 0.0
        if self._variable_cardinality_enabled():
            move_start = time.perf_counter()
            birth_count, death_count = self._apply_continuous_rj_birth_death(
                evidence_data,
                target_beta=target_beta,
            )
            birth_death_elapsed = time.perf_counter() - move_start
        split_merge_start = time.perf_counter()
        split_count = 0
        merge_count = 0
        if self._variable_cardinality_enabled():
            split_count, merge_count = self._apply_continuous_rj_split_merge(
                evidence_data,
                target_beta=target_beta,
            )
        split_merge_elapsed = time.perf_counter() - split_merge_start
        multi_component_start = time.perf_counter()
        multi_split_count = 0
        multi_merge_count = 0
        if self._variable_cardinality_enabled():
            multi_split_count, multi_merge_count = (
                self._apply_continuous_rj_multi_component(
                    evidence_data,
                    target_beta=target_beta,
                )
            )
        multi_component_elapsed = time.perf_counter() - multi_component_start
        block_start = time.perf_counter()
        block_count = 0
        block_cardinality_change_count = 0
        if self._variable_cardinality_enabled():
            (
                block_count,
                block_cardinality_change_count,
            ) = self._apply_continuous_rj_block_independence(
                evidence_data,
                target_beta=target_beta,
            )
        block_elapsed = time.perf_counter() - block_start
        position_start = time.perf_counter()
        position_count = self._apply_continuous_rj_global_position_moves(
            evidence_data,
            target_beta=target_beta,
        )
        position_elapsed = time.perf_counter() - position_start
        local_position_start = time.perf_counter()
        local_position_count = self._apply_continuous_rj_local_position_moves(
            evidence_data,
            target_beta=target_beta,
        )
        local_position_elapsed = time.perf_counter() - local_position_start
        strength_start = time.perf_counter()
        strength_count = self._apply_continuous_rj_strength_moves(
            evidence_data,
            target_beta=target_beta,
        )
        strength_elapsed = time.perf_counter() - strength_start
        current_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        outer_weight_array_equal = bool(
            np.array_equal(original_log_weights, current_log_weights)
        )
        with np.errstate(invalid="ignore"):
            outer_weight_differences = np.where(
                original_log_weights == current_log_weights,
                0.0,
                np.abs(original_log_weights - current_log_weights),
            )
        outer_weight_differences = np.where(
            np.isfinite(outer_weight_differences),
            outer_weight_differences,
            float("inf"),
        )
        outer_weight_max_abs_diff = (
            float(np.max(outer_weight_differences))
            if outer_weight_differences.size
            else 0.0
        )
        position_proposal = self._active_continuous_rj_position_proposal()
        strength_proposal = self._active_continuous_rj_strength_proposal()
        proposal_probabilities = np.asarray(
            position_proposal.chart_probabilities,
            dtype=np.float64,
        )
        proposal_entropy = -float(
            np.sum(
                proposal_probabilities
                * np.log(np.maximum(proposal_probabilities, 1.0e-300)),
                dtype=np.float64,
            )
        )
        self.last_birth_count += int(birth_count)
        self.last_death_count += int(death_count)
        self.last_structural_timing_s = {
            "total": float(time.perf_counter() - structural_start),
            "response_dictionary": float(response_elapsed),
            "rj_birth_death": float(birth_death_elapsed),
            "rj_position": float(position_elapsed),
            "rj_global_position": float(position_elapsed),
            "rj_local_position": float(local_position_elapsed),
            "rj_strength": float(strength_elapsed),
            "rj_split_merge": float(split_merge_elapsed),
            "rj_multi_component": float(multi_component_elapsed),
            "rj_block_independence": float(block_elapsed),
            "target_beta": float(target_beta),
            "rj_birth_attempted": float(
                self._structural_rj_move_counts["birth_attempted"]
            ),
            "rj_birth_accepted": float(birth_count),
            "rj_death_attempted": float(
                self._structural_rj_move_counts["death_attempted"]
            ),
            "rj_death_accepted": float(death_count),
            "rj_global_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_global_position_accepted": float(position_count),
            "rj_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_position_accepted": float(position_count),
            "rj_local_position_attempted": float(
                self._structural_rj_move_counts["local_position_attempted"]
            ),
            "rj_local_position_movable": float(
                self._structural_rj_move_counts["local_position_movable"]
            ),
            "rj_local_position_accepted": float(local_position_count),
            "rj_strength_attempted": float(
                self._structural_rj_move_counts["strength_attempted"]
            ),
            "rj_strength_accepted": float(strength_count),
            "rj_split_attempted": float(
                self._structural_rj_move_counts["split_attempted"]
            ),
            "rj_split_accepted": float(split_count),
            "rj_merge_attempted": float(
                self._structural_rj_move_counts["merge_attempted"]
            ),
            "rj_merge_accepted": float(merge_count),
            "rj_multi_split_attempted": float(
                self._structural_rj_move_counts["multi_split_attempted"]
            ),
            "rj_multi_split_accepted": float(multi_split_count),
            "rj_multi_merge_attempted": float(
                self._structural_rj_move_counts["multi_merge_attempted"]
            ),
            "rj_multi_merge_accepted": float(multi_merge_count),
            "rj_block_attempted": float(
                self._structural_rj_move_counts["block_attempted"]
            ),
            "rj_block_accepted": float(block_count),
            "rj_block_cardinality_changed": float(block_cardinality_change_count),
            "rj_block_attempted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "block_attempted_weight_mass",
                    0.0,
                )
            ),
            "rj_block_accepted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "block_accepted_weight_mass",
                    0.0,
                )
            ),
            "rj_block_cardinality_changed_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "block_cardinality_changed_weight_mass",
                    0.0,
                )
            ),
            "rj_birth_attempted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "birth_attempted_weight_mass",
                    0.0,
                )
            ),
            "rj_birth_accepted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "birth_accepted_weight_mass",
                    0.0,
                )
            ),
            "rj_death_attempted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "death_attempted_weight_mass",
                    0.0,
                )
            ),
            "rj_death_accepted_weight_mass": float(
                self.last_structural_transition_weight_mass.get(
                    "death_accepted_weight_mass",
                    0.0,
                )
            ),
            "rj_position_proposal_prior_weight": float(
                position_proposal.prior_component_probability
            ),
            "rj_position_proposal_data_informative": float(
                position_proposal.data_informative
            ),
            "rj_position_proposal_max_chart_mass": float(
                np.max(proposal_probabilities)
            ),
            "rj_position_proposal_entropy": float(proposal_entropy),
            "rj_strength_proposal_prior_weight": float(
                strength_proposal.prior_component_probability
            ),
            "rj_strength_proposal_data_informative": float(
                strength_proposal.data_informative
            ),
            "rj_strength_proposal_sigma_cps_1m": float(strength_proposal.data_sigma),
            "rj_strength_proposal_location_min_cps_1m": float(
                np.min(strength_proposal.data_locations_by_chart)
            ),
            "rj_strength_proposal_location_max_cps_1m": float(
                np.max(strength_proposal.data_locations_by_chart)
            ),
            "outer_log_weight_max_abs_diff": float(outer_weight_max_abs_diff),
            "outer_log_weight_array_equal": float(outer_weight_array_equal),
            "weights_preserved": float(outer_weight_array_equal),
        }
        device_diagnostics = self.last_structural_device_diagnostics
        diagnostic_names = (
            "mh_acceptance_calls",
            "mh_acceptance_rows",
            "state_scatter_calls",
            "state_scatter_rows",
            "group_gather_calls",
            "tpht_proposal_rows",
            "tpht_refinement_bound_rejected_rows",
            "tpht_first_stage_station_evaluations",
            "tpht_first_stage_rejected_rows",
            "tpht_full_history_rows",
            "tpht_exact_rejected_rows",
            "tpht_exact_station_evaluations",
            "tpht_block_evaluation_calls",
            "tpht_refinement_rounds",
            "tpht_staged_replay_rows",
            "tpht_maximum_block_level",
            "tpht_debug_oracle_rows",
        )
        for name in diagnostic_names:
            self.last_structural_timing_s[f"device_{name}"] = float(
                device_diagnostics.get(name, 0)
            )
        self.last_structural_rejection_diagnostics = (
            self._summarize_structural_mh_components()
        )
        if not outer_weight_array_equal:
            raise RuntimeError("rj_mh rejuvenation must not alter PF weights.")


__all__ = ["StructuralRJSweepMixin"]
