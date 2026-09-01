"""Joint exact-RJ rejuvenation and tempered SMC algorithms."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from pf.cardinality_policy import (
    HARD_CAP_POSTERIOR_MASS_LIMIT,
    hard_cap_mass_is_acceptable,
)
from pf.estimator_sampling import _stratified_joint_cardinality_draws
from pf.estimator_types import JointStationObservation
from pf.exact_mh import ExactMHDecision, run_exact_mh_acceptance_torch
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    JointRowIdentity,
)
from pf.particle_filter_tempering import TemperingIncrementRequiresRejuvenation
from pf.joint_transport_cache import (
    JointTransportCache,
)
from pf.randomness import named_random_generator
from pf.resampling import systematic_resample
from pf.state import IsotopeState
from pf.structural_rj import (
    ContinuousBlockStrengthProposal,
    shifted_log_strength_random_walk_log_reverse_ratio,
)

if TYPE_CHECKING:
    import torch


_JOINT_GATE_NO_PROGRESS_SWEEP_LIMIT = 8
_JOINT_GATE_NO_PROGRESS_WALL_TIME_S = 900.0


class JointRejuvenationMixin:
    """Provide joint state moves, resampling, mixing, and tempering."""

    @staticmethod
    def _adjacent_cardinality_transition_counts(
        rejection_diagnostics: object,
    ) -> dict[str, int]:
        """Count attempted and accepted adjacent-cardinality proposal rows.

        Structural kernels already summarize their vectorized MH rows by
        current and proposed cardinality. This helper only packages those
        existing summaries; it neither evaluates a target nor changes a move.
        """
        counts = {
            "k_to_k_minus_1_attempted_count": 0,
            "k_to_k_minus_1_accepted_count": 0,
            "k_minus_1_to_k_attempted_count": 0,
            "k_minus_1_to_k_accepted_count": 0,
        }
        if not isinstance(rejection_diagnostics, Mapping):
            return counts
        for raw_move in rejection_diagnostics.values():
            if not isinstance(raw_move, Mapping):
                continue
            transitions = raw_move.get("by_cardinality_transition", {})
            if not isinstance(transitions, Mapping):
                continue
            for transition, raw_summary in transitions.items():
                if not isinstance(raw_summary, Mapping):
                    continue
                parts = str(transition).split("->", maxsplit=1)
                if len(parts) != 2:
                    continue
                try:
                    current = int(parts[0])
                    proposed = int(parts[1])
                except ValueError:
                    continue
                if abs(proposed - current) != 1:
                    continue
                attempted = raw_summary.get("attempted", 0)
                accepted = raw_summary.get("accepted", 0)
                if (
                    isinstance(attempted, (bool, np.bool_))
                    or not isinstance(attempted, (int, np.integer))
                    or int(attempted) < 0
                    or isinstance(accepted, (bool, np.bool_))
                    or not isinstance(accepted, (int, np.integer))
                    or int(accepted) < 0
                    or int(accepted) > int(attempted)
                ):
                    raise RuntimeError(
                        "Structural cardinality-transition counts are invalid."
                    )
                direction = (
                    "k_to_k_minus_1"
                    if proposed == current - 1
                    else "k_minus_1_to_k"
                )
                counts[f"{direction}_attempted_count"] += int(attempted)
                counts[f"{direction}_accepted_count"] += int(accepted)
                transition = f"k_transition_{current}_to_{proposed}"
                attempted_key = f"{transition}_attempted_count"
                accepted_key = f"{transition}_accepted_count"
                counts[attempted_key] = counts.get(attempted_key, 0) + int(
                    attempted
                )
                counts[accepted_key] = counts.get(accepted_key, 0) + int(
                    accepted
                )
        return counts

    def _joint_lineage_recovery_masks(
        self,
        *,
        particle_count: int,
    ) -> dict[str, NDArray[np.bool_]]:
        """Return validated isotope recovery markers aligned to joint rows."""
        raw = self._joint_lineage_recovery_certified_mask_by_isotope
        isotope_order = self.joint_isotope_order()
        if set(raw) != set(isotope_order):
            raise RuntimeError(
                "Lineage-recovery markers do not match the joint isotope set."
            )
        masks: dict[str, NDArray[np.bool_]] = {}
        for isotope in isotope_order:
            value = np.asarray(raw[isotope])
            if value.dtype != np.bool_ or value.shape != (particle_count,):
                raise RuntimeError(
                    "Lineage-recovery markers do not match authenticated PF rows."
                )
            masks[isotope] = value
        return masks

    def _synchronize_joint_lineage_recovery_epoch(
        self,
        cumulative_lineage_ids: NDArray[np.int64],
    ) -> bool:
        """Start one recovery epoch exactly when ancestry first collapses."""
        lineage = np.asarray(cumulative_lineage_ids)
        particle_count = int(self.pf_config.num_particles)
        if (
            lineage.dtype != np.int64
            or lineage.shape != (particle_count,)
            or np.any(lineage < 0)
        ):
            raise RuntimeError(
                "Cumulative lineage IDs do not match authenticated PF rows."
            )
        masks = self._joint_lineage_recovery_masks(
            particle_count=particle_count,
        )
        collapsed = bool(np.unique(lineage).size <= 1)
        active = bool(self._joint_lineage_recovery_active)
        if collapsed and not active:
            self._joint_lineage_recovery_epoch = int(
                self._joint_lineage_recovery_epoch
            ) + 1
            self._joint_lineage_recovery_certified_mask_by_isotope = {
                isotope: np.zeros(particle_count, dtype=np.bool_)
                for isotope in self.joint_isotope_order()
            }
            self._joint_lineage_recovery_active = True
        elif not collapsed:
            if active or any(np.any(mask) for mask in masks.values()):
                raise RuntimeError(
                    "Lineage-recovery provenance is active before ancestry collapse."
                )
        return collapsed

    def _reindex_joint_lineage_recovery_provenance(
        self,
        indices: NDArray[np.int64],
    ) -> None:
        """Propagate recovery descendants through one batched resample."""
        raw_indices = np.asarray(indices)
        particle_count = int(self.pf_config.num_particles)
        if (
            raw_indices.dtype != np.int64
            or raw_indices.shape != (particle_count,)
            or np.any(raw_indices < 0)
            or np.any(raw_indices >= particle_count)
        ):
            raise RuntimeError(
                "Lineage-recovery resampling indices do not match PF rows."
            )
        ancestor_indices = np.ascontiguousarray(raw_indices)
        masks = self._joint_lineage_recovery_masks(
            particle_count=particle_count,
        )
        self._joint_lineage_recovery_certified_mask_by_isotope = {
            isotope: np.ascontiguousarray(mask[ancestor_indices])
            for isotope, mask in masks.items()
        }

    def _record_joint_lineage_recovery_acceptance(
        self,
        *,
        isotope: str,
        accepted_mask: NDArray[np.bool_],
        changed_rows: NDArray[np.int64],
    ) -> None:
        """Certify changed descendants of exact full-support isotope moves."""
        if not bool(self._joint_lineage_recovery_active):
            return
        particle_count = int(self.pf_config.num_particles)
        masks = self._joint_lineage_recovery_masks(
            particle_count=particle_count,
        )
        accepted = np.asarray(accepted_mask)
        if accepted.dtype != np.bool_ or accepted.shape != (particle_count,):
            raise RuntimeError(
                "Full-support acceptance markers do not match joint PF rows."
            )
        rows = np.asarray(changed_rows)
        if (
            rows.dtype != np.int64
            or rows.ndim != 1
            or np.any(rows < 0)
            or np.any(rows >= particle_count)
        ):
            raise RuntimeError(
                "Changed full-support rows do not match authenticated PF rows."
            )
        changed = np.zeros(particle_count, dtype=np.bool_)
        changed[rows] = True
        masks[str(isotope)] |= accepted & changed

    def _joint_lineage_recovery_summary(
        self,
    ) -> dict[str, tuple[int, float]]:
        """Return surviving certified row counts and posterior weight masses."""
        particle_count = int(self.pf_config.num_particles)
        masks = self._joint_lineage_recovery_masks(
            particle_count=particle_count,
        )
        weights = self._strict_joint_particle_weights()
        return {
            isotope: (
                int(np.count_nonzero(mask)),
                float(np.sum(weights[mask], dtype=np.float64)),
            )
            for isotope, mask in masks.items()
        }

    def _joint_packed_strength_state(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return all isotope strengths in the fixed joint slot layout."""
        strengths_by_isotope: list[NDArray[np.float64]] = []
        masks_by_isotope: list[NDArray[np.bool_]] = []
        for isotope in self.joint_isotope_order():
            _, strengths, active_mask, _, _ = self.filters[
                isotope
            ]._packed_continuous_surface_state_arrays()
            strengths_by_isotope.append(np.asarray(strengths, dtype=np.float64))
            masks_by_isotope.append(np.asarray(active_mask, dtype=np.bool_))
        return (
            np.concatenate(strengths_by_isotope, axis=1),
            np.concatenate(masks_by_isotope, axis=1),
        )

    def _joint_packed_strength_state_torch(
        self,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return authoritative joint strength and mask tensors on CUDA."""
        import torch

        strengths: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        device = None
        dtype = None
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            state = filt._structural_rj_device_state
            if (
                state is None
                or not filt._structural_rj_device_state_authoritative
                or not bool(state["strengths"].is_cuda)
            ):
                raise RuntimeError(
                    "Joint CUDA strengths require station-authoritative state."
                )
            if device is None:
                device = state["strengths"].device
                dtype = state["strengths"].dtype
            elif (
                state["strengths"].device != device
                or state["strengths"].dtype != dtype
            ):
                raise RuntimeError("Joint isotope CUDA states do not align.")
            strengths.append(state["strengths"])
            masks.append(state["mask"])
        return torch.cat(strengths, dim=1), torch.cat(masks, dim=1)

    def _joint_torch_generator_required(self) -> "torch.Generator":
        """Return the active sweep-local joint CUDA random generator."""
        import torch

        generator = self._joint_torch_generator
        if not isinstance(generator, torch.Generator):
            raise RuntimeError("Joint CUDA rejuvenation has no random generator.")
        return generator

    def _joint_strength_exact_decision_torch(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_indices: "torch.Tensor",
        scale_ps: "torch.Tensor",
        base_target_log_likelihood: "torch.Tensor",
        log_non_likelihood_ratio: "torch.Tensor",
        support: "torch.Tensor",
        target_beta: float,
    ) -> ExactMHDecision:
        """Evaluate an all-isotope strength proposal with one exact MH test."""
        import torch

        cache = self._joint_structural_transport_cache
        if not isinstance(cache, JointTransportCache) or not torch.is_tensor(
            cache[0]
        ):
            raise RuntimeError(
                "Joint strength exact MH requires the fixed Torch cache."
            )
        rows = particle_indices.to(device=cache[0].device, dtype=torch.long)
        scale = scale_ps.to(device=cache[0].device, dtype=cache[0].dtype)
        base = base_target_log_likelihood.reshape(-1)
        non_likelihood = log_non_likelihood_ratio.reshape(-1)
        feasible = support.to(dtype=torch.bool).reshape(-1)
        row_count = int(rows.numel())
        if (
            row_count <= 0
            or tuple(scale.shape)
            != (row_count, int(cache[0].shape[2]))
            or tuple(base.shape) != (row_count,)
            or tuple(non_likelihood.shape) != (row_count,)
            or tuple(feasible.shape) != (row_count,)
            or cache.station_count != len(stations)
        ):
            raise ValueError("Joint strength exact-MH arrays are not aligned.")
        filt = self.filters[self.joint_isotope_order()[0]]
        feasible_rows = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        proposed_target = torch.full_like(base, float("-inf"))
        proposed_station = torch.index_select(
            cache.station_log_likelihood[:, : cache.station_count],
            0,
            rows,
        ).clone()
        if int(feasible_rows.numel()):
            selected_rows = torch.index_select(rows, 0, feasible_rows)
            selected_scale = torch.index_select(scale, 0, feasible_rows)[
                :, None, :, None
            ]
            selected_target, selected_station = (
                self._joint_history_log_likelihood_torch(
                    filt=filt,
                    stations=stations,
                    total_nvsl=(
                        torch.index_select(cache[0], 0, selected_rows)
                        * selected_scale
                    ),
                    uncollided_nvsl=(
                        torch.index_select(cache[1], 0, selected_rows)
                        * selected_scale
                    ),
                    features_nvslf=torch.index_select(
                        cache[2],
                        0,
                        selected_rows,
                    ),
                    target_beta=float(target_beta),
                    return_station_log_likelihood=True,
                )
            )
            proposed_target.index_copy_(0, feasible_rows, selected_target)
            proposed_station.index_copy_(0, feasible_rows, selected_station)
        decision = run_exact_mh_acceptance_torch(
            current_target_log_likelihood=base,
            proposed_target_log_likelihood=proposed_target,
            proposed_station_log_likelihood=proposed_station,
            log_non_likelihood_ratio=non_likelihood,
            support=feasible,
            generator=self._joint_torch_generator_required(),
        )
        self.last_joint_device_mh_acceptance_calls += 1
        self.last_joint_device_mh_acceptance_rows += row_count
        return decision

    def _commit_joint_strength_block_torch(
        self,
        accepted_rows: "torch.Tensor",
        proposed_strengths_ps: "torch.Tensor",
        old_strengths_ps: "torch.Tensor",
    ) -> None:
        """Commit accepted joint strengths and rescale CUDA cache rows."""
        import torch

        rows = accepted_rows.reshape(-1).to(dtype=torch.long)
        if int(rows.numel()) == 0:
            return
        proposed = proposed_strengths_ps
        old = old_strengths_ps
        slots_per_isotope = int(self.pf_config.cardinality_capacity)
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            filt = self.filters[isotope]
            state = filt._structural_rj_device_state
            if state is None or not filt._structural_rj_device_state_authoritative:
                raise RuntimeError("Accepted joint strengths lost CUDA authority.")
            slot = slice(
                isotope_index * slots_per_isotope,
                (isotope_index + 1) * slots_per_isotope,
            )
            state["strengths"].index_copy_(0, rows, proposed[:, slot])
            filt._structural_rj_device_state_dirty = True
        cache = self._joint_structural_transport_cache
        if cache is None or not torch.is_tensor(cache[0]):
            raise RuntimeError("Joint CUDA strength cache disappeared.")
        scale = torch.where(old > 0.0, proposed / old, torch.ones_like(old))
        scale = scale[:, None, :, None]
        for cached_values in cache[:2]:
            updated = torch.index_select(cached_values, 0, rows) * scale
            cached_values.index_copy_(0, rows, updated)
        self._joint_persistent_structural_transport_cache = cache
        signatures = self._joint_station_cache_signatures(
            self._active_joint_station_history or ()
        )
        if isinstance(cache, JointTransportCache):
            if signatures is None or signatures != cache.station_signatures:
                raise RuntimeError("Joint strength commit lost its station identity.")
            state_sha256 = self._joint_structural_state_sha256()
            cache.invalidate_station_likelihood(rows)
            cache.update_state_identity(
                state_sha256=state_sha256,
                row_generation=self._joint_row_generation,
            )

    def _apply_joint_strength_block_torch(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: "torch.Tensor",
    ) -> "torch.Tensor":
        """Apply the exact all-isotope strength move entirely on CUDA."""
        import torch

        probability = float(self.pf_config.joint_strength_block_probability)
        current_target = current_target_log_likelihood
        if probability <= 0.0:
            return current_target
        current_strengths, active_mask = self._joint_packed_strength_state_torch()
        generator = self._joint_torch_generator_required()
        attempted = (
            torch.rand(
                (int(current_strengths.shape[0]),),
                device=current_strengths.device,
                dtype=current_strengths.dtype,
                generator=generator,
            )
            < probability
        ) & torch.any(active_mask, dim=1)
        rows = torch.nonzero(attempted, as_tuple=False).reshape(-1)
        weights = self._strict_joint_particle_weights()
        rows_host = rows.detach().cpu().numpy().astype(np.int64, copy=False)
        self.last_joint_strength_block_attempted_weight_mass += float(
            np.sum(weights[rows_host], dtype=np.float64)
        )
        if int(rows.numel()) == 0:
            return current_target
        minimum = float(self.pf_config.strength_prior.minimum_cps_1m)
        old = current_strengths[rows]
        mask = active_mask[rows]
        shifted = old - minimum
        valid = torch.all(~mask | (shifted > 0.0), dim=1)
        noise = torch.randn(
            old.shape,
            device=old.device,
            dtype=old.dtype,
            generator=generator,
        ) * float(self.pf_config.joint_strength_block_log_sigma)
        proposed = torch.where(
            mask,
            minimum + shifted * torch.exp(noise),
            old,
        )
        log_prior_ratio = torch.zeros(
            int(rows.numel()),
            device=old.device,
            dtype=old.dtype,
        )
        slots_per_isotope = int(self.pf_config.cardinality_capacity)
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            slot = slice(
                isotope_index * slots_per_isotope,
                (isotope_index + 1) * slots_per_isotope,
            )
            isotope_mask = mask[:, slot]
            filt = self.filters[isotope]
            safe_proposed = torch.where(
                isotope_mask,
                proposed[:, slot],
                torch.full_like(proposed[:, slot], filt._strength_prior.mean),
            )
            safe_old = torch.where(
                isotope_mask,
                old[:, slot],
                torch.full_like(old[:, slot], filt._strength_prior.mean),
            )
            log_prior_ratio += torch.sum(
                torch.where(
                    isotope_mask,
                    filt._continuous_rj_strength_log_prior_torch(safe_proposed)
                    - filt._continuous_rj_strength_log_prior_torch(safe_old),
                    torch.zeros_like(safe_old),
                ),
                dim=1,
            )
        safe_shifted = torch.clamp(
            shifted,
            min=torch.finfo(shifted.dtype).tiny,
        )
        proposed_shifted = proposed - minimum
        log_proposal_ratio = torch.sum(
            torch.where(
                mask,
                torch.log(
                    torch.clamp(
                        proposed_shifted,
                        min=torch.finfo(proposed.dtype).tiny,
                    )
                )
                - torch.log(safe_shifted),
                torch.zeros_like(old),
            ),
            dim=1,
        )
        valid &= torch.isfinite(log_prior_ratio) & torch.isfinite(
            log_proposal_ratio
        )
        scale = torch.where(mask, proposed / old, torch.ones_like(old))
        decision = self._joint_strength_exact_decision_torch(
            stations,
            particle_indices=rows,
            scale_ps=scale,
            base_target_log_likelihood=current_target[rows],
            log_non_likelihood_ratio=(log_prior_ratio + log_proposal_ratio),
            support=valid,
            target_beta=float(target_beta),
        )
        proposed_target = decision.proposed_target_log_likelihood
        accepted = decision.accepted
        accepted_rows = rows[accepted]
        if int(accepted_rows.numel()):
            self._commit_joint_strength_block_torch(
                accepted_rows,
                proposed[accepted],
                old[accepted],
            )
            current_target = current_target.clone()
            current_target[accepted_rows] = proposed_target[accepted]
            cache = self._joint_structural_transport_cache
            if not isinstance(cache, JointTransportCache):
                raise RuntimeError("Joint strength exact-MH cache disappeared.")
            cache.set_station_likelihood(
                decision.proposed_station_log_likelihood[accepted],
                rows=accepted_rows,
            )
            accepted_host = accepted_rows.detach().cpu().numpy().astype(
                np.int64,
                copy=False,
            )
            self.last_joint_strength_block_accepted_weight_mass += float(
                np.sum(weights[accepted_host], dtype=np.float64)
            )
        return current_target

    def _assign_joint_state_rows_torch(
        self,
        candidate: Mapping[str, Mapping[str, "torch.Tensor"]],
        particle_rows: "torch.Tensor",
        *,
        local_rows: "torch.Tensor" | None = None,
    ) -> None:
        """Assign fixed-capacity candidate rows to authoritative CUDA state."""
        import torch

        selected_local = (
            torch.arange(
                int(particle_rows.numel()),
                device=particle_rows.device,
            )
            if local_rows is None
            else local_rows.to(device=particle_rows.device, dtype=torch.long)
        )
        rows = particle_rows[selected_local]
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            state = filt._structural_rj_device_state
            if state is None or not filt._structural_rj_device_state_authoritative:
                raise RuntimeError("Joint state assignment lost CUDA authority.")
            isotope_candidate = candidate[isotope]
            for name in (
                "positions",
                "strengths",
                "mask",
                "chart_ids",
                "surface_uv",
                "cardinalities",
            ):
                value = isotope_candidate[name]
                state[name].index_copy_(
                    0,
                    rows,
                    torch.index_select(value, 0, selected_local),
                )
            filt._structural_rj_device_state_dirty = True

    def _cross_isotope_strength_centers_torch(
        self,
        filt: IsotopeParticleFilter,
        chart_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return the exact cross-isotope block proposal centers on CUDA."""
        import torch

        cardinality = int(chart_ids.shape[1])
        if cardinality < 1:
            raise ValueError("Cross-isotope strength centers require K > 0.")
        proposal = filt._active_continuous_rj_strength_proposal()
        locations = torch.tensor(
            proposal.data_locations_by_chart,
            device=chart_ids.device,
            dtype=filt._structural_rj_device_state["strengths"].dtype,
        )[chart_ids]
        minimum = float(filt._strength_prior.minimum)
        return minimum + (locations - minimum) / float(cardinality)

    def _joint_cross_isotope_exact_decision_torch(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_rows: "torch.Tensor",
        old_states: Mapping[str, Mapping[str, "torch.Tensor"]],
        proposed_states: Mapping[str, Mapping[str, "torch.Tensor"]],
        base_target_log_likelihood: "torch.Tensor",
        log_non_likelihood_ratio: "torch.Tensor",
        support: "torch.Tensor",
        target_beta: float,
    ) -> tuple[
        ExactMHDecision,
        tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
        "torch.Tensor",
    ]:
        """Evaluate a joint state proposal with one exact full-history test."""
        import torch

        cache = self._joint_structural_transport_cache
        if not isinstance(cache, JointTransportCache) or not torch.is_tensor(
            cache[0]
        ):
            raise RuntimeError(
                "Cross-isotope exact MH requires the fixed Torch cache."
            )
        rows = particle_rows.to(device=cache[0].device, dtype=torch.long)
        base = base_target_log_likelihood.reshape(-1)
        non_likelihood = log_non_likelihood_ratio.reshape(-1)
        feasible = support.to(dtype=torch.bool).reshape(-1)
        row_count = int(rows.numel())
        if (
            row_count <= 0
            or tuple(base.shape) != (row_count,)
            or tuple(non_likelihood.shape) != (row_count,)
            or tuple(feasible.shape) != (row_count,)
            or cache.station_count != len(stations)
        ):
            raise ValueError("Cross-isotope exact-MH arrays are not aligned.")
        isotope_order = self.joint_isotope_order()
        replacement_active_slot_mask = torch.cat(
            [proposed_states[isotope]["mask"] for isotope in isotope_order],
            dim=1,
        ).contiguous()
        feasible_local = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        view_count = sum(int(station.fe_indices.size) for station in stations)
        slot_count = int(cache[0].shape[2])
        line_count = int(cache[0].shape[3])
        feature_count = int(cache[2].shape[4])
        replacement = (
            torch.zeros(
                (row_count, view_count, slot_count, line_count),
                device=cache[0].device,
                dtype=cache[0].dtype,
            ),
            torch.zeros(
                (row_count, view_count, slot_count, line_count),
                device=cache[1].device,
                dtype=cache[1].dtype,
            ),
            torch.zeros(
                (
                    row_count,
                    view_count,
                    slot_count,
                    line_count,
                    feature_count,
                ),
                device=cache[2].device,
                dtype=cache[2].dtype,
            ),
        )
        proposed_target = torch.full_like(base, float("-inf"))
        proposed_station = torch.index_select(
            cache.station_log_likelihood[:, : cache.station_count],
            0,
            rows,
        ).clone()
        if int(feasible_local.numel()):
            selected_rows = torch.index_select(rows, 0, feasible_local)
            selected_host = selected_rows.detach().cpu().numpy().astype(
                np.int64,
                copy=False,
            )
            self._assign_joint_state_rows_torch(
                proposed_states,
                rows,
                local_rows=feasible_local,
            )
            try:
                isotope_replacements: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ] = []
                for isotope in isotope_order:
                    station_components = [
                        self._joint_isotope_station_transport_components_torch(
                            station,
                            isotope,
                            particle_indices=selected_host,
                        )
                        for station in stations
                    ]
                    isotope_replacements.append(
                        tuple(
                            torch.cat(
                                [
                                    components[index]
                                    for components in station_components
                                ],
                                dim=1,
                            ).contiguous()
                            for index in range(3)
                        )
                    )
                selected_replacement = tuple(
                    torch.cat(
                        [
                            components[index]
                            for components in isotope_replacements
                        ],
                        dim=2,
                    ).contiguous()
                    for index in range(3)
                )
            finally:
                self._assign_joint_state_rows_torch(
                    old_states,
                    rows,
                    local_rows=feasible_local,
                )
            selected_target, selected_station = (
                self._joint_history_slot_overlay_log_likelihood_torch(
                    filt=self.filters[isotope_order[0]],
                    stations=stations,
                    accepted_total_nvsl=cache[0],
                    accepted_uncollided_nvsl=cache[1],
                    accepted_features_nvslf=cache[2],
                    replacement_total_nvrl=selected_replacement[0],
                    replacement_uncollided_nvrl=selected_replacement[1],
                    replacement_features_nvrlf=selected_replacement[2],
                    particle_indices=selected_rows,
                    slot_start=0,
                    slot_stop=slot_count,
                    replacement_active_slot_mask=torch.index_select(
                        replacement_active_slot_mask,
                        0,
                        feasible_local,
                    ),
                    target_beta=float(target_beta),
                    return_station_log_likelihood=True,
                )
            )
            proposed_target.index_copy_(0, feasible_local, selected_target)
            proposed_station.index_copy_(0, feasible_local, selected_station)
            for destination, selected in zip(
                replacement,
                selected_replacement,
                strict=True,
            ):
                destination.index_copy_(0, feasible_local, selected)
        decision = run_exact_mh_acceptance_torch(
            current_target_log_likelihood=base,
            proposed_target_log_likelihood=proposed_target,
            proposed_station_log_likelihood=proposed_station,
            log_non_likelihood_ratio=non_likelihood,
            support=feasible,
            generator=self._joint_torch_generator_required(),
        )
        self.last_joint_device_mh_acceptance_calls += 1
        self.last_joint_device_mh_acceptance_rows += row_count
        return decision, replacement, replacement_active_slot_mask

    def _apply_joint_cross_isotope_state_block_torch(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: "torch.Tensor",
    ) -> "torch.Tensor":
        """Apply the exact joint isotope-state independence move on CUDA."""
        import torch

        probability = float(
            self.pf_config.joint_cross_isotope_state_block_probability
        )
        current_target = current_target_log_likelihood
        if probability <= 0.0:
            return current_target
        generator = self._joint_torch_generator_required()
        particle_count = int(self.pf_config.num_particles)
        rows = torch.nonzero(
            torch.rand(
                (particle_count,),
                device=current_target.device,
                dtype=current_target.dtype,
                generator=generator,
            )
            < probability,
            as_tuple=False,
        ).reshape(-1)
        weights = self._strict_joint_particle_weights()
        rows_host = rows.detach().cpu().numpy().astype(np.int64, copy=False)
        self.last_joint_cross_isotope_state_attempted_weight_mass += float(
            np.sum(weights[rows_host], dtype=np.float64)
        )
        if int(rows.numel()) == 0:
            return current_target
        isotope_order = self.joint_isotope_order()
        row_count = int(rows.numel())
        old_states: dict[str, dict[str, torch.Tensor]] = {}
        proposed_states: dict[str, dict[str, torch.Tensor]] = {}
        current_log_prior = torch.zeros_like(current_target[rows])
        current_log_proposal = torch.zeros_like(current_log_prior)
        proposed_log_prior = torch.zeros_like(current_log_prior)
        proposed_log_proposal = torch.zeros_like(current_log_prior)
        for isotope in isotope_order:
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            cardinality_prior = filt._structural_rj_cardinality_prior
            state = filt._structural_rj_device_state
            if (
                atlas is None
                or cardinality_prior is None
                or state is None
                or not filt._structural_rj_device_state_authoritative
            ):
                raise RuntimeError(
                    "Joint isotope-state proposals require CUDA surface priors."
                )
            evidence = self._joint_history_structural_geometry(
                isotope,
                stations,
            )
            self._active_joint_structural_geometry = evidence
            try:
                filt._structural_rj_position_proposal = (
                    filt._build_continuous_rj_position_proposal(
                        evidence,
                        target_beta=float(target_beta),
                    )
                )
                old_states[isotope] = {
                    name: torch.index_select(value, 0, rows).clone()
                    for name, value in state.items()
                    if name
                    in {
                        "positions",
                        "strengths",
                        "mask",
                        "chart_ids",
                        "surface_uv",
                        "cardinalities",
                    }
                }
                old_cardinalities = old_states[isotope]["cardinalities"]
                for value in torch.unique(old_cardinalities).tolist():
                    cardinality = int(value)
                    selected = torch.nonzero(
                        old_cardinalities == cardinality,
                        as_tuple=False,
                    ).reshape(-1)
                    charts = old_states[isotope]["chart_ids"][
                        selected,
                        :cardinality,
                    ]
                    strengths = old_states[isotope]["strengths"][
                        selected,
                        :cardinality,
                    ]
                    current_log_prior[selected] += (
                        filt._continuous_rj_block_log_prior_torch(
                            charts,
                            strengths,
                        )
                    )
                    base = (
                        float(cardinality_prior.log_prob(cardinality))
                        + math.lgamma(float(cardinality) + 1.0)
                    )
                    current_log_proposal[selected] += base
                    if cardinality:
                        centers = self._cross_isotope_strength_centers_torch(
                            filt,
                            charts,
                        )
                        current_log_proposal[selected] += torch.sum(
                            filt._continuous_rj_position_proposal_log_density_torch(
                                charts
                            ),
                            dim=1,
                        ) + filt._continuous_rj_block_strength_log_density_torch(
                            strengths,
                            centers,
                        )
                cardinality_probabilities = torch.tensor(
                    cardinality_prior.probabilities,
                    device=current_target.device,
                    dtype=current_target.dtype,
                )
                cardinalities = torch.multinomial(
                    cardinality_probabilities,
                    row_count,
                    replacement=True,
                    generator=generator,
                )
                capacity = int(filt.config.hard_max_sources or 0)
                candidate = {
                    "positions": torch.zeros(
                        (row_count, capacity, 3),
                        device=current_target.device,
                        dtype=current_target.dtype,
                    ),
                    "strengths": torch.zeros(
                        (row_count, capacity),
                        device=current_target.device,
                        dtype=current_target.dtype,
                    ),
                    "mask": torch.zeros(
                        (row_count, capacity),
                        device=current_target.device,
                        dtype=torch.bool,
                    ),
                    "chart_ids": torch.zeros(
                        (row_count, capacity),
                        device=current_target.device,
                        dtype=torch.long,
                    ),
                    "surface_uv": torch.zeros(
                        (row_count, capacity, 2),
                        device=current_target.device,
                        dtype=current_target.dtype,
                    ),
                    "cardinalities": cardinalities.clone(),
                }
                position_proposal = (
                    filt._active_continuous_rj_position_proposal()
                )
                position_probabilities = torch.tensor(
                    position_proposal.chart_probabilities,
                    device=current_target.device,
                    dtype=current_target.dtype,
                )
                for value in torch.unique(cardinalities).tolist():
                    cardinality = int(value)
                    selected = torch.nonzero(
                        cardinalities == cardinality,
                        as_tuple=False,
                    ).reshape(-1)
                    selected_count = int(selected.numel())
                    source_count = selected_count * cardinality
                    if source_count:
                        charts = torch.multinomial(
                            position_probabilities,
                            source_count,
                            replacement=True,
                            generator=generator,
                        ).reshape(selected_count, cardinality)
                        uv = torch.rand(
                            (selected_count, cardinality, 2),
                            device=current_target.device,
                            dtype=current_target.dtype,
                            generator=generator,
                        )
                        positions = filt._continuous_rj_positions_torch(
                            charts,
                            uv,
                        )
                        centers = self._cross_isotope_strength_centers_torch(
                            filt,
                            charts,
                        )
                        strengths = (
                            filt._continuous_rj_sample_block_strength_torch(
                                centers,
                                generator=generator,
                            )
                        )
                        charts, uv, positions, strengths = (
                            filt._continuous_rj_canonicalize_tensors(
                                charts,
                                uv,
                                positions,
                                strengths,
                            )
                        )
                    else:
                        charts = candidate["chart_ids"][selected, :0]
                        uv = candidate["surface_uv"][selected, :0]
                        positions = candidate["positions"][selected, :0]
                        strengths = candidate["strengths"][selected, :0]
                    proposed_log_prior[selected] += (
                        filt._continuous_rj_block_log_prior_torch(
                            charts,
                            strengths,
                        )
                    )
                    base = (
                        float(cardinality_prior.log_prob(cardinality))
                        + math.lgamma(float(cardinality) + 1.0)
                    )
                    proposed_log_proposal[selected] += base
                    if cardinality:
                        centers = self._cross_isotope_strength_centers_torch(
                            filt,
                            charts,
                        )
                        proposed_log_proposal[selected] += torch.sum(
                            filt._continuous_rj_position_proposal_log_density_torch(
                                charts
                            ),
                            dim=1,
                        ) + filt._continuous_rj_block_strength_log_density_torch(
                            strengths,
                            centers,
                        )
                        candidate["positions"][selected, :cardinality] = positions
                        candidate["strengths"][selected, :cardinality] = strengths
                        candidate["mask"][selected, :cardinality] = True
                        candidate["chart_ids"][selected, :cardinality] = charts
                        candidate["surface_uv"][selected, :cardinality] = uv
                proposed_states[isotope] = candidate
            finally:
                self._active_joint_structural_geometry = None
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None

        cache = self._joint_structural_transport_cache
        if not isinstance(cache, JointTransportCache) or not torch.is_tensor(cache[0]):
            raise RuntimeError(
                "Cross-isotope CUDA moves require the fixed exact transport cache."
            )
        delta_prior = proposed_log_prior - current_log_prior
        proposal_ratio = current_log_proposal - proposed_log_proposal
        support = (
            torch.isfinite(current_log_prior)
            & torch.isfinite(current_log_proposal)
            & torch.isfinite(proposed_log_prior)
            & torch.isfinite(proposed_log_proposal)
        )
        decision, exact_replacement, exact_active_mask = (
            self._joint_cross_isotope_exact_decision_torch(
                stations,
                particle_rows=rows,
                old_states=old_states,
                proposed_states=proposed_states,
                base_target_log_likelihood=current_target[rows],
                log_non_likelihood_ratio=delta_prior + proposal_ratio,
                support=support,
                target_beta=float(target_beta),
            )
        )
        proposed_target = decision.proposed_target_log_likelihood
        delta_likelihood = decision.diagnostic_delta_log_likelihood
        log_ratio = decision.diagnostic_log_acceptance_ratio
        accepted = decision.accepted
        accepted_rows = rows[accepted]
        if int(accepted_rows.numel()):
            accepted_local = torch.nonzero(
                accepted,
                as_tuple=False,
            ).reshape(-1)
            self._assign_joint_state_rows_torch(
                proposed_states,
                rows,
                local_rows=accepted_local,
            )
            accepted_host = accepted_rows.detach().cpu().numpy().astype(
                np.int64,
                copy=False,
            )
            accepted_replacement = tuple(
                values[accepted] for values in exact_replacement
            )
            accepted_active_slot_mask = exact_active_mask[accepted]
            cache.replace_slot_rows(
                rows=accepted_rows,
                slot_start=0,
                slot_stop=int(cache[0].shape[2]),
                replacement=accepted_replacement,
                active_slot_mask=accepted_active_slot_mask,
            )
            state_sha256 = self._joint_structural_state_sha256()
            cache.update_state_identity(
                state_sha256=state_sha256,
                row_generation=self._joint_row_generation,
            )
            self._joint_persistent_structural_transport_cache = cache
            current_target = current_target.clone()
            current_target[accepted_rows] = proposed_target[accepted]
            cache.set_station_likelihood(
                decision.proposed_station_log_likelihood[accepted],
                rows=accepted_rows,
            )
            self.last_joint_cross_isotope_state_accepted_weight_mass += float(
                np.sum(weights[accepted_host], dtype=np.float64)
            )
            self._invalidate_posterior_summary_cache()
        diagnostic_payload = torch.stack(
            (
                delta_likelihood,
                delta_prior,
                proposal_ratio,
                log_ratio,
                support.to(dtype=current_target.dtype),
                accepted.to(dtype=current_target.dtype),
            ),
            dim=1,
        ).detach().cpu().numpy()
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )

        def _quantiles(column: int) -> dict[str, float] | None:
            """Return finite quantiles for one transferred diagnostic term."""
            values = np.asarray(
                diagnostic_payload[:, column],
                dtype=np.float64,
            )
            finite = np.isfinite(values)
            if not np.any(finite):
                return None
            return {
                label: float(value)
                for label, value in zip(
                    ("min", "p10", "median", "p90", "max"),
                    np.quantile(values[finite], quantile_levels),
                    strict=True,
                )
            }

        support_host = diagnostic_payload[:, 4] != 0.0
        accepted_mask_host = diagnostic_payload[:, 5] != 0.0
        finite_ratio = np.isfinite(diagnostic_payload[:, 3])
        self.last_joint_cross_isotope_state_rejection_diagnostics = {
            "attempted": row_count,
            "accepted": int(np.count_nonzero(accepted_mask_host)),
            "support_rejected": int(np.count_nonzero(~support_host)),
            "nonfinite_rejected": int(
                np.count_nonzero(support_host & ~finite_ratio)
            ),
            "mh_random_rejected": int(
                np.count_nonzero(
                    support_host
                    & finite_ratio
                    & ~accepted_mask_host
                )
            ),
            "component_quantiles": {
                "delta_log_likelihood": _quantiles(0),
                "delta_log_prior": _quantiles(1),
                "log_reverse_minus_forward": _quantiles(2),
                "log_jacobian": {
                    label: 0.0
                    for label in ("min", "p10", "median", "p90", "max")
                },
                "log_acceptance_ratio": _quantiles(3),
            },
        }
        self._assert_joint_particle_alignment()
        return current_target

    def _joint_strength_block_target(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_indices: NDArray[np.int64],
        scale_ps: NDArray[np.float64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate one strength-only proposal from the accepted GPU cache.

        The immutable unit transport, surface geometry, and line layout do not
        change when strengths move.  Scaling cached source columns is therefore
        exactly equivalent to transport recomputation and avoids new kernels.
        Rows are evaluated in bounded batches to cap temporary GPU memory.
        """
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint strength moves require an active cache.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        scale = np.asarray(scale_ps, dtype=np.float64)
        if scale.shape != (indices.size, int(cache[0].shape[2])):
            raise ValueError("Joint strength scales do not match cache slots.")
        result = np.empty(indices.size, dtype=np.float64)
        batch_size = int(self.pf_config.joint_strength_block_batch_size)
        filt = self.filters[self.joint_isotope_order()[0]]
        cache_is_torch = hasattr(cache[0], "detach")
        for start in range(0, indices.size, batch_size):
            stop = min(start + batch_size, indices.size)
            selected = indices[start:stop]
            selected_scale = scale[start:stop]
            if cache_is_torch:
                import torch

                index_tensor = torch.as_tensor(
                    selected,
                    device=cache[0].device,
                    dtype=torch.long,
                )
                scale_tensor = torch.as_tensor(
                    selected_scale,
                    device=cache[0].device,
                    dtype=cache[0].dtype,
                )[:, None, :, None]
                total = torch.index_select(cache[0], 0, index_tensor) * scale_tensor
                uncollided = (
                    torch.index_select(cache[1], 0, index_tensor) * scale_tensor
                )
                features = torch.index_select(cache[2], 0, index_tensor)
                batch_result = self._joint_history_log_likelihood_torch(
                    filt=filt,
                    stations=stations,
                    total_nvsl=total,
                    uncollided_nvsl=uncollided,
                    features_nvslf=features,
                    target_beta=float(target_beta),
                )
                result[start:stop] = (
                    batch_result.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            else:
                slot_scale = selected_scale[:, None, :, None]
                result[start:stop] = self._joint_history_log_likelihood_numpy(
                    filt=filt,
                    stations=stations,
                    total_nvsl=np.asarray(cache[0][selected]) * slot_scale,
                    uncollided_nvsl=(np.asarray(cache[1][selected]) * slot_scale),
                    features_nvslf=np.asarray(cache[2][selected]),
                    target_beta=float(target_beta),
                )
        return result

    def _joint_active_cache_target(
        self,
        stations: Sequence[JointStationObservation],
        *,
        particle_indices: NDArray[np.int64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate selected rows of the currently active joint GPU cache."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("A joint structural cache is not active.")
        rows = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= int(cache[0].shape[0])):
            raise ValueError("Joint cache particle indices are invalid.")
        filt = self.filters[self.joint_isotope_order()[0]]
        if hasattr(cache[0], "detach"):
            import torch

            selected = torch.as_tensor(
                rows,
                device=cache[0].device,
                dtype=torch.long,
            )
            result = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=cache[0],
                uncollided_nvsl=cache[1],
                features_nvslf=cache[2],
                target_beta=float(target_beta),
                particle_indices=selected,
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=np.asarray(cache[0][rows]),
            uncollided_nvsl=np.asarray(cache[1][rows]),
            features_nvslf=np.asarray(cache[2][rows]),
            target_beta=float(target_beta),
        )

    def _commit_joint_strength_block(
        self,
        accepted_rows: NDArray[np.int64],
        proposed_strengths_ps: NDArray[np.float64],
        old_strengths_ps: NDArray[np.float64],
    ) -> None:
        """Commit accepted joint strengths and rescale their cache rows."""
        rows = np.asarray(accepted_rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            return
        proposed = np.asarray(proposed_strengths_ps, dtype=np.float64)
        old = np.asarray(old_strengths_ps, dtype=np.float64)
        slots_per_isotope = self.pf_config.cardinality_capacity
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            filt = self.filters[isotope]
            cardinalities = filt._continuous_rj_cardinalities_numpy()[rows]
            slot_start = isotope_index * slots_per_isotope
            for cardinality in np.unique(cardinalities).tolist():
                local = np.flatnonzero(cardinalities == int(cardinality))
                isotope_rows = rows[local]
                charts, uv, positions, _ = filt._continuous_rj_group_arrays(
                    isotope_rows,
                    int(cardinality),
                )
                strengths = proposed[
                    local,
                    slot_start : slot_start + int(cardinality),
                ]
                filt._commit_continuous_rj_states(
                    isotope_rows,
                    np.ones(isotope_rows.size, dtype=np.bool_),
                    charts,
                    uv,
                    positions,
                    strengths,
                )
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint strength cache disappeared during commit.")
        scale = np.divide(
            proposed,
            old,
            out=np.ones_like(proposed),
            where=old > 0.0,
        )
        if hasattr(cache[0], "detach"):
            import torch

            index_tensor = torch.as_tensor(
                rows,
                device=cache[0].device,
                dtype=torch.long,
            )
            scale_tensor = torch.as_tensor(
                scale,
                device=cache[0].device,
                dtype=cache[0].dtype,
            )[:, None, :, None]
            for cached_values in cache[:2]:
                updated = (
                    torch.index_select(
                        cached_values,
                        0,
                        index_tensor,
                    )
                    * scale_tensor
                )
                cached_values.index_copy_(0, index_tensor, updated)
        else:
            for cached_values in cache[:2]:
                cached_values[rows] *= scale[:, None, :, None]
        self._joint_persistent_structural_transport_cache = cache
        signatures = self._joint_station_cache_signatures(
            self._active_joint_station_history or ()
        )
        if isinstance(cache, JointTransportCache):
            if signatures is None or signatures != cache.station_signatures:
                raise RuntimeError("Joint strength commit lost its station identity.")
            state_sha256 = self._joint_structural_state_sha256()
            cache.invalidate_station_likelihood(rows)
            cache.update_state_identity(
                state_sha256=state_sha256,
                row_generation=self._joint_row_generation,
            )

    def _joint_mh_acceptance_mask(
        self,
        log_ratio: NDArray[np.float64],
        *,
        rng: np.random.Generator,
        support: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Draw a joint MH mask on CUDA when the active cache is CUDA-backed."""
        ratios = np.asarray(log_ratio, dtype=np.float64).reshape(-1)
        with np.errstate(divide="ignore"):
            log_uniforms = np.log(rng.random(ratios.size))
        thresholds = np.minimum(ratios, 0.0)
        feasible = None
        if support is not None:
            feasible = np.asarray(support, dtype=np.bool_).reshape(-1)
            if feasible.size != ratios.size:
                raise ValueError("Joint MH support must align with log_ratio.")
        cache = self._joint_structural_transport_cache
        if (
            cache is None
            or not hasattr(cache[0], "detach")
            or not bool(cache[0].is_cuda)
        ):
            accepted = log_uniforms < thresholds
            if feasible is not None:
                accepted &= feasible
            return np.asarray(accepted, dtype=np.bool_)
        import torch

        accepted_tensor = torch.as_tensor(
            log_uniforms,
            device=cache[0].device,
            dtype=torch.float64,
        ) < torch.as_tensor(
            thresholds,
            device=cache[0].device,
            dtype=torch.float64,
        )
        if feasible is not None:
            accepted_tensor &= torch.as_tensor(
                feasible,
                device=cache[0].device,
                dtype=torch.bool,
            )
        self.last_joint_device_mh_acceptance_calls += 1
        self.last_joint_device_mh_acceptance_rows += int(ratios.size)
        return accepted_tensor.detach().cpu().numpy().astype(np.bool_, copy=False)

    def _apply_joint_strength_block(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: object,
    ) -> object:
        """Apply one exact all-isotope strength move without conservation.

        Each active source receives a symmetric Gaussian increment in shifted
        log-strength coordinates.  Isotope totals are not constrained or
        transferred.  The shared spectrum likelihood alone decides whether a
        simultaneous decrease in one isotope and increase in another is useful.
        """
        if hasattr(current_target_log_likelihood, "detach"):
            return self._apply_joint_strength_block_torch(
                stations,
                target_beta=target_beta,
                current_target_log_likelihood=current_target_log_likelihood,
            )
        probability = float(self.pf_config.joint_strength_block_probability)
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        if probability <= 0.0:
            return current_target
        current_strengths, active_mask = self._joint_packed_strength_state()
        rng = self._joint_random_generator
        attempted = (rng.random(current_strengths.shape[0]) < probability) & np.any(
            active_mask, axis=1
        )
        rows = np.flatnonzero(attempted).astype(np.int64, copy=False)
        weights = self._strict_joint_particle_weights()
        self.last_joint_strength_block_attempted_weight_mass += float(
            np.sum(weights[rows])
        )
        if rows.size == 0:
            return current_target
        minimum = float(self.pf_config.strength_prior.minimum_cps_1m)
        old = current_strengths[rows]
        mask = active_mask[rows]
        shifted = old - minimum
        valid = np.all(~mask | (shifted > 0.0), axis=1)
        noise = rng.normal(
            loc=0.0,
            scale=float(self.pf_config.joint_strength_block_log_sigma),
            size=old.shape,
        )
        proposed = old.copy()
        proposed[mask] = minimum + shifted[mask] * np.exp(noise[mask])
        log_prior_ratio = np.zeros(rows.size, dtype=np.float64)
        slots_per_isotope = self.pf_config.cardinality_capacity
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            slot = slice(
                isotope_index * slots_per_isotope,
                (isotope_index + 1) * slots_per_isotope,
            )
            isotope_mask = mask[:, slot]
            prior = self.filters[isotope]._strength_prior
            safe_proposed = np.where(
                isotope_mask,
                proposed[:, slot],
                prior.mean,
            )
            safe_old = np.where(
                isotope_mask,
                old[:, slot],
                prior.mean,
            )
            log_prior_ratio += np.sum(
                np.where(
                    isotope_mask,
                    np.asarray(prior.log_prob(safe_proposed))
                    - np.asarray(prior.log_prob(safe_old)),
                    0.0,
                ),
                axis=1,
                dtype=np.float64,
            )
        log_proposal_ratio = shifted_log_strength_random_walk_log_reverse_ratio(
            old,
            proposed,
            mask,
            minimum_strength=minimum,
        )
        valid &= np.isfinite(log_prior_ratio) & np.isfinite(log_proposal_ratio)
        scale = np.ones_like(old)
        scale[mask] = proposed[mask] / old[mask]
        proposed_target = np.full(rows.size, float("-inf"), dtype=np.float64)
        valid_local = np.flatnonzero(valid)
        if valid_local.size:
            proposed_target[valid_local] = self._joint_strength_block_target(
                stations,
                particle_indices=rows[valid_local],
                scale_ps=scale[valid_local],
                target_beta=float(target_beta),
            )
        log_ratio = (
            proposed_target
            - current_target[rows]
            + log_prior_ratio
            + log_proposal_ratio
        )
        accepted = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
            support=valid,
        )
        accepted_rows = rows[accepted]
        if accepted_rows.size:
            self._commit_joint_strength_block(
                accepted_rows,
                proposed[accepted],
                old[accepted],
            )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted]
            self.last_joint_strength_block_accepted_weight_mass += float(
                np.sum(weights[accepted_rows])
            )
        return current_target

    def _apply_joint_cross_isotope_state_block(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: object,
    ) -> object:
        """Jointly propose independent isotope states without strength transfer.

        Each isotope retains its own cardinality, surface-position, and strength
        priors. The proposal is coupled only by making one simultaneous MH
        decision under the shared full-spectrum likelihood; no activity or
        detector-cps quantity is conserved between isotope labels.
        """
        if hasattr(current_target_log_likelihood, "detach"):
            return self._apply_joint_cross_isotope_state_block_torch(
                stations,
                target_beta=target_beta,
                current_target_log_likelihood=current_target_log_likelihood,
            )
        probability = float(self.pf_config.joint_cross_isotope_state_block_probability)
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        if probability <= 0.0:
            return current_target
        particle_count = int(self.pf_config.num_particles)
        rng = self._joint_random_generator
        rows = np.flatnonzero(rng.random(particle_count) < probability).astype(
            np.int64, copy=False
        )
        weights = self._strict_joint_particle_weights()
        self.last_joint_cross_isotope_state_attempted_weight_mass += float(
            np.sum(weights[rows], dtype=np.float64)
        )
        if rows.size == 0:
            return current_target
        isotope_order = self.joint_isotope_order()
        old_states: dict[str, dict[str, NDArray[Any]]] = {}
        proposed_states: dict[str, dict[str, NDArray[Any]]] = {}
        current_log_prior = np.zeros(rows.size, dtype=np.float64)
        current_log_proposal = np.zeros(rows.size, dtype=np.float64)
        proposed_log_prior = np.zeros(rows.size, dtype=np.float64)
        proposed_log_proposal = np.zeros(rows.size, dtype=np.float64)
        for isotope in isotope_order:
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            cardinality_prior = filt._structural_rj_cardinality_prior
            if atlas is None or cardinality_prior is None:
                raise RuntimeError(
                    "Joint isotope-state proposals require exact surface priors."
                )
            evidence = self._joint_history_structural_geometry(
                isotope,
                stations,
            )
            self._active_joint_structural_geometry = evidence
            try:
                filt._structural_rj_position_proposal = (
                    filt._build_continuous_rj_position_proposal(
                        evidence,
                        target_beta=float(target_beta),
                    )
                )
                (
                    packed_positions,
                    packed_strengths,
                    packed_mask,
                    packed_charts,
                    packed_uv,
                ) = filt._packed_continuous_surface_state_arrays()
                old_cardinalities = np.sum(
                    packed_mask[rows],
                    axis=1,
                    dtype=np.int64,
                )
                old_states[isotope] = {
                    "cardinalities": old_cardinalities.copy(),
                    "chart_ids": np.ascontiguousarray(packed_charts[rows]),
                    "surface_uv": np.ascontiguousarray(packed_uv[rows]),
                    "positions": np.ascontiguousarray(packed_positions[rows]),
                    "strengths": np.ascontiguousarray(packed_strengths[rows]),
                }
                for cardinality in np.unique(old_cardinalities).tolist():
                    selected = np.flatnonzero(old_cardinalities == int(cardinality))
                    charts = packed_charts[
                        rows[selected],
                        : int(cardinality),
                    ]
                    strengths = packed_strengths[
                        rows[selected],
                        : int(cardinality),
                    ]
                    prior, proposal = filt._continuous_rj_block_log_densities(
                        charts,
                        strengths,
                    )
                    current_log_prior[selected] += prior
                    if int(cardinality) == 0:
                        current_log_proposal[selected] += proposal
                    else:
                        scalar_strength_proposal = (
                            filt._active_continuous_rj_strength_proposal()
                        )
                        minimum_strength = float(filt._strength_prior.minimum)
                        conditional_locations = minimum_strength + (
                            scalar_strength_proposal.data_locations_by_chart[charts]
                            - minimum_strength
                        ) / float(cardinality)
                        strength_block = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(scalar_strength_proposal.data_sigma),
                            prior_component_probability=float(
                                scalar_strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        current_log_proposal[selected] += (
                            float(cardinality_prior.log_prob(int(cardinality)))
                            + math.lgamma(float(cardinality) + 1.0)
                            + np.sum(
                                filt._active_continuous_rj_position_proposal()
                                .log_density(charts),
                                axis=1,
                            )
                            + strength_block.log_density(strengths)
                        )
                cardinalities = rng.choice(
                    cardinality_prior.probabilities.size,
                    size=rows.size,
                    replace=True,
                    p=cardinality_prior.probabilities,
                ).astype(np.int64, copy=False)
                capacity = int(filt.config.hard_max_sources or 0)
                proposed_chart_ids = np.zeros(
                    (rows.size, capacity),
                    dtype=np.int64,
                )
                proposed_surface_uv = np.zeros(
                    (rows.size, capacity, 2),
                    dtype=np.float64,
                )
                proposed_positions = np.zeros(
                    (rows.size, capacity, 3),
                    dtype=np.float64,
                )
                proposed_strengths = np.zeros(
                    (rows.size, capacity),
                    dtype=np.float64,
                )
                for cardinality in np.unique(cardinalities).tolist():
                    selected = np.flatnonzero(cardinalities == int(cardinality))
                    source_count = int(selected.size) * int(cardinality)
                    chart_ids, surface_uv, _ = atlas.sample(
                        source_count,
                        rng=rng,
                        chart_probabilities=(
                            filt._active_continuous_rj_position_proposal()
                            .chart_probabilities
                        ),
                    )
                    charts_batch = chart_ids.reshape(
                        selected.size,
                        int(cardinality),
                    )
                    uv_batch = surface_uv.reshape(
                        selected.size,
                        int(cardinality),
                        2,
                    )
                    if int(cardinality) == 0:
                        strength_batch = np.empty(
                            (selected.size, 0),
                            dtype=np.float64,
                        )
                        prior, proposal = filt._continuous_rj_block_log_densities(
                            charts_batch,
                            strength_batch,
                        )
                        proposed_log_prior[selected] += prior
                        proposed_log_proposal[selected] += proposal
                    else:
                        scalar_strength_proposal = (
                            filt._active_continuous_rj_strength_proposal()
                        )
                        minimum_strength = float(filt._strength_prior.minimum)
                        conditional_locations = minimum_strength + (
                            scalar_strength_proposal.data_locations_by_chart[
                                charts_batch
                            ]
                            - minimum_strength
                        ) / float(cardinality)
                        strength_block = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(scalar_strength_proposal.data_sigma),
                            prior_component_probability=float(
                                scalar_strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        strength_batch = strength_block.sample(rng=rng)
                        prior, _ = filt._continuous_rj_block_log_densities(
                            charts_batch,
                            strength_batch,
                        )
                        proposed_log_prior[selected] += prior
                        proposed_log_proposal[selected] += (
                            float(cardinality_prior.log_prob(int(cardinality)))
                            + math.lgamma(float(cardinality) + 1.0)
                            + np.sum(
                                filt._active_continuous_rj_position_proposal()
                                .log_density(charts_batch),
                                axis=1,
                            )
                            + strength_block.log_density(strength_batch)
                        )
                    positions_batch = atlas.positions_xyz(charts_batch, uv_batch)
                    canonical = filt._continuous_rj_canonicalize_rows(
                        charts_batch,
                        uv_batch,
                        positions_batch,
                        strength_batch,
                    )
                    if int(cardinality):
                        proposed_chart_ids[
                            selected,
                            : int(cardinality),
                        ] = canonical[0]
                        proposed_surface_uv[
                            selected,
                            : int(cardinality),
                        ] = canonical[1]
                        proposed_positions[
                            selected,
                            : int(cardinality),
                        ] = canonical[2]
                        proposed_strengths[
                            selected,
                            : int(cardinality),
                        ] = canonical[3]
                proposed_states[isotope] = {
                    "cardinalities": cardinalities,
                    "chart_ids": proposed_chart_ids,
                    "surface_uv": proposed_surface_uv,
                    "positions": proposed_positions,
                    "strengths": proposed_strengths,
                }
            finally:
                self._active_joint_structural_geometry = None
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None

        def _assign(
            candidate: Mapping[str, Mapping[str, NDArray[Any]]],
            local_rows: NDArray[np.int64] | None = None,
        ) -> None:
            """Assign fixed-capacity candidate rows without Python particles."""
            selected_local = (
                np.arange(rows.size, dtype=np.int64)
                if local_rows is None
                else np.asarray(local_rows, dtype=np.int64).reshape(-1)
            )
            for isotope in isotope_order:
                filt = self.filters[isotope]
                state = candidate[isotope]
                cardinalities = np.asarray(
                    state["cardinalities"],
                    dtype=np.int64,
                )
                for cardinality in np.unique(
                    cardinalities[selected_local]
                ).tolist():
                    local = selected_local[
                        cardinalities[selected_local] == int(cardinality)
                    ]
                    filt._commit_continuous_rj_states(
                        rows[local],
                        np.ones(local.size, dtype=np.bool_),
                        np.asarray(state["chart_ids"])[
                            local,
                            : int(cardinality),
                        ],
                        np.asarray(state["surface_uv"])[
                            local,
                            : int(cardinality),
                        ],
                        np.asarray(state["positions"])[
                            local,
                            : int(cardinality),
                        ],
                        np.asarray(state["strengths"])[
                            local,
                            : int(cardinality),
                        ],
                    )

        _assign(proposed_states)
        try:
            for isotope in isotope_order:
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=rows,
                )
            proposed_target = self._joint_active_cache_target(
                stations,
                particle_indices=rows,
                target_beta=float(target_beta),
            )
        finally:
            _assign(old_states)
            for isotope in isotope_order:
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=rows,
                )
        delta_likelihood = np.full(rows.size, float("-inf"), dtype=np.float64)
        proposed_finite = np.isfinite(proposed_target)
        current_finite = np.isfinite(current_target[rows])
        both_finite = proposed_finite & current_finite
        delta_likelihood[both_finite] = (
            proposed_target[both_finite] - current_target[rows][both_finite]
        )
        delta_likelihood[proposed_finite & ~current_finite] = float("inf")
        delta_prior = proposed_log_prior - current_log_prior
        proposal_ratio = current_log_proposal - proposed_log_proposal
        support = (
            np.isfinite(current_log_prior)
            & np.isfinite(current_log_proposal)
            & np.isfinite(proposed_log_prior)
            & np.isfinite(proposed_log_proposal)
        )
        log_ratio = delta_likelihood + delta_prior + proposal_ratio
        accepted = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
            support=support,
        )
        accepted_rows = rows[accepted]
        if accepted_rows.size:
            accepted_local = np.flatnonzero(accepted).astype(
                np.int64,
                copy=False,
            )
            _assign(proposed_states, accepted_local)
            for isotope in isotope_order:
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope,
                    particle_indices=accepted_rows,
                )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted]
            self.last_joint_cross_isotope_state_accepted_weight_mass += float(
                np.sum(weights[accepted_rows], dtype=np.float64)
            )
            self._invalidate_posterior_summary_cache()
        finite_ratio = np.isfinite(log_ratio)
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )

        def _quantiles(values: NDArray[np.float64]) -> dict[str, float] | None:
            """Return finite rejection-diagnostic quantiles for one MH term."""
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            finite = np.isfinite(array)
            if not np.any(finite):
                return None
            return {
                label: float(value)
                for label, value in zip(
                    ("min", "p10", "median", "p90", "max"),
                    np.quantile(array[finite], quantile_levels),
                    strict=True,
                )
            }

        self.last_joint_cross_isotope_state_rejection_diagnostics = {
            "attempted": int(rows.size),
            "accepted": int(accepted_rows.size),
            "support_rejected": int(np.count_nonzero(~support)),
            "nonfinite_rejected": int(np.count_nonzero(support & ~finite_ratio)),
            "mh_random_rejected": int(
                np.count_nonzero(support & finite_ratio & ~accepted)
            ),
            "component_quantiles": {
                "delta_log_likelihood": _quantiles(delta_likelihood),
                "delta_log_prior": _quantiles(delta_prior),
                "log_reverse_minus_forward": _quantiles(proposal_ratio),
                "log_jacobian": {
                    label: 0.0 for label in ("min", "p10", "median", "p90", "max")
                },
                "log_acceptance_ratio": _quantiles(log_ratio),
            },
        }
        self._assert_joint_particle_alignment()
        return current_target

    def _resample_joint_particles(
        self,
        normalized_log_weights: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Apply one systematic ancestor vector to every isotope state row."""
        log_weights = np.asarray(
            normalized_log_weights,
            dtype=np.float64,
        ).reshape(-1)
        self._assign_joint_log_weights(log_weights)
        raw_indices = np.asarray(
            systematic_resample(
                log_weights,
                rng=self._joint_random_generator,
            )
        )
        if (
            raw_indices.shape != log_weights.shape
            or not np.issubdtype(raw_indices.dtype, np.integer)
            or np.any(raw_indices < 0)
            or np.any(raw_indices >= log_weights.size)
        ):
            raise RuntimeError(
                "Joint systematic resampling returned invalid ancestor indices."
            )
        indices = np.asarray(
            raw_indices,
            dtype=np.int64,
        )
        uniform_log_weight = float(-np.log(max(indices.size, 1)))
        reference_particles = self.filters[
            self.joint_isotope_order()[0]
        ].continuous_particles
        parent_identities = tuple(
            particle.joint_row_identity for particle in reference_particles
        )
        if not all(
            isinstance(identity, JointRowIdentity) for identity in parent_identities
        ):
            raise RuntimeError(
                "Joint resampling requires authenticated parent row identities."
            )
        new_identities = tuple(
            parent_identities[int(index)].resampled_child(ordinal=row)
            for row, index in enumerate(indices.tolist())
        )
        persistent_cache = self._joint_persistent_structural_transport_cache
        if isinstance(persistent_cache, JointTransportCache):
            persistent_cache.stage_reindex_rows(indices)
        new_particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            filt._reindex_continuous_rj_device_state(indices)
            previous = filt.continuous_particles
            new_particles_by_isotope[isotope] = [
                IsotopeParticle(
                    state=previous[int(index)].state.copy(),
                    log_weight=uniform_log_weight,
                    joint_row_identity=new_identities[row],
                )
                for row, index in enumerate(indices.tolist())
            ]
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            filt.continuous_particles = new_particles_by_isotope[isotope]
            filt.last_resample_count += 1
            filt._resample_count_in_observation += 1
        self._reindex_joint_lineage_recovery_provenance(indices)
        new_generation = new_identities[0].generation
        if isinstance(persistent_cache, JointTransportCache):
            persistent_cache.commit_staged_reindex()
            self._joint_persistent_structural_transport_cache = persistent_cache
            self._joint_structural_transport_cache = persistent_cache
            self.last_joint_persistent_cache_reindex_count += 1
        elif persistent_cache is not None:
            # Explicit tuple caches exist only in small deterministic test
            # oracles. Production refresh always creates JointTransportCache.
            if hasattr(persistent_cache[0], "detach"):
                import torch

                index_tensor = torch.as_tensor(
                    indices,
                    device=persistent_cache[0].device,
                    dtype=torch.long,
                )
                reindexed_cache = tuple(
                    torch.index_select(values, 0, index_tensor).contiguous()
                    for values in persistent_cache
                )
            else:
                reindexed_cache = tuple(
                    np.ascontiguousarray(values[indices]) for values in persistent_cache
                )
            self._joint_persistent_structural_transport_cache = reindexed_cache
            self._joint_structural_transport_cache = reindexed_cache
            self.last_joint_persistent_cache_reindex_count += 1
        self._joint_row_generation = new_generation
        self.last_joint_resample_indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        if isinstance(
            self._joint_persistent_structural_transport_cache,
            JointTransportCache,
        ):
            state_sha256 = self._joint_structural_state_sha256()
            self._joint_persistent_structural_transport_cache.update_state_identity(
                state_sha256=state_sha256,
                row_generation=new_generation,
            )
        self._invalidate_posterior_summary_cache()
        self._assert_joint_particle_alignment()
        return self.last_joint_resample_indices

    def _joint_mixing_snapshot(self) -> dict[str, object]:
        """Capture batched joint states for one exact rejuvenation diagnostic."""
        weights = self._strict_joint_particle_weights()
        state_blocks: list[NDArray[np.float64]] = []
        cardinality_columns: list[NDArray[np.int64]] = []
        isotope_states: dict[str, dict[str, NDArray[Any]]] = {}
        transition_mass: dict[str, float] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            positions, strengths, mask, chart_ids, surface_uv = (
                filt._packed_continuous_surface_state_arrays()
            )
            cardinalities = np.sum(mask, axis=1, dtype=np.int64)
            cardinality_columns.append(cardinalities)
            state_blocks.extend(
                (
                    chart_ids.astype(np.float64, copy=False),
                    surface_uv.reshape(surface_uv.shape[0], -1),
                    strengths,
                    mask.astype(np.float64, copy=False),
                )
            )
            isotope_states[isotope] = {
                "positions": positions,
                "strengths": strengths,
                "mask": mask,
                "cardinalities": cardinalities,
            }
            for key, value in filt.last_structural_transition_weight_mass.items():
                transition_mass[f"{isotope}.{key}"] = float(value)
        state_matrix = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(block, dtype=np.float64).reshape(
                        weights.size,
                        -1,
                    )
                    for block in state_blocks
                ],
                axis=1,
            ),
            dtype=np.float64,
        )
        cardinality_matrix = np.stack(cardinality_columns, axis=1)
        return {
            "weights": weights,
            "state_matrix": state_matrix,
            "cardinality_matrix": cardinality_matrix,
            "isotope_states": isotope_states,
            "transition_mass": transition_mass,
        }

    @staticmethod
    def _joint_isotope_cache_state(
        filt: IsotopeParticleFilter,
    ) -> tuple[Any, ...]:
        """Copy compact row state used to find exact accepted cache deltas."""
        device_state = filt._structural_rj_device_state
        if (
            device_state is not None
            and filt._structural_rj_device_state_authoritative
        ):
            return tuple(
                device_state[name].clone()
                for name in (
                    "positions",
                    "strengths",
                    "mask",
                    "chart_ids",
                    "surface_uv",
                )
            )
        positions, strengths, mask, chart_ids, surface_uv = (
            filt._packed_continuous_surface_state_arrays()
        )
        return tuple(
            np.ascontiguousarray(values).copy()
            for values in (positions, strengths, mask, chart_ids, surface_uv)
        )

    @staticmethod
    def _joint_changed_cache_rows(
        before: tuple[Any, ...],
        filt: IsotopeParticleFilter,
    ) -> NDArray[np.int64]:
        """Return PF rows whose accepted isotope state changed exactly."""
        if before and hasattr(before[0], "detach"):
            import torch

            state = filt._structural_rj_device_state
            if state is None or not filt._structural_rj_device_state_authoritative:
                raise RuntimeError("CUDA cache delta lost authoritative state.")
            after = tuple(
                state[name]
                for name in (
                    "positions",
                    "strengths",
                    "mask",
                    "chart_ids",
                    "surface_uv",
                )
            )
            if len(before) != len(after) or any(
                first.shape != second.shape
                for first, second in zip(before, after, strict=True)
            ):
                raise RuntimeError("CUDA isotope cache-state shapes changed.")
            changed = torch.zeros(
                int(after[0].shape[0]),
                device=after[0].device,
                dtype=torch.bool,
            )
            for first, second in zip(before, after, strict=True):
                changed |= torch.any(
                    first.reshape(int(first.shape[0]), -1)
                    != second.reshape(int(second.shape[0]), -1),
                    dim=1,
                )
            return (
                torch.nonzero(changed, as_tuple=False)
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
        after = JointRejuvenationMixin._joint_isotope_cache_state(filt)
        if len(before) != len(after) or any(
            first.shape != second.shape
            for first, second in zip(before, after, strict=True)
        ):
            raise RuntimeError("Isotope cache-state shapes changed unexpectedly.")
        changed = np.zeros(after[0].shape[0], dtype=bool)
        for first, second in zip(before, after, strict=True):
            changed |= np.any(
                first.reshape(first.shape[0], -1)
                != second.reshape(second.shape[0], -1),
                axis=1,
            )
        return np.flatnonzero(changed).astype(np.int64, copy=False)

    @staticmethod
    def _weighted_correlation(
        before: NDArray[np.float64],
        after: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return a finite weighted lag-one correlation diagnostic."""
        first = np.asarray(before, dtype=np.float64).reshape(-1)
        second = np.asarray(after, dtype=np.float64).reshape(-1)
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        finite = np.isfinite(first) & np.isfinite(second)
        if not np.any(finite):
            return 0.0
        first = first[finite]
        second = second[finite]
        mass = mass[finite]
        mass_sum = float(np.sum(mass))
        if not np.isfinite(mass_sum) or mass_sum <= 0.0:
            return 0.0
        mass = mass / mass_sum
        first_centered = first - float(np.sum(mass * first))
        second_centered = second - float(np.sum(mass * second))
        first_var = float(np.sum(mass * np.square(first_centered)))
        second_var = float(np.sum(mass * np.square(second_centered)))
        if first_var <= 0.0 or second_var <= 0.0:
            return float(np.array_equal(first, second))
        correlation = float(
            np.sum(mass * first_centered * second_centered)
            / math.sqrt(first_var * second_var)
        )
        return float(np.clip(correlation, -1.0, 1.0))

    @staticmethod
    def _weighted_vector_lag1_correlation(
        before: NDArray[np.float64],
        after: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return one weighted lag-one correlation for vector-valued states."""
        first = np.asarray(before, dtype=np.float64)
        second = np.asarray(after, dtype=np.float64)
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        if first.ndim != 2 or second.shape != first.shape:
            raise ValueError(
                "Vector lag-one correlation requires matching row matrices."
            )
        if first.shape[0] != mass.size:
            raise ValueError(
                "Vector lag-one correlation weights must match matrix rows."
            )
        finite_rows = (
            np.all(np.isfinite(first), axis=1)
            & np.all(np.isfinite(second), axis=1)
            & np.isfinite(mass)
        )
        if not np.any(finite_rows):
            return 0.0
        first = first[finite_rows]
        second = second[finite_rows]
        mass = mass[finite_rows]
        mass_sum = float(np.sum(mass))
        if not np.isfinite(mass_sum) or mass_sum <= 0.0:
            return 0.0
        mass = mass / mass_sum
        first_centered = first - np.sum(
            mass[:, None] * first,
            axis=0,
            keepdims=True,
        )
        second_centered = second - np.sum(
            mass[:, None] * second,
            axis=0,
            keepdims=True,
        )
        first_variance = float(np.sum(mass[:, None] * np.square(first_centered)))
        second_variance = float(np.sum(mass[:, None] * np.square(second_centered)))
        if first_variance <= 0.0 or second_variance <= 0.0:
            return float(np.array_equal(first, second))
        covariance = float(np.sum(mass[:, None] * first_centered * second_centered))
        correlation = covariance / math.sqrt(first_variance * second_variance)
        return float(np.clip(correlation, -1.0, 1.0))

    def _joint_mixing_diagnostics(
        self,
        before: Mapping[str, object],
        after: Mapping[str, object],
        *,
        target_before: NDArray[np.float64],
        target_after: NDArray[np.float64],
    ) -> dict[str, float]:
        """Measure row diversity and movement without using ancestry labels."""
        weights = np.asarray(before["weights"], dtype=np.float64)
        before_matrix = np.asarray(
            before["state_matrix"],
            dtype=np.float64,
        )
        after_matrix = np.asarray(
            after["state_matrix"],
            dtype=np.float64,
        )
        if before_matrix.shape != after_matrix.shape:
            raise RuntimeError(
                "Joint rejuvenation changed the aligned particle layout."
            )
        state_changed = np.any(before_matrix != after_matrix, axis=1)
        before_cardinality = np.asarray(
            before["cardinality_matrix"],
            dtype=np.int64,
        )
        after_cardinality = np.asarray(
            after["cardinality_matrix"],
            dtype=np.int64,
        )
        k_changed = np.any(
            before_cardinality != after_cardinality,
            axis=1,
        )
        _, inverse = np.unique(
            after_cardinality,
            axis=0,
            return_inverse=True,
        )
        k_mass = np.bincount(
            inverse,
            weights=weights,
            minlength=int(np.max(inverse)) + 1,
        )
        positive_k_mass = k_mass[k_mass > 0.0]
        k_entropy = -float(np.sum(positive_k_mass * np.log(positive_k_mass)))
        position_row_esjd = np.zeros(weights.size, dtype=np.float64)
        strength_row_esjd = np.zeros(weights.size, dtype=np.float64)
        before_states = before["isotope_states"]
        after_states = after["isotope_states"]
        if not isinstance(before_states, Mapping) or not isinstance(
            after_states,
            Mapping,
        ):
            raise RuntimeError("Joint mixing snapshots lost isotope states.")
        for isotope in self.joint_isotope_order():
            first = before_states[isotope]
            second = after_states[isotope]
            if not isinstance(first, Mapping) or not isinstance(
                second,
                Mapping,
            ):
                raise RuntimeError("Joint mixing isotope snapshots are malformed.")
            first_mask = np.asarray(first["mask"], dtype=bool)
            second_mask = np.asarray(second["mask"], dtype=bool)
            same_cardinality = np.asarray(
                first["cardinalities"], dtype=np.int64
            ) == np.asarray(second["cardinalities"], dtype=np.int64)
            matched = first_mask & second_mask & same_cardinality[:, None]
            position_delta = np.asarray(
                second["positions"], dtype=np.float64
            ) - np.asarray(first["positions"], dtype=np.float64)
            position_row_esjd += np.sum(
                np.where(
                    matched,
                    np.sum(np.square(position_delta), axis=2),
                    0.0,
                ),
                axis=1,
            )
            first_strength = np.asarray(
                first["strengths"],
                dtype=np.float64,
            )
            second_strength = np.asarray(
                second["strengths"],
                dtype=np.float64,
            )
            log_strength_delta = np.zeros_like(first_strength)
            log_strength_delta[matched] = np.log(second_strength[matched]) - np.log(
                first_strength[matched]
            )
            strength_row_esjd += np.sum(
                np.square(log_strength_delta),
                axis=1,
            )
        transition_before = before["transition_mass"]
        transition_after = after["transition_mass"]
        if not isinstance(transition_before, Mapping) or not isinstance(
            transition_after,
            Mapping,
        ):
            raise RuntimeError("Joint transition-mass snapshots are malformed.")
        transition_delta = {
            str(key): float(value) - float(transition_before.get(key, 0.0))
            for key, value in transition_after.items()
        }
        accepted_structure_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    (
                        "birth_accepted_weight_mass",
                        "death_accepted_weight_mass",
                        "split_accepted_weight_mass",
                        "merge_accepted_weight_mass",
                    )
                )
            )
        )
        full_support_attempted_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    (
                        "birth_attempted_weight_mass",
                        "global_position_attempted_weight_mass",
                        "block_attempted_weight_mass",
                    )
                )
            )
        )
        full_support_accepted_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    (
                        "birth_accepted_weight_mass",
                        "global_position_accepted_weight_mass",
                        "block_accepted_weight_mass",
                    )
                )
            )
        )
        boundary_inward_attempted_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    "ordinary_boundary_inward_attempted_weight_mass"
                )
            )
        )
        boundary_inward_supported_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    "ordinary_boundary_inward_supported_weight_mass"
                )
            )
        )
        boundary_inward_finite_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    "ordinary_boundary_inward_finite_weight_mass"
                )
            )
        )
        boundary_inward_accepted_mass = float(
            sum(
                value
                for key, value in transition_delta.items()
                if key.endswith(
                    "ordinary_boundary_inward_accepted_weight_mass"
                )
            )
        )
        diagnostics = {
            "distinct_joint_state_count": float(
                np.unique(after_matrix, axis=0).shape[0]
            ),
            "joint_k_vector_count": float(
                np.unique(
                    after_cardinality,
                    axis=0,
                ).shape[0]
            ),
            "joint_k_entropy": k_entropy,
            "state_change_weight_mass": float(np.sum(weights[state_changed])),
            "k_transition_weight_mass": float(np.sum(weights[k_changed])),
            "accepted_structure_weight_mass": accepted_structure_mass,
            "full_support_attempted_weight_mass": full_support_attempted_mass,
            "full_support_accepted_weight_mass": full_support_accepted_mass,
            "ordinary_boundary_inward_attempted_weight_mass": (
                boundary_inward_attempted_mass
            ),
            "ordinary_boundary_inward_supported_weight_mass": (
                boundary_inward_supported_mass
            ),
            "ordinary_boundary_inward_finite_weight_mass": (
                boundary_inward_finite_mass
            ),
            "ordinary_boundary_inward_accepted_weight_mass": (
                boundary_inward_accepted_mass
            ),
            "surface_position_esjd_m2": float(np.sum(weights * position_row_esjd)),
            "log_strength_esjd": float(np.sum(weights * strength_row_esjd)),
            "target_log_likelihood_lag1_correlation": (
                self._weighted_correlation(
                    target_before,
                    target_after,
                    weights,
                )
            ),
            "joint_k_vector_lag1_correlation": (
                self._weighted_vector_lag1_correlation(
                    before_cardinality,
                    after_cardinality,
                    weights,
                )
            ),
        }
        ordinary_boundary = np.zeros(weights.size, dtype=np.bool_)
        ordinary_boundary_escape = np.zeros(weights.size, dtype=np.bool_)
        hard_boundary = np.zeros(weights.size, dtype=np.bool_)
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            ordinary_maximum = int(self.filters[isotope].config.max_sources or 0)
            hard_maximum = int(self.filters[isotope].config.hard_max_sources or 0)
            at_boundary = before_cardinality[:, isotope_index] >= ordinary_maximum
            escaped = at_boundary & (
                after_cardinality[:, isotope_index]
                < before_cardinality[:, isotope_index]
            )
            at_hard_boundary = after_cardinality[:, isotope_index] >= hard_maximum
            isotope_k_changed = (
                before_cardinality[:, isotope_index]
                != after_cardinality[:, isotope_index]
            )
            ordinary_boundary |= at_boundary
            ordinary_boundary_escape |= escaped
            hard_boundary |= at_hard_boundary
            diagnostics[f"k_transition_weight_mass.{isotope}"] = float(
                np.sum(weights[isotope_k_changed])
            )
            diagnostics[f"ordinary_boundary_weight_mass.{isotope}"] = float(
                np.sum(weights[at_boundary])
            )
            diagnostics[f"ordinary_boundary_escape_weight_mass.{isotope}"] = float(
                np.sum(weights[escaped])
            )
            diagnostics[f"hard_boundary_weight_mass.{isotope}"] = float(
                np.sum(weights[at_hard_boundary])
            )
            transition_suffixes_by_metric = {
                "accepted_structure_weight_mass": (
                    "birth_accepted_weight_mass",
                    "death_accepted_weight_mass",
                    "split_accepted_weight_mass",
                    "merge_accepted_weight_mass",
                ),
                "ordinary_boundary_inward_attempted_weight_mass": (
                    "ordinary_boundary_inward_attempted_weight_mass",
                ),
                "ordinary_boundary_inward_supported_weight_mass": (
                    "ordinary_boundary_inward_supported_weight_mass",
                ),
                "ordinary_boundary_inward_finite_weight_mass": (
                    "ordinary_boundary_inward_finite_weight_mass",
                ),
                "ordinary_boundary_inward_accepted_weight_mass": (
                    "ordinary_boundary_inward_accepted_weight_mass",
                ),
                "full_support_attempted_weight_mass": (
                    "birth_attempted_weight_mass",
                    "global_position_attempted_weight_mass",
                    "block_attempted_weight_mass",
                ),
                "full_support_accepted_weight_mass": (
                    "birth_accepted_weight_mass",
                    "global_position_accepted_weight_mass",
                    "block_accepted_weight_mass",
                ),
            }
            for metric, transition_suffixes in (
                transition_suffixes_by_metric.items()
            ):
                diagnostics[f"{metric}.{isotope}"] = float(
                    sum(
                        value
                        for key, value in transition_delta.items()
                        if key.startswith(f"{isotope}.")
                        and key.endswith(transition_suffixes)
                    )
                )
        diagnostics["ordinary_boundary_weight_mass"] = float(
            np.sum(weights[ordinary_boundary])
        )
        diagnostics["ordinary_boundary_escape_weight_mass"] = float(
            np.sum(weights[ordinary_boundary_escape])
        )
        diagnostics["hard_boundary_weight_mass"] = float(
            np.sum(weights[hard_boundary])
        )
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            diagnostics[f"k_lag1_correlation.{isotope}"] = self._weighted_correlation(
                before_cardinality[:, isotope_index],
                after_cardinality[:, isotope_index],
                weights,
            )
        return diagnostics

    def _joint_rejuvenate(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
    ) -> dict[str, float]:
        """Apply conditional exact-RJ sweeps under one station bridge target."""
        sweep_start_s = time.perf_counter()
        active = tuple(stations)
        if not active:
            return {}
        state_before = self._joint_mixing_snapshot()
        diagnostics: dict[str, float] = {}
        cross_state_attempted_start = float(
            self.last_joint_cross_isotope_state_attempted_weight_mass
        )
        cross_state_accepted_start = float(
            self.last_joint_cross_isotope_state_accepted_weight_mass
        )
        strength_block_attempted_start = float(
            self.last_joint_strength_block_attempted_weight_mass
        )
        strength_block_accepted_start = float(
            self.last_joint_strength_block_accepted_weight_mass
        )
        active_station_ids = tuple(id(station) for station in active)
        if active_station_ids != self._joint_torch_context_station_ids:
            self._joint_torch_observation_context_cache.clear()
            self._joint_torch_context_station_ids = active_station_ids
        self._joint_cuda_accepted_unit_transport_cache.clear()
        self.last_joint_staged_transport_commit_rows = 0
        self._active_joint_station_history = active
        newest_start = sum(int(station.fe_indices.size) for station in active[:-1])
        try:
            cache_refresh_start_s = time.perf_counter()
            self._refresh_joint_structural_transport_cache(active)
            cache_refresh_wall_s = time.perf_counter() - cache_refresh_start_s
            isotope_order = self.joint_isotope_order()
            cache = self._joint_structural_transport_cache
            if cache is None:
                raise RuntimeError(
                    "Joint structural transport cache was not initialized."
                )
            initial_target_start_s = time.perf_counter()
            cache_is_torch = hasattr(cache[0], "detach")
            cache_is_cuda = cache_is_torch and bool(cache[0].is_cuda)
            if cache_is_torch:
                import torch

                if isinstance(cache, JointTransportCache):
                    current_target_tensor = (
                        self._joint_initial_cached_history_target_torch(
                            filt=self.filters[isotope_order[0]],
                            stations=active,
                            cache=cache,
                            target_beta=float(target_beta),
                        )
                    )
                else:
                    current_target_tensor = (
                        self._joint_history_log_likelihood_torch(
                            filt=self.filters[isotope_order[0]],
                            stations=active,
                            total_nvsl=cache[0],
                            uncollided_nvsl=cache[1],
                            features_nvslf=cache[2],
                            target_beta=float(target_beta),
                        )
                    )
                if cache_is_cuda:
                    joint_seed = int(
                        self._joint_random_generator.integers(
                            0,
                            np.iinfo(np.int64).max,
                            dtype=np.int64,
                        )
                    )
                    joint_generator = torch.Generator(device=cache[0].device)
                    joint_generator.manual_seed(joint_seed)
                    self._joint_torch_generator = joint_generator
                    current_target_log_likelihood = current_target_tensor
                else:
                    current_target_log_likelihood = (
                        current_target_tensor.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
            else:
                current_target_log_likelihood = (
                    self._joint_history_log_likelihood_numpy(
                        filt=self.filters[isotope_order[0]],
                        stations=active,
                        total_nvsl=cache[0],
                        uncollided_nvsl=cache[1],
                        features_nvslf=cache[2],
                        target_beta=float(target_beta),
                    )
                )
            current_station_log_likelihood = (
                cache.station_log_likelihood[:, : cache.station_count].clone()
                if isinstance(cache, JointTransportCache) and cache_is_torch
                else None
            )
            initial_target_wall_s = time.perf_counter() - initial_target_start_s
            if hasattr(current_target_log_likelihood, "detach"):
                target_before = (
                    current_target_log_likelihood.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                    .copy()
                )
            else:
                target_before = np.asarray(
                    current_target_log_likelihood,
                    dtype=np.float64,
                ).copy()
            isotope_wall_s: dict[str, float] = {}
            adjacent_transition_counts_by_isotope: dict[
                str,
                dict[str, int],
            ] = {}
            for isotope in isotope_order:
                isotope_start_s = time.perf_counter()
                filt = self.filters[isotope]
                cache_state_before = self._joint_isotope_cache_state(filt)
                evidence = self._joint_history_structural_geometry(
                    isotope,
                    active,
                )
                self._validate_joint_structural_geometry(evidence, active)
                self._active_joint_structural_geometry = evidence
                try:
                    filt._initialize_continuous_rj_device_state(cache[0])
                    filt.apply_structural_moves(
                        evidence,
                        target_beta=float(target_beta),
                        tempering_start_row=int(newest_start),
                        current_target_log_likelihood=(current_target_log_likelihood),
                        current_station_log_likelihood=(
                            current_station_log_likelihood
                        ),
                    )
                finally:
                    self._active_joint_structural_geometry = None
                adjacent_transition_counts_by_isotope[str(isotope)] = (
                    self._adjacent_cardinality_transition_counts(
                        filt.last_structural_rejection_diagnostics
                    )
                )
                self._assert_joint_particle_alignment()
                updated_target = (
                    filt.last_structural_target_log_likelihood_device
                    if cache_is_cuda
                    else filt.last_structural_target_log_likelihood
                )
                if updated_target is None:
                    raise RuntimeError(
                        "Joint structural sweep did not return its target cache."
                    )
                current_target_log_likelihood = updated_target
                updated_station_target = (
                    filt.last_structural_station_log_likelihood_device
                )
                if cache_is_cuda and updated_station_target is None:
                    raise RuntimeError(
                        "CUDA structural sweep did not return its station cache."
                    )
                changed_rows = self._joint_changed_cache_rows(
                    cache_state_before,
                    filt,
                )
                self._record_joint_lineage_recovery_acceptance(
                    isotope=str(isotope),
                    accepted_mask=(
                        filt.last_structural_full_support_accepted_mask
                    ),
                    changed_rows=changed_rows,
                )
                if changed_rows.size:
                    if cache_is_cuda:
                        self._joint_commit_staged_cuda_transport_cache_isotope(
                            filt=filt,
                            data=evidence,
                            stations=active,
                            particle_indices=changed_rows,
                            sweep_entry_state=cache_state_before,
                        )
                        if not isinstance(cache, JointTransportCache):
                            raise RuntimeError(
                                "CUDA exact-MH commit requires JointTransportCache."
                            )
                        import torch

                        changed_tensor = torch.as_tensor(
                            changed_rows,
                            device=cache[0].device,
                            dtype=torch.long,
                        )
                        if updated_station_target is not None:
                            cache.set_station_likelihood(
                                torch.index_select(
                                    updated_station_target,
                                    0,
                                    changed_tensor,
                                ),
                                rows=changed_tensor,
                            )
                    else:
                        self._refresh_joint_structural_transport_cache_isotope(
                            active,
                            isotope,
                            particle_indices=changed_rows,
                        )
                if updated_station_target is not None:
                    current_station_log_likelihood = updated_station_target
                isotope_wall_s[isotope] = time.perf_counter() - isotope_start_s
            strength_start_s = time.perf_counter()
            if float(self.pf_config.joint_strength_block_probability) > 0.0:
                current_target_log_likelihood = self._apply_joint_strength_block(
                    active,
                    target_beta=float(target_beta),
                    current_target_log_likelihood=(current_target_log_likelihood),
                )
            strength_wall_s = time.perf_counter() - strength_start_s
            cross_isotope_start_s = time.perf_counter()
            if float(self.pf_config.joint_cross_isotope_state_block_probability) > 0.0:
                current_target_log_likelihood = (
                    self._apply_joint_cross_isotope_state_block(
                        active,
                        target_beta=float(target_beta),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                )
            cross_isotope_wall_s = time.perf_counter() - cross_isotope_start_s
            station_cache_refresh_start_s = time.perf_counter()
            station_cache_refresh_rows = 0
            if isinstance(cache, JointTransportCache) and cache_is_torch:
                import torch

                invalid_rows = torch.nonzero(
                    torch.any(
                        torch.isnan(
                            cache.station_log_likelihood[:, : cache.station_count]
                        ),
                        dim=1,
                    ),
                    as_tuple=False,
                ).reshape(-1)
                station_cache_refresh_rows = int(invalid_rows.numel())
                if station_cache_refresh_rows:
                    committed_target, station_values = (
                        self._joint_history_log_likelihood_torch(
                            filt=self.filters[isotope_order[0]],
                            stations=active,
                            total_nvsl=cache[0],
                            uncollided_nvsl=cache[1],
                            features_nvslf=cache[2],
                            target_beta=float(target_beta),
                            return_station_log_likelihood=True,
                            particle_indices=invalid_rows,
                        )
                    )
                    expected_target = torch.index_select(
                        torch.as_tensor(
                            current_target_log_likelihood,
                            device=cache[0].device,
                            dtype=cache[0].dtype,
                        ),
                        0,
                        invalid_rows,
                    )
                    if not torch.allclose(
                        committed_target,
                        expected_target,
                        rtol=2.0e-12,
                        atol=1.0e-8,
                    ):
                        maximum_error = float(
                            torch.max(
                                torch.abs(committed_target - expected_target)
                            ).item()
                        )
                        raise RuntimeError(
                            "Accepted slot-overlay target differs from its exact "
                            f"committed-state rebase (max error {maximum_error:.6g})."
                        )
                    cache.set_station_likelihood(
                        station_values,
                        rows=invalid_rows,
                    )
            station_cache_refresh_wall_s = (
                time.perf_counter() - station_cache_refresh_start_s
            )
            mixing_start_s = time.perf_counter()
            if hasattr(current_target_log_likelihood, "detach"):
                target_after = (
                    current_target_log_likelihood.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                target_after = np.asarray(
                    current_target_log_likelihood,
                    dtype=np.float64,
                )
            diagnostics = self._joint_mixing_diagnostics(
                state_before,
                self._joint_mixing_snapshot(),
                target_before=target_before,
                target_after=target_after,
            )
            for isotope, transition_counts in (
                adjacent_transition_counts_by_isotope.items()
            ):
                for name, value in transition_counts.items():
                    diagnostics[f"{name}.{isotope}"] = float(value)
            diagnostics["cross_isotope_state_attempted_weight_mass"] = float(
                self.last_joint_cross_isotope_state_attempted_weight_mass
                - cross_state_attempted_start
            )
            diagnostics["cross_isotope_state_accepted_weight_mass"] = float(
                self.last_joint_cross_isotope_state_accepted_weight_mass
                - cross_state_accepted_start
            )
            diagnostics["joint_strength_attempted_weight_mass"] = float(
                self.last_joint_strength_block_attempted_weight_mass
                - strength_block_attempted_start
            )
            diagnostics["joint_strength_accepted_weight_mass"] = float(
                self.last_joint_strength_block_accepted_weight_mass
                - strength_block_accepted_start
            )
            diagnostics["wall_s.cache_refresh"] = float(cache_refresh_wall_s)
            diagnostics["wall_s.initial_target"] = float(initial_target_wall_s)
            for isotope, elapsed_s in isotope_wall_s.items():
                diagnostics[f"wall_s.isotope.{isotope}"] = float(elapsed_s)
            diagnostics["wall_s.joint_strength"] = float(strength_wall_s)
            diagnostics["wall_s.cross_isotope"] = float(cross_isotope_wall_s)
            diagnostics["wall_s.station_likelihood_rebase"] = float(
                station_cache_refresh_wall_s
            )
            diagnostics["station_likelihood_rebase_rows"] = float(
                station_cache_refresh_rows
            )
            if isinstance(cache, JointTransportCache):
                diagnostics["transport_cache_allocated_bytes"] = float(
                    cache.allocated_bytes
                )
                diagnostics["transport_cache_valid_views"] = float(
                    cache.valid_view_count
                )
            diagnostics["wall_s.mixing_diagnostics"] = float(
                time.perf_counter() - mixing_start_s
            )
            diagnostics["wall_s.total"] = float(
                time.perf_counter() - sweep_start_s
            )
            diagnostics["staged_transport_commit_rows"] = float(
                self.last_joint_staged_transport_commit_rows
            )
        finally:
            self._invalidate_posterior_summary_cache()
            for filt in self.filters.values():
                filt._clear_continuous_rj_device_state()
            self._active_joint_structural_geometry = None
            self._joint_structural_transport_cache = None
            self._active_joint_station_history = None
            self._joint_torch_generator = None
        return diagnostics

    def _joint_inward_move_rejection_summary(self) -> dict[str, object]:
        """Return compact latest-sweep diagnostics for inward RJ proposals."""
        summary: dict[str, object] = {}
        filters = getattr(self, "filters", {})
        for isotope in self.joint_isotope_order():
            moves: dict[str, object] = {}
            filt = filters.get(isotope) if isinstance(filters, Mapping) else None
            rejection = getattr(
                filt,
                "last_structural_rejection_diagnostics",
                {},
            )
            if not isinstance(rejection, Mapping):
                summary[str(isotope)] = moves
                continue
            for move, raw_move in rejection.items():
                if not isinstance(raw_move, Mapping):
                    continue
                transitions = raw_move.get("by_cardinality_transition", {})
                if not isinstance(transitions, Mapping):
                    continue
                inward: dict[str, object] = {}
                for transition, raw_transition in transitions.items():
                    if not isinstance(raw_transition, Mapping):
                        continue
                    parts = str(transition).split("->", maxsplit=1)
                    if len(parts) != 2:
                        continue
                    try:
                        source_count = int(parts[0])
                        destination_count = int(parts[1])
                    except ValueError:
                        continue
                    if destination_count >= source_count:
                        continue
                    ratio = raw_transition.get("component_quantiles", {})
                    ratio_values = (
                        ratio.get("log_acceptance_ratio")
                        if isinstance(ratio, Mapping)
                        else None
                    )
                    inward[str(transition)] = {
                        "attempted": int(raw_transition.get("attempted", 0)),
                        "accepted": int(raw_transition.get("accepted", 0)),
                        "support_rejected": int(
                            raw_transition.get("support_rejected", 0)
                        ),
                        "nonfinite_rejected": int(
                            raw_transition.get("nonfinite_rejected", 0)
                        ),
                        "mh_random_rejected": int(
                            raw_transition.get("mh_random_rejected", 0)
                        ),
                        "log_acceptance_median": (
                            None
                            if not isinstance(ratio_values, Mapping)
                            else float(ratio_values.get("median", float("nan")))
                        ),
                        "log_acceptance_p90": (
                            None
                            if not isinstance(ratio_values, Mapping)
                            else float(ratio_values.get("p90", float("nan")))
                        ),
                    }
                if inward:
                    moves[str(move)] = inward
            summary[str(isotope)] = moves
        return summary

    def _joint_rejuvenate_adaptive(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        station_start_s: float,
        enforce_lineage_recovery: bool = False,
    ) -> None:
        """Run exact sweeps until movement and lineage-recovery gates pass."""
        if enforce_lineage_recovery and not bool(
            getattr(self, "_joint_lineage_recovery_active", False)
        ):
            raise RuntimeError(
                "Lineage recovery cannot be enforced outside an active epoch."
            )
        if enforce_lineage_recovery and not math.isclose(
            float(target_beta),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "Lineage recovery is a final-target station health gate."
            )
        minimum_sweeps = int(self.pf_config.joint_rejuvenation_min_sweeps)
        wall_time_limit_s = float(
            self.pf_config.joint_smc_rejuvenation_wall_time_limit_s
        )
        cumulative_state_change_mass = 0.0
        cumulative_surface_position_esjd = 0.0
        cumulative_log_strength_esjd = 0.0
        cumulative_k_transition_mass = 0.0
        cumulative_boundary_escape_mass = 0.0
        cumulative_boundary_inward_supported_mass = 0.0
        isotope_order = tuple(str(value) for value in self.joint_isotope_order())
        cumulative_k_transition_mass_by_isotope = {
            isotope: 0.0 for isotope in isotope_order
        }
        station_boundary_inward_attempted_mass_by_isotope = {
            isotope: float(
                sum(
                    float(
                        record.get(
                            "ordinary_boundary_inward_attempted_weight_mass."
                            f"{isotope}",
                            0.0,
                        )
                    )
                    for record in self.last_joint_rejuvenation_diagnostics
                )
            )
            for isotope in isotope_order
        }
        station_boundary_inward_supported_mass_by_isotope = {
            isotope: float(
                sum(
                    float(
                        record.get(
                            "ordinary_boundary_inward_supported_weight_mass."
                            f"{isotope}",
                            0.0,
                        )
                    )
                    for record in self.last_joint_rejuvenation_diagnostics
                )
            )
            for isotope in isotope_order
        }
        station_boundary_inward_finite_mass_by_isotope = {
            isotope: float(
                sum(
                    float(
                        record.get(
                            "ordinary_boundary_inward_finite_weight_mass."
                            f"{isotope}",
                            0.0,
                        )
                    )
                    for record in self.last_joint_rejuvenation_diagnostics
                )
            )
            for isotope in isotope_order
        }
        station_boundary_inward_accepted_mass_by_isotope = {
            isotope: float(
                sum(
                    float(
                        record.get(
                            "ordinary_boundary_inward_accepted_weight_mass."
                            f"{isotope}",
                            0.0,
                        )
                    )
                    for record in self.last_joint_rejuvenation_diagnostics
                )
            )
            for isotope in isotope_order
        }
        cumulative_full_support_attempted_mass_by_isotope = {
            isotope: 0.0 for isotope in isotope_order
        }
        cumulative_full_support_accepted_mass_by_isotope = {
            isotope: 0.0 for isotope in isotope_order
        }
        boundary_mass_by_isotope = {isotope: 0.0 for isotope in isotope_order}
        hard_boundary_mass_by_isotope = {
            isotope: 0.0 for isotope in isotope_order
        }
        self.last_joint_rejuvenation_mixing_incomplete = True
        self.last_joint_structural_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete_by_isotope = {
            isotope: False for isotope in isotope_order
        }
        previous_unmet_gates: tuple[str, ...] | None = None
        previous_gate_progress = float("-inf")
        no_progress_sweeps = 0
        no_progress_start_s = time.perf_counter()
        for sweep_index in itertools.count():
            sweep_start_s = time.perf_counter()
            diagnostics = self._joint_rejuvenate(
                stations,
                target_beta=target_beta,
            )
            sweep_elapsed_s = time.perf_counter() - sweep_start_s
            if not isinstance(diagnostics, Mapping):
                raise RuntimeError(
                    "Joint rejuvenation did not return mixing diagnostics."
                )
            record = {str(key): float(value) for key, value in diagnostics.items()}
            record.update(
                {
                    "target_beta": float(target_beta),
                    "sweep_index": float(sweep_index + 1),
                    "station_elapsed_s": float(time.perf_counter() - station_start_s),
                }
            )
            self.last_joint_rejuvenation_diagnostics.append(record)
            state_change_mass = float(
                diagnostics.get("state_change_weight_mass", 0.0)
            )
            surface_position_esjd = float(
                diagnostics.get("surface_position_esjd_m2", 0.0)
            )
            log_strength_esjd = float(
                diagnostics.get("log_strength_esjd", 0.0)
            )
            cumulative_state_change_mass += state_change_mass
            cumulative_surface_position_esjd += surface_position_esjd
            cumulative_log_strength_esjd += log_strength_esjd
            k_transition_mass = float(
                diagnostics.get("k_transition_weight_mass", 0.0)
            )
            cumulative_k_transition_mass += k_transition_mass
            cumulative_boundary_escape_mass += float(
                diagnostics.get(
                    "ordinary_boundary_escape_weight_mass",
                    0.0,
                )
            )
            cumulative_boundary_inward_supported_mass += float(
                diagnostics.get(
                    "ordinary_boundary_inward_supported_weight_mass",
                    0.0,
                )
            )
            for isotope in isotope_order:
                cumulative_k_transition_mass_by_isotope[isotope] += float(
                    diagnostics.get(
                        f"k_transition_weight_mass.{isotope}",
                        0.0,
                    )
                )
                station_boundary_inward_attempted_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        "ordinary_boundary_inward_attempted_weight_mass."
                        f"{isotope}",
                        0.0,
                    )
                )
                station_boundary_inward_finite_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        "ordinary_boundary_inward_finite_weight_mass."
                        f"{isotope}",
                        0.0,
                    )
                )
                station_boundary_inward_supported_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        "ordinary_boundary_inward_supported_weight_mass."
                        f"{isotope}",
                        0.0,
                    )
                )
                station_boundary_inward_accepted_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        "ordinary_boundary_inward_accepted_weight_mass."
                        f"{isotope}",
                        0.0,
                    )
                )
                cumulative_full_support_attempted_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        f"full_support_attempted_weight_mass.{isotope}",
                        0.0,
                    )
                )
                cumulative_full_support_accepted_mass_by_isotope[
                    isotope
                ] += float(
                    diagnostics.get(
                        f"full_support_accepted_weight_mass.{isotope}",
                        0.0,
                    )
                )
                boundary_mass_by_isotope[isotope] = float(
                    diagnostics.get(
                        f"ordinary_boundary_weight_mass.{isotope}",
                        0.0,
                    )
                )
                hard_boundary_mass_by_isotope[isotope] = float(
                    diagnostics.get(
                        f"hard_boundary_weight_mass.{isotope}",
                        0.0,
                    )
                )
            record["cumulative_k_transition_weight_mass"] = float(
                cumulative_k_transition_mass
            )
            record["cumulative_boundary_escape_weight_mass"] = float(
                cumulative_boundary_escape_mass
            )
            record["cumulative_boundary_inward_supported_weight_mass"] = float(
                cumulative_boundary_inward_supported_mass
            )
            record["cumulative_state_change_weight_mass"] = float(
                cumulative_state_change_mass
            )
            record["cumulative_surface_position_esjd_m2"] = float(
                cumulative_surface_position_esjd
            )
            record["cumulative_log_strength_esjd"] = float(
                cumulative_log_strength_esjd
            )
            station_elapsed = time.perf_counter() - station_start_s
            isotope_wall_s = sum(
                float(value)
                for key, value in diagnostics.items()
                if str(key).startswith("wall_s.isotope.")
            )
            print(
                "[joint-smc] rejuvenation-sweep-done "
                f"beta={target_beta:.12g} "
                f"sweep={sweep_index + 1} "
                f"sweep_s={sweep_elapsed_s:.3f} "
                f"station_s={station_elapsed:.3f} "
                f"cache_s={diagnostics.get('wall_s.cache_refresh', 0.0):.3f} "
                f"target_s={diagnostics.get('wall_s.initial_target', 0.0):.3f} "
                f"isotopes_s={isotope_wall_s:.3f} "
                "staged_rows="
                f"{int(diagnostics.get('staged_transport_commit_rows', 0.0))} "
                f"strength_s={diagnostics.get('wall_s.joint_strength', 0.0):.3f} "
                f"cross_s={diagnostics.get('wall_s.cross_isotope', 0.0):.3f}",
                flush=True,
            )
            if sweep_index + 1 < minimum_sweeps:
                continue
            continuous_movement_sufficient = (
                cumulative_state_change_mass
                >= float(
                    self.pf_config.joint_rejuvenation_min_state_change_weight_mass
                )
                and (
                    cumulative_surface_position_esjd
                    >= float(
                        self.pf_config.joint_rejuvenation_min_surface_esjd_m2
                    )
                    or cumulative_log_strength_esjd
                    >= float(
                        self.pf_config.joint_rejuvenation_min_log_strength_esjd
                    )
                )
            )
            final_target = bool(
                math.isclose(
                    float(target_beta),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            ordinary_boundary_diagnostic_by_isotope: dict[str, bool] = {}
            inward_proposal_integrity_by_isotope: dict[str, bool] = {}
            hard_boundary_saturated_by_isotope: dict[str, bool] = {}
            for isotope in isotope_order:
                ordinary_boundary_diagnostic = bool(
                    final_target
                    and self.pf_config.variable_cardinality
                    and boundary_mass_by_isotope[isotope]
                    > HARD_CAP_POSTERIOR_MASS_LIMIT
                )
                hard_boundary_saturated = bool(
                    final_target
                    and self.pf_config.variable_cardinality
                    and not hard_cap_mass_is_acceptable(
                        hard_boundary_mass_by_isotope[isotope]
                    )
                )
                inward_proposal_integrity = bool(
                    not ordinary_boundary_diagnostic
                    or (
                        station_boundary_inward_attempted_mass_by_isotope[
                            isotope
                        ]
                        > 0.0
                        and station_boundary_inward_supported_mass_by_isotope[
                            isotope
                        ]
                        > 0.0
                        and station_boundary_inward_finite_mass_by_isotope[
                            isotope
                        ]
                        > 0.0
                    )
                )
                ordinary_boundary_diagnostic_by_isotope[isotope] = (
                    ordinary_boundary_diagnostic
                )
                inward_proposal_integrity_by_isotope[isotope] = (
                    inward_proposal_integrity
                )
                hard_boundary_saturated_by_isotope[isotope] = (
                    hard_boundary_saturated
                )
                record[f"structural_movement_required.{isotope}"] = float(
                    hard_boundary_saturated
                )
                record[f"structural_movement_sufficient.{isotope}"] = float(
                    not hard_boundary_saturated
                )
                record[
                    f"ordinary_boundary_diagnostic_active.{isotope}"
                ] = float(ordinary_boundary_diagnostic)
                record[
                    f"inward_proposal_integrity_sufficient.{isotope}"
                ] = float(inward_proposal_integrity)
                record[f"hard_boundary_saturated.{isotope}"] = float(
                    hard_boundary_saturated
                )
                record[f"cumulative_k_transition_weight_mass.{isotope}"] = float(
                    cumulative_k_transition_mass_by_isotope[isotope]
                )
                record[
                    "station_boundary_inward_attempted_weight_mass."
                    f"{isotope}"
                ] = float(
                    station_boundary_inward_attempted_mass_by_isotope[isotope]
                )
                record[
                    "station_boundary_inward_supported_weight_mass."
                    f"{isotope}"
                ] = float(
                    station_boundary_inward_supported_mass_by_isotope[isotope]
                )
                record[
                    "station_boundary_inward_finite_weight_mass."
                    f"{isotope}"
                ] = float(
                    station_boundary_inward_finite_mass_by_isotope[isotope]
                )
                record[
                    "station_boundary_inward_accepted_weight_mass."
                    f"{isotope}"
                ] = float(
                    station_boundary_inward_accepted_mass_by_isotope[isotope]
                )
            minimum_surviving_mass = (
                float(
                    self.pf_config.joint_lineage_recovery_min_surviving_weight_mass
                )
                if enforce_lineage_recovery
                else 0.0
            )
            recovery_summary = (
                self._joint_lineage_recovery_summary()
                if bool(
                    getattr(self, "_joint_lineage_recovery_active", False)
                )
                else {isotope: (0, 0.0) for isotope in isotope_order}
            )
            recovery_sufficient_by_isotope = {
                isotope: bool(
                    recovery_summary[isotope][0] > 0
                    and recovery_summary[isotope][1] >= minimum_surviving_mass
                )
                for isotope in isotope_order
            }
            minimum_distinct_joint_states = (
                int(
                    math.ceil(
                        float(self.pf_config.target_ess_ratio)
                        * int(self.pf_config.num_particles)
                    )
                )
                if enforce_lineage_recovery
                else 0
            )
            distinct_joint_state_count = int(
                diagnostics.get("distinct_joint_state_count", 0.0)
            )
            lineage_recovery_sufficient = bool(
                not enforce_lineage_recovery
                or (
                    distinct_joint_state_count >= minimum_distinct_joint_states
                    and all(recovery_sufficient_by_isotope.values())
                )
            )
            record["lineage_recovery_required"] = float(
                enforce_lineage_recovery
            )
            record["lineage_recovery_sufficient"] = float(
                lineage_recovery_sufficient
            )
            record["lineage_recovery_epoch"] = float(
                getattr(self, "_joint_lineage_recovery_epoch", 0)
            )
            record["lineage_recovery_min_surviving_weight_mass"] = float(
                minimum_surviving_mass
            )
            record["minimum_distinct_joint_states"] = float(
                minimum_distinct_joint_states
            )
            for isotope in isotope_order:
                record[
                    f"cumulative_full_support_attempted_weight_mass.{isotope}"
                ] = float(
                    cumulative_full_support_attempted_mass_by_isotope[isotope]
                )
                record[
                    f"cumulative_full_support_accepted_weight_mass.{isotope}"
                ] = float(
                    cumulative_full_support_accepted_mass_by_isotope[isotope]
                )
                certified_count, surviving_mass = recovery_summary[isotope]
                record[f"lineage_recovery_certified_row_count.{isotope}"] = float(
                    certified_count
                )
                record[f"lineage_recovery_surviving_weight_mass.{isotope}"] = float(
                    surviving_mass
                )
                record[f"lineage_recovery_sufficient.{isotope}"] = float(
                    recovery_sufficient_by_isotope[isotope]
                )
            record["continuous_movement_sufficient"] = float(
                continuous_movement_sufficient
            )
            record["structural_movement_required"] = float(
                any(hard_boundary_saturated_by_isotope.values())
            )
            record["hard_cap_posterior_mass_limit"] = (
                HARD_CAP_POSTERIOR_MASS_LIMIT
            )
            record["structural_movement_sufficient"] = float(
                not any(hard_boundary_saturated_by_isotope.values())
            )
            record["ordinary_boundary_diagnostic_active"] = float(
                any(ordinary_boundary_diagnostic_by_isotope.values())
            )
            record["inward_proposal_integrity_sufficient"] = float(
                all(inward_proposal_integrity_by_isotope.values())
            )
            structural_diagnostics_by_isotope = {
                isotope: {
                    "boundary_weight_mass": boundary_mass_by_isotope[isotope],
                    "hard_boundary_weight_mass": hard_boundary_mass_by_isotope[
                        isotope
                    ],
                    "call_k_transition_weight_mass": (
                        cumulative_k_transition_mass_by_isotope[isotope]
                    ),
                    "station_inward_attempted_weight_mass": (
                        station_boundary_inward_attempted_mass_by_isotope[isotope]
                    ),
                    "station_inward_supported_weight_mass": (
                        station_boundary_inward_supported_mass_by_isotope[isotope]
                    ),
                    "station_inward_finite_weight_mass": (
                        station_boundary_inward_finite_mass_by_isotope[isotope]
                    ),
                    "station_inward_accepted_weight_mass": (
                        station_boundary_inward_accepted_mass_by_isotope[isotope]
                    ),
                }
                for isotope in isotope_order
            }
            hard_boundary_failures = {
                isotope: diagnostics
                for isotope, diagnostics in structural_diagnostics_by_isotope.items()
                if final_target
                and not hard_cap_mass_is_acceptable(
                    diagnostics["hard_boundary_weight_mass"]
                )
            }
            if hard_boundary_failures:
                self.last_joint_rejuvenation_mixing_incomplete = True
                self.last_joint_structural_mixing_incomplete = True
                self.last_joint_structural_mixing_incomplete_by_isotope = {
                    isotope: isotope in hard_boundary_failures
                    for isotope in isotope_order
                }
                record["terminated_at_hard_cap"] = 1.0
                print(
                    "[joint-smc] sampler-quality-failed "
                    "reason=hard-cardinality-cap "
                    f"target_beta={target_beta:.12g}, "
                    "hard_cap_posterior_mass_limit="
                    f"{HARD_CAP_POSTERIOR_MASS_LIMIT:.12g}, "
                    f"diagnostics_by_isotope={structural_diagnostics_by_isotope}",
                    flush=True,
                )
            inward_integrity_failures = {
                isotope: structural_diagnostics_by_isotope[isotope]
                for isotope in isotope_order
                if ordinary_boundary_diagnostic_by_isotope[isotope]
                and not inward_proposal_integrity_by_isotope[isotope]
            }
            if inward_integrity_failures:
                self.last_joint_rejuvenation_mixing_incomplete = True
                self.last_joint_structural_mixing_incomplete = True
                self.last_joint_structural_mixing_incomplete_by_isotope = {
                    isotope: isotope in inward_integrity_failures
                    for isotope in isotope_order
                }
                raise RuntimeError(
                    "Joint SMC inward-proposal integrity failed at the ordinary "
                    "cardinality boundary. At least one inward proposal must be "
                    "attempted, support-feasible, and have a finite MH ratio; "
                    "acceptance is not required: "
                    f"target_beta={target_beta:.12g}, "
                    f"diagnostics_by_isotope={inward_integrity_failures}, "
                    "latest_inward_moves="
                    f"{self._joint_inward_move_rejection_summary()}."
                )
            if hard_boundary_failures:
                break
            unmet_gate_progress: dict[str, float] = {}
            if not continuous_movement_sufficient:
                state_threshold = float(
                    self.pf_config.joint_rejuvenation_min_state_change_weight_mass
                )
                position_threshold = float(
                    self.pf_config.joint_rejuvenation_min_surface_esjd_m2
                )
                strength_threshold = float(
                    self.pf_config.joint_rejuvenation_min_log_strength_esjd
                )
                state_progress = min(
                    cumulative_state_change_mass / max(state_threshold, 1.0e-300),
                    1.0,
                )
                geometry_progress = max(
                    min(
                        cumulative_surface_position_esjd
                        / max(position_threshold, 1.0e-300),
                        1.0,
                    ),
                    min(
                        cumulative_log_strength_esjd
                        / max(strength_threshold, 1.0e-300),
                        1.0,
                    ),
                )
                unmet_gate_progress["continuous"] = 0.5 * (
                    state_progress + geometry_progress
                )
            if enforce_lineage_recovery and not lineage_recovery_sufficient:
                diversity_progress = min(
                    distinct_joint_state_count
                    / max(minimum_distinct_joint_states, 1),
                    1.0,
                )
                unmet_gate_progress["lineage.diversity"] = diversity_progress
                for isotope in isotope_order:
                    if recovery_sufficient_by_isotope[isotope]:
                        continue
                    certified_count, surviving_mass = recovery_summary[isotope]
                    certificate_progress = float(certified_count > 0)
                    mass_progress = min(
                        surviving_mass / max(minimum_surviving_mass, 1.0e-300),
                        1.0,
                    )
                    unmet_gate_progress[f"lineage.{isotope}"] = 0.5 * (
                        certificate_progress + mass_progress
                    )
            unmet_signature = tuple(sorted(unmet_gate_progress))
            gate_progress = float(sum(unmet_gate_progress.values()))
            if unmet_signature:
                if (
                    unmet_signature != previous_unmet_gates
                    or gate_progress > previous_gate_progress + 1.0e-12
                ):
                    no_progress_sweeps = 0
                    no_progress_start_s = time.perf_counter()
                else:
                    no_progress_sweeps += 1
                previous_unmet_gates = unmet_signature
                previous_gate_progress = gate_progress
                no_progress_elapsed_s = time.perf_counter() - no_progress_start_s
                record["gate_no_progress_sweeps"] = float(no_progress_sweeps)
                record["gate_no_progress_elapsed_s"] = float(
                    no_progress_elapsed_s
                )
                for gate, progress in unmet_gate_progress.items():
                    record[f"gate_progress.{gate}"] = float(progress)
                print(
                    "[joint-smc] gate-wait "
                    f"beta={target_beta:.12g} "
                    f"sweep={sweep_index + 1} "
                    f"unmet={','.join(unmet_signature)} "
                    f"no_progress_sweeps={no_progress_sweeps} "
                    f"no_progress_s={no_progress_elapsed_s:.3f}",
                    flush=True,
                )
                if (
                    no_progress_sweeps >= _JOINT_GATE_NO_PROGRESS_SWEEP_LIMIT
                    or no_progress_elapsed_s
                    >= _JOINT_GATE_NO_PROGRESS_WALL_TIME_S
                ):
                    self.last_joint_rejuvenation_mixing_incomplete = True
                    self.last_joint_structural_mixing_incomplete = False
                    self.last_joint_structural_mixing_incomplete_by_isotope = {
                        isotope: False for isotope in isotope_order
                    }
                    record["terminated_after_no_progress"] = 1.0
                    print(
                        "[joint-smc] sampler-quality-warning "
                        "reason=mixing-no-progress "
                        f"target_beta={target_beta:.12g}, "
                        f"unmet_gates={unmet_gate_progress}, "
                        f"no_progress_sweeps={no_progress_sweeps}, "
                        f"no_progress_elapsed_s={no_progress_elapsed_s:.3f}",
                        flush=True,
                    )
                    break
            if station_elapsed >= wall_time_limit_s:
                self.last_joint_smc_wall_time_limit_exceeded = True
                self.last_joint_rejuvenation_mixing_incomplete = True
                self.last_joint_structural_mixing_incomplete = False
                self.last_joint_structural_mixing_incomplete_by_isotope = {
                    isotope: False for isotope in isotope_order
                }
                record["terminated_at_wall_time_limit"] = 1.0
                print(
                    "[joint-smc] sampler-quality-warning "
                    "reason=rejuvenation-wall-time-limit "
                    f"elapsed_s={station_elapsed:.3f}, "
                    f"limit_s={wall_time_limit_s:.3f}, "
                    f"sweeps={sweep_index + 1}",
                    flush=True,
                )
                break
            if (
                continuous_movement_sufficient
                and lineage_recovery_sufficient
            ):
                self.last_joint_rejuvenation_mixing_incomplete = False
                self.last_joint_structural_mixing_incomplete_by_isotope = {
                    isotope: False for isotope in isotope_order
                }
                break

    def _apply_joint_guided_initialization(
        self,
        station: JointStationObservation,
    ) -> None:
        """Replace the first product-prior draw by an exact full-support IS draw."""
        if (
            self._joint_guided_initialization_applied
            or self._joint_station_history
            or not bool(self.pf_config.joint_guided_initialization)
            or not bool(self.pf_config.variable_cardinality)
        ):
            return
        initial_sha256 = self._joint_initial_product_prior_state_sha256
        current_matrix = np.asarray(
            self._joint_mixing_snapshot()["state_matrix"],
            dtype=np.float64,
        )
        current_sha256 = hashlib.sha256(
            np.ascontiguousarray(current_matrix).tobytes(order="C")
        ).hexdigest()
        if initial_sha256 is None or current_sha256 != initial_sha256:
            self._joint_guided_initialization_applied = True
            print(
                "[joint-smc] guided-initialization skipped "
                "reason=initial-state-was-explicitly-replaced",
                flush=True,
            )
            return
        order = self.joint_isotope_order()
        particle_count = int(self.pf_config.num_particles)
        reference_particles = self.filters[order[0]].continuous_particles
        identities = tuple(
            particle.joint_row_identity for particle in reference_particles
        )
        if len(identities) != particle_count or not all(
            isinstance(identity, JointRowIdentity) for identity in identities
        ):
            raise RuntimeError(
                "Guided initialization requires authenticated joint rows."
            )
        geometry = self._joint_station_structural_geometry(station)
        joint_log_prior_density = np.zeros(
            particle_count,
            dtype=np.float64,
        )
        joint_log_guided_density = np.zeros(
            particle_count,
            dtype=np.float64,
        )
        requested_prior_probability = float(
            self.pf_config.joint_guided_initialization_prior_row_probability
        )
        prior_row_count = min(
            particle_count,
            max(1, int(round(requested_prior_probability * particle_count))),
        )
        realized_prior_probability = prior_row_count / float(particle_count)
        component_rng = named_random_generator(
            self.random_seed,
            "joint_guided_initialization_component",
        )
        prior_rows = np.zeros(particle_count, dtype=bool)
        prior_rows[component_rng.permutation(particle_count)[:prior_row_count]] = True
        particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        cardinality_priors = tuple(
            self.filters[isotope]._structural_rj_cardinality_prior for isotope in order
        )
        if any(prior is None for prior in cardinality_priors):
            raise RuntimeError(
                "Guided initialization requires every cardinality prior."
            )
        cardinality_rng = named_random_generator(
            self.random_seed,
            "joint_guided_initialization_cardinality_vectors",
        )
        joint_cardinalities = _stratified_joint_cardinality_draws(
            tuple(
                np.asarray(prior.probabilities, dtype=np.float64)
                for prior in cardinality_priors
                if prior is not None
            ),
            particle_count,
            rng=cardinality_rng,
        )
        original_particles = {
            isotope: self.filters[isotope].continuous_particles for isotope in order
        }
        for isotope in order:
            self.filters[isotope].continuous_particles = [
                IsotopeParticle(
                    state=IsotopeState(
                        num_sources=0,
                        strengths=np.zeros(0, dtype=np.float64),
                        surface_chart_ids=np.zeros(0, dtype=np.int64),
                        surface_uv=np.zeros((0, 2), dtype=np.float64),
                    ),
                    log_weight=0.0,
                    joint_row_identity=identities[row],
                )
                for row in range(particle_count)
            ]
        self._active_joint_station_history = (station,)
        try:
            for isotope_index, isotope in enumerate(order):
                filt = self.filters[isotope]
                atlas = filt._structural_rj_surface_atlas
                cardinality_prior = filt._structural_rj_cardinality_prior
                if atlas is None or cardinality_prior is None:
                    raise RuntimeError(
                        "Guided initialization requires continuous surface "
                        "and cardinality priors."
                    )
                rng = named_random_generator(
                    self.random_seed,
                    "joint_guided_initialization",
                    isotope,
                )
                if filt.config.variable_cardinality:
                    cardinalities = joint_cardinalities[:, isotope_index].copy()
                else:
                    cardinalities = np.full(
                        particle_count,
                        int(filt.config.init_num_sources[0]),
                        dtype=np.int64,
                    )
                position_proposal = filt._build_continuous_rj_position_proposal(
                    geometry,
                    target_beta=1.0,
                )
                strength_proposal = filt._active_continuous_rj_strength_proposal()
                offsets = np.concatenate(
                    (
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(cardinalities, dtype=np.int64),
                    )
                )
                total_sources = int(offsets[-1])
                source_rows = np.repeat(
                    np.arange(particle_count, dtype=np.int64),
                    cardinalities,
                )
                source_uses_prior = prior_rows[source_rows]
                chart_ids = np.empty(total_sources, dtype=np.int64)
                surface_uv = np.empty((total_sources, 2), dtype=np.float64)
                strengths = np.empty(total_sources, dtype=np.float64)
                for use_prior in (True, False):
                    source_indices = np.flatnonzero(source_uses_prior == use_prior)
                    if source_indices.size == 0:
                        continue
                    sampled_chart_ids, sampled_uv, _ = atlas.sample(
                        int(source_indices.size),
                        rng=rng,
                        chart_probabilities=(
                            None if use_prior else position_proposal.chart_probabilities
                        ),
                    )
                    chart_ids[source_indices] = sampled_chart_ids
                    surface_uv[source_indices] = sampled_uv
                log_cardinality_density = np.log(
                    cardinality_prior.probabilities[cardinalities]
                )
                isotope_log_prior_density = log_cardinality_density.copy()
                isotope_log_guided_density = log_cardinality_density.copy()
                if total_sources:
                    source_log_prior = atlas.log_chart_probabilities[chart_ids]
                    source_log_guided = position_proposal.log_density(chart_ids)
                    isotope_log_prior_density += np.bincount(
                        source_rows,
                        weights=source_log_prior,
                        minlength=particle_count,
                    )
                    isotope_log_guided_density += np.bincount(
                        source_rows,
                        weights=source_log_guided,
                        minlength=particle_count,
                    )
                    minimum_strength = float(filt._strength_prior.minimum)
                    for cardinality in np.unique(
                        cardinalities[cardinalities > 0]
                    ).tolist():
                        state_rows = np.flatnonzero(cardinalities == int(cardinality))
                        source_indices = (
                            offsets[state_rows, None]
                            + np.arange(
                                int(cardinality),
                                dtype=np.int64,
                            )[None, :]
                        )
                        state_charts = chart_ids[source_indices]
                        single_source_locations = np.asarray(
                            strength_proposal.data_locations_by_chart[state_charts],
                            dtype=np.float64,
                        )
                        conditional_locations = minimum_strength + (
                            single_source_locations - minimum_strength
                        ) / float(cardinality)
                        block_proposal = ContinuousBlockStrengthProposal(
                            minimum=minimum_strength,
                            maximum=float(filt._strength_prior.maximum),
                            data_locations=conditional_locations,
                            data_sigma=float(strength_proposal.data_sigma),
                            prior_component_probability=float(
                                strength_proposal.prior_component_probability
                            ),
                            prior_family=str(filt._strength_prior.family),
                            prior_gamma_shape=float(filt._strength_prior.gamma_shape),
                            prior_gamma_scale=float(filt._strength_prior.gamma_scale),
                        )
                        sampled = block_proposal.sample(rng=rng)
                        state_prior_rows = prior_rows[state_rows]
                        if np.any(state_prior_rows):
                            sampled[state_prior_rows] = np.asarray(
                                filt._strength_prior.sample(
                                    (
                                        int(np.count_nonzero(state_prior_rows)),
                                        int(cardinality),
                                    ),
                                    rng=rng,
                                ),
                                dtype=np.float64,
                            )
                        strengths[source_indices] = sampled
                        isotope_log_prior_density[state_rows] += np.sum(
                            np.asarray(
                                filt._strength_prior.log_prob(sampled),
                                dtype=np.float64,
                            ),
                            axis=1,
                        )
                        isotope_log_guided_density[state_rows] += (
                            block_proposal.log_density(sampled)
                        )
                joint_log_prior_density += isotope_log_prior_density
                joint_log_guided_density += isotope_log_guided_density
                particles: list[IsotopeParticle] = []
                for row, cardinality in enumerate(cardinalities.tolist()):
                    begin = int(offsets[row])
                    end = int(offsets[row + 1])
                    state = IsotopeState(
                        num_sources=int(cardinality),
                        strengths=np.asarray(
                            strengths[begin:end],
                            dtype=np.float64,
                        ).copy(),
                        surface_chart_ids=np.asarray(
                            chart_ids[begin:end],
                            dtype=np.int64,
                        ).copy(),
                        surface_uv=np.asarray(
                            surface_uv[begin:end],
                            dtype=np.float64,
                        ).copy(),
                    )
                    filt._canonicalize_structural_rj_state(state)
                    particles.append(
                        IsotopeParticle(
                            state=state,
                            log_weight=0.0,
                            joint_row_identity=identities[row],
                        )
                    )
                particles_by_isotope[isotope] = particles
                filt.continuous_particles = particles
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None
                if isotope_index + 1 < len(order) and realized_prior_probability < 1.0:
                    explained = self._joint_station_expected_means_torch(station)
                    guided_indices = np.flatnonzero(~prior_rows)
                    if guided_indices.size == 0:
                        raise RuntimeError(
                            "Sequential guided initialization has no guided "
                            "rows from which to form a residual."
                        )
                    import torch

                    selected = torch.as_tensor(
                        guided_indices,
                        dtype=torch.long,
                        device=explained.device,
                    )
                    reference = torch.mean(
                        torch.index_select(explained, 0, selected),
                        dim=0,
                    )
                    self._joint_birth_proposal_reference_mean_vb = (
                        reference.detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                    if (
                        self._joint_birth_proposal_reference_mean_vb.shape
                        != station.spectrum_vb.shape
                        or np.any(
                            ~np.isfinite(self._joint_birth_proposal_reference_mean_vb)
                        )
                        or np.any(self._joint_birth_proposal_reference_mean_vb < 0.0)
                    ):
                        raise RuntimeError(
                            "Sequential guided reference mean is invalid."
                        )
                    self._joint_birth_proposal_station_score_cache.clear()
                    self._joint_birth_proposal_station_score_cache_order.clear()
        finally:
            self._active_joint_station_history = None
            self._joint_birth_proposal_reference_mean_vb = None
            for filt in self.filters.values():
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None
            for isotope, particles in original_particles.items():
                self.filters[isotope].continuous_particles = particles
        if realized_prior_probability >= 1.0:
            joint_log_proposal_density = joint_log_prior_density.copy()
        else:
            joint_log_proposal_density = np.logaddexp(
                math.log(realized_prior_probability) + joint_log_prior_density,
                math.log1p(-realized_prior_probability) + joint_log_guided_density,
            )
        log_importance_ratio = joint_log_prior_density - joint_log_proposal_density
        if np.any(~np.isfinite(log_importance_ratio)):
            raise RuntimeError(
                "Guided initialization importance correction is invalid."
            )
        normalized_log_weights = log_importance_ratio - logsumexp(log_importance_ratio)
        for isotope in order:
            particles = particles_by_isotope[isotope]
            for row, particle in enumerate(particles):
                particle.log_weight = float(normalized_log_weights[row])
            self.filters[isotope].continuous_particles = particles
        normalized_weights = np.exp(normalized_log_weights)
        self.last_joint_guided_initialization_ess = float(
            1.0 / np.sum(np.square(normalized_weights))
        )
        self._joint_guided_initialization_applied = True
        self._invalidate_posterior_summary_cache()
        self._assert_joint_particle_alignment()

    def _joint_tempered_station_update(
        self,
        station: JointStationObservation,
    ) -> None:
        """Run one station update and materialize CUDA state at its boundary."""
        try:
            self._joint_tempered_station_update_impl(station)
        finally:
            for filt in self.filters.values():
                filt._end_continuous_rj_station_device_state()

    def _joint_tempered_station_update_impl(
        self,
        station: JointStationObservation,
    ) -> None:
        """Assimilate one station with common weights and aligned SMC ancestors."""
        import torch

        all_stations = tuple((*self._joint_station_history, station))
        for filt in self.filters.values():
            filt.reset_step_stats()
        self.last_joint_rejuvenation_diagnostics = []
        self.last_joint_smc_wall_time_limit_exceeded = False
        self.last_joint_rejuvenation_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete_by_isotope = {
            str(isotope): False for isotope in self.joint_isotope_order()
        }
        station_start = time.perf_counter()
        station_cache_reuse_start = int(
            self.last_joint_station_transport_cache_reuse_count
        )
        self._apply_joint_guided_initialization(station)
        self.last_joint_resample_indices = np.empty(0, dtype=np.int64)
        reference_filter = self.filters[self.joint_isotope_order()[0]]
        common_log_weights = self._assert_joint_particle_alignment()
        self._refresh_joint_structural_transport_cache(all_stations)
        station_log_likelihood = self._joint_station_log_likelihood_torch(station)
        for filt in self.filters.values():
            filt._begin_continuous_rj_station_device_state(
                station_log_likelihood,
            )
        device = station_log_likelihood.device
        log_weights = torch.as_tensor(
            common_log_weights,
            dtype=torch.float64,
            device=device,
        )
        log_weights = reference_filter._normalized_log_weights_torch(log_weights)
        self._assign_joint_log_weights(log_weights.detach().cpu().numpy())
        initial_ess = reference_filter._ess_from_logw_torch(log_weights)
        target_ess = float(self.pf_config.target_ess_ratio) * int(log_weights.numel())
        particle_count = int(log_weights.numel())
        station_ancestor_ids = np.arange(particle_count, dtype=np.int64)
        if self._joint_cumulative_lineage_ids is None:
            if self._joint_station_history:
                raise RuntimeError(
                    "Cumulative PF lineage is missing after prior station updates."
                )
            self._joint_cumulative_lineage_ids = np.arange(
                particle_count,
                dtype=np.int64,
            )
        cumulative_lineage_ids = (
            np.asarray(
                self._joint_cumulative_lineage_ids,
                dtype=np.int64,
            )
            .reshape(-1)
            .copy()
        )
        if cumulative_lineage_ids.shape != (particle_count,):
            raise RuntimeError(
                "Cumulative PF lineage does not match aligned particle rows."
            )
        self._synchronize_joint_lineage_recovery_epoch(cumulative_lineage_ids)
        resamples = 0
        steps: list[dict[str, float]] = []
        view_count = int(station.fe_indices.size)

        def _station_likelihood() -> "torch.Tensor":
            """Return the exact full-station target for aligned states."""
            self._refresh_joint_structural_transport_cache(all_stations)
            return self._joint_station_log_likelihood_torch(station).to(
                device=device,
                dtype=torch.float64,
            )

        station_log_likelihood = station_log_likelihood.to(
            device=device,
            dtype=torch.float64,
        )
        if initial_ess <= target_ess + 1.0e-9:
            indices = self._resample_joint_particles(log_weights.detach().cpu().numpy())
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            self._synchronize_joint_lineage_recovery_epoch(
                cumulative_lineage_ids,
            )
            resamples += 1
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=0.0,
                station_start_s=station_start,
            )
            station_log_likelihood = _station_likelihood()
            log_weights = torch.full(
                (particle_count,),
                -math.log(max(particle_count, 1)),
                dtype=torch.float64,
                device=device,
            )
        max_steps = int(self.pf_config.max_temper_steps)
        beta_total = 0.0
        while beta_total < 1.0 - 1.0e-12:
            if len(steps) >= max_steps:
                raise RuntimeError(
                    "Joint full-station SMC reached max_temper_steps before "
                    "the station reached beta=1."
                )
            likelihood = station_log_likelihood
            try:
                delta_beta, proposed_log_weights, ess = (
                    reference_filter._select_delta_beta(
                        logw_prev=log_weights,
                        ll_t=likelihood,
                        remaining=1.0 - beta_total,
                        target_ess=target_ess,
                    )
                )
            except TemperingIncrementRequiresRejuvenation:
                current_ess = reference_filter._ess_from_logw_torch(log_weights)
                resampled = current_ess < particle_count - 1.0e-9
                if resampled:
                    indices = self._resample_joint_particles(
                        log_weights.detach().cpu().numpy()
                    )
                    station_ancestor_ids = station_ancestor_ids[indices]
                    cumulative_lineage_ids = cumulative_lineage_ids[indices]
                    self._synchronize_joint_lineage_recovery_epoch(
                        cumulative_lineage_ids,
                    )
                    resamples += 1
                    log_weights = torch.full(
                        (particle_count,),
                        -math.log(particle_count),
                        dtype=torch.float64,
                        device=device,
                    )
                self._joint_rejuvenate_adaptive(
                    all_stations,
                    target_beta=float(beta_total),
                    station_start_s=station_start,
                )
                station_log_likelihood = _station_likelihood()
                steps.append(
                    {
                        "station_view_count": float(view_count),
                        "station_beta": float(beta_total),
                        "beta_total": float(beta_total),
                        "delta_beta": 0.0,
                        "ess": float(current_ess),
                        "resampled": float(resampled),
                        "recovery_rejuvenation": 1.0,
                        "station_unique_ancestors": float(
                            np.unique(station_ancestor_ids).size
                        ),
                        "cumulative_unique_ancestors": float(
                            np.unique(cumulative_lineage_ids).size
                        ),
                    }
                )
                continue
            log_weights = proposed_log_weights
            self._assign_joint_log_weights(log_weights.detach().cpu().numpy())
            beta_total += float(delta_beta)
            step = {
                "station_view_count": float(view_count),
                "station_beta": float(beta_total),
                "beta_total": float(beta_total),
                "delta_beta": float(delta_beta),
                "ess": float(ess),
                "resampled": 0.0,
                "recovery_rejuvenation": 0.0,
                "station_unique_ancestors": float(
                    np.unique(station_ancestor_ids).size
                ),
                "cumulative_unique_ancestors": float(
                    np.unique(cumulative_lineage_ids).size
                ),
            }
            steps.append(step)
            if beta_total >= 1.0 - 1.0e-12:
                beta_total = 1.0
                break
            indices = self._resample_joint_particles(
                log_weights.detach().cpu().numpy()
            )
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            self._synchronize_joint_lineage_recovery_epoch(
                cumulative_lineage_ids,
            )
            resamples += 1
            step["resampled"] = 1.0
            step["station_unique_ancestors"] = float(
                np.unique(station_ancestor_ids).size
            )
            step["cumulative_unique_ancestors"] = float(
                np.unique(cumulative_lineage_ids).size
            )
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=float(beta_total),
                station_start_s=station_start,
            )
            station_log_likelihood = _station_likelihood()
            log_weights = torch.full(
                (particle_count,),
                -math.log(max(particle_count, 1)),
                dtype=torch.float64,
                device=device,
            )
        lineage_recovery_required = self._synchronize_joint_lineage_recovery_epoch(
            cumulative_lineage_ids,
        )
        self._joint_rejuvenate_adaptive(
            all_stations,
            target_beta=1.0,
            station_start_s=station_start,
            enforce_lineage_recovery=lineage_recovery_required,
        )
        normalized = self._strict_joint_particle_weights()
        final_ess = 1.0 / float(np.sum(normalized**2))
        if final_ess + 1.0e-9 < target_ess:
            raise RuntimeError("Completed joint tempering did not preserve target ESS.")
        station_unique_ancestors = int(np.unique(station_ancestor_ids).size)
        cumulative_unique_ancestors = int(np.unique(cumulative_lineage_ids).size)
        self._joint_cumulative_lineage_ids = cumulative_lineage_ids
        self.last_joint_temper_steps = steps
        self.last_joint_station_unique_ancestor_count = station_unique_ancestors
        self.last_joint_cumulative_unique_ancestor_count = cumulative_unique_ancestors
        for filt in self.filters.values():
            filt.last_temper_steps = [dict(step) for step in steps]
            filt.last_temper_resample_count = int(resamples)
            filt.last_temper_min_ess = float(
                min((step["ess"] for step in steps), default=final_ess)
            )
            filt.last_station_unique_ancestor_count = station_unique_ancestors
            filt.last_cumulative_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_ess_pre = float(initial_ess)
            filt.last_ess = float(final_ess)
            filt.last_ess_post = float(final_ess)
            filt.last_resample_ess = bool(resamples)
        self._promote_joint_birth_proposal_station(station)
        self._joint_station_history.append(station)
        self._assert_joint_particle_alignment()
        station_cache_reuses = (
            self.last_joint_station_transport_cache_reuse_count
            - station_cache_reuse_start
        )
        print(
            "[joint-smc] station-update-done "
            f"station={len(all_stations) - 1} "
            f"elapsed_s={time.perf_counter() - station_start:.3f} "
            f"temper_steps={len(steps)} "
            f"resamples={resamples} "
            "station_transport_cache_reuses="
            f"{station_cache_reuses} "
            f"final_ess={final_ess:.6f}",
            flush=True,
        )
