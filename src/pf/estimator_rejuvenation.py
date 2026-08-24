"""Joint exact-RJ rejuvenation and tempered SMC algorithms."""

from __future__ import annotations

import hashlib
import math
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from pf.estimator_compat import runtime_estimator_export
from pf.estimator_sampling import _stratified_joint_cardinality_draws
from pf.estimator_types import JointStationObservation
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    JointRowIdentity,
    TemperingIncrementRequiresRejuvenation,
)
from pf.randomness import named_random_generator
from pf.state import IsotopeState
from pf.structural_rj import (
    ContinuousBlockStrengthProposal,
    cross_isotope_transfer_log_proposal,
    shifted_log_strength_random_walk_log_reverse_ratio,
)

if TYPE_CHECKING:
    import torch


class JointRejuvenationMixin:
    """Provide joint state moves, resampling, mixing, and tempering."""

    @staticmethod
    def _joint_isotope_state_log_prior(
        filt: IsotopeParticleFilter,
        state: IsotopeState,
    ) -> float:
        """Return the exact labeled continuous-state prior log density."""
        atlas = filt._structural_rj_surface_atlas
        cardinality_prior = filt._structural_rj_cardinality_prior
        if atlas is None or cardinality_prior is None:
            raise RuntimeError("Cross-isotope transfer priors are unavailable.")
        cardinality = int(state.num_sources)
        value = float(cardinality_prior.log_prob(cardinality)) + math.lgamma(
            float(cardinality) + 1.0
        )
        if cardinality:
            value += float(
                np.sum(
                    atlas.log_chart_probabilities[state.surface_chart_ids]
                    + np.asarray(
                        filt._strength_prior.log_prob(state.strengths),
                        dtype=np.float64,
                    )
                )
            )
        return value

    @staticmethod
    def _cross_isotope_transferred_states(
        donor_filter: IsotopeParticleFilter,
        receiver_filter: IsotopeParticleFilter,
        donor_state: IsotopeState,
        receiver_state: IsotopeState,
        transferred_indices: NDArray[np.int64],
    ) -> tuple[IsotopeState, IsotopeState]:
        """Return canonical donor/receiver states after identity transfer."""
        selected = np.asarray(transferred_indices, dtype=np.int64).reshape(-1)
        donor_count = int(donor_state.num_sources)
        if (
            selected.size == 0
            or np.unique(selected).size != selected.size
            or np.any(selected < 0)
            or np.any(selected >= donor_count)
        ):
            raise ValueError("Cross-isotope transferred indices are invalid.")
        keep = np.ones(donor_count, dtype=bool)
        keep[selected] = False
        new_donor = IsotopeState(
            num_sources=int(np.sum(keep)),
            strengths=donor_state.strengths[keep],
            surface_chart_ids=donor_state.surface_chart_ids[keep],
            surface_uv=donor_state.surface_uv[keep],
        )
        new_receiver = IsotopeState(
            num_sources=int(receiver_state.num_sources + selected.size),
            strengths=np.concatenate(
                (receiver_state.strengths, donor_state.strengths[selected])
            ),
            surface_chart_ids=np.concatenate(
                (
                    receiver_state.surface_chart_ids,
                    donor_state.surface_chart_ids[selected],
                )
            ),
            surface_uv=np.concatenate(
                (receiver_state.surface_uv, donor_state.surface_uv[selected]),
                axis=0,
            ),
        )
        donor_filter._canonicalize_structural_rj_state(new_donor)
        receiver_filter._canonicalize_structural_rj_state(new_receiver)
        return new_donor, new_receiver

    def _evaluate_cross_isotope_receiver_states(
        self,
        *,
        stations: Sequence[JointStationObservation],
        receiver_states: Mapping[int, IsotopeState],
        receiver_by_row: NDArray[np.int64],
        isotope_order: tuple[str, ...],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Evaluate batched receiver states with temporary donor cache active."""
        proposed_target = np.full(
            receiver_by_row.size,
            float("nan"),
            dtype=np.float64,
        )
        for receiver_index, isotope in enumerate(isotope_order):
            rows_for_isotope = np.flatnonzero(receiver_by_row == receiver_index)
            if rows_for_isotope.size == 0:
                continue
            filt = self.filters[isotope]
            evidence = self._joint_history_structural_geometry(
                isotope,
                stations,
            )
            cardinalities = np.asarray(
                [receiver_states[int(row)].num_sources for row in rows_for_isotope],
                dtype=np.int64,
            )
            for cardinality in np.unique(cardinalities).tolist():
                local = np.flatnonzero(cardinalities == int(cardinality))
                rows = rows_for_isotope[local]
                states = [receiver_states[int(row)] for row in rows]
                if int(cardinality):
                    chart_ids = np.stack(
                        [state.surface_chart_ids for state in states],
                        axis=0,
                    )
                    strengths = np.stack(
                        [state.strengths for state in states],
                        axis=0,
                    )
                    positions = np.stack(
                        [filt.continuous_state_positions(state) for state in states],
                        axis=0,
                    )
                else:
                    chart_ids = np.empty((rows.size, 0), dtype=np.int64)
                    strengths = np.empty((rows.size, 0), dtype=np.float64)
                    positions = np.empty((rows.size, 0, 3), dtype=np.float64)
                proposed_target[rows] = self._joint_structural_target_evaluator(
                    filt=filt,
                    data=evidence,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    strengths_pk=strengths,
                    particle_indices=rows,
                    target_beta=float(target_beta),
                    tempering_start_row=sum(
                        int(station.fe_indices.size) for station in stations[:-1]
                    ),
                )
        evaluated_rows = np.asarray(
            sorted(receiver_states),
            dtype=np.int64,
        )
        if np.any(~np.isfinite(proposed_target[evaluated_rows])):
            raise RuntimeError("Cross-isotope target evaluation was incomplete.")
        return proposed_target

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
                    newest_prefix_count=(self._active_joint_tempering_prefix_count),
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
                    newest_prefix_count=(self._active_joint_tempering_prefix_count),
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
                total_nvsl=torch.index_select(cache[0], 0, selected),
                uncollided_nvsl=torch.index_select(cache[1], 0, selected),
                features_nvslf=torch.index_select(cache[2], 0, selected),
                target_beta=float(target_beta),
                newest_prefix_count=self._active_joint_tempering_prefix_count,
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=np.asarray(cache[0][rows]),
            uncollided_nvsl=np.asarray(cache[1][rows]),
            features_nvslf=np.asarray(cache[2][rows]),
            target_beta=float(target_beta),
            newest_prefix_count=self._active_joint_tempering_prefix_count,
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
            cardinalities = np.asarray(
                [filt.continuous_particles[int(row)].state.num_sources for row in rows],
                dtype=np.int64,
            )
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
        if signatures is None:
            self._joint_persistent_structural_state_sha256 = None
        else:
            self._joint_persistent_structural_station_signature = signatures
            self._joint_persistent_structural_state_sha256 = (
                self._joint_structural_state_sha256()
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
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply one exact all-isotope strength move without conservation.

        Each active source receives a symmetric Gaussian increment in shifted
        log-strength coordinates.  Isotope totals are not constrained or
        transferred.  The shared spectrum likelihood alone decides whether a
        simultaneous decrease in one isotope and increase in another is useful.
        """
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
        minimum = float(self.pf_config.strength_prior_min_cps_1m)
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

    def _apply_joint_cross_isotope_transfer(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply an exact multi-source isotope-identity transfer proposal.

        A uniformly selected subset of one isotope is relabeled as another
        isotope while retaining its continuous surface coordinates and
        strengths.  The state-dependent subset and group-size probabilities,
        both isotope priors, and the full joint likelihood are all included in
        the MH ratio.  This crosses the likelihood barrier created when a clear
        peak has been assigned to the wrong isotope during early resampling.
        """
        probability = float(self.pf_config.joint_cross_isotope_transfer_probability)
        isotope_order = self.joint_isotope_order()
        if probability <= 0.0 or len(isotope_order) < 2:
            return np.asarray(current_target_log_likelihood, dtype=np.float64)
        particle_count = int(self.pf_config.num_particles)
        weights = self._strict_joint_particle_weights()
        rng = self._joint_random_generator
        attempted = rng.random(particle_count) < probability
        donor_by_row = rng.integers(
            0,
            len(isotope_order),
            size=particle_count,
        )
        receiver_offset = rng.integers(
            1,
            len(isotope_order),
            size=particle_count,
        )
        receiver_by_row = (donor_by_row + receiver_offset) % len(isotope_order)
        cardinalities = np.stack(
            [
                np.asarray(
                    [
                        particle.state.num_sources
                        for particle in self.filters[isotope].continuous_particles
                    ],
                    dtype=np.int64,
                )
                for isotope in isotope_order
            ],
            axis=1,
        )
        row_indices = np.arange(particle_count, dtype=np.int64)
        donor_cardinality = cardinalities[row_indices, donor_by_row]
        receiver_cardinality = cardinalities[row_indices, receiver_by_row]
        maximum_sources = self.pf_config.cardinality_capacity
        maximum_group = np.minimum.reduce(
            (
                donor_cardinality,
                maximum_sources - receiver_cardinality,
                np.full(
                    particle_count,
                    int(self.pf_config.joint_cross_isotope_transfer_max_group),
                    dtype=np.int64,
                ),
            )
        )
        feasible = attempted & (maximum_group > 0)
        attempted_rows = np.flatnonzero(attempted)
        rows = np.flatnonzero(feasible)
        self.last_joint_cross_isotope_attempted_weight_mass += float(
            np.sum(weights[attempted_rows])
        )
        if rows.size == 0:
            self.last_joint_cross_isotope_rejection_diagnostics = {
                "attempted": int(attempted_rows.size),
                "accepted": 0,
                "support_rejected": int(attempted_rows.size),
                "geometry_support_rejected": 0,
                "strength_support_rejected": 0,
                "other_support_rejected": int(attempted_rows.size),
                "nonfinite_rejected": 0,
                "mh_random_rejected": 0,
                "component_quantiles": {},
                "by_isotope_cardinality_transfer": {},
            }
            return np.asarray(current_target_log_likelihood, dtype=np.float64)
        group_sizes = np.ones(particle_count, dtype=np.int64)
        group_sizes[rows] = (
            np.floor(rng.random(rows.size) * maximum_group[rows]).astype(np.int64) + 1
        )
        donor_states: dict[int, IsotopeState] = {}
        receiver_states: dict[int, IsotopeState] = {}
        old_donor_states: dict[int, IsotopeState] = {}
        old_receiver_states: dict[int, IsotopeState] = {}
        log_forward = np.full(particle_count, float("-inf"), dtype=np.float64)
        log_reverse = np.full(particle_count, float("-inf"), dtype=np.float64)
        log_prior_ratio = np.zeros(particle_count, dtype=np.float64)
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            donor_state = donor_filter.continuous_particles[row].state
            receiver_state = receiver_filter.continuous_particles[row].state
            group_size = int(group_sizes[row])
            selected = np.sort(
                rng.choice(
                    int(donor_state.num_sources),
                    size=group_size,
                    replace=False,
                )
            )
            proposed_donor, proposed_receiver = self._cross_isotope_transferred_states(
                donor_filter,
                receiver_filter,
                donor_state,
                receiver_state,
                selected,
            )
            donor_states[row] = proposed_donor
            receiver_states[row] = proposed_receiver
            old_donor_states[row] = donor_state.copy()
            old_receiver_states[row] = receiver_state.copy()
            log_forward[row] = cross_isotope_transfer_log_proposal(
                donor_cardinality=int(donor_state.num_sources),
                receiver_cardinality=int(receiver_state.num_sources),
                group_size=group_size,
                maximum_sources=maximum_sources,
                maximum_group_size=int(
                    self.pf_config.joint_cross_isotope_transfer_max_group
                ),
            )
            log_reverse[row] = cross_isotope_transfer_log_proposal(
                donor_cardinality=int(proposed_receiver.num_sources),
                receiver_cardinality=int(proposed_donor.num_sources),
                group_size=group_size,
                maximum_sources=maximum_sources,
                maximum_group_size=int(
                    self.pf_config.joint_cross_isotope_transfer_max_group
                ),
            )
            old_prior = self._joint_isotope_state_log_prior(
                donor_filter,
                donor_state,
            ) + self._joint_isotope_state_log_prior(
                receiver_filter,
                receiver_state,
            )
            new_prior = self._joint_isotope_state_log_prior(
                donor_filter,
                proposed_donor,
            ) + self._joint_isotope_state_log_prior(
                receiver_filter,
                proposed_receiver,
            )
            log_prior_ratio[row] = new_prior - old_prior

        touched_donors = sorted(set(donor_by_row[rows].tolist()))
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            donor_filter.continuous_particles[row].state = donor_states[row]
        try:
            for isotope_index in touched_donors:
                isotope_rows = rows[donor_by_row[rows] == isotope_index]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
            proposed_target = self._evaluate_cross_isotope_receiver_states(
                stations=stations,
                receiver_states=receiver_states,
                receiver_by_row=np.where(
                    feasible,
                    receiver_by_row,
                    -1,
                ),
                isotope_order=isotope_order,
                target_beta=float(target_beta),
            )
        finally:
            for row in rows.tolist():
                donor_filter = self.filters[isotope_order[donor_by_row[row]]]
                donor_filter.continuous_particles[row].state = old_donor_states[row]
            for isotope_index in touched_donors:
                isotope_rows = rows[donor_by_row[rows] == isotope_index]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
        current_target = np.asarray(
            current_target_log_likelihood,
            dtype=np.float64,
        )
        log_ratio = (
            proposed_target[rows]
            - current_target[rows]
            + log_prior_ratio[rows]
            + log_reverse[rows]
            - log_forward[rows]
        )
        accepted_local = self._joint_mh_acceptance_mask(
            log_ratio,
            rng=rng,
        )
        accepted_rows = rows[accepted_local]
        for row in accepted_rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            donor_filter.continuous_particles[row].state = donor_states[row]
            receiver_filter.continuous_particles[row].state = receiver_states[row]
        if accepted_rows.size:
            touched = sorted(
                set(donor_by_row[accepted_rows].tolist())
                | set(receiver_by_row[accepted_rows].tolist())
            )
            for isotope_index in touched:
                isotope_rows = accepted_rows[
                    (donor_by_row[accepted_rows] == isotope_index)
                    | (receiver_by_row[accepted_rows] == isotope_index)
                ]
                self._refresh_joint_structural_transport_cache_isotope(
                    stations,
                    isotope_order[isotope_index],
                    particle_indices=isotope_rows,
                )
            current_target = current_target.copy()
            current_target[accepted_rows] = proposed_target[accepted_rows]
            self.last_joint_cross_isotope_accepted_weight_mass += float(
                np.sum(weights[accepted_rows])
            )
            self._invalidate_posterior_summary_cache()
        diagnostic_delta_likelihood = np.full(
            particle_count,
            float("nan"),
            dtype=np.float64,
        )
        diagnostic_log_ratio = np.full_like(
            diagnostic_delta_likelihood,
            float("nan"),
        )
        diagnostic_delta_likelihood[rows] = (
            proposed_target[rows] - current_target_log_likelihood[rows]
        )
        diagnostic_log_ratio[rows] = log_ratio
        diagnostic_accepted = np.zeros(particle_count, dtype=np.bool_)
        diagnostic_accepted[accepted_rows] = True
        diagnostic_strength_support = np.ones(
            particle_count,
            dtype=np.bool_,
        )
        for row in rows.tolist():
            donor_filter = self.filters[isotope_order[donor_by_row[row]]]
            receiver_filter = self.filters[isotope_order[receiver_by_row[row]]]
            diagnostic_strength_support[row] = bool(
                np.all(
                    donor_filter._strength_prior.in_support(donor_states[row].strengths)
                )
                and np.all(
                    receiver_filter._strength_prior.in_support(
                        receiver_states[row].strengths
                    )
                )
            )
        diagnostic_support = (
            feasible
            & diagnostic_strength_support
            & np.isfinite(log_forward)
            & np.isfinite(log_reverse)
            & np.isfinite(log_prior_ratio)
        )
        diagnostic_proposal_ratio = np.full(
            particle_count,
            float("nan"),
            dtype=np.float64,
        )
        diagnostic_proposal_ratio[rows] = log_reverse[rows] - log_forward[rows]
        self.last_joint_cross_isotope_rejection_diagnostics = (
            self._summarize_joint_cross_isotope_transfer(
                attempted_rows=attempted_rows,
                donor_by_row=donor_by_row,
                receiver_by_row=receiver_by_row,
                donor_cardinality=donor_cardinality,
                receiver_cardinality=receiver_cardinality,
                group_sizes=group_sizes,
                isotope_order=isotope_order,
                delta_log_likelihood=diagnostic_delta_likelihood,
                delta_log_prior=log_prior_ratio,
                log_reverse_minus_forward=diagnostic_proposal_ratio,
                log_acceptance_ratio=diagnostic_log_ratio,
                support_feasible=diagnostic_support,
                strength_support_feasible=diagnostic_strength_support,
                accepted=diagnostic_accepted,
            )
        )
        self._assert_joint_particle_alignment()
        return current_target

    @staticmethod
    def _summarize_joint_cross_isotope_transfer(
        *,
        attempted_rows: NDArray[np.int64],
        donor_by_row: NDArray[np.int64],
        receiver_by_row: NDArray[np.int64],
        donor_cardinality: NDArray[np.int64],
        receiver_cardinality: NDArray[np.int64],
        group_sizes: NDArray[np.int64],
        isotope_order: Sequence[str],
        delta_log_likelihood: NDArray[np.float64],
        delta_log_prior: NDArray[np.float64],
        log_reverse_minus_forward: NDArray[np.float64],
        log_acceptance_ratio: NDArray[np.float64],
        support_feasible: NDArray[np.bool_],
        strength_support_feasible: NDArray[np.bool_],
        accepted: NDArray[np.bool_],
    ) -> dict[str, object]:
        """Summarize exact isotope-transfer MH terms on attempted rows."""
        selected = np.asarray(attempted_rows, dtype=np.int64).reshape(-1)
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )
        numeric = {
            "delta_log_likelihood": np.asarray(
                delta_log_likelihood,
                dtype=np.float64,
            ),
            "delta_log_prior": np.asarray(
                delta_log_prior,
                dtype=np.float64,
            ),
            "log_reverse_minus_forward": np.asarray(
                log_reverse_minus_forward,
                dtype=np.float64,
            ),
            "log_jacobian": np.zeros(
                np.asarray(delta_log_likelihood).shape,
                dtype=np.float64,
            ),
            "log_acceptance_ratio": np.asarray(
                log_acceptance_ratio,
                dtype=np.float64,
            ),
        }
        support = np.asarray(support_feasible, dtype=np.bool_)
        strength_support = np.asarray(
            strength_support_feasible,
            dtype=np.bool_,
        )
        acceptance = np.asarray(accepted, dtype=np.bool_)

        def _rows_summary(rows: NDArray[np.int64]) -> dict[str, object]:
            """Summarize one batched subset of transfer attempts."""
            finite_all = support[rows].copy()
            component_quantiles: dict[
                str,
                dict[str, float | int] | None,
            ] = {}
            for name, all_values in numeric.items():
                values = all_values[rows]
                finite = np.isfinite(values)
                finite_all &= finite
                if not np.any(finite):
                    component_quantiles[name] = None
                    continue
                finite_values = values[finite]
                resolved = np.quantile(finite_values, quantile_levels)
                component_quantiles[name] = {
                    "finite_count": int(finite_values.size),
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    **{
                        label: float(value)
                        for label, value in zip(
                            ("min", "p10", "median", "p90", "max"),
                            resolved,
                            strict=True,
                        )
                    },
                }
            return {
                "attempted": int(rows.size),
                "accepted": int(np.count_nonzero(acceptance[rows])),
                "support_rejected": int(np.count_nonzero(~support[rows])),
                "geometry_support_rejected": 0,
                "strength_support_rejected": int(
                    np.count_nonzero(~strength_support[rows])
                ),
                "other_support_rejected": int(
                    np.count_nonzero(strength_support[rows] & ~support[rows])
                ),
                "nonfinite_rejected": int(
                    np.count_nonzero(support[rows] & ~finite_all)
                ),
                "mh_random_rejected": int(
                    np.count_nonzero(support[rows] & finite_all & ~acceptance[rows])
                ),
                "component_quantiles": component_quantiles,
            }

        summary = _rows_summary(selected)
        transitions: dict[str, object] = {}
        # Isotope pairs and group sizes are bounded by the configured isotope
        # set and max transfer group, so this packages batched arrays only.
        if selected.size:
            labels = np.stack(
                (
                    np.asarray(donor_by_row, dtype=np.int64),
                    np.asarray(receiver_by_row, dtype=np.int64),
                    np.asarray(donor_cardinality, dtype=np.int64),
                    np.asarray(receiver_cardinality, dtype=np.int64),
                    np.asarray(group_sizes, dtype=np.int64),
                ),
                axis=1,
            )
            for donor, receiver, donor_k, receiver_k, group_size in np.unique(
                labels[selected],
                axis=0,
            ).tolist():
                matching = selected[
                    np.all(
                        labels[selected]
                        == np.asarray(
                            (
                                donor,
                                receiver,
                                donor_k,
                                receiver_k,
                                group_size,
                            ),
                            dtype=np.int64,
                        ),
                        axis=1,
                    )
                ]
                key = (
                    f"{isotope_order[int(donor)]}:{int(donor_k)}"
                    f"->{int(donor_k) - int(group_size)}|"
                    f"{isotope_order[int(receiver)]}:{int(receiver_k)}"
                    f"->{int(receiver_k) + int(group_size)}"
                )
                transitions[key] = _rows_summary(matching)
        summary["by_isotope_cardinality_transfer"] = transitions
        return summary

    def _apply_joint_cross_isotope_state_block(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        current_target_log_likelihood: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Jointly propose independent isotope states without strength transfer.

        Each isotope retains its own cardinality, surface-position, and strength
        priors. The proposal is coupled only by making one simultaneous MH
        decision under the shared full-spectrum likelihood; no activity or
        detector-cps quantity is conserved between isotope labels.
        """
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
        old_states: dict[str, list[IsotopeState]] = {}
        proposed_states: dict[str, list[IsotopeState]] = {}
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
                old = [filt.continuous_particles[int(row)].state.copy() for row in rows]
                old_states[isotope] = old
                old_cardinalities = np.asarray(
                    [state.num_sources for state in old],
                    dtype=np.int64,
                )
                for cardinality in np.unique(old_cardinalities).tolist():
                    selected = np.flatnonzero(old_cardinalities == int(cardinality))
                    states = [old[int(index)] for index in selected]
                    charts = (
                        np.stack(
                            [state.surface_chart_ids for state in states],
                            axis=0,
                        )
                        if int(cardinality)
                        else np.empty((selected.size, 0), dtype=np.int64)
                    )
                    strengths = (
                        np.stack(
                            [state.strengths for state in states],
                            axis=0,
                        )
                        if int(cardinality)
                        else np.empty((selected.size, 0), dtype=np.float64)
                    )
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
                generated: list[IsotopeState | None] = [None] * rows.size
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
                    for local_index, destination in enumerate(selected.tolist()):
                        state = IsotopeState(
                            num_sources=int(cardinality),
                            strengths=strength_batch[local_index],
                            surface_chart_ids=charts_batch[local_index],
                            surface_uv=uv_batch[local_index],
                        )
                        filt._canonicalize_structural_rj_state(state)
                        generated[int(destination)] = state
                if any(state is None for state in generated):
                    raise RuntimeError(
                        "Joint isotope-state proposal generation was incomplete."
                    )
                proposed_states[isotope] = [
                    state for state in generated if isinstance(state, IsotopeState)
                ]
            finally:
                self._active_joint_structural_geometry = None
                filt._structural_rj_position_proposal = None
                filt._structural_rj_strength_proposal = None

        def _assign(candidate: Mapping[str, Sequence[IsotopeState]]) -> None:
            """Assign selected variable-length rows before a batched refresh."""
            for isotope in isotope_order:
                filt = self.filters[isotope]
                states = candidate[isotope]
                for local_index, row in enumerate(rows.tolist()):
                    filt.continuous_particles[int(row)].state = states[
                        local_index
                    ].copy()

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
            for isotope in isotope_order:
                filt = self.filters[isotope]
                states = proposed_states[isotope]
                for local_index in np.flatnonzero(accepted).tolist():
                    row = int(rows[local_index])
                    filt.continuous_particles[row].state = states[local_index].copy()
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
        systematic_resample = runtime_estimator_export("systematic_resample")
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
        new_particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
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
        persistent_cache = self._joint_persistent_structural_transport_cache
        if persistent_cache is not None:
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
        self._joint_row_generation = new_identities[0].generation
        self.last_joint_resample_indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        self._joint_persistent_structural_state_sha256 = (
            self._joint_structural_state_sha256()
            if self._joint_persistent_structural_transport_cache is not None
            else None
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
    ) -> tuple[NDArray[Any], ...]:
        """Copy compact row state used to find exact accepted cache deltas."""
        _, strengths, mask, chart_ids, surface_uv = (
            filt._packed_continuous_surface_state_arrays()
        )
        return tuple(
            np.ascontiguousarray(values).copy()
            for values in (strengths, mask, chart_ids, surface_uv)
        )

    @staticmethod
    def _joint_changed_cache_rows(
        before: tuple[NDArray[Any], ...],
        filt: IsotopeParticleFilter,
    ) -> NDArray[np.int64]:
        """Return PF rows whose accepted isotope state changed exactly."""
        estimator_type = runtime_estimator_export("RotatingShieldPFEstimator")
        after = estimator_type._joint_isotope_cache_state(filt)
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
        for isotope_index, isotope in enumerate(self.joint_isotope_order()):
            ordinary_maximum = int(self.filters[isotope].config.max_sources or 0)
            at_boundary = before_cardinality[:, isotope_index] >= ordinary_maximum
            escaped = at_boundary & (
                after_cardinality[:, isotope_index]
                < before_cardinality[:, isotope_index]
            )
            ordinary_boundary |= at_boundary
            ordinary_boundary_escape |= escaped
            diagnostics[f"ordinary_boundary_weight_mass.{isotope}"] = float(
                np.sum(weights[at_boundary])
            )
            diagnostics[f"ordinary_boundary_escape_weight_mass.{isotope}"] = float(
                np.sum(weights[escaped])
            )
        diagnostics["ordinary_boundary_weight_mass"] = float(
            np.sum(weights[ordinary_boundary])
        )
        diagnostics["ordinary_boundary_escape_weight_mass"] = float(
            np.sum(weights[ordinary_boundary_escape])
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
        newest_prefix_count: int | None = None,
    ) -> dict[str, float]:
        """Apply conditional exact-RJ sweeps under one prefix bridge target."""
        active = tuple(stations)
        if not active:
            return {}
        state_before = self._joint_mixing_snapshot()
        diagnostics: dict[str, float] = {}
        cross_attempted_start = float(
            self.last_joint_cross_isotope_attempted_weight_mass
        )
        cross_accepted_start = float(self.last_joint_cross_isotope_accepted_weight_mass)
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
        self._active_joint_station_history = active
        self._active_joint_tempering_prefix_count = newest_prefix_count
        newest_start = sum(int(station.fe_indices.size) for station in active[:-1])
        try:
            self._refresh_joint_structural_transport_cache(active)
            isotope_order = self.joint_isotope_order()
            cache = self._joint_structural_transport_cache
            if cache is None:
                raise RuntimeError(
                    "Joint structural transport cache was not initialized."
                )
            cache_is_torch = hasattr(cache[0], "detach")
            if cache_is_torch:
                current_target_tensor = self._joint_history_log_likelihood_torch(
                    filt=self.filters[isotope_order[0]],
                    stations=active,
                    total_nvsl=cache[0],
                    uncollided_nvsl=cache[1],
                    features_nvslf=cache[2],
                    target_beta=float(target_beta),
                    newest_prefix_count=newest_prefix_count,
                )
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
                        newest_prefix_count=newest_prefix_count,
                    )
                )
            target_before = np.asarray(
                current_target_log_likelihood,
                dtype=np.float64,
            ).copy()
            last_changed_rows = np.empty(0, dtype=np.int64)
            for isotope_index, isotope in enumerate(isotope_order):
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
                    )
                finally:
                    self._active_joint_structural_geometry = None
                self._assert_joint_particle_alignment()
                updated_target = filt.last_structural_target_log_likelihood
                if updated_target is None:
                    raise RuntimeError(
                        "Joint structural sweep did not return its target cache."
                    )
                current_target_log_likelihood = np.asarray(
                    updated_target,
                    dtype=np.float64,
                )
                changed_rows = self._joint_changed_cache_rows(
                    cache_state_before,
                    filt,
                )
                last_changed_rows = changed_rows
                if isotope_index + 1 < len(isotope_order):
                    self._refresh_joint_structural_transport_cache_isotope(
                        active,
                        isotope,
                        particle_indices=changed_rows,
                    )
            if last_changed_rows.size:
                self._refresh_joint_structural_transport_cache_isotope(
                    active,
                    isotope_order[-1],
                    particle_indices=last_changed_rows,
                )
            if float(self.pf_config.joint_strength_block_probability) > 0.0:
                current_target_log_likelihood = self._apply_joint_strength_block(
                    active,
                    target_beta=float(target_beta),
                    current_target_log_likelihood=(current_target_log_likelihood),
                )
            if float(self.pf_config.joint_cross_isotope_state_block_probability) > 0.0:
                current_target_log_likelihood = (
                    self._apply_joint_cross_isotope_state_block(
                        active,
                        target_beta=float(target_beta),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                )
            if float(self.pf_config.joint_cross_isotope_transfer_probability) > 0.0:
                current_target_log_likelihood = (
                    self._apply_joint_cross_isotope_transfer(
                        active,
                        target_beta=float(target_beta),
                        current_target_log_likelihood=(current_target_log_likelihood),
                    )
                )
            diagnostics = self._joint_mixing_diagnostics(
                state_before,
                self._joint_mixing_snapshot(),
                target_before=target_before,
                target_after=current_target_log_likelihood,
            )
            diagnostics["cross_isotope_attempted_weight_mass"] = float(
                self.last_joint_cross_isotope_attempted_weight_mass
                - cross_attempted_start
            )
            diagnostics["cross_isotope_accepted_weight_mass"] = float(
                self.last_joint_cross_isotope_accepted_weight_mass
                - cross_accepted_start
            )
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
        finally:
            self._invalidate_posterior_summary_cache()
            for filt in self.filters.values():
                filt._clear_continuous_rj_device_state()
            self._active_joint_structural_geometry = None
            self._joint_structural_transport_cache = None
            self._active_joint_station_history = None
            self._active_joint_tempering_prefix_count = None
        return diagnostics

    def _joint_rejuvenate_adaptive(
        self,
        stations: Sequence[JointStationObservation],
        *,
        target_beta: float,
        newest_prefix_count: int,
        station_start_s: float,
    ) -> None:
        """Run exact sweeps until movement is adequate or the soft budget binds."""
        minimum_sweeps = int(self.pf_config.joint_rejuvenation_min_sweeps)
        maximum_sweeps = int(self.pf_config.joint_rejuvenation_max_sweeps)
        soft_budget_s = float(self.pf_config.joint_smc_soft_wall_time_s)
        cumulative_k_transition_mass = 0.0
        cumulative_boundary_escape_mass = 0.0
        self.last_joint_rejuvenation_mixing_incomplete = True
        self.last_joint_structural_mixing_incomplete = False
        for sweep_index in range(maximum_sweeps):
            diagnostics = self._joint_rejuvenate(
                stations,
                target_beta=target_beta,
                newest_prefix_count=newest_prefix_count,
            )
            if not isinstance(diagnostics, Mapping):
                return
            record = {str(key): float(value) for key, value in diagnostics.items()}
            record.update(
                {
                    "prefix_count": float(newest_prefix_count),
                    "target_beta": float(target_beta),
                    "sweep_index": float(sweep_index + 1),
                    "station_elapsed_s": float(time.perf_counter() - station_start_s),
                }
            )
            self.last_joint_rejuvenation_diagnostics.append(record)
            k_transition_mass = float(diagnostics.get("k_transition_weight_mass", 0.0))
            cumulative_k_transition_mass += k_transition_mass
            boundary_mass = float(diagnostics.get("ordinary_boundary_weight_mass", 0.0))
            cumulative_boundary_escape_mass += float(
                diagnostics.get(
                    "ordinary_boundary_escape_weight_mass",
                    0.0,
                )
            )
            record["cumulative_k_transition_weight_mass"] = float(
                cumulative_k_transition_mass
            )
            record["cumulative_boundary_escape_weight_mass"] = float(
                cumulative_boundary_escape_mass
            )
            if sweep_index + 1 < minimum_sweeps:
                continue
            state_mass = float(diagnostics.get("state_change_weight_mass", 0.0))
            position_esjd = float(diagnostics.get("surface_position_esjd_m2", 0.0))
            strength_esjd = float(diagnostics.get("log_strength_esjd", 0.0))
            continuous_movement_sufficient = state_mass >= float(
                self.pf_config.joint_rejuvenation_min_state_change_weight_mass
            ) and (
                position_esjd
                >= float(self.pf_config.joint_rejuvenation_min_surface_esjd_m2)
                or strength_esjd
                >= float(self.pf_config.joint_rejuvenation_min_log_strength_esjd)
            )
            minimum_k_mass = float(
                self.pf_config.joint_rejuvenation_min_k_transition_weight_mass
            )
            boundary_requires_structure = bool(
                self.pf_config.variable_cardinality
            ) and boundary_mass > float(
                self.pf_config.joint_rejuvenation_boundary_mass_threshold
            )
            structural_movement_sufficient = not boundary_requires_structure or (
                cumulative_k_transition_mass >= minimum_k_mass
                and cumulative_boundary_escape_mass >= minimum_k_mass
            )
            record["continuous_movement_sufficient"] = float(
                continuous_movement_sufficient
            )
            record["structural_movement_required"] = float(boundary_requires_structure)
            record["structural_movement_sufficient"] = float(
                structural_movement_sufficient
            )
            if continuous_movement_sufficient and structural_movement_sufficient:
                self.last_joint_rejuvenation_mixing_incomplete = False
                break
            station_elapsed = time.perf_counter() - station_start_s
            if station_elapsed >= soft_budget_s and not boundary_requires_structure:
                self.last_joint_smc_soft_budget_exceeded = True
                print(
                    "[joint-smc] soft-budget "
                    f"elapsed_s={station_elapsed:.3f} "
                    f"budget_s={soft_budget_s:.3f} "
                    "action=skip-optional-rejuvenation-only",
                    flush=True,
                )
                break
            if sweep_index + 1 == maximum_sweeps:
                self.last_joint_rejuvenation_mixing_incomplete = bool(
                    not continuous_movement_sufficient
                    or not structural_movement_sufficient
                )
                self.last_joint_structural_mixing_incomplete = bool(
                    boundary_requires_structure and not structural_movement_sufficient
                )
                if self.last_joint_structural_mixing_incomplete:
                    print(
                        "[joint-smc] structural-mixing-incomplete "
                        f"boundary_mass={boundary_mass:.12g} "
                        "cumulative_k_transition_mass="
                        f"{cumulative_k_transition_mass:.12g} "
                        "cumulative_boundary_escape_mass="
                        f"{cumulative_boundary_escape_mass:.12g} "
                        f"sweeps={maximum_sweeps}",
                        flush=True,
                    )

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
        """Assimilate one station with common weights and aligned SMC ancestors."""
        import torch

        all_stations = tuple((*self._joint_station_history, station))
        for filt in self.filters.values():
            filt.reset_step_stats()
        self.last_joint_rejuvenation_diagnostics = []
        self.last_joint_smc_soft_budget_exceeded = False
        self.last_joint_rejuvenation_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete = False
        station_start = time.perf_counter()
        self._apply_joint_guided_initialization(station)
        self.last_joint_resample_indices = np.empty(0, dtype=np.int64)
        reference_filter = self.filters[self.joint_isotope_order()[0]]
        common_log_weights = self._assert_joint_particle_alignment()
        prefix_log_likelihood = self._joint_station_prefix_log_likelihood_torch(station)
        device = prefix_log_likelihood.device
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
        resamples = 0
        steps: list[dict[str, float]] = []
        view_count = int(station.fe_indices.size)

        def _prefix_likelihood() -> "torch.Tensor":
            """Return exact prefix targets for the current aligned states."""
            return self._joint_station_prefix_log_likelihood_torch(station).to(
                device=device,
                dtype=torch.float64,
            )

        def _prefix_increment(
            values: "torch.Tensor",
            prefix_count: int,
        ) -> "torch.Tensor":
            """Return a numerically defined exact adjacent-prefix increment."""
            previous = values[prefix_count - 1]
            current = values[prefix_count]
            previous_finite = torch.isfinite(previous)
            current_finite = torch.isfinite(current)
            if bool(torch.any(current_finite & ~previous_finite).item()):
                raise RuntimeError(
                    "Adding a view restored target mass after an impossible "
                    "prefix; the full-spectrum prefix contract is invalid."
                )
            increment = torch.full_like(current, float("-inf"))
            both_finite = previous_finite & current_finite
            increment[both_finite] = current[both_finite] - previous[both_finite]
            increment[~previous_finite & ~current_finite] = 0.0
            if bool(torch.any(torch.isnan(increment)).item()) or bool(
                torch.any(torch.isinf(increment) & (increment > 0.0)).item()
            ):
                raise RuntimeError(
                    "Adjacent full-spectrum prefix increment is invalid."
                )
            return increment

        prefix_log_likelihood = prefix_log_likelihood.to(
            device=device,
            dtype=torch.float64,
        )
        if initial_ess <= target_ess + 1.0e-9:
            indices = self._resample_joint_particles(log_weights.detach().cpu().numpy())
            station_ancestor_ids = station_ancestor_ids[indices]
            cumulative_lineage_ids = cumulative_lineage_ids[indices]
            resamples += 1
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=0.0,
                newest_prefix_count=1,
                station_start_s=station_start,
            )
            prefix_log_likelihood = _prefix_likelihood()
            log_weights = torch.full(
                (particle_count,),
                -math.log(max(particle_count, 1)),
                dtype=torch.float64,
                device=device,
            )
        max_steps = int(self.pf_config.max_temper_steps)
        for prefix_count in range(1, view_count + 1):
            prefix_step_start = len(steps)
            beta_total = 0.0
            likelihood = _prefix_increment(
                prefix_log_likelihood,
                prefix_count,
            )
            while beta_total < 1.0 - 1.0e-12:
                prefix_step_count = len(steps) - prefix_step_start
                if prefix_step_count >= max_steps:
                    raise RuntimeError(
                        "Joint view-prefix SMC reached max_temper_steps "
                        f"within prefix {prefix_count}/{view_count} "
                        "before that prefix reached beta=1."
                    )
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
                        newest_prefix_count=prefix_count,
                        station_start_s=station_start,
                    )
                    prefix_log_likelihood = _prefix_likelihood()
                    likelihood = _prefix_increment(
                        prefix_log_likelihood,
                        prefix_count,
                    )
                    recovery_step = {
                        "prefix_count": float(prefix_count),
                        "prefix_view_count": float(view_count),
                        "station_beta": float(
                            (prefix_count - 1 + beta_total) / view_count
                        ),
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
                    steps.append(recovery_step)
                    continue
                log_weights = proposed_log_weights
                self._assign_joint_log_weights(log_weights.detach().cpu().numpy())
                beta_total += float(delta_beta)
                step = {
                    "prefix_count": float(prefix_count),
                    "prefix_view_count": float(view_count),
                    "station_beta": float((prefix_count - 1 + beta_total) / view_count),
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
                    newest_prefix_count=prefix_count,
                    station_start_s=station_start,
                )
                prefix_log_likelihood = _prefix_likelihood()
                likelihood = _prefix_increment(
                    prefix_log_likelihood,
                    prefix_count,
                )
                log_weights = torch.full(
                    (particle_count,),
                    -math.log(max(particle_count, 1)),
                    dtype=torch.float64,
                    device=device,
                )
            self._joint_rejuvenate_adaptive(
                all_stations,
                target_beta=1.0,
                newest_prefix_count=prefix_count,
                station_start_s=station_start,
            )
            if prefix_count < view_count:
                prefix_log_likelihood = _prefix_likelihood()
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
        # Backward-compatible field now has the conservative cumulative
        # meaning; station-local ancestry is reported separately.
        self.last_joint_unique_ancestor_count = cumulative_unique_ancestors
        for filt in self.filters.values():
            filt.last_temper_steps = [dict(step) for step in steps]
            filt.last_temper_resample_count = int(resamples)
            filt.last_temper_min_ess = float(
                min((step["ess"] for step in steps), default=final_ess)
            )
            filt.last_station_unique_ancestor_count = station_unique_ancestors
            filt.last_cumulative_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_unique_ancestor_count = cumulative_unique_ancestors
            filt.last_ess_pre = float(initial_ess)
            filt.last_ess = float(final_ess)
            filt.last_ess_post = float(final_ess)
            filt.last_resample_ess = bool(resamples)
        self._promote_joint_birth_proposal_station(station)
        self._joint_station_history.append(station)
        self._assert_joint_particle_alignment()
        print(
            "[joint-smc] station-update-done "
            f"station={len(all_stations) - 1} "
            f"elapsed_s={time.perf_counter() - station_start:.3f} "
            f"temper_steps={len(steps)} "
            f"resamples={resamples} "
            f"final_ess={final_ess:.6f}",
            flush=True,
        )
