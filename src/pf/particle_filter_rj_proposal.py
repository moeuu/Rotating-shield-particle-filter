"""Sweep-fixed continuous position and strength proposals for exact-RJ PF."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from pf.particle_types import StructuralGeometryBatch
from pf.structural_rj import (
    ContinuousBlockStrengthProposal,
    ContinuousStrengthProposal,
    ContinuousSurfacePositionProposal,
)


class StructuralRJProposalMixin:
    """Build the batched proposal distributions frozen for an RJ sweep."""

    def _build_continuous_rj_position_proposal(
        self,
        data: StructuralGeometryBatch,
        *,
        target_beta: float,
    ) -> ContinuousSurfacePositionProposal:
        """Build the sweep-fixed full-spectrum residual proposal.

        Chart centers are evaluated only to define a proposal density.  The
        accepted state remains continuous in chart ``(u, v)`` and every MH/RJ
        target evaluation uses its exact XYZ.  A positive area-prior mixture
        gives global support, while the estimator supplies batched
        full-spectrum residual evidence and a chart-conditional strength
        location.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        evaluator = self._joint_proposal_evaluator
        if evaluator is None:
            raise RuntimeError(
                "Continuous exact-RJ requires the estimator-owned "
                "full-spectrum residual proposal evaluator."
            )
        prior_probabilities = np.asarray(
            atlas.chart_probabilities,
            dtype=np.float64,
        )
        alignment_scores, strength_locations, informative = evaluator(
            filt=self,
            data=data,
            chart_centers_xyz=np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            ),
            target_beta=target_beta,
        )
        alignment = np.asarray(
            alignment_scores,
            dtype=np.float64,
        ).reshape(-1)
        locations = np.asarray(
            strength_locations,
            dtype=np.float64,
        ).reshape(-1)
        if (
            alignment.shape != (atlas.chart_count,)
            or locations.shape != (atlas.chart_count,)
            or np.any(~np.isfinite(alignment))
            or np.any(alignment < 0.0)
            or np.any(~np.isfinite(locations))
            or np.any(
                ~np.asarray(self._strength_prior.in_support(locations), dtype=bool)
            )
        ):
            raise ValueError("Full-spectrum residual proposal arrays are invalid.")
        proposal = ContinuousSurfacePositionProposal(
            area_prior_probabilities=prior_probabilities,
            alignment_scores=(
                alignment if bool(informative) else np.zeros_like(alignment)
            ),
            prior_component_probability=float(
                self.config.structural_rj_position_proposal_prior_weight
            ),
        )
        strength_proposal = ContinuousStrengthProposal(
            minimum=float(self._strength_prior.minimum),
            maximum=float(self._strength_prior.maximum),
            data_locations_by_chart=locations,
            data_sigma=float(self.config.structural_rj_strength_proposal_sigma_fraction)
            * (
                float(self._strength_prior.finite_upper_quantile())
                - float(self._strength_prior.minimum)
                if self._strength_prior.family == "shifted_gamma"
                else float(self._strength_prior.maximum)
                - float(self._strength_prior.minimum)
            ),
            prior_component_probability=float(
                self.config.structural_rj_strength_proposal_prior_weight
            ),
            data_informative=bool(informative),
            prior_family=str(self._strength_prior.family),
            prior_gamma_shape=float(self._strength_prior.gamma_shape),
            prior_gamma_scale=float(self._strength_prior.gamma_scale),
        )
        self._last_structural_rj_position_proposal = proposal
        self._structural_rj_strength_proposal = strength_proposal
        self._last_structural_rj_strength_proposal = strength_proposal
        self.last_structural_rj_proposal_snapshot_sha256 = (
            self._continuous_rj_proposal_snapshot_sha256(
                proposal,
                strength_proposal,
            )
        )
        return proposal

    @staticmethod
    def _continuous_rj_proposal_snapshot_sha256(
        position_proposal: ContinuousSurfacePositionProposal,
        strength_proposal: ContinuousStrengthProposal,
    ) -> str:
        """Hash every frozen parameter used by birth/death proposal densities."""
        digest = hashlib.sha256(b"continuous_surface_birth_proposal_snapshot_v2\0")
        digest.update(str(strength_proposal.prior_family).encode("utf-8"))
        digest.update(b"\0")
        arrays = (
            position_proposal.area_prior_probabilities,
            position_proposal.alignment_scores,
            position_proposal.chart_probabilities,
            strength_proposal.data_locations_by_chart,
            np.asarray(
                [
                    position_proposal.prior_component_probability,
                    strength_proposal.minimum,
                    strength_proposal.maximum,
                    strength_proposal.data_sigma,
                    strength_proposal.prior_component_probability,
                    float(strength_proposal.data_informative),
                    strength_proposal.prior_gamma_shape,
                    strength_proposal.prior_gamma_scale,
                ],
                dtype="<f8",
            ),
        )
        for value in arrays:
            array = np.ascontiguousarray(value, dtype="<f8")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _active_continuous_rj_position_proposal(
        self,
    ) -> ContinuousSurfacePositionProposal:
        """Return the proposal frozen at the start of this structural sweep."""
        proposal = self._structural_rj_position_proposal
        if proposal is None:
            raise RuntimeError(
                "Continuous RJ position proposal was not frozen for this sweep."
            )
        return proposal

    def _active_continuous_rj_strength_proposal(
        self,
    ) -> ContinuousStrengthProposal:
        """Return the strength proposal frozen with the current sweep."""
        proposal = self._structural_rj_strength_proposal
        if proposal is None:
            raise RuntimeError(
                "Continuous RJ strength proposal was not frozen for this sweep."
            )
        return proposal

    def _continuous_rj_conditional_block_strength_proposal(
        self,
        data: StructuralGeometryBatch,
        *,
        chart_ids: NDArray[np.int64],
        positions: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
        cache_current_state: bool = False,
    ) -> ContinuousBlockStrengthProposal:
        """Build an all-history conditional full-support strength proposal.

        The chart-wise residual proposal supplies relative source weights.  A
        physically defined grid over the proper prior's total-strength scale
        is then evaluated under the exact current history target in one batch.
        The best total scale becomes the deterministic center of a block
        mixture.  This changes proposal efficiency only: both the prior block
        and conditional block densities are retained explicitly in the MH
        ratio.
        """
        charts = np.asarray(chart_ids, dtype=np.int64)
        coordinates = np.asarray(positions, dtype=np.float64)
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if (
            charts.ndim != 2
            or coordinates.shape != charts.shape + (3,)
            or charts.shape[0] != indices.size
            or charts.shape[1] < 1
        ):
            raise ValueError(
                "Conditional block strengths require aligned nonempty states."
            )
        row_count, cardinality = charts.shape
        scalar_proposal = self._active_continuous_rj_strength_proposal()
        base = np.asarray(
            scalar_proposal.data_locations_by_chart[charts],
            dtype=np.float64,
        )
        minimum = float(self._strength_prior.minimum)
        excess = np.maximum(
            base - minimum,
            np.finfo(np.float64).eps * max(1.0, float(self._strength_prior.mean)),
        )
        relative = excess / np.sum(excess, axis=1, keepdims=True)
        grid_size = int(self.config.structural_rj_strength_proposal_grid_size)
        probabilities = np.linspace(
            0.005,
            0.995,
            grid_size,
            dtype=np.float64,
        )
        if self._strength_prior.family == "shifted_gamma":
            from scipy.special import gammaincinv

            total_excess = float(self._strength_prior.gamma_scale) * gammaincinv(
                float(cardinality) * float(self._strength_prior.gamma_shape),
                probabilities,
            )
        else:
            total_excess = (
                probabilities
                * float(cardinality)
                * (float(self._strength_prior.maximum) - minimum)
            )
        candidate_strengths = (
            minimum + relative[:, None, :] * total_excess[None, :, None]
        )
        if self._strength_prior.family == "bounded_uniform":
            candidate_strengths = np.minimum(
                candidate_strengths,
                float(self._strength_prior.maximum),
            )
        centers = np.empty((row_count, cardinality), dtype=np.float64)
        cached_rows = np.zeros(row_count, dtype=np.bool_)
        center_cache = self._structural_rj_current_block_strength_centers
        cardinality_cache = self._structural_rj_current_block_strength_cardinalities
        if cache_current_state and center_cache is not None:
            if (
                cardinality_cache is None
                or center_cache.shape
                != (len(self.continuous_particles), self.config.hard_max_sources)
                or cardinality_cache.shape != (len(self.continuous_particles),)
                or np.any(indices < 0)
                or np.any(indices >= center_cache.shape[0])
            ):
                raise RuntimeError(
                    "Current block-strength proposal cache is misaligned."
                )
            cached_rows = cardinality_cache[indices] == cardinality
            centers[cached_rows] = center_cache[
                indices[cached_rows],
                :cardinality,
            ]
        evaluation_rows = np.flatnonzero(~cached_rows)
        if evaluation_rows.size:
            evaluation_positions = coordinates[evaluation_rows]
            evaluation_charts = charts[evaluation_rows]
            evaluation_strengths = candidate_strengths[evaluation_rows]
            evaluation_indices = indices[evaluation_rows]
            if self._joint_strength_grid_target_evaluator is not None:
                conditional_target = np.asarray(
                    self._joint_strength_grid_target_evaluator(
                        filt=self,
                        data=data,
                        positions_pks=evaluation_positions,
                        chart_ids_pk=evaluation_charts,
                        strengths_pgk=evaluation_strengths,
                        particle_indices=evaluation_indices,
                        target_beta=float(target_beta),
                        tempering_start_row=(self._structural_rj_tempering_start_row),
                    ),
                    dtype=np.float64,
                )
                if conditional_target.shape != (
                    evaluation_rows.size,
                    grid_size,
                ):
                    raise ValueError(
                        "Joint strength-grid target evaluator returned an "
                        "invalid shape."
                    )
            else:
                candidate_count = evaluation_rows.size * grid_size
                expanded_positions = np.broadcast_to(
                    evaluation_positions[:, None, :, :],
                    (
                        evaluation_rows.size,
                        grid_size,
                        cardinality,
                        3,
                    ),
                ).reshape(candidate_count, cardinality, 3)
                expanded_charts = np.broadcast_to(
                    evaluation_charts[:, None, :],
                    (evaluation_rows.size, grid_size, cardinality),
                ).reshape(candidate_count, cardinality)
                expanded_indices = np.repeat(evaluation_indices, grid_size)
                flat_strengths = evaluation_strengths.reshape(
                    candidate_count,
                    cardinality,
                )
                conditional_target = self._continuous_rj_group_log_likelihood(
                    data,
                    expanded_positions,
                    flat_strengths,
                    chart_ids=expanded_charts,
                    particle_indices=expanded_indices,
                    target_beta=target_beta,
                ).reshape(evaluation_rows.size, grid_size)
            conditional_target += np.sum(
                np.asarray(
                    self._strength_prior.log_prob(evaluation_strengths),
                    dtype=np.float64,
                ),
                axis=2,
            )
            valid_rows = np.any(np.isfinite(conditional_target), axis=1)
            best_columns = np.argmax(
                np.where(
                    np.isfinite(conditional_target),
                    conditional_target,
                    float("-inf"),
                ),
                axis=1,
            )
            evaluated_centers = evaluation_strengths[
                np.arange(evaluation_rows.size, dtype=np.int64),
                best_columns,
            ].copy()
            evaluated_centers[~valid_rows] = base[evaluation_rows][~valid_rows]
            centers[evaluation_rows] = evaluated_centers
            if cache_current_state and center_cache is not None:
                center_cache[
                    evaluation_indices,
                    :cardinality,
                ] = evaluated_centers
                cardinality_cache[evaluation_indices] = cardinality
        return ContinuousBlockStrengthProposal(
            minimum=minimum,
            maximum=float(self._strength_prior.maximum),
            data_locations=centers,
            data_sigma=float(scalar_proposal.data_sigma),
            prior_component_probability=float(
                scalar_proposal.prior_component_probability
            ),
            prior_family=str(self._strength_prior.family),
            prior_gamma_shape=float(self._strength_prior.gamma_shape),
            prior_gamma_scale=float(self._strength_prior.gamma_scale),
        )


__all__ = ["StructuralRJProposalMixin"]
