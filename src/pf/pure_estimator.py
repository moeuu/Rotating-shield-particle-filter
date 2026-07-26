"""Expose the sequential particle-filter estimator used by scientific runtimes."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from pf.estimator import (
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator as _PFEstimatorCore,
)
from pf.posterior import (
    PFPointEstimate,
    PFPosteriorSnapshot,
    PFSourceMode,
    cardinality_distribution_from_states,
    posterior_point_estimate_from_states,
)
from pf.profiles import (
    ProposalOrigin,
    apply_profile_to_config,
    resolve_structural_transition_provenance,
)
from pf.provenance import canonical_json_bytes, repository_commit, sha256_json


class PurePFBoundaryError(RuntimeError):
    """Signal a violation of the sequential PF result contract."""


class PurePFEstimator(_PFEstimatorCore):
    """Run causal PF updates and report only the resulting PF posterior."""

    planner_belief_sources: tuple[str, ...] = ("pf_posterior", "pf_tentative")
    allowed_proposal_origins: tuple[ProposalOrigin, ...] = (
        ProposalOrigin.PF_BIRTH,
        ProposalOrigin.PF_RESIDUAL,
        ProposalOrigin.PF_SPLIT,
    )

    def __init__(
        self,
        *args: Any,
        measurement_log_schema_version: int = 1,
        config_hash: str | None = None,
        resolved_config_hash: str | None = None,
        measurement_log_sha256: str = "unavailable",
        random_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the PF and its immutable result provenance."""
        positional_args = list(args)
        if "pf_config" in kwargs:
            pure_config = kwargs["pf_config"]
            if pure_config is None:
                pure_config = RotatingShieldPFConfig()
                kwargs["pf_config"] = pure_config
        elif len(positional_args) > 4:
            pure_config = positional_args[4]
            if pure_config is None:
                pure_config = RotatingShieldPFConfig()
                positional_args[4] = pure_config
        else:
            pure_config = RotatingShieldPFConfig()
            kwargs["pf_config"] = pure_config
        capabilities = apply_profile_to_config(pure_config)
        super().__init__(*positional_args, **kwargs)
        if apply_profile_to_config(self.pf_config) != capabilities:
            raise PurePFBoundaryError(
                "PF capabilities changed during estimator initialization."
            )
        self.profile_capabilities = capabilities
        self.measurement_log_schema_version = int(measurement_log_schema_version)
        if self.measurement_log_schema_version != 1:
            raise ValueError(
                "PurePFEstimator supports MeasurementLog schema version 1."
            )
        self.resolved_config_hash = (
            str(resolved_config_hash)
            if resolved_config_hash is not None
            else sha256_json(self.pf_config)
        )
        self.config_hash = (
            str(config_hash)
            if config_hash is not None
            else str(self.resolved_config_hash)
        )
        self.repository_commit = repository_commit()
        self.measurement_log_sha256 = str(measurement_log_sha256)
        self.random_seed = int(random_seed)

    @property
    def estimator_variant(self) -> str:
        """Return the resolved scientific PF variant."""
        return str(self.pf_config.estimator_profile)

    def structural_transition_diagnostics(self) -> dict[str, bool | str]:
        """Return target-preservation provenance for structural moves."""
        return resolve_structural_transition_provenance(
            self.pf_config,
            capabilities=self.profile_capabilities,
        ).to_dict()

    def accepts_proposal_origin(self, origin: ProposalOrigin | str) -> bool:
        """Return whether a proposal origin may alter this PF."""
        try:
            resolved = (
                origin if isinstance(origin, ProposalOrigin) else ProposalOrigin(origin)
            )
        except ValueError:
            return False
        return resolved in self.allowed_proposal_origins

    def posterior_cardinality_distribution(self) -> dict[str, dict[int, float]]:
        """Return source-count posterior mass for every active isotope."""
        result: dict[str, dict[int, float]] = {}
        for isotope, filt in self.filters.items():
            states = [particle.state for particle in filt.continuous_particles]
            result[str(isotope)] = cardinality_distribution_from_states(
                states,
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
            )
        return result

    def posterior_point_estimate(self) -> dict[str, PFPointEstimate]:
        """Return deterministic PF point estimates and uncertainty."""
        result: dict[str, PFPointEstimate] = {}
        for isotope, filt in self.filters.items():
            states = [particle.state for particle in filt.continuous_particles]
            result[str(isotope)] = posterior_point_estimate_from_states(
                states,
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
                position_projector=filt._project_positions_to_source_prior,
            )
        return result

    def posterior_modes(self) -> dict[str, tuple[PFSourceMode, ...]]:
        """Return aligned PF posterior modes for every active isotope."""
        return {
            isotope: estimate.modes
            for isotope, estimate in self.posterior_point_estimate().items()
        }

    def posterior_snapshot(self) -> PFPosteriorSnapshot:
        """Return a schema-v1 PF posterior result with reproducibility metadata."""
        log_digest = str(self.measurement_log_sha256).strip().lower()
        if len(log_digest) != 64 or any(
            character not in "0123456789abcdef" for character in log_digest
        ):
            raise PurePFBoundaryError(
                "A publishable PF posterior requires a finalized "
                "MeasurementLog SHA-256 digest."
            )
        return PFPosteriorSnapshot(
            estimator_variant=self.estimator_variant,
            isotopes=self.posterior_point_estimate(),
            planner_belief_sources=self.planner_belief_sources,
            repository_commit=self.repository_commit,
            measurement_log_schema_version=self.measurement_log_schema_version,
            config_hash=self.config_hash,
            resolved_config_hash=self.resolved_config_hash,
            measurement_log_sha256=self.measurement_log_sha256,
            random_seed=self.random_seed,
            profile_capability_map=self.profile_capabilities.to_dict(),
            record_count=len(self.measurements),
            structural_transition_provenance=(
                self.structural_transition_diagnostics()
            ),
        )

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project the PF posterior into the historical array result format."""
        result: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, point_estimate in self.posterior_point_estimate().items():
            if not point_estimate.modes:
                result[isotope] = (
                    np.zeros((0, 3), dtype=float),
                    np.zeros(0, dtype=float),
                )
                continue
            result[isotope] = (
                np.asarray(
                    [mode.position_mean_xyz for mode in point_estimate.modes],
                    dtype=float,
                ),
                np.asarray(
                    [mode.strength_mean_cps_1m for mode in point_estimate.modes],
                    dtype=float,
                ),
            )
        return result

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the PF posterior projection for visualization."""
        return self.estimates()

    def serialized_state(self) -> bytes:
        """Return canonical bytes for causality and determinism tests."""
        isotope_payload: dict[str, Any] = {}
        for isotope, filt in sorted(self.filters.items()):
            particles: list[dict[str, Any]] = []
            for particle in filt.continuous_particles:
                state = particle.state
                particles.append(
                    {
                        "log_weight": float(particle.log_weight),
                        "num_sources": int(state.num_sources),
                        "positions": np.asarray(state.positions, dtype=float),
                        "strengths": np.asarray(state.strengths, dtype=float),
                        "background": float(state.background),
                        "ages": state.ages,
                        "support_scores": state.support_scores,
                        "tentative_sources": state.tentative_sources,
                        "verification_fail_streaks": (
                            state.verification_fail_streaks
                        ),
                    }
                )
            isotope_payload[str(isotope)] = particles
        measurement_history = [
            {
                "z_k": measurement.z_k,
                "pose_idx": int(measurement.pose_idx),
                "fe_index": measurement.fe_index,
                "pb_index": measurement.pb_index,
                "live_time_s": float(measurement.live_time_s),
                "z_variance_k": measurement.z_variance_k,
                "z_covariance_k": measurement.z_covariance_k,
                "station_sequence_id": measurement.station_sequence_id,
                "station_view_index": measurement.station_view_index,
                "runtime_likelihood_route_by_isotope": (
                    measurement.runtime_likelihood_route_by_isotope
                ),
                "runtime_spectrum_variance_used_by_isotope": (
                    measurement.runtime_spectrum_variance_used_by_isotope
                ),
                "station_view_covariance_by_isotope": (
                    measurement.station_view_covariance_by_isotope
                ),
            }
            for measurement in self.measurements
        ]
        return canonical_json_bytes(
            {
                "schema_version": 1,
                "estimator_variant": self.estimator_variant,
                "measurement_count": len(self.measurements),
                "measurement_poses_xyz": [
                    np.asarray(pose, dtype=float) for pose in self.poses
                ],
                "measurement_pose_indices": [
                    int(measurement.pose_idx) for measurement in self.measurements
                ],
                "measurement_history_sha256": sha256_json(measurement_history),
                "deferred_pose_update_active": bool(self._defer_resample_birth),
                "deferred_measurement_count": int(self._deferred_measurement_count),
                "isotopes": isotope_payload,
            }
        )


RotatingShieldPurePFEstimator = PurePFEstimator
RotatingShieldPFEstimator = PurePFEstimator


__all__ = [
    "PurePFBoundaryError",
    "PurePFEstimator",
    "RotatingShieldPFConfig",
    "RotatingShieldPFEstimator",
    "RotatingShieldPurePFEstimator",
]
