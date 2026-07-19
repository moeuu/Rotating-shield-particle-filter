"""Expose the clean sequential PF estimator used by scientific runtimes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, NoReturn, Tuple

import numpy as np
from numpy.typing import NDArray

from pf.estimator import (
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator as _LegacyEstimatorShell,
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
    """Signal an attempt to call a batch estimator through the pure PF API."""


class PurePFEstimator(_LegacyEstimatorShell):
    """Run only causal PF updates while preserving the established PF kernels.

    The inherited shell supplies the high-fidelity observation kernel, batched
    likelihood, resampling, structural moves, and continuous 3-D planner APIs.
    Every all-history/batch hook is overridden here so it cannot participate in
    state updates, planning, mission control, or reporting.
    """

    planner_belief_sources: tuple[str, ...] = ("pf_posterior", "pf_tentative")
    allowed_proposal_origins: tuple[ProposalOrigin, ...] = (
        ProposalOrigin.PF_BIRTH,
        ProposalOrigin.PF_RESIDUAL,
        ProposalOrigin.PF_SPLIT,
    )
    forbidden_batch_entry_points: tuple[str, ...] = (
        "_solve_report_strengths",
        "_solve_report_strengths_batch",
        "_augment_report_candidates_with_mle_rescue",
        "_select_report_clusters_by_model_order",
        "_refine_report_surface_positions",
        "_refit_reported_strengths",
        "_all_history_dictionary_candidates",
        "_runtime_report_rescue_estimate",
        "fit_surface_map",
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
        """Initialize the clean PF and its immutable purity provenance."""
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
                "Pure PF capabilities changed during legacy-shell initialization."
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
        self.batch_methods_invoked: list[str] = []

    @property
    def estimator_variant(self) -> str:
        """Return the resolved scientific PF variant."""
        return str(self.pf_config.estimator_profile)

    def structural_transition_diagnostics(self) -> dict[str, bool | str]:
        """Return truthful target-preservation provenance for structural moves."""
        return resolve_structural_transition_provenance(
            self.pf_config,
            capabilities=self.profile_capabilities,
        ).to_dict()

    def accepts_proposal_origin(self, origin: ProposalOrigin | str) -> bool:
        """Return whether a proposal origin is allowed to alter this PF."""
        try:
            resolved = (
                origin if isinstance(origin, ProposalOrigin) else ProposalOrigin(origin)
            )
        except ValueError:
            return False
        return resolved in self.allowed_proposal_origins

    def _reject_batch_estimation(self, method_name: str) -> NoReturn:
        """Record and reject an inherited all-history batch-estimation call."""
        self.batch_methods_invoked.append(str(method_name))
        raise PurePFBoundaryError(
            f"{method_name} is outside the PurePFEstimator boundary."
        )

    def _solve_report_strengths(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject the inherited all-history report-strength optimizer."""
        del args, kwargs
        self._reject_batch_estimation("_solve_report_strengths")

    def _solve_report_strengths_batch(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject the inherited batched all-history strength optimizer."""
        del args, kwargs
        self._reject_batch_estimation("_solve_report_strengths_batch")

    def _augment_report_candidates_with_mle_rescue(
        self, *args: Any, **kwargs: Any
    ) -> NoReturn:
        """Reject inherited report-MLE position candidate augmentation."""
        del args, kwargs
        self._reject_batch_estimation("_augment_report_candidates_with_mle_rescue")

    def _select_report_clusters_by_model_order(
        self, *args: Any, **kwargs: Any
    ) -> NoReturn:
        """Reject inherited batch model-order selection for final reports."""
        del args, kwargs
        self._reject_batch_estimation("_select_report_clusters_by_model_order")

    def _refine_report_surface_positions(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject inherited all-history surface-position refinement."""
        del args, kwargs
        self._reject_batch_estimation("_refine_report_surface_positions")

    def _refit_reported_strengths(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject inherited final-report position and strength refitting."""
        del args, kwargs
        self._reject_batch_estimation("_refit_reported_strengths")

    def _all_history_dictionary_candidates(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject inherited all-history dictionary proposals."""
        del args, kwargs
        self._reject_batch_estimation("_all_history_dictionary_candidates")

    def _runtime_report_rescue_estimate(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject inherited runtime report-MLE rescue estimation."""
        del args, kwargs
        self._reject_batch_estimation("_runtime_report_rescue_estimate")

    def refresh_sparse_poisson_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Return no evidence because all-history sparse fits are outside pure PF."""
        self._last_sparse_poisson_evidence_diagnostics = {}
        self._last_joint_sparse_poisson_evidence_diagnostics = {}
        self.last_sparse_poisson_refresh_wall_s = 0.0
        self.last_sparse_poisson_refresh_stage_wall_s = {}
        return {}

    def sparse_poisson_evidence_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return an empty mapping without invoking a sparse estimator."""
        return {}

    def _complete_spectrum_payload_with_configured_responses(
        self,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Reject direct spectrum-bin likelihoods at the pure count-PF boundary."""
        if payload is not None:
            raise PurePFBoundaryError(
                "Pure PF profiles accept response_poisson isotope counts only; "
                "raw spectrum bins are reserved for standalone MLE/ablations."
            )
        return None

    def report_model_order_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return no batch model-order diagnostics in a pure PF."""
        return {}

    def report_model_order_ready(self) -> bool:
        """Return false because readiness is not defined by a batch model order."""
        return False

    def runtime_report_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return no report/MLE rescue modes to the planner."""
        return {}

    def planning_surface_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return no surface-map modes to the planner."""
        self._last_planning_surface_rescue_mode_counts = {}
        return {}

    def _sync_sparse_evidence_cardinality_protection(
        self,
        isotope: str,
        filt: Any,
    ) -> tuple[bool, int]:
        """Disable sparse-evidence cardinality protection for pure PF updates."""
        del isotope, filt
        return False, 0

    def _runtime_global_birth_rescue_candidates(
        self,
        isotope: str,
        filt: Any,
        data: Any,
    ) -> NDArray[np.float64]:
        """Return no all-history/global rescue candidates."""
        del isotope, filt, data
        return np.zeros((0, 3), dtype=float)

    def _inject_runtime_report_rescue(self, isotope: str, filt: Any) -> None:
        """Reject report-derived particle injection by construction."""
        del isotope, filt

    def _run_isotope_structural_update(
        self,
        task: tuple[str, Any, Any, Any, Any],
    ) -> None:
        """Run causal PF refit/moves without batch candidates or feedback."""
        _isotope, filt, refit_data, support_data, birth_data = task
        if self.profile_capabilities.conditional_strength_profile and bool(
            self.pf_config.conditional_strength_refit
        ):
            filt.refit_strengths_for_particles(
                refit_data,
                iters=self.pf_config.conditional_strength_refit_iters,
                eps=self.pf_config.refit_eps,
                suppress_prune_after_refit=bool(
                    self.pf_config.birth_residual_suppress_death
                ),
            )
        filt.apply_birth_death(
            support_data=support_data,
            birth_data=birth_data,
            candidate_positions=self.candidate_sources,
            global_birth_candidates=np.zeros((0, 3), dtype=float),
            global_birth_candidate_counts=None,
            allow_structural_birth_proposals=True,
        )

    def record_report_snapshot(
        self,
        *,
        label: str,
        allow_heavy_estimate: bool = True,
    ) -> None:
        """Avoid the historical mixed best-report cache in pure PF variants."""
        del label, allow_heavy_estimate

    def final_report_estimate(
        self,
        *,
        total_measurements: int | None = None,
        use_best_so_far: bool = False,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the current PF posterior projection as the final report."""
        del total_measurements, use_best_so_far
        return self.estimates()

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
        """Return deterministic PF-only point estimates and uncertainty."""
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
        """Return a schema-v1 PF posterior result with purity provenance."""
        log_digest = str(self.measurement_log_sha256).strip().lower()
        if len(log_digest) != 64 or any(
            character not in "0123456789abcdef" for character in log_digest
        ):
            raise PurePFBoundaryError(
                "A publishable pure-PF posterior requires a finalized "
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
            structural_transition_provenance=(self.structural_transition_diagnostics()),
        )

    def estimates(
        self,
        *,
        use_pre_finalize_guard: bool = True,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project the PF posterior report into the historical array API."""
        del use_pre_finalize_guard
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
        """Return the PF posterior projection for visualization compatibility."""
        return self.estimates()

    def pruned_estimates(
        self,
        method: str = "none",
        params: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Apply display-only strength thresholding without likelihood refits."""
        del method, kwargs
        threshold = max(0.0, float((params or {}).get("min_strength_abs", 0.0)))
        result: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, (positions, strengths) in self.estimates().items():
            keep = np.asarray(strengths, dtype=float) >= threshold
            result[isotope] = (
                np.asarray(positions, dtype=float)[keep].copy(),
                np.asarray(strengths, dtype=float)[keep].copy(),
            )
        return result

    def fit_surface_map(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Reject surface-map reconstruction at the pure PF boundary."""
        del args, kwargs
        self._reject_batch_estimation("fit_surface_map")

    def serialized_state(self) -> bytes:
        """Return a canonical byte representation for causality/determinism tests."""
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
                        "low_q_streaks": state.low_q_streaks,
                        "support_scores": state.support_scores,
                        "tentative_sources": state.tentative_sources,
                        "verification_fail_streaks": state.verification_fail_streaks,
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


# The scientific/default class name is kept explicit for downstream adapters.
RotatingShieldPurePFEstimator = PurePFEstimator
RotatingShieldPFEstimator = PurePFEstimator


__all__ = [
    "PurePFBoundaryError",
    "PurePFEstimator",
    "RotatingShieldPFConfig",
    "RotatingShieldPFEstimator",
    "RotatingShieldPurePFEstimator",
]
