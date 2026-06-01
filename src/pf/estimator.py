"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import itertools
import re
import time
from typing import Dict, List, Sequence, Tuple, Any
import copy
import os

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.model import EnvironmentConfig
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kinds
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.likelihood import expected_counts_per_source
from pf.particle_filter import IsotopeParticleFilter, MeasurementData, PFConfig
from pf.resampling import systematic_resample
from pf.state import IsotopeState


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a weighted quantile for non-negative planning statistics."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        raise ValueError("weights must have the same size as values.")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return 0.0
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.quantile(values, np.clip(float(quantile), 0.0, 1.0)))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, np.clip(float(quantile), 0.0, 1.0), side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def _measurement_vector(
    values: float | NDArray[np.float64],
    count: int,
    name: str,
    *,
    min_value: float | None = None,
    allow_scalar: bool = True,
) -> NDArray[np.float64]:
    """Return a validated one-value-per-measurement vector."""
    expected = max(int(count), 0)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        if expected == 0:
            return np.zeros(0, dtype=float)
        raise ValueError(f"{name} must contain one value per measurement.")
    if arr.size == 1 and expected != 1 and allow_scalar:
        arr = np.full(expected, float(arr[0]), dtype=float)
    elif arr.size != expected:
        scalar_text = "scalar or " if allow_scalar else ""
        raise ValueError(f"{name} must be {scalar_text}one value per measurement.")
    if min_value is not None:
        arr = np.maximum(arr, float(min_value))
    return np.asarray(arr, dtype=float)


@dataclass
class RotatingShieldPFConfig:
    """
    Configuration parameters for the rotating-shield PF (Sec. 3.4–3.5).

    Users can tune convergence thresholds and planning settings:
        - max_sources: cap on sources per isotope
        - ig_threshold: max IG below which rotation stops (Eq. 3.49)
        - max_dwell_time_s: per-pose dwell cap
        - credible_volume_threshold: max ellipsoid volume for positional credible regions
        - lambda_cost: motion-cost weight in Eq. 3.51
        - position_sigma: Gaussian jitter for positions (meters)
        - alpha_weights: isotope weights for IG criteria
        - death_low_q_streak: steps below min_strength before death is allowed
        - death_delta_ll_threshold: ΔLL threshold required to kill weak sources
        - support_ema_alpha: EMA weight for per-source ΔLL support
        - support_window: measurement window for per-source support scoring
        - birth_window: measurement window for residual-driven birth proposals
        - birth_softmax_temp: temperature for residual proposal sampling
        - birth_min_score: score floor for residual proposal sampling
        - birth_enable: enable birth/death/split/merge moves
        - birth_topk_particles: number of top-weight particles for residual mix
        - birth_use_weighted_topk: weight residual mix by particle weights
        - birth_min_sep_m: minimum separation between sources during birth
        - birth_detector_min_sep_m: minimum separation from measured detector poses
        - source_detector_exclusion_m: hard exclusion around measured detector poses
        - birth_candidate_jitter_sigma: position jitter (m) for birth candidates
        - birth_num_local_jitter: local jitter samples per candidate
        - birth_alpha: damping factor for new source strength
        - birth_q_max: clamp max for new source strength
        - birth_q_min: clamp min for new source strength
        - birth_max_per_update: cap accepted birth proposals per structural update
        - birth_delta_ll_threshold: likelihood-gain floor for accepting birth
        - birth_complexity_penalty: extra model-complexity penalty for birth
        - birth_bic_penalty_params: source parameter count for BIC birth penalty
        - birth_residual_clip_quantile: clip residuals at this quantile
        - birth_residual_gate_p_value: chi-square p-value for residual birth evidence
        - birth_residual_min_support: minimum independent residual-supported measurements
        - birth_residual_support_sigma: per-measurement residual z-score support floor
        - birth_min_distinct_stations: minimum robot stations with residual birth evidence
        - birth_candidate_support_fraction: per-candidate residual overlap floor
        - birth_refit_residual_gate: require residuals to survive fixed-position strength refit
        - birth_refit_residual_min_fraction: residual fraction retained after refit for birth
        - birth_use_shield_coded_residual: rank birth candidates by full shield-coded residual response
        - birth_existing_response_corr_max: reject birth candidates collinear with existing responses
        - birth_count_distance_prior_weight: soft proposal weight for high unit-response candidates
        - birth_count_distance_strength_weight: soft penalty for candidates needing high fitted strength
        - birth_count_distance_log_clip: robust log-ratio clip for count-distance proposal terms
        - birth_count_distance_strength_sigma: log-strength scale for the high-strength penalty
        - birth_residual_always_try: try residual birth when the statistical gate passes
        - birth_residual_expand_structural_particles: expand residual birth proposals beyond normal top-k
        - birth_residual_expanded_structural_topk_particles: cap residual-gated structural proposals
        - birth_residual_acceptance_complexity_scale: scale complexity penalty after residual gate
        - birth_residual_suppress_death: delay death while positive residual birth evidence remains
        - birth_matching_pursuit_max_new_sources: max sequential residual births per particle
        - birth_matching_pursuit_topk_candidates: candidates evaluated per matching-pursuit birth
        - birth_jitter_topk_candidates: base residual-supported candidates jittered for birth
        - birth_global_rescue_enable: add station-level MLE-style surface candidates to runtime birth
        - birth_global_rescue_max_candidates: max global surface candidates used by runtime birth
        - birth_global_rescue_min_residual_fraction: residual fraction needed before runtime global rescue
        - birth_global_rescue_dedup_radius_m: distance used to merge runtime global birth candidates
        - birth_global_rescue_forced_min_delta_ll: minimum ΔLL for forced runtime global rescue births
        - birth_global_rescue_min_support: support count override for global rescue births
        - birth_global_rescue_min_distinct_poses: distinct-view override for global rescue births
        - birth_global_rescue_min_distinct_stations: station-count override for global rescue births
        - runtime_report_rescue_enable: inject report-level rescue modes after each station
        - runtime_report_rescue_particle_fraction: PF particle fraction reserved for runtime rescue
        - runtime_report_rescue_min_particles_per_source: minimum injected particles per rescued source
        - runtime_report_rescue_weight: posterior mass assigned to injected rescue particles
        - runtime_report_rescue_jitter_sigma_m: surface-projected rescue-particle position jitter
        - runtime_report_rescue_quarantine_enable: inject unstable rescue modes with reduced mass
        - runtime_report_rescue_quarantine_weight: posterior mass for quarantined rescue modes
        - runtime_report_rescue_candidate_weight: posterior mass for BIC-selected report modes that are not yet fully verified
        - runtime_report_rescue_memory_enable: retain recent report-rescue modes across stations
        - runtime_report_rescue_memory_decay: station-to-station rescue-memory score decay
        - runtime_report_rescue_memory_max_sources: max remembered rescue modes per isotope
        - report_mle_rescue_surface_quota_enable: reserve rescue slots across known surface strata
        - report_mle_rescue_surface_quota_min_score_fraction: score floor for stratum quota candidates
        - report_mle_rescue_surface_quota_per_stratum: reserved rescue slots per surface stratum
        - residual_decomposition_enable: enable raw/peak-suppressed residual layers
        - peak_suppression_enable: use strong-source leave-one-out residual layers
        - peak_suppression_min_source_fraction: source fraction defining suppressible peaks
        - peak_suppression_factor: fraction of a strong source contribution added back
        - residual_decomposition_max_layers: max residual layers used for birth proposals
        - pseudo_source_verification_enable: verify tentative birth sources before pruning
        - pseudo_source_min_delta_ll: leave-one-out ΔLL floor for confirming sources
        - pseudo_source_min_distinct_views: distinct shield views needed to confirm sources
        - pseudo_source_fail_grace_stations: failed verifications before pruning tentative sources
        - pseudo_source_corr_max: response-correlation ceiling against stronger sources
        - pseudo_source_temporal_sep_min: whitened temporal-code separation that can confirm high-correlation sources
        - report_exclude_unverified_sources: hide tentative birth sources from report estimates
        - source_prune_refit_after_remove: refit remaining strengths before pruning
        - source_prune_bic_penalty_params: source parameter count for BIC prune gain
        - refit_after_moves: refit strengths after birth/kill/split/merge
        - refit_iters: iterations for strength refit
        - refit_eps: epsilon for refit stability
        - weak_source_prune_min_expected_count: prune floor-strength sources below this support
        - weak_source_prune_min_fraction: prune floor-strength sources below this source fraction
        - weak_source_prune_min_age: source age before weak-source pruning is allowed
        - weak_source_prune_require_observable: only prune after enough observable measurements
        - weak_source_prune_min_observable_measurements: visible-view count needed before weak pruning
        - weak_source_prune_observable_count: per-measurement visible-count threshold
        - weak_source_prune_observable_fraction: per-measurement visible-fraction threshold
        - weak_source_prune_visibility_reference_strength: optional fixed cps@1m for visibility checks
        - conditional_strength_refit: refit strengths at station finalization
        - conditional_strength_refit_window: recent measurements used for strength refit
        - conditional_strength_refit_iters: iterations for conditional strength refit
        - conditional_strength_refit_reweight: reweight particles by profile-likelihood gain
        - conditional_strength_refit_cardinality_neutral_reweight: keep model-order mass during refit reweighting
        - conditional_strength_refit_reweight_clip: robust clip for profile-likelihood correction
        - conditional_strength_refit_min_count: minimum positive count for strength refit
        - conditional_strength_refit_min_snr: minimum count SNR for strength refit
        - conditional_strength_refit_prior_weight: MAP strength-prior weight
        - conditional_strength_refit_prior_rel_sigma: relative strength-prior sigma
        - source_strength_prior_mean: absolute source-rate prior mean shared by runtime/report refits
        - source_strength_prior_weight: absolute source-rate prior weight to suppress one-source absorption
        - source_strength_prior_rel_sigma: relative sigma for the absolute source-rate prior
        - source_strength_absorption_penalty_weight: legacy prior-based runtime high-strength penalty
        - source_strength_absorption_q_multiple: legacy prior-based runtime high-strength scale
        - source_strength_observation_overshoot_penalty_weight: data-driven runtime overshoot penalty
        - source_strength_observation_overshoot_sigma: count-sigma margin for runtime overshoot tests
        - source_strength_observation_overshoot_quantile: visible-view quantile for runtime overshoot bounds
        - source_strength_observation_overshoot_min_visible_fraction: response visibility threshold fraction
        - source_strength_observation_overshoot_min_visible_measurements: visible views needed for runtime overshoot
        - report_strength_refit: refit reported strengths conditioned on reported positions
        - report_strength_refit_iters: multiplicative Poisson regression iterations
        - report_strength_refit_eps: numerical floor for reported-strength regression
        - report_strength_refit_use_all_measurements: include shielded and low-count postures in report refit
        - report_strength_refit_preserve_cardinality: keep posterior clusters during report refit
        - report_strength_refit_prior_weight: MAP strength-prior weight for report refit
        - report_strength_refit_prior_rel_sigma: relative strength-prior sigma for report refit
        - report_strength_absorption_penalty_weight: legacy prior-based report high-strength penalty
        - report_strength_absorption_q_multiple: legacy prior-based report high-strength scale
        - report_strength_observation_overshoot_penalty_weight: data-driven report overshoot penalty
        - report_strength_observation_overshoot_sigma: count-sigma margin for report overshoot tests
        - report_strength_observation_overshoot_quantile: visible-view quantile for report overshoot bounds
        - report_strength_observation_overshoot_min_visible_fraction: response visibility threshold fraction
        - report_strength_observation_overshoot_min_visible_measurements: visible views needed for report overshoot
        - report_surface_local_refine: bounded local surface refinement before report BIC
        - report_surface_local_refine_radius_m: local refinement search radius
        - report_surface_local_refine_grid_steps: half-grid steps for local refinement
        - report_surface_local_refine_max_candidates_per_source: stencil cap per source
        - report_surface_local_refine_max_sources: source-slot cap for local refinement
        - report_surface_local_refine_min_ll_gain: minimum accepted local-refine LL gain
        - report_mle_rescue_enable: add MLE-style global/residual surface candidates before report BIC
        - report_mle_rescue_max_candidates: total report candidates kept after rescue
        - report_mle_rescue_max_posterior_candidates: unverified posterior clusters added as rescue candidates
        - report_mle_rescue_max_residual_candidates: global/residual-ranked surface candidates added as rescue candidates
        - report_mle_rescue_dedup_radius_m: distance used to merge duplicate rescue candidates
        - report_mle_rescue_min_residual_fraction: residual fraction needed before surface-grid rescue
        - report_mle_rescue_visibility_weight: blend weight for visible-window rescue scoring
        - report_mle_rescue_min_visible_measurements: visible measurements needed for rescue candidates
        - report_mle_rescue_visible_count: per-measurement count threshold for visible rescue support
        - report_mle_rescue_visibility_reference_strength: optional fixed cps@1m for rescue visibility checks
        - report_cluster_model_selection: remove redundant reported clusters by refit-after-remove
        - report_cluster_bic_penalty_params: source parameter count for reported cluster BIC
        - report_cluster_delta_ll_threshold: tolerated report-model likelihood loss
        - report_model_order_require_posterior_match: require PF K posterior to match report K for stopping
        - report_model_order_prune_particles: apply report-level BIC rejections to PF source slots
        - report_model_order_particle_prune_radius_m: source-slot radius for BIC-driven PF pruning
        - report_model_order_zero_source_min_bic_margin: BIC margin required before a count-supported isotope is treated as absent
        - report_model_order_workers: workers for report-level subset BIC scoring
        - report_model_order_parallel_min_subsets: subset count needed before parallel scoring
        - min_age_to_split: minimum age before split proposals
        - use_clustered_output: use clustered estimate when birth is enabled
        - cluster_eps_m: clustering radius in meters
        - cluster_min_samples: minimum samples per cluster
        - split_prob: probability of split proposals per particle
        - split_strength_min: minimum strength for split candidates
        - split_position_sigma: position jitter for split proposals
        - split_strength_min_frac: min split fraction for q1/q2
        - split_strength_max_frac: max split fraction for q1/q2
        - split_delta_ll_threshold: ΔLL threshold for split acceptance
        - split_complexity_penalty: extra model-complexity penalty for split
        - split_residual_guided: use posterior residual candidates for split moves
        - split_residual_always_try: always test residual split on selected particles
        - split_residual_candidate_count: residual candidates evaluated per split
        - merge_prob: probability of merge proposals per particle
        - merge_distance_max: max distance for merge candidates
        - merge_delta_ll_threshold: ΔLL threshold for merge acceptance
        - merge_response_corr_min: response-correlation floor for merge candidates
        - merge_search_topk_pairs: max response-redundant pairs tested per merge move
        - structural_proposal_topk_particles: posterior-support cap for split/merge proposals
        - structural_trial_workers: worker count for deterministic split/merge trial chunks
        - structural_trial_parallel_min_trials: minimum trial count before worker chunks
        - source_position_prior: "volume" or "surface" PF source-position support
        - init_num_sources: inclusive range for initial source count per particle
        - init_grid_spacing_m: grid spacing for deterministic particle initialization
        - init_grid_repeats: repeated strength samples per deterministic grid point
        - roughening_k: roughening coefficient for post-resample position jitter
        - min_sigma_pos: minimum roughening sigma (meters)
        - max_sigma_pos: maximum roughening sigma (meters)
        - roughening_decay: multiplier decay per resample within an observation
        - roughening_min_mult: minimum multiplier for roughening decay
        - init_strength_log_mean: log-normal median for fallback strength initialization
        - init_strength_log_sigma: log-normal spread for fallback strength initialization
        - strength_log_sigma: log-space jitter for strengths
        - adaptive_strength_prior: rescale early strength particles from observed counts
        - adaptive_strength_prior_steps: number of first measurements allowed to rescale strengths
        - adaptive_strength_prior_min_counts: Poisson upper-count floor for zero/weak observations
        - adaptive_strength_prior_log_sigma: log-normal proposal spread around count-matched strength
        - adaptive_strength_prior_max_upscale: per-update upper strength multiplier
        - pose_min_observation_quantile: posterior quantile used for observability guarantees
        - orientation_k: maximum number of orientations to execute per pose
        - min_rotations_per_pose: minimum orientations before IG early stopping
        - orientation_selection_mode: "eig"
        - planning_particles: particle count used for orientation scoring (None = all)
        - planning_method: how to select planning particles (top_weight/resample)
        - use_gpu: enable torch acceleration for continuous kernel evaluation
        - gpu_device: torch device string (e.g., "cuda" or "cpu")
        - gpu_dtype: torch dtype string ("float32" or "float64")
        - target_ess_ratio: target ESS/N for tempered updates
        - max_temper_steps: max sub-steps for tempered updates
        - min_delta_beta: minimum delta_beta for tempering
        - use_tempering: enable ESS-targeted likelihood tempering
        - max_resamples_per_observation: cap resamples per observation update
        - temper_resample_cooldown_steps: substeps to skip resampling after resample
        - temper_resample_force_ratio: ESS/N ratio forcing resample despite cooldown
        - disable_regularize_on_temper_resample: skip roughening on temper resamples
        - deferred_resample_roughening_scale: roughening scale during station-burst resampling
        - cardinality_preserving_resample: preserve posterior source-count mass during resampling
        - mode_preserving_resample: keep distinct source modes during resampling
        - mode_preserving_max_modes: max spatial source modes protected per resample
        - mode_preserving_particles_per_mode: particles retained per protected mode
        - mode_preserving_radius_m: spatial clustering radius for protected source modes
        - mode_preserving_min_weight_fraction: minimum mode support fraction to protect
        - mode_preserving_cardinality_strata: keep source-count hypotheses during mode protection
        - mode_preserving_min_particles_per_cardinality: particles protected per source count
        - adapt_cooldown_steps: block particle-count shrink steps after resampling
        - eig_num_samples: Monte-Carlo samples for EIG (Eq. 3.44)
        - planning_eig_samples: Monte-Carlo samples for EIG inside planning rollouts
        - planning_rollout_particles: particle cap for IG evaluation in rollouts
        - planning_rollout_method: selection method for rollout particles
        - preselect_*: optional surrogate stage settings for candidate reduction
        - use_fast_gpu_rollout: enable approximate fast GPU rollouts for uncertainty prediction
        - ig_workers: number of parallel workers for IG grid evaluation (0 = auto)
        - use_tempering: enable ESS-targeted tempered updates in the PF
        - measurement_scale_by_isotope: isotope-wise source response scales
        - count_likelihood_model: "poisson", "gaussian", or "student_t"
        - transport_model_rel_sigma: relative model mismatch from scatter/build-up omissions
        - transport_model_abs_sigma: absolute transport-model mismatch floor in counts
        - spectrum_count_rel_sigma: relative spectrum-decomposition count uncertainty
        - spectrum_count_abs_sigma: additive spectrum-decomposition count uncertainty
        - low_count_abs_sigma: extra low-count uncertainty floor in counts
        - low_count_transition_counts: count scale where the low-count floor decays
        - count_likelihood_df: Student-t degrees of freedom for robust count likelihood
        - history_estimate_interval: exact report-history stride; 0 disables history
        - candidate_response_cache_max_entries: LRU entries for deterministic candidate responses
        - parallel_isotope_updates: run independent isotope structural updates in parallel
        - parallel_isotope_workers: worker count for parallel isotope structural updates
        - label_enable: enable label alignment for continuous particles
        - label_alignment_iters: iterations for label alignment refinement
        - label_pos_weight: position cost weight for label alignment
        - label_strength_weight: strength cost weight for label alignment
        - label_missing_cost: missing-source cost for label alignment
        - label_pos_scale: optional position scale for label alignment
        - label_strength_scale: optional strength scale for label alignment
        - converge_enable: enable per-isotope convergence gating
        - converge_window: window length for convergence checks
        - converge_map_move_eps_m: MMSE position stability threshold (meters)
        - converge_ess_ratio_high: ESS/N threshold for convergence
        - converge_ll_improve_eps: LL improvement tolerance
        - converge_min_steps: minimum steps before convergence
        - converge_require_all: if True, all criteria must hold; else any two
    """

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    ess_low: float = 0.5
    ess_high: float = 0.9
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: Dict[str, float] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | Dict[str, float] = 0.0
    transport_model_abs_sigma: float | Dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_transition_counts: float | Dict[str, float] = 0.0
    count_likelihood_df: float = 5.0
    min_strength: float = 0.01
    p_birth: float = 0.05
    p_kill: float = 0.1
    death_low_q_streak: int = 10
    death_delta_ll_threshold: float = 0.0
    support_ema_alpha: float = 0.3
    support_window: int = 1
    birth_window: int = 10
    birth_softmax_temp: float = 1.0
    birth_min_score: float = 1e-12
    birth_enable: bool = True
    birth_topk_particles: int = 10
    birth_use_weighted_topk: bool = True
    birth_min_sep_m: float = 0.8
    birth_detector_min_sep_m: float = 1.0
    source_detector_exclusion_m: float = 0.0
    birth_candidate_jitter_sigma: float = 0.5
    birth_num_local_jitter: int = 8
    birth_alpha: float = 0.2
    birth_q_max: float = 3e5
    birth_q_min: float = 1e2
    birth_max_per_update: int | None = None
    birth_delta_ll_threshold: float = 0.0
    birth_complexity_penalty: float = 0.0
    birth_bic_penalty_params: int = 4
    structural_update_min_counts: float = 0.0
    structural_update_min_snr: float = 0.0
    birth_min_distinct_poses: int = 1
    birth_residual_clip_quantile: float = 0.95
    birth_residual_gate_p_value: float = 0.05
    birth_residual_min_support: int = 2
    birth_residual_support_sigma: float = 1.0
    birth_min_distinct_stations: int = 1
    birth_candidate_support_fraction: float = 0.05
    birth_refit_residual_gate: bool = True
    birth_refit_residual_min_fraction: float = 0.5
    birth_use_shield_coded_residual: bool = True
    birth_existing_response_corr_max: float = 1.0
    birth_response_condition_max: float = 0.0
    birth_count_distance_prior_weight: float = 0.5
    birth_count_distance_strength_weight: float = 0.25
    birth_count_distance_log_clip: float = 3.0
    birth_count_distance_strength_sigma: float = 2.0
    birth_residual_always_try: bool = True
    birth_residual_expand_structural_particles: bool = True
    birth_residual_expanded_structural_topk_particles: int | None = 256
    birth_residual_acceptance_complexity_scale: float = 0.0
    birth_residual_force_proposal_on_gate: bool = True
    birth_residual_forced_min_delta_ll: float = -50.0
    birth_residual_force_relax_candidate_masks: bool = True
    birth_residual_suppress_death: bool = True
    birth_matching_pursuit_max_new_sources: int = 3
    birth_matching_pursuit_topk_candidates: int = 16
    birth_jitter_topk_candidates: int | None = 512
    birth_global_rescue_enable: bool = False
    birth_global_rescue_max_candidates: int = 8
    birth_global_rescue_min_residual_fraction: float = 0.005
    birth_global_rescue_dedup_radius_m: float = 0.5
    birth_global_rescue_forced_min_delta_ll: float = 0.0
    birth_global_rescue_min_support: int | None = None
    birth_global_rescue_min_distinct_poses: int | None = None
    birth_global_rescue_min_distinct_stations: int | None = None
    high_strength_split_enable: bool = True
    high_strength_split_q_multiple: float = 2.0
    high_strength_split_offset_m: float = 1.5
    high_strength_split_candidate_count: int = 12
    runtime_report_rescue_enable: bool = False
    runtime_report_rescue_particle_fraction: float = 0.15
    runtime_report_rescue_min_particles_per_source: int = 4
    runtime_report_rescue_weight: float = 0.10
    runtime_report_rescue_jitter_sigma_m: float = 0.10
    runtime_report_rescue_quarantine_enable: bool = True
    runtime_report_rescue_quarantine_weight: float = 0.02
    runtime_report_rescue_candidate_weight: float = 0.06
    runtime_report_rescue_memory_enable: bool = True
    runtime_report_rescue_memory_decay: float = 0.90
    runtime_report_rescue_memory_max_sources: int = 0
    report_mle_rescue_surface_quota_enable: bool = True
    report_mle_rescue_surface_quota_min_score_fraction: float = 0.0
    report_mle_rescue_surface_quota_per_stratum: int = 1
    report_mle_rescue_spatial_quota_enable: bool = True
    report_mle_rescue_spatial_quota_tile_m: float = 2.5
    report_mle_rescue_spatial_quota_per_tile: int = 1
    residual_decomposition_enable: bool = True
    peak_suppression_enable: bool = True
    peak_suppression_min_source_fraction: float = 0.25
    peak_suppression_factor: float = 1.0
    residual_decomposition_max_layers: int = 4
    pseudo_source_verification_enable: bool = True
    pseudo_source_min_delta_ll: float = 0.0
    pseudo_source_min_distinct_views: int = 2
    pseudo_source_fail_grace_stations: int = 2
    pseudo_source_corr_max: float = 0.995
    pseudo_source_temporal_sep_min: float = 0.0
    pseudo_source_quarantine_on_suppress: bool = True
    pseudo_source_quarantine_excludes_runtime: bool = False
    report_exclude_unverified_sources: bool = False
    source_prune_min_distinct_stations: int = 2
    source_prune_min_distinct_views: int = 2
    source_prune_fail_grace_stations: int = 2
    source_prune_delta_ll_threshold: float = 0.0
    source_prune_refit_after_remove: bool = True
    source_prune_bic_penalty_params: int = 4
    refit_after_moves: bool = True
    refit_iters: int = 3
    refit_eps: float = 1e-12
    weak_source_prune_min_expected_count: float = 0.0
    weak_source_prune_min_fraction: float = 0.0
    weak_source_prune_min_age: int = 0
    weak_source_prune_require_observable: bool = True
    weak_source_prune_min_observable_measurements: int = 1
    weak_source_prune_observable_count: float = 0.0
    weak_source_prune_observable_fraction: float = 0.0
    weak_source_prune_visibility_reference_strength: float = 0.0
    conditional_strength_refit: bool = True
    conditional_strength_refit_window: int = 10
    conditional_strength_refit_iters: int = 3
    conditional_strength_refit_reweight: bool = False
    conditional_strength_refit_cardinality_neutral_reweight: bool = True
    conditional_strength_refit_reweight_clip: float = 50.0
    conditional_strength_refit_min_count: float = 5.0
    conditional_strength_refit_min_snr: float = 1.0
    conditional_strength_refit_prior_weight: float = 0.0
    conditional_strength_refit_prior_rel_sigma: float = 2.0
    source_strength_prior_mean: float = 0.0
    source_strength_prior_weight: float = 0.0
    source_strength_prior_rel_sigma: float = 1.0
    source_strength_absorption_penalty_weight: float = 0.0
    source_strength_absorption_q_multiple: float = 4.0
    source_strength_observation_overshoot_penalty_weight: float = 0.0
    source_strength_observation_overshoot_sigma: float = 5.0
    source_strength_observation_overshoot_quantile: float = 0.05
    source_strength_observation_overshoot_min_visible_fraction: float = 0.05
    source_strength_observation_overshoot_min_visible_measurements: int = 3
    birth_stage_single_station_as_quarantine: bool = True
    report_strength_refit: bool = False
    report_strength_refit_iters: int = 64
    report_strength_refit_eps: float = 1.0e-9
    report_strength_refit_use_all_measurements: bool = True
    report_strength_refit_preserve_cardinality: bool = False
    report_strength_refit_prior_weight: float = 0.0
    report_strength_refit_prior_rel_sigma: float = 2.0
    report_strength_absorption_penalty_weight: float = 0.0
    report_strength_absorption_q_multiple: float = 4.0
    report_strength_observation_overshoot_penalty_weight: float = 0.0
    report_strength_observation_overshoot_sigma: float = 5.0
    report_strength_observation_overshoot_quantile: float = 0.05
    report_strength_observation_overshoot_min_visible_fraction: float = 0.05
    report_strength_observation_overshoot_min_visible_measurements: int = 3
    report_surface_local_refine: bool = False
    report_surface_local_refine_radius_m: float = 0.5
    report_surface_local_refine_grid_steps: int = 1
    report_surface_local_refine_max_candidates_per_source: int = 27
    report_surface_local_refine_max_sources: int = 0
    report_surface_local_refine_min_ll_gain: float = 0.0
    report_mle_rescue_enable: bool = False
    report_mle_rescue_max_candidates: int = 12
    report_mle_rescue_max_posterior_candidates: int = 8
    report_mle_rescue_max_residual_candidates: int = 8
    report_mle_rescue_dedup_radius_m: float = 0.5
    report_mle_rescue_min_residual_fraction: float = 0.01
    report_mle_rescue_visibility_weight: float = 0.0
    report_mle_rescue_min_visible_measurements: int = 1
    report_mle_rescue_visible_count: float = 0.0
    report_mle_rescue_visibility_reference_strength: float = 0.0
    report_cluster_model_selection: bool = True
    report_cluster_bic_penalty_params: int = 4
    report_cluster_delta_ll_threshold: float = 0.0
    report_cluster_model_selection_max_candidates: int = 12
    report_model_order_require_posterior_match: bool = True
    report_model_order_prune_particles: bool = False
    report_model_order_particle_prune_radius_m: float = 0.0
    report_model_order_min_bic_margin: float = 0.0
    report_model_order_zero_source_min_bic_margin: float = 10.0
    report_model_order_condition_max: float = 0.0
    report_model_order_workers: int = 1
    report_model_order_parallel_min_subsets: int = 128
    report_pre_finalize_guard: bool = True
    history_estimate_interval: int = 1
    candidate_response_cache_max_entries: int = 24
    min_age_to_split: int = 5
    use_clustered_output: bool = True
    cluster_eps_m: float = 0.8
    cluster_min_samples: int = 20
    cluster_report_max_points: int = 6000
    cluster_exact_max_points: int = 5000
    split_prob: float = 0.05
    split_strength_min: float = 0.1
    split_position_sigma: float = 0.25
    split_strength_min_frac: float = 0.3
    split_strength_max_frac: float = 0.7
    split_delta_ll_threshold: float = 0.0
    split_complexity_penalty: float = 0.0
    split_residual_guided: bool = True
    split_residual_always_try: bool = False
    split_residual_candidate_count: int = 8
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
    merge_response_corr_min: float = 0.995
    merge_search_topk_pairs: int = 8
    structural_proposal_topk_particles: int | None = None
    structural_trial_workers: int = 1
    structural_trial_parallel_min_trials: int = 8
    short_time_s: float = 0.5  # Recommended short-time measurement (Sec. 3.4.3).
    ig_threshold: float = 1e-3  # ΔIG stopping threshold (Sec. 3.4.4).
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    lambda_cost: float = 1.0  # Motion-cost weight (Eq. 3.51).
    alpha_weights: Dict[str, float] | None = None  # EIG isotope weights alpha_h.
    credible_volume_threshold: float = 1e-3  # Max 95% credible volume for convergence.
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 16
    min_delta_beta: float = 1e-3
    use_tempering: bool = True
    max_resamples_per_observation: int = 2
    temper_resample_cooldown_steps: int = 2
    temper_resample_force_ratio: float = 0.1
    disable_regularize_on_temper_resample: bool = False
    deferred_resample_roughening_scale: float = 0.15
    cardinality_preserving_resample: bool = True
    cardinality_preserving_min_stations: int = 0
    cardinality_preserving_require_confirmed_structure: bool = False
    mode_preserving_resample: bool = True
    mode_preserving_max_modes: int = 6
    mode_preserving_particles_per_mode: int = 3
    mode_preserving_radius_m: float = 1.5
    mode_preserving_min_weight_fraction: float = 1e-4
    mode_preserving_surface_strata: bool = True
    mode_preserving_height_bin_m: float = 2.0
    mode_preserving_high_surface_extra_particles: int = 0
    mode_preserving_high_surface_z_fraction: float = 0.75
    mode_preserving_support_score_weight: float = 0.0
    mode_preserving_tentative_boost: float = 1.0
    mode_preserving_residual_boost: float = 1.0
    mode_preserving_cardinality_strata: bool = True
    mode_preserving_min_particles_per_cardinality: int = 2
    adapt_cooldown_steps: int = 0
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "volume"
    init_num_sources: Tuple[int, int] = (0, 3)
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
    roughening_k: float = 0.5
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_strength_log_mean: float = 9.0
    init_strength_log_sigma: float = 1.0
    strength_log_sigma: float = 0.3
    adaptive_strength_prior: bool = False
    adaptive_strength_prior_steps: int = 3
    adaptive_strength_prior_min_counts: float = 3.0
    adaptive_strength_prior_log_sigma: float = 0.7
    adaptive_strength_prior_max_upscale: float = 10.0
    pose_min_observation_counts: float = 0.0
    pose_min_observation_penalty_scale: float = 1.0
    pose_min_observation_aggregate: str = "max"
    pose_min_observation_max_particles: int | None = None
    pose_min_observation_quantile: float = 0.25
    orientation_k: int = 8
    min_rotations_per_pose: int = 0
    orientation_selection_mode: str = "eig"
    planning_particles: int | None = None
    planning_method: str = "top_weight"
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    eig_num_samples: int = 50
    planning_eig_samples: int | None = None
    planning_rollout_particles: int | None = None
    planning_rollout_method: str | None = None
    preselect_orientations: bool = False
    preselect_metric: str = "var_log_lambda"
    preselect_delta: float = 0.05
    preselect_k_min: int = 8
    preselect_k_max: int = 16
    use_fast_gpu_rollout: bool = False
    ig_workers: int = 0
    parallel_isotope_updates: bool = True
    parallel_isotope_workers: int | None = None
    label_enable: bool = True
    label_alignment_iters: int = 2
    label_pos_weight: float = 1.0
    label_strength_weight: float = 0.2
    label_missing_cost: float = 1e3
    label_pos_scale: float | None = None
    label_strength_scale: float | None = None
    converge_enable: bool = False
    converge_window: int = 8
    converge_map_move_eps_m: float = 0.4
    converge_ess_ratio_high: float = 0.2
    converge_ll_improve_eps: float = 1e5
    converge_min_steps: int = 30
    converge_require_all: bool = True
    converge_cardinality_var_max: float = 0.05
    converge_require_no_tentative: bool = True
    converge_freeze_updates: bool = False
    converge_min_stations: int = 0
    converge_cluster_spread_max_m: float = 0.0
    converge_cluster_min_support_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize estimator configuration values."""
        if self.min_particles is None:
            self.min_particles = max(1, int(self.num_particles * 0.5))
        if self.max_particles is None:
            self.max_particles = max(self.num_particles, int(self.num_particles * 2.0))
        self.ess_low = float(self.ess_low)
        self.ess_high = float(self.ess_high)
        if not 0.0 < self.ess_low < self.ess_high < 1.0:
            raise ValueError(
                "ess_low and ess_high must satisfy 0 < ess_low < ess_high < 1."
            )
        self.init_grid_repeats = max(1, int(self.init_grid_repeats))
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")
        self.adaptive_strength_prior_steps = int(self.adaptive_strength_prior_steps)
        if self.adaptive_strength_prior_steps < 0:
            raise ValueError("adaptive_strength_prior_steps must be >= 0.")
        self.adaptive_strength_prior_min_counts = float(
            self.adaptive_strength_prior_min_counts
        )
        if self.adaptive_strength_prior_min_counts < 0.0:
            raise ValueError("adaptive_strength_prior_min_counts must be >= 0.")
        self.adaptive_strength_prior_log_sigma = float(
            self.adaptive_strength_prior_log_sigma
        )
        if self.adaptive_strength_prior_log_sigma < 0.0:
            raise ValueError("adaptive_strength_prior_log_sigma must be >= 0.")
        self.adaptive_strength_prior_max_upscale = float(
            self.adaptive_strength_prior_max_upscale
        )
        if self.adaptive_strength_prior_max_upscale < 1.0:
            raise ValueError("adaptive_strength_prior_max_upscale must be >= 1.")
        self.pose_min_observation_counts = float(self.pose_min_observation_counts)
        if self.pose_min_observation_counts < 0.0:
            raise ValueError("pose_min_observation_counts must be >= 0.")
        self.pose_min_observation_penalty_scale = float(
            self.pose_min_observation_penalty_scale
        )
        if self.pose_min_observation_penalty_scale < 0.0:
            raise ValueError("pose_min_observation_penalty_scale must be >= 0.")
        self.pose_min_observation_aggregate = (
            str(self.pose_min_observation_aggregate).strip().lower()
        )
        if self.pose_min_observation_aggregate not in {"max", "mean"}:
            raise ValueError("pose_min_observation_aggregate must be max or mean.")
        if self.pose_min_observation_max_particles is not None:
            self.pose_min_observation_max_particles = int(
                self.pose_min_observation_max_particles
            )
            if self.pose_min_observation_max_particles < 0:
                raise ValueError("pose_min_observation_max_particles must be >= 0.")
        self.pose_min_observation_quantile = float(self.pose_min_observation_quantile)
        if not 0.0 <= self.pose_min_observation_quantile <= 1.0:
            raise ValueError("pose_min_observation_quantile must be in [0, 1].")
        normalized_likelihood = str(self.count_likelihood_model).strip().lower()
        if normalized_likelihood in {"normal"}:
            normalized_likelihood = "gaussian"
        if normalized_likelihood in {"robust", "robust_gaussian", "t"}:
            normalized_likelihood = "student_t"
        if normalized_likelihood not in {"poisson", "gaussian", "student_t"}:
            raise ValueError(
                "count_likelihood_model must be poisson, gaussian, or student_t."
            )
        self.count_likelihood_model = normalized_likelihood
        self.count_likelihood_df = max(float(self.count_likelihood_df), 1.0)
        self.birth_residual_gate_p_value = float(self.birth_residual_gate_p_value)
        if self.birth_residual_gate_p_value < 0.0:
            raise ValueError("birth_residual_gate_p_value must be >= 0.")
        self.birth_residual_gate_p_value = min(self.birth_residual_gate_p_value, 1.0)
        self.birth_residual_min_support = max(1, int(self.birth_residual_min_support))
        self.birth_residual_support_sigma = max(
            0.0,
            float(self.birth_residual_support_sigma),
        )
        self.birth_candidate_support_fraction = float(
            np.clip(float(self.birth_candidate_support_fraction), 0.0, 1.0)
        )
        self.source_detector_exclusion_m = max(
            0.0,
            float(self.source_detector_exclusion_m),
        )
        self.birth_refit_residual_min_fraction = max(
            0.0,
            float(self.birth_refit_residual_min_fraction),
        )
        self.birth_existing_response_corr_max = float(
            np.clip(float(self.birth_existing_response_corr_max), 0.0, 1.0)
        )
        self.birth_response_condition_max = max(
            0.0,
            float(self.birth_response_condition_max),
        )
        self.structural_update_min_snr = max(
            0.0,
            float(self.structural_update_min_snr),
        )
        self.converge_cluster_spread_max_m = max(
            0.0,
            float(self.converge_cluster_spread_max_m),
        )
        self.converge_cluster_min_support_fraction = float(
            np.clip(float(self.converge_cluster_min_support_fraction), 0.0, 1.0)
        )
        self.birth_count_distance_prior_weight = max(
            0.0,
            float(self.birth_count_distance_prior_weight),
        )
        self.birth_count_distance_strength_weight = max(
            0.0,
            float(self.birth_count_distance_strength_weight),
        )
        self.birth_count_distance_log_clip = max(
            0.0,
            float(self.birth_count_distance_log_clip),
        )
        self.birth_count_distance_strength_sigma = max(
            1.0e-12,
            float(self.birth_count_distance_strength_sigma),
        )
        self.birth_matching_pursuit_max_new_sources = max(
            1,
            int(self.birth_matching_pursuit_max_new_sources),
        )
        self.birth_matching_pursuit_topk_candidates = max(
            1,
            int(self.birth_matching_pursuit_topk_candidates),
        )
        if self.birth_residual_expanded_structural_topk_particles is not None:
            expanded_topk = int(self.birth_residual_expanded_structural_topk_particles)
            self.birth_residual_expanded_structural_topk_particles = (
                None if expanded_topk <= 0 else expanded_topk
            )
        self.mode_preserving_max_modes = max(
            0,
            int(self.mode_preserving_max_modes),
        )
        self.cardinality_preserving_min_stations = max(
            0,
            int(self.cardinality_preserving_min_stations),
        )
        self.cardinality_preserving_require_confirmed_structure = bool(
            self.cardinality_preserving_require_confirmed_structure
        )
        self.deferred_resample_roughening_scale = max(
            0.0,
            float(self.deferred_resample_roughening_scale),
        )
        self.mode_preserving_particles_per_mode = max(
            0,
            int(self.mode_preserving_particles_per_mode),
        )
        self.mode_preserving_radius_m = max(
            1.0e-6,
            float(self.mode_preserving_radius_m),
        )
        self.mode_preserving_min_weight_fraction = max(
            0.0,
            float(self.mode_preserving_min_weight_fraction),
        )
        self.mode_preserving_surface_strata = bool(
            self.mode_preserving_surface_strata
        )
        self.mode_preserving_height_bin_m = max(
            0.0,
            float(self.mode_preserving_height_bin_m),
        )
        self.mode_preserving_high_surface_extra_particles = max(
            0,
            int(self.mode_preserving_high_surface_extra_particles),
        )
        self.mode_preserving_high_surface_z_fraction = float(
            np.clip(float(self.mode_preserving_high_surface_z_fraction), 0.0, 1.0)
        )
        self.mode_preserving_support_score_weight = max(
            0.0,
            float(self.mode_preserving_support_score_weight),
        )
        self.mode_preserving_tentative_boost = max(
            1.0,
            float(self.mode_preserving_tentative_boost),
        )
        self.mode_preserving_residual_boost = max(
            1.0,
            float(self.mode_preserving_residual_boost),
        )
        self.mode_preserving_cardinality_strata = bool(
            self.mode_preserving_cardinality_strata
        )
        self.mode_preserving_min_particles_per_cardinality = max(
            0,
            int(self.mode_preserving_min_particles_per_cardinality),
        )
        if isinstance(self.source_position_prior, bool):
            prior = "surface" if self.source_position_prior else "volume"
        else:
            prior = str(self.source_position_prior).strip().lower()
        if prior in {"surface_constrained", "surface-constrained", "surfaces"}:
            prior = "surface"
        if prior not in {"volume", "surface"}:
            raise ValueError("source_position_prior must be 'volume' or 'surface'.")
        self.source_position_prior = prior
        if (
            bool(self.birth_enable)
            and self.max_sources is not None
            and int(self.max_sources) > 1
        ):
            self.use_clustered_output = True
        if self.birth_jitter_topk_candidates is not None:
            self.birth_jitter_topk_candidates = max(
                1,
                int(self.birth_jitter_topk_candidates),
            )
        self.birth_global_rescue_enable = bool(self.birth_global_rescue_enable)
        self.birth_global_rescue_max_candidates = max(
            0,
            int(self.birth_global_rescue_max_candidates),
        )
        self.birth_global_rescue_min_residual_fraction = max(
            0.0,
            float(self.birth_global_rescue_min_residual_fraction),
        )
        self.birth_global_rescue_dedup_radius_m = max(
            0.0,
            float(self.birth_global_rescue_dedup_radius_m),
        )
        self.birth_global_rescue_forced_min_delta_ll = float(
            self.birth_global_rescue_forced_min_delta_ll
        )
        if self.birth_global_rescue_min_support is not None:
            self.birth_global_rescue_min_support = max(
                1,
                int(self.birth_global_rescue_min_support),
            )
        if self.birth_global_rescue_min_distinct_poses is not None:
            self.birth_global_rescue_min_distinct_poses = max(
                1,
                int(self.birth_global_rescue_min_distinct_poses),
            )
        if self.birth_global_rescue_min_distinct_stations is not None:
            self.birth_global_rescue_min_distinct_stations = max(
                1,
                int(self.birth_global_rescue_min_distinct_stations),
            )
        self.high_strength_split_enable = bool(self.high_strength_split_enable)
        self.high_strength_split_q_multiple = max(
            1.0,
            float(self.high_strength_split_q_multiple),
        )
        self.high_strength_split_offset_m = max(
            1.0e-6,
            float(self.high_strength_split_offset_m),
        )
        self.high_strength_split_candidate_count = max(
            1,
            int(self.high_strength_split_candidate_count),
        )
        self.runtime_report_rescue_enable = bool(self.runtime_report_rescue_enable)
        self.runtime_report_rescue_particle_fraction = float(
            np.clip(float(self.runtime_report_rescue_particle_fraction), 0.0, 1.0)
        )
        self.runtime_report_rescue_min_particles_per_source = max(
            1,
            int(self.runtime_report_rescue_min_particles_per_source),
        )
        self.runtime_report_rescue_weight = float(
            np.clip(float(self.runtime_report_rescue_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_jitter_sigma_m = max(
            0.0,
            float(self.runtime_report_rescue_jitter_sigma_m),
        )
        self.runtime_report_rescue_quarantine_enable = bool(
            self.runtime_report_rescue_quarantine_enable
        )
        self.runtime_report_rescue_quarantine_weight = float(
            np.clip(float(self.runtime_report_rescue_quarantine_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_candidate_weight = float(
            np.clip(float(self.runtime_report_rescue_candidate_weight), 0.0, 0.5)
        )
        self.runtime_report_rescue_memory_enable = bool(
            self.runtime_report_rescue_memory_enable
        )
        self.runtime_report_rescue_memory_decay = float(
            np.clip(float(self.runtime_report_rescue_memory_decay), 0.0, 1.0)
        )
        self.runtime_report_rescue_memory_max_sources = max(
            0,
            int(self.runtime_report_rescue_memory_max_sources),
        )
        self.report_mle_rescue_surface_quota_enable = bool(
            self.report_mle_rescue_surface_quota_enable
        )
        self.report_mle_rescue_surface_quota_min_score_fraction = max(
            0.0,
            float(self.report_mle_rescue_surface_quota_min_score_fraction),
        )
        self.report_mle_rescue_surface_quota_per_stratum = max(
            1,
            int(self.report_mle_rescue_surface_quota_per_stratum),
        )
        self.report_mle_rescue_spatial_quota_enable = bool(
            self.report_mle_rescue_spatial_quota_enable
        )
        self.report_mle_rescue_spatial_quota_tile_m = max(
            1.0e-6,
            float(self.report_mle_rescue_spatial_quota_tile_m),
        )
        self.report_mle_rescue_spatial_quota_per_tile = max(
            1,
            int(self.report_mle_rescue_spatial_quota_per_tile),
        )
        self.residual_decomposition_enable = bool(self.residual_decomposition_enable)
        self.peak_suppression_enable = bool(self.peak_suppression_enable)
        self.peak_suppression_min_source_fraction = float(
            np.clip(float(self.peak_suppression_min_source_fraction), 0.0, 1.0)
        )
        self.peak_suppression_factor = float(
            np.clip(float(self.peak_suppression_factor), 0.0, 1.0)
        )
        self.residual_decomposition_max_layers = max(
            1,
            int(self.residual_decomposition_max_layers),
        )
        self.pseudo_source_verification_enable = bool(
            self.pseudo_source_verification_enable
        )
        self.pseudo_source_min_delta_ll = float(self.pseudo_source_min_delta_ll)
        self.pseudo_source_min_distinct_views = max(
            1,
            int(self.pseudo_source_min_distinct_views),
        )
        self.pseudo_source_fail_grace_stations = max(
            0,
            int(self.pseudo_source_fail_grace_stations),
        )
        self.pseudo_source_corr_max = float(
            np.clip(float(self.pseudo_source_corr_max), 0.0, 1.0)
        )
        self.pseudo_source_temporal_sep_min = max(
            0.0,
            float(self.pseudo_source_temporal_sep_min),
        )
        self.pseudo_source_quarantine_on_suppress = bool(
            self.pseudo_source_quarantine_on_suppress
        )
        self.pseudo_source_quarantine_excludes_runtime = bool(
            self.pseudo_source_quarantine_excludes_runtime
        )
        self.report_exclude_unverified_sources = bool(
            self.report_exclude_unverified_sources
        )
        self.source_prune_min_distinct_stations = max(
            1,
            int(self.source_prune_min_distinct_stations),
        )
        self.source_prune_min_distinct_views = max(
            1,
            int(self.source_prune_min_distinct_views),
        )
        self.source_prune_fail_grace_stations = max(
            1,
            int(self.source_prune_fail_grace_stations),
        )
        self.source_prune_delta_ll_threshold = float(
            self.source_prune_delta_ll_threshold
        )
        self.source_prune_refit_after_remove = bool(
            self.source_prune_refit_after_remove
        )
        self.source_prune_bic_penalty_params = max(
            0,
            int(self.source_prune_bic_penalty_params),
        )
        self.weak_source_prune_min_expected_count = max(
            0.0,
            float(self.weak_source_prune_min_expected_count),
        )
        self.weak_source_prune_min_fraction = max(
            0.0,
            float(self.weak_source_prune_min_fraction),
        )
        self.weak_source_prune_min_age = max(0, int(self.weak_source_prune_min_age))
        self.weak_source_prune_require_observable = bool(
            self.weak_source_prune_require_observable
        )
        self.weak_source_prune_min_observable_measurements = max(
            1,
            int(self.weak_source_prune_min_observable_measurements),
        )
        self.weak_source_prune_observable_count = max(
            0.0,
            float(self.weak_source_prune_observable_count),
        )
        self.weak_source_prune_observable_fraction = max(
            0.0,
            float(self.weak_source_prune_observable_fraction),
        )
        self.weak_source_prune_visibility_reference_strength = max(
            0.0,
            float(self.weak_source_prune_visibility_reference_strength),
        )
        self.birth_residual_always_try = bool(self.birth_residual_always_try)
        self.birth_residual_expand_structural_particles = bool(
            self.birth_residual_expand_structural_particles
        )
        self.birth_residual_acceptance_complexity_scale = float(
            np.clip(
                float(self.birth_residual_acceptance_complexity_scale),
                0.0,
                1.0,
            )
        )
        self.birth_residual_suppress_death = bool(self.birth_residual_suppress_death)
        if self.birth_max_per_update is not None:
            self.birth_max_per_update = max(0, int(self.birth_max_per_update))
        self.birth_delta_ll_threshold = float(self.birth_delta_ll_threshold)
        self.birth_complexity_penalty = max(0.0, float(self.birth_complexity_penalty))
        self.birth_bic_penalty_params = max(0, int(self.birth_bic_penalty_params))
        self.conditional_strength_refit_window = max(
            1,
            int(self.conditional_strength_refit_window),
        )
        self.conditional_strength_refit_iters = max(
            1,
            int(self.conditional_strength_refit_iters),
        )
        self.conditional_strength_refit_reweight_clip = max(
            0.0,
            float(self.conditional_strength_refit_reweight_clip),
        )
        self.conditional_strength_refit_min_count = max(
            0.0,
            float(self.conditional_strength_refit_min_count),
        )
        self.conditional_strength_refit_min_snr = max(
            0.0,
            float(self.conditional_strength_refit_min_snr),
        )
        self.conditional_strength_refit_prior_weight = max(
            0.0,
            float(self.conditional_strength_refit_prior_weight),
        )
        self.conditional_strength_refit_prior_rel_sigma = max(
            1.0e-6,
            float(self.conditional_strength_refit_prior_rel_sigma),
        )
        self.source_strength_prior_mean = max(
            0.0,
            float(self.source_strength_prior_mean),
        )
        self.source_strength_prior_weight = max(
            0.0,
            float(self.source_strength_prior_weight),
        )
        self.source_strength_prior_rel_sigma = max(
            1.0e-6,
            float(self.source_strength_prior_rel_sigma),
        )
        self.source_strength_absorption_penalty_weight = max(
            0.0,
            float(self.source_strength_absorption_penalty_weight),
        )
        self.source_strength_absorption_q_multiple = max(
            1.0,
            float(self.source_strength_absorption_q_multiple),
        )
        self.source_strength_observation_overshoot_penalty_weight = max(
            0.0,
            float(self.source_strength_observation_overshoot_penalty_weight),
        )
        self.source_strength_observation_overshoot_sigma = max(
            0.0,
            float(self.source_strength_observation_overshoot_sigma),
        )
        self.source_strength_observation_overshoot_quantile = float(
            np.clip(
                float(self.source_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        self.source_strength_observation_overshoot_min_visible_fraction = max(
            0.0,
            float(self.source_strength_observation_overshoot_min_visible_fraction),
        )
        self.source_strength_observation_overshoot_min_visible_measurements = max(
            1,
            int(self.source_strength_observation_overshoot_min_visible_measurements),
        )
        self.birth_stage_single_station_as_quarantine = bool(
            self.birth_stage_single_station_as_quarantine
        )
        self.report_strength_refit_iters = max(1, int(self.report_strength_refit_iters))
        self.report_strength_refit_eps = max(
            1.0e-15,
            float(self.report_strength_refit_eps),
        )
        self.report_strength_refit_use_all_measurements = bool(
            self.report_strength_refit_use_all_measurements
        )
        self.report_strength_refit_preserve_cardinality = bool(
            self.report_strength_refit_preserve_cardinality
        )
        self.report_strength_refit_prior_weight = max(
            0.0,
            float(self.report_strength_refit_prior_weight),
        )
        self.report_strength_refit_prior_rel_sigma = max(
            1.0e-6,
            float(self.report_strength_refit_prior_rel_sigma),
        )
        self.report_strength_absorption_penalty_weight = max(
            0.0,
            float(self.report_strength_absorption_penalty_weight),
        )
        self.report_strength_absorption_q_multiple = max(
            1.0,
            float(self.report_strength_absorption_q_multiple),
        )
        self.report_strength_observation_overshoot_penalty_weight = max(
            0.0,
            float(self.report_strength_observation_overshoot_penalty_weight),
        )
        self.report_strength_observation_overshoot_sigma = max(
            0.0,
            float(self.report_strength_observation_overshoot_sigma),
        )
        self.report_strength_observation_overshoot_quantile = float(
            np.clip(
                float(self.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        self.report_strength_observation_overshoot_min_visible_fraction = max(
            0.0,
            float(self.report_strength_observation_overshoot_min_visible_fraction),
        )
        self.report_strength_observation_overshoot_min_visible_measurements = max(
            1,
            int(self.report_strength_observation_overshoot_min_visible_measurements),
        )
        self.report_mle_rescue_enable = bool(self.report_mle_rescue_enable)
        self.report_mle_rescue_max_candidates = max(
            1,
            int(self.report_mle_rescue_max_candidates),
        )
        self.report_mle_rescue_max_posterior_candidates = max(
            0,
            int(self.report_mle_rescue_max_posterior_candidates),
        )
        self.report_mle_rescue_max_residual_candidates = max(
            0,
            int(self.report_mle_rescue_max_residual_candidates),
        )
        self.report_surface_local_refine = bool(self.report_surface_local_refine)
        self.report_surface_local_refine_radius_m = max(
            0.0,
            float(self.report_surface_local_refine_radius_m),
        )
        self.report_surface_local_refine_grid_steps = max(
            0,
            int(self.report_surface_local_refine_grid_steps),
        )
        self.report_surface_local_refine_max_candidates_per_source = max(
            1,
            int(self.report_surface_local_refine_max_candidates_per_source),
        )
        self.report_surface_local_refine_max_sources = max(
            0,
            int(self.report_surface_local_refine_max_sources),
        )
        self.report_surface_local_refine_min_ll_gain = max(
            0.0,
            float(self.report_surface_local_refine_min_ll_gain),
        )
        self.report_mle_rescue_dedup_radius_m = max(
            0.0,
            float(self.report_mle_rescue_dedup_radius_m),
        )
        self.report_mle_rescue_min_residual_fraction = max(
            0.0,
            float(self.report_mle_rescue_min_residual_fraction),
        )
        self.report_mle_rescue_visibility_weight = float(
            np.clip(float(self.report_mle_rescue_visibility_weight), 0.0, 1.0)
        )
        self.report_mle_rescue_min_visible_measurements = max(
            1,
            int(self.report_mle_rescue_min_visible_measurements),
        )
        self.report_mle_rescue_visible_count = max(
            0.0,
            float(self.report_mle_rescue_visible_count),
        )
        self.report_mle_rescue_visibility_reference_strength = max(
            0.0,
            float(self.report_mle_rescue_visibility_reference_strength),
        )
        self.report_cluster_model_selection = bool(self.report_cluster_model_selection)
        self.report_cluster_bic_penalty_params = max(
            0,
            int(self.report_cluster_bic_penalty_params),
        )
        self.report_cluster_delta_ll_threshold = float(
            self.report_cluster_delta_ll_threshold
        )
        self.report_cluster_model_selection_max_candidates = max(
            1,
            int(self.report_cluster_model_selection_max_candidates),
        )
        self.report_model_order_require_posterior_match = bool(
            self.report_model_order_require_posterior_match
        )
        self.report_model_order_prune_particles = bool(
            self.report_model_order_prune_particles
        )
        self.report_model_order_particle_prune_radius_m = max(
            0.0,
            float(self.report_model_order_particle_prune_radius_m),
        )
        self.report_model_order_min_bic_margin = max(
            0.0,
            float(self.report_model_order_min_bic_margin),
        )
        self.report_model_order_zero_source_min_bic_margin = max(
            0.0,
            float(self.report_model_order_zero_source_min_bic_margin),
        )
        self.report_model_order_condition_max = max(
            0.0,
            float(self.report_model_order_condition_max),
        )
        self.report_model_order_workers = max(
            1,
            int(self.report_model_order_workers),
        )
        self.report_model_order_parallel_min_subsets = max(
            1,
            int(self.report_model_order_parallel_min_subsets),
        )
        self.report_pre_finalize_guard = bool(self.report_pre_finalize_guard)
        self.history_estimate_interval = max(0, int(self.history_estimate_interval))
        self.candidate_response_cache_max_entries = max(
            0,
            int(self.candidate_response_cache_max_entries),
        )
        self.converge_cardinality_var_max = max(
            0.0,
            float(self.converge_cardinality_var_max),
        )
        self.converge_require_no_tentative = bool(self.converge_require_no_tentative)
        self.converge_freeze_updates = bool(self.converge_freeze_updates)
        self.converge_min_stations = max(0, int(self.converge_min_stations))
        self.split_residual_guided = bool(self.split_residual_guided)
        self.split_complexity_penalty = max(0.0, float(self.split_complexity_penalty))
        self.split_residual_always_try = bool(self.split_residual_always_try)
        self.split_residual_candidate_count = max(
            1,
            int(self.split_residual_candidate_count),
        )
        self.merge_response_corr_min = float(
            np.clip(float(self.merge_response_corr_min), 0.0, 1.0)
        )
        self.merge_search_topk_pairs = max(1, int(self.merge_search_topk_pairs))
        self.structural_trial_workers = max(1, int(self.structural_trial_workers))
        self.structural_trial_parallel_min_trials = max(
            1,
            int(self.structural_trial_parallel_min_trials),
        )
        self.parallel_isotope_updates = bool(self.parallel_isotope_updates)
        if self.parallel_isotope_workers is not None:
            self.parallel_isotope_workers = max(1, int(self.parallel_isotope_workers))


@dataclass(frozen=True)
class MeasurementRecord:
    """Store a single isotope-wise measurement and metadata."""

    z_k: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float
    fe_index: int | None = None
    pb_index: int | None = None
    z_variance_k: Dict[str, float] | None = None
    ig_value: float | None = None


class RotatingShieldPFEstimator:
    """
    Online source estimator using parallel PFs with shield rotation (Sec. 3.4–3.6).

    - Maintains one PF per isotope.
    - Updates each PF with pose/orientation and Poisson weight updates.
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        candidate_sources: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: Dict[str, object] | None,
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: Dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_samples: int = 1,
    ) -> None:
        """Initialize per-isotope filters and shared measurement-model state."""
        self.all_isotopes = list(isotopes)
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = float(obstacle_height_m)
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = max(float(obstacle_buildup_coeff), 0.0)
        self.detector_radius_m = max(float(detector_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        # Measurement poses are appended incrementally.
        self.poses: List[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: KernelPrecomputer | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        ] = []
        self.history_scores: List[float] = []
        self.measurements: List[MeasurementRecord] = []
        self.last_strength_prior_diagnostics: Dict[str, Dict[str, float]] = {}
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        self._previous_deferred_measurement_count = 0
        self._pre_finalize_guard_estimates: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._last_report_model_order_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._runtime_report_rescue_modes: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], float],
        ] = {}
        self._runtime_report_rescue_memory: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = 0.0
        self._report_cache_revision = 0
        self._report_estimate_cache: dict[
            tuple[int, bool],
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
        ] = {}
        self._candidate_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._candidate_response_cache_order: list[tuple[Any, ...]] = []

    @staticmethod
    def _copy_estimate_map(
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return a deep array copy of a per-isotope estimate mapping."""
        return {
            isotope: (
                np.asarray(positions, dtype=float).copy(),
                np.asarray(strengths, dtype=float).copy(),
            )
            for isotope, (positions, strengths) in estimates.items()
        }

    def _invalidate_report_cache(self) -> None:
        """Invalidate cached report estimates after any PF/report state change."""
        self._report_cache_revision += 1
        self._report_estimate_cache.clear()

    def _record_history_estimate(self, measurement_count: int) -> None:
        """Record an exact report estimate when the configured history stride allows it."""
        interval = max(
            0,
            int(getattr(self.pf_config, "history_estimate_interval", 1)),
        )
        if interval <= 0:
            return
        count = max(0, int(measurement_count))
        if count <= 0 or count % interval != 0:
            return
        self.history_estimates.append(self.estimates())

    def _candidate_response_source_key(
        self,
        sources: NDArray[np.float64],
    ) -> tuple[str, int] | None:
        """Return a stable cache key for the full shared candidate-source grid."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        candidate_arr = np.asarray(self.candidate_sources, dtype=float).reshape(
            -1,
            3,
        )
        if (
            source_arr.shape == candidate_arr.shape
            and source_arr.size > 0
            and np.shares_memory(source_arr, candidate_arr)
        ):
            return ("candidate_sources", int(source_arr.shape[0]))
        return None

    @staticmethod
    def _measurement_geometry_digest(data: MeasurementData) -> bytes:
        """Return a compact digest for measurement geometry arrays used by responses."""
        digest = hashlib.blake2b(digest_size=16)
        for array in (
            np.asarray(data.detector_positions, dtype=np.float64),
            np.asarray(data.live_times, dtype=np.float64),
            np.asarray(data.fe_indices, dtype=np.int64),
            np.asarray(data.pb_indices, dtype=np.int64),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.digest()

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts, reusing deterministic full-grid responses."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._candidate_response_source_key(source_arr)
        scale = float(self.response_scale_for_isotope(isotope))
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.candidate_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(filt.continuous_kernel)),
                source_key,
                self._measurement_geometry_digest(data),
                float(scale),
            )
            cached = self._candidate_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        counts = expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=source_arr,
            strengths=strength_arr,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=scale,
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._candidate_response_cache[cache_key] = counts_arr.copy()
            self._candidate_response_cache_order.append(cache_key)
            max_entries = max(
                0,
                int(self.pf_config.candidate_response_cache_max_entries),
            )
            while len(self._candidate_response_cache_order) > max_entries:
                old_key = self._candidate_response_cache_order.pop(0)
                if old_key not in self._candidate_response_cache_order:
                    self._candidate_response_cache.pop(old_key, None)
        return counts_arr

    def _cached_candidate_grid_counts(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength responses for the full source-candidate grid."""
        pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        if pool.size == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=pool,
            strengths=np.ones(pool.shape[0], dtype=float),
        )
        return np.asarray(counts, dtype=float)

    def _resolve_mu_by_isotope(
        self, mu_by_isotope: Dict[str, object] | None
    ) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
            """Return a normalized isotope key for attenuation lookup."""
            return re.sub(r"[^A-Za-z0-9]", "", name).upper()

        canonical_by_norm = {
            "CS137": "Cs-137",
            "CO60": "Co-60",
            "EU154": "Eu-154",
        }

        resolved: Dict[str, object] = {}
        if mu_by_isotope is not None:
            resolved.update(mu_by_isotope)
        normalized: Dict[str, object] = {}
        for key, value in resolved.items():
            normalized[_norm_key(key)] = value
        isotope_names = (
            self.all_isotopes if hasattr(self, "all_isotopes") else self.isotopes
        )
        if isotope_names:
            still_missing: List[str] = []
            for iso in isotope_names:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(
                        HVL_TVL_TABLE_MM, isotopes=[canonical]
                    )
                    if canonical in table_vals:
                        resolved[iso] = table_vals[canonical]
                        normalized[norm] = table_vals[canonical]
                        if canonical not in resolved:
                            resolved[canonical] = table_vals[canonical]
                        continue
                still_missing.append(iso)
            if still_missing:
                missing_list = ", ".join(still_missing)
                raise ValueError(
                    "mu_by_isotope is missing entries for isotopes: "
                    f"{missing_list}. Ensure isotope names match the HVL/TVL table keys."
                )
        return resolved

    def _ensure_kernel_cache(self) -> None:
        """Build the discrete kernel cache when it is first needed."""
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = KernelPrecomputer(
            candidate_sources=self.candidate_sources,
            poses=poses_arr,
            orientations=self.normals,
            shield_params=self.shield_params,
            mu_by_isotope=self.mu_by_isotope,
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
        )
        pf_conf = self._build_pf_config()
        if self.filters:
            for iso in self.isotopes:
                if iso in self.filters:
                    self.filters[iso].set_kernel(self.kernel_cache)
                else:
                    self.filters[iso] = self._build_filter(iso, pf_conf)
        else:
            for iso in self.isotopes:
                self.filters[iso] = self._build_filter(iso, pf_conf)

    def _build_filter(self, isotope: str, pf_conf: PFConfig) -> IsotopeParticleFilter:
        """Build an isotope filter with shared PF observation-model settings."""
        return IsotopeParticleFilter(
            isotope,
            kernel=self.kernel_cache,
            config=pf_conf,
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
        )

    def _build_pf_config(self) -> PFConfig:
        """Build a per-isotope PFConfig from the estimator configuration."""
        return PFConfig(
            num_particles=self.pf_config.num_particles,
            min_particles=self.pf_config.min_particles,
            max_particles=self.pf_config.max_particles,
            ess_low=self.pf_config.ess_low,
            ess_high=self.pf_config.ess_high,
            max_sources=self.pf_config.max_sources,
            resample_threshold=self.pf_config.resample_threshold,
            position_sigma=self.pf_config.position_sigma,
            strength_sigma=self.pf_config.strength_sigma,
            background_sigma=self.pf_config.background_sigma,
            background_level=self.pf_config.background_level,
            measurement_scale_by_isotope=self.pf_config.measurement_scale_by_isotope,
            count_likelihood_model=self.pf_config.count_likelihood_model,
            transport_model_rel_sigma=self.pf_config.transport_model_rel_sigma,
            transport_model_abs_sigma=self.pf_config.transport_model_abs_sigma,
            spectrum_count_rel_sigma=self.pf_config.spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=self.pf_config.spectrum_count_abs_sigma,
            low_count_abs_sigma=self.pf_config.low_count_abs_sigma,
            low_count_transition_counts=self.pf_config.low_count_transition_counts,
            count_likelihood_df=self.pf_config.count_likelihood_df,
            min_strength=self.pf_config.min_strength,
            p_birth=self.pf_config.p_birth,
            p_kill=self.pf_config.p_kill,
            death_low_q_streak=self.pf_config.death_low_q_streak,
            death_delta_ll_threshold=self.pf_config.death_delta_ll_threshold,
            support_ema_alpha=self.pf_config.support_ema_alpha,
            support_window=self.pf_config.support_window,
            birth_window=self.pf_config.birth_window,
            birth_softmax_temp=self.pf_config.birth_softmax_temp,
            birth_min_score=self.pf_config.birth_min_score,
            birth_enable=self.pf_config.birth_enable,
            birth_topk_particles=self.pf_config.birth_topk_particles,
            birth_use_weighted_topk=self.pf_config.birth_use_weighted_topk,
            birth_min_sep_m=self.pf_config.birth_min_sep_m,
            birth_detector_min_sep_m=self.pf_config.birth_detector_min_sep_m,
            source_detector_exclusion_m=self.pf_config.source_detector_exclusion_m,
            birth_candidate_jitter_sigma=self.pf_config.birth_candidate_jitter_sigma,
            birth_num_local_jitter=self.pf_config.birth_num_local_jitter,
            birth_alpha=self.pf_config.birth_alpha,
            birth_q_max=self.pf_config.birth_q_max,
            birth_q_min=self.pf_config.birth_q_min,
            birth_max_per_update=self.pf_config.birth_max_per_update,
            birth_delta_ll_threshold=self.pf_config.birth_delta_ll_threshold,
            birth_complexity_penalty=self.pf_config.birth_complexity_penalty,
            birth_bic_penalty_params=self.pf_config.birth_bic_penalty_params,
            structural_update_min_counts=self.pf_config.structural_update_min_counts,
            structural_update_min_snr=self.pf_config.structural_update_min_snr,
            birth_min_distinct_poses=self.pf_config.birth_min_distinct_poses,
            birth_residual_clip_quantile=self.pf_config.birth_residual_clip_quantile,
            birth_residual_gate_p_value=self.pf_config.birth_residual_gate_p_value,
            birth_residual_min_support=self.pf_config.birth_residual_min_support,
            birth_residual_support_sigma=self.pf_config.birth_residual_support_sigma,
            birth_min_distinct_stations=self.pf_config.birth_min_distinct_stations,
            birth_candidate_support_fraction=self.pf_config.birth_candidate_support_fraction,
            birth_refit_residual_gate=self.pf_config.birth_refit_residual_gate,
            birth_refit_residual_min_fraction=self.pf_config.birth_refit_residual_min_fraction,
            birth_use_shield_coded_residual=self.pf_config.birth_use_shield_coded_residual,
            birth_existing_response_corr_max=self.pf_config.birth_existing_response_corr_max,
            birth_response_condition_max=self.pf_config.birth_response_condition_max,
            birth_count_distance_prior_weight=(
                self.pf_config.birth_count_distance_prior_weight
            ),
            birth_count_distance_strength_weight=(
                self.pf_config.birth_count_distance_strength_weight
            ),
            birth_count_distance_log_clip=self.pf_config.birth_count_distance_log_clip,
            birth_count_distance_strength_sigma=(
                self.pf_config.birth_count_distance_strength_sigma
            ),
            birth_residual_always_try=self.pf_config.birth_residual_always_try,
            birth_residual_expand_structural_particles=(
                self.pf_config.birth_residual_expand_structural_particles
            ),
            birth_residual_expanded_structural_topk_particles=(
                self.pf_config.birth_residual_expanded_structural_topk_particles
            ),
            birth_residual_acceptance_complexity_scale=(
                self.pf_config.birth_residual_acceptance_complexity_scale
            ),
            birth_residual_force_proposal_on_gate=(
                self.pf_config.birth_residual_force_proposal_on_gate
            ),
            birth_residual_forced_min_delta_ll=(
                self.pf_config.birth_residual_forced_min_delta_ll
            ),
            birth_residual_force_relax_candidate_masks=(
                self.pf_config.birth_residual_force_relax_candidate_masks
            ),
            birth_residual_suppress_death=self.pf_config.birth_residual_suppress_death,
            birth_matching_pursuit_max_new_sources=(
                self.pf_config.birth_matching_pursuit_max_new_sources
            ),
            birth_matching_pursuit_topk_candidates=(
                self.pf_config.birth_matching_pursuit_topk_candidates
            ),
            birth_jitter_topk_candidates=self.pf_config.birth_jitter_topk_candidates,
            birth_global_rescue_enable=self.pf_config.birth_global_rescue_enable,
            birth_global_rescue_max_candidates=(
                self.pf_config.birth_global_rescue_max_candidates
            ),
            birth_global_rescue_min_residual_fraction=(
                self.pf_config.birth_global_rescue_min_residual_fraction
            ),
            birth_global_rescue_dedup_radius_m=(
                self.pf_config.birth_global_rescue_dedup_radius_m
            ),
            birth_global_rescue_forced_min_delta_ll=(
                self.pf_config.birth_global_rescue_forced_min_delta_ll
            ),
            birth_global_rescue_min_support=(
                self.pf_config.birth_global_rescue_min_support
            ),
            birth_global_rescue_min_distinct_poses=(
                self.pf_config.birth_global_rescue_min_distinct_poses
            ),
            birth_global_rescue_min_distinct_stations=(
                self.pf_config.birth_global_rescue_min_distinct_stations
            ),
            high_strength_split_enable=self.pf_config.high_strength_split_enable,
            high_strength_split_q_multiple=(
                self.pf_config.high_strength_split_q_multiple
            ),
            high_strength_split_offset_m=(
                self.pf_config.high_strength_split_offset_m
            ),
            high_strength_split_candidate_count=(
                self.pf_config.high_strength_split_candidate_count
            ),
            runtime_report_rescue_enable=self.pf_config.runtime_report_rescue_enable,
            runtime_report_rescue_particle_fraction=(
                self.pf_config.runtime_report_rescue_particle_fraction
            ),
            runtime_report_rescue_min_particles_per_source=(
                self.pf_config.runtime_report_rescue_min_particles_per_source
            ),
            runtime_report_rescue_weight=self.pf_config.runtime_report_rescue_weight,
            runtime_report_rescue_jitter_sigma_m=(
                self.pf_config.runtime_report_rescue_jitter_sigma_m
            ),
            residual_decomposition_enable=(
                self.pf_config.residual_decomposition_enable
            ),
            peak_suppression_enable=self.pf_config.peak_suppression_enable,
            peak_suppression_min_source_fraction=(
                self.pf_config.peak_suppression_min_source_fraction
            ),
            peak_suppression_factor=self.pf_config.peak_suppression_factor,
            residual_decomposition_max_layers=(
                self.pf_config.residual_decomposition_max_layers
            ),
            pseudo_source_verification_enable=(
                self.pf_config.pseudo_source_verification_enable
            ),
            pseudo_source_min_delta_ll=self.pf_config.pseudo_source_min_delta_ll,
            pseudo_source_min_distinct_views=(
                self.pf_config.pseudo_source_min_distinct_views
            ),
            pseudo_source_fail_grace_stations=(
                self.pf_config.pseudo_source_fail_grace_stations
            ),
            pseudo_source_corr_max=self.pf_config.pseudo_source_corr_max,
            pseudo_source_temporal_sep_min=(
                self.pf_config.pseudo_source_temporal_sep_min
            ),
            pseudo_source_quarantine_on_suppress=(
                self.pf_config.pseudo_source_quarantine_on_suppress
            ),
            pseudo_source_quarantine_excludes_runtime=(
                self.pf_config.pseudo_source_quarantine_excludes_runtime
            ),
            report_exclude_unverified_sources=(
                self.pf_config.report_exclude_unverified_sources
            ),
            source_prune_min_distinct_stations=(
                self.pf_config.source_prune_min_distinct_stations
            ),
            source_prune_min_distinct_views=(
                self.pf_config.source_prune_min_distinct_views
            ),
            source_prune_fail_grace_stations=(
                self.pf_config.source_prune_fail_grace_stations
            ),
            source_prune_delta_ll_threshold=(
                self.pf_config.source_prune_delta_ll_threshold
            ),
            source_prune_refit_after_remove=(
                self.pf_config.source_prune_refit_after_remove
            ),
            source_prune_bic_penalty_params=(
                self.pf_config.source_prune_bic_penalty_params
            ),
            refit_after_moves=self.pf_config.refit_after_moves,
            refit_iters=self.pf_config.refit_iters,
            refit_eps=self.pf_config.refit_eps,
            weak_source_prune_min_expected_count=self.pf_config.weak_source_prune_min_expected_count,
            weak_source_prune_min_fraction=self.pf_config.weak_source_prune_min_fraction,
            weak_source_prune_min_age=self.pf_config.weak_source_prune_min_age,
            weak_source_prune_require_observable=(
                self.pf_config.weak_source_prune_require_observable
            ),
            weak_source_prune_min_observable_measurements=(
                self.pf_config.weak_source_prune_min_observable_measurements
            ),
            weak_source_prune_observable_count=(
                self.pf_config.weak_source_prune_observable_count
            ),
            weak_source_prune_observable_fraction=(
                self.pf_config.weak_source_prune_observable_fraction
            ),
            weak_source_prune_visibility_reference_strength=(
                self.pf_config.weak_source_prune_visibility_reference_strength
            ),
            conditional_strength_refit=self.pf_config.conditional_strength_refit,
            conditional_strength_refit_window=self.pf_config.conditional_strength_refit_window,
            conditional_strength_refit_iters=self.pf_config.conditional_strength_refit_iters,
            conditional_strength_refit_reweight=self.pf_config.conditional_strength_refit_reweight,
            conditional_strength_refit_cardinality_neutral_reweight=(
                self.pf_config.conditional_strength_refit_cardinality_neutral_reweight
            ),
            conditional_strength_refit_reweight_clip=self.pf_config.conditional_strength_refit_reweight_clip,
            conditional_strength_refit_min_count=self.pf_config.conditional_strength_refit_min_count,
            conditional_strength_refit_min_snr=self.pf_config.conditional_strength_refit_min_snr,
            conditional_strength_refit_prior_weight=self.pf_config.conditional_strength_refit_prior_weight,
            conditional_strength_refit_prior_rel_sigma=self.pf_config.conditional_strength_refit_prior_rel_sigma,
            source_strength_prior_mean=self.pf_config.source_strength_prior_mean,
            source_strength_prior_weight=self.pf_config.source_strength_prior_weight,
            source_strength_prior_rel_sigma=(
                self.pf_config.source_strength_prior_rel_sigma
            ),
            source_strength_absorption_penalty_weight=(
                self.pf_config.source_strength_absorption_penalty_weight
            ),
            source_strength_absorption_q_multiple=(
                self.pf_config.source_strength_absorption_q_multiple
            ),
            source_strength_observation_overshoot_penalty_weight=(
                self.pf_config.source_strength_observation_overshoot_penalty_weight
            ),
            source_strength_observation_overshoot_sigma=(
                self.pf_config.source_strength_observation_overshoot_sigma
            ),
            source_strength_observation_overshoot_quantile=(
                self.pf_config.source_strength_observation_overshoot_quantile
            ),
            source_strength_observation_overshoot_min_visible_fraction=(
                self.pf_config.source_strength_observation_overshoot_min_visible_fraction
            ),
            source_strength_observation_overshoot_min_visible_measurements=(
                self.pf_config.source_strength_observation_overshoot_min_visible_measurements
            ),
            birth_stage_single_station_as_quarantine=(
                self.pf_config.birth_stage_single_station_as_quarantine
            ),
            min_age_to_split=self.pf_config.min_age_to_split,
            use_clustered_output=self.pf_config.use_clustered_output,
            cluster_eps_m=self.pf_config.cluster_eps_m,
            cluster_min_samples=self.pf_config.cluster_min_samples,
            cluster_report_max_points=self.pf_config.cluster_report_max_points,
            cluster_exact_max_points=self.pf_config.cluster_exact_max_points,
            split_prob=self.pf_config.split_prob,
            split_strength_min=self.pf_config.split_strength_min,
            split_position_sigma=self.pf_config.split_position_sigma,
            split_strength_min_frac=self.pf_config.split_strength_min_frac,
            split_strength_max_frac=self.pf_config.split_strength_max_frac,
            split_delta_ll_threshold=self.pf_config.split_delta_ll_threshold,
            split_complexity_penalty=self.pf_config.split_complexity_penalty,
            split_residual_guided=self.pf_config.split_residual_guided,
            split_residual_always_try=self.pf_config.split_residual_always_try,
            split_residual_candidate_count=self.pf_config.split_residual_candidate_count,
            merge_prob=self.pf_config.merge_prob,
            merge_distance_max=self.pf_config.merge_distance_max,
            merge_delta_ll_threshold=self.pf_config.merge_delta_ll_threshold,
            merge_response_corr_min=self.pf_config.merge_response_corr_min,
            merge_search_topk_pairs=self.pf_config.merge_search_topk_pairs,
            structural_proposal_topk_particles=(
                self.pf_config.structural_proposal_topk_particles
            ),
            structural_trial_workers=self.pf_config.structural_trial_workers,
            structural_trial_parallel_min_trials=(
                self.pf_config.structural_trial_parallel_min_trials
            ),
            target_ess_ratio=self.pf_config.target_ess_ratio,
            max_temper_steps=self.pf_config.max_temper_steps,
            min_delta_beta=self.pf_config.min_delta_beta,
            max_resamples_per_observation=self.pf_config.max_resamples_per_observation,
            temper_resample_cooldown_steps=self.pf_config.temper_resample_cooldown_steps,
            temper_resample_force_ratio=self.pf_config.temper_resample_force_ratio,
            disable_regularize_on_temper_resample=self.pf_config.disable_regularize_on_temper_resample,
            deferred_resample_roughening_scale=(
                self.pf_config.deferred_resample_roughening_scale
            ),
            cardinality_preserving_resample=self.pf_config.cardinality_preserving_resample,
            cardinality_preserving_min_stations=(
                self.pf_config.cardinality_preserving_min_stations
            ),
            cardinality_preserving_require_confirmed_structure=(
                self.pf_config.cardinality_preserving_require_confirmed_structure
            ),
            mode_preserving_resample=self.pf_config.mode_preserving_resample,
            mode_preserving_max_modes=self.pf_config.mode_preserving_max_modes,
            mode_preserving_particles_per_mode=(
                self.pf_config.mode_preserving_particles_per_mode
            ),
            mode_preserving_radius_m=self.pf_config.mode_preserving_radius_m,
            mode_preserving_min_weight_fraction=(
                self.pf_config.mode_preserving_min_weight_fraction
            ),
            mode_preserving_surface_strata=(
                self.pf_config.mode_preserving_surface_strata
            ),
            mode_preserving_height_bin_m=self.pf_config.mode_preserving_height_bin_m,
            mode_preserving_high_surface_extra_particles=(
                self.pf_config.mode_preserving_high_surface_extra_particles
            ),
            mode_preserving_high_surface_z_fraction=(
                self.pf_config.mode_preserving_high_surface_z_fraction
            ),
            mode_preserving_support_score_weight=(
                self.pf_config.mode_preserving_support_score_weight
            ),
            mode_preserving_tentative_boost=(
                self.pf_config.mode_preserving_tentative_boost
            ),
            mode_preserving_residual_boost=(
                self.pf_config.mode_preserving_residual_boost
            ),
            mode_preserving_cardinality_strata=(
                self.pf_config.mode_preserving_cardinality_strata
            ),
            mode_preserving_min_particles_per_cardinality=(
                self.pf_config.mode_preserving_min_particles_per_cardinality
            ),
            adapt_cooldown_steps=self.pf_config.adapt_cooldown_steps,
            position_min=self.pf_config.position_min,
            position_max=self.pf_config.position_max,
            source_position_prior=self.pf_config.source_position_prior,
            init_num_sources=self.pf_config.init_num_sources,
            init_grid_spacing_m=self.pf_config.init_grid_spacing_m,
            init_grid_repeats=self.pf_config.init_grid_repeats,
            roughening_k=self.pf_config.roughening_k,
            min_sigma_pos=self.pf_config.min_sigma_pos,
            max_sigma_pos=self.pf_config.max_sigma_pos,
            roughening_decay=self.pf_config.roughening_decay,
            roughening_min_mult=self.pf_config.roughening_min_mult,
            init_strength_log_mean=self.pf_config.init_strength_log_mean,
            init_strength_log_sigma=self.pf_config.init_strength_log_sigma,
            strength_log_sigma=self.pf_config.strength_log_sigma,
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
            use_tempering=self.pf_config.use_tempering,
            label_enable=self.pf_config.label_enable,
            label_alignment_iters=self.pf_config.label_alignment_iters,
            label_pos_weight=self.pf_config.label_pos_weight,
            label_strength_weight=self.pf_config.label_strength_weight,
            label_missing_cost=self.pf_config.label_missing_cost,
            label_pos_scale=self.pf_config.label_pos_scale,
            label_strength_scale=self.pf_config.label_strength_scale,
            converge_enable=self.pf_config.converge_enable,
            converge_window=self.pf_config.converge_window,
            converge_map_move_eps_m=self.pf_config.converge_map_move_eps_m,
            converge_ess_ratio_high=self.pf_config.converge_ess_ratio_high,
            converge_ll_improve_eps=self.pf_config.converge_ll_improve_eps,
            converge_min_steps=self.pf_config.converge_min_steps,
            converge_require_all=self.pf_config.converge_require_all,
            converge_cardinality_var_max=self.pf_config.converge_cardinality_var_max,
            converge_require_no_tentative=self.pf_config.converge_require_no_tentative,
            converge_freeze_updates=self.pf_config.converge_freeze_updates,
            converge_min_stations=self.pf_config.converge_min_stations,
            converge_cluster_spread_max_m=(
                self.pf_config.converge_cluster_spread_max_m
            ),
            converge_cluster_min_support_fraction=(
                self.pf_config.converge_cluster_min_support_fraction
            ),
        )

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        if not gpu_utils.torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether CUDA-backed estimator math is available."""
        from pf import gpu_utils

        return bool(self.pf_config.use_gpu and gpu_utils.torch_available())

    def response_scale_for_isotope(self, isotope: str) -> float:
        """Return the configured source response scale for one isotope."""
        scales = self.pf_config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(isotope, 1.0)), 0.0)

    def adapt_strength_prior_to_observation(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Rescale early source-strength particles using the current count observation.

        The adaptation uses only spectrum-derived counts and the same forward
        model used by the PF. For each particle, source positions and relative
        source-strength proportions are kept fixed, while the total strength is
        set to the value implied by z ~= T * sum_j q_j K_j. This creates a
        count-conditioned proposal over strength without inserting any ground
        truth source information or a scenario-specific cps value.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        return self._adapt_strength_prior_at_detector(
            z_k=z_k,
            detector_pos=np.asarray(self.poses[pose_idx], dtype=float),
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_variance_k=z_variance_k,
        )

    def _adapt_strength_prior_at_detector(
        self,
        z_k: Dict[str, float],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Apply the count-conditioned strength proposal at an explicit detector position."""
        self.last_strength_prior_diagnostics = {}
        if not bool(self.pf_config.adaptive_strength_prior):
            return {}
        max_steps = int(self.pf_config.adaptive_strength_prior_steps)
        if max_steps <= 0 or len(self.measurements) >= max_steps:
            return {}
        live_time = float(live_time_s)
        if live_time <= 0.0:
            return {}
        detector = np.asarray(detector_pos, dtype=float)
        min_counts = float(self.pf_config.adaptive_strength_prior_min_counts)
        log_sigma = float(self.pf_config.adaptive_strength_prior_log_sigma)
        max_upscale = float(self.pf_config.adaptive_strength_prior_max_upscale)
        min_strength = max(float(self.pf_config.min_strength), 0.0)
        eps = 1e-12
        diagnostics: Dict[str, Dict[str, float]] = {}
        for iso, filt in self.filters.items():
            if iso not in z_k:
                continue
            particles = list(filt.continuous_particles)
            if not particles:
                continue
            observed_counts = max(float(z_k.get(iso, 0.0)), 0.0)
            target_counts = max(observed_counts, min_counts)
            floor_only_target = observed_counts < min_counts
            obs_variance = (
                0.0
                if z_variance_k is None
                else max(float(z_variance_k.get(iso, target_counts)), 0.0)
            )
            relative_count_variance = obs_variance / max(target_counts**2, eps)
            effective_log_sigma = float(
                np.sqrt(log_sigma**2 + np.log1p(relative_count_variance))
            )
            states = [particle.state for particle in particles]
            total_strengths = np.zeros(len(states), dtype=float)
            source_counts = np.zeros(len(states), dtype=float)
            eligible = np.zeros(len(states), dtype=bool)
            for index, state in enumerate(states):
                num_sources = int(state.num_sources)
                if num_sources <= 0 or state.strengths.size < num_sources:
                    continue
                strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(np.sum(strengths))
                total_strengths[index] = total_strength
                eligible[index] = total_strength > eps
            if np.any(eligible):
                expected_counts = self.expected_counts_pair_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time,
                    states=states,
                )
                backgrounds = np.asarray(
                    [float(state.background) for state in states],
                    dtype=float,
                )
                source_counts = np.maximum(
                    np.asarray(expected_counts, dtype=float) - live_time * backgrounds,
                    0.0,
                )
            before_totals: list[float] = []
            after_totals: list[float] = []
            for index, particle in enumerate(particles):
                state = particle.state
                num_sources = int(state.num_sources)
                if num_sources <= 0 or state.strengths.size < num_sources:
                    continue
                strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(total_strengths[index])
                if total_strength <= eps:
                    continue
                proportions = strengths / total_strength
                unit_counts = float(source_counts[index]) / total_strength
                if not np.isfinite(unit_counts) or unit_counts <= eps:
                    continue
                proposed_total = target_counts / unit_counts
                if effective_log_sigma > 0.0:
                    proposed_total *= float(
                        np.random.lognormal(mean=0.0, sigma=effective_log_sigma)
                    )
                if total_strength > eps:
                    if floor_only_target:
                        proposed_total = min(proposed_total, total_strength)
                    else:
                        proposed_total = min(
                            proposed_total,
                            total_strength * max_upscale,
                        )
                proposed_total = max(float(proposed_total), min_strength * num_sources)
                if not np.isfinite(proposed_total):
                    continue
                before_totals.append(total_strength)
                after_totals.append(proposed_total)
                state.strengths[:num_sources] = proportions * proposed_total
            if after_totals:
                diagnostics[iso] = {
                    "observed_counts": float(observed_counts),
                    "target_counts": float(target_counts),
                    "observation_count_variance": float(obs_variance),
                    "effective_log_sigma": float(effective_log_sigma),
                    "floor_only_target": float(floor_only_target),
                    "max_upscale": float(max_upscale),
                    "before_median_strength": float(np.median(before_totals)),
                    "after_median_strength": float(np.median(after_totals)),
                    "particles_changed": float(len(after_totals)),
                }
        self.last_strength_prior_diagnostics = diagnostics
        return diagnostics

    def _continuous_kernel(self) -> ContinuousKernel:
        """Build a ContinuousKernel matching the estimator observation model."""
        return ContinuousKernel(
            mu_by_isotope=self.mu_by_isotope,
            shield_params=self.shield_params,
            use_gpu=bool(self.pf_config.use_gpu),
            gpu_device=str(self.pf_config.gpu_device),
            gpu_dtype=str(self.pf_config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
        )

    def expected_counts_pair_for_states(
        self,
        isotope: str,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ_{k,h}^{(n)} for an isotope over a list of states at a pose.

        Uses torch acceleration when enabled; otherwise falls back to CPU kernels.
        """
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[pose_idx], dtype=float)
        return self.expected_counts_pair_for_states_at_detector(
            isotope=isotope,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            states=states,
        )

    def expected_counts_pair_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ for a state subset at an arbitrary detector position.

        This helper keeps candidate-pose and shield-selection scoring on the
        same GPU-accelerated transport approximation as normal PF updates,
        even when a planning particle subset is used.
        """
        if not states:
            return np.zeros(0, dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            values = np.zeros(len(states), dtype=float)
            source_scale = self.response_scale_for_isotope(isotope)
            for idx, state in enumerate(states):
                rate = float(state.background)
                for pos, strength in zip(
                    state.positions[: state.num_sources],
                    state.strengths[: state.num_sources],
                ):
                    rate += (
                        source_scale
                        * float(strength)
                        * kernel.kernel_value_pair(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            source_pos=pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                        )
                    )
                values[idx] = float(live_time_s) * rate
            return values
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        mu_fe, mu_pb = kernel._mu_values(isotope=isotope)
        shield_params = kernel.shield_params
        lam_t = gpu_utils.expected_counts_pair_torch(
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            mu_fe=mu_fe,
            mu_pb=mu_pb,
            thickness_fe_cm=shield_params.thickness_fe_cm,
            thickness_pb_cm=shield_params.thickness_pb_cm,
            inner_radius_fe_cm=shield_params.inner_radius_fe_cm,
            inner_radius_pb_cm=shield_params.inner_radius_pb_cm,
            shield_geometry_model=shield_params.shield_geometry_model,
            use_angle_attenuation=shield_params.use_angle_attenuation,
            live_time_s=live_time_s,
            device=device,
            dtype=dtype,
            source_scale=self.response_scale_for_isotope(isotope),
            detector_radius_m=kernel.detector_radius_m,
            detector_aperture_samples=kernel.detector_aperture_samples,
            buildup_fe_coeff=shield_params.buildup_fe_coeff,
            buildup_pb_coeff=shield_params.buildup_pb_coeff,
            **kernel.obstacle_gpu_kwargs(isotope),
        )
        return lam_t.detach().cpu().numpy()

    def expected_counts_all_pairs_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute expected counts for all Fe/Pb orientation pairs for state subsets.

        The returned array is shaped ``(num_pairs, num_states)`` and uses the same
        continuous kernel, spherical-octant shield geometry, detector aperture,
        response scale, and obstacle attenuation as the per-pair helper.
        """
        num_pairs = int(self.num_orientations) * int(self.num_orientations)
        if not states:
            return np.zeros((num_pairs, 0), dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            rows: list[NDArray[np.float64]] = []
            for fe_index in range(int(self.num_orientations)):
                for pb_index in range(int(self.num_orientations)):
                    rows.append(
                        self.expected_counts_pair_for_states_at_detector(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                            live_time_s=live_time_s,
                            states=states,
                        )
                    )
            return np.vstack(rows).astype(float, copy=False)

        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        mu_fe, mu_pb = kernel._mu_values(isotope=isotope)
        shield_params = kernel.shield_params
        lam_t = gpu_utils.expected_counts_all_pairs_torch(
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            mu_fe=mu_fe,
            mu_pb=mu_pb,
            thickness_fe_cm=shield_params.thickness_fe_cm,
            thickness_pb_cm=shield_params.thickness_pb_cm,
            inner_radius_fe_cm=shield_params.inner_radius_fe_cm,
            inner_radius_pb_cm=shield_params.inner_radius_pb_cm,
            shield_geometry_model=shield_params.shield_geometry_model,
            use_angle_attenuation=shield_params.use_angle_attenuation,
            live_time_s=live_time_s,
            device=device,
            dtype=dtype,
            source_scale=self.response_scale_for_isotope(isotope),
            detector_radius_m=kernel.detector_radius_m,
            detector_aperture_samples=kernel.detector_aperture_samples,
            buildup_fe_coeff=shield_params.buildup_fe_coeff,
            buildup_pb_coeff=shield_params.buildup_pb_coeff,
            **kernel.obstacle_gpu_kwargs(isotope),
        )
        return lam_t.detach().cpu().numpy().astype(float, copy=False)

    def shield_selection_batch_grids(
        self,
        pose_idx: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
        include_count_quantiles: bool = True,
    ) -> tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """
        Return all-pair shield-signature and observability count grids.

        This batches the same per-pair expected-count calculation used by
        ``orientation_signature_separation_score`` and
        ``expected_observation_counts_by_isotope_at_pair``. It is a planning
        acceleration only; it does not change the observation model used for PF
        updates.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[int(pose_idx)], dtype=float)
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        floor = max(float(variance_floor), eps)
        signature_flat = np.zeros(num_pairs, dtype=float)
        count_quantiles: Dict[str, NDArray[np.float64]] = {}
        if particles_by_isotope is None:
            particles_by_isotope = self.planning_particles(
                max_particles=max_particles,
                method=self.pf_config.planning_method,
            )

        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            means = lambdas @ weights_arr
            centered = lambdas - means[:, None]
            variances = (centered * centered) @ weights_arr
            signature_flat += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * np.maximum(variances, 0.0)
                / np.maximum(means, floor)
            )
            if include_count_quantiles:
                quantile = float(self.pf_config.pose_min_observation_quantile)
                count_quantiles[iso] = np.asarray(
                    [
                        _weighted_quantile(lambdas[pair_idx], weights_arr, quantile)
                        for pair_idx in range(num_pairs)
                    ],
                    dtype=float,
                ).reshape(num_orients, num_orients)
        return (
            np.maximum(signature_flat, 0.0).reshape(num_orients, num_orients),
            count_quantiles,
        )

    def expected_observation_counts_by_isotope_at_pose(
        self,
        pose_xyz: NDArray[np.float64],
        *,
        live_time_s: float,
        fe_pb_pairs: Sequence[tuple[int, int]] | None = None,
        aggregate: str = "max",
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """
        Return posterior-mean expected counts for each isotope at a candidate pose.

        The value for one isotope is computed from the same inverse-square,
        spherical shield, and obstacle attenuation model used by PF updates.
        Across shield pairs, ``aggregate="max"`` returns the best achievable
        expected count at that pose, while ``aggregate="mean"`` returns the
        orientation-average expected count. Each pair uses a weighted posterior
        quantile rather than the posterior mean, so a few high-strength outlier
        particles cannot make the pose look observable for every isotope.
        """
        detector = np.asarray(pose_xyz, dtype=float)
        if detector.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        live_time = float(live_time_s)
        if live_time <= 0.0:
            return {iso: 0.0 for iso in self.isotopes}
        aggregate = str(aggregate).strip().lower()
        if aggregate not in {"max", "mean"}:
            raise ValueError("aggregate must be max or mean.")
        num_orients = max(1, int(self.num_orientations))
        if fe_pb_pairs is None:
            pairs = [
                (fe_index, pb_index)
                for fe_index in range(num_orients)
                for pb_index in range(num_orients)
            ]
        else:
            pairs = [(int(fe), int(pb)) for fe, pb in fe_pb_pairs]
        if not pairs:
            return {iso: 0.0 for iso in self.isotopes}
        particles = self.planning_particles(max_particles=max_particles)
        counts_by_isotope: Dict[str, float] = {}
        eps = 1e-12
        for iso in self.isotopes:
            filt = self.filters.get(iso)
            use_gpu_quantile = False
            if (
                max_particles is None
                and filt is not None
                and filt.continuous_particles
                and self.pf_config.use_gpu
            ):
                try:
                    use_gpu_quantile = bool(self._gpu_enabled())
                except RuntimeError:
                    use_gpu_quantile = False
            if use_gpu_quantile:
                weights_arr = np.asarray(filt.continuous_weights, dtype=float)
                weight_sum = float(np.sum(weights_arr))
                if weight_sum <= eps:
                    weights_arr = np.ones(len(weights_arr), dtype=float) / max(
                        len(weights_arr),
                        1,
                    )
                else:
                    weights_arr = weights_arr / weight_sum
                pair_means = []
                for fe_index, pb_index in pairs:
                    lambdas = filt._continuous_expected_counts_pair_at_pose(
                        detector_pos=detector,
                        fe_index=fe_index,
                        pb_index=pb_index,
                        live_time_s=live_time,
                    )
                    pair_means.append(
                        _weighted_quantile(
                            lambdas,
                            weights_arr,
                            self.pf_config.pose_min_observation_quantile,
                        )
                    )
                if aggregate == "mean":
                    counts_by_isotope[iso] = float(np.mean(pair_means))
                else:
                    counts_by_isotope[iso] = float(np.max(pair_means))
                continue
            if iso not in particles:
                counts_by_isotope[iso] = 0.0
                continue
            states, weights = particles[iso]
            if not states:
                counts_by_isotope[iso] = 0.0
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            pair_means: list[float] = []
            for fe_index, pb_index in pairs:
                lambdas = self.expected_counts_pair_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time,
                    states=states,
                )
                pair_means.append(
                    _weighted_quantile(
                        lambdas,
                        weights_arr,
                        self.pf_config.pose_min_observation_quantile,
                    )
                )
            if aggregate == "mean":
                counts_by_isotope[iso] = float(np.mean(pair_means))
            else:
                counts_by_isotope[iso] = float(np.max(pair_means))
        return counts_by_isotope

    def expected_observation_counts_by_isotope_at_pair(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """Return posterior-quantile expected counts for one Fe/Pb pair."""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        pose = np.asarray(self.poses[int(pose_idx)], dtype=float)
        return self.expected_observation_counts_by_isotope_at_pose(
            pose,
            live_time_s=float(live_time_s),
            fe_pb_pairs=[(int(fe_index), int(pb_index))],
            aggregate="max",
            max_particles=max_particles,
        )

    def orientation_signature_separation_score(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
    ) -> float:
        """
        Return a shield-signature separation score for one orientation pair.

        The score is a weighted posterior variance of predicted counts,
        normalized by the mean count scale. It favors shield postures whose
        response differs across currently plausible source hypotheses.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        score = 0.0
        floor = max(float(variance_floor), eps)
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lambdas = self.expected_counts_pair_for_states(
                isotope=iso,
                pose_idx=int(pose_idx),
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            mean = float(np.sum(weights_arr * lambdas))
            var = float(np.sum(weights_arr * (lambdas - mean) ** 2))
            score += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * max(var, 0.0)
                / max(mean, floor)
            )
        return float(max(score, 0.0))

    def planning_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]:
        """
        Select per-isotope particle subsets for orientation evaluation.

        Args:
            max_particles: cap on particles per isotope; None uses config default.
            method: "top_weight" or "resample"; None uses config default.
            rng: optional RNG for resampling.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if max_particles is None:
            max_particles = self.pf_config.planning_particles
        method = method or self.pf_config.planning_method
        rng = rng or np.random.default_rng()
        subsets: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] = {}
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            weights = weights / total
            n_particles = len(weights)
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= n_particles
            ):
                states = [p.state.copy() for p in filt.continuous_particles]
                subsets[iso] = (states, weights)
                continue
            if method == "top_weight":
                idx = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[idx]
                sel_weights = sel_weights / max(np.sum(sel_weights), 1e-12)
            elif method == "resample":
                idx = rng.choice(n_particles, size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max_particles
            else:
                raise ValueError(
                    f"Unknown planning particle selection method: {method}"
                )
            states = [filt.continuous_particles[i].state.copy() for i in idx]
            subsets[iso] = (states, sel_weights)
        return subsets

    def weight_entropy_ratio(
        self,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
    ) -> float:
        """
        Return the mean normalized weight entropy across isotopes.

        The entropy ratio is H(w)/log(N) in [0, 1]. Lower values indicate a more
        concentrated posterior (less multi-modality).
        """
        entropies: List[float] = []
        eps = 1e-12
        for iso, filt in self.filters.items():
            if particles_by_isotope is not None and iso in particles_by_isotope:
                _, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                weights = filt.continuous_weights
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(float(np.sum(weights)), eps)
            if weights.size == 1:
                entropies.append(0.0)
                continue
            entropy = float(-np.sum(weights * np.log(weights + eps)))
            entropies.append(entropy / max(np.log(weights.size), eps))
        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def add_measurement_pose(
        self, pose: NDArray[np.float64], reset_filters: bool = True
    ) -> None:
        """Register a new measurement pose and invalidate the kernel cache."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}
        self._invalidate_report_cache()

    def restrict_isotopes(
        self,
        active_isotopes: Sequence[str],
        *,
        allow_empty: bool = False,
    ) -> None:
        """
        Restrict estimator state to the specified isotopes.

        This drops filters and cached estimates for isotopes that are not in
        active_isotopes while preserving the original isotope ordering. When
        allow_empty is true, no isotope PFs remain active until add_isotopes()
        is called by the spectrum-detection gate.
        """
        active_set = set(active_isotopes)
        if not active_set and not allow_empty:
            raise ValueError("active_isotopes must contain at least one isotope.")
        self.isotopes = [iso for iso in self.all_isotopes if iso in active_set]
        if self.filters:
            self.filters = {
                iso: filt for iso, filt in self.filters.items() if iso in active_set
            }
        if self.history_estimates:
            self.history_estimates = [
                {iso: val for iso, val in est.items() if iso in active_set}
                for est in self.history_estimates
            ]
        self._invalidate_report_cache()

    def add_isotopes(self, new_isotopes: Sequence[str]) -> None:
        """
        Add isotopes to the estimator and initialize their PF filters.

        This is useful when new isotopes are detected after an initial restriction.
        """
        requested = set(new_isotopes)
        active_set = set(self.isotopes) | requested
        to_add = [
            iso
            for iso in self.all_isotopes
            if iso in requested and iso not in self.isotopes
        ]
        if not to_add:
            return
        self.isotopes = [iso for iso in self.all_isotopes if iso in active_set]
        if self.kernel_cache is None and self.poses:
            self._ensure_kernel_cache()
        if self.kernel_cache is None:
            return
        pf_conf = self._build_pf_config()
        for iso in to_add:
            if iso not in self.filters:
                self.filters[iso] = self._build_filter(iso, pf_conf)
        self._invalidate_report_cache()

    def update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> None:
        """
        Update per-isotope PFs using isotope-wise counts z_k.

        z_k must come from the spectrum unfolding pipeline (Sec. 2.5.7); this method
        never fabricates observations from geometric kernels or ground truth.
        """
        raise RuntimeError(
            "Single-orientation updates are disabled. Use update_pair or short_time_update "
            "with Fe/Pb indices to preserve the 64-orientation shield model."
        )

    def predict(self) -> None:
        """Run the prediction step for all PFs."""
        for f in self.filters.values():
            f.predict()
        self._invalidate_report_cache()

    def short_time_update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float | None = None,
    ) -> None:
        """
        Apply a short-time measurement update (Sec. 3.4.3).

        - Use shield orientations (RFe, RPb) and isotope-wise counts z_k.
        - T_k defaults to pf_config.short_time_s unless specified.
        - z_k must come from the spectrum pipeline (Sec. 2.5.7), not from geometry.
        """
        duration = (
            live_time_s if live_time_s is not None else self.pf_config.short_time_s
        )
        fe_index = octant_index_from_rotation(RFe)
        pb_index = octant_index_from_rotation(RPb)
        self.update_pair(
            z_k=z_k,
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=duration,
        )

    def update_pair(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
    ) -> None:
        """
        Update PFs using Fe/Pb orientation indices (RFe, RPb) and isotope-wise counts z_k.

        This feeds the continuous 3D PF path (Sec. 3.3.3) with Λ computed via expected_counts_pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if self._defer_resample_birth:
            self.last_strength_prior_diagnostics = {}
        else:
            self.adapt_strength_prior_to_observation(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_variance_k=z_variance_k,
            )
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            # Use continuous PF update that relies on spectrum-unfolded counts.
            self.filters[iso].update_continuous_pair(
                z_obs=val,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
                defer_resample=bool(self._defer_resample_birth),
            )
        self._invalidate_report_cache()
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=None
                if z_variance_k is None
                else {iso: float(v) for iso, v in z_variance_k.items()},
                ig_value=None,
            )
        )
        if self._defer_resample_birth:
            self._deferred_measurement_count += 1
        else:
            self._apply_birth_death()
        self._invalidate_report_cache()
        if not self._defer_resample_birth:
            self._record_history_estimate(len(self.measurements))

    def begin_deferred_pose_update(self) -> None:
        """Start a station-level update that delays only structural moves."""
        self._defer_resample_birth = True
        self._deferred_measurement_count = 0

    def finalize_deferred_pose_update(self) -> int:
        """
        Finish a station-level delayed update and return finalized measurements.

        During a delayed update, each shield posture updates particle weights
        immediately and may resample on ESS. This method then performs
        station-level adaptation, label alignment, and residual-gated
        birth/death once.
        """
        count = int(self._deferred_measurement_count)
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        if count <= 0:
            return 0
        pre_finalize_estimates = self.estimates(use_pre_finalize_guard=False)
        for filt in self.filters.values():
            filt.finalize_deferred_update()
        birth_context_count = count + max(
            0,
            int(self._previous_deferred_measurement_count),
        )
        self._apply_birth_death(birth_window_override=birth_context_count)
        self._invalidate_report_cache()
        post_finalize_estimates = self.estimates(use_pre_finalize_guard=False)
        self._update_pre_finalize_guard(
            pre_finalize_estimates,
            post_finalize_estimates,
        )
        self._invalidate_report_cache()
        self._previous_deferred_measurement_count = count
        self._record_history_estimate(len(self.measurements))
        return count

    def update_pair_sequence(
        self,
        records: Sequence[
            tuple[Dict[str, float], int, int, float, Dict[str, float] | None]
        ],
        *,
        pose_idx: int,
    ) -> None:
        """
        Jointly update PFs from a same-pose shield-orientation sequence.

        Each record is ``(z_k, fe_index, pb_index, live_time_s, z_variance_k)``.
        The joint update uses the product likelihood over all postures and only
        applies birth/death after the full shield program is observed.
        """
        if not records:
            return
        sequence_start = time.perf_counter()
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        for z_k, fe_index, pb_index, live_time_s, z_variance_k in records:
            self.adapt_strength_prior_to_observation(
                z_k=z_k,
                pose_idx=pose_idx,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=float(live_time_s),
                z_variance_k=z_variance_k,
            )
        step_idx = len(self.measurements)
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                int,
                int,
            ]
        ] = []
        for iso, filt in self.filters.items():
            z_arr = np.asarray(
                [float(z_k.get(iso, 0.0)) for z_k, _, _, _, _ in records],
                dtype=float,
            )
            var_arr = np.asarray(
                [
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                    for _, _, _, _, z_variance_k in records
                ],
                dtype=float,
            )
            fe_arr = np.asarray(
                [int(fe_index) for _, fe_index, _, _, _ in records],
                dtype=int,
            )
            pb_arr = np.asarray(
                [int(pb_index) for _, _, pb_index, _, _ in records],
                dtype=int,
            )
            live_arr = np.asarray(
                [float(live_time_s) for _, _, _, live_time_s, _ in records],
                dtype=float,
            )
            tasks.append(
                (
                    iso,
                    filt,
                    z_arr,
                    fe_arr,
                    pb_arr,
                    live_arr,
                    var_arr,
                    int(pose_idx),
                    int(step_idx),
                )
            )
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_pair_sequence_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_pair_sequence_update(task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(self._run_isotope_pair_sequence_update, tasks))
        self.last_pair_sequence_update_wall_s = time.perf_counter() - sequence_start
        self._invalidate_report_cache()
        for z_k, fe_index, pb_index, live_time_s, z_variance_k in records:
            self.measurements.append(
                MeasurementRecord(
                    z_k={iso: float(v) for iso, v in z_k.items()},
                    pose_idx=pose_idx,
                    orient_idx=int(fe_index),
                    live_time_s=float(live_time_s),
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                    z_variance_k=None
                    if z_variance_k is None
                    else {iso: float(v) for iso, v in z_variance_k.items()},
                    ig_value=None,
                )
            )
        self._apply_birth_death()
        self._invalidate_report_cache()
        self._record_history_estimate(len(self.measurements))

    @staticmethod
    def _run_isotope_pair_sequence_update(
        task: tuple[
            str,
            IsotopeParticleFilter,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
            int,
        ],
    ) -> None:
        """Run one isotope's same-station shield-program likelihood update."""
        (
            _isotope,
            filt,
            z_arr,
            fe_arr,
            pb_arr,
            live_arr,
            var_arr,
            pose_idx,
            step_idx,
        ) = task
        filt.update_continuous_pair_sequence(
            z_obs=z_arr,
            pose_idx=pose_idx,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_times_s=live_arr,
            observation_count_variances=var_arr,
            step_idx=step_idx,
        )

    def update_pair_at_pose(
        self,
        z_k: Dict[str, float],
        detector_pos: NDArray[np.float64],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
    ) -> None:
        """
        Update PFs using explicit detector position without rebuilding the kernel cache.

        This avoids kernel-cache growth with many poses by using per-pose updates.
        """
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(detector_pos, dtype=float)
        if not self.filters:
            pf_conf = self._build_pf_config()
            for iso in self.isotopes:
                self.filters[iso] = IsotopeParticleFilter(
                    iso,
                    kernel=None,
                    config=pf_conf,
                    obstacle_grid=self.obstacle_grid,
                    obstacle_height_m=self.obstacle_height_m,
                    obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
                    obstacle_buildup_coeff=self.obstacle_buildup_coeff,
                    detector_radius_m=self.detector_radius_m,
                    detector_aperture_samples=self.detector_aperture_samples,
                )
        self._adapt_strength_prior_at_detector(
            z_k=z_k,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_variance_k=z_variance_k,
        )
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            self.filters[iso].update_continuous_pair_at_pose(
                z_obs=val,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
            )
        self._invalidate_report_cache()
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=None
                if z_variance_k is None
                else {iso: float(v) for iso, v in z_variance_k.items()},
                ig_value=None,
            )
        )
        self._apply_birth_death()
        self._invalidate_report_cache()
        self._record_history_estimate(len(self.measurements))

    def _measurement_data_for_iso(
        self,
        isotope: str,
        window: int | None,
    ) -> MeasurementData | None:
        """Build measurement arrays for a single isotope with an optional window."""
        if not self.measurements:
            return None
        if window is None or window <= 0:
            records = self.measurements
        else:
            records = self.measurements[-int(window) :]
        if not records:
            return None
        z_list = []
        poses = []
        fe_indices = []
        pb_indices = []
        live_times = []
        variance_list = []
        for rec in records:
            z_list.append(float(rec.z_k.get(isotope, 0.0)))
            if rec.z_variance_k is None:
                variance_list.append(max(float(rec.z_k.get(isotope, 0.0)), 1.0))
            else:
                variance_list.append(
                    max(float(rec.z_variance_k.get(isotope, 1.0)), 1.0)
                )
            poses.append(self.poses[rec.pose_idx])
            live_times.append(float(rec.live_time_s))
            if rec.fe_index is not None and rec.pb_index is not None:
                fe_indices.append(int(rec.fe_index))
                pb_indices.append(int(rec.pb_index))
            else:
                fe_indices.append(int(rec.orient_idx))
                pb_indices.append(int(rec.orient_idx))
        return MeasurementData(
            z_k=np.asarray(z_list, dtype=float),
            observation_variances=np.asarray(variance_list, dtype=float),
            detector_positions=np.asarray(poses, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            live_times=np.asarray(live_times, dtype=float),
        )

    def _background_counts_for_report_refit(
        self,
        isotope: str,
        live_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return background counts used by reported-strength refitting."""
        background_rate = 0.0
        filt = self.filters.get(isotope)
        if filt is not None and filt.continuous_particles:
            background_rate = float(filt.best_particle().state.background)
        elif filt is not None:
            level = filt.config.background_level
            if isinstance(level, dict):
                background_rate = float(level.get(isotope, 0.0))
            else:
                background_rate = float(level)
        return np.maximum(background_rate, 0.0) * np.asarray(live_times, dtype=float)

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by surface-candidate diagnostics."""
        hi = np.asarray(self.pf_config.position_max, dtype=float).reshape(3)
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    @staticmethod
    def _response_design_observability_stats(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> dict[str, float | int]:
        """Return condition and correlation statistics for a response design."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] == 0:
            return {
                "candidate_count": int(design_arr.shape[1] if design_arr.ndim == 2 else 0),
                "active_candidate_count": 0,
                "weak_column_count": 0,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        weak_count = int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) <= 1:
            return {
                "candidate_count": int(design_arr.shape[1]),
                "active_candidate_count": int(np.count_nonzero(valid)),
                "weak_column_count": weak_count,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
            condition = (
                float(np.max(positive) / max(float(np.min(positive)), eps))
                if positive.size
                else float("inf")
            )
        except np.linalg.LinAlgError:
            condition = float("inf")
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        upper_values = corr[upper] if upper[0].size else np.zeros(0, dtype=float)
        max_corr = float(np.max(upper_values)) if upper_values.size else 0.0
        return {
            "candidate_count": int(design_arr.shape[1]),
            "active_candidate_count": int(np.count_nonzero(valid)),
            "weak_column_count": weak_count,
            "condition_number": condition,
            "max_abs_correlation": max_corr,
            "ambiguous_pair_count_corr_ge_0p99": int(
                np.count_nonzero(upper_values >= 0.99)
            ),
            "ambiguous_pair_count_corr_ge_0p995": int(
                np.count_nonzero(upper_values >= 0.995)
            ),
        }

    def surface_candidate_observability_diagnostics(
        self,
        *,
        window: int | None = None,
        max_candidates: int = 256,
    ) -> dict[str, dict[str, Any]]:
        """Return truth-independent observability diagnostics over surface candidates."""
        diagnostics: dict[str, dict[str, Any]] = {}
        if self.candidate_sources.size == 0:
            return diagnostics
        self._ensure_kernel_cache()
        pool_all = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        sample_count = max(1, min(int(max_candidates), int(pool_all.shape[0])))
        if pool_all.shape[0] > sample_count:
            sample_indices = np.linspace(
                0,
                pool_all.shape[0] - 1,
                sample_count,
                dtype=np.int64,
            )
            pool = pool_all[sample_indices]
        else:
            pool = pool_all
        env = self._source_prior_environment()
        surface_kinds = source_surface_kinds(
            pool,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        surface_counts = {
            str(kind): int(np.count_nonzero(surface_kinds == kind))
            for kind in ("floor", "ceiling", "wall", "obstacle_side", "obstacle_top")
        }
        surface_counts["off_surface"] = int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
        eps = max(float(self.pf_config.refit_eps), 1.0e-12)
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window)
            if data is None or data.z_k.size == 0:
                diagnostics[isotope] = {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": 0,
                    "surface_counts": surface_counts,
                }
                continue
            candidate_counts = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pool,
                strengths=np.ones(pool.shape[0], dtype=float),
            )
            variances = _measurement_vector(
                data.observation_variances,
                data.z_k.size,
                "observation_variances",
                min_value=1.0,
            )
            whitened = np.asarray(candidate_counts, dtype=float) / np.sqrt(
                variances[:, None]
            )
            stats = self._response_design_observability_stats(whitened, eps=eps)
            stats.update(
                {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": int(data.z_k.size),
                    "surface_counts": surface_counts,
                    "window": None if window is None else int(window),
                }
            )
            diagnostics[isotope] = stats
        return diagnostics

    def _report_absolute_strength_prior_terms(
        self,
        shape: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return absolute report-strength prior precision and mean arrays."""
        mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        weight = max(float(self.pf_config.source_strength_prior_weight), 0.0)
        if mean <= 0.0 or weight <= 0.0:
            zeros = np.zeros(shape, dtype=float)
            return zeros, zeros
        rel_sigma = max(float(self.pf_config.source_strength_prior_rel_sigma), 1.0e-6)
        sigma = max(rel_sigma * mean, 1.0e-12)
        precision = weight / (sigma * sigma)
        return (
            np.full(shape, precision, dtype=float),
            np.full(shape, mean, dtype=float),
        )

    def _apply_report_strength_absorption_guard(
        self,
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply a one-sided high-strength guard to report refit strengths."""
        q_arr = np.maximum(np.asarray(strengths, dtype=float), 0.0)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_absorption_penalty_weight),
        )
        mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if weight <= 0.0 or mean <= 0.0 or q_arr.size == 0:
            return q_arr
        multiple = max(
            1.0,
            float(self.pf_config.report_strength_absorption_q_multiple),
        )
        soft_cap = mean * multiple
        over = q_arr > soft_cap
        if not np.any(over):
            return q_arr
        guarded = q_arr.copy()
        guarded[over] = (guarded[over] + weight * soft_cap) / (1.0 + weight)
        return guarded

    def _report_observation_strength_bounds(
        self,
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Return data-driven per-source strength bounds from visible measurements."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        bg_arr = np.asarray(background, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_variances, dtype=float).reshape(-1)
        if (
            design_arr.ndim != 2
            or z_arr.size != design_arr.shape[0]
            or bg_arr.size != z_arr.size
            or var_arr.size != z_arr.size
        ):
            return np.full(design_arr.shape[-1], np.inf, dtype=float)
        sigma = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_sigma),
        )
        allowed = np.maximum(z_arr - bg_arr, 0.0) + sigma * np.sqrt(
            np.maximum(var_arr, 1.0)
        )
        max_response = np.max(design_arr, axis=0)
        visible_fraction = max(
            0.0,
            float(
                self.pf_config.report_strength_observation_overshoot_min_visible_fraction
            ),
        )
        visible_threshold = np.maximum(float(eps), visible_fraction * max_response)
        visible = design_arr > visible_threshold[None, :]
        min_visible = max(
            1,
            int(
                self.pf_config.report_strength_observation_overshoot_min_visible_measurements
            ),
        )
        quantile = float(
            np.clip(
                float(self.pf_config.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        ratios = np.divide(
            allowed[:, None],
            np.maximum(design_arr, float(eps)),
            out=np.full_like(design_arr, np.inf, dtype=float),
            where=visible,
        )
        visible_count = np.sum(visible, axis=0)
        ratios = np.where(visible, ratios, np.inf)
        sorted_ratios = np.sort(ratios, axis=0)
        count_for_index = np.maximum(visible_count, 1)
        positions = quantile * np.maximum(count_for_index - 1, 0)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        lower = np.take_along_axis(
            sorted_ratios,
            lower_indices[None, :],
            axis=0,
        )[0]
        upper = np.take_along_axis(
            sorted_ratios,
            upper_indices[None, :],
            axis=0,
        )[0]
        deltas = np.subtract(
            upper,
            lower,
            out=np.zeros_like(lower),
            where=np.isfinite(lower) & np.isfinite(upper),
        )
        bounds = lower + deltas * (positions - lower_indices)
        bounds = np.where(visible_count > 0, bounds, np.inf)
        return np.where(
            (visible_count >= min_visible) & np.isfinite(bounds) & (bounds >= 0.0),
            bounds,
            np.inf,
        )

    def _apply_report_observation_overshoot_guard(
        self,
        strengths: NDArray[np.float64],
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Shrink report strengths only when their response over-explains data."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_penalty_weight),
        )
        if weight <= 0.0 or q_arr.size == 0:
            return q_arr
        bounds = self._report_observation_strength_bounds(
            design=design,
            z_obs=z_obs,
            background=background,
            observation_variances=observation_variances,
            eps=eps,
        )
        if bounds.shape != q_arr.shape:
            return q_arr
        high = q_arr > bounds
        if not np.any(high):
            return q_arr
        guarded = q_arr.copy()
        guarded[high] = bounds[high] + (guarded[high] - bounds[high]) / (
            1.0 + weight
        )
        return np.where(np.isfinite(guarded), np.maximum(guarded, 0.0), q_arr)

    def _apply_report_observation_overshoot_guard_batch(
        self,
        strengths: NDArray[np.float64],
        *,
        design_batch: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        eps: float,
    ) -> NDArray[np.float64]:
        """Shrink batched report strengths that visibly over-explain observations."""
        q_arr = np.asarray(strengths, dtype=float)
        weight = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_penalty_weight),
        )
        if weight <= 0.0 or q_arr.size == 0:
            return q_arr
        design_arr = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        if design_arr.ndim != 3 or design_arr.shape[0] != q_arr.shape[0]:
            return q_arr
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        bg_arr = np.asarray(background, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_variances, dtype=float).reshape(-1)
        if (
            z_arr.size != design_arr.shape[1]
            or bg_arr.size != z_arr.size
            or var_arr.size != z_arr.size
            or design_arr.shape[2] != q_arr.shape[1]
        ):
            return q_arr
        sigma = max(
            0.0,
            float(self.pf_config.report_strength_observation_overshoot_sigma),
        )
        allowed = np.maximum(z_arr - bg_arr, 0.0) + sigma * np.sqrt(
            np.maximum(var_arr, 1.0)
        )
        max_response = np.max(design_arr, axis=1)
        visible_fraction = max(
            0.0,
            float(
                self.pf_config.report_strength_observation_overshoot_min_visible_fraction
            ),
        )
        visible_threshold = np.maximum(float(eps), visible_fraction * max_response)
        visible = design_arr > visible_threshold[:, None, :]
        ratios = np.divide(
            allowed[None, :, None],
            np.maximum(design_arr, float(eps)),
            out=np.full_like(design_arr, np.inf, dtype=float),
            where=visible,
        )
        quantile = float(
            np.clip(
                float(self.pf_config.report_strength_observation_overshoot_quantile),
                0.0,
                1.0,
            )
        )
        visible_count = np.sum(visible, axis=1)
        ratios = np.where(visible, ratios, np.inf)
        sorted_ratios = np.sort(ratios, axis=1)
        count_for_index = np.maximum(visible_count, 1)
        positions = quantile * np.maximum(count_for_index - 1, 0)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        lower = np.take_along_axis(
            sorted_ratios,
            lower_indices[:, None, :],
            axis=1,
        )[:, 0, :]
        upper = np.take_along_axis(
            sorted_ratios,
            upper_indices[:, None, :],
            axis=1,
        )[:, 0, :]
        deltas = np.subtract(
            upper,
            lower,
            out=np.zeros_like(lower),
            where=np.isfinite(lower) & np.isfinite(upper),
        )
        bounds = lower + deltas * (positions - lower_indices)
        bounds = np.where(visible_count > 0, bounds, np.inf)
        min_visible = max(
            1,
            int(
                self.pf_config.report_strength_observation_overshoot_min_visible_measurements
            ),
        )
        bounds = np.where(
            (visible_count >= min_visible) & np.isfinite(bounds) & (bounds >= 0.0),
            bounds,
            np.inf,
        )
        high = q_arr > bounds
        if not np.any(high):
            return q_arr
        guarded = q_arr.copy()
        guarded[high] = bounds[high] + (guarded[high] - bounds[high]) / (
            1.0 + weight
        )
        return np.where(np.isfinite(guarded), np.maximum(guarded, 0.0), q_arr)

    def _solve_report_strengths(
        self,
        *,
        design: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        initial_strengths: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """Return non-negative strengths for fixed reported cluster positions."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        z_arr = np.maximum(np.asarray(z_obs, dtype=float).reshape(-1), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] != z_arr.size:
            return np.asarray(initial_strengths, dtype=float).reshape(-1)
        bg_arr = _measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        source_count = int(design_arr.shape[1])
        if source_count <= 0:
            return np.zeros(0, dtype=float)
        q = np.maximum(np.asarray(initial_strengths, dtype=float).reshape(-1), eps)
        if q.size == 0:
            q = np.full(source_count, eps, dtype=float)
        elif q.size != source_count:
            raise ValueError("initial_strengths must have one value per source.")
        if not np.any(np.isfinite(q)):
            q = np.full(source_count, eps, dtype=float)
        prior_q = np.maximum(np.where(np.isfinite(q), q, eps), eps)
        if q_max > 0.0:
            prior_q = np.minimum(prior_q, q_max)
        prior_weight = max(
            0.0,
            float(self.pf_config.report_strength_refit_prior_weight),
        )
        prior_rel_sigma = max(
            1.0e-6,
            float(self.pf_config.report_strength_refit_prior_rel_sigma),
        )
        abs_precision, abs_mean = self._report_absolute_strength_prior_terms(
            prior_q.shape
        )
        local_precision = np.zeros(source_count, dtype=float)
        if prior_weight > 0.0:
            prior_sigma = np.maximum(prior_rel_sigma * prior_q, eps)
            local_precision = prior_weight / np.maximum(prior_sigma**2, eps)
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior_q + abs_precision * abs_mean,
            np.maximum(prior_precision, eps),
            out=prior_q.copy(),
            where=prior_precision > 0.0,
        )
        column_sum = np.sum(design_arr, axis=0)
        observable = column_sum > eps
        signal_total = max(float(np.sum(z_arr - bg_arr)), 0.0)
        weak_or_invalid = ~np.isfinite(q) | (q <= eps)
        if np.any(weak_or_invalid) and signal_total > 0.0:
            denom = max(float(np.sum(column_sum[observable])), eps)
            q[weak_or_invalid & observable] = signal_total / denom
        q[~observable] = 0.0
        obs_variances = _measurement_vector(
            observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        obs_weights = 1.0 / obs_variances
        gram = (design_arr.T * obs_weights[None, :]) @ design_arr
        rhs = (design_arr.T * obs_weights[None, :]) @ (z_arr - bg_arr)
        if np.any(prior_precision > 0.0):
            gram = gram + np.diag(prior_precision)
            rhs = rhs + prior_precision * prior_target
        try:
            direct = np.linalg.solve(
                gram + np.eye(source_count, dtype=float) * eps,
                rhs,
            )
            direct = np.where(np.isfinite(direct), direct, 0.0)
            if np.any(direct > 0.0):
                q = np.maximum(direct, 0.0)
                q = self._apply_report_strength_absorption_guard(q)
                q = self._apply_report_observation_overshoot_guard(
                    q,
                    design=design_arr,
                    z_obs=z_arr,
                    background=bg_arr,
                    observation_variances=obs_variances,
                    eps=eps,
                )
                q[~observable] = 0.0
        except np.linalg.LinAlgError:
            pass
        for _ in range(int(self.pf_config.report_strength_refit_iters)):
            lam = np.maximum(bg_arr + design_arr @ q, eps)
            ratio = np.divide(z_arr, lam, out=np.zeros_like(z_arr), where=lam > 0.0)
            numerator = design_arr.T @ ratio
            denominator = np.maximum(column_sum, eps)
            if np.any(prior_precision > 0.0):
                numerator = numerator + prior_precision * prior_target
                denominator = denominator + prior_precision * np.maximum(q, eps)
            q = q * np.clip(numerator / denominator, 0.0, np.inf)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard(
                q,
                design=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q[~observable] = 0.0
            if q_max > 0.0:
                q = np.minimum(q, q_max)
            q = np.where(np.isfinite(q), q, 0.0)
        return np.maximum(q, 0.0)

    def _solve_report_strengths_batch(
        self,
        *,
        design_batch: NDArray[np.float64],
        z_obs: NDArray[np.float64],
        background: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        initial_strengths: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """
        Return non-negative fixed-position strengths for a batch of subsets.

        This is a batched NumPy implementation of ``_solve_report_strengths``.
        It evaluates the same weighted least-squares initialization and the same
        multiplicative Poisson-regression iterations, but across many source
        subsets at once.
        """
        design_arr = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        z_arr = np.maximum(np.asarray(z_obs, dtype=float).reshape(-1), 0.0)
        if design_arr.ndim != 3 or design_arr.shape[1] != z_arr.size:
            return np.asarray(initial_strengths, dtype=float)
        bg_arr = _measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        batch_count, _, source_count = design_arr.shape
        if source_count <= 0:
            return np.zeros((batch_count, 0), dtype=float)
        q = np.maximum(np.asarray(initial_strengths, dtype=float), eps)
        if q.shape == (source_count,):
            q = np.broadcast_to(q[None, :], (batch_count, source_count)).copy()
        elif q.shape != (batch_count, source_count):
            raise ValueError("initial_strengths must have shape B x S.")
        q = np.where(np.isfinite(q), q, eps)
        prior_q = np.maximum(np.where(np.isfinite(q), q, eps), eps)
        if q_max > 0.0:
            prior_q = np.minimum(prior_q, q_max)
        prior_weight = max(
            0.0,
            float(self.pf_config.report_strength_refit_prior_weight),
        )
        prior_rel_sigma = max(
            1.0e-6,
            float(self.pf_config.report_strength_refit_prior_rel_sigma),
        )
        local_precision = np.zeros((batch_count, source_count), dtype=float)
        if prior_weight > 0.0:
            prior_sigma = np.maximum(prior_rel_sigma * prior_q, eps)
            local_precision = prior_weight / np.maximum(prior_sigma**2, eps)
        abs_precision, abs_mean = self._report_absolute_strength_prior_terms(
            prior_q.shape
        )
        prior_precision = local_precision + abs_precision
        prior_target = np.divide(
            local_precision * prior_q + abs_precision * abs_mean,
            np.maximum(prior_precision, eps),
            out=prior_q.copy(),
            where=prior_precision > 0.0,
        )
        column_sum = np.sum(design_arr, axis=1)
        observable = column_sum > eps
        signal_total = max(float(np.sum(z_arr - bg_arr)), 0.0)
        weak_or_invalid = ~np.isfinite(q) | (q <= eps)
        if signal_total > 0.0:
            denom = np.maximum(
                np.sum(np.where(observable, column_sum, 0.0), axis=1),
                eps,
            )
            q = np.where(
                weak_or_invalid & observable,
                signal_total / denom[:, None],
                q,
            )
        q = np.where(observable, q, 0.0)
        obs_variances = _measurement_vector(
            observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        obs_weights = 1.0 / obs_variances
        weighted_design = design_arr * obs_weights[None, :, None]
        gram = np.einsum("bmi,bmj->bij", weighted_design, design_arr)
        rhs = np.einsum("bmk,m->bk", design_arr, obs_weights * (z_arr - bg_arr))
        if np.any(prior_precision > 0.0):
            eye = np.eye(source_count, dtype=float)[None, :, :]
            gram = gram + eye * prior_precision[:, None, :]
            rhs = rhs + prior_precision * prior_target
        try:
            eye = np.eye(source_count, dtype=float)[None, :, :]
            direct = np.linalg.solve(gram + eye * eps, rhs[:, :, None])[:, :, 0]
            direct = np.where(np.isfinite(direct), direct, 0.0)
            positive_rows = np.any(direct > 0.0, axis=1)
            q[positive_rows] = np.maximum(direct[positive_rows], 0.0)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard_batch(
                q,
                design_batch=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q = np.where(observable, q, 0.0)
        except np.linalg.LinAlgError:
            pass
        for _ in range(int(self.pf_config.report_strength_refit_iters)):
            lam = np.maximum(
                bg_arr[None, :] + np.einsum("bmk,bk->bm", design_arr, q),
                eps,
            )
            ratio = np.divide(
                z_arr[None, :],
                lam,
                out=np.zeros_like(lam),
                where=lam > 0.0,
            )
            numerator = np.einsum("bmk,bm->bk", design_arr, ratio)
            denominator = np.maximum(column_sum, eps)
            if np.any(prior_precision > 0.0):
                numerator = numerator + prior_precision * prior_target
                denominator = denominator + prior_precision * np.maximum(q, eps)
            q = q * np.clip(numerator / denominator, 0.0, np.inf)
            q = self._apply_report_strength_absorption_guard(q)
            q = self._apply_report_observation_overshoot_guard_batch(
                q,
                design_batch=design_arr,
                z_obs=z_arr,
                background=bg_arr,
                observation_variances=obs_variances,
                eps=eps,
            )
            q = np.where(observable, q, 0.0)
            if q_max > 0.0:
                q = np.minimum(q, q_max)
            q = np.where(np.isfinite(q), q, 0.0)
        return np.maximum(q, 0.0)

    @staticmethod
    def _dedupe_report_candidates(
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        radius_m: float,
        max_candidates: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return report candidates after deterministic radius de-duplication."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if pos_arr.shape[0] == 0 or q_arr.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        if q_arr.size != pos_arr.shape[0]:
            raise ValueError("strengths must have one value per report candidate.")
        finite = np.all(np.isfinite(pos_arr), axis=1) & np.isfinite(q_arr)
        pos_arr = pos_arr[finite]
        q_arr = np.maximum(q_arr[finite], 0.0)
        limit = max(1, int(max_candidates))
        radius = max(float(radius_m), 0.0)
        kept_pos: list[NDArray[np.float64]] = []
        kept_q: list[float] = []
        for pos, strength in zip(pos_arr, q_arr):
            if len(kept_pos) >= limit:
                break
            if radius > 0.0 and kept_pos:
                distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
                if np.any(distances <= radius):
                    continue
            kept_pos.append(np.asarray(pos, dtype=float))
            kept_q.append(max(float(strength), 0.0))
        if not kept_pos:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return np.vstack(kept_pos), np.asarray(kept_q, dtype=float)

    def _surface_rescue_strata(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.object_]:
        """Return coarse source-surface strata used for rescue diversity quotas."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        if pos_arr.size == 0:
            return np.zeros(0, dtype=object)
        env = self._source_prior_environment()
        kinds = source_surface_kinds(
            pos_arr,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            tolerance_m=1.0e-5,
        )
        strata = np.asarray(kinds, dtype=object).copy()
        room_z = max(float(env.size_z), 1.0e-9)
        high_wall = (
            np.asarray(kinds, dtype=object) == "wall"
        ) & (pos_arr[:, 2] >= 0.5 * room_z)
        strata[high_wall] = "high_wall"
        strata[np.equal(strata, None)] = "off_surface"
        return strata

    def _surface_spatial_rescue_keys(
        self,
        positions: NDArray[np.float64],
        strata: NDArray[np.object_],
    ) -> NDArray[np.object_]:
        """Return ceiling/high-surface tile keys for spatial rescue quotas."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        strata_arr = np.asarray(strata, dtype=object).reshape(-1)
        if pos_arr.shape[0] != strata_arr.size or pos_arr.size == 0:
            return np.full(pos_arr.shape[0], None, dtype=object)
        if not bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            return np.full(pos_arr.shape[0], None, dtype=object)
        tile_m = max(float(self.pf_config.report_mle_rescue_spatial_quota_tile_m), 1e-6)
        high_strata = {"ceiling", "high_wall", "obstacle_top"}
        tile_xy = np.floor(pos_arr[:, :2] / tile_m).astype(np.int64)
        keys = np.full(pos_arr.shape[0], None, dtype=object)
        for idx, stratum in enumerate(strata_arr):
            key = str(stratum)
            if key not in high_strata:
                continue
            keys[idx] = f"{key}:x{int(tile_xy[idx, 0])}:y{int(tile_xy[idx, 1])}"
        return keys

    def _surface_stratified_rescue_indices(
        self,
        pool: NDArray[np.float64],
        scores: NDArray[np.float64],
        valid: NDArray[np.bool_],
        *,
        max_candidates: int,
        strata: NDArray[np.object_] | None = None,
        spatial_keys: NDArray[np.object_] | None = None,
    ) -> NDArray[np.int64]:
        """Return top candidate indices while reserving slots for surface strata."""
        limit = max(0, int(max_candidates))
        if limit <= 0:
            return np.zeros(0, dtype=np.int64)
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        if score_arr.size != valid_arr.size or score_arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        valid_indices = np.flatnonzero(valid_arr & np.isfinite(score_arr))
        if valid_indices.size == 0:
            return np.zeros(0, dtype=np.int64)
        order = valid_indices[np.argsort(score_arr[valid_indices])[::-1]]
        if not bool(self.pf_config.report_mle_rescue_surface_quota_enable):
            return order[:limit].astype(np.int64, copy=False)
        selected: list[int] = []
        best_score = max(float(score_arr[order[0]]), 1.0e-300)
        min_fraction = max(
            0.0,
            float(self.pf_config.report_mle_rescue_surface_quota_min_score_fraction),
        )
        per_stratum = max(
            1,
            int(getattr(self.pf_config, "report_mle_rescue_surface_quota_per_stratum", 1)),
        )
        pool_arr = np.asarray(pool, dtype=float).reshape(-1, 3)
        strata_arr = (
            self._surface_rescue_strata(pool_arr)
            if strata is None
            else np.asarray(strata, dtype=object).reshape(-1)
        )
        if strata_arr.size != score_arr.size:
            return order[:limit].astype(np.int64, copy=False)

        def _append(index: int) -> bool:
            """Append an index if it is still eligible."""
            if len(selected) >= limit or int(index) in selected:
                return False
            if score_arr[int(index)] < best_score * min_fraction:
                return False
            selected.append(int(index))
            return True

        _append(int(order[0]))
        for stratum in (
            "ceiling",
            "high_wall",
            "obstacle_top",
            "obstacle_side",
            "wall",
            "floor",
            "off_surface",
        ):
            if len(selected) >= limit:
                break
            matches = order[np.asarray(strata_arr[order], dtype=object) == stratum]
            if matches.size:
                for match in matches[:per_stratum]:
                    _append(int(match))
                    if len(selected) >= limit:
                        break
        spatial_arr = (
            self._surface_spatial_rescue_keys(pool_arr, strata_arr)
            if spatial_keys is None
            else np.asarray(spatial_keys, dtype=object).reshape(-1)
        )
        if bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            per_tile = max(
                1,
                int(self.pf_config.report_mle_rescue_spatial_quota_per_tile),
            )
            key_counts: dict[str, int] = {}
            for selected_idx in selected:
                key = spatial_arr[int(selected_idx)]
                if key is None:
                    continue
                key_text = str(key)
                key_counts[key_text] = key_counts.get(key_text, 0) + 1
            for index in order:
                if len(selected) >= limit:
                    break
                key = spatial_arr[int(index)]
                if key is None:
                    continue
                key_text = str(key)
                if key_counts.get(key_text, 0) >= per_tile:
                    continue
                if _append(int(index)):
                    key_counts[key_text] = key_counts.get(key_text, 0) + 1
        for index in order:
            if len(selected) >= limit:
                break
            _append(int(index))
        return np.asarray(selected, dtype=np.int64)

    def _rescue_visibility_reference_strength(self) -> float:
        """Return the configured absolute rescue visibility reference, if any."""
        configured = max(
            0.0,
            float(self.pf_config.report_mle_rescue_visibility_reference_strength),
        )
        if configured > 0.0:
            return configured
        weak_reference = max(
            0.0,
            float(self.pf_config.weak_source_prune_visibility_reference_strength),
        )
        if weak_reference > 0.0:
            return weak_reference
        prior_mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if prior_mean > 0.0:
            return prior_mean
        return 0.0

    def _visibility_adjusted_rescue_scores(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual: NDArray[np.float64],
        weights: NDArray[np.float64],
        base_scores: NDArray[np.float64],
        eps: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], dict[str, int | float]]:
        """Blend all-history and visible-window scores for rescue candidates."""
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        score_arr = np.maximum(np.asarray(base_scores, dtype=float).reshape(-1), 0.0)
        if counts.ndim != 2 or counts.shape[1] != score_arr.size:
            return score_arr, np.ones(score_arr.size, dtype=bool), {}
        residual_arr = np.maximum(np.asarray(residual, dtype=float).reshape(-1), 0.0)
        weight_arr = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
        if residual_arr.size != counts.shape[0] or weight_arr.size != counts.shape[0]:
            return score_arr, np.ones(score_arr.size, dtype=bool), {}
        visible_count_floor = max(
            float(self.pf_config.report_mle_rescue_visible_count),
            float(self.pf_config.weak_source_prune_observable_count),
            float(self.pf_config.weak_source_prune_min_expected_count),
            0.0,
        )
        all_weights = weight_arr[:, None]
        all_numerator = np.sum(all_weights * residual_arr[:, None] * counts, axis=0)
        all_denominator = np.sum(all_weights * counts * counts, axis=0)
        all_q_hat = np.divide(
            all_numerator,
            np.maximum(all_denominator, eps),
            out=np.zeros_like(all_numerator, dtype=float),
            where=all_denominator > eps,
        )
        ref_strength = self._rescue_visibility_reference_strength()
        if ref_strength > 0.0:
            reference_counts = counts * ref_strength
            reference_mode = "configured"
        else:
            reference_counts = counts * np.maximum(all_q_hat[None, :], 0.0)
            reference_mode = "data_driven_qhat"
        visible = (
            reference_counts >= visible_count_floor
            if visible_count_floor > 0.0
            else reference_counts > 0.0
        )
        visible_counts = np.count_nonzero(visible, axis=0)
        min_visible = max(
            1,
            int(self.pf_config.report_mle_rescue_min_visible_measurements),
        )
        valid_visible = visible_counts >= min_visible
        blend = float(self.pf_config.report_mle_rescue_visibility_weight)
        if blend <= 0.0:
            return score_arr, valid_visible, {
                "rescue_visibility_min_visible": int(min_visible),
                "rescue_visibility_valid_candidates": int(
                    np.count_nonzero(valid_visible)
                ),
                "rescue_visibility_reference_mode": reference_mode,
                "rescue_visibility_reference_strength": float(ref_strength),
                "rescue_visibility_qhat_median": float(
                    np.median(all_q_hat[all_q_hat > 0.0])
                    if np.any(all_q_hat > 0.0)
                    else 0.0
                ),
            }
        visible_weights = weight_arr[:, None] * visible
        numerator = np.sum(
            visible_weights * residual_arr[:, None] * counts,
            axis=0,
        )
        denominator = np.sum(visible_weights * counts * counts, axis=0)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, eps),
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > eps,
        )
        visible_scores = np.maximum(numerator * np.maximum(q_hat, 0.0), 0.0)
        adjusted = (1.0 - blend) * score_arr + blend * visible_scores
        return adjusted, valid_visible, {
            "rescue_visibility_weight": float(blend),
            "rescue_visibility_min_visible": int(min_visible),
            "rescue_visibility_valid_candidates": int(np.count_nonzero(valid_visible)),
            "rescue_visibility_reference_mode": reference_mode,
            "rescue_visibility_reference_strength": float(ref_strength),
            "rescue_visibility_qhat_median": float(
                np.median(all_q_hat[all_q_hat > 0.0])
                if np.any(all_q_hat > 0.0)
                else 0.0
            ),
        }

    def _next_surface_stratified_rescue_index(
        self,
        pool: NDArray[np.float64],
        scores: NDArray[np.float64],
        valid: NDArray[np.bool_],
        selected_indices: Sequence[int],
        strata: NDArray[np.object_] | None = None,
        spatial_keys: NDArray[np.object_] | None = None,
    ) -> int | None:
        """Return the next greedy rescue index with surface-stratum diversity."""
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        valid_indices = np.flatnonzero(valid_arr & np.isfinite(score_arr))
        if valid_indices.size == 0:
            return None
        order = valid_indices[np.argsort(score_arr[valid_indices])[::-1]]
        if not bool(self.pf_config.report_mle_rescue_surface_quota_enable):
            return int(order[0])
        pool_arr = np.asarray(pool, dtype=float).reshape(-1, 3)
        strata_arr = (
            self._surface_rescue_strata(pool_arr)
            if strata is None
            else np.asarray(strata, dtype=object).reshape(-1)
        )
        if strata_arr.size != score_arr.size:
            return int(order[0])
        selected = np.asarray(list(selected_indices), dtype=int)
        selected_strata_counts: dict[str, int] = {}
        if selected.size:
            for value in strata_arr[selected]:
                key = str(value)
                selected_strata_counts[key] = selected_strata_counts.get(key, 0) + 1
        best_score = max(float(score_arr[order[0]]), 1.0e-300)
        min_fraction = max(
            0.0,
            float(self.pf_config.report_mle_rescue_surface_quota_min_score_fraction),
        )
        per_stratum = max(
            1,
            int(getattr(self.pf_config, "report_mle_rescue_surface_quota_per_stratum", 1)),
        )
        for stratum in (
            "ceiling",
            "high_wall",
            "obstacle_top",
            "obstacle_side",
            "wall",
            "floor",
            "off_surface",
        ):
            if selected_strata_counts.get(stratum, 0) >= per_stratum:
                continue
            matches = order[np.asarray(strata_arr[order], dtype=object) == stratum]
            if matches.size and score_arr[int(matches[0])] >= best_score * min_fraction:
                return int(matches[0])
        spatial_arr = (
            self._surface_spatial_rescue_keys(pool_arr, strata_arr)
            if spatial_keys is None
            else np.asarray(spatial_keys, dtype=object).reshape(-1)
        )
        if bool(self.pf_config.report_mle_rescue_spatial_quota_enable):
            per_tile = max(
                1,
                int(self.pf_config.report_mle_rescue_spatial_quota_per_tile),
            )
            selected_key_counts: dict[str, int] = {}
            if selected.size:
                for key in spatial_arr[selected]:
                    if key is None:
                        continue
                    key_text = str(key)
                    selected_key_counts[key_text] = (
                        selected_key_counts.get(key_text, 0) + 1
                    )
            for index in order:
                key = spatial_arr[int(index)]
                if key is None:
                    continue
                key_text = str(key)
                if selected_key_counts.get(key_text, 0) >= per_tile:
                    continue
                if score_arr[int(index)] >= best_score * min_fraction:
                    return int(index)
        return int(order[0])

    def _surface_stratum_count_payload(
        self,
        positions: NDArray[np.float64],
    ) -> dict[str, int]:
        """Return JSON-safe counts for rescue-selected surface strata."""
        strata = self._surface_rescue_strata(positions)
        payload: dict[str, int] = {}
        for value in strata:
            key = str(value)
            payload[key] = int(payload.get(key, 0) + 1)
        return payload

    def _rank_residual_surface_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        residual: NDArray[np.float64],
        existing_positions: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        max_candidates: int | None = None,
        min_residual_fraction: float | None = None,
        dedup_radius_m: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Return residual-ranked surface candidates for MLE-style report rescue."""
        max_candidates = max(
            0,
            int(
                self.pf_config.report_mle_rescue_max_residual_candidates
                if max_candidates is None
                else max_candidates
            ),
        )
        if max_candidates <= 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        residual_arr = np.maximum(np.asarray(residual, dtype=float).reshape(-1), 0.0)
        if residual_arr.size != data.z_k.size:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        residual_sum = float(np.sum(residual_arr))
        reference_sum = max(
            float(np.sum(np.maximum(np.asarray(data.z_k, dtype=float), 0.0))),
            float(np.sum(np.maximum(np.asarray(background, dtype=float), 0.0))),
            eps,
        )
        min_fraction = max(
            0.0,
            float(
                self.pf_config.report_mle_rescue_min_residual_fraction
                if min_residual_fraction is None
                else min_residual_fraction
            ),
        )
        residual_fraction = residual_sum / reference_sum
        if residual_fraction < min_fraction:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "residual_sum": residual_sum,
                    "residual_reference_sum": reference_sum,
                    "residual_fraction": residual_fraction,
                    "residual_candidates_skipped": True,
                },
            )
        full_pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        available = np.ones(full_pool.shape[0], dtype=bool)
        if existing_positions.size:
            existing = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            distances = np.linalg.norm(
                full_pool[:, None, :] - existing[None, :, :],
                axis=2,
            )
            dedup_radius = max(
                float(
                    self.pf_config.report_mle_rescue_dedup_radius_m
                    if dedup_radius_m is None
                    else dedup_radius_m
                ),
                0.0,
            )
            if dedup_radius > 0.0:
                available &= np.min(distances, axis=1) > dedup_radius
        if not np.any(available):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        pool = full_pool[available]
        candidate_counts_full = self._cached_candidate_grid_counts(
            filt=filt,
            isotope=isotope,
            data=data,
        )
        if (
            candidate_counts_full.ndim != 2
            or candidate_counts_full.shape[1] != full_pool.shape[0]
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        candidate_counts = candidate_counts_full[:, available]
        candidate_counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        variances = _measurement_vector(
            data.observation_variances,
            data.z_k.size,
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / variances
        weighted_residual = weights * residual_arr
        numerator = weighted_residual @ candidate_counts
        denominator = weights @ (candidate_counts * candidate_counts)
        q_hat = np.divide(
            numerator,
            np.maximum(denominator, eps),
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > eps,
        )
        q_hat = np.maximum(np.where(np.isfinite(q_hat), q_hat, 0.0), 0.0)
        scores = numerator * q_hat
        scores, visibility_valid, visibility_stats = (
            self._visibility_adjusted_rescue_scores(
                candidate_counts=candidate_counts,
                residual=residual_arr,
                weights=weights,
                base_scores=scores,
                eps=eps,
            )
        )
        valid = (
            np.isfinite(scores)
            & (scores > 0.0)
            & (q_hat > 0.0)
            & visibility_valid
        )
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        selected = self._surface_stratified_rescue_indices(
            pool,
            scores,
            valid,
            max_candidates=max_candidates,
            strata=self._surface_rescue_strata(pool),
        )
        stats = {
            "residual_sum": residual_sum,
            "residual_reference_sum": reference_sum,
            "residual_fraction": residual_fraction,
            "residual_candidate_pool": int(pool.shape[0]),
            "residual_candidate_count": int(selected.size),
            "residual_candidate_best_score": (
                float(scores[selected[0]]) if selected.size else 0.0
            ),
            "residual_candidate_surface_counts": (
                self._surface_stratum_count_payload(pool[selected])
                if selected.size
                else {}
            ),
        }
        stats.update(visibility_stats)
        return pool[selected], q_hat[selected], stats

    def _rank_global_surface_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        existing_positions: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
        max_candidates: int | None = None,
        min_residual_fraction: float | None = None,
        dedup_radius_m: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Return greedy surface candidates from all observations for report rescue."""
        max_candidates_value = max(
            0,
            int(
                self.pf_config.report_mle_rescue_max_residual_candidates
                if max_candidates is None
                else max_candidates
            ),
        )
        if max_candidates_value <= 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        background_arr = _measurement_vector(
            background,
            z_obs.size,
            "background",
            min_value=0.0,
        )
        initial_residual = np.maximum(z_obs - background_arr, 0.0)
        reference_sum = max(float(np.sum(z_obs)), float(np.sum(background_arr)), eps)
        min_fraction = max(
            0.0,
            float(
                self.pf_config.report_mle_rescue_min_residual_fraction
                if min_residual_fraction is None
                else min_residual_fraction
            ),
        )
        initial_fraction = float(np.sum(initial_residual)) / reference_sum
        if initial_fraction < min_fraction:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "global_rescue_initial_residual_fraction": initial_fraction,
                    "global_rescue_candidates_skipped": True,
                },
            )
        pool = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        dedup_radius = max(
            float(
                self.pf_config.report_mle_rescue_dedup_radius_m
                if dedup_radius_m is None
                else dedup_radius_m
            ),
            0.0,
        )
        unavailable = np.zeros(pool.shape[0], dtype=bool)
        if existing_positions.size and dedup_radius > 0.0:
            existing = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            distances = np.linalg.norm(pool[:, None, :] - existing[None, :, :], axis=2)
            unavailable |= np.min(distances, axis=1) <= dedup_radius
        candidate_counts = self._cached_candidate_grid_counts(
            filt=filt,
            isotope=isotope,
            data=data,
        )
        candidate_counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if candidate_counts.ndim != 2 or candidate_counts.shape[1] != pool.shape[0]:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), {}
        variances = _measurement_vector(
            data.observation_variances,
            z_obs.size,
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / variances
        denominator = weights @ (candidate_counts * candidate_counts)
        existing_count = 0
        existing_residual_fraction = initial_fraction
        if existing_positions.size:
            existing_arr = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
            existing_count = int(existing_arr.shape[0])
            existing_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=existing_arr,
                strengths=np.ones(existing_arr.shape[0], dtype=float),
            )
            existing_design = np.maximum(
                np.asarray(existing_design, dtype=float),
                0.0,
            )
            if (
                existing_design.ndim == 2
                and existing_design.shape[0] == z_obs.size
                and existing_design.shape[1] == existing_arr.shape[0]
            ):
                existing_q = self._solve_report_strengths(
                    design=existing_design,
                    z_obs=z_obs,
                    background=background_arr,
                    observation_variances=variances,
                    initial_strengths=np.full(
                        existing_arr.shape[0],
                        max(float(self.pf_config.birth_q_min), 1.0),
                        dtype=float,
                    ),
                    eps=eps,
                    q_max=q_max,
                )
                initial_residual = np.maximum(
                    z_obs - (background_arr + existing_design @ existing_q),
                    0.0,
                )
                existing_residual_fraction = (
                    float(np.sum(initial_residual)) / reference_sum
                )
        strata = self._surface_rescue_strata(pool)
        spatial_keys = self._surface_spatial_rescue_keys(pool, strata)
        selected_indices: list[int] = []
        selected_q: list[float] = []
        selected_design = np.zeros((z_obs.size, 0), dtype=float)
        residual = initial_residual
        best_score = 0.0
        final_fraction = initial_fraction
        visibility_stats: dict[str, int | float] = {}
        for _ in range(max_candidates_value):
            residual_sum = float(np.sum(residual))
            final_fraction = residual_sum / reference_sum
            if final_fraction < min_fraction:
                break
            numerator = (weights * residual) @ candidate_counts
            q_hat = np.divide(
                numerator,
                np.maximum(denominator, eps),
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > eps,
            )
            q_hat = np.maximum(np.where(np.isfinite(q_hat), q_hat, 0.0), 0.0)
            if q_max > 0.0:
                q_hat = np.minimum(q_hat, q_max)
            scores = numerator * q_hat
            scores, visibility_valid, visibility_stats = (
                self._visibility_adjusted_rescue_scores(
                    candidate_counts=candidate_counts,
                    residual=residual,
                    weights=weights,
                    base_scores=scores,
                    eps=eps,
                )
            )
            valid = (
                np.isfinite(scores)
                & (scores > 0.0)
                & (q_hat > eps)
                & (~unavailable)
                & visibility_valid
            )
            if not np.any(valid):
                break
            next_idx = self._next_surface_stratified_rescue_index(
                pool,
                scores,
                valid,
                selected_indices,
                strata=strata,
                spatial_keys=spatial_keys,
            )
            if next_idx is None:
                break
            best_idx = int(next_idx)
            best_score = max(best_score, float(scores[best_idx]))
            selected_indices.append(best_idx)
            selected_q.append(float(q_hat[best_idx]))
            selected_design = candidate_counts[:, selected_indices]
            selected_q_arr = self._solve_report_strengths(
                design=selected_design,
                z_obs=z_obs,
                background=background_arr,
                observation_variances=variances,
                initial_strengths=np.asarray(selected_q, dtype=float),
                eps=eps,
                q_max=q_max,
            )
            selected_q = [float(value) for value in selected_q_arr]
            residual = np.maximum(
                z_obs - (background_arr + selected_design @ selected_q_arr),
                0.0,
            )
            if dedup_radius > 0.0:
                distances = np.linalg.norm(
                    pool - pool[best_idx][None, :],
                    axis=1,
                )
                unavailable |= distances <= dedup_radius
            else:
                unavailable[best_idx] = True
        if not selected_indices:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                {
                    "global_rescue_initial_residual_fraction": initial_fraction,
                    "global_rescue_candidate_pool": int(pool.shape[0]),
                    "global_rescue_candidate_count": 0,
                },
            )
        final_fraction = float(np.sum(residual)) / reference_sum
        selected = np.asarray(selected_indices, dtype=int)
        final_q = np.asarray(selected_q, dtype=float)
        stats = {
            "global_rescue_initial_residual_fraction": initial_fraction,
            "global_rescue_existing_source_count": int(existing_count),
            "global_rescue_existing_residual_fraction": float(
                existing_residual_fraction
            ),
            "global_rescue_final_residual_fraction": float(final_fraction),
            "global_rescue_candidate_pool": int(pool.shape[0]),
            "global_rescue_candidate_count": int(selected.size),
            "global_rescue_best_score": float(best_score),
            "global_rescue_surface_counts": self._surface_stratum_count_payload(
                pool[selected]
            ),
        }
        stats.update(visibility_stats)
        return pool[selected], final_q, stats

    def _augment_report_candidates_with_mle_rescue(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """Add MLE-style global and residual surface candidates for report BIC."""
        base_pos = np.asarray(positions, dtype=float).reshape(-1, 3)
        base_q = np.asarray(strengths, dtype=float).reshape(-1)
        max_total = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                int(self.pf_config.report_cluster_model_selection_max_candidates),
            ),
        )
        dedup_radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        all_pos = [base_pos] if base_pos.size else []
        all_q = [np.maximum(base_q, eps)] if base_q.size else []
        global_pos, global_q, global_stats = self._rank_global_surface_candidates(
            isotope,
            filt,
            data,
            existing_positions=base_pos,
            background=background,
            eps=eps,
            q_max=q_max,
        )
        if global_pos.size:
            all_pos.append(global_pos)
            all_q.append(np.maximum(global_q, eps))
        posterior_count = 0
        max_posterior = max(
            0,
            int(self.pf_config.report_mle_rescue_max_posterior_candidates),
        )
        if max_posterior > 0 and hasattr(filt, "estimate_clustered"):
            try:
                post_pos, post_q = filt.estimate_clustered(
                    max_k=max_posterior,
                    include_report_excluded=True,
                )
            except RuntimeError:
                post_pos = np.zeros((0, 3), dtype=float)
                post_q = np.zeros(0, dtype=float)
            if post_pos.size and post_q.size:
                posterior_count = int(post_pos.shape[0])
                all_pos.append(np.asarray(post_pos, dtype=float).reshape(-1, 3))
                all_q.append(np.maximum(np.asarray(post_q, dtype=float).reshape(-1), eps))
        if all_pos:
            seed_pos = np.vstack(all_pos)
            seed_q = np.concatenate(all_q)
        else:
            seed_pos = np.zeros((0, 3), dtype=float)
            seed_q = np.zeros(0, dtype=float)
        seed_pos, seed_q = self._dedupe_report_candidates(
            seed_pos,
            seed_q,
            radius_m=dedup_radius,
            max_candidates=max_total,
        )
        residual_stats: dict[str, Any] = {}
        residual_pos = np.zeros((0, 3), dtype=float)
        residual_q = np.zeros(0, dtype=float)
        if seed_pos.size and seed_pos.shape[0] < max_total:
            seed_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=seed_pos,
                strengths=np.ones(seed_pos.shape[0], dtype=float),
            )
            seed_q_fit = self._solve_report_strengths(
                design=seed_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=seed_q,
                eps=eps,
                q_max=q_max,
            )
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - (
                    np.asarray(background, dtype=float).reshape(-1)
                    + np.asarray(seed_design, dtype=float) @ seed_q_fit
                ),
                0.0,
            )
            residual_pos, residual_q, residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=seed_pos,
                    background=background,
                    eps=eps,
                )
            )
        elif seed_pos.shape[0] == 0:
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - np.asarray(background, dtype=float).reshape(-1),
                0.0,
            )
            residual_pos, residual_q, residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=seed_pos,
                    background=background,
                    eps=eps,
                )
            )
        if residual_pos.size:
            merged_pos = (
                np.vstack([seed_pos, residual_pos]) if seed_pos.size else residual_pos
            )
            merged_q = np.concatenate([seed_q, residual_q]) if seed_q.size else residual_q
        else:
            merged_pos = seed_pos
            merged_q = seed_q
        final_pos, final_q = self._dedupe_report_candidates(
            merged_pos,
            merged_q,
            radius_m=dedup_radius,
            max_candidates=max_total,
        )
        stats = {
            "mle_rescue_enabled": True,
            "mle_rescue_base_candidates": int(base_pos.shape[0]),
            "mle_rescue_global_candidates": int(global_pos.shape[0]),
            "mle_rescue_posterior_candidates": int(posterior_count),
            "mle_rescue_residual_candidates": int(residual_pos.shape[0]),
            "mle_rescue_candidate_limit": int(max_total),
            "mle_rescue_candidate_count": int(final_pos.shape[0]),
        }
        stats.update(global_stats)
        stats.update(residual_stats)
        return final_pos, final_q, stats

    def _report_cluster_log_likelihood(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
    ) -> float:
        """Return the report-level count log-likelihood for fixed clusters."""
        lam = np.asarray(background, dtype=float).reshape(-1) + np.asarray(
            design,
            dtype=float,
        ) @ np.asarray(strengths, dtype=float).reshape(-1)
        return float(
            filt._count_log_likelihood_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        )

    @staticmethod
    def _report_design_condition_number(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> float:
        """Return a scale-normalized condition number for reported sources."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[1] <= 1:
            return 1.0
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        if np.count_nonzero(valid) <= 1:
            return float("inf")
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
        except np.linalg.LinAlgError:
            return float("inf")
        positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
        if positive.size <= 0:
            return float("inf")
        return float(np.max(positive) / max(float(np.min(positive)), eps))

    @staticmethod
    def _report_design_condition_numbers_batch(
        design_batch: NDArray[np.float64],
        *,
        eps: float,
    ) -> NDArray[np.float64]:
        """Return condition numbers for a stack of response designs."""
        designs = np.maximum(np.asarray(design_batch, dtype=float), 0.0)
        if designs.ndim != 3:
            return np.zeros(0, dtype=float)
        batch_count = int(designs.shape[0])
        source_count = int(designs.shape[2])
        if source_count <= 1:
            return np.ones(batch_count, dtype=float)
        floor = max(float(eps), 1.0e-12)
        column_norm = np.linalg.norm(designs, axis=1)
        valid = column_norm > floor
        valid_count = np.count_nonzero(valid, axis=1)
        normalized = np.divide(
            designs,
            np.maximum(column_norm[:, None, :], floor),
            out=np.zeros_like(designs, dtype=float),
            where=valid[:, None, :],
        )
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.full(batch_count, float("inf"), dtype=float)
        positive = singular_values > floor
        max_positive = np.max(
            np.where(positive, singular_values, 0.0),
            axis=1,
        )
        min_positive = np.min(
            np.where(positive, singular_values, float("inf")),
            axis=1,
        )
        condition = np.divide(
            max_positive,
            np.maximum(min_positive, float(eps)),
            out=np.full(batch_count, float("inf"), dtype=float),
            where=np.isfinite(min_positive) & (min_positive > 0.0),
        )
        condition[valid_count <= 1] = float("inf")
        return condition

    @staticmethod
    def _report_design_max_abs_correlation(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> float:
        """Return the maximum normalized response-column correlation."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[1] <= 1:
            return 0.0
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        if np.count_nonzero(valid) <= 1:
            return 0.0
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        if upper[0].size == 0:
            return 0.0
        return float(np.max(corr[upper]))

    def _select_report_clusters_by_model_order(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """
        Select reported clusters by exhaustive BIC model-order comparison.

        The comparison is over all subsets of the currently reported candidate
        clusters up to a configurable cap.  It is not tied to any particular
        source count: K=0,1,...,N are all evaluated, and the selected K is the
        one with the highest refit Poisson log-likelihood minus the standard
        half-BIC parameter penalty.
        """
        design_arr = np.asarray(design, dtype=float)
        q_initial = np.asarray(strengths, dtype=float).reshape(-1)
        source_count = int(q_initial.size)
        q_full = np.zeros(source_count, dtype=float)
        keep = np.zeros(source_count, dtype=bool)
        if source_count <= 0:
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": 0,
                "selected_count": 0,
                "method": "empty",
            }
            return keep, q_full
        if not bool(self.pf_config.report_cluster_model_selection):
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": source_count,
                "method": "disabled",
            }
            return np.ones(source_count, dtype=bool), q_initial
        max_exhaustive = max(
            1,
            int(self.pf_config.report_cluster_model_selection_max_candidates),
        )
        configured_max_sources = self.pf_config.max_sources
        max_model_sources = source_count
        if configured_max_sources is not None:
            max_model_sources = min(
                source_count,
                max(0, int(configured_max_sources)),
            )
        if max_model_sources <= 0:
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": 0,
                "method": "source_cap_zero",
                "max_model_sources": int(max_model_sources),
            }
            return keep, q_full
        if source_count > max_exhaustive:
            keep_greedy, q_greedy = self._select_report_clusters_by_refit_after_remove(
                filt,
                data,
                design=design_arr,
                strengths=q_initial,
                background=background,
                eps=eps,
                q_max=q_max,
            )
            selected_count = int(np.count_nonzero(keep_greedy))
            if selected_count > max_model_sources:
                selected = np.flatnonzero(keep_greedy)
                q_selected = np.asarray(q_greedy, dtype=float).reshape(-1)[selected]
                order = np.argsort(np.maximum(q_selected, 0.0))[::-1]
                keep_limited = np.zeros_like(keep_greedy, dtype=bool)
                keep_limited[selected[order[:max_model_sources]]] = True
                q_limited = np.zeros_like(q_greedy, dtype=float)
                q_limited[keep_limited] = q_greedy[keep_limited]
                keep_greedy = keep_limited
                q_greedy = q_limited
                selected_count = int(np.count_nonzero(keep_greedy))
            self._last_report_model_order_diagnostics[isotope] = {
                "candidate_count": source_count,
                "selected_count": selected_count,
                "method": "greedy_refit_after_remove",
                "max_exhaustive_candidates": max_exhaustive,
                "max_model_sources": int(max_model_sources),
            }
            return keep_greedy, q_greedy

        measurement_count = int(data.z_k.size)
        penalty_params = int(self.pf_config.report_cluster_bic_penalty_params)
        background_ll = self._report_cluster_log_likelihood(
            filt,
            data,
            design=np.zeros((measurement_count, 0), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=background,
        )
        best: dict[str, Any] = {
            "criterion": background_ll,
            "ll": background_ll,
            "indices": tuple(),
            "strengths": np.zeros(0, dtype=float),
            "condition_number": 1.0,
        }
        best_by_k: dict[int, dict[str, Any]] = {
            0: {
                "criterion": float(background_ll),
                "ll": float(background_ll),
                "condition_number": 1.0,
                "indices": [],
            }
        }
        subset_tasks: list[tuple[int, int, tuple[int, ...], float]] = []
        ordinal = 0
        for k in range(1, max_model_sources + 1):
            penalty = filt._bic_model_penalty(
                measurement_count,
                penalty_params * k,
            )
            for indices in itertools.combinations(range(source_count), k):
                subset_tasks.append(
                    (ordinal, k, tuple(int(i) for i in indices), penalty)
                )
                ordinal += 1

        def _score_subset(
            task: tuple[int, int, tuple[int, ...], float],
        ) -> dict[str, Any]:
            """Score one fixed source subset without mutating estimator state."""
            task_ordinal, k, indices, penalty = task
            subset = np.asarray(indices, dtype=int)
            subset_design = design_arr[:, subset]
            subset_initial = q_initial[subset]
            subset_q = self._solve_report_strengths(
                design=subset_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=subset_initial,
                eps=eps,
                q_max=q_max,
            )
            ll_value = self._report_cluster_log_likelihood(
                filt,
                data,
                design=subset_design,
                strengths=subset_q,
                background=background,
            )
            criterion = float(ll_value - penalty)
            condition_number = self._report_design_condition_number(
                subset_design,
                eps=eps,
            )
            return {
                "ordinal": int(task_ordinal),
                "k": int(k),
                "criterion": float(criterion),
                "ll": float(ll_value),
                "condition_number": float(condition_number),
                "indices": tuple(int(i) for i in indices),
                "strengths": np.asarray(subset_q, dtype=float),
            }

        def _score_subset_batch(
            tasks: list[tuple[int, int, tuple[int, ...], float]],
        ) -> list[dict[str, Any]]:
            """Score same-cardinality subsets using batched NumPy operations."""
            results: list[dict[str, Any]] = []
            tasks_by_k: dict[int, list[tuple[int, int, tuple[int, ...], float]]] = {}
            for task in tasks:
                tasks_by_k.setdefault(int(task[1]), []).append(task)
            for k, grouped_tasks in sorted(tasks_by_k.items()):
                indices = np.asarray(
                    [task[2] for task in grouped_tasks],
                    dtype=int,
                )
                if indices.size == 0:
                    continue
                subset_designs = np.asarray(
                    design_arr[:, indices],
                    dtype=float,
                ).transpose(1, 0, 2)
                subset_initial = q_initial[indices]
                subset_q = self._solve_report_strengths_batch(
                    design_batch=subset_designs,
                    z_obs=data.z_k,
                    background=background,
                    observation_variances=data.observation_variances,
                    initial_strengths=subset_initial,
                    eps=eps,
                    q_max=q_max,
                )
                lam_bm = np.maximum(
                    np.asarray(background, dtype=float)[None, :]
                    + np.einsum("bmk,bk->bm", subset_designs, subset_q),
                    eps,
                )
                ll_values = filt._count_log_likelihood_matrix_np(
                    data.z_k,
                    lam_bm.T,
                    observation_count_variance=data.observation_variances,
                )
                condition_numbers = self._report_design_condition_numbers_batch(
                    subset_designs,
                    eps=eps,
                )
                for local_idx, task in enumerate(grouped_tasks):
                    task_ordinal, _, task_indices, penalty = task
                    ll_value = float(ll_values[local_idx])
                    criterion = float(ll_value - penalty)
                    results.append(
                        {
                            "ordinal": int(task_ordinal),
                            "k": int(k),
                            "criterion": float(criterion),
                            "ll": float(ll_value),
                            "condition_number": float(
                                condition_numbers[local_idx]
                            ),
                            "indices": tuple(int(i) for i in task_indices),
                            "strengths": np.asarray(
                                subset_q[local_idx],
                                dtype=float,
                            ),
                        }
                    )
            return results

        workers = max(1, int(self.pf_config.report_model_order_workers))
        parallel_min = max(
            1,
            int(self.pf_config.report_model_order_parallel_min_subsets),
        )
        use_batched_subset_scoring = workers > 1 and len(subset_tasks) >= parallel_min
        if workers > 1 and len(subset_tasks) >= parallel_min:
            subset_results = _score_subset_batch(subset_tasks)
        else:
            subset_results = [_score_subset(task) for task in subset_tasks]

        for result in sorted(subset_results, key=lambda item: int(item["ordinal"])):
            k = int(result["k"])
            indices = tuple(int(i) for i in result["indices"])
            criterion = float(result["criterion"])
            ll_value = float(result["ll"])
            condition_number = float(result["condition_number"])
            existing = best_by_k.get(k)
            if existing is None or criterion > float(existing["criterion"]):
                best_by_k[k] = {
                    "criterion": float(criterion),
                    "ll": float(ll_value),
                    "condition_number": float(condition_number),
                    "indices": [int(i) for i in indices],
                }
            if criterion > float(best["criterion"]):
                best = {
                    "criterion": criterion,
                    "ll": float(ll_value),
                    "indices": tuple(int(i) for i in indices),
                    "strengths": np.asarray(result["strengths"], dtype=float),
                    "condition_number": float(condition_number),
                }
        selected_indices = tuple(int(i) for i in best["indices"])
        if selected_indices:
            keep[np.asarray(selected_indices, dtype=int)] = True
            q_full[np.asarray(selected_indices, dtype=int)] = np.asarray(
                best["strengths"],
                dtype=float,
            )
            selected_design = design_arr[:, np.asarray(selected_indices, dtype=int)]
            selected_max_corr = self._report_design_max_abs_correlation(
                selected_design,
                eps=eps,
            )
        else:
            selected_max_corr = 0.0
        selected_count = int(np.count_nonzero(keep))
        simpler_criteria = [
            float(stats["criterion"])
            for k, stats in best_by_k.items()
            if k < selected_count
        ]
        simpler_margin = (
            float(best["criterion"]) - max(simpler_criteria)
            if simpler_criteria
            else float("inf")
        )
        runner_up_criteria = [
            float(stats["criterion"])
            for stats in best_by_k.values()
            if list(stats["indices"]) != [int(i) for i in selected_indices]
        ]
        runner_up_margin = (
            float(best["criterion"]) - max(runner_up_criteria)
            if runner_up_criteria
            else float("inf")
        )
        z_arr = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        bg_arr = _measurement_vector(
            background,
            z_arr.size,
            "background",
            min_value=0.0,
        )
        variances = _measurement_vector(
            data.observation_variances,
            z_arr.size,
            "observation_variances",
            min_value=1.0,
        )
        signal = np.maximum(z_arr - bg_arr, 0.0)
        signal_total = float(np.sum(signal))
        signal_max = float(np.max(signal)) if signal.size else 0.0
        signal_snr = float(
            signal_total / np.sqrt(max(float(np.sum(variances)), 1.0e-12))
        )
        selected_residual_fraction = 0.0
        selected_residual_chi2 = 0.0
        if selected_indices:
            selected_design_for_residual = design_arr[
                :,
                np.asarray(selected_indices, dtype=int),
            ]
            selected_q_for_residual = np.asarray(best["strengths"], dtype=float)
            selected_prediction = (
                bg_arr + selected_design_for_residual @ selected_q_for_residual
            )
        else:
            selected_prediction = bg_arr
        selected_residual = np.maximum(z_arr - selected_prediction, 0.0)
        reference_signal = max(float(np.sum(z_arr)), float(np.sum(bg_arr)), eps)
        selected_residual_fraction = float(np.sum(selected_residual)) / reference_signal
        selected_residual_chi2 = float(
            np.sum((selected_residual * selected_residual) / variances)
        )
        zero_source_ready_margin = max(
            float(self.pf_config.report_model_order_min_bic_margin),
            float(self.pf_config.report_model_order_zero_source_min_bic_margin),
        )
        count_supported_zero = (
            selected_count == 0
            and (
                signal_total
                >= max(float(self.pf_config.structural_update_min_counts), 0.0)
                or signal_max
                >= max(float(self.pf_config.conditional_strength_refit_min_count), 0.0)
                or signal_snr
                >= max(float(self.pf_config.structural_update_min_snr), 0.0)
            )
        )
        model_order_ready = True
        min_margin = float(self.pf_config.report_model_order_min_bic_margin)
        if min_margin > 0.0:
            if np.isfinite(runner_up_margin) and runner_up_margin < min_margin:
                model_order_ready = False
        if count_supported_zero and (
            not np.isfinite(runner_up_margin)
            or runner_up_margin < zero_source_ready_margin
        ):
            model_order_ready = False
        if selected_count > 1:
            if np.isfinite(simpler_margin) and simpler_margin < min_margin:
                model_order_ready = False
            max_condition = float(self.pf_config.report_model_order_condition_max)
            if (
                max_condition > 0.0
                and float(best["condition_number"]) > max_condition
            ):
                model_order_ready = False
        self._last_report_model_order_diagnostics[isotope] = {
            "candidate_count": source_count,
            "selected_count": selected_count,
            "selected_indices": [int(i) for i in selected_indices],
            "method": "exhaustive_bic",
            "max_model_sources": int(max_model_sources),
            "workers": int(workers if len(subset_tasks) >= parallel_min else 1),
            "evaluation_mode": (
                "batched_numpy" if use_batched_subset_scoring else "serial"
            ),
            "evaluated_subsets": int(len(subset_tasks) + 1),
            "background_ll": float(background_ll),
            "best_ll": float(best["ll"]),
            "best_criterion": float(best["criterion"]),
            "condition_number": float(best["condition_number"]),
            "selected_max_response_correlation": float(selected_max_corr),
            "criterion_margin_to_simpler": float(simpler_margin),
            "criterion_margin_to_runner_up": float(runner_up_margin),
            "observed_signal_total_counts": float(signal_total),
            "observed_signal_max_count": float(signal_max),
            "observed_signal_snr": float(signal_snr),
            "selected_positive_residual_fraction": float(selected_residual_fraction),
            "selected_positive_residual_chi2": float(selected_residual_chi2),
            "zero_source_ready_margin": float(zero_source_ready_margin),
            "count_supported_zero_source": bool(count_supported_zero),
            "model_order_ready": bool(model_order_ready),
            "best_by_k": {
                str(k): {
                    "criterion": float(stats["criterion"]),
                    "ll": float(stats["ll"]),
                    "condition_number": float(stats["condition_number"]),
                    "indices": [int(i) for i in stats["indices"]],
                }
                for k, stats in sorted(best_by_k.items())
            },
        }
        return keep, q_full

    def _select_report_clusters_by_refit_after_remove(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        design: NDArray[np.float64],
        strengths: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Drop redundant reported clusters using refit-after-remove BIC tests."""
        design_arr = np.asarray(design, dtype=float)
        q = np.asarray(strengths, dtype=float).reshape(-1)
        source_count = int(q.size)
        keep = np.ones(source_count, dtype=bool)
        if (
            source_count <= 1
            or not bool(self.pf_config.report_cluster_model_selection)
            or data.z_k.size == 0
        ):
            return keep, q
        penalty_gain = filt._bic_model_penalty(
            int(data.z_k.size),
            int(self.pf_config.report_cluster_bic_penalty_params),
        )
        allowed_loss = penalty_gain + float(
            self.pf_config.report_cluster_delta_ll_threshold
        )
        while int(np.count_nonzero(keep)) > 1:
            active_idx = np.flatnonzero(keep)
            active_design = design_arr[:, active_idx]
            active_q = q[active_idx]
            ll_full = self._report_cluster_log_likelihood(
                filt,
                data,
                design=active_design,
                strengths=active_q,
                background=background,
            )
            best_global_idx: int | None = None
            best_loss = np.inf
            best_trial_q: NDArray[np.float64] | None = None
            for local_idx, global_idx in enumerate(active_idx):
                trial_local_keep = np.ones(active_idx.size, dtype=bool)
                trial_local_keep[local_idx] = False
                trial_design = active_design[:, trial_local_keep]
                trial_initial = active_q[trial_local_keep]
                trial_q = self._solve_report_strengths(
                    design=trial_design,
                    z_obs=data.z_k,
                    background=background,
                    observation_variances=data.observation_variances,
                    initial_strengths=trial_initial,
                    eps=eps,
                    q_max=q_max,
                )
                ll_without = self._report_cluster_log_likelihood(
                    filt,
                    data,
                    design=trial_design,
                    strengths=trial_q,
                    background=background,
                )
                loss = float(ll_full - ll_without)
                if np.isfinite(loss) and loss <= allowed_loss and loss < best_loss:
                    best_loss = loss
                    best_global_idx = int(global_idx)
                    best_trial_q = trial_q
            if best_global_idx is None or best_trial_q is None:
                break
            keep[best_global_idx] = False
            q[np.flatnonzero(keep)] = best_trial_q
        return keep, q

    def _apply_report_model_order_particle_prune(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        positions: NDArray[np.float64],
        selected_by_model: NDArray[np.bool_],
    ) -> int:
        """Apply report-level BIC cluster rejections to PF source slots."""
        if not bool(self.pf_config.report_model_order_prune_particles):
            return 0
        selected = np.asarray(selected_by_model, dtype=bool).reshape(-1)
        if selected.size == 0 or np.all(selected) or not np.any(selected):
            return 0
        radius = float(self.pf_config.report_model_order_particle_prune_radius_m)
        if radius <= 0.0:
            radius = max(float(self.pf_config.cluster_eps_m), 1.0e-6)
        removed = filt.apply_report_model_order_cluster_prune(
            np.asarray(positions, dtype=float),
            selected,
            radius_m=radius,
        )
        if removed <= 0:
            return 0
        filt.refresh_weights_from_measurements(data)
        diagnostics = self._last_report_model_order_diagnostics.get(isotope)
        if isinstance(diagnostics, dict):
            diagnostics["particle_pruned_source_slots"] = int(removed)
            diagnostics["particle_prune_radius_m"] = float(radius)
        self._invalidate_report_cache()
        return int(removed)

    def _report_surface_local_candidates(
        self,
        filt: IsotopeParticleFilter,
        position_xyz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return a bounded local surface stencil around one report candidate.

        The runtime count response is still evaluated by the normal PF kernel.
        The only approximation here is the finite local candidate stencil, which
        is capped by configuration so it cannot become a global surface search.
        """
        center = np.asarray(position_xyz, dtype=float).reshape(1, 3)
        radius = max(float(self.pf_config.report_surface_local_refine_radius_m), 0.0)
        steps = max(0, int(self.pf_config.report_surface_local_refine_grid_steps))
        max_points = max(
            1,
            int(self.pf_config.report_surface_local_refine_max_candidates_per_source),
        )
        if radius <= 0.0 or steps <= 0:
            projected = filt._project_positions_to_source_prior(center)
            return np.asarray(projected, dtype=float).reshape(-1, 3)
        axis = np.linspace(-radius, radius, 2 * steps + 1, dtype=float)
        mesh = np.meshgrid(axis, axis, axis, indexing="ij")
        offsets = np.stack([item.reshape(-1) for item in mesh], axis=1)
        offset_norm = np.linalg.norm(offsets, axis=1)
        order = np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0], offset_norm))
        offsets = offsets[order[:max_points]]
        projected = filt._project_positions_to_source_prior(center + offsets)
        projected = np.asarray(projected, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(projected), axis=1)
        projected = projected[finite]
        if projected.size == 0:
            return np.zeros((0, 3), dtype=float)
        _, unique_indices = np.unique(
            np.round(projected, 9),
            axis=0,
            return_index=True,
        )
        return projected[np.sort(unique_indices)]

    def _refine_report_surface_positions(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData,
        *,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        design: NDArray[np.float64],
        background: NDArray[np.float64],
        eps: float,
        q_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """
        Refine report positions with a bounded, batched local surface search.

        The coordinate sweep over source slots is intentionally small and
        configuration-capped; each source's local stencil is scored in one
        vectorized likelihood batch.  This avoids the combinatorial explosion of
        a joint continuous all-surface search while preserving the PF response
        model and observation likelihood.
        """
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3).copy()
        q_arr = np.asarray(strengths, dtype=float).reshape(-1).copy()
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0).copy()
        if (
            not bool(self.pf_config.report_surface_local_refine)
            or pos_arr.size == 0
            or q_arr.size != pos_arr.shape[0]
            or design_arr.ndim != 2
            or design_arr.shape[1] != pos_arr.shape[0]
        ):
            return pos_arr, q_arr, {"surface_local_refine_enabled": False}
        source_count = int(pos_arr.shape[0])
        max_sources = int(self.pf_config.report_surface_local_refine_max_sources)
        if max_sources > 0 and source_count > max_sources:
            source_order = np.argsort(np.maximum(q_arr, 0.0))[::-1][:max_sources]
        else:
            source_order = np.arange(source_count, dtype=int)
        if source_order.size == 0:
            return pos_arr, q_arr, {"surface_local_refine_enabled": True}
        current_q = self._solve_report_strengths(
            design=design_arr,
            z_obs=data.z_k,
            background=background,
            observation_variances=data.observation_variances,
            initial_strengths=q_arr,
            eps=eps,
            q_max=q_max,
        )
        current_ll = self._report_cluster_log_likelihood(
            filt,
            data,
            design=design_arr,
            strengths=current_q,
            background=background,
        )
        obs_variances = _measurement_vector(
            data.observation_variances,
            int(data.z_k.size),
            "observation_variances",
            min_value=1.0,
        )
        weights = 1.0 / obs_variances
        accepted = 0
        tested = 0
        total_gain = 0.0
        min_gain = max(
            0.0,
            float(self.pf_config.report_surface_local_refine_min_ll_gain),
        )
        for source_idx_raw in source_order:
            source_idx = int(source_idx_raw)
            local_positions = self._report_surface_local_candidates(
                filt,
                pos_arr[source_idx],
            )
            if local_positions.shape[0] <= 1:
                continue
            tested += int(local_positions.shape[0])
            local_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=local_positions,
                strengths=np.ones(local_positions.shape[0], dtype=float),
            )
            local_design = np.maximum(np.asarray(local_design, dtype=float), 0.0)
            if (
                local_design.ndim != 2
                or local_design.shape[1] != local_positions.shape[0]
            ):
                continue
            other_mask = np.ones(source_count, dtype=bool)
            other_mask[source_idx] = False
            base = np.asarray(background, dtype=float).reshape(-1).copy()
            if np.any(other_mask):
                base = base + design_arr[:, other_mask] @ current_q[other_mask]
            residual = np.asarray(data.z_k, dtype=float).reshape(-1) - base
            numerator = (weights * residual) @ local_design
            denominator = weights @ (local_design * local_design)
            local_q = np.divide(
                numerator,
                np.maximum(denominator, eps),
                out=np.zeros(local_positions.shape[0], dtype=float),
                where=denominator > eps,
            )
            local_q = np.maximum(np.where(np.isfinite(local_q), local_q, 0.0), 0.0)
            if q_max > 0.0:
                local_q = np.minimum(local_q, q_max)
            lam_matrix = np.maximum(
                base[:, None] + local_design * local_q[None, :],
                eps,
            )
            ll_values = filt._count_log_likelihood_matrix_np(
                data.z_k,
                lam_matrix,
                observation_count_variance=data.observation_variances,
            )
            if ll_values.size == 0:
                continue
            best_local = int(np.argmax(ll_values))
            best_ll = float(ll_values[best_local])
            if not np.isfinite(best_ll) or best_ll <= current_ll + min_gain:
                continue
            trial_positions = pos_arr.copy()
            trial_positions[source_idx] = local_positions[best_local]
            trial_design = design_arr.copy()
            trial_design[:, source_idx] = local_design[:, best_local]
            trial_q = self._solve_report_strengths(
                design=trial_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=current_q,
                eps=eps,
                q_max=q_max,
            )
            trial_ll = self._report_cluster_log_likelihood(
                filt,
                data,
                design=trial_design,
                strengths=trial_q,
                background=background,
            )
            if not np.isfinite(trial_ll) or trial_ll <= current_ll + min_gain:
                continue
            total_gain += float(trial_ll - current_ll)
            pos_arr = trial_positions
            design_arr = trial_design
            current_q = trial_q
            current_ll = float(trial_ll)
            accepted += 1
        stats = {
            "surface_local_refine_enabled": True,
            "surface_local_refine_sources_considered": int(source_order.size),
            "surface_local_refine_candidates_tested": int(tested),
            "surface_local_refine_accept_count": int(accepted),
            "surface_local_refine_ll_gain": float(total_gain),
        }
        return pos_arr, current_q, stats

    def _refit_reported_strengths(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Refit reported source strengths with non-negative Poisson regression.

        PF particles estimate source existence and position.  Conditioned on the
        reported positions, the intensity parameters are linear in the expected
        counts, so a multiplicative Poisson regression update gives a
        Rao-Blackwellized strength estimate without changing the transport model
        or inventing source-specific thresholds.
        """
        pos_arr = np.asarray(positions, dtype=float)
        str_arr = np.asarray(strengths, dtype=float).reshape(-1)
        if not bool(self.pf_config.report_strength_refit):
            return pos_arr, str_arr
        rescue_enabled = bool(self.pf_config.report_mle_rescue_enable)
        if (pos_arr.size == 0 or str_arr.size == 0) and not rescue_enabled:
            return pos_arr, str_arr
        if pos_arr.size == 0:
            pos_arr = np.zeros((0, 3), dtype=float)
            str_arr = np.zeros(0, dtype=float)
        if pos_arr.shape[0] != str_arr.size:
            return pos_arr, str_arr
        data = self._measurement_data_for_iso(isotope, None)
        if data is None or data.z_k.size == 0:
            return pos_arr, str_arr
        filt = self.filters.get(isotope)
        if filt is None:
            return pos_arr, str_arr
        if not bool(self.pf_config.report_strength_refit_use_all_measurements):
            refit_data = filt._signal_bearing_refit_data(data)
            if refit_data is None or refit_data.z_k.size == 0:
                return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
            data = refit_data
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        background = _measurement_vector(
            background,
            z_obs.size,
            "background",
            min_value=0.0,
        )
        eps = float(self.pf_config.report_strength_refit_eps)
        q_max = float(getattr(self.pf_config, "birth_q_max", 0.0))
        rescue_stats: dict[str, Any] = {}
        if rescue_enabled:
            pos_arr, str_arr, rescue_stats = (
                self._augment_report_candidates_with_mle_rescue(
                    isotope,
                    filt,
                    data,
                    positions=pos_arr,
                    strengths=str_arr,
                    background=background,
                    eps=eps,
                    q_max=q_max,
                )
            )
            if pos_arr.size == 0 or str_arr.size == 0:
                return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        unit_strengths = np.ones(str_arr.size, dtype=float)
        design = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=pos_arr,
            strengths=unit_strengths,
        )
        design = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design.ndim != 2 or design.shape[1] != str_arr.size:
            return pos_arr, str_arr
        column_sum = np.sum(design, axis=0)
        observable = column_sum > eps
        if not np.any(observable):
            return pos_arr, np.zeros_like(str_arr, dtype=float)
        q = self._solve_report_strengths(
            design=design,
            z_obs=z_obs,
            background=background,
            observation_variances=data.observation_variances,
            initial_strengths=str_arr,
            eps=eps,
            q_max=q_max,
        )
        if bool(self.pf_config.report_surface_local_refine):
            pos_arr, q, refine_stats = self._refine_report_surface_positions(
                isotope,
                filt,
                data,
                positions=pos_arr,
                strengths=q,
                design=design,
                background=background,
                eps=eps,
                q_max=q_max,
            )
            rescue_stats.update(refine_stats)
            str_arr = q.copy()
            unit_strengths = np.ones(str_arr.size, dtype=float)
            design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pos_arr,
                strengths=unit_strengths,
            )
            design = np.maximum(np.asarray(design, dtype=float), 0.0)
            if design.ndim != 2 or design.shape[1] != str_arr.size:
                return pos_arr, str_arr
            column_sum = np.sum(design, axis=0)
            observable = column_sum > eps
            if not np.any(observable):
                return pos_arr, np.zeros_like(str_arr, dtype=float)
            q = self._solve_report_strengths(
                design=design,
                z_obs=z_obs,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=str_arr,
                eps=eps,
                q_max=q_max,
            )
        q_regression = q.copy()
        selected_by_model, q = self._select_report_clusters_by_model_order(
            isotope,
            filt,
            data,
            design=design,
            strengths=q,
            background=background,
            eps=eps,
            q_max=q_max,
        )
        diagnostics = self._last_report_model_order_diagnostics.get(isotope)
        if rescue_stats and isinstance(diagnostics, dict):
            diagnostics.update(rescue_stats)
        support_floor = max(float(self.pf_config.min_strength), 0.0) * (1.0 + 1.0e-6)
        preserve_cardinality = bool(
            self.pf_config.report_strength_refit_preserve_cardinality
        )
        if preserve_cardinality:
            original_supported = np.asarray(str_arr, dtype=float) > support_floor
            keep = observable & original_supported
            if int(np.count_nonzero(keep)) > 1:
                active = np.flatnonzero(keep)
                duplicate_tol = max(
                    1.0e-6,
                    1.0e-3 * float(self.pf_config.cluster_eps_m),
                )
                assigned: set[int] = set()
                for idx in active:
                    idx_int = int(idx)
                    if idx_int in assigned:
                        continue
                    distances = np.linalg.norm(pos_arr[active] - pos_arr[idx_int], axis=1)
                    group = active[distances <= duplicate_tol]
                    assigned.update(int(item) for item in group)
                    if group.size <= 1:
                        continue
                    selected_group = group[selected_by_model[group]]
                    survivor = int(
                        selected_group[0] if selected_group.size > 0 else group[0]
                    )
                    for duplicate_idx in group:
                        if int(duplicate_idx) != survivor:
                            keep[int(duplicate_idx)] = False
            q_report = q.copy()
            collapsed = keep & (q_report <= support_floor)
            q_report[collapsed] = q_regression[collapsed]
            still_collapsed = keep & (q_report <= support_floor)
            q_report[still_collapsed] = np.asarray(str_arr, dtype=float)[still_collapsed]
            diagnostics = self._last_report_model_order_diagnostics.get(isotope)
            if isinstance(diagnostics, dict):
                raw_model_count = int(np.count_nonzero(selected_by_model))
                reported_count = int(np.count_nonzero(keep))
                raw_model_indices = [
                    int(idx) for idx in np.flatnonzero(selected_by_model)
                ]
                reported_indices = [int(idx) for idx in np.flatnonzero(keep)]
                model_order_overridden = bool(np.any(keep & ~selected_by_model))
                diagnostics["preserve_cardinality"] = True
                diagnostics["model_selected_count"] = raw_model_count
                diagnostics["model_selected_indices"] = raw_model_indices
                diagnostics["selected_count"] = reported_count
                diagnostics["selected_indices"] = reported_indices
                diagnostics["reported_count_after_preserve"] = reported_count
                diagnostics["model_order_overridden"] = model_order_overridden
                if model_order_overridden:
                    diagnostics["model_order_ready"] = False
        else:
            keep = observable & selected_by_model & (q > support_floor)
            q_report = q
            pruned_slots = self._apply_report_model_order_particle_prune(
                isotope,
                filt,
                data,
                pos_arr,
                selected_by_model,
            )
            diagnostics = self._last_report_model_order_diagnostics.get(isotope)
            if isinstance(diagnostics, dict):
                diagnostics.setdefault("particle_pruned_source_slots", int(pruned_slots))
        if not np.any(keep):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return pos_arr[keep], q_report[keep]

    def runtime_report_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return the latest station-level report-rescue modes for planning."""
        return {
            isotope: (positions.copy(), strengths.copy(), float(weight))
            for isotope, (positions, strengths, weight) in (
                self._runtime_report_rescue_modes
            ).items()
        }

    def planning_surface_rescue_modes(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Return residual-ranked global surface candidates for DSS-PP planning."""
        if not (
            bool(self.pf_config.birth_global_rescue_enable)
            or bool(self.pf_config.report_mle_rescue_enable)
        ):
            return {}
        payload: Dict[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64], float],
        ] = {}
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window=None)
            if data is None or data.z_k.size == 0:
                continue
            positions = self._runtime_global_birth_rescue_candidates(
                isotope,
                filt,
                data,
            )
            if positions.size == 0 and bool(self.pf_config.report_mle_rescue_enable):
                try:
                    existing_positions, _existing_strengths = filt.estimate_clustered(
                        max_k=max(1, int(self.pf_config.max_sources or 1)),
                        include_report_excluded=True,
                    )
                except (RuntimeError, ValueError, AttributeError):
                    if filt.continuous_particles:
                        best = filt.best_particle().state
                        existing_positions = np.asarray(
                            best.positions[: best.num_sources],
                            dtype=float,
                        )
                    else:
                        existing_positions = np.zeros((0, 3), dtype=float)
                background = self._background_counts_for_report_refit(
                    isotope,
                    data.live_times,
                )
                positions, _q_init, _stats = self._rank_global_surface_candidates(
                    isotope,
                    filt,
                    data,
                    existing_positions=np.asarray(
                        existing_positions,
                        dtype=float,
                    ).reshape(-1, 3),
                    background=background,
                    eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                    q_max=float(self.pf_config.birth_q_max),
                    max_candidates=max(
                        1,
                        int(self.pf_config.birth_global_rescue_max_candidates),
                    ),
                    min_residual_fraction=(
                        self.pf_config.birth_global_rescue_min_residual_fraction
                    ),
                    dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                )
            pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
            if pos_arr.size == 0:
                continue
            background = self._background_counts_for_report_refit(
                isotope,
                data.live_times,
            )
            design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pos_arr,
                strengths=np.ones(pos_arr.shape[0], dtype=float),
            )
            q_arr = self._solve_report_strengths(
                design=design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=np.full(
                    pos_arr.shape[0],
                    max(float(self.pf_config.birth_q_min), 1.0),
                    dtype=float,
                ),
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
            )
            q_arr = np.maximum(np.asarray(q_arr, dtype=float).reshape(-1), 0.0)
            valid = (
                np.isfinite(pos_arr).all(axis=1)
                & np.isfinite(q_arr)
                & (q_arr > max(float(self.pf_config.min_strength), 0.0))
            )
            if not np.any(valid):
                continue
            payload[str(isotope)] = (
                pos_arr[valid].copy(),
                q_arr[valid].copy(),
                float(self.pf_config.runtime_report_rescue_quarantine_weight),
            )
        return payload

    def _runtime_report_rescue_estimate(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a report-rescue model using all measurements collected so far."""
        if (
            not bool(self.pf_config.runtime_report_rescue_enable)
            or not bool(self.pf_config.report_strength_refit)
            or not bool(self.pf_config.report_mle_rescue_enable)
            or not self.measurements
        ):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        max_k = max(
            1,
            int(self.pf_config.report_mle_rescue_max_candidates),
            int(self.pf_config.max_sources or 1),
        )
        try:
            base_pos, base_q = filt.estimate_clustered(
                max_k=max_k,
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if not filt.continuous_particles:
                base_pos = np.zeros((0, 3), dtype=float)
                base_q = np.zeros(0, dtype=float)
            else:
                best = filt.best_particle().state
                count = max(0, int(best.num_sources))
                base_pos = np.asarray(best.positions[:count], dtype=float).reshape(-1, 3)
                base_q = np.asarray(best.strengths[:count], dtype=float).reshape(-1)
        prune_enabled = bool(self.pf_config.report_model_order_prune_particles)
        self.pf_config.report_model_order_prune_particles = False
        try:
            pos_arr, q_arr = self._refit_reported_strengths(
                isotope,
                np.asarray(base_pos, dtype=float).reshape(-1, 3),
                np.asarray(base_q, dtype=float).reshape(-1),
            )
        finally:
            self.pf_config.report_model_order_prune_particles = prune_enabled
        if pos_arr.size == 0 or q_arr.size == 0:
            return self._runtime_unresolved_surface_rescue_estimate(isotope, filt)
        return (
            np.asarray(pos_arr, dtype=float).reshape(-1, 3),
            np.asarray(q_arr, dtype=float).reshape(-1),
        )

    def _runtime_unresolved_surface_rescue_estimate(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return quarantined surface-grid rescue modes for count-supported isotopes."""
        unresolved = self.unresolved_structural_evidence().get(str(isotope), {})
        rescue_reasons = {
            "isotope_absence",
            "report_underfit",
            "birth_residual",
            "runtime_rescue_unverified",
        }
        if not any(reason in unresolved for reason in rescue_reasons):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        data = self._measurement_data_for_iso(isotope, window=None)
        if data is None or data.z_k.size == 0 or self.candidate_sources.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        try:
            existing_positions, _existing_strengths = filt.estimate_clustered(
                max_k=max(1, int(self.pf_config.max_sources or 1)),
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if filt.continuous_particles:
                state = filt.best_particle().state
                existing_positions = np.asarray(
                    state.positions[: state.num_sources],
                    dtype=float,
                )
            else:
                existing_positions = np.zeros((0, 3), dtype=float)
        existing_positions = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        eps = max(float(self.pf_config.report_strength_refit_eps), 1.0e-12)
        q_max = float(getattr(self.pf_config, "birth_q_max", 0.0))
        max_candidates = max(
            1,
            min(
                int(self.pf_config.report_mle_rescue_max_candidates),
                int(self.pf_config.max_sources or self.pf_config.report_mle_rescue_max_candidates),
            ),
        )
        global_pos, global_q, _global_stats = self._rank_global_surface_candidates(
            isotope,
            filt,
            data,
            existing_positions=existing_positions,
            background=background,
            eps=eps,
            q_max=q_max,
            max_candidates=max_candidates,
            min_residual_fraction=0.0,
            dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
        )
        if global_pos.size and global_q.size:
            return self._dedupe_report_candidates(
                np.asarray(global_pos, dtype=float).reshape(-1, 3),
                np.asarray(global_q, dtype=float).reshape(-1),
                radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                max_candidates=max_candidates,
            )
        residual = np.maximum(
            np.asarray(data.z_k, dtype=float).reshape(-1)
            - np.asarray(background, dtype=float).reshape(-1),
            0.0,
        )
        residual_pos, residual_q, _residual_stats = self._rank_residual_surface_candidates(
            isotope,
            filt,
            data,
            residual=residual,
            existing_positions=existing_positions,
            background=background,
            eps=eps,
            max_candidates=max_candidates,
            min_residual_fraction=0.0,
            dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
        )
        if residual_pos.size == 0 or residual_q.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return self._dedupe_report_candidates(
            np.asarray(residual_pos, dtype=float).reshape(-1, 3),
            np.asarray(residual_q, dtype=float).reshape(-1),
            radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            max_candidates=max_candidates,
        )

    def _runtime_report_rescue_memory_limit(self) -> int:
        """Return the per-isotope rescue-memory source limit."""
        configured = max(0, int(self.pf_config.runtime_report_rescue_memory_max_sources))
        if configured > 0:
            return configured
        return max(
            1,
            int(self.pf_config.report_mle_rescue_max_candidates),
            int(self.pf_config.max_sources or 1),
        )

    def _merge_runtime_report_rescue_memory(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Merge current station rescue modes with decayed rescue memory."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if pos_arr.shape[0] != q_arr.size:
            pos_arr = np.zeros((0, 3), dtype=float)
            q_arr = np.zeros(0, dtype=float)
        if not bool(self.pf_config.runtime_report_rescue_memory_enable):
            if pos_arr.size == 0:
                self._runtime_report_rescue_memory.pop(isotope, None)
            return pos_arr, q_arr

        memory = self._runtime_report_rescue_memory.get(isotope)
        decay = float(self.pf_config.runtime_report_rescue_memory_decay)
        if memory is None:
            mem_pos = np.zeros((0, 3), dtype=float)
            mem_q = np.zeros(0, dtype=float)
            mem_score = np.zeros(0, dtype=float)
        else:
            mem_pos, mem_q, mem_score = memory
            mem_pos = np.asarray(mem_pos, dtype=float).reshape(-1, 3)
            mem_q = np.maximum(np.asarray(mem_q, dtype=float).reshape(-1), 0.0)
            mem_score = (
                np.maximum(np.asarray(mem_score, dtype=float).reshape(-1), 0.0)
                * decay
            )
            valid_mem = (
                np.isfinite(mem_pos).all(axis=1)
                & np.isfinite(mem_q)
                & np.isfinite(mem_score)
                & (mem_q > 0.0)
                & (mem_score > max(float(self.pf_config.min_strength), 0.0))
            )
            mem_pos = mem_pos[valid_mem]
            mem_q = mem_q[valid_mem]
            mem_score = mem_score[valid_mem]

        valid_current = (
            np.isfinite(pos_arr).all(axis=1)
            & np.isfinite(q_arr)
            & (q_arr > max(float(self.pf_config.min_strength), 0.0))
        )
        pos_arr = pos_arr[valid_current]
        q_arr = q_arr[valid_current]
        current_score = q_arr.copy()
        if mem_pos.size and pos_arr.size:
            merged_pos = np.vstack([pos_arr, mem_pos])
            merged_q = np.concatenate([q_arr, mem_q])
            merged_score = np.concatenate([current_score, mem_score])
        elif pos_arr.size:
            merged_pos = pos_arr
            merged_q = q_arr
            merged_score = current_score
        elif mem_pos.size:
            merged_pos = mem_pos
            merged_q = mem_q
            merged_score = mem_score
        else:
            self._runtime_report_rescue_memory.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

        order = np.argsort(merged_score)[::-1]
        radius = max(float(self.pf_config.report_mle_rescue_dedup_radius_m), 0.0)
        limit = self._runtime_report_rescue_memory_limit()
        kept_pos: list[NDArray[np.float64]] = []
        kept_q: list[float] = []
        kept_score: list[float] = []
        for idx in order:
            if len(kept_pos) >= limit:
                break
            pos = merged_pos[int(idx)]
            if radius > 0.0 and kept_pos:
                distances = np.linalg.norm(np.vstack(kept_pos) - pos[None, :], axis=1)
                if np.any(distances <= radius):
                    continue
            kept_pos.append(np.asarray(pos, dtype=float))
            kept_q.append(float(merged_q[int(idx)]))
            kept_score.append(float(merged_score[int(idx)]))
        if not kept_pos:
            self._runtime_report_rescue_memory.pop(isotope, None)
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        final_pos = np.vstack(kept_pos)
        final_q = np.asarray(kept_q, dtype=float)
        final_score = np.asarray(kept_score, dtype=float)
        self._runtime_report_rescue_memory[isotope] = (
            final_pos.copy(),
            final_q.copy(),
            final_score.copy(),
        )
        return final_pos, final_q

    def _runtime_report_rescue_injection_weight(
        self,
        isotope: str,
        source_count: int,
    ) -> float:
        """Return the PF mass assigned to a runtime rescue hypothesis."""
        full_weight = float(self.pf_config.runtime_report_rescue_weight)
        candidate_weight = float(
            self.pf_config.runtime_report_rescue_candidate_weight
        )
        if not bool(self.pf_config.runtime_report_rescue_quarantine_enable):
            return full_weight
        quarantine_weight = float(self.pf_config.runtime_report_rescue_quarantine_weight)
        diagnostics = self._last_report_model_order_diagnostics.get(str(isotope), {})
        selected_count = int(diagnostics.get("selected_count", source_count))
        candidate_count = int(diagnostics.get("candidate_count", source_count))
        ready = bool(diagnostics.get("model_order_ready", False))
        method = str(diagnostics.get("method", ""))
        min_margin = max(float(self.pf_config.report_model_order_min_bic_margin), 0.0)
        runner_up_margin = float(
            diagnostics.get("criterion_margin_to_runner_up", np.inf)
        )
        simpler_margin = float(
            diagnostics.get("criterion_margin_to_simpler", np.inf)
        )
        margin_ok = (
            min_margin <= 0.0
            or not np.isfinite(runner_up_margin)
            or runner_up_margin >= min_margin
        )
        if selected_count > 1 and np.isfinite(simpler_margin):
            margin_ok = bool(margin_ok and simpler_margin >= min_margin)
        unresolved = self.unresolved_structural_evidence().get(str(isotope), {})
        if (
            ready
            and not unresolved
            and selected_count == int(source_count)
            and candidate_count >= selected_count
        ):
            return full_weight
        if (
            selected_count > 0
            and selected_count == int(source_count)
            and candidate_count >= selected_count
            and method not in {"", "empty", "source_cap_zero"}
            and margin_ok
        ):
            return float(max(quarantine_weight, min(candidate_weight, full_weight)))
        return quarantine_weight

    def _inject_runtime_report_rescue(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
    ) -> int:
        """Inject station-level report-rescue modes into PF particles."""
        positions, strengths = self._runtime_report_rescue_estimate(isotope, filt)
        positions, strengths = self._merge_runtime_report_rescue_memory(
            isotope,
            positions,
            strengths,
        )
        if positions.size == 0 or strengths.size == 0:
            self._runtime_report_rescue_modes.pop(isotope, None)
            return 0
        injection_weight = self._runtime_report_rescue_injection_weight(
            isotope,
            int(np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]),
        )
        self._runtime_report_rescue_modes[isotope] = (
            positions.copy(),
            strengths.copy(),
            float(max(injection_weight, 0.0)),
        )
        if injection_weight <= 0.0:
            return 0
        injected = filt.inject_runtime_report_rescue_particles(
            positions,
            strengths,
            particle_fraction=self.pf_config.runtime_report_rescue_particle_fraction,
            min_particles_per_source=(
                self.pf_config.runtime_report_rescue_min_particles_per_source
            ),
            total_weight=injection_weight,
            jitter_sigma_m=self.pf_config.runtime_report_rescue_jitter_sigma_m,
        )
        if injected > 0:
            self._runtime_report_rescue_modes[isotope] = (
                positions.copy(),
                strengths.copy(),
                float(injection_weight),
            )
            self._invalidate_report_cache()
        else:
            self._runtime_report_rescue_modes[isotope] = (
                positions.copy(),
                strengths.copy(),
                float(max(injection_weight, 0.0)),
            )
        return int(injected)

    def _global_birth_candidate_counts_for_update(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        candidates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Return detector-filtered global birth candidates and response counts."""
        candidate_arr = np.asarray(candidates, dtype=float).reshape(-1, 3)
        if data is None or data.z_k.size == 0 or candidate_arr.size == 0:
            return candidate_arr, None
        filtered = filt._exclude_birth_candidates_near_detectors(
            candidate_arr,
            data,
        )
        if filtered.size == 0:
            return np.zeros((0, 3), dtype=float), None
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=isotope,
            data=data,
            sources=filtered,
            strengths=np.ones(filtered.shape[0], dtype=float),
        )
        return filtered, np.asarray(counts, dtype=float)

    def _run_isotope_structural_update(
        self,
        task: tuple[
            str,
            IsotopeParticleFilter,
            MeasurementData | None,
            MeasurementData | None,
            MeasurementData | None,
        ],
    ) -> None:
        """Run one isotope's deferred strength refit and birth/death update."""
        isotope, filt, refit_data, support_data, birth_data = task
        if bool(self.pf_config.conditional_strength_refit):
            filt.refit_strengths_for_particles(
                refit_data,
                iters=self.pf_config.conditional_strength_refit_iters,
                eps=self.pf_config.refit_eps,
                suppress_prune_after_refit=bool(
                    self.pf_config.birth_residual_suppress_death
                ),
            )
        proposal_data = birth_data if birth_data is not None else support_data
        global_birth_candidates = self._runtime_global_birth_rescue_candidates(
            isotope,
            filt,
            proposal_data,
        )
        global_birth_candidate_counts = None
        if global_birth_candidates.size:
            (
                global_birth_candidates,
                global_birth_candidate_counts,
            ) = self._global_birth_candidate_counts_for_update(
                isotope,
                filt,
                proposal_data,
                global_birth_candidates,
            )
        filt.apply_birth_death(
            support_data=support_data,
            birth_data=birth_data,
            candidate_positions=self.candidate_sources,
            global_birth_candidates=global_birth_candidates,
            global_birth_candidate_counts=global_birth_candidate_counts,
        )
        self._inject_runtime_report_rescue(isotope, filt)

    def _runtime_global_birth_rescue_candidates(
        self,
        isotope: str,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
    ) -> NDArray[np.float64]:
        """Return surface-grid candidates for station-level global birth rescue."""
        if (
            not bool(self.pf_config.birth_global_rescue_enable)
            or data is None
            or data.z_k.size == 0
        ):
            return np.zeros((0, 3), dtype=float)
        max_candidates = max(0, int(self.pf_config.birth_global_rescue_max_candidates))
        if max_candidates <= 0:
            return np.zeros((0, 3), dtype=float)
        try:
            existing_positions, existing_strengths = filt.estimate_clustered(
                max_k=max(max_candidates, int(self.pf_config.max_sources or 1)),
                include_report_excluded=True,
            )
        except (RuntimeError, ValueError, AttributeError):
            if filt.continuous_particles:
                best_state = filt.best_particle().state
                existing_positions = np.asarray(
                    best_state.positions[: best_state.num_sources],
                    dtype=float,
                )
                existing_strengths = np.asarray(
                    best_state.strengths[: best_state.num_sources],
                    dtype=float,
                )
            else:
                existing_positions = np.zeros((0, 3), dtype=float)
                existing_strengths = np.zeros(0, dtype=float)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        existing_positions = np.asarray(existing_positions, dtype=float).reshape(-1, 3)
        existing_strengths = np.maximum(
            np.asarray(existing_strengths, dtype=float).reshape(-1),
            0.0,
        )
        positions = np.zeros((0, 3), dtype=float)
        if self.candidate_sources.size:
            positions, _strengths, _stats = self._rank_global_surface_candidates(
                isotope,
                filt,
                data,
                existing_positions=existing_positions,
                background=background,
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
                max_candidates=max_candidates,
                min_residual_fraction=(
                    self.pf_config.birth_global_rescue_min_residual_fraction
                ),
                dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            )
        residual_positions = np.zeros((0, 3), dtype=float)
        if existing_positions.size and self.candidate_sources.size:
            existing_design = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=existing_positions,
                strengths=np.ones(existing_positions.shape[0], dtype=float),
            )
            existing_q = self._solve_report_strengths(
                design=existing_design,
                z_obs=data.z_k,
                background=background,
                observation_variances=data.observation_variances,
                initial_strengths=np.full(
                    existing_positions.shape[0],
                    max(float(self.pf_config.birth_q_min), 1.0),
                    dtype=float,
                ),
                eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                q_max=float(self.pf_config.birth_q_max),
            )
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float).reshape(-1)
                - (
                    np.asarray(background, dtype=float).reshape(-1)
                    + np.asarray(existing_design, dtype=float) @ existing_q
                ),
                0.0,
            )
            residual_positions, _residual_strengths, _residual_stats = (
                self._rank_residual_surface_candidates(
                    isotope,
                    filt,
                    data,
                    residual=residual,
                    existing_positions=existing_positions,
                    background=background,
                    eps=max(float(self.pf_config.refit_eps), 1.0e-12),
                    max_candidates=max_candidates,
                    min_residual_fraction=(
                        self.pf_config.birth_global_rescue_min_residual_fraction
                    ),
                    dedup_radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
                )
            )
        merged = (
            np.vstack([positions, residual_positions])
            if positions.size and residual_positions.size
            else positions
            if positions.size
            else residual_positions
        )
        split_positions = self._runtime_high_strength_split_candidates(
            filt,
            existing_positions,
            existing_strengths,
            data=data,
        )
        if split_positions.size:
            merged = (
                np.vstack([merged, split_positions])
                if np.asarray(merged).size
                else split_positions
            )
        if merged.size == 0:
            return np.zeros((0, 3), dtype=float)
        final_positions, _final_strengths = self._dedupe_report_candidates(
            np.asarray(merged, dtype=float).reshape(-1, 3),
            np.ones(np.asarray(merged).reshape(-1, 3).shape[0], dtype=float),
            radius_m=self.pf_config.birth_global_rescue_dedup_radius_m,
            max_candidates=max_candidates,
        )
        return np.asarray(final_positions, dtype=float).reshape(-1, 3)

    def _runtime_high_strength_split_candidates(
        self,
        filt: IsotopeParticleFilter,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        data: MeasurementData | None = None,
    ) -> NDArray[np.float64]:
        """Return surface-projected split candidates around over-strong modes."""
        if not bool(self.pf_config.high_strength_split_enable):
            return np.zeros((0, 3), dtype=float)
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if pos_arr.shape[0] == 0 or pos_arr.shape[0] != q_arr.size:
            return np.zeros((0, 3), dtype=float)
        prior_mean = max(float(self.pf_config.source_strength_prior_mean), 0.0)
        if prior_mean > 0.0:
            reference = max(
                prior_mean,
                float(self.pf_config.split_strength_min),
                1.0,
            )
            threshold = reference * max(
                float(self.pf_config.high_strength_split_q_multiple),
                1.0,
            )
            high = q_arr >= threshold
        else:
            high = self._observation_limited_high_strength_mask(
                filt,
                data,
                pos_arr,
                q_arr,
            )
        if not np.any(high):
            return np.zeros((0, 3), dtype=float)
        count = max(1, int(self.pf_config.high_strength_split_candidate_count))
        offset = max(float(self.pf_config.high_strength_split_offset_m), 1.0e-6)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        xy_offsets = np.column_stack(
            [
                np.cos(angles) * offset,
                np.sin(angles) * offset,
                np.zeros(count, dtype=float),
            ]
        )
        axial_offsets = np.asarray(
            [
                [0.0, 0.0, offset],
                [0.0, 0.0, -offset],
            ],
            dtype=float,
        )
        offsets = np.vstack([xy_offsets, axial_offsets])
        candidates = (pos_arr[high, None, :] + offsets[None, :, :]).reshape(-1, 3)
        projected = filt._project_positions_to_source_prior(candidates)
        parent_positions = pos_arr[high]
        parent_repeated = np.repeat(parent_positions, offsets.shape[0], axis=0)
        distances = np.linalg.norm(projected - parent_repeated, axis=1)
        keep = (
            np.isfinite(projected).all(axis=1)
            & (distances >= 0.5 * max(float(self.pf_config.birth_min_sep_m), 0.0))
        )
        if not np.any(keep):
            return np.zeros((0, 3), dtype=float)
        kept = projected[keep]
        deduped, _ = self._dedupe_report_candidates(
            kept,
            np.ones(kept.shape[0], dtype=float),
            radius_m=max(float(self.pf_config.birth_global_rescue_dedup_radius_m), 0.0),
            max_candidates=max(1, int(self.pf_config.birth_global_rescue_max_candidates)),
        )
        return np.asarray(deduped, dtype=float).reshape(-1, 3)

    def _observation_limited_high_strength_mask(
        self,
        filt: IsotopeParticleFilter,
        data: MeasurementData | None,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return sources whose individual response is too large for observations."""
        pos_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
        q_arr = np.maximum(np.asarray(strengths, dtype=float).reshape(-1), 0.0)
        if data is None or data.z_k.size == 0 or pos_arr.shape[0] != q_arr.size:
            return np.zeros(q_arr.size, dtype=bool)
        if q_arr.size == 0:
            return np.zeros(0, dtype=bool)
        counts = self._cached_expected_counts_per_source(
            filt=filt,
            isotope=filt.isotope,
            data=data,
            sources=pos_arr,
            strengths=np.ones(pos_arr.shape[0], dtype=float),
        )
        counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.shape != (data.z_k.size, q_arr.size):
            return np.zeros(q_arr.size, dtype=bool)
        try:
            background_rate = float(filt.best_particle().state.background)
        except (RuntimeError, ValueError, AttributeError):
            background_rate = 0.0
        background = (
            np.asarray(data.live_times, dtype=float).reshape(-1, 1)
            * max(background_rate, 0.0)
        )
        guarded = filt._apply_runtime_observation_overshoot_guard(
            q_arr[None, :],
            counts[:, None, :],
            data,
            background,
            eps=max(float(self.pf_config.refit_eps), 1.0e-12),
        )
        if guarded.shape != (1, q_arr.size):
            return np.zeros(q_arr.size, dtype=bool)
        shrink = np.asarray(guarded[0], dtype=float) < q_arr * (1.0 - 1.0e-9)
        return np.asarray(shrink, dtype=bool)

    def _structural_update_worker_count(self, task_count: int) -> int:
        """Return the worker count for independent per-isotope structural updates."""
        if task_count <= 1 or not bool(self.pf_config.parallel_isotope_updates):
            return 1
        configured = self.pf_config.parallel_isotope_workers
        if configured is None:
            configured = os.cpu_count() or 1
        return max(1, min(int(configured), int(task_count)))

    def _apply_birth_death(self, birth_window_override: int | None = None) -> None:
        """Apply per-isotope birth/death updates using recent measurements."""
        structural_start = time.perf_counter()
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                MeasurementData | None,
                MeasurementData | None,
                MeasurementData | None,
            ]
        ] = []
        birth_window = (
            self.pf_config.birth_window
            if birth_window_override is None
            else max(1, int(birth_window_override))
        )
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            refit_data = (
                self._measurement_data_for_iso(
                    iso,
                    self.pf_config.conditional_strength_refit_window,
                )
                if bool(self.pf_config.conditional_strength_refit)
                else None
            )
            support_data = self._measurement_data_for_iso(
                iso, self.pf_config.support_window
            )
            birth_data = self._measurement_data_for_iso(iso, birth_window)
            tasks.append((iso, filt, refit_data, support_data, birth_data))
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_structural_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_structural_update(task)
            self.last_structural_update_wall_s = time.perf_counter() - structural_start
            return
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(self._run_isotope_structural_update, tasks))
        self.last_structural_update_wall_s = time.perf_counter() - structural_start

    def _update_pre_finalize_guard(
        self,
        pre_finalize: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
        post_finalize: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """
        Preserve reporting candidates if station finalization collapses them.

        The guard is non-destructive: it only affects reported estimates.  PF
        particles remain governed by their likelihoods, but the final report can
        compare pre/post finalize posterior modes and avoid losing a
        multi-source hypothesis solely due to a late resample/refit/prune step.
        """
        if not bool(self.pf_config.report_pre_finalize_guard):
            self._pre_finalize_guard_estimates.clear()
            return
        max_sources = self.pf_config.max_sources
        for isotope in self.isotopes:
            pre_pos, pre_q = pre_finalize.get(
                isotope,
                (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
            )
            post_pos, _ = post_finalize.get(
                isotope,
                (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
            )
            pre_count = int(np.asarray(pre_pos).shape[0])
            post_count = int(np.asarray(post_pos).shape[0])
            if max_sources is not None and pre_count > int(max_sources):
                self._pre_finalize_guard_estimates.pop(isotope, None)
                continue
            if pre_count > post_count and pre_count > 0:
                self._pre_finalize_guard_estimates[isotope] = (
                    np.asarray(pre_pos, dtype=float).copy(),
                    np.asarray(pre_q, dtype=float).copy(),
                )
            elif post_count >= pre_count:
                self._pre_finalize_guard_estimates.pop(isotope, None)

    def estimates(
        self,
        *,
        use_pre_finalize_guard: bool = True,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return per-isotope position/strength estimates for reporting."""
        cache_revision = int(self._report_cache_revision)
        cache_key = (cache_revision, bool(use_pre_finalize_guard))
        cached = self._report_estimate_cache.get(cache_key)
        if cached is not None:
            return self._copy_estimate_map(cached)
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, filt in self.filters.items():
            use_clustered = bool(
                filt.config.birth_enable and filt.config.use_clustered_output
            )
            if use_clustered and hasattr(filt, "estimate_clustered"):
                try:
                    clustered = filt.estimate_clustered()
                    if clustered[0].shape[0] > 0:
                        estimates[isotope] = self._guarded_report_estimate(
                            isotope,
                            *self._refit_reported_strengths(
                                isotope,
                                clustered[0],
                                clustered[1],
                            ),
                            use_pre_finalize_guard=use_pre_finalize_guard,
                        )
                        continue
                except RuntimeError:
                    estimates[isotope] = (
                        np.zeros((0, 3), dtype=float),
                        np.zeros(0, dtype=float),
                    )
                    continue
            raw_positions, raw_strengths = filt.estimate()
            estimates[isotope] = self._guarded_report_estimate(
                isotope,
                *self._refit_reported_strengths(isotope, raw_positions, raw_strengths),
                use_pre_finalize_guard=use_pre_finalize_guard,
            )
        if int(self._report_cache_revision) == cache_revision:
            self._report_estimate_cache[cache_key] = self._copy_estimate_map(estimates)
        return self._copy_estimate_map(estimates)

    def _guarded_report_estimate(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        use_pre_finalize_guard: bool,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return protected pre-finalize estimates when post-finalize collapsed."""
        pos_arr = np.asarray(positions, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        if not use_pre_finalize_guard:
            return pos_arr, q_arr
        guarded = self._pre_finalize_guard_estimates.get(isotope)
        if guarded is None:
            return pos_arr, q_arr
        guard_pos, guard_q = guarded
        if guard_pos.shape[0] <= pos_arr.shape[0]:
            return pos_arr, q_arr
        refit_pos, refit_q = self._refit_reported_strengths(
            isotope,
            guard_pos,
            guard_q,
        )
        if refit_pos.shape[0] >= guard_pos.shape[0]:
            return refit_pos, refit_q
        return guard_pos.copy(), guard_q.copy()

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def report_model_order_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return the latest report-level model-order diagnostics."""
        return copy.deepcopy(self._last_report_model_order_diagnostics)

    def report_model_order_ready(self) -> bool:
        """Return True when report-level multi-source model orders are stable."""
        diagnostics = self.report_model_order_diagnostics()
        if not diagnostics:
            return not bool(self.unresolved_structural_evidence())
        for stats in diagnostics.values():
            candidate_count = int(stats.get("candidate_count", 0))
            selected_count = int(stats.get("selected_count", 0))
            if max(candidate_count, selected_count) <= 1:
                continue
            if not bool(stats.get("model_order_ready", True)):
                return False
        return not bool(self.unresolved_structural_evidence())

    def unresolved_isotope_evidence(
        self,
        *,
        window: int | None = None,
        min_total_counts: float = 25.0,
        min_max_count: float = 5.0,
        min_snr: float = 2.0,
    ) -> dict[str, dict[str, Any]]:
        """Return isotopes whose observations remain unsupported by zero-source PF MAPs."""
        evidence: dict[str, dict[str, Any]] = {}
        total_floor = max(float(min_total_counts), 0.0)
        max_floor = max(float(min_max_count), 0.0)
        snr_floor = max(float(min_snr), 0.0)
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window)
            if data is None or data.z_k.size == 0:
                continue
            counts = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
            variances = np.maximum(
                np.asarray(data.observation_variances, dtype=float).reshape(-1),
                1.0,
            )
            total_counts = float(np.sum(counts))
            max_count = float(np.max(counts)) if counts.size else 0.0
            snr = float(total_counts / np.sqrt(max(float(np.sum(variances)), 1.0e-12)))
            if not filt.continuous_particles:
                map_count = 0
                source_probability = 0.0
                map_confidence = 1.0
            else:
                weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
                weights_sum = float(np.sum(weights))
                if weights.size == 0 or weights_sum <= 0.0:
                    weights = np.full(
                        len(filt.continuous_particles),
                        1.0 / max(len(filt.continuous_particles), 1),
                        dtype=float,
                    )
                else:
                    weights = weights / weights_sum
                source_counts = np.asarray(
                    [int(particle.state.num_sources) for particle in filt.continuous_particles],
                    dtype=int,
                )
                unique, inverse = np.unique(source_counts, return_inverse=True)
                probs = np.zeros(unique.size, dtype=float)
                np.add.at(probs, inverse, weights)
                best = int(np.argmax(probs)) if probs.size else 0
                map_count = int(unique[best]) if unique.size else 0
                map_confidence = float(probs[best]) if probs.size else 1.0
                source_probability = float(np.sum(weights[source_counts > 0]))
            total_ratio = (
                total_counts / total_floor if total_floor > 0.0 else float(total_counts > 0.0)
            )
            max_ratio = max_count / max_floor if max_floor > 0.0 else float(max_count > 0.0)
            snr_ratio = snr / snr_floor if snr_floor > 0.0 else float(snr > 0.0)
            count_floor_met = total_counts >= total_floor or max_count >= max_floor
            snr_floor_met = snr >= snr_floor
            if snr_floor <= 0.0:
                observed = count_floor_met
            elif total_floor <= 0.0 and max_floor <= 0.0:
                observed = snr_floor_met
            else:
                observed = count_floor_met and snr_floor_met
            if map_count <= 0 and observed:
                evidence[str(isotope)] = {
                    "reason": "observed_counts_without_map_source",
                    "total_counts": total_counts,
                    "max_count": max_count,
                    "count_snr": snr,
                    "map_source_count": int(map_count),
                    "map_cardinality_confidence": map_confidence,
                    "source_probability": source_probability,
                    "budget": float(max(total_ratio, max_ratio, snr_ratio, 1.0) - 1.0),
                    "min_total_counts": total_floor,
                    "min_max_count": max_floor,
                    "min_snr": snr_floor,
                }
        return evidence

    def unresolved_structural_evidence(self) -> dict[str, dict[str, Any]]:
        """
        Return isotope-wise pseudo-source or birth-residual evidence that still needs views.

        Report-level BIC can only compare candidates that survived into the current
        report set.  If residual birth or pseudo-source verification is still asking
        for discriminative views, a one-candidate BIC result must not be treated as
        a settled model order.
        """
        unresolved: dict[str, dict[str, Any]] = {}
        discriminative_reasons = {
            "needs_discriminative_views",
            "insufficient_distinct_views",
            "high_response_corr",
            "too_young_to_prune",
        }
        support_floor = max(1, int(self.pf_config.birth_residual_min_support))
        for isotope, filt in self.filters.items():
            payload: dict[str, Any] = {}
            report_stats = self._last_report_model_order_diagnostics.get(
                str(isotope),
                {},
            )
            if isinstance(report_stats, dict):
                selected_count = int(report_stats.get("selected_count", 0))
                count_supported_zero = bool(
                    report_stats.get("count_supported_zero_source", False)
                )
                ready = bool(report_stats.get("model_order_ready", True))
                if selected_count == 0 and count_supported_zero and not ready:
                    payload["report_underfit"] = {
                        "reason": "count_supported_zero_source",
                        "observed_signal_total_counts": float(
                            report_stats.get("observed_signal_total_counts", 0.0)
                        ),
                        "observed_signal_max_count": float(
                            report_stats.get("observed_signal_max_count", 0.0)
                        ),
                        "observed_signal_snr": float(
                            report_stats.get("observed_signal_snr", 0.0)
                        ),
                        "criterion_margin_to_runner_up": float(
                            report_stats.get(
                                "criterion_margin_to_runner_up",
                                float("inf"),
                            )
                        ),
                        "zero_source_ready_margin": float(
                            report_stats.get("zero_source_ready_margin", 0.0)
                        ),
                    }
                residual_fraction = float(
                    report_stats.get("selected_positive_residual_fraction", 0.0)
                )
                min_fraction = max(
                    float(self.pf_config.report_mle_rescue_min_residual_fraction),
                    0.0,
                )
                if (
                    selected_count > 0
                    and not ready
                    and residual_fraction >= min_fraction
                ):
                    payload["report_underfit"] = {
                        "reason": "selected_model_positive_residual",
                        "selected_count": int(selected_count),
                        "selected_positive_residual_fraction": float(
                            residual_fraction
                        ),
                        "selected_positive_residual_chi2": float(
                            report_stats.get("selected_positive_residual_chi2", 0.0)
                        ),
                    }
            reasons = getattr(filt, "last_pseudo_source_fail_reasons", {})
            reason_payload = (
                {str(reason): int(count) for reason, count in reasons.items()}
                if isinstance(reasons, dict)
                else {}
            )
            unresolved_pseudo = {
                reason: count
                for reason, count in reason_payload.items()
                if reason in discriminative_reasons and int(count) > 0
            }
            if unresolved_pseudo:
                payload["pseudo_source_fail_reasons"] = unresolved_pseudo
            birth_gate_passed = bool(
                getattr(filt, "last_birth_residual_gate_passed", False)
            )
            birth_support = int(getattr(filt, "last_birth_residual_support", 0))
            if birth_gate_passed and birth_support >= support_floor:
                payload["birth_residual"] = {
                    "gate_passed": True,
                    "support": int(birth_support),
                    "support_floor": int(support_floor),
                    "chi2": float(getattr(filt, "last_birth_residual_chi2", 0.0)),
                    "p_value": float(
                        getattr(filt, "last_birth_residual_p_value", 1.0)
                    ),
                }
            rescue_entry = self._runtime_report_rescue_modes.get(str(isotope))
            if rescue_entry is not None:
                rescue_pos, _rescue_q, rescue_weight = rescue_entry
                rescue_count = int(np.asarray(rescue_pos, dtype=float).reshape(-1, 3).shape[0])
                diagnostics = self._last_report_model_order_diagnostics.get(
                    str(isotope),
                    {},
                )
                selected_count = int(diagnostics.get("selected_count", 0))
                report_ready = bool(diagnostics.get("model_order_ready", False))
                quarantine_weight = float(
                    self.pf_config.runtime_report_rescue_quarantine_weight
                )
                if rescue_count > max(selected_count, 0) or (
                    rescue_count > 0
                    and not report_ready
                    and float(rescue_weight) <= quarantine_weight + 1.0e-12
                ):
                    payload["runtime_rescue_unverified"] = {
                        "rescue_source_count": int(rescue_count),
                        "report_selected_count": int(selected_count),
                        "rescue_weight": float(rescue_weight),
                        "quarantine_weight": float(quarantine_weight),
                    }
            if payload:
                unresolved[str(isotope)] = payload
        absent_evidence = self.unresolved_isotope_evidence(
            min_total_counts=max(float(self.pf_config.structural_update_min_counts), 25.0),
            min_max_count=max(float(self.pf_config.conditional_strength_refit_min_count), 5.0),
            min_snr=max(float(self.pf_config.structural_update_min_snr), 2.0),
        )
        for isotope, payload in absent_evidence.items():
            unresolved.setdefault(str(isotope), {})["isotope_absence"] = payload
        return unresolved

    def step_diagnostics(
        self,
        top_k: int = 3,
        *,
        include_estimates: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/kill counts, and the source
        count distribution.  When include_estimates is false, the routine avoids
        report-only clustered MMSE recomputation so per-measurement health logs
        cannot stall the runtime path.
        """
        diagnostics: Dict[str, Dict[str, Any]] = {}
        eps = 1e-12
        k = max(0, int(top_k))
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                diagnostics[iso] = {
                    "ess_pre": 0.0,
                    "resampled": False,
                    "ess_post": None,
                    "n_after_adapt": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "mode_preserved_count": int(
                        getattr(filt, "last_mode_preserved_count", 0)
                    ),
                    "mode_preserving_strata_summary": dict(
                        getattr(filt, "last_mode_preserving_strata_summary", {})
                    ),
                    "mode_preserving_selected_strata": list(
                        getattr(filt, "last_mode_preserving_selected_strata", [])
                    ),
                    "mode_preserving_cardinality_summary": dict(
                        getattr(filt, "last_mode_preserving_cardinality_summary", {})
                    ),
                    "mode_preserving_selected_cardinalities": list(
                        getattr(
                            filt,
                            "last_mode_preserving_selected_cardinalities",
                            [],
                        )
                    ),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "kill_count": int(getattr(filt, "last_kill_count", 0)),
                    "weak_source_prune_occlusion_protected": int(
                        getattr(
                            filt,
                            "last_weak_source_prune_occlusion_protected",
                            0,
                        )
                    ),
                    "birth_residual_chi2": float(
                        getattr(filt, "last_birth_residual_chi2", 0.0)
                    ),
                    "birth_residual_p_value": float(
                        getattr(filt, "last_birth_residual_p_value", 1.0)
                    ),
                    "birth_residual_support": int(
                        getattr(filt, "last_birth_residual_support", 0)
                    ),
                    "birth_residual_distinct_poses": int(
                        getattr(filt, "last_birth_residual_distinct_poses", 0)
                    ),
                    "birth_residual_distinct_stations": int(
                        getattr(filt, "last_birth_residual_distinct_stations", 0)
                    ),
                    "birth_residual_gate_passed": bool(
                        getattr(filt, "last_birth_residual_gate_passed", False)
                    ),
                    "birth_residual_refit_fraction": float(
                        getattr(filt, "last_birth_residual_refit_fraction", 1.0)
                    ),
                    "birth_residual_refit_gate_passed": bool(
                        getattr(filt, "last_birth_residual_refit_gate_passed", True)
                    ),
                    "birth_residual_layer": str(
                        getattr(filt, "last_birth_residual_layer", "none")
                    ),
                    "birth_residual_layer_count": int(
                        getattr(filt, "last_birth_residual_layer_count", 0)
                    ),
                    "birth_forced_attempts": int(
                        getattr(filt, "last_birth_forced_attempts", 0)
                    ),
                    "birth_forced_accepts": int(
                        getattr(filt, "last_birth_forced_accepts", 0)
                    ),
                    "birth_forced_mask_relaxations": int(
                        getattr(filt, "last_birth_forced_mask_relaxations", 0)
                    ),
                    "birth_forced_no_candidate": int(
                        getattr(filt, "last_birth_forced_no_candidate", 0)
                    ),
                    "birth_forced_rejected": int(
                        getattr(filt, "last_birth_forced_rejected", 0)
                    ),
                    "birth_forced_best_delta": float(
                        getattr(filt, "last_birth_forced_best_delta", -np.inf)
                    ),
                    "birth_global_rescue_candidates": int(
                        getattr(filt, "last_birth_global_rescue_candidates", 0)
                    ),
                    "birth_global_rescue_attempts": int(
                        getattr(filt, "last_birth_global_rescue_attempts", 0)
                    ),
                    "birth_global_rescue_accepts": int(
                        getattr(filt, "last_birth_global_rescue_accepts", 0)
                    ),
                    "birth_global_rescue_rejected": int(
                        getattr(filt, "last_birth_global_rescue_rejected", 0)
                    ),
                    "birth_global_rescue_best_delta": float(
                        getattr(filt, "last_birth_global_rescue_best_delta", -np.inf)
                    ),
                    "runtime_report_rescue_candidates": int(
                        getattr(filt, "last_runtime_report_rescue_candidates", 0)
                    ),
                    "runtime_report_rescue_sources": int(
                        getattr(filt, "last_runtime_report_rescue_sources", 0)
                    ),
                    "runtime_report_rescue_injected": int(
                        getattr(filt, "last_runtime_report_rescue_injected", 0)
                    ),
                    "runtime_report_rescue_weight": float(
                        getattr(filt, "last_runtime_report_rescue_weight", 0.0)
                    ),
                    "birth_structural_eligible": int(
                        getattr(filt, "last_birth_structural_eligible", 0)
                    ),
                    "pseudo_source_verified": int(
                        getattr(filt, "last_pseudo_source_verified", 0)
                    ),
                    "pseudo_source_failed": int(
                        getattr(filt, "last_pseudo_source_failed", 0)
                    ),
                    "pseudo_source_pruned": int(
                        getattr(filt, "last_pseudo_source_pruned", 0)
                    ),
                    "pseudo_source_quarantined": int(
                        getattr(filt, "last_pseudo_source_quarantined", 0)
                    ),
                    "pseudo_source_quarantine_active": int(
                        getattr(filt, "last_pseudo_source_quarantine_active", 0)
                    ),
                    "pseudo_source_fail_reasons": dict(
                        getattr(filt, "last_pseudo_source_fail_reasons", {})
                    ),
                    "structural_timing_s": dict(
                        getattr(filt, "last_structural_timing_s", {})
                    ),
                    "temper_steps": [],
                    "temper_resamples": 0,
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "r_weighted_mean": 0.0,
                    "r_weighted_var": 0.0,
                    "r_probability_by_count": {},
                    "r_particle_count_by_count": {},
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "mmse": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "top_k": [],
                    "converged": bool(getattr(filt, "is_converged", False)),
                    "updates_skipped": int(getattr(filt, "updates_skipped", 0)),
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                weights = weights / total
            elif weights.size:
                weights = np.full(weights.size, 1.0 / float(weights.size), dtype=float)
            r_vals = np.array(
                [p.state.num_sources for p in filt.continuous_particles], dtype=float
            )
            if weights.size != r_vals.size and r_vals.size:
                weights = np.full(r_vals.size, 1.0 / float(r_vals.size), dtype=float)
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
            r_int = r_vals.astype(int, copy=False)
            r_weighted_mean = float(np.sum(weights * r_vals)) if r_vals.size else 0.0
            r_weighted_var = (
                float(np.sum(weights * (r_vals - r_weighted_mean) ** 2))
                if r_vals.size
                else 0.0
            )
            r_probability_by_count = {
                str(int(value)): float(np.sum(weights[r_int == int(value)]))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            r_particle_count_by_count = {
                str(int(value)): int(np.count_nonzero(r_int == int(value)))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            ess_pre = getattr(filt, "last_ess_pre", None)
            if ess_pre is None and weights.size:
                ess_pre = float(1.0 / max(np.sum(weights**2), eps))
            if ess_pre is None:
                ess_pre = 0.0
            resampled = bool(getattr(filt, "last_resample_ess", False))
            ess_post = getattr(filt, "last_ess_post", None)
            n_after_adapt = getattr(filt, "last_n_after_adapt", None)
            if n_after_adapt is None:
                n_after_adapt = int(len(filt.continuous_particles))
            best_state = filt.state_without_quarantined_sources(
                filt.best_particle().state
            )
            best_source_count = max(0, int(best_state.num_sources))
            map_positions = best_state.positions[:best_source_count].copy()
            map_strengths = best_state.strengths[:best_source_count].copy()
            if include_estimates:
                try:
                    if bool(
                        filt.config.birth_enable and filt.config.use_clustered_output
                    ) and hasattr(filt, "estimate_clustered"):
                        mmse_positions, mmse_strengths = filt.estimate_clustered()
                    else:
                        mmse_positions, mmse_strengths = filt.estimate()
                except RuntimeError:
                    mmse_positions = np.zeros((0, 3), dtype=float)
                    mmse_strengths = np.zeros(0, dtype=float)
            else:
                mmse_positions = np.zeros((0, 3), dtype=float)
                mmse_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.state_without_quarantined_sources(
                        filt.continuous_particles[int(idx)].state
                    )
                    source_count = max(0, int(state.num_sources))
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": source_count,
                            "positions": state.positions[:source_count].copy(),
                            "strengths": state.strengths[:source_count].copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "n_after_adapt": int(n_after_adapt),
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "mode_preserved_count": int(
                    getattr(filt, "last_mode_preserved_count", 0)
                ),
                "mode_preserving_strata_summary": dict(
                    getattr(filt, "last_mode_preserving_strata_summary", {})
                ),
                "mode_preserving_selected_strata": list(
                    getattr(filt, "last_mode_preserving_selected_strata", [])
                ),
                "mode_preserving_cardinality_summary": dict(
                    getattr(filt, "last_mode_preserving_cardinality_summary", {})
                ),
                "mode_preserving_selected_cardinalities": list(
                    getattr(
                        filt,
                        "last_mode_preserving_selected_cardinalities",
                        [],
                    )
                ),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "kill_count": int(getattr(filt, "last_kill_count", 0)),
                "weak_source_prune_occlusion_protected": int(
                    getattr(
                        filt,
                        "last_weak_source_prune_occlusion_protected",
                        0,
                    )
                ),
                "birth_residual_chi2": float(
                    getattr(filt, "last_birth_residual_chi2", 0.0)
                ),
                "birth_residual_p_value": float(
                    getattr(filt, "last_birth_residual_p_value", 1.0)
                ),
                "birth_residual_support": int(
                    getattr(filt, "last_birth_residual_support", 0)
                ),
                "birth_residual_distinct_poses": int(
                    getattr(filt, "last_birth_residual_distinct_poses", 0)
                ),
                "birth_residual_distinct_stations": int(
                    getattr(filt, "last_birth_residual_distinct_stations", 0)
                ),
                "birth_residual_gate_passed": bool(
                    getattr(filt, "last_birth_residual_gate_passed", False)
                ),
                "birth_residual_refit_fraction": float(
                    getattr(filt, "last_birth_residual_refit_fraction", 1.0)
                ),
                "birth_residual_refit_gate_passed": bool(
                    getattr(filt, "last_birth_residual_refit_gate_passed", True)
                ),
                "birth_residual_layer": str(
                    getattr(filt, "last_birth_residual_layer", "none")
                ),
                "birth_residual_layer_count": int(
                    getattr(filt, "last_birth_residual_layer_count", 0)
                ),
                "birth_forced_attempts": int(
                    getattr(filt, "last_birth_forced_attempts", 0)
                ),
                "birth_forced_accepts": int(
                    getattr(filt, "last_birth_forced_accepts", 0)
                ),
                "birth_forced_mask_relaxations": int(
                    getattr(filt, "last_birth_forced_mask_relaxations", 0)
                ),
                "birth_forced_no_candidate": int(
                    getattr(filt, "last_birth_forced_no_candidate", 0)
                ),
                "birth_forced_rejected": int(
                    getattr(filt, "last_birth_forced_rejected", 0)
                ),
                "birth_forced_best_delta": float(
                    getattr(filt, "last_birth_forced_best_delta", -np.inf)
                ),
                "birth_global_rescue_candidates": int(
                    getattr(filt, "last_birth_global_rescue_candidates", 0)
                ),
                "birth_global_rescue_attempts": int(
                    getattr(filt, "last_birth_global_rescue_attempts", 0)
                ),
                "birth_global_rescue_accepts": int(
                    getattr(filt, "last_birth_global_rescue_accepts", 0)
                ),
                "birth_global_rescue_rejected": int(
                    getattr(filt, "last_birth_global_rescue_rejected", 0)
                ),
                "birth_global_rescue_best_delta": float(
                    getattr(filt, "last_birth_global_rescue_best_delta", -np.inf)
                ),
                "runtime_report_rescue_candidates": int(
                    getattr(filt, "last_runtime_report_rescue_candidates", 0)
                ),
                "runtime_report_rescue_sources": int(
                    getattr(filt, "last_runtime_report_rescue_sources", 0)
                ),
                "runtime_report_rescue_injected": int(
                    getattr(filt, "last_runtime_report_rescue_injected", 0)
                ),
                "runtime_report_rescue_weight": float(
                    getattr(filt, "last_runtime_report_rescue_weight", 0.0)
                ),
                "birth_structural_eligible": int(
                    getattr(filt, "last_birth_structural_eligible", 0)
                ),
                "pseudo_source_verified": int(
                    getattr(filt, "last_pseudo_source_verified", 0)
                ),
                "pseudo_source_failed": int(
                    getattr(filt, "last_pseudo_source_failed", 0)
                ),
                "pseudo_source_pruned": int(
                    getattr(filt, "last_pseudo_source_pruned", 0)
                ),
                "pseudo_source_quarantined": int(
                    getattr(filt, "last_pseudo_source_quarantined", 0)
                ),
                "pseudo_source_quarantine_active": int(
                    getattr(filt, "last_pseudo_source_quarantine_active", 0)
                ),
                "pseudo_source_fail_reasons": dict(
                    getattr(filt, "last_pseudo_source_fail_reasons", {})
                ),
                "structural_timing_s": dict(
                    getattr(filt, "last_structural_timing_s", {})
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "r_mean": r_mean,
                "r_var": r_var,
                "r_weighted_mean": r_weighted_mean,
                "r_weighted_var": r_weighted_var,
                "r_probability_by_count": r_probability_by_count,
                "r_particle_count_by_count": r_particle_count_by_count,
                "map": (map_positions, map_strengths),
                "mmse": (mmse_positions, mmse_strengths),
                "top_k": top_entries,
                "converged": bool(getattr(filt, "is_converged", False)),
                "updates_skipped": int(getattr(filt, "updates_skipped", 0)),
            }
        return diagnostics

    def isotope_log_likelihood_gain(
        self, window: int | None = None
    ) -> Dict[str, float]:
        """
        Return per-isotope log-likelihood gain vs background-only (evidence mixing).
        """
        if not self.measurements:
            return {iso: 0.0 for iso in self.filters}
        estimates = self.pruned_estimates(method="deltall")
        gains: Dict[str, float] = {}
        for iso, filt in self.filters.items():
            data = self._measurement_data_for_iso(iso, window)
            if data is None or data.z_k.size == 0:
                gains[iso] = 0.0
                continue
            positions, strengths = estimates.get(iso, (np.zeros((0, 3)), np.zeros(0)))
            if filt.continuous_particles:
                background_rate = float(filt.best_particle().state.background)
            else:
                background_rate = 0.0
            background_counts = background_rate * data.live_times
            if positions.size == 0:
                gains[iso] = 0.0
                continue
            lambda_m = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=iso,
                data=data,
                sources=positions,
                strengths=strengths,
            )
            lambda_total = background_counts + np.sum(lambda_m, axis=1)
            ll = filt._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
            ll_bg = filt._count_log_likelihood_np(
                data.z_k,
                background_counts,
                observation_count_variance=data.observation_variances,
            )
            gains[iso] = float(ll - ll_bg)
        return gains

    def isotopes_by_evidence(
        self, min_delta_ll: float = 0.0, window: int | None = None
    ) -> List[str]:
        """
        Return isotopes whose LL gain exceeds min_delta_ll for the given window.
        """
        gains = self.isotope_log_likelihood_gain(window=window)
        return [iso for iso, gain in gains.items() if gain >= float(min_delta_ll)]

    @property
    def num_orientations(self) -> int:
        """Return the number of shield orientation normals."""
        return self.normals.shape[0]

    def orientation_information_gain(
        self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0
    ) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        IG_k(phi) ~= 0.5 * log(1 + Var[Lambda_k(phi)] / E[Lambda_k(phi)]) aggregated over isotopes.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            use_continuous = bool(filt.continuous_particles)
            if use_continuous:
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
                w = filt.continuous_weights
            else:
                lam = np.zeros(0, dtype=float)
                w = np.zeros(0, dtype=float)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            ig_total += 0.5 * float(np.log1p(var / max(mean, eps)))
        return ig_total

    def max_orientation_information_gain(
        self, pose_idx: int, live_time_s: float = 1.0
    ) -> float:
        """Return max_phi IG_k(phi) at pose k (Eq. 3.45 surrogate)."""
        scores = [
            self.orientation_information_gain(
                pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s
            )
            for oidx in range(self.num_orientations)
        ]
        return float(np.max(scores)) if scores else 0.0

    def orientation_expected_information_gain(
        self,
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        rng: np.random.Generator | None = None,
        detector_pos: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Monte-Carlo approximation of EIG (Eq. 3.44) for a Fe/Pb orientation pair.

        - Uses continuous particles and ContinuousKernel expected counts (Eq. 3.41).
        - For each isotope h: IG_h = H(w_h) - E_z[H(w'_h(z; RFe, RPb))].
        - Global IG = Σ_h α_h IG_h, with α_h uniform if not provided.
        - If detector_pos is provided, pose_idx is ignored.
        """
        if detector_pos is None:
            if self.kernel_cache is None:
                self._ensure_kernel_cache()
            detector_pos = self.kernel_cache.poses[pose_idx]
        detector_pos = np.asarray(detector_pos, dtype=float)
        rng = rng or np.random.default_rng()
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        if not self._can_use_gpu():
            return self._orientation_expected_information_gain_cpu(
                pose_idx=pose_idx,
                detector_pos=detector_pos,
                fe_idx=fe_idx,
                pb_idx=pb_idx,
                live_time_s=live_time_s,
                num_samples=int(num_samples),
                alpha_by_isotope=alpha_by_isotope,
                particles_by_isotope=particles_by_isotope,
                rng=rng,
                eps=eps,
            )
        kernel = self._continuous_kernel()
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        # normalize alphas
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils as gpu_mod
        import torch as torch_mod

        gpu_utils = gpu_mod
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        torch = torch_mod

        def _compute_lam_torch(
            states: Sequence[IsotopeState], isotope: str
        ) -> "torch.Tensor":
            """Compute expected counts for a state subset on the torch backend."""
            if not states:
                return torch.zeros(0, device=device, dtype=dtype)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            mu_fe, mu_pb = kernel._mu_values(isotope=isotope)
            shield_params = kernel.shield_params
            return gpu_utils.expected_counts_pair_torch(
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                mu_fe=mu_fe,
                mu_pb=mu_pb,
                thickness_fe_cm=shield_params.thickness_fe_cm,
                thickness_pb_cm=shield_params.thickness_pb_cm,
                inner_radius_fe_cm=shield_params.inner_radius_fe_cm,
                inner_radius_pb_cm=shield_params.inner_radius_pb_cm,
                shield_geometry_model=shield_params.shield_geometry_model,
                use_angle_attenuation=shield_params.use_angle_attenuation,
                live_time_s=live_time_s,
                device=device,
                dtype=dtype,
                source_scale=self.response_scale_for_isotope(isotope),
                detector_radius_m=kernel.detector_radius_m,
                detector_aperture_samples=kernel.detector_aperture_samples,
                buildup_fe_coeff=shield_params.buildup_fe_coeff,
                buildup_pb_coeff=shield_params.buildup_pb_coeff,
                **kernel.obstacle_gpu_kwargs(isotope),
            )

        total_ig = 0.0
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights = np.asarray(weights, dtype=float)
            weights = weights / max(np.sum(weights), eps)
            lam_t = _compute_lam_torch(states, iso)
            weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
            weight_sum = torch.sum(weights_t)
            if float(weight_sum) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum
            H_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
            if num_samples <= 0:
                H_post_mean = torch.zeros((), device=device, dtype=dtype)
            else:
                idx = torch.multinomial(weights_t, num_samples, replacement=True)
                z = torch.poisson(lam_t[idx])
                logw = (
                    torch.log(weights_t + eps)
                    + z.unsqueeze(1) * torch.log(lam_t + eps)
                    - lam_t
                )
                logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                w_post = torch.exp(logw)
                H_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                H_post_mean = torch.mean(H_post)
            ig_h = float((H_prior - H_post_mean).item())
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)

    def orientation_expected_information_gain_grid(
        self,
        pose_idx: int,
        *,
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute MC-EIG for all Fe/Pb orientation pairs using shared lambdas.

        This evaluates the same likelihood-entropy estimator as
        ``orientation_expected_information_gain`` but avoids recomputing the
        continuous expected-count kernel separately for every orientation pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        if not self._can_use_gpu():
            from measurement.shielding import generate_octant_rotation_matrices

            rot_mats = generate_octant_rotation_matrices()
            scores = np.zeros((num_orients, num_orients), dtype=float)
            for fe_idx, RFe in enumerate(rot_mats[:num_orients]):
                for pb_idx, RPb in enumerate(rot_mats[:num_orients]):
                    scores[fe_idx, pb_idx] = self.orientation_expected_information_gain(
                        pose_idx=pose_idx,
                        RFe=RFe,
                        RPb=RPb,
                        live_time_s=live_time_s,
                        num_samples=num_samples,
                        alpha_by_isotope=alpha_by_isotope,
                        particles_by_isotope=particles_by_isotope,
                    )
            return scores

        detector_pos = np.asarray(self.kernel_cache.poses[int(pose_idx)], dtype=float)
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        iso_data: Dict[str, tuple["torch.Tensor", "torch.Tensor"]] = {}
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas_np = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            lam_all = torch.as_tensor(lambdas_np, device=device, dtype=dtype)
            weights_t = torch.as_tensor(weights_arr, device=device, dtype=dtype)
            weight_sum_t = torch.sum(weights_t)
            if float(weight_sum_t.detach().cpu().item()) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum_t
            iso_data[iso] = (lam_all, weights_t)
        if not iso_data:
            return np.zeros((num_orients, num_orients), dtype=float)

        scores = np.zeros(num_pairs, dtype=float)
        for pair_idx in range(num_pairs):
            total_ig = 0.0
            for iso, (lam_all, weights_t) in iso_data.items():
                lam_t = lam_all[pair_idx]
                log_weights = torch.log(weights_t + eps)
                h_prior = -torch.sum(weights_t * log_weights)
                if int(num_samples) <= 0:
                    h_post_mean = torch.zeros((), device=device, dtype=dtype)
                else:
                    idx = torch.multinomial(
                        weights_t, int(num_samples), replacement=True
                    )
                    z = torch.poisson(lam_t[idx])
                    logw = (
                        log_weights.view(1, -1)
                        + z.unsqueeze(1) * torch.log(lam_t + eps).view(1, -1)
                        - lam_t.view(1, -1)
                    )
                    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                    w_post = torch.exp(logw)
                    h_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                    h_post_mean = torch.mean(h_post)
                total_ig += float(alphas.get(iso, 0.0)) * float(
                    (h_prior - h_post_mean).item()
                )
            scores[pair_idx] = total_ig
        return scores.reshape(num_orients, num_orients)

    def _orientation_expected_information_gain_cpu(
        self,
        *,
        pose_idx: int,
        detector_pos: NDArray[np.float64],
        fe_idx: int,
        pb_idx: int,
        live_time_s: float,
        num_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None,
        rng: np.random.Generator,
        eps: float,
    ) -> float:
        """Compute orientation EIG on CPU using the same expected-count kernel."""
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(value) for value in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
        total_ig = 0.0
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lam = self.expected_counts_pair_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                fe_index=int(fe_idx),
                pb_index=int(pb_idx),
                live_time_s=float(live_time_s),
                states=states,
            )
            lam = np.maximum(np.asarray(lam, dtype=float).reshape(-1), eps)
            h_prior = -float(np.sum(weights_arr * np.log(weights_arr + eps)))
            if num_samples <= 0:
                h_post_mean = 0.0
            else:
                sample_indices = rng.choice(
                    weights_arr.size,
                    size=int(num_samples),
                    replace=True,
                    p=weights_arr,
                )
                z_samples = rng.poisson(lam[sample_indices])
                logw = (
                    np.log(weights_arr + eps)[None, :]
                    + z_samples[:, None] * np.log(lam + eps)[None, :]
                    - lam[None, :]
                )
                logw -= logw.max(axis=1, keepdims=True)
                w_post = np.exp(logw)
                w_post /= np.maximum(np.sum(w_post, axis=1, keepdims=True), eps)
                h_post = -np.sum(w_post * np.log(w_post + eps), axis=1)
                h_post_mean = float(np.mean(h_post))
            total_ig += alphas.get(iso, 0.0) * (h_prior - h_post_mean)
        return float(total_ig)

    def _strength_matrix(self, filt: IsotopeParticleFilter) -> NDArray[np.float64]:
        """
        Build a (N, max_r) matrix of source strengths for variance computation (Eq. 3.38 surrogate).
        """
        max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
        mat = np.zeros((len(filt.continuous_particles), max_r), dtype=float)
        for i, p in enumerate(filt.continuous_particles):
            r = p.state.num_sources
            if r > 0:
                mat[i, :r] = p.state.strengths
        return mat

    def expected_uncertainty_after_pose(
        self,
        pose_idx: int,
        fe_index: int | None = None,
        pb_index: int | None = None,
        orient_idx: int = 0,
        live_time_s: float = 1.0,
        num_samples: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo estimate of E[U | q_cand] where U = Σ_h Σ_m Var(q_{h,m}) (Eq. 3.38 surrogate).

        Draw hypothetical Poisson observations at pose q_cand and average posterior variance of strengths.
        Uses either Fe/Pb indices (if provided) or orient_idx into the kernel orientations.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        rng = rng or np.random.default_rng()
        eps = 1e-12
        total_U = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            if fe_index is not None and pb_index is not None:
                lam = filt._continuous_expected_counts_pair(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            else:
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
            strengths_mat = self._strength_matrix(filt)
            U_accum = 0.0
            for _ in range(num_samples):
                n = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[n])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= logsumexp(logw)
                w_post = np.exp(logw)
                if strengths_mat.size == 0:
                    continue
                mean = np.sum(w_post[:, None] * strengths_mat, axis=0)
                var = np.sum(w_post[:, None] * (strengths_mat - mean) ** 2, axis=0)
                U_accum += float(np.sum(var))
            total_U += U_accum / max(num_samples, 1)
        return float(total_U)

    def expected_uncertainty_after_pose_xyz(
        self,
        pose_xyz: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float = 1.0,
        num_samples: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo estimate of E[U | pose_xyz] for an explicit detector position.

        Uses Fe/Pb indices to compute expected counts without relying on pose indices.
        """
        detector_pos = np.asarray(pose_xyz, dtype=float)
        if detector_pos.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        rng = rng or np.random.default_rng()
        num_samples = max(int(num_samples), 1)
        eps = 1e-12
        total_U = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(np.sum(weights), eps)
            lam = filt._continuous_expected_counts_pair_at_pose(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
            if lam.size == 0:
                continue
            strengths_mat = self._strength_matrix(filt)
            U_accum = 0.0
            for _ in range(num_samples):
                n = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[n])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= logsumexp(logw)
                w_post = np.exp(logw)
                if strengths_mat.size == 0:
                    continue
                mean = np.sum(w_post[:, None] * strengths_mat, axis=0)
                var = np.sum(w_post[:, None] * (strengths_mat - mean) ** 2, axis=0)
                U_accum += float(np.sum(var))
            total_U += U_accum / max(num_samples, 1)
        return float(total_U)

    def expected_uncertainty_after_rotation(
        self,
        pose_xyz: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        n_rollouts: int = 64,
        orient_selection: str = "IG",
        return_debug: bool = False,
        rng_seed: int | None = None,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Estimate E[U_after-rotation | pose_xyz] by Monte Carlo rollouts.

        This method has no side effects on the estimator state. Rotation policy:
        - choose the next orientation by maximizing IG
        - stop if max IG < tau_ig
        - stop if accumulated live time reaches tmax_s

        rng_seed can be set to make rollouts deterministic for debugging.
        """
        detector_pos = np.asarray(pose_xyz, dtype=float)
        if detector_pos.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        if orient_selection.lower() != "ig":
            raise ValueError("Only orient_selection='IG' is supported.")
        n_rollouts = int(n_rollouts)
        use_mean_measurement = n_rollouts <= 0
        rollouts = max(1, n_rollouts)
        if rng_seed is None:
            rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        else:
            rng = np.random.default_rng(int(rng_seed))
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        alphas = self.pf_config.alpha_weights
        eig_samples = (
            self.pf_config.planning_eig_samples
            if self.pf_config.planning_eig_samples is not None
            else self.pf_config.eig_num_samples
        )
        rollout_particles = self.pf_config.planning_rollout_particles
        if rollout_particles is None:
            rollout_particles = self.pf_config.planning_particles
        rollout_method = (
            self.pf_config.planning_rollout_method or self.pf_config.planning_method
        )

        fast_result = self._expected_uncertainty_after_rotation_fast(
            detector_pos=detector_pos,
            live_time_per_rot_s=live_time_per_rot_s,
            tau_ig=tau_ig,
            tmax_s=tmax_s,
            rollouts=rollouts,
            eig_samples=eig_samples,
            alpha_by_isotope=alphas,
            rollout_particles=rollout_particles,
            rollout_method=rollout_method,
            use_mean_measurement=use_mean_measurement,
            rng=rng,
            return_debug=return_debug,
        )
        if fast_result is not None:
            return fast_result

        def _select_best_orientation(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[int, int, float]:
            """Return the (fe_idx, pb_idx) pair with the maximum EIG at the given pose."""
            best_ig = -np.inf
            best_fe = 0
            best_pb = 0
            particles_by_iso = None
            if rollout_particles is not None and rollout_particles > 0:
                particles_by_iso = estimator.planning_particles(
                    max_particles=int(rollout_particles),
                    method=rollout_method,
                    rng=rng_local,
                )
            for fe_idx in range(num_fe):
                for pb_idx in range(num_pb):
                    ig_val = estimator.orientation_expected_information_gain(
                        pose_idx=0,
                        RFe=RFe_candidates[fe_idx],
                        RPb=RPb_candidates[pb_idx],
                        live_time_s=live_time_per_rot_s,
                        num_samples=eig_samples,
                        alpha_by_isotope=alphas,
                        particles_by_isotope=particles_by_iso,
                        rng=rng_local,
                        detector_pos=detector_pos,
                    )
                    if ig_val > best_ig:
                        best_ig = ig_val
                        best_fe = fe_idx
                        best_pb = pb_idx
            return best_fe, best_pb, float(best_ig)

        def _simulate_measurement(
            estimator: "RotatingShieldPFEstimator",
            fe_idx: int,
            pb_idx: int,
            rng_local: np.random.Generator,
        ) -> Dict[str, float]:
            """Simulate isotope-wise Poisson observations at the candidate pose."""
            z_k: Dict[str, float] = {}
            for iso, filt in estimator.filters.items():
                if not filt.continuous_particles:
                    z_k[iso] = 0.0
                    continue
                lam = filt._continuous_expected_counts_pair_at_pose(
                    detector_pos=detector_pos,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_per_rot_s,
                )
                if lam.size == 0:
                    z_k[iso] = 0.0
                    continue
                weights = filt.continuous_weights
                if use_mean_measurement:
                    z_k[iso] = float(np.sum(weights * lam))
                else:
                    idx = int(rng_local.choice(len(lam), p=weights))
                    z_k[iso] = float(rng_local.poisson(lam[idx]))
            return z_k

        def _run_once(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[float, Dict[str, Any]]:
            """Run a single rotation rollout and return uncertainty plus debug metadata."""
            elapsed = 0.0
            rotations = 0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
                fe_idx, pb_idx, ig_val = _select_best_orientation(estimator, rng_local)
                iterations.append(
                    {
                        "fe_idx": fe_idx,
                        "pb_idx": pb_idx,
                        "ig": ig_val,
                        "elapsed": elapsed,
                    }
                )
                if ig_val < tau_ig:
                    break
                z_k = _simulate_measurement(estimator, fe_idx, pb_idx, rng_local)
                for iso, val in z_k.items():
                    if iso not in estimator.filters:
                        continue
                    estimator.filters[iso].update_continuous_pair_at_pose(
                        z_obs=val,
                        detector_pos=detector_pos,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=live_time_per_rot_s,
                    )
                elapsed += live_time_per_rot_s
                rotations += 1
            return estimator.global_uncertainty(), {
                "iterations": iterations,
                "elapsed": elapsed,
                "num_rotations": rotations,
            }

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(rollouts):
            estimator_copy = copy.deepcopy(self)
            u_val, debug = _run_once(estimator_copy, rng)
            u_vals.append(u_val)
            debug_rollouts.append(debug)
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def _expected_uncertainty_after_rotation_fast(
        self,
        detector_pos: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        rollouts: int,
        eig_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        rollout_particles: int | None,
        rollout_method: str | None,
        use_mean_measurement: bool,
        rng: np.random.Generator,
        return_debug: bool,
    ) -> float | Tuple[float, Dict[str, Any]] | None:
        """
        Fast GPU rollout evaluation using precomputed lambdas and index-based updates.

        Returns None when the fast path cannot be used.
        """
        if not self.pf_config.use_fast_gpu_rollout:
            return None
        if not self.pf_config.use_gpu:
            return None
        self._gpu_enabled()
        from pf import gpu_utils
        import torch
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        fe_indices = np.repeat(np.arange(num_fe), num_pb)
        pb_indices = np.tile(np.arange(num_pb), num_fe)
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        planning_subset = self.planning_particles(
            max_particles=rollout_particles,
            method=rollout_method,
            rng=rng,
        )

        iso_data: Dict[str, Dict[str, Any]] = {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            if iso in planning_subset and planning_subset[iso][0]:
                states, weights = planning_subset[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = np.asarray(filt.continuous_weights, dtype=float)
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0 or not states:
                continue
            weights = weights / max(np.sum(weights), eps)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            mu_fe, mu_pb = filt.continuous_kernel._mu_values(isotope=iso)
            shield_params = filt.continuous_kernel.shield_params
            lam_all = gpu_utils.expected_counts_all_pairs_torch(
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                mu_fe=mu_fe,
                mu_pb=mu_pb,
                thickness_fe_cm=shield_params.thickness_fe_cm,
                thickness_pb_cm=shield_params.thickness_pb_cm,
                inner_radius_fe_cm=shield_params.inner_radius_fe_cm,
                inner_radius_pb_cm=shield_params.inner_radius_pb_cm,
                shield_geometry_model=shield_params.shield_geometry_model,
                use_angle_attenuation=shield_params.use_angle_attenuation,
                live_time_s=live_time_per_rot_s,
                device=device,
                dtype=dtype,
                source_scale=self.response_scale_for_isotope(iso),
                detector_radius_m=filt.continuous_kernel.detector_radius_m,
                detector_aperture_samples=filt.continuous_kernel.detector_aperture_samples,
                buildup_fe_coeff=shield_params.buildup_fe_coeff,
                buildup_pb_coeff=shield_params.buildup_pb_coeff,
                **filt.continuous_kernel.obstacle_gpu_kwargs(iso),
            )
            iso_data[iso] = {
                "lam": lam_all,
                "strengths": strengths,
                "weights": weights,
                "num_particles": weights.size,
                "resample_threshold": filt.config.resample_threshold,
            }
        if not iso_data:
            return 0.0 if not return_debug else (0.0, {"rollouts": [], "u_vals": []})

        def _select_subset(
            weights: NDArray[np.float64],
            indices: NDArray[np.int64],
            max_particles: int | None,
            method: str | None,
            rng_local: np.random.Generator,
        ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
            """Return subset indices and normalized weights for EIG evaluation."""
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= len(weights)
            ):
                return indices, weights
            method = method or "top_weight"
            if method == "top_weight":
                sel = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[sel]
                sel_weights = sel_weights / max(np.sum(sel_weights), eps)
                return indices[sel], sel_weights
            if method == "resample":
                sel = rng_local.choice(len(weights), size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max(
                    max_particles, 1
                )
                return indices[sel], sel_weights
            raise ValueError(f"Unknown planning particle selection method: {method}")

        def _ig_scores_from_lam(
            lam_all: "torch.Tensor",
            subset_indices: NDArray[np.int64],
            subset_weights: NDArray[np.float64],
            num_samples: int,
        ) -> "torch.Tensor":
            """Compute IG scores for all orientations from precomputed lambdas."""
            if num_samples <= 0:
                weights_t = torch.as_tensor(
                    subset_weights, device=lam_all.device, dtype=lam_all.dtype
                )
                weights_t = weights_t / torch.sum(weights_t)
                h_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
                return torch.full(
                    (lam_all.shape[0],),
                    h_prior,
                    device=lam_all.device,
                    dtype=lam_all.dtype,
                )
            idx_t = torch.as_tensor(
                subset_indices, device=lam_all.device, dtype=torch.long
            )
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(
                subset_weights, device=lam_all.device, dtype=lam_all.dtype
            )
            weights_t = weights_t / torch.sum(weights_t)
            log_weights = torch.log(weights_t + eps)
            h_prior = -torch.sum(weights_t * log_weights)
            weights_row = weights_t.expand(lam_sel.shape[0], -1)
            idx_samples = torch.multinomial(weights_row, num_samples, replacement=True)
            lam_samples = torch.gather(lam_sel, 1, idx_samples)
            z = torch.poisson(lam_samples)
            log_lam = torch.log(lam_sel + eps)
            logw = (
                log_weights.view(1, 1, -1)
                + z.unsqueeze(2) * log_lam.unsqueeze(1)
                - lam_sel.unsqueeze(1)
            )
            logw = logw - torch.logsumexp(logw, dim=2, keepdim=True)
            w_post = torch.exp(logw)
            h_post = -torch.sum(w_post * torch.log(w_post + eps), dim=2)
            h_post_mean = torch.mean(h_post, dim=1)
            return h_prior - h_post_mean

        def _update_weights(
            lam_curr: NDArray[np.float64],
            weights: NDArray[np.float64],
            z_obs: float,
        ) -> NDArray[np.float64]:
            """Update weights using Poisson log-likelihood and normalize."""
            logw = np.log(weights + eps) + z_obs * np.log(lam_curr + eps) - lam_curr
            logw -= np.max(logw)
            w = np.exp(logw)
            total = np.sum(w)
            if total <= 0.0:
                return np.ones_like(weights) / max(len(weights), 1)
            return w / total

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(int(rollouts)):
            weights_by_iso: Dict[str, NDArray[np.float64]] = {}
            indices_by_iso: Dict[str, NDArray[np.int64]] = {}
            for iso, data in iso_data.items():
                n_particles = int(data["num_particles"])
                weights_by_iso[iso] = data["weights"].copy()
                indices_by_iso[iso] = np.arange(n_particles, dtype=int)
            elapsed = 0.0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
                total_ig: "torch.Tensor" | None = None
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    subset_idx, subset_w = _select_subset(
                        weights=weights,
                        indices=indices,
                        max_particles=rollout_particles,
                        method=rollout_method,
                        rng_local=rng,
                    )
                    if subset_w.size == 0:
                        continue
                    ig_scores = _ig_scores_from_lam(
                        lam_all=data["lam"],
                        subset_indices=subset_idx,
                        subset_weights=subset_w,
                        num_samples=int(eig_samples),
                    )
                    weight = float(alphas.get(iso, 0.0))
                    ig_scores = ig_scores * weight
                    if total_ig is None:
                        total_ig = ig_scores
                    else:
                        total_ig = total_ig + ig_scores
                if total_ig is None:
                    break
                best_orient = int(torch.argmax(total_ig).item())
                best_ig = float(total_ig[best_orient].detach().cpu().item())
                iterations.append(
                    {
                        "fe_idx": int(fe_indices[best_orient]),
                        "pb_idx": int(pb_indices[best_orient]),
                        "ig": best_ig,
                        "elapsed": elapsed,
                    }
                )
                if best_ig < tau_ig:
                    break
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                    lam_curr_t = torch.index_select(data["lam"][best_orient], 0, idx_t)
                    lam_curr = lam_curr_t.detach().cpu().numpy()
                    if lam_curr.size == 0:
                        continue
                    if use_mean_measurement:
                        z_obs = float(np.sum(weights * lam_curr))
                    else:
                        idx = int(rng.choice(len(lam_curr), p=weights))
                        z_obs = float(rng.poisson(lam_curr[idx]))
                    weights = _update_weights(lam_curr, weights, z_obs)
                    ess = 1.0 / max(np.sum(weights**2), eps)
                    if ess < float(data["resample_threshold"]) * len(weights):
                        resampled = systematic_resample(np.log(weights + eps))
                        indices = indices[resampled]
                        weights = np.ones_like(weights) / max(len(weights), 1)
                    weights_by_iso[iso] = weights
                    indices_by_iso[iso] = indices
                elapsed += live_time_per_rot_s
            total_u = 0.0
            for iso, data in iso_data.items():
                weights = weights_by_iso[iso]
                indices = indices_by_iso[iso]
                if weights.size == 0:
                    continue
                idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                strengths_t = torch.index_select(data["strengths"], 0, idx_t)
                weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
                weights_t = weights_t / torch.sum(weights_t)
                mean = torch.sum(weights_t[:, None] * strengths_t, dim=0)
                var = torch.sum(weights_t[:, None] * (strengths_t - mean) ** 2, dim=0)
                total_u += float(torch.sum(var).detach().cpu().item())
            u_vals.append(total_u)
            debug_rollouts.append(
                {
                    "iterations": iterations,
                    "elapsed": elapsed,
                    "num_rotations": len(iterations),
                }
            )
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def expected_uncertainty_after_rotation_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        *,
        tau_ig: float,
        t_max_s: float,
        t_short_s: float,
        num_rollouts: int = 0,
        use_mean_measurement: bool = True,
        rng_seed: int | None = 0,
        return_debug: bool = False,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Backward-compatible wrapper for expected_uncertainty_after_rotation.
        """
        n_rollouts = int(num_rollouts)
        if n_rollouts <= 0 and not use_mean_measurement:
            n_rollouts = 1
        if rng_seed is not None:
            np.random.seed(rng_seed)
        return self.expected_uncertainty_after_rotation(
            pose_xyz=detector_pos,
            live_time_per_rot_s=t_short_s,
            tau_ig=tau_ig,
            tmax_s=t_max_s,
            n_rollouts=n_rollouts,
            orient_selection="IG",
            return_debug=return_debug,
            rng_seed=rng_seed,
        )

    def estimate_change_norm(self) -> float:
        """
        Return ||Δs|| + ||Δq|| between the last two estimates (Sec. 3.6 convergence check).
        """
        if len(self.history_estimates) < 2:
            return float("inf")
        prev = self.history_estimates[-2]
        curr = self.history_estimates[-1]
        diff = 0.0
        for iso in self.isotopes:
            prev_pos, prev_str = prev.get(iso, (None, None))
            curr_pos, curr_str = curr.get(iso, (None, None))
            if prev_pos is None or curr_pos is None:
                continue
            m = min(len(prev_pos), len(curr_pos))
            if m > 0:
                diff += float(np.linalg.norm(prev_pos[:m] - curr_pos[:m]))
                diff += float(np.linalg.norm(prev_str[:m] - curr_str[:m]))
        return diff

    def global_uncertainty(self) -> float:
        """
        Return global uncertainty U = Σ_h Σ_j Var(q_{h,j}) (Sec. 3.6).
        """
        total = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            self._gpu_enabled()
            from pf import gpu_utils
            import torch

            device = gpu_utils.resolve_device(self.pf_config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
            states = [p.state for p in filt.continuous_particles]
            _, strengths_t, _, _ = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            weights = torch.as_tensor(
                filt.continuous_weights, device=device, dtype=dtype
            )
            weight_sum = torch.sum(weights)
            if float(weight_sum) <= 0.0:
                weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
            else:
                weights = weights / weight_sum
            mean = torch.sum(weights[:, None] * strengths_t, dim=0)
            var = torch.sum(weights[:, None] * (strengths_t - mean) ** 2, dim=0)
            total += float(torch.sum(var).detach().cpu().item())
        return total

    def credible_region_volumes(
        self, confidence: float = 0.95
    ) -> Dict[str, List[float]]:
        """
        Compute 3D positional credible region volumes for each isotope/source (Sec. 3.5).

        For each source index m (up to max_r across particles), compute weighted mean/cov
        of positions and return ellipsoid volume using chi-square threshold. Used by
        should_stop_shield_rotation/should_stop_exploration to enforce small positional
        uncertainty before declaring convergence.
        """
        volumes: Dict[str, List[float]] = {}
        chi2_thresh = float(chi2.ppf(confidence, df=3))
        for iso, filt in self.filters.items():
            vols: List[float] = []
            if not filt.continuous_particles:
                volumes[iso] = vols
                continue
            w = filt.continuous_weights
            max_r = max(
                (p.state.num_sources for p in filt.continuous_particles), default=0
            )
            for j in range(max_r):
                positions = []
                weights = []
                for wi, p in zip(w, filt.continuous_particles):
                    if p.state.num_sources > j:
                        positions.append(p.state.positions[j])
                        weights.append(wi)
                if not positions:
                    continue
                pos_arr = np.vstack(positions)
                weights_arr = np.asarray(weights)
                weights_arr = weights_arr / max(np.sum(weights_arr), 1e-12)
                mean = np.sum(weights_arr[:, None] * pos_arr, axis=0)
                centered = pos_arr - mean
                cov = centered.T @ (centered * weights_arr[:, None])
                # Ellipsoid volume = 4/3 π sqrt(det(cov * chi2_thresh))
                det_val = np.linalg.det(cov * chi2_thresh)
                if det_val < 0:
                    vol = 0.0
                else:
                    vol = float((4.0 / 3.0) * np.pi * np.sqrt(det_val + 1e-12))
                vols.append(vol)
            volumes[iso] = vols
        return volumes

    def should_stop_shield_rotation(
        self,
        pose_idx: int,
        ig_threshold: float = 1e-3,
        change_tol: float = 1e-2,
        uncertainty_tol: float = 1e-3,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        Stop shield rotation when convergence criteria are met (Sec. 3.5–3.6).

        - max IG_k(φ) below threshold
        - estimate change ||Δs|| + ||Δq|| < change_tol
        - global uncertainty U below threshold
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if len(self.history_estimates) < 2:
            return False
        ig_scores = []
        for oidx in range(self.num_orientations):
            ig_scores.append(
                self.orientation_information_gain(
                    pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s
                )
            )
        max_ig = max(ig_scores) if ig_scores else 0.0
        dwell_time = sum(
            rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx
        )
        # Credible region volumes check (Sec. 3.5)
        volumes = self.credible_region_volumes()
        max_volume = 0.0
        for vols in volumes.values():
            if vols:
                max_volume = max(max_volume, max(vols))
        return (
            (max_ig < ig_threshold)
            and (self.estimate_change_norm() < change_tol)
            and (self.global_uncertainty() < uncertainty_tol)
            and (max_volume < self.pf_config.credible_volume_threshold)
            or (dwell_time >= self.pf_config.max_dwell_time_s)
        )

    def should_stop_exploration(
        self,
        ig_threshold: float = 5e-4,
        change_tol: float = 5e-3,
        uncertainty_tol: float = 5e-4,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        Stop the overall exploration (Sec. 3.6) based on IG and uncertainty convergence.

        - Max IG at the last pose is small
        - Estimate change is small
        - Global uncertainty U is small
        """
        if not self.poses:
            return False
        last_pose_idx = len(self.poses) - 1
        return self.should_stop_shield_rotation(
            pose_idx=last_pose_idx,
            ig_threshold=ig_threshold,
            change_tol=change_tol,
            uncertainty_tol=uncertainty_tol,
            live_time_s=live_time_s,
        )

    def prune_spurious_sources(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, NDArray[np.bool_]]:
        """
        Compute spurious-source keep masks using delta-LL, best-case residual gating, or legacy dominance.

        For each isotope h and each candidate source, the pruning method is applied using
        per-measurement expected counts from the continuous kernel. Optionally drop sources
        below max(min_strength_abs, min_strength_ratio * max_strength).

        Method params (passed via params):
        - deltaLL: deltaLL_min, penalty_d (BIC-style), epsilon
        - bestcase: alpha, lambda_min, lrt_threshold, epsilon
        - legacy: tau_mix, epsilon
        """
        from pf.mixing import prune_spurious_sources_continuous

        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
        )
        return keep_masks

    def pruned_estimates(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Return non-destructively pruned estimates derived from MMSE outputs.

        This uses prune_spurious_sources_continuous() in estimate space and does not
        mutate particle states.
        """
        from pf.mixing import prune_spurious_sources_continuous

        est = self.estimates()
        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
            estimate_snapshot=est,
        )
        pruned: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for iso, (pos, strg) in est.items():
            keep = keep_masks.get(iso)
            if keep is None or keep.size == 0:
                pruned[iso] = (pos, strg)
            else:
                keep_arr = np.asarray(keep, dtype=bool).reshape(-1)
                if keep_arr.size != pos.shape[0]:
                    raise ValueError("report model-order keep mask must match estimate count.")
                diagnostics = self._last_report_model_order_diagnostics.get(iso)
                if isinstance(diagnostics, dict) and bool(
                    diagnostics.get("preserve_cardinality", False)
                ):
                    target_count = min(
                        int(diagnostics.get("selected_count", 0)),
                        int(pos.shape[0]),
                    )
                    if target_count > int(np.count_nonzero(keep_arr)):
                        selected_indices = diagnostics.get("selected_indices", [])
                        selected_iter = (
                            selected_indices
                            if isinstance(selected_indices, list)
                            else []
                        )
                        for idx in selected_iter:
                            idx_int = int(idx)
                            if 0 <= idx_int < keep_arr.size:
                                keep_arr[idx_int] = True
                            if int(np.count_nonzero(keep_arr)) >= target_count:
                                break
                    if target_count > int(np.count_nonzero(keep_arr)):
                        dropped = np.flatnonzero(~keep_arr)
                        strengths = np.asarray(strg, dtype=float).reshape(-1)
                        if dropped.size and strengths.size == keep_arr.size:
                            order = dropped[np.argsort(strengths[dropped])[::-1]]
                        else:
                            order = dropped
                        for idx_int in order:
                            keep_arr[int(idx_int)] = True
                            if int(np.count_nonzero(keep_arr)) >= target_count:
                                break
                pruned[iso] = (pos[keep_arr], strg[keep_arr])
        return pruned
