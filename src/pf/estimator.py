"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
import hashlib
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple
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
from pf.likelihood import (
    CountLikelihoodSpec,
    OBSERVATION_COUNT_VARIANCE_ADDITIONAL,
    OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL,
    count_log_likelihood_terms_np,
    count_log_likelihood_terms_torch,
    expected_counts_per_source,
    predictive_count_likelihood_variance,
    predictive_count_likelihood_variance_torch,
    normalize_observation_count_variance_semantics,
)
from pf.particle_filter import IsotopeParticleFilter, MeasurementData, PFConfig
from pf.posterior_uncertainty import posterior_mode_uncertainty_batched
from pf.reporting import measurement_vector
from pf.resampling import systematic_resample
from pf.state import IsotopeState

if TYPE_CHECKING:
    import torch


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
        - support_ema_alpha: EMA weight for per-source ΔLL support
        - birth_softmax_temp: temperature for residual proposal sampling
        - birth_min_score: score floor for residual proposal sampling
        - birth_enable: enable birth/death/split/merge moves
        - birth_topk_particles: number of top-weight particles for residual mix
        - birth_use_weighted_topk: weight residual mix by particle weights
        - birth_min_sep_m: minimum separation between sources during birth
        - birth_detector_min_sep_m: minimum separation from measured detector poses
        - source_detector_exclusion_m: prioritize evidence-prunable sources near detector poses
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
        - birth_use_shield_coded_residual: rank birth candidates by full shield-coded residual response
        - birth_count_distance_prior_weight: soft proposal weight for high unit-response candidates
        - birth_count_distance_strength_weight: soft penalty for candidates needing high fitted strength
        - birth_count_distance_log_clip: robust log-ratio clip for count-distance proposal terms
        - birth_count_distance_strength_sigma: log-strength scale for the high-strength penalty
        - birth_residual_expand_structural_particles: expand residual birth proposals beyond normal top-k
        - birth_residual_expanded_structural_topk_particles: cap residual-gated structural proposals
        - birth_matching_pursuit_max_new_sources: max sequential residual births per particle
        - birth_matching_pursuit_topk_candidates: candidates evaluated per matching-pursuit birth
        - birth_jitter_topk_candidates: base residual-supported candidates jittered for birth
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
        - source_prune_bic_penalty_params: source parameter count for BIC prune gain
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
        - split_residual_candidate_count: residual candidates evaluated per split
        - merge_prob: probability of merge proposals per particle
        - merge_distance_max: max distance for merge candidates
        - merge_delta_ll_threshold: ΔLL threshold for merge acceptance
        - merge_response_corr_min: response-correlation floor for merge candidates
        - merge_search_topk_pairs: max response-redundant pairs tested per merge move
        - structural_proposal_topk_particles: posterior-support cap for split/merge proposals
        - structural_trial_workers: worker count for deterministic split/merge trial chunks
        - structural_trial_parallel_min_trials: minimum trial count before worker chunks
        - structural_kernel_mode: heuristic legacy moves or target-preserving RJ-MH
        - structural_rj_patch_spacing_m: finite surface-patch spacing for RJ-MH
        - structural_rj_move_probability: per-particle RJ birth/death attempt rate
        - structural_rj_birth_probability: interior-state birth move weight
        - structural_rj_death_probability: interior-state death move weight
        - structural_rj_position_move_probability: global within-K move rate
        - structural_rj_local_position_move_probability: adjacent-patch move rate
        - structural_rj_strength_move_probability: within-K strength-move attempt rate
        - structural_cardinality_prior_probs: optional positive prior masses for K
        - source_position_prior: environment-surface PF source-position support
        - init_num_sources: inclusive range for initial source count per particle
        - init_grid_spacing_m: grid spacing for deterministic particle initialization
        - init_grid_repeats: repeated strength samples per deterministic grid point
        - init_joint_position_design: independent or Latin-hypercube source tuples
        - init_joint_position_retries: complete tuples tested per anchor
        - init_source_min_separation_m: prior minimum within-isotope spacing
        - roughening_k: roughening coefficient for post-resample position jitter
        - min_sigma_pos: minimum roughening sigma (meters)
        - max_sigma_pos: maximum roughening sigma (meters)
        - roughening_decay: multiplier decay per resample within an observation
        - roughening_min_mult: minimum multiplier for roughening decay
        - init_strength_prior: lognormal, uniform, or log_uniform strength prior
        - init_strength_min: optional lower source-strength support in cps@1m
        - init_strength_max: optional upper source-strength support in cps@1m
        - init_strength_log_mean: log-normal median for fallback strength initialization
        - init_strength_log_sigma: log-normal spread for fallback strength initialization
        - strength_log_sigma: log-space jitter for strengths
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
        - measurement_scale_by_isotope_and_pair: shield-pair response scales
        - count_likelihood_model: "poisson", "gaussian", or "student_t"
        - transport_model_rel_sigma: relative model mismatch from scatter/build-up omissions
        - transport_model_abs_sigma: absolute transport-model mismatch floor in counts
        - spectrum_count_rel_sigma: relative spectrum-decomposition count uncertainty
        - spectrum_count_abs_sigma: additive spectrum-decomposition count uncertainty
        - observation_count_variance_includes_counting_noise: whether propagated
          extraction variance already contains its source-equivalent Poisson term
        - observation_count_variance_semantics: explicit covariance meaning;
          complete_statistical prevents adding a second Poisson term
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

    estimator_profile: str = "pf_strict"
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
    measurement_scale_by_isotope_and_pair: Dict[str, Dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | Dict[str, float] = 0.0
    transport_model_abs_sigma: float | Dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_transition_counts: float | Dict[str, float] = 0.0
    observation_count_variance_includes_counting_noise: bool = False
    observation_count_variance_semantics: str = ""
    count_likelihood_df: float = 5.0
    shield_contrast_likelihood_enable: bool = False
    shield_contrast_likelihood_weight: float = 1.0
    shield_contrast_log_sigma_floor: float = 0.5
    shield_contrast_log_sigma_ceiling: float = 2.0
    shield_contrast_min_count: float = 25.0
    shield_contrast_min_views: int = 2
    shield_contrast_likelihood_df: float = 5.0
    shield_view_ratio_likelihood_enable: bool = False
    shield_view_ratio_likelihood_weight: float = 1.0
    shield_view_ratio_likelihood_concentration: float = 128.0
    shield_view_ratio_likelihood_min_total_count: float = 25.0
    shield_view_ratio_likelihood_min_views: int = 2
    station_view_covariance_enable: bool = False
    station_view_correlated_spectrum_fraction: float = 0.0
    direct_spectrum_likelihood_enable: bool = True
    spectrum_likelihood_bin_chunk: int = 512
    min_strength: float = 0.01
    p_birth: float = 0.05
    p_kill: float = 0.1
    support_ema_alpha: float = 0.3
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
    birth_q_max: float = 5e6
    birth_q_min: float = 1e2
    birth_max_per_update: int | None = None
    birth_delta_ll_threshold: float = 0.0
    birth_complexity_penalty: float = 0.0
    birth_bic_penalty_params: int = 4
    birth_min_distinct_poses: int = 1
    birth_residual_clip_quantile: float = 0.95
    birth_residual_gate_p_value: float = 0.05
    birth_residual_min_support: int = 2
    birth_residual_support_sigma: float = 1.0
    birth_min_distinct_stations: int = 1
    birth_candidate_support_fraction: float = 0.05
    birth_use_shield_coded_residual: bool = True
    birth_count_distance_prior_weight: float = 0.5
    birth_count_distance_strength_weight: float = 0.25
    birth_count_distance_log_clip: float = 3.0
    birth_count_distance_strength_sigma: float = 2.0
    birth_residual_expand_structural_particles: bool = True
    birth_residual_expanded_structural_topk_particles: int | None = 256
    birth_matching_pursuit_max_new_sources: int = 3
    birth_matching_pursuit_topk_candidates: int = 16
    birth_orthogonalize_residual_candidates: bool = False
    birth_orthogonal_candidate_corr_max: float = 0.98
    birth_jitter_topk_candidates: int | None = 512
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
    source_prune_min_distinct_stations: int = 2
    source_prune_min_distinct_views: int = 2
    source_prune_fail_grace_stations: int = 2
    source_prune_delta_ll_threshold: float = 0.0
    source_prune_bic_penalty_params: int = 4
    birth_stage_single_station_as_quarantine: bool = True
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
    split_residual_candidate_count: int = 8
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
    merge_response_corr_min: float = 0.995
    merge_search_topk_pairs: int = 8
    structural_proposal_topk_particles: int | None = None
    structural_trial_workers: int = 1
    structural_trial_parallel_min_trials: int = 8
    structural_kernel_mode: str = "heuristic"
    structural_rj_patch_spacing_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_strength_move_probability: float = 1.0
    structural_cardinality_prior_probs: tuple[float, ...] | list[float] | None = None
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
    mode_preserving_dynamic_cardinality_allocation: bool = False
    mode_preserving_dynamic_cardinality_extra_particles: int = 0
    mode_preserving_dynamic_cardinality_min_mass: float = 0.02
    mode_preserving_dynamic_cardinality_entropy_min: float = 0.5
    mode_preserving_dynamic_spatial_allocation: bool = False
    mode_preserving_dynamic_spatial_extra_particles: int = 0
    mode_preserving_dynamic_spatial_min_score_fraction: float = 0.005
    adapt_cooldown_steps: int = 0
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "surface"
    init_num_sources: Tuple[int, int] = (0, 3)
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
    init_joint_position_design: str = "independent"
    init_joint_position_retries: int = 1
    init_source_min_separation_m: float = 0.0
    roughening_k: float = 0.5
    surface_rejuvenation_enable: bool = True
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_strength_prior: str = "lognormal"
    init_strength_min: float = 0.0
    init_strength_max: float | None = None
    init_strength_log_mean: float = 9.0
    init_strength_log_sigma: float = 1.0
    strength_log_sigma: float = 0.3
    observation_covariance_projection_enable: bool = True
    observation_covariance_projection_weight: float = 1.0
    observation_covariance_projection_max_corr: float = 0.999
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
        self.init_joint_position_design = (
            str(self.init_joint_position_design).strip().lower().replace("-", "_")
        )
        if self.init_joint_position_design not in {"independent", "latin_hypercube"}:
            raise ValueError(
                "init_joint_position_design must be independent or latin_hypercube."
            )
        self.init_joint_position_retries = max(
            1,
            int(self.init_joint_position_retries),
        )
        self.init_source_min_separation_m = max(
            float(self.init_source_min_separation_m),
            0.0,
        )
        self.init_strength_prior = (
            str(self.init_strength_prior).strip().lower().replace("-", "_")
        )
        if self.init_strength_prior not in {"lognormal", "uniform", "log_uniform"}:
            raise ValueError(
                "init_strength_prior must be lognormal, uniform, or log_uniform."
            )
        self.init_strength_min = max(float(self.init_strength_min), 0.0)
        self.init_strength_max = (
            None if self.init_strength_max is None else float(self.init_strength_max)
        )
        if (
            self.init_strength_max is not None
            and self.init_strength_max < self.init_strength_min
        ):
            raise ValueError("init_strength_max must be >= init_strength_min.")
        if self.init_strength_prior in {"uniform", "log_uniform"}:
            if self.init_strength_max is None or not np.isfinite(
                self.init_strength_max
            ):
                raise ValueError("bounded strength priors require a finite maximum.")
        if self.init_strength_prior == "log_uniform" and self.init_strength_min <= 0.0:
            raise ValueError("log_uniform strength prior requires a positive minimum.")
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")
        self.observation_covariance_projection_enable = bool(
            self.observation_covariance_projection_enable
        )
        self.observation_covariance_projection_weight = max(
            0.0,
            float(self.observation_covariance_projection_weight),
        )
        self.observation_covariance_projection_max_corr = float(
            np.clip(
                float(self.observation_covariance_projection_max_corr),
                0.0,
                1.0,
            )
        )
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
        self.observation_count_variance_semantics = (
            normalize_observation_count_variance_semantics(
                self.observation_count_variance_semantics,
                includes_counting_noise=(
                    self.observation_count_variance_includes_counting_noise
                ),
            )
        )
        self.observation_count_variance_includes_counting_noise = (
            self.observation_count_variance_semantics
            != OBSERVATION_COUNT_VARIANCE_ADDITIONAL
        )
        if (
            self.observation_count_variance_semantics
            == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL
            and self.count_likelihood_model == "poisson"
        ):
            raise ValueError(
                "complete_statistical observation variance requires gaussian "
                "or student_t count likelihood."
            )
        self.count_likelihood_df = max(float(self.count_likelihood_df), 1.0)
        self.shield_contrast_likelihood_enable = bool(
            self.shield_contrast_likelihood_enable
        )
        self.shield_view_ratio_likelihood_enable = bool(
            self.shield_view_ratio_likelihood_enable
        )
        if (
            self.observation_count_variance_semantics
            == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL
        ):
            # These auxiliary terms reuse the same shield-view counts without
            # the supplied cross-view covariance. Applying them would count the
            # observation twice under complete statistical semantics.
            self.shield_contrast_likelihood_enable = False
            self.shield_view_ratio_likelihood_enable = False
        self.shield_view_ratio_likelihood_weight = max(
            0.0,
            float(self.shield_view_ratio_likelihood_weight),
        )
        self.shield_view_ratio_likelihood_concentration = max(
            1.0e-6,
            float(self.shield_view_ratio_likelihood_concentration),
        )
        self.shield_view_ratio_likelihood_min_total_count = max(
            0.0,
            float(self.shield_view_ratio_likelihood_min_total_count),
        )
        self.shield_view_ratio_likelihood_min_views = max(
            2,
            int(self.shield_view_ratio_likelihood_min_views),
        )
        self.station_view_covariance_enable = bool(self.station_view_covariance_enable)
        self.station_view_correlated_spectrum_fraction = max(
            0.0,
            float(self.station_view_correlated_spectrum_fraction),
        )
        self.direct_spectrum_likelihood_enable = bool(
            self.direct_spectrum_likelihood_enable
        )
        self.spectrum_likelihood_bin_chunk = max(
            1,
            int(self.spectrum_likelihood_bin_chunk),
        )
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
        self.birth_orthogonalize_residual_candidates = bool(
            self.birth_orthogonalize_residual_candidates
        )
        self.birth_orthogonal_candidate_corr_max = float(
            np.clip(float(self.birth_orthogonal_candidate_corr_max), 0.0, 1.0)
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
        self.mode_preserving_surface_strata = bool(self.mode_preserving_surface_strata)
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
        self.mode_preserving_dynamic_cardinality_allocation = bool(
            self.mode_preserving_dynamic_cardinality_allocation
        )
        self.mode_preserving_dynamic_cardinality_extra_particles = max(
            0,
            int(self.mode_preserving_dynamic_cardinality_extra_particles),
        )
        self.mode_preserving_dynamic_cardinality_min_mass = max(
            0.0,
            float(self.mode_preserving_dynamic_cardinality_min_mass),
        )
        self.mode_preserving_dynamic_cardinality_entropy_min = max(
            0.0,
            float(self.mode_preserving_dynamic_cardinality_entropy_min),
        )
        self.mode_preserving_dynamic_spatial_allocation = bool(
            self.mode_preserving_dynamic_spatial_allocation
        )
        self.mode_preserving_dynamic_spatial_extra_particles = max(
            0,
            int(self.mode_preserving_dynamic_spatial_extra_particles),
        )
        self.mode_preserving_dynamic_spatial_min_score_fraction = max(
            0.0,
            float(self.mode_preserving_dynamic_spatial_min_score_fraction),
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
        self.surface_rejuvenation_enable = bool(self.surface_rejuvenation_enable)
        self.structural_kernel_mode = (
            str(self.structural_kernel_mode).strip().lower().replace("-", "_")
        )
        if self.structural_kernel_mode not in {"heuristic", "rj_mh"}:
            raise ValueError(
                "structural_kernel_mode must be 'heuristic' or 'rj_mh'."
            )
        self.structural_rj_patch_spacing_m = float(
            self.structural_rj_patch_spacing_m
        )
        if (
            not np.isfinite(self.structural_rj_patch_spacing_m)
            or self.structural_rj_patch_spacing_m <= 0.0
        ):
            raise ValueError("structural_rj_patch_spacing_m must be positive.")
        for probability_field in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
        ):
            probability = float(getattr(self, probability_field))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_field} must be in [0, 1].")
            setattr(self, probability_field, probability)
        if self.structural_cardinality_prior_probs is not None:
            if not isinstance(
                self.structural_cardinality_prior_probs,
                (tuple, list),
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must be a tuple, "
                    "list, or None."
                )
            cardinality_prior = tuple(
                float(value)
                for value in self.structural_cardinality_prior_probs
            )
            if not cardinality_prior or any(
                not np.isfinite(value) or value <= 0.0
                for value in cardinality_prior
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain only "
                    "positive finite values."
                )
            self.structural_cardinality_prior_probs = cardinality_prior
        self.cardinality_preserving_resample = bool(
            self.cardinality_preserving_resample
        )
        self.mode_preserving_resample = bool(self.mode_preserving_resample)
        if self.structural_kernel_mode == "rj_mh" and bool(self.birth_enable):
            if self.source_position_prior != "surface":
                raise ValueError(
                    "structural_kernel_mode='rj_mh' requires "
                    "source_position_prior='surface'."
                )
            if self.max_sources is None or int(self.max_sources) < 1:
                raise ValueError(
                    "structural_kernel_mode='rj_mh' requires a finite "
                    "positive max_sources."
                )
            expected_cardinalities = int(self.max_sources) + 1
            if (
                self.structural_cardinality_prior_probs is not None
                and len(self.structural_cardinality_prior_probs)
                != expected_cardinalities
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain "
                    "max_sources + 1 entries in rj_mh mode."
                )
            initial_lower, initial_upper = self.init_num_sources
            if (
                int(initial_lower) != 0
                or int(initial_upper) != int(self.max_sources)
            ):
                raise ValueError(
                    "structural_kernel_mode='rj_mh' requires "
                    "init_num_sources=(0, max_sources)."
                )
            if (
                self.structural_rj_birth_probability <= 0.0
                or self.structural_rj_death_probability <= 0.0
            ):
                raise ValueError(
                    "structural_kernel_mode='rj_mh' with structural moves "
                    "enabled requires positive birth and death probabilities."
                )
            incompatible = {
                "split_prob": float(self.split_prob) != 0.0,
                "merge_prob": float(self.merge_prob) != 0.0,
                "surface_rejuvenation_enable": self.surface_rejuvenation_enable,
                "mode_preserving_resample": self.mode_preserving_resample,
                "cardinality_preserving_resample": (
                    self.cardinality_preserving_resample
                ),
                "pseudo_source_verification_enable": bool(
                    self.pseudo_source_verification_enable
                ),
                "source_detector_exclusion_m": (
                    self.source_detector_exclusion_m > 0.0
                ),
                "init_source_min_separation_m": (
                    self.init_source_min_separation_m > 0.0
                ),
            }
            enabled_incompatible = [
                name for name, enabled in incompatible.items() if enabled
            ]
            if enabled_incompatible:
                joined = ", ".join(enabled_incompatible)
                raise ValueError(
                    "structural_kernel_mode='rj_mh' requires incompatible "
                    f"options to be disabled or zero: {joined}."
                )
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
        self.source_prune_bic_penalty_params = max(
            0,
            int(self.source_prune_bic_penalty_params),
        )
        self.birth_residual_expand_structural_particles = bool(
            self.birth_residual_expand_structural_particles
        )
        if self.birth_max_per_update is not None:
            self.birth_max_per_update = max(0, int(self.birth_max_per_update))
        self.birth_delta_ll_threshold = float(self.birth_delta_ll_threshold)
        self.birth_complexity_penalty = max(0.0, float(self.birth_complexity_penalty))
        self.birth_bic_penalty_params = max(0, int(self.birth_bic_penalty_params))
        self.birth_stage_single_station_as_quarantine = bool(
            self.birth_stage_single_station_as_quarantine
        )
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
    z_covariance_k: Dict[str, Dict[str, float]] | None = None
    ig_value: float | None = None
    spectrum_counts: tuple[float, ...] | None = None
    spectrum_variance: tuple[float, ...] | None = None
    spectrum_background: tuple[float, ...] | None = None
    spectrum_background_source: str | None = None
    spectrum_background_observation_independent: bool = False
    spectrum_response_templates_by_isotope: Dict[str, tuple[float, ...]] | None = None
    detector_position_xyz_m: tuple[float, float, float] | None = None
    station_sequence_id: int | None = None
    station_view_index: int | None = None
    runtime_likelihood_route_by_isotope: Dict[str, str] | None = None
    runtime_spectrum_variance_used_by_isotope: Dict[str, bool] | None = None
    station_view_covariance_by_isotope: (
        Dict[str, tuple[tuple[float, ...], ...]] | None
    ) = None


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
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: Dict[str, object] | None = None,
        transport_response_model: Dict[str, object] | None = None,
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
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = max(float(detector_aperture_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        self.detector_aperture_sampling = str(detector_aperture_sampling)
        self.source_extent_radius_m = max(float(source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(source_extent_samples), 1)
        self.line_mu_by_isotope = line_mu_by_isotope
        self.transport_response_model = transport_response_model
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
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_pair_sequence_stage_wall_s: Dict[str, float] = {}
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = 0.0
        self._candidate_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._candidate_response_cache_order: list[tuple[Any, ...]] = []
        self._candidate_response_prefix_cache: dict[
            tuple[Any, ...],
            dict[str, NDArray[np.float64]],
        ] = {}
        self._candidate_response_prefix_cache_order: list[tuple[Any, ...]] = []
        self._configured_response_kernel_registry: dict[str, ContinuousKernel] = {}
        self._configured_spectrum_response_registry: dict[
            tuple[str, int],
            tuple[float, ...],
        ] = {}

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

    @staticmethod
    def _response_scale_vector(
        scale: NDArray[np.float64] | float,
        measurement_count: int,
    ) -> NDArray[np.float64]:
        """Return one response-scale value per measurement row."""
        count = max(0, int(measurement_count))
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if count == 0:
            return np.zeros(0, dtype=float)
        if scale_arr.size == 0:
            return np.ones(count, dtype=float)
        if scale_arr.size == 1 and count != 1:
            return np.full(count, float(scale_arr[0]), dtype=float)
        if scale_arr.size != count:
            raise ValueError(
                "source_scale must be scalar or one value per measurement."
            )
        return scale_arr.astype(float, copy=False)

    def _store_candidate_response_cache(
        self,
        cache_key: tuple[Any, ...],
        counts: NDArray[np.float64],
    ) -> None:
        """Store an exact deterministic candidate response with LRU eviction."""
        self._candidate_response_cache[cache_key] = np.asarray(
            counts,
            dtype=float,
        ).copy()
        self._candidate_response_cache_order.append(cache_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_cache_order) > max_entries:
            old_key = self._candidate_response_cache_order.pop(0)
            if old_key not in self._candidate_response_cache_order:
                self._candidate_response_cache.pop(old_key, None)

    @staticmethod
    def _candidate_response_prefix_matches(
        payload: Mapping[str, NDArray[np.float64]],
        data: MeasurementData,
        scale: NDArray[np.float64],
        row_count: int,
    ) -> bool:
        """Return True when cached response rows match the requested prefix."""
        rows = max(0, int(row_count))
        if rows < 0:
            return False
        try:
            return bool(
                np.allclose(
                    np.asarray(payload["detector_positions"], dtype=float)[:rows],
                    np.asarray(data.detector_positions, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    np.asarray(payload["live_times"], dtype=float)[:rows],
                    np.asarray(data.live_times, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.array_equal(
                    np.asarray(payload["fe_indices"], dtype=int)[:rows],
                    np.asarray(data.fe_indices, dtype=int)[:rows],
                )
                and np.array_equal(
                    np.asarray(payload["pb_indices"], dtype=int)[:rows],
                    np.asarray(data.pb_indices, dtype=int)[:rows],
                )
                and np.allclose(
                    np.asarray(payload["source_scale"], dtype=float)[:rows],
                    np.asarray(scale, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _store_candidate_response_prefix_cache(
        self,
        prefix_key: tuple[Any, ...],
        data: MeasurementData,
        scale: NDArray[np.float64],
        counts: NDArray[np.float64],
    ) -> None:
        """Store a full-history response prefix for incremental extension."""
        self._candidate_response_prefix_cache[prefix_key] = {
            "detector_positions": np.asarray(
                data.detector_positions, dtype=float
            ).copy(),
            "live_times": np.asarray(data.live_times, dtype=float).copy(),
            "fe_indices": np.asarray(data.fe_indices, dtype=int).copy(),
            "pb_indices": np.asarray(data.pb_indices, dtype=int).copy(),
            "source_scale": np.asarray(scale, dtype=float).copy(),
            "counts": np.asarray(counts, dtype=float).copy(),
        }
        self._candidate_response_prefix_cache_order.append(prefix_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_prefix_cache_order) > max_entries:
            old_key = self._candidate_response_prefix_cache_order.pop(0)
            if old_key not in self._candidate_response_prefix_cache_order:
                self._candidate_response_prefix_cache.pop(old_key, None)

    def _cached_expected_counts_for_kernel(
        self,
        *,
        kernel: ContinuousKernel,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched expected counts from a particle-independent kernel."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._candidate_response_source_key(source_arr)
        scale = self.response_scales_for_measurements(
            isotope,
            data.fe_indices,
            data.pb_indices,
        )
        measurement_count = int(np.asarray(data.live_times, dtype=float).size)
        scale_arr = self._response_scale_vector(scale, measurement_count)
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.candidate_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        prefix_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(kernel)),
                source_key,
                self._measurement_geometry_digest(data),
                tuple(scale_arr.round(12).tolist()),
            )
            cached = self._candidate_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            prefix_key = (str(isotope), int(id(kernel)), source_key)
            prefix_payload = self._candidate_response_prefix_cache.get(prefix_key)
            if isinstance(prefix_payload, dict):
                cached_counts = np.asarray(
                    prefix_payload.get("counts", np.zeros((0, 0))),
                    dtype=float,
                )
                cached_rows = int(cached_counts.shape[0])
                if (
                    cached_rows >= measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        measurement_count,
                    )
                ):
                    counts_arr = cached_counts[:measurement_count].copy()
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr
                if (
                    cached_rows < measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        cached_rows,
                    )
                ):
                    suffix_counts = expected_counts_per_source(
                        kernel=kernel,
                        isotope=isotope,
                        detector_positions=np.asarray(
                            data.detector_positions,
                            dtype=float,
                        )[cached_rows:],
                        sources=source_arr,
                        strengths=strength_arr,
                        live_times=np.asarray(data.live_times, dtype=float)[
                            cached_rows:
                        ],
                        fe_indices=np.asarray(data.fe_indices, dtype=int)[cached_rows:],
                        pb_indices=np.asarray(data.pb_indices, dtype=int)[cached_rows:],
                        source_scale=scale_arr[cached_rows:],
                    )
                    counts_arr = np.vstack(
                        [
                            cached_counts,
                            np.asarray(suffix_counts, dtype=float),
                        ]
                    )
                    self._store_candidate_response_prefix_cache(
                        prefix_key,
                        data,
                        scale_arr,
                        counts_arr,
                    )
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr.copy()
        counts = expected_counts_per_source(
            kernel=kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=source_arr,
            strengths=strength_arr,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=scale_arr,
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._store_candidate_response_cache(cache_key, counts_arr)
            if prefix_key is not None:
                self._store_candidate_response_prefix_cache(
                    prefix_key,
                    data,
                    scale_arr,
                    counts_arr,
                )
        return counts_arr

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts using an active filter's physical kernel."""
        return self._cached_expected_counts_for_kernel(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            data=data,
            sources=sources,
            strengths=strengths,
        )

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
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
        )

    def _build_pf_config(self) -> PFConfig:
        """Build the per-isotope config from the shared PF field contract."""
        shared_values = {
            field.name: getattr(self.pf_config, field.name)
            for field in fields(PFConfig)
            if hasattr(self.pf_config, field.name)
        }
        return PFConfig(**shared_values)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        if not gpu_utils.torch_device_available(self.pf_config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether torch-backed estimator math is available."""
        from pf import gpu_utils

        return bool(
            self.pf_config.use_gpu
            and gpu_utils.torch_device_available(self.pf_config.gpu_device)
        )

    def response_scale_for_isotope(
        self,
        isotope: str,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the configured source response scale for one isotope."""
        pair_id = self._shield_pair_id(
            fe_index=fe_index,
            pb_index=pb_index,
            shield_pair_id=shield_pair_id,
        )
        pair_scales = self.pf_config.measurement_scale_by_isotope_and_pair
        if pair_id is not None and isinstance(pair_scales, Mapping):
            iso_pair_scales = pair_scales.get(str(isotope), {})
            if isinstance(iso_pair_scales, Mapping):
                value = iso_pair_scales.get(int(pair_id))
                if value is None:
                    value = iso_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.pf_config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(isotope, 1.0)), 0.0)

    def response_scales_for_measurements(
        self,
        isotope: str,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale per Fe/Pb measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self.response_scale_for_isotope(
                    isotope,
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _shield_pair_id(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> int | None:
        """Return the canonical shield-pair id when a pair is available."""
        if shield_pair_id is not None:
            return int(shield_pair_id)
        if fe_index is None or pb_index is None:
            return None
        return int(fe_index) * int(self.num_orientations) + int(pb_index)

    def _project_observation_covariance_to_variance(
        self,
        z_k: Mapping[str, float],
        z_variance_k: Mapping[str, float] | None,
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> tuple[Dict[str, float] | None, Dict[str, Dict[str, float]] | None]:
        """
        Return diagonal PF variances that conservatively cover isotope covariance.

        The per-isotope filters are conditionally independent, while the
        response-Poisson spectrum regression reports a same-spectrum covariance
        across isotope count channels.  This projection keeps the observed count
        means unchanged and uses a Gershgorin-style diagonal envelope,
        ``var_i + sum_j |cov_ij|``, so ignoring off-diagonal terms cannot make
        the independent isotope filters more confident than the structured
        covariance supports.
        """
        if z_variance_k is None and z_covariance_k is None:
            return None, None
        isotopes = [str(isotope) for isotope in z_k]
        if not isotopes:
            return {}, None
        base_variances = np.asarray(
            [
                max(
                    float(
                        z_variance_k.get(
                            isotope, max(float(z_k.get(isotope, 0.0)), 1.0)
                        )
                        if z_variance_k is not None
                        else max(float(z_k.get(isotope, 0.0)), 1.0)
                    ),
                    1.0,
                )
                for isotope in isotopes
            ],
            dtype=float,
        )
        if z_covariance_k is None or not bool(
            self.pf_config.observation_covariance_projection_enable
        ):
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                self._sanitize_observation_covariance(
                    isotopes,
                    base_variances,
                    z_covariance_k,
                ),
            )
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                None,
            )
        offdiag_abs = np.sum(np.abs(covariance), axis=1) - np.abs(np.diag(covariance))
        projected = np.maximum(
            base_variances,
            np.diag(covariance)
            + float(self.pf_config.observation_covariance_projection_weight)
            * offdiag_abs,
        )
        return (
            {
                isotope: float(variance)
                for isotope, variance in zip(isotopes, projected)
            },
            {
                row_iso: {
                    col_iso: float(covariance[row_idx, col_idx])
                    for col_idx, col_iso in enumerate(isotopes)
                }
                for row_idx, row_iso in enumerate(isotopes)
            },
        )

    def _observation_covariance_matrix(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]],
    ) -> NDArray[np.float64] | None:
        """Build a symmetric isotope covariance matrix for one spectrum."""
        covariance = np.diag(np.maximum(np.asarray(base_variances, dtype=float), 1.0))
        index_by_isotope = {str(isotope): idx for idx, isotope in enumerate(isotopes)}
        for row_iso, row_payload in z_covariance_k.items():
            if str(row_iso) not in index_by_isotope or not isinstance(
                row_payload,
                Mapping,
            ):
                continue
            row_idx = index_by_isotope[str(row_iso)]
            for col_iso, raw_value in row_payload.items():
                col_key = str(col_iso)
                if col_key not in index_by_isotope:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                col_idx = index_by_isotope[col_key]
                covariance[row_idx, col_idx] = value
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), base_variances))
        diag = np.maximum(np.diag(covariance), 1.0)
        corr_limit = float(self.pf_config.observation_covariance_projection_max_corr)
        if corr_limit < 1.0:
            scale = np.sqrt(diag[:, None] * diag[None, :])
            corr = np.divide(
                covariance,
                scale,
                out=np.zeros_like(covariance, dtype=float),
                where=scale > 0.0,
            )
            corr = np.clip(corr, -corr_limit, corr_limit)
            covariance = corr * scale
            np.fill_diagonal(covariance, diag)
        return covariance

    def _sanitize_observation_covariance(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> Dict[str, Dict[str, float]] | None:
        """Return a JSON-safe covariance payload when one was supplied."""
        if z_covariance_k is None:
            return None
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return None
        return {
            row_iso: {
                col_iso: float(covariance[row_idx, col_idx])
                for col_idx, col_iso in enumerate(isotopes)
            }
            for row_idx, row_iso in enumerate(isotopes)
        }

    @staticmethod
    def _sanitize_spectrum_vector(
        values: object,
        *,
        name: str,
        expected_size: int | None = None,
    ) -> tuple[float, ...] | None:
        """Return a finite non-negative spectrum vector payload."""
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        if expected_size is not None and arr.size != int(expected_size):
            raise ValueError(f"{name} must have {int(expected_size)} bins.")
        arr = np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0)
        return tuple(float(value) for value in arr)

    @staticmethod
    def _looks_like_spectrum_payload(payload: object) -> bool:
        """Return True when a mapping carries direct spectrum-bin fields."""
        if not isinstance(payload, Mapping):
            return False
        keys = {str(key) for key in payload.keys()}
        return bool(
            {
                "spectrum_counts",
                "spectrum_variance",
                "spectrum_background",
                "spectrum_response_templates_by_isotope",
            }
            & keys
        )

    @staticmethod
    def _sanitize_spectrum_payload(
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Return a normalized spectrum-bin payload for measurement history."""
        if payload is None:
            return None
        spectrum_counts = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_counts"),
            name="spectrum_counts",
        )
        if spectrum_counts is None:
            return None
        bin_count = len(spectrum_counts)
        spectrum_variance = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_variance"),
            name="spectrum_variance",
            expected_size=bin_count,
        )
        spectrum_background = RotatingShieldPFEstimator._sanitize_spectrum_vector(
            payload.get("spectrum_background"),
            name="spectrum_background",
            expected_size=bin_count,
        )
        template_payload = payload.get("spectrum_response_templates_by_isotope", {})
        templates: dict[str, tuple[float, ...]] = {}
        if isinstance(template_payload, Mapping):
            for isotope, values in template_payload.items():
                template = RotatingShieldPFEstimator._sanitize_spectrum_vector(
                    values,
                    name=f"spectrum_response_templates_by_isotope[{isotope}]",
                    expected_size=bin_count,
                )
                if template is not None:
                    templates[str(isotope)] = template
        return {
            "spectrum_counts": spectrum_counts,
            "spectrum_variance": spectrum_variance,
            "spectrum_background": spectrum_background,
            "spectrum_background_source": str(
                payload.get("spectrum_background_source", "unspecified")
            ),
            "spectrum_background_observation_independent": bool(
                payload.get(
                    "spectrum_background_observation_independent",
                    False,
                )
            ),
            "spectrum_response_templates_by_isotope": templates,
        }

    def register_configured_isotope_spectrum_responses(
        self,
        templates_by_isotope: Mapping[str, object],
    ) -> tuple[str, ...]:
        """Retain configured spectral responses independently of active PF filters.

        Entries are keyed by isotope and spectrum-bin count so an exact response is
        reused only for histories with the same binning.  Partial registration is
        allowed because response providers may populate configured isotopes in
        batches; unknown isotopes are rejected to prevent silent model mismatch.
        """
        configured = set(self.configured_isotope_order())
        registered: list[str] = []
        for isotope, values in templates_by_isotope.items():
            isotope_key = str(isotope)
            if isotope_key not in configured:
                raise ValueError(
                    f"Cannot register spectrum response for unconfigured isotope "
                    f"{isotope_key!r}."
                )
            response = self._sanitize_spectrum_vector(
                values,
                name=f"configured_spectrum_response[{isotope_key}]",
            )
            if response is None:
                continue
            self._configured_spectrum_response_registry[
                (isotope_key, len(response))
            ] = response
            registered.append(isotope_key)
        return tuple(registered)

    def configured_isotope_spectrum_response(
        self,
        isotope: str,
        *,
        bin_count: int,
    ) -> NDArray[np.float64] | None:
        """Return a copied configured spectral response for an exact binning."""
        isotope_key = str(isotope)
        if isotope_key not in self.configured_isotope_order():
            raise KeyError(f"Isotope {isotope_key!r} is not configured.")
        response = self._configured_spectrum_response_registry.get(
            (isotope_key, int(bin_count))
        )
        if response is None:
            return None
        return np.asarray(response, dtype=float).copy()

    def _complete_spectrum_payload_with_configured_responses(
        self,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Register exact templates and fill configured responses of equal binning."""
        if payload is None:
            return None
        counts = payload.get("spectrum_counts")
        if counts is None:
            return dict(payload)
        bin_count = int(np.asarray(counts, dtype=float).size)
        raw_templates = payload.get("spectrum_response_templates_by_isotope", {})
        templates = (
            {str(key): value for key, value in raw_templates.items()}
            if isinstance(raw_templates, Mapping)
            else {}
        )
        configured = set(self.configured_isotope_order())
        configured_templates = {
            isotope: values
            for isotope, values in templates.items()
            if isotope in configured
        }
        self.register_configured_isotope_spectrum_responses(configured_templates)
        completed = dict(templates)
        for isotope in self.configured_isotope_order():
            response = self.configured_isotope_spectrum_response(
                isotope,
                bin_count=bin_count,
            )
            if response is not None:
                completed[isotope] = tuple(float(value) for value in response)
        return {
            **dict(payload),
            "spectrum_response_templates_by_isotope": completed,
        }

    @staticmethod
    def _pf_spectrum_update_payload_for_isotope(
        isotope: str,
        z_k: Mapping[str, float],
        spectrum_payload: Mapping[str, object] | None,
    ) -> dict[str, NDArray[np.float64]] | None:
        """Return target-isotope spectrum arrays for a PF weight update."""
        if spectrum_payload is None:
            return None
        counts_raw = spectrum_payload.get("spectrum_counts")
        templates_raw = spectrum_payload.get("spectrum_response_templates_by_isotope")
        if counts_raw is None or not isinstance(templates_raw, Mapping):
            return None
        isotope_key = str(isotope)
        if isotope_key not in templates_raw:
            return None
        counts = np.asarray(counts_raw, dtype=float).reshape(-1)
        target_template = np.asarray(templates_raw[isotope_key], dtype=float).reshape(
            -1
        )
        if counts.size == 0 or target_template.size != counts.size:
            return None
        background_raw = spectrum_payload.get("spectrum_background")
        if background_raw is None:
            background = np.zeros_like(counts, dtype=float)
        else:
            background = np.asarray(background_raw, dtype=float).reshape(-1)
            if background.size != counts.size:
                background = np.zeros_like(counts, dtype=float)
        for other_isotope, other_template_raw in templates_raw.items():
            other_key = str(other_isotope)
            if other_key == isotope_key:
                continue
            other_template = np.asarray(other_template_raw, dtype=float).reshape(-1)
            if other_template.size != counts.size:
                continue
            other_counts = max(float(z_k.get(other_key, 0.0)), 0.0)
            if other_counts > 0.0:
                background = background + other_counts * np.maximum(
                    np.where(np.isfinite(other_template), other_template, 0.0),
                    0.0,
                )
        variance = None
        variance_raw = spectrum_payload.get("spectrum_variance")
        if variance_raw is not None:
            variance_candidate = np.asarray(variance_raw, dtype=float).reshape(-1)
            if variance_candidate.size == counts.size:
                variance = np.maximum(
                    np.where(np.isfinite(variance_candidate), variance_candidate, 0.0),
                    0.0,
                )
        payload = {
            "spectrum_counts": np.maximum(
                np.where(np.isfinite(counts), counts, 0.0),
                0.0,
            ),
            "spectrum_response_template": np.maximum(
                np.where(np.isfinite(target_template), target_template, 0.0),
                0.0,
            ),
            "spectrum_background": np.maximum(
                np.where(np.isfinite(background), background, 0.0),
                0.0,
            ),
        }
        if variance is not None:
            payload["spectrum_variance"] = variance
        return payload

    @staticmethod
    def _stack_pf_spectrum_sequence_payloads(
        payloads: Sequence[dict[str, NDArray[np.float64]] | None],
    ) -> dict[str, NDArray[np.float64]] | None:
        """Stack per-view PF spectrum payloads into KxB arrays."""
        if not payloads or any(payload is None for payload in payloads):
            return None
        concrete = [payload for payload in payloads if payload is not None]
        if not concrete:
            return None
        bin_count = int(concrete[0]["spectrum_counts"].reshape(-1).size)
        if bin_count <= 0:
            return None
        required_keys = (
            "spectrum_counts",
            "spectrum_response_template",
            "spectrum_background",
        )
        stacked: dict[str, NDArray[np.float64]] = {}
        for key in required_keys:
            rows = [
                np.asarray(payload[key], dtype=float).reshape(-1)
                for payload in concrete
            ]
            if any(row.size != bin_count for row in rows):
                return None
            stacked[key] = np.vstack(rows)
        if any("spectrum_variance" in payload for payload in concrete):
            variance_rows = [
                (
                    np.zeros(bin_count, dtype=float)
                    if "spectrum_variance" not in payload
                    else np.asarray(
                        payload["spectrum_variance"],
                        dtype=float,
                    ).reshape(-1)
                )
                for payload in concrete
            ]
            if any(row.size != bin_count for row in variance_rows):
                return None
            stacked["spectrum_variance"] = np.vstack(variance_rows)
        return stacked

    @staticmethod
    def _recorded_runtime_likelihood_route(
        filt: IsotopeParticleFilter,
    ) -> str:
        """Return the validated likelihood route selected by a runtime update."""
        route = str(filt.last_spectrum_likelihood_route)
        if route not in {"count", "count_covariance", "direct_spectrum"}:
            return "count"
        return route

    def _runtime_likelihood_routes_for_records(
        self,
        isotope: str,
        records: Sequence[MeasurementRecord],
        spectrum_payloads: Sequence[dict[str, NDArray[np.float64]] | None],
    ) -> NDArray[np.str_]:
        """Return the exact per-row runtime likelihood route for one isotope."""
        if len(records) != len(spectrum_payloads):
            raise ValueError("Records and spectrum payloads must have matching length.")
        filters = getattr(self, "filters", {})
        filt = filters.get(str(isotope)) if isinstance(filters, Mapping) else None
        pf_config = getattr(self, "pf_config", None)
        if filt is not None:
            direct_enabled = filt._direct_spectrum_likelihood_enabled()
        elif pf_config is None:
            direct_enabled = True
        else:
            direct_enabled = (
                IsotopeParticleFilter._direct_spectrum_likelihood_config_enabled(
                    pf_config,
                    str(isotope),
                )
            )
        routes: list[str] = []
        explicit_routes: list[bool] = []
        for record, spectrum_payload in zip(records, spectrum_payloads):
            explicit = record.runtime_likelihood_route_by_isotope
            route = None if explicit is None else explicit.get(str(isotope))
            explicit_routes.append(route is not None)
            if route is None:
                route = (
                    "direct_spectrum"
                    if direct_enabled and spectrum_payload is not None
                    else "count"
                )
            normalized = str(route)
            if normalized not in {
                "count",
                "count_covariance",
                "direct_spectrum",
            }:
                raise ValueError(
                    f"Unsupported recorded runtime likelihood route: {normalized!r}."
                )
            routes.append(normalized)
        sequence_ids = self._station_sequence_ids_for_records(records)
        route_array = np.asarray(routes, dtype="<U16")
        explicit_array = np.asarray(explicit_routes, dtype=bool)
        likelihood_config = getattr(filt, "config", pf_config)
        configured_station_covariance = bool(
            getattr(
                likelihood_config,
                "station_view_covariance_enable",
                False,
            )
        ) and float(
            getattr(
                likelihood_config,
                "station_view_correlated_spectrum_fraction",
                0.0,
            )
        ) > 0.0
        for sequence_id in np.unique(sequence_ids):
            block_mask = sequence_ids == int(sequence_id)
            if np.any(explicit_array[block_mask]) or np.count_nonzero(block_mask) < 2:
                continue
            supplied_station_covariance = any(
                isinstance(
                    records[int(index)].station_view_covariance_by_isotope,
                    Mapping,
                )
                and str(isotope)
                in records[int(index)].station_view_covariance_by_isotope
                for index in np.flatnonzero(block_mask)
            )
            if supplied_station_covariance or configured_station_covariance:
                route_array[block_mask] = "count_covariance"
        return route_array

    @staticmethod
    def _runtime_spectrum_variance_usage_for_records(
        isotope: str,
        records: Sequence[MeasurementRecord],
        spectrum_payloads: Sequence[dict[str, NDArray[np.float64]] | None],
        runtime_routes: NDArray[np.str_],
    ) -> NDArray[np.bool_]:
        """
        Return whether each runtime direct-spectrum row used a variance array.

        A joint station update supplies one stacked variance array to every row
        when any row in that station has a variance payload. Independent
        updates retain their row-local ``None`` semantics. New records store
        the exact runtime choice; station grouping provides a compatible
        fallback for legacy or directly constructed records.
        """
        routes = np.asarray(runtime_routes, dtype=str).reshape(-1)
        if len(records) != len(spectrum_payloads) or routes.size != len(records):
            raise ValueError(
                "Records, spectrum payloads, and routes must have matching length."
            )
        if not records:
            return np.zeros(0, dtype=bool)
        sequence_ids = RotatingShieldPFEstimator._station_sequence_ids_for_records(
            records
        )
        result = np.zeros(len(records), dtype=bool)
        for sequence_id in np.unique(sequence_ids):
            block_mask = sequence_ids == int(sequence_id)
            block_routes = routes[block_mask]
            if np.any(block_routes != block_routes[:1]):
                raise ValueError(
                    "Rows in one station sequence must share one runtime "
                    "likelihood route."
                )
            if block_routes[0] != "direct_spectrum":
                continue
            block_indices = np.flatnonzero(block_mask)
            explicit_values = [
                bool(explicit[str(isotope)])
                for index in block_indices
                if (
                    (explicit := records[int(index)].runtime_spectrum_variance_used_by_isotope)
                    is not None
                    and str(isotope) in explicit
                )
            ]
            if explicit_values:
                if any(value != explicit_values[0] for value in explicit_values):
                    raise ValueError(
                        "Rows in one station sequence recorded inconsistent "
                        "spectrum-variance usage."
                    )
                variance_used = bool(explicit_values[0])
            else:
                variance_used = any(
                    spectrum_payloads[int(index)] is not None
                    and "spectrum_variance" in spectrum_payloads[int(index)]
                    for index in block_indices
                )
            result[block_mask] = variance_used
        return result

    @staticmethod
    def _stack_pf_spectrum_history_payloads(
        payloads: Sequence[dict[str, NDArray[np.float64]] | None],
        runtime_routes: NDArray[np.str_],
        spectrum_variance_used: NDArray[np.bool_],
    ) -> dict[str, NDArray[np.float64]] | None:
        """
        Stack only rows that actually used the direct-spectrum runtime route.

        Count-route rows receive zero placeholders and are excluded by their
        explicit route before likelihood evaluation. Spectrum math remains
        batched; the metadata pass only resolves optional row payloads.
        """
        routes = np.asarray(runtime_routes, dtype=str).reshape(-1)
        if routes.size != len(payloads):
            raise ValueError(
                "runtime_likelihood_routes must contain one route per payload."
            )
        variance_used = np.asarray(spectrum_variance_used, dtype=bool).reshape(-1)
        if variance_used.size != len(payloads):
            raise ValueError(
                "spectrum_variance_used must contain one flag per payload."
            )
        direct_mask = routes == "direct_spectrum"
        if not np.any(direct_mask):
            return None
        direct_indices = np.flatnonzero(direct_mask)
        direct_payloads = [payloads[int(index)] for index in direct_indices]
        if any(payload is None for payload in direct_payloads):
            raise ValueError(
                "A recorded direct-spectrum route is missing its spectrum payload."
            )
        stacked_direct = (
            RotatingShieldPFEstimator._stack_pf_spectrum_sequence_payloads(
                direct_payloads
            )
        )
        if stacked_direct is None:
            raise ValueError("Recorded direct-spectrum payload rows are inconsistent.")
        if (
            np.any(variance_used[direct_indices])
            and "spectrum_variance" not in stacked_direct
        ):
            stacked_direct["spectrum_variance"] = np.zeros_like(
                stacked_direct["spectrum_counts"],
                dtype=float,
            )
        bin_count = int(stacked_direct["spectrum_counts"].shape[1])
        stacked_history: dict[str, NDArray[np.float64]] = {}
        for key, direct_values in stacked_direct.items():
            history_values = np.zeros(
                (len(payloads), bin_count),
                dtype=float,
            )
            history_values[direct_indices, :] = np.asarray(
                direct_values,
                dtype=float,
            )
            stacked_history[key] = history_values
        return stacked_history

    @staticmethod
    def _normalize_pair_sequence_record(
        record: Sequence[object],
    ) -> tuple[
        Dict[str, float],
        int,
        int,
        float,
        Dict[str, float] | None,
        Dict[str, Dict[str, float]] | None,
        dict[str, object] | None,
    ]:
        """Return a canonical same-pose shield-program observation record."""
        spectrum_payload = None
        if len(record) == 5:
            z_k, fe_index, pb_index, live_time_s, z_variance_k = record
            z_covariance_k = None
        elif len(record) == 6:
            z_k, fe_index, pb_index, live_time_s, z_variance_k, sixth = record
            if RotatingShieldPFEstimator._looks_like_spectrum_payload(sixth):
                z_covariance_k = None
                spectrum_payload = sixth
            else:
                z_covariance_k = sixth
        elif len(record) == 7:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
                spectrum_payload,
            ) = record
        else:
            raise ValueError(
                "Pair sequence records must have 5 fields "
                "(z, fe, pb, live, variance), 6 fields with covariance or "
                "spectrum payload, or 7 fields with both."
            )
        return (
            {str(isotope): float(value) for isotope, value in dict(z_k).items()},
            int(fe_index),
            int(pb_index),
            float(live_time_s),
            None
            if z_variance_k is None
            else {
                str(isotope): float(value)
                for isotope, value in dict(z_variance_k).items()
            },
            None
            if z_covariance_k is None
            else {
                str(row_iso): {
                    str(col_iso): float(value)
                    for col_iso, value in dict(row_payload).items()
                }
                for row_iso, row_payload in dict(z_covariance_k).items()
            },
            RotatingShieldPFEstimator._sanitize_spectrum_payload(
                spectrum_payload if isinstance(spectrum_payload, Mapping) else None
            ),
        )

    @staticmethod
    def _view_covariance_for_isotope(
        isotope: str,
        *,
        sequence_length: int,
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None,
    ) -> NDArray[np.float64] | None:
        """Return a same-station shield-view covariance matrix for one isotope."""
        if z_view_covariance_by_isotope is None:
            return None
        payload = z_view_covariance_by_isotope.get(str(isotope))
        if payload is None:
            return None
        covariance = np.asarray(payload, dtype=float)
        expected_shape = (int(sequence_length), int(sequence_length))
        if covariance.shape != expected_shape:
            raise ValueError(
                "z_view_covariance_by_isotope entries must be shaped K x K."
            )
        return 0.5 * (covariance + covariance.T)

    def continuous_kernel(
        self,
        *,
        detector_aperture_samples: int | None = None,
        use_gpu: bool | None = None,
    ) -> ContinuousKernel:
        """Build the shared ContinuousKernel for PF, planning, and diagnostics."""
        active_aperture_samples = (
            int(self.detector_aperture_samples)
            if detector_aperture_samples is None
            else int(detector_aperture_samples)
        )
        active_use_gpu = (
            bool(self.pf_config.use_gpu) if use_gpu is None else bool(use_gpu)
        )
        return ContinuousKernel(
            mu_by_isotope=self.mu_by_isotope,
            shield_params=self.shield_params,
            orientations=self.normals,
            use_gpu=active_use_gpu,
            gpu_device=str(self.pf_config.gpu_device),
            gpu_dtype=str(self.pf_config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=max(1, active_aperture_samples),
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
        )

    def configured_isotope_order(self) -> tuple[str, ...]:
        """Return the stable configured isotope order, including inactive PFs."""
        return tuple(dict.fromkeys(str(isotope) for isotope in self.all_isotopes))

    def configured_isotope_response_kernel(self, isotope: str) -> ContinuousKernel:
        """Return a shared physical response kernel without creating a PF filter."""
        isotope_key = str(isotope)
        configured = self.configured_isotope_order()
        if isotope_key not in configured:
            raise KeyError(f"Isotope {isotope_key!r} is not configured.")
        kernel = self._configured_response_kernel_registry.get(isotope_key)
        if kernel is not None:
            return kernel
        shared_kernel = next(
            iter(self._configured_response_kernel_registry.values()),
            None,
        )
        if shared_kernel is None:
            shared_kernel = self.continuous_kernel()
        for configured_isotope in configured:
            self._configured_response_kernel_registry.setdefault(
                configured_isotope,
                shared_kernel,
            )
        return self._configured_response_kernel_registry[isotope_key]

    def configured_isotope_response_counts(
        self,
        isotope: str,
        data: MeasurementData,
        source_positions: NDArray[np.float64],
        strengths: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return batched configured-isotope responses independent of PF state.

        Measurement rows and source positions are evaluated by the same continuous
        transport, obstacle, aperture, shield, and calibrated response-scale model
        used by active PF filters.  Candidate positions remain batched; no particle
        state or active-isotope gate is read.
        """
        positions = np.asarray(source_positions, dtype=float).reshape(-1, 3)
        if strengths is None:
            strength_values = np.ones(positions.shape[0], dtype=float)
        else:
            strength_values = np.asarray(strengths, dtype=float).reshape(-1)
        if strength_values.size != positions.shape[0]:
            raise ValueError("strengths must contain one value per source position.")
        return self._cached_expected_counts_for_kernel(
            kernel=self.configured_isotope_response_kernel(str(isotope)),
            isotope=str(isotope),
            data=data,
            sources=positions,
            strengths=strength_values,
        )

    def configured_isotope_measurement_history(
        self,
        isotope: str,
        *,
        window: int | None = None,
    ) -> MeasurementData | None:
        """Return count/geometry history for a configured, possibly inactive isotope."""
        isotope_key = str(isotope)
        if isotope_key not in self.configured_isotope_order():
            raise KeyError(f"Isotope {isotope_key!r} is not configured.")
        return self._measurement_data_for_iso(isotope_key, window)

    def _continuous_kernel(self) -> ContinuousKernel:
        """Build a ContinuousKernel matching the estimator observation model."""
        return self.continuous_kernel()

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
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            values = np.zeros(len(states), dtype=float)
            source_scale = self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            )
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
        lam_t = kernel.expected_counts_pair_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
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
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
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
        num_orients = int(self.num_orientations)
        fe_indices = np.repeat(np.arange(num_orients), num_orients)
        pb_indices = np.tile(np.arange(num_orients), num_orients)
        lam_t = kernel.expected_counts_all_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            live_time_s=live_time_s,
            source_scale=self.response_scales_for_measurements(
                isotope,
                fe_indices,
                pb_indices,
            ),
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy().astype(float, copy=False)

    def expected_counts_all_pairs_for_states_at_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """Compute all-pair counts for many detector positions in one batch.

        The result is shaped ``(detector, pair, state)``.  Particle source slots
        are packed once and the shared continuous kernel evaluates every
        detector/source/pair response together.  Callers should pass bounded
        detector chunks when many planning poses are under consideration.
        """
        detectors = np.asarray(detector_positions, dtype=float)
        if detectors.size == 0:
            detectors = np.zeros((0, 3), dtype=float)
        if detectors.ndim != 2 or detectors.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (D, 3).")
        num_pairs = int(self.num_orientations) * int(self.num_orientations)
        state_count = len(states)
        if state_count == 0:
            return np.zeros((detectors.shape[0], num_pairs, 0), dtype=float)

        max_sources = max(
            (max(0, int(state.num_sources)) for state in states), default=0
        )
        backgrounds = np.asarray(
            [max(float(state.background), 0.0) for state in states],
            dtype=float,
        )
        source_rates = np.zeros(
            (detectors.shape[0], num_pairs, state_count),
            dtype=float,
        )
        if max_sources > 0 and detectors.shape[0] > 0:
            positions = np.zeros((state_count, max_sources, 3), dtype=float)
            strengths = np.zeros((state_count, max_sources), dtype=float)
            for state_index, state in enumerate(states):
                source_count = min(max(0, int(state.num_sources)), max_sources)
                if source_count <= 0:
                    continue
                positions[state_index, :source_count, :] = np.asarray(
                    state.positions[:source_count],
                    dtype=float,
                )
                strengths[state_index, :source_count] = np.maximum(
                    np.asarray(state.strengths[:source_count], dtype=float),
                    0.0,
                )
            kernel_values = (
                self._continuous_kernel().kernel_values_all_pairs_for_detectors(
                    isotope=isotope,
                    detector_positions=detectors,
                    sources=positions.reshape(-1, 3),
                )
            )
            expected_shape = (
                detectors.shape[0],
                num_pairs,
                state_count * max_sources,
            )
            if kernel_values.shape != expected_shape:
                raise RuntimeError(
                    "Batched all-pair kernel returned an unexpected shape: "
                    f"{kernel_values.shape} != {expected_shape}."
                )
            source_rates = np.einsum(
                "daps,ps->dap",
                kernel_values.reshape(
                    detectors.shape[0],
                    num_pairs,
                    state_count,
                    max_sources,
                ),
                strengths,
                optimize=True,
            )

        num_orients = int(self.num_orientations)
        fe_indices = np.repeat(np.arange(num_orients), num_orients)
        pb_indices = np.tile(np.arange(num_orients), num_orients)
        source_scales = self.response_scales_for_measurements(
            isotope,
            fe_indices,
            pb_indices,
        )
        rates = backgrounds[None, None, :] + source_scales[None, :, None] * np.maximum(
            source_rates, 0.0
        )
        return np.maximum(float(live_time_s) * rates, 0.0)

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
        z_covariance_k: Dict[str, Dict[str, float]] | None = None,
        spectrum_payload: Mapping[str, object] | None = None,
    ) -> None:
        """
        Update PFs using Fe/Pb orientation indices (RFe, RPb) and isotope-wise counts z_k.

        Configured isotopes omitted from ``z_k`` are observed as zero, matching
        joint-sequence and structural-history semantics. This feeds the
        continuous 3D PF path with expected counts from the shield pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        effective_variance_k, sanitized_covariance_k = (
            self._project_observation_covariance_to_variance(
                z_k,
                z_variance_k,
                z_covariance_k,
            )
        )
        sanitized_spectrum_payload = None
        if spectrum_payload is not None:
            sanitized_spectrum_payload = (
                self._complete_spectrum_payload_with_configured_responses(
                    self._sanitize_spectrum_payload(spectrum_payload)
                )
            )
        runtime_likelihood_routes: dict[str, str] = {}
        runtime_spectrum_variance_used: dict[str, bool] = {}
        for iso, filt in self.filters.items():
            val = float(z_k.get(iso, 0.0))
            pf_spectrum_payload = (
                None
                if sanitized_spectrum_payload is None
                else self._pf_spectrum_update_payload_for_isotope(
                    iso,
                    z_k,
                    sanitized_spectrum_payload,
                )
            )
            # Use continuous PF update that relies on spectrum-unfolded counts.
            filt.update_continuous_pair(
                z_obs=val,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0
                    if effective_variance_k is None
                    else float(effective_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
                defer_resample=bool(self._defer_resample_birth),
                **({} if pf_spectrum_payload is None else pf_spectrum_payload),
            )
            runtime_likelihood_routes[str(iso)] = (
                self._recorded_runtime_likelihood_route(filt)
            )
            runtime_spectrum_variance_used[str(iso)] = bool(
                runtime_likelihood_routes[str(iso)] == "direct_spectrum"
                and pf_spectrum_payload is not None
                and "spectrum_variance" in pf_spectrum_payload
            )
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k={
                    str(iso): (
                        0.0
                        if effective_variance_k is None
                        else float(effective_variance_k.get(iso, 0.0))
                    )
                    for iso in self.filters
                },
                z_covariance_k=sanitized_covariance_k,
                ig_value=None,
                spectrum_counts=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_counts")
                ),
                spectrum_variance=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_variance")
                ),
                spectrum_background=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get("spectrum_background")
                ),
                spectrum_background_source=(
                    None
                    if sanitized_spectrum_payload is None
                    else str(
                        sanitized_spectrum_payload.get(
                            "spectrum_background_source",
                            "unspecified",
                        )
                    )
                ),
                spectrum_background_observation_independent=(
                    False
                    if sanitized_spectrum_payload is None
                    else bool(
                        sanitized_spectrum_payload.get(
                            "spectrum_background_observation_independent",
                            False,
                        )
                    )
                ),
                spectrum_response_templates_by_isotope=(
                    None
                    if sanitized_spectrum_payload is None
                    else sanitized_spectrum_payload.get(
                        "spectrum_response_templates_by_isotope"
                    )
                ),
                station_sequence_id=int(len(self.measurements)),
                station_view_index=0,
                runtime_likelihood_route_by_isotope=runtime_likelihood_routes,
                runtime_spectrum_variance_used_by_isotope=(
                    runtime_spectrum_variance_used
                ),
            )
        )
        if self._defer_resample_birth:
            self._deferred_measurement_count += 1
        else:
            self._apply_structural_moves()
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
        for filt in self.filters.values():
            filt.finalize_deferred_update()
        self._apply_structural_moves()
        self._record_history_estimate(len(self.measurements))
        return count

    def update_pair_sequence(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """
        Jointly update PFs from a same-pose shield-orientation sequence.

        Each record is ``(z_k, fe_index, pb_index, live_time_s, z_variance_k)``.
        A sixth ``z_covariance_k`` field may be supplied for same-spectrum
        isotope covariance.
        A seventh spectrum payload field may be supplied for the direct
        spectrum-bin PF likelihood.
        ``z_view_covariance_by_isotope`` may also supply KxK same-station
        shield-view covariance for each isotope. The joint update uses one
        station-level likelihood over all postures and only applies birth/death
        after the full shield program is observed.
        """
        if not records:
            return
        sequence_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        stage_start = sequence_start
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        normalized_records = []
        for record in records:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
                spectrum_payload,
            ) = self._normalize_pair_sequence_record(record)
            if spectrum_payload is not None:
                spectrum_payload = (
                    self._complete_spectrum_payload_with_configured_responses(
                        spectrum_payload
                    )
                )
            effective_variance_k, sanitized_covariance_k = (
                self._project_observation_covariance_to_variance(
                    z_k,
                    z_variance_k,
                    z_covariance_k,
                )
            )
            normalized_records.append(
                (
                    z_k,
                    fe_index,
                    pb_index,
                    live_time_s,
                    effective_variance_k,
                    sanitized_covariance_k,
                    spectrum_payload,
                )
            )
        stage_wall["normalize_records"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
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
                NDArray[np.float64] | None,
                int,
                int,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
                NDArray[np.float64] | None,
            ]
        ] = []
        for iso, filt in self.filters.items():
            z_arr = np.asarray(
                [
                    float(z_k.get(iso, 0.0))
                    for z_k, _, _, _, _, _, _ in normalized_records
                ],
                dtype=float,
            )
            var_arr = np.asarray(
                [
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                    for _, _, _, _, z_variance_k, _, _ in normalized_records
                ],
                dtype=float,
            )
            fe_arr = np.asarray(
                [int(fe_index) for _, fe_index, _, _, _, _, _ in normalized_records],
                dtype=int,
            )
            pb_arr = np.asarray(
                [int(pb_index) for _, _, pb_index, _, _, _, _ in normalized_records],
                dtype=int,
            )
            live_arr = np.asarray(
                [
                    float(live_time_s)
                    for _, _, _, live_time_s, _, _, _ in normalized_records
                ],
                dtype=float,
            )
            view_covariance = self._view_covariance_for_isotope(
                iso,
                sequence_length=z_arr.size,
                z_view_covariance_by_isotope=z_view_covariance_by_isotope,
            )
            sequence_spectrum_payload = None
            if any(record[6] is not None for record in normalized_records):
                sequence_spectrum_payload = self._stack_pf_spectrum_sequence_payloads(
                    [
                        self._pf_spectrum_update_payload_for_isotope(
                            iso,
                            z_k,
                            spectrum_payload,
                        )
                        for (
                            z_k,
                            _fe_index,
                            _pb_index,
                            _live_time_s,
                            _z_variance_k,
                            _z_covariance_k,
                            spectrum_payload,
                        ) in normalized_records
                    ]
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
                    view_covariance,
                    int(pose_idx),
                    int(step_idx),
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_counts"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_response_template"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload["spectrum_background"],
                    None
                    if sequence_spectrum_payload is None
                    else sequence_spectrum_payload.get("spectrum_variance"),
                )
            )
        stage_wall["build_isotope_tasks"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_pair_sequence_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_pair_sequence_update(task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(self._run_isotope_pair_sequence_update, tasks))
        self.last_pair_sequence_update_wall_s = time.perf_counter() - stage_start
        stage_wall["isotope_sequence_update"] = self.last_pair_sequence_update_wall_s
        stage_start = time.perf_counter()
        runtime_likelihood_routes = {
            str(isotope): self._recorded_runtime_likelihood_route(filt)
            for isotope, filt in self.filters.items()
        }
        runtime_spectrum_variance_used = {
            str(task[0]): bool(
                runtime_likelihood_routes.get(str(task[0])) == "direct_spectrum"
                and task[13] is not None
            )
            for task in tasks
        }
        sequence_view_covariance = {
            str(task[0]): tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(task[7], dtype=float)
            )
            for task in tasks
            if task[7] is not None
        }
        for view_index, normalized_record in enumerate(normalized_records):
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
                spectrum_payload,
            ) = normalized_record
            self.measurements.append(
                MeasurementRecord(
                    z_k={iso: float(v) for iso, v in z_k.items()},
                    pose_idx=pose_idx,
                    orient_idx=int(fe_index),
                    live_time_s=float(live_time_s),
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                    z_variance_k={
                        str(iso): (
                            0.0
                            if z_variance_k is None
                            else float(z_variance_k.get(iso, 0.0))
                        )
                        for iso in self.filters
                    },
                    z_covariance_k=z_covariance_k,
                    ig_value=None,
                    spectrum_counts=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_counts")
                    ),
                    spectrum_variance=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_variance")
                    ),
                    spectrum_background=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get("spectrum_background")
                    ),
                    spectrum_background_source=(
                        None
                        if spectrum_payload is None
                        else str(
                            spectrum_payload.get(
                                "spectrum_background_source",
                                "unspecified",
                            )
                        )
                    ),
                    spectrum_background_observation_independent=(
                        False
                        if spectrum_payload is None
                        else bool(
                            spectrum_payload.get(
                                "spectrum_background_observation_independent",
                                False,
                            )
                        )
                    ),
                    spectrum_response_templates_by_isotope=(
                        None
                        if spectrum_payload is None
                        else spectrum_payload.get(
                            "spectrum_response_templates_by_isotope"
                        )
                    ),
                    station_sequence_id=int(step_idx),
                    station_view_index=int(view_index),
                    runtime_likelihood_route_by_isotope=runtime_likelihood_routes,
                    runtime_spectrum_variance_used_by_isotope=(
                        runtime_spectrum_variance_used
                    ),
                    station_view_covariance_by_isotope=(
                        None
                        if not sequence_view_covariance
                        else dict(sequence_view_covariance)
                    ),
                )
            )
        stage_wall["append_measurements"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._apply_structural_moves()
        stage_wall["structural_moves"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._record_history_estimate(len(self.measurements))
        stage_wall["history_estimate"] = time.perf_counter() - stage_start
        stage_wall["total"] = time.perf_counter() - sequence_start
        self.last_pair_sequence_stage_wall_s = {
            key: float(value) for key, value in stage_wall.items()
        }

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
            NDArray[np.float64] | None,
            int,
            int,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
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
            view_covariance,
            pose_idx,
            step_idx,
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
        ) = task
        filt.update_continuous_pair_sequence(
            z_obs=z_arr,
            pose_idx=pose_idx,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_times_s=live_arr,
            observation_count_variances=var_arr,
            observation_count_covariance=view_covariance,
            step_idx=step_idx,
            spectrum_counts=spectrum_counts,
            spectrum_response_template=spectrum_response_template,
            spectrum_background=spectrum_background,
            spectrum_variance=spectrum_variance,
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
        z_covariance_k: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        """
        Update PFs using explicit detector position without rebuilding the kernel cache.

        Configured isotopes omitted from ``z_k`` are observed as zero. This
        avoids kernel-cache growth with many poses by using per-pose updates.
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
                    detector_aperture_radius_m=self.detector_aperture_radius_m,
                    detector_aperture_samples=self.detector_aperture_samples,
                    detector_aperture_sampling=self.detector_aperture_sampling,
                    source_extent_radius_m=self.source_extent_radius_m,
                    source_extent_samples=self.source_extent_samples,
                    line_mu_by_isotope=self.line_mu_by_isotope,
                    transport_response_model=self.transport_response_model,
                )
        effective_variance_k, sanitized_covariance_k = (
            self._project_observation_covariance_to_variance(
                z_k,
                z_variance_k,
                z_covariance_k,
            )
        )
        runtime_likelihood_routes: dict[str, str] = {}
        for iso, filt in self.filters.items():
            val = float(z_k.get(iso, 0.0))
            filt.update_continuous_pair_at_pose(
                z_obs=val,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0
                    if effective_variance_k is None
                    else float(effective_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
            )
            runtime_likelihood_routes[str(iso)] = (
                self._recorded_runtime_likelihood_route(filt)
            )
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k={
                    str(iso): (
                        0.0
                        if effective_variance_k is None
                        else float(effective_variance_k.get(iso, 0.0))
                    )
                    for iso in self.filters
                },
                z_covariance_k=sanitized_covariance_k,
                ig_value=None,
                detector_position_xyz_m=tuple(float(value) for value in detector_pos),
                station_sequence_id=int(len(self.measurements)),
                station_view_index=0,
                runtime_likelihood_route_by_isotope=runtime_likelihood_routes,
                runtime_spectrum_variance_used_by_isotope={
                    str(iso): False for iso in runtime_likelihood_routes
                },
            )
        )
        self._apply_structural_moves()
        self._record_history_estimate(len(self.measurements))

    @staticmethod
    def _station_view_covariance_for_records(
        isotope: str,
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.float64] | None:
        """Rebuild supplied station-view covariance for selected history rows."""
        grouped_rows: dict[
            int,
            list[
                tuple[
                    int,
                    int,
                    NDArray[np.float64],
                ]
            ],
        ] = {}
        for row_index, record in enumerate(records):
            if (
                record.station_sequence_id is None
                or record.station_view_index is None
                or record.station_view_covariance_by_isotope is None
            ):
                continue
            covariance_payload = record.station_view_covariance_by_isotope.get(
                str(isotope)
            )
            if covariance_payload is None:
                continue
            covariance = np.asarray(covariance_payload, dtype=float)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError(
                    "Stored station-view covariance must be a square matrix."
                )
            view_index = int(record.station_view_index)
            if view_index < 0 or view_index >= covariance.shape[0]:
                raise ValueError(
                    "Stored station-view index lies outside its covariance matrix."
                )
            grouped_rows.setdefault(int(record.station_sequence_id), []).append(
                (int(row_index), view_index, covariance)
            )
        if not grouped_rows:
            return None
        combined = np.zeros((len(records), len(records)), dtype=float)
        for rows in grouped_rows.values():
            reference = rows[0][2]
            if any(
                covariance.shape != reference.shape
                or not np.array_equal(covariance, reference)
                for _, _, covariance in rows[1:]
            ):
                raise ValueError(
                    "Stored station-view covariance changed within one sequence."
                )
            history_indices = np.fromiter(
                (row_index for row_index, _, _ in rows),
                dtype=np.int64,
                count=len(rows),
            )
            view_indices = np.fromiter(
                (view_index for _, view_index, _ in rows),
                dtype=np.int64,
                count=len(rows),
            )
            combined[np.ix_(history_indices, history_indices)] = reference[
                np.ix_(view_indices, view_indices)
            ]
        return combined

    @staticmethod
    def _station_sequence_ids_for_records(
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.int64]:
        """
        Return an explicit runtime-likelihood block ID for every history row.

        Joint sequence records retain their shared ID. Legacy records without
        an ID are assigned distinct synthetic blocks, matching independent
        per-row runtime updates instead of inferring a block from coordinates.
        """
        explicit_ids = [
            int(record.station_sequence_id)
            for record in records
            if record.station_sequence_id is not None
        ]
        next_synthetic_id = max(explicit_ids, default=-1) + 1
        sequence_ids = np.empty(len(records), dtype=np.int64)
        for row_index, record in enumerate(records):
            if record.station_sequence_id is None:
                sequence_ids[row_index] = int(next_synthetic_id)
                next_synthetic_id += 1
            else:
                sequence_ids[row_index] = int(record.station_sequence_id)
        return sequence_ids

    @staticmethod
    def _record_spectrum_payload(
        record: MeasurementRecord,
    ) -> dict[str, object] | None:
        """Return the stored spectrum payload needed by a PF likelihood."""
        if (
            record.spectrum_counts is None
            or record.spectrum_response_templates_by_isotope is None
        ):
            return None
        return {
            "spectrum_counts": record.spectrum_counts,
            "spectrum_variance": record.spectrum_variance,
            "spectrum_background": record.spectrum_background,
            "spectrum_response_templates_by_isotope": (
                record.spectrum_response_templates_by_isotope
            ),
        }

    def _measurement_data_for_iso(
        self,
        isotope: str,
        window: int | None,
        records: Sequence[MeasurementRecord] | None = None,
    ) -> MeasurementData | None:
        """Build measurement arrays for a single isotope with an optional window."""
        if records is None and not self.measurements:
            return None
        if records is not None:
            selected_records = list(records)
        elif window is None or window <= 0:
            selected_records = self.measurements
        else:
            selected_records = self.measurements[-int(window) :]
        if not selected_records:
            return None
        spectrum_payloads = [
            self._pf_spectrum_update_payload_for_isotope(
                isotope,
                record.z_k,
                self._record_spectrum_payload(record),
            )
            for record in selected_records
        ]
        runtime_likelihood_routes = self._runtime_likelihood_routes_for_records(
            isotope,
            selected_records,
            spectrum_payloads,
        )
        spectrum_variance_present = (
            self._runtime_spectrum_variance_usage_for_records(
                isotope,
                selected_records,
                spectrum_payloads,
                runtime_likelihood_routes,
            )
        )
        spectrum_payload = self._stack_pf_spectrum_history_payloads(
            spectrum_payloads,
            runtime_likelihood_routes,
            spectrum_variance_present,
        )
        view_covariance = self._station_view_covariance_for_records(
            isotope,
            selected_records,
        )
        station_sequence_ids = self._station_sequence_ids_for_records(
            selected_records
        )
        z_list = []
        poses = []
        fe_indices = []
        pb_indices = []
        live_times = []
        variance_list = []
        for rec in selected_records:
            z_list.append(float(rec.z_k.get(isotope, 0.0)))
            if rec.z_variance_k is None:
                variance_list.append(0.0)
            else:
                variance_value = float(rec.z_variance_k.get(isotope, 0.0))
                variance_list.append(
                    max(variance_value, 0.0)
                    if np.isfinite(variance_value)
                    else 0.0
                )
            poses.append(
                self.poses[rec.pose_idx]
                if rec.detector_position_xyz_m is None
                else rec.detector_position_xyz_m
            )
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
            station_sequence_ids=station_sequence_ids,
            runtime_likelihood_routes=runtime_likelihood_routes,
            observation_count_covariance=view_covariance,
            spectrum_counts=(
                None
                if spectrum_payload is None
                else spectrum_payload["spectrum_counts"]
            ),
            spectrum_response_template=(
                None
                if spectrum_payload is None
                else spectrum_payload["spectrum_response_template"]
            ),
            spectrum_background=(
                None
                if spectrum_payload is None
                else spectrum_payload["spectrum_background"]
            ),
            spectrum_variance=(
                None
                if spectrum_payload is None
                else spectrum_payload.get("spectrum_variance")
            ),
            spectrum_variance_present=spectrum_variance_present,
        )

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
                "candidate_count": int(
                    design_arr.shape[1] if design_arr.ndim == 2 else 0
                ),
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
        eps = 1.0e-12
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
            variances = measurement_vector(
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

    def _run_isotope_structural_update(
        self,
        task: tuple[
            str,
            IsotopeParticleFilter,
            MeasurementData | None,
        ],
    ) -> None:
        """Run one isotope's PF-native structural update."""
        _isotope, filt, evidence_data = task
        filt.apply_structural_moves(
            evidence_data=evidence_data,
            candidate_positions=self.candidate_sources,
            allow_structural_birth_proposals=True,
        )

    def _structural_update_worker_count(self, task_count: int) -> int:
        """Return the worker count for independent per-isotope structural updates."""
        if task_count <= 1 or not bool(self.pf_config.parallel_isotope_updates):
            return 1
        configured = self.pf_config.parallel_isotope_workers
        if configured is None:
            configured = os.cpu_count() or 1
        return max(1, min(int(configured), int(task_count)))

    def _apply_structural_moves(self) -> None:
        """Apply per-isotope structural updates using all measurement evidence."""
        structural_start = time.perf_counter()
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                MeasurementData | None,
            ]
        ] = []
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(
                filt.config, "converge_enable", False
            ):
                continue
            structural_data = self._measurement_data_for_iso(iso, None)
            tasks.append((iso, filt, structural_data))
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

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the current PF posterior projection without a secondary solver."""
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, filt in self.filters.items():
            use_clustered = bool(
                filt.config.birth_enable and filt.config.use_clustered_output
            )
            positions, strengths = (
                filt.estimate_clustered() if use_clustered else filt.estimate()
            )
            estimates[isotope] = (
                np.asarray(positions, dtype=float).reshape(-1, 3).copy(),
                np.asarray(strengths, dtype=float).reshape(-1).copy(),
            )
        return estimates

    def posterior_source_uncertainty(
        self,
        reported_estimates: Mapping[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ]
        | None = None,
        *,
        match_radius_m: float | None = None,
        surface_tolerance_m: float = 1.0e-5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return JSON-safe posterior 3-D diagnostics for reported source modes.

        Particle source slots are matched to the nearest reported mode in one
        batched distance calculation.  Existence mass is unconditional particle
        mass, while location, covariance, z quantiles, ellipsoid, and surface
        probabilities are conditional on a matched source being present.  Each
        mode includes availability flags so downstream evaluation can exclude
        unsupported summaries; the ellipsoid payload identifies itself as a
        Gaussian-equivalent covariance summary rather than an empirical credible
        region.
        """
        estimate_map = (
            self.estimates() if reported_estimates is None else dict(reported_estimates)
        )
        radius = (
            max(float(self.pf_config.cluster_eps_m), 1.0e-6)
            if match_radius_m is None
            else float(match_radius_m)
        )
        environment = self._source_prior_environment()
        output: Dict[str, List[Dict[str, Any]]] = {}
        for isotope, estimate in estimate_map.items():
            positions = np.asarray(estimate[0], dtype=float)
            strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
            if positions.size == 0:
                positions = np.zeros((0, 3), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("reported estimate positions must have shape (M, 3).")
            if strengths.size != positions.shape[0]:
                raise ValueError(
                    "reported estimate strengths must have one value per mode."
                )
            if np.any(~np.isfinite(strengths)):
                raise ValueError("reported estimate strengths must be finite.")

            filt = self.filters.get(isotope)
            if filt is None or not filt.continuous_particles:
                packed_positions = np.zeros((0, 0, 3), dtype=float)
                packed_mask = np.zeros((0, 0), dtype=bool)
                weights = np.zeros(0, dtype=float)
            else:
                from pf import gpu_utils
                import torch

                report_states = [
                    particle.state for particle in filt.continuous_particles
                ]
                positions_tensor, _, _, mask_tensor = gpu_utils.pack_states(
                    report_states,
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
                packed_positions = positions_tensor.detach().cpu().numpy()
                packed_mask = mask_tensor.detach().cpu().numpy().astype(bool)
                weights = np.asarray(filt.continuous_weights, dtype=float)

            diagnostics = posterior_mode_uncertainty_batched(
                packed_positions,
                packed_mask,
                weights,
                positions,
                environment=environment,
                obstacle_grid=self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                match_radius_m=radius,
                surface_tolerance_m=surface_tolerance_m,
            )
            for mode_index, diagnostic in enumerate(diagnostics):
                diagnostic["reported_strength_cps_1m"] = float(strengths[mode_index])
            output[isotope] = diagnostics
        return output

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

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
                    [
                        int(particle.state.num_sources)
                        for particle in filt.continuous_particles
                    ],
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
                total_counts / total_floor
                if total_floor > 0.0
                else float(total_counts > 0.0)
            )
            max_ratio = (
                max_count / max_floor if max_floor > 0.0 else float(max_count > 0.0)
            )
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
        """Return PF-native structural evidence that still needs measurements."""
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
                    "p_value": float(getattr(filt, "last_birth_residual_p_value", 1.0)),
                }
            if payload:
                unresolved[str(isotope)] = payload
        absent_evidence = self.unresolved_isotope_evidence(
            min_total_counts=25.0,
            min_max_count=5.0,
            min_snr=2.0,
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
                    "mode_preserving_dynamic_spatial_summary": list(
                        getattr(
                            filt,
                            "last_mode_preserving_dynamic_spatial_summary",
                            [],
                        )
                    ),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "kill_count": int(getattr(filt, "last_kill_count", 0)),
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
                    "birth_residual_layer": str(
                        getattr(filt, "last_birth_residual_layer", "none")
                    ),
                    "birth_residual_layer_count": int(
                        getattr(filt, "last_birth_residual_layer_count", 0)
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
            best_state = filt.best_particle().state
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
                    state = filt.continuous_particles[int(idx)].state
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
                "mode_preserving_dynamic_spatial_summary": list(
                    getattr(
                        filt,
                        "last_mode_preserving_dynamic_spatial_summary",
                        [],
                    )
                ),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "kill_count": int(getattr(filt, "last_kill_count", 0)),
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
                "birth_residual_layer": str(
                    getattr(filt, "last_birth_residual_layer", "none")
                ),
                "birth_residual_layer_count": int(
                    getattr(filt, "last_birth_residual_layer_count", 0)
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
        estimates = self.estimates()
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

    def count_likelihood_spec_for_isotope(
        self,
        isotope: str,
    ) -> CountLikelihoodSpec:
        """Return the normalized runtime count-likelihood spec for an isotope."""
        filt = self.filters[str(isotope)]
        return CountLikelihoodSpec(**filt._count_likelihood_kwargs())

    @staticmethod
    def _sample_planning_count_observations_np(
        selected_lambdas: NDArray[np.float64],
        predictive_variance: NDArray[np.float64] | float,
        *,
        spec: CountLikelihoodSpec,
        rng: np.random.Generator,
        epsilon: float = 1.0e-12,
    ) -> NDArray[np.float64]:
        """Sample non-negative future counts from the configured planning model."""
        lambdas = np.maximum(np.asarray(selected_lambdas, dtype=float), epsilon)
        if spec.model == "poisson":
            return np.asarray(rng.poisson(lambdas), dtype=float)
        scale = np.sqrt(
            np.maximum(np.asarray(predictive_variance, dtype=float), epsilon)
        )
        if spec.model == "gaussian":
            observations = rng.normal(loc=lambdas, scale=scale)
        else:
            df = max(float(spec.student_t_df), 1.0 + epsilon)
            observations = lambdas + scale * rng.standard_t(
                df,
                size=lambdas.shape,
            )
        return np.maximum(np.asarray(observations, dtype=float), 0.0)

    @staticmethod
    def _sample_planning_count_observations_torch(
        selected_lambdas: "torch.Tensor",
        predictive_variance: "torch.Tensor",
        *,
        spec: CountLikelihoodSpec,
        epsilon: float = 1.0e-12,
    ) -> "torch.Tensor":
        """Return the Torch equivalent of planning count observation sampling."""
        import torch

        lambdas = torch.clamp(selected_lambdas.to(dtype=torch.float64), min=epsilon)
        if spec.model == "poisson":
            return torch.poisson(lambdas)
        scale = torch.sqrt(
            torch.clamp(
                predictive_variance.to(device=lambdas.device, dtype=torch.float64),
                min=epsilon,
            )
        )
        if spec.model == "gaussian":
            observations = lambdas + scale * torch.randn_like(lambdas)
        else:
            df = max(float(spec.student_t_df), 1.0 + epsilon)
            distribution = torch.distributions.StudentT(
                torch.as_tensor(df, device=lambdas.device, dtype=torch.float64)
            )
            noise = distribution.sample(lambdas.shape)
            observations = lambdas + scale * noise
        return torch.clamp(observations, min=0.0)

    @staticmethod
    def _planning_eig_from_lambdas_np(
        lambdas_ap: NDArray[np.float64],
        weights_p: NDArray[np.float64],
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        rng: np.random.Generator,
        observations_as: NDArray[np.float64] | None = None,
        epsilon: float = 1.0e-12,
    ) -> NDArray[np.float64]:
        """Return batched action EIG using the configured count likelihood."""
        lambdas = np.maximum(np.asarray(lambdas_ap, dtype=float), epsilon)
        if lambdas.ndim == 1:
            lambdas = lambdas.reshape(1, -1)
        if lambdas.ndim != 2:
            raise ValueError("lambdas_ap must have shape action x particle.")
        weights = np.maximum(np.asarray(weights_p, dtype=float).reshape(-1), 0.0)
        if weights.size != lambdas.shape[1]:
            raise ValueError("weights_p must have one value per particle.")
        weight_sum = float(np.sum(weights))
        if weight_sum <= epsilon:
            weights = np.full(weights.size, 1.0 / max(weights.size, 1), dtype=float)
        else:
            weights = weights / weight_sum
        log_weights = np.log(weights + epsilon)
        h_prior = -float(np.sum(weights * log_weights))
        action_count = int(lambdas.shape[0])
        if observations_as is None:
            sample_count = int(num_samples)
            if sample_count <= 0:
                return np.full(action_count, h_prior, dtype=float)
            sample_indices = rng.choice(
                weights.size,
                size=(action_count, sample_count),
                replace=True,
                p=weights,
            )
            selected = np.take_along_axis(lambdas, sample_indices, axis=1)
            predictive_variance = predictive_count_likelihood_variance(
                selected,
                spec=spec,
                epsilon=epsilon,
            )
            observations = (
                RotatingShieldPFEstimator._sample_planning_count_observations_np(
                    selected,
                    predictive_variance,
                    spec=spec,
                    rng=rng,
                    epsilon=epsilon,
                )
            )
        else:
            observations = np.asarray(observations_as, dtype=float)
            if observations.ndim == 1 and action_count == 1:
                observations = observations.reshape(1, -1)
            if observations.ndim != 2 or observations.shape[0] != action_count:
                raise ValueError("observations_as must have shape action x sample.")
            if observations.shape[1] == 0:
                return np.full(action_count, h_prior, dtype=float)
        likelihood_terms = count_log_likelihood_terms_np(
            observations[:, :, None],
            lambdas[:, None, :],
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = log_weights[None, None, :] + likelihood_terms
        log_posterior -= logsumexp(log_posterior, axis=2, keepdims=True)
        posterior = np.exp(log_posterior)
        h_post = -np.sum(
            posterior * np.log(posterior + epsilon),
            axis=2,
        )
        return np.full(action_count, h_prior, dtype=float) - np.mean(h_post, axis=1)

    @staticmethod
    def _planning_eig_from_lambdas_torch(
        lambdas_ap: "torch.Tensor",
        weights_p: "torch.Tensor",
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        observations_as: "torch.Tensor | None" = None,
        epsilon: float = 1.0e-12,
    ) -> "torch.Tensor":
        """Return Torch action EIG equivalent to the batched NumPy helper."""
        import torch

        lambdas = torch.clamp(lambdas_ap.to(dtype=torch.float64), min=epsilon)
        if lambdas.ndim == 1:
            lambdas = lambdas.reshape(1, -1)
        if lambdas.ndim != 2:
            raise ValueError("lambdas_ap must have shape action x particle.")
        weights = torch.clamp(
            weights_p.to(device=lambdas.device, dtype=torch.float64).reshape(-1),
            min=0.0,
        )
        if int(weights.numel()) != int(lambdas.shape[1]):
            raise ValueError("weights_p must have one value per particle.")
        weight_sum = torch.sum(weights)
        if float(weight_sum.detach().cpu().item()) <= epsilon:
            weights = torch.full_like(weights, 1.0 / max(int(weights.numel()), 1))
        else:
            weights = weights / weight_sum
        log_weights = torch.log(weights + epsilon)
        h_prior = -torch.sum(weights * log_weights)
        action_count = int(lambdas.shape[0])
        if observations_as is None:
            sample_count = int(num_samples)
            if sample_count <= 0:
                return torch.full(
                    (action_count,),
                    float(h_prior.detach().cpu().item()),
                    device=lambdas.device,
                    dtype=torch.float64,
                )
            sample_indices = torch.multinomial(
                weights.expand(action_count, -1),
                sample_count,
                replacement=True,
            )
            selected = torch.gather(lambdas, 1, sample_indices)
            predictive_variance = predictive_count_likelihood_variance_torch(
                selected,
                spec=spec,
                epsilon=epsilon,
            )
            observations = (
                RotatingShieldPFEstimator._sample_planning_count_observations_torch(
                    selected,
                    predictive_variance,
                    spec=spec,
                    epsilon=epsilon,
                )
            )
        else:
            observations = observations_as.to(
                device=lambdas.device,
                dtype=torch.float64,
            )
            if observations.ndim == 1 and action_count == 1:
                observations = observations.reshape(1, -1)
            if observations.ndim != 2 or int(observations.shape[0]) != action_count:
                raise ValueError("observations_as must have shape action x sample.")
            if int(observations.shape[1]) == 0:
                return h_prior.expand(action_count).clone()
        likelihood_terms = count_log_likelihood_terms_torch(
            observations.unsqueeze(2),
            lambdas.unsqueeze(1),
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = log_weights.view(1, 1, -1) + likelihood_terms
        log_posterior = log_posterior - torch.logsumexp(
            log_posterior,
            dim=2,
            keepdim=True,
        )
        posterior = torch.exp(log_posterior)
        h_post = -torch.sum(
            posterior * torch.log(posterior + epsilon),
            dim=2,
        )
        return h_prior - torch.mean(h_post, dim=1)

    @staticmethod
    def _expected_strength_uncertainty_from_lambdas_np(
        lambdas_p: NDArray[np.float64],
        weights_p: NDArray[np.float64],
        strengths_pm: NDArray[np.float64],
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        rng: np.random.Generator,
        epsilon: float = 1.0e-12,
    ) -> float:
        """Return batched expected posterior strength variance for one action."""
        lambdas = np.maximum(np.asarray(lambdas_p, dtype=float).reshape(-1), epsilon)
        weights = np.maximum(np.asarray(weights_p, dtype=float).reshape(-1), 0.0)
        strengths = np.asarray(strengths_pm, dtype=float)
        if strengths.ndim != 2 or strengths.shape[0] != lambdas.size:
            raise ValueError("strengths_pm must have one row per particle.")
        if weights.size != lambdas.size:
            raise ValueError("weights_p must have one value per particle.")
        sample_count = int(num_samples)
        if sample_count <= 0 or strengths.size == 0:
            return 0.0
        weight_sum = float(np.sum(weights))
        if weight_sum <= epsilon:
            weights = np.full(weights.size, 1.0 / max(weights.size, 1), dtype=float)
        else:
            weights = weights / weight_sum
        sample_indices = rng.choice(
            weights.size,
            size=sample_count,
            replace=True,
            p=weights,
        )
        selected_lambdas = lambdas[sample_indices]
        predictive_variance = predictive_count_likelihood_variance(
            selected_lambdas,
            spec=spec,
            epsilon=epsilon,
        )
        observations = RotatingShieldPFEstimator._sample_planning_count_observations_np(
            selected_lambdas,
            predictive_variance,
            spec=spec,
            rng=rng,
            epsilon=epsilon,
        )
        likelihood_terms = count_log_likelihood_terms_np(
            observations[:, None],
            lambdas[None, :],
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = np.log(weights + epsilon)[None, :] + likelihood_terms
        log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior_mean = posterior @ strengths
        posterior_second = posterior @ (strengths * strengths)
        posterior_variance = np.maximum(
            posterior_second - posterior_mean * posterior_mean,
            0.0,
        )
        return float(np.mean(np.sum(posterior_variance, axis=1)))

    def orientation_information_gain(
        self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0
    ) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        The denominator is the configured likelihood's predictive variance at
        the posterior-mean count. It reduces to the original mean-count
        denominator exactly for a Poisson likelihood.
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
            predictive_variance = float(
                predictive_count_likelihood_variance(
                    np.asarray(mean, dtype=float),
                    spec=self.count_likelihood_spec_for_isotope(iso),
                    epsilon=eps,
                )
            )
            ig_total += 0.5 * float(np.log1p(var / predictive_variance))
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
            return kernel.expected_counts_pair_for_packed_states_torch(
                isotope=isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                live_time_s=live_time_s,
                source_scale=self.response_scale_for_isotope(
                    isotope,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                ),
                device=device,
                dtype=dtype,
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
            ig_h = float(
                self._planning_eig_from_lambdas_torch(
                    lam_t,
                    weights_t,
                    spec=self.count_likelihood_spec_for_isotope(iso),
                    num_samples=int(num_samples),
                    epsilon=eps,
                )[0]
                .detach()
                .cpu()
                .item()
            )
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
            detector_pos = np.asarray(
                self.kernel_cache.poses[int(pose_idx)],
                dtype=float,
            )
            sample_count = (
                self.pf_config.eig_num_samples
                if num_samples is None
                else int(num_samples)
            )
            eps = 1.0e-12
            alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
            alpha_sum = sum(float(value) for value in alphas.values()) or 1.0
            alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
            rng = np.random.default_rng()
            scores = np.zeros(num_pairs, dtype=float)
            for iso, filt in self.filters.items():
                if getattr(filt, "is_converged", False) and getattr(
                    filt.config,
                    "converge_enable",
                    False,
                ):
                    continue
                if particles_by_isotope is not None and iso in particles_by_isotope:
                    states, weights = particles_by_isotope[iso]
                else:
                    if not filt.continuous_particles:
                        continue
                    states = [particle.state for particle in filt.continuous_particles]
                    weights = filt.continuous_weights
                if not states:
                    continue
                lambdas = self.expected_counts_all_pairs_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector_pos,
                    live_time_s=float(live_time_s),
                    states=states,
                )
                scores += float(alphas.get(iso, 0.0)) * (
                    self._planning_eig_from_lambdas_np(
                        lambdas,
                        np.asarray(weights, dtype=float),
                        spec=self.count_likelihood_spec_for_isotope(iso),
                        num_samples=sample_count,
                        rng=rng,
                        epsilon=eps,
                    )
                )
            return scores.reshape(num_orients, num_orients)

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
        iso_data: Dict[
            str,
            tuple["torch.Tensor", "torch.Tensor", CountLikelihoodSpec],
        ] = {}
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
            iso_data[iso] = (
                lam_all,
                weights_t,
                self.count_likelihood_spec_for_isotope(iso),
            )
        if not iso_data:
            return np.zeros((num_orients, num_orients), dtype=float)

        scores_t = torch.zeros(num_pairs, device=device, dtype=torch.float64)
        for iso, (lam_all, weights_t, spec) in iso_data.items():
            scores_t = scores_t + float(alphas.get(iso, 0.0)) * (
                self._planning_eig_from_lambdas_torch(
                    lam_all,
                    weights_t,
                    spec=spec,
                    num_samples=int(num_samples),
                    epsilon=eps,
                )
            )
        return scores_t.detach().cpu().numpy().reshape(num_orients, num_orients)

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
            ig_h = float(
                self._planning_eig_from_lambdas_np(
                    lam,
                    weights_arr,
                    spec=self.count_likelihood_spec_for_isotope(iso),
                    num_samples=int(num_samples),
                    rng=rng,
                    epsilon=eps,
                )[0]
            )
            total_ig += alphas.get(iso, 0.0) * ig_h
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

        Draw hypothetical observations from the configured count model and
        average posterior strength variance. Uses either Fe/Pb indices or the
        legacy single-orientation index.
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
            total_U += self._expected_strength_uncertainty_from_lambdas_np(
                lam,
                np.asarray(weights, dtype=float),
                strengths_mat,
                spec=self.count_likelihood_spec_for_isotope(iso),
                num_samples=int(num_samples),
                rng=rng,
                epsilon=eps,
            )
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

        Uses Fe/Pb indices and the configured count likelihood without relying
        on pose indices.
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
            total_U += self._expected_strength_uncertainty_from_lambdas_np(
                lam,
                weights,
                strengths_mat,
                spec=self.count_likelihood_spec_for_isotope(iso),
                num_samples=num_samples,
                rng=rng,
                epsilon=eps,
            )
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
        # Planning rollouts must never advance or reseed the global NumPy stream
        # used by the sequential PF.  Otherwise a live planner call between two
        # observations makes same-seed MeasurementLog replay diverge.
        rng = (
            np.random.default_rng()
            if rng_seed is None
            else np.random.default_rng(int(rng_seed))
        )
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
            """Simulate isotope-wise observations from each runtime count model."""
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
                    spec = estimator.count_likelihood_spec_for_isotope(iso)
                    selected_lambda = np.asarray(lam[idx], dtype=float)
                    predictive_variance = predictive_count_likelihood_variance(
                        selected_lambda,
                        spec=spec,
                    )
                    z_k[iso] = float(
                        estimator._sample_planning_count_observations_np(
                            selected_lambda,
                            predictive_variance,
                            spec=spec,
                            rng=rng_local,
                        )
                    )
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
            lam_all = filt.continuous_kernel.expected_counts_all_pairs_for_packed_states_torch(
                isotope=iso,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                live_time_s=live_time_per_rot_s,
                source_scale=self.response_scales_for_measurements(
                    iso,
                    fe_indices,
                    pb_indices,
                ),
                device=device,
                dtype=dtype,
            )
            iso_data[iso] = {
                "lam": lam_all,
                "strengths": strengths,
                "weights": weights,
                "num_particles": weights.size,
                "resample_threshold": filt.config.resample_threshold,
                "likelihood_spec": self.count_likelihood_spec_for_isotope(iso),
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
            spec: CountLikelihoodSpec,
        ) -> "torch.Tensor":
            """Compute model-aware IG for all orientations from cached lambdas."""
            idx_t = torch.as_tensor(
                subset_indices, device=lam_all.device, dtype=torch.long
            )
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(
                subset_weights, device=lam_all.device, dtype=lam_all.dtype
            )
            return self._planning_eig_from_lambdas_torch(
                lam_sel,
                weights_t,
                spec=spec,
                num_samples=int(num_samples),
                epsilon=eps,
            )

        def _update_weights(
            lam_curr: NDArray[np.float64],
            weights: NDArray[np.float64],
            z_obs: float,
            spec: CountLikelihoodSpec,
        ) -> NDArray[np.float64]:
            """Update rollout weights using the configured count likelihood."""
            likelihood_terms = count_log_likelihood_terms_np(
                np.asarray(z_obs, dtype=float),
                lam_curr,
                spec=spec,
                epsilon=eps,
            )
            logw = np.log(weights + eps) + likelihood_terms
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
                        spec=data["likelihood_spec"],
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
                        spec = data["likelihood_spec"]
                        selected_lambda = np.asarray(lam_curr[idx], dtype=float)
                        predictive_variance = predictive_count_likelihood_variance(
                            selected_lambda,
                            spec=spec,
                            epsilon=eps,
                        )
                        z_obs = float(
                            self._sample_planning_count_observations_np(
                                selected_lambda,
                                predictive_variance,
                                spec=spec,
                                rng=rng,
                                epsilon=eps,
                            )
                        )
                    weights = _update_weights(
                        lam_curr,
                        weights,
                        z_obs,
                        data["likelihood_spec"],
                    )
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
