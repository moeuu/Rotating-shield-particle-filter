"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple, Any
import copy
import os

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
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


@dataclass
class RotatingShieldPFConfig:
    """
    Configuration parameters for the rotating-shield PF (Sec. 3.4–3.5).

    Users can tune convergence thresholds and planning settings:
        - max_sources: optional cap on the number of sources per isotope (None = no cap)
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
        - birth_residual_clip_quantile: clip residuals at this quantile
        - birth_residual_gate_p_value: chi-square p-value for residual birth evidence
        - birth_residual_min_support: minimum independent residual-supported measurements
        - birth_residual_support_sigma: per-measurement residual z-score support floor
        - birth_min_distinct_stations: minimum robot stations with residual birth evidence
        - birth_candidate_support_fraction: per-candidate residual overlap floor
        - birth_refit_residual_gate: require residuals to survive fixed-position strength refit
        - birth_refit_residual_min_fraction: residual fraction retained after refit for birth
        - birth_jitter_topk_candidates: base residual-supported candidates jittered for birth
        - refit_after_moves: refit strengths after birth/kill/split/merge
        - refit_iters: iterations for strength refit
        - refit_eps: epsilon for refit stability
        - weak_source_prune_min_expected_count: prune floor-strength sources below this support
        - weak_source_prune_min_fraction: prune floor-strength sources below this source fraction
        - conditional_strength_refit: refit strengths at station finalization
        - conditional_strength_refit_window: recent measurements used for strength refit
        - conditional_strength_refit_iters: iterations for conditional strength refit
        - conditional_strength_refit_reweight: reweight particles by profile-likelihood gain
        - conditional_strength_refit_reweight_clip: robust clip for profile-likelihood correction
        - conditional_strength_refit_min_count: minimum positive count for strength refit
        - conditional_strength_refit_min_snr: minimum count SNR for strength refit
        - conditional_strength_refit_prior_weight: MAP strength-prior weight
        - conditional_strength_refit_prior_rel_sigma: relative strength-prior sigma
        - report_strength_refit: refit reported strengths conditioned on reported positions
        - report_strength_refit_iters: multiplicative Poisson regression iterations
        - report_strength_refit_eps: numerical floor for reported-strength regression
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
        - split_residual_guided: use posterior residual candidates for split moves
        - split_residual_candidate_count: residual candidates evaluated per split
        - merge_prob: probability of merge proposals per particle
        - merge_distance_max: max distance for merge candidates
        - merge_delta_ll_threshold: ΔLL threshold for merge acceptance
        - merge_response_corr_min: response-correlation floor for merge candidates
        - merge_search_topk_pairs: max response-redundant pairs tested per merge move
        - structural_proposal_topk_particles: posterior-support cap for split/merge proposals
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
        - spectrum_count_rel_sigma: relative spectrum-decomposition count uncertainty
        - spectrum_count_abs_sigma: additive spectrum-decomposition count uncertainty
        - count_likelihood_df: Student-t degrees of freedom for robust count likelihood
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
    max_sources: int | None = None
    resample_threshold: float = 0.5
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: Dict[str, float] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | Dict[str, float] = 0.0
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
    structural_update_min_counts: float = 0.0
    birth_min_distinct_poses: int = 1
    birth_residual_clip_quantile: float = 0.95
    birth_residual_gate_p_value: float = 0.05
    birth_residual_min_support: int = 2
    birth_residual_support_sigma: float = 1.0
    birth_min_distinct_stations: int = 1
    birth_candidate_support_fraction: float = 0.05
    birth_refit_residual_gate: bool = True
    birth_refit_residual_min_fraction: float = 0.5
    birth_jitter_topk_candidates: int | None = 512
    refit_after_moves: bool = True
    refit_iters: int = 3
    refit_eps: float = 1e-12
    weak_source_prune_min_expected_count: float = 0.0
    weak_source_prune_min_fraction: float = 0.0
    conditional_strength_refit: bool = True
    conditional_strength_refit_window: int = 10
    conditional_strength_refit_iters: int = 3
    conditional_strength_refit_reweight: bool = False
    conditional_strength_refit_reweight_clip: float = 50.0
    conditional_strength_refit_min_count: float = 5.0
    conditional_strength_refit_min_snr: float = 1.0
    conditional_strength_refit_prior_weight: float = 0.0
    conditional_strength_refit_prior_rel_sigma: float = 2.0
    report_strength_refit: bool = False
    report_strength_refit_iters: int = 64
    report_strength_refit_eps: float = 1.0e-9
    min_age_to_split: int = 5
    use_clustered_output: bool = True
    cluster_eps_m: float = 0.8
    cluster_min_samples: int = 20
    split_prob: float = 0.05
    split_strength_min: float = 0.1
    split_position_sigma: float = 0.25
    split_strength_min_frac: float = 0.3
    split_strength_max_frac: float = 0.7
    split_delta_ll_threshold: float = 0.0
    split_residual_guided: bool = True
    split_residual_candidate_count: int = 8
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
    merge_response_corr_min: float = 0.995
    merge_search_topk_pairs: int = 8
    structural_proposal_topk_particles: int | None = None
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
    adapt_cooldown_steps: int = 0
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
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

    def __post_init__(self) -> None:
        if self.min_particles is None:
            self.min_particles = max(1, int(self.num_particles * 0.5))
        if self.max_particles is None:
            self.max_particles = max(self.num_particles, int(self.num_particles * 2.0))
        self.ess_low = float(self.ess_low)
        self.ess_high = float(self.ess_high)
        if not 0.0 < self.ess_low < self.ess_high < 1.0:
            raise ValueError("ess_low and ess_high must satisfy 0 < ess_low < ess_high < 1.")
        self.init_grid_repeats = max(1, int(self.init_grid_repeats))
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")
        self.adaptive_strength_prior_steps = int(self.adaptive_strength_prior_steps)
        if self.adaptive_strength_prior_steps < 0:
            raise ValueError("adaptive_strength_prior_steps must be >= 0.")
        self.adaptive_strength_prior_min_counts = float(self.adaptive_strength_prior_min_counts)
        if self.adaptive_strength_prior_min_counts < 0.0:
            raise ValueError("adaptive_strength_prior_min_counts must be >= 0.")
        self.adaptive_strength_prior_log_sigma = float(self.adaptive_strength_prior_log_sigma)
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
        self.pose_min_observation_aggregate = str(
            self.pose_min_observation_aggregate
        ).strip().lower()
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
            raise ValueError("count_likelihood_model must be poisson, gaussian, or student_t.")
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
        if self.birth_jitter_topk_candidates is not None:
            self.birth_jitter_topk_candidates = max(
                1,
                int(self.birth_jitter_topk_candidates),
            )
        self.weak_source_prune_min_expected_count = max(
            0.0,
            float(self.weak_source_prune_min_expected_count),
        )
        self.weak_source_prune_min_fraction = max(
            0.0,
            float(self.weak_source_prune_min_fraction),
        )
        if self.birth_max_per_update is not None:
            self.birth_max_per_update = max(0, int(self.birth_max_per_update))
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
        self.report_strength_refit_iters = max(1, int(self.report_strength_refit_iters))
        self.report_strength_refit_eps = max(
            1.0e-15,
            float(self.report_strength_refit_eps),
        )
        self.split_residual_guided = bool(self.split_residual_guided)
        self.split_residual_candidate_count = max(
            1,
            int(self.split_residual_candidate_count),
        )
        self.merge_response_corr_min = float(
            np.clip(float(self.merge_response_corr_min), 0.0, 1.0)
        )
        self.merge_search_topk_pairs = max(1, int(self.merge_search_topk_pairs))
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
        self.history_estimates: List[Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]] = []
        self.history_scores: List[float] = []
        self.measurements: List[MeasurementRecord] = []
        self.last_strength_prior_diagnostics: Dict[str, Dict[str, float]] = {}
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        self._previous_deferred_measurement_count = 0

    def _resolve_mu_by_isotope(self, mu_by_isotope: Dict[str, object] | None) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
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
        if self.isotopes:
            still_missing: List[str] = []
            for iso in self.isotopes:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=[canonical])
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
            spectrum_count_rel_sigma=self.pf_config.spectrum_count_rel_sigma,
            spectrum_count_abs_sigma=self.pf_config.spectrum_count_abs_sigma,
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
            structural_update_min_counts=self.pf_config.structural_update_min_counts,
            birth_min_distinct_poses=self.pf_config.birth_min_distinct_poses,
            birth_residual_clip_quantile=self.pf_config.birth_residual_clip_quantile,
            birth_residual_gate_p_value=self.pf_config.birth_residual_gate_p_value,
            birth_residual_min_support=self.pf_config.birth_residual_min_support,
            birth_residual_support_sigma=self.pf_config.birth_residual_support_sigma,
            birth_min_distinct_stations=self.pf_config.birth_min_distinct_stations,
            birth_candidate_support_fraction=self.pf_config.birth_candidate_support_fraction,
            birth_refit_residual_gate=self.pf_config.birth_refit_residual_gate,
            birth_refit_residual_min_fraction=self.pf_config.birth_refit_residual_min_fraction,
            birth_jitter_topk_candidates=self.pf_config.birth_jitter_topk_candidates,
            refit_after_moves=self.pf_config.refit_after_moves,
            refit_iters=self.pf_config.refit_iters,
            refit_eps=self.pf_config.refit_eps,
            weak_source_prune_min_expected_count=self.pf_config.weak_source_prune_min_expected_count,
            weak_source_prune_min_fraction=self.pf_config.weak_source_prune_min_fraction,
            conditional_strength_refit=self.pf_config.conditional_strength_refit,
            conditional_strength_refit_window=self.pf_config.conditional_strength_refit_window,
            conditional_strength_refit_iters=self.pf_config.conditional_strength_refit_iters,
            conditional_strength_refit_reweight=self.pf_config.conditional_strength_refit_reweight,
            conditional_strength_refit_reweight_clip=self.pf_config.conditional_strength_refit_reweight_clip,
            conditional_strength_refit_min_count=self.pf_config.conditional_strength_refit_min_count,
            conditional_strength_refit_min_snr=self.pf_config.conditional_strength_refit_min_snr,
            conditional_strength_refit_prior_weight=self.pf_config.conditional_strength_refit_prior_weight,
            conditional_strength_refit_prior_rel_sigma=self.pf_config.conditional_strength_refit_prior_rel_sigma,
            min_age_to_split=self.pf_config.min_age_to_split,
            use_clustered_output=self.pf_config.use_clustered_output,
            cluster_eps_m=self.pf_config.cluster_eps_m,
            cluster_min_samples=self.pf_config.cluster_min_samples,
            split_prob=self.pf_config.split_prob,
            split_strength_min=self.pf_config.split_strength_min,
            split_position_sigma=self.pf_config.split_position_sigma,
            split_strength_min_frac=self.pf_config.split_strength_min_frac,
            split_strength_max_frac=self.pf_config.split_strength_max_frac,
            split_delta_ll_threshold=self.pf_config.split_delta_ll_threshold,
            split_residual_guided=self.pf_config.split_residual_guided,
            split_residual_candidate_count=self.pf_config.split_residual_candidate_count,
            merge_prob=self.pf_config.merge_prob,
            merge_distance_max=self.pf_config.merge_distance_max,
            merge_delta_ll_threshold=self.pf_config.merge_delta_ll_threshold,
            merge_response_corr_min=self.pf_config.merge_response_corr_min,
            merge_search_topk_pairs=self.pf_config.merge_search_topk_pairs,
            structural_proposal_topk_particles=(
                self.pf_config.structural_proposal_topk_particles
            ),
            target_ess_ratio=self.pf_config.target_ess_ratio,
            max_temper_steps=self.pf_config.max_temper_steps,
            min_delta_beta=self.pf_config.min_delta_beta,
            max_resamples_per_observation=self.pf_config.max_resamples_per_observation,
            temper_resample_cooldown_steps=self.pf_config.temper_resample_cooldown_steps,
            temper_resample_force_ratio=self.pf_config.temper_resample_force_ratio,
            disable_regularize_on_temper_resample=self.pf_config.disable_regularize_on_temper_resample,
            adapt_cooldown_steps=self.pf_config.adapt_cooldown_steps,
            position_min=self.pf_config.position_min,
            position_max=self.pf_config.position_max,
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
        )

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in RotatingShieldPFConfig.")
        if not gpu_utils.torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

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
        kernel = self._continuous_kernel()
        min_counts = float(self.pf_config.adaptive_strength_prior_min_counts)
        log_sigma = float(self.pf_config.adaptive_strength_prior_log_sigma)
        max_upscale = float(self.pf_config.adaptive_strength_prior_max_upscale)
        min_strength = max(float(self.pf_config.min_strength), 0.0)
        eps = 1e-12
        diagnostics: Dict[str, Dict[str, float]] = {}
        for iso, filt in self.filters.items():
            if iso not in z_k:
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
            source_scale = self.response_scale_for_isotope(iso)
            before_totals: list[float] = []
            after_totals: list[float] = []
            for particle in filt.continuous_particles:
                state = particle.state
                num_sources = int(state.num_sources)
                if num_sources <= 0 or state.strengths.size < num_sources:
                    continue
                strengths = np.maximum(
                    np.asarray(state.strengths[:num_sources], dtype=float),
                    0.0,
                )
                total_strength = float(np.sum(strengths))
                if total_strength <= eps:
                    proportions = np.ones(num_sources, dtype=float) / num_sources
                else:
                    proportions = strengths / total_strength
                unit_rate = 0.0
                for position, proportion in zip(
                    state.positions[:num_sources],
                    proportions,
                ):
                    kernel_value = kernel.kernel_value_pair(
                        isotope=iso,
                        detector_pos=detector,
                        source_pos=np.asarray(position, dtype=float),
                        fe_index=fe_index,
                        pb_index=pb_index,
                    )
                    unit_rate += float(proportion) * source_scale * float(kernel_value)
                unit_counts = live_time * unit_rate
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
                    rate += source_scale * float(strength) * kernel.kernel_value_pair(
                        isotope=isotope,
                        detector_pos=detector_pos,
                        source_pos=pos,
                        fe_index=fe_index,
                        pb_index=pb_index,
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
            if (
                use_gpu_quantile
            ):
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
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
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
            if getattr(filt, "is_converged", False) and getattr(filt.config, "converge_enable", False):
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
            if getattr(filt, "is_converged", False) and getattr(filt.config, "converge_enable", False):
                continue
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            weights = weights / total
            n_particles = len(weights)
            if max_particles is None or max_particles <= 0 or max_particles >= n_particles:
                states = [p.state for p in filt.continuous_particles]
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
                raise ValueError(f"Unknown planning particle selection method: {method}")
            states = [filt.continuous_particles[i].state for i in idx]
            subsets[iso] = (states, sel_weights)
        return subsets

    def weight_entropy_ratio(
        self,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
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

    def add_measurement_pose(self, pose: NDArray[np.float64], reset_filters: bool = True) -> None:
        """Register a new measurement pose and invalidate the kernel cache."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}

    def restrict_isotopes(self, active_isotopes: Sequence[str]) -> None:
        """
        Restrict estimator state to the specified isotopes.

        This drops filters and cached estimates for isotopes that are not in
        active_isotopes while preserving the original isotope ordering.
        """
        active_set = set(active_isotopes)
        if not active_set:
            raise ValueError("active_isotopes must contain at least one isotope.")
        self.isotopes = [iso for iso in self.isotopes if iso in active_set]
        if self.filters:
            self.filters = {iso: filt for iso, filt in self.filters.items() if iso in active_set}
        if self.history_estimates:
            self.history_estimates = [
                {iso: val for iso, val in est.items() if iso in active_set} for est in self.history_estimates
            ]

    def add_isotopes(self, new_isotopes: Sequence[str]) -> None:
        """
        Add isotopes to the estimator and initialize their PF filters.

        This is useful when new isotopes are detected after an initial restriction.
        """
        to_add = [iso for iso in new_isotopes if iso not in self.isotopes]
        if not to_add:
            return
        self.isotopes.extend(to_add)
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
        duration = live_time_s if live_time_s is not None else self.pf_config.short_time_s
        fe_index = octant_index_from_rotation(RFe)
        pb_index = octant_index_from_rotation(RPb)
        self.update_pair(z_k=z_k, pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=duration)

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
                    0.0
                    if z_variance_k is None
                    else float(z_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
                defer_resample=bool(self._defer_resample_birth),
            )
        self.history_estimates.append(self.estimates())
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

    def begin_deferred_pose_update(self) -> None:
        """Start a station-level update that delays resampling and birth/death."""
        self._defer_resample_birth = True
        self._deferred_measurement_count = 0

    def finalize_deferred_pose_update(self) -> int:
        """
        Finish a station-level delayed update and return finalized measurements.

        During a delayed update, each shield posture updates particle weights
        immediately. This method then performs the station-level resampling,
        particle adaptation, label alignment, and residual-gated birth/death once.
        """
        count = int(self._deferred_measurement_count)
        self._defer_resample_birth = False
        self._deferred_measurement_count = 0
        if count <= 0:
            return 0
        for filt in self.filters.values():
            filt.finalize_deferred_update()
        birth_context_count = count + max(
            0,
            int(self._previous_deferred_measurement_count),
        )
        self._apply_birth_death(birth_window_override=birth_context_count)
        self._previous_deferred_measurement_count = count
        self.history_estimates.append(self.estimates())
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
        for iso, filt in self.filters.items():
            z_arr = np.asarray(
                [float(z_k.get(iso, 0.0)) for z_k, _, _, _, _ in records],
                dtype=float,
            )
            var_arr = np.asarray(
                [
                    0.0
                    if z_variance_k is None
                    else float(z_variance_k.get(iso, 0.0))
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
            filt.update_continuous_pair_sequence(
                z_obs=z_arr,
                pose_idx=pose_idx,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_times_s=live_arr,
                observation_count_variances=var_arr,
                step_idx=step_idx,
            )
        self.history_estimates.append(self.estimates())
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
                    0.0
                    if z_variance_k is None
                    else float(z_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
            )
        self.history_estimates.append(self.estimates())
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
        if pos_arr.size == 0 or str_arr.size == 0:
            return pos_arr, str_arr
        if pos_arr.shape[0] != str_arr.size:
            return pos_arr, str_arr
        data = self._measurement_data_for_iso(isotope, None)
        if data is None or data.z_k.size == 0:
            return pos_arr, str_arr
        filt = self.filters.get(isotope)
        if filt is None:
            return pos_arr, str_arr
        refit_data = filt._signal_bearing_refit_data(data)
        if refit_data is None or refit_data.z_k.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        data = refit_data
        unit_strengths = np.ones(str_arr.size, dtype=float)
        design = expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=pos_arr,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self.response_scale_for_isotope(isotope),
        )
        design = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design.ndim != 2 or design.shape[1] != str_arr.size:
            return pos_arr, str_arr
        column_sum = np.sum(design, axis=0)
        observable = column_sum > float(self.pf_config.report_strength_refit_eps)
        if not np.any(observable):
            return pos_arr, np.zeros_like(str_arr, dtype=float)
        z_obs = np.maximum(np.asarray(data.z_k, dtype=float).reshape(-1), 0.0)
        background = self._background_counts_for_report_refit(isotope, data.live_times)
        if background.size != z_obs.size:
            background = np.resize(background, z_obs.size)
        eps = float(self.pf_config.report_strength_refit_eps)
        q = np.maximum(str_arr, eps)
        if not np.any(np.isfinite(q)):
            q = np.full(str_arr.size, eps, dtype=float)
        signal_total = max(float(np.sum(z_obs - background)), 0.0)
        weak_or_invalid = ~np.isfinite(q) | (q <= eps)
        if np.any(weak_or_invalid) and signal_total > 0.0:
            denom = max(float(np.sum(column_sum[observable])), eps)
            q[weak_or_invalid & observable] = signal_total / denom
        q[~observable] = 0.0
        obs_weights = 1.0 / np.maximum(data.observation_variances, 1.0)
        gram = (design.T * obs_weights[None, :]) @ design
        rhs = (design.T * obs_weights[None, :]) @ (z_obs - background)
        try:
            direct = np.linalg.solve(
                gram + np.eye(str_arr.size, dtype=float) * eps,
                rhs,
            )
            direct = np.where(np.isfinite(direct), direct, 0.0)
            if np.any(direct > 0.0):
                q = np.maximum(direct, 0.0)
                q[~observable] = 0.0
        except np.linalg.LinAlgError:
            pass
        q_max = float(getattr(self.pf_config, "birth_q_max", 0.0))
        for _ in range(int(self.pf_config.report_strength_refit_iters)):
            lam = background + design @ q
            lam = np.maximum(lam, eps)
            ratio = np.divide(z_obs, lam, out=np.zeros_like(z_obs), where=lam > 0.0)
            numerator = design.T @ ratio
            denominator = np.maximum(column_sum, eps)
            update = numerator / denominator
            q = q * np.clip(update, 0.0, np.inf)
            q[~observable] = 0.0
            if q_max > 0.0:
                q = np.minimum(q, q_max)
            q = np.where(np.isfinite(q), q, 0.0)
        q = np.maximum(q, 0.0)
        support_floor = max(float(self.pf_config.min_strength), 0.0) * (
            1.0 + 1.0e-6
        )
        keep = observable & (q > support_floor)
        if not np.any(keep):
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return pos_arr[keep], q[keep]

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
        _, filt, refit_data, support_data, birth_data = task
        if bool(self.pf_config.conditional_strength_refit):
            filt.refit_strengths_for_particles(
                refit_data,
                iters=self.pf_config.conditional_strength_refit_iters,
                eps=self.pf_config.refit_eps,
            )
        filt.apply_birth_death(
            support_data=support_data,
            birth_data=birth_data,
            candidate_positions=self.candidate_sources,
        )

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
            if getattr(filt, "is_converged", False) and getattr(filt.config, "converge_enable", False):
                continue
            refit_data = (
                self._measurement_data_for_iso(
                    iso,
                    self.pf_config.conditional_strength_refit_window,
                )
                if bool(self.pf_config.conditional_strength_refit)
                else None
            )
            support_data = self._measurement_data_for_iso(iso, self.pf_config.support_window)
            birth_data = self._measurement_data_for_iso(iso, birth_window)
            tasks.append((iso, filt, refit_data, support_data, birth_data))
        worker_count = self._structural_update_worker_count(len(tasks))
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_structural_update(task)
            return
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(self._run_isotope_structural_update, tasks))

    def estimates(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return per-isotope position/strength estimates for reporting."""
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, filt in self.filters.items():
            use_clustered = bool(
                filt.config.birth_enable and filt.config.use_clustered_output
            )
            if use_clustered and hasattr(filt, "estimate_clustered"):
                try:
                    clustered = filt.estimate_clustered()
                    if clustered[0].shape[0] > 0:
                        estimates[isotope] = self._refit_reported_strengths(
                            isotope,
                            clustered[0],
                            clustered[1],
                        )
                        continue
                except RuntimeError:
                    estimates[isotope] = (
                        np.zeros((0, 3), dtype=float),
                        np.zeros(0, dtype=float),
                    )
                    continue
            raw_positions, raw_strengths = filt.estimate()
            estimates[isotope] = self._refit_reported_strengths(
                isotope,
                raw_positions,
                raw_strengths,
            )
        return estimates

    def estimate_all(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def step_diagnostics(self, top_k: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/kill counts, r distribution
        (mean/variance), MAP/MMSE estimates, and top-k particle summaries.
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
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "kill_count": int(getattr(filt, "last_kill_count", 0)),
                    "birth_residual_chi2": float(getattr(filt, "last_birth_residual_chi2", 0.0)),
                    "birth_residual_p_value": float(getattr(filt, "last_birth_residual_p_value", 1.0)),
                    "birth_residual_support": int(getattr(filt, "last_birth_residual_support", 0)),
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
                    "temper_steps": [],
                    "temper_resamples": 0,
                    "r_mean": 0.0,
                    "r_var": 0.0,
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
            r_vals = np.array([p.state.num_sources for p in filt.continuous_particles], dtype=float)
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
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
            map_positions = best_state.positions.copy()
            map_strengths = best_state.strengths.copy()
            try:
                if bool(filt.config.birth_enable and filt.config.use_clustered_output) and hasattr(
                    filt, "estimate_clustered"
                ):
                    mmse_positions, mmse_strengths = filt.estimate_clustered()
                else:
                    mmse_positions, mmse_strengths = filt.estimate()
            except RuntimeError:
                mmse_positions = np.zeros((0, 3), dtype=float)
                mmse_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.continuous_particles[int(idx)].state
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": int(state.num_sources),
                            "positions": state.positions.copy(),
                            "strengths": state.strengths.copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "n_after_adapt": int(n_after_adapt),
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "kill_count": int(getattr(filt, "last_kill_count", 0)),
                "birth_residual_chi2": float(getattr(filt, "last_birth_residual_chi2", 0.0)),
                "birth_residual_p_value": float(getattr(filt, "last_birth_residual_p_value", 1.0)),
                "birth_residual_support": int(getattr(filt, "last_birth_residual_support", 0)),
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
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "r_mean": r_mean,
                "r_var": r_var,
                "map": (map_positions, map_strengths),
                "mmse": (mmse_positions, mmse_strengths),
                "top_k": top_entries,
                "converged": bool(getattr(filt, "is_converged", False)),
                "updates_skipped": int(getattr(filt, "updates_skipped", 0)),
            }
        return diagnostics

    def isotope_log_likelihood_gain(self, window: int | None = None) -> Dict[str, float]:
        """
        Return per-isotope log-likelihood gain vs background-only (evidence mixing).
        """
        if not self.measurements:
            return {iso: 0.0 for iso in self.filters}
        estimates = self.pruned_estimates(method="legacy")
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
            lambda_m = expected_counts_per_source(
                kernel=filt.continuous_kernel,
                isotope=iso,
                detector_positions=data.detector_positions,
                sources=positions,
                strengths=strengths,
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self.response_scale_for_isotope(iso),
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

    def isotopes_by_evidence(self, min_delta_ll: float = 0.0, window: int | None = None) -> List[str]:
        """
        Return isotopes whose LL gain exceeds min_delta_ll for the given window.
        """
        gains = self.isotope_log_likelihood_gain(window=window)
        return [iso for iso, gain in gains.items() if gain >= float(min_delta_ll)]

    @property
    def num_orientations(self) -> int:
        return self.normals.shape[0]

    def orientation_information_gain(self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        IG_k(phi) ~= 0.5 * log(1 + Var[Lambda_k(phi)] / E[Lambda_k(phi)]) aggregated over isotopes.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            if getattr(filt, "is_converged", False) and getattr(filt.config, "converge_enable", False):
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

    def max_orientation_information_gain(self, pose_idx: int, live_time_s: float = 1.0) -> float:
        """Return max_phi IG_k(phi) at pose k (Eq. 3.45 surrogate)."""
        scores = [
            self.orientation_information_gain(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
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
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
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
        num_samples = self.pf_config.eig_num_samples if num_samples is None else num_samples
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
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

        def _compute_lam_torch(states: Sequence[IsotopeState], isotope: str) -> "torch.Tensor":
            if not states:
                return torch.zeros(0, device=device, dtype=dtype)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
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
            if getattr(filt, "is_converged", False) and getattr(filt.config, "converge_enable", False):
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
                logw = torch.log(weights_t + eps) + z.unsqueeze(1) * torch.log(lam_t + eps) - lam_t
                logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                w_post = torch.exp(logw)
                H_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                H_post_mean = torch.mean(H_post)
            ig_h = float((H_prior - H_post_mean).item())
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
                    pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
                )
            else:
                lam = filt._continuous_expected_counts(pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
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
        rollout_method = self.pf_config.planning_rollout_method or self.pf_config.planning_method

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
        self._gpu_enabled()
        from pf import gpu_utils
        import torch
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        num_orients = num_fe * num_pb
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
            lam_list = []
            for fe_idx, pb_idx in zip(fe_indices, pb_indices):
                lam_t = gpu_utils.expected_counts_pair_torch(
                    detector_pos=detector_pos,
                    positions=positions,
                    strengths=strengths,
                    backgrounds=backgrounds,
                    mask=mask,
                    fe_index=int(fe_idx),
                    pb_index=int(pb_idx),
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
                lam_list.append(lam_t)
            if not lam_list:
                continue
            lam_all = torch.stack(lam_list, dim=0)
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
            if max_particles is None or max_particles <= 0 or max_particles >= len(weights):
                return indices, weights
            method = method or "top_weight"
            if method == "top_weight":
                sel = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[sel]
                sel_weights = sel_weights / max(np.sum(sel_weights), eps)
                return indices[sel], sel_weights
            if method == "resample":
                sel = rng_local.choice(len(weights), size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max(max_particles, 1)
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
                weights_t = torch.as_tensor(subset_weights, device=lam_all.device, dtype=lam_all.dtype)
                weights_t = weights_t / torch.sum(weights_t)
                h_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
                return torch.full((lam_all.shape[0],), h_prior, device=lam_all.device, dtype=lam_all.dtype)
            idx_t = torch.as_tensor(subset_indices, device=lam_all.device, dtype=torch.long)
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(subset_weights, device=lam_all.device, dtype=lam_all.dtype)
            weights_t = weights_t / torch.sum(weights_t)
            log_weights = torch.log(weights_t + eps)
            h_prior = -torch.sum(weights_t * log_weights)
            weights_row = weights_t.expand(lam_sel.shape[0], -1)
            idx_samples = torch.multinomial(weights_row, num_samples, replacement=True)
            lam_samples = torch.gather(lam_sel, 1, idx_samples)
            z = torch.poisson(lam_samples)
            log_lam = torch.log(lam_sel + eps)
            logw = log_weights.view(1, 1, -1) + z.unsqueeze(2) * log_lam.unsqueeze(1) - lam_sel.unsqueeze(1)
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
            _, strengths_t, _, _ = gpu_utils.pack_states(states, device=device, dtype=dtype)
            weights = torch.as_tensor(filt.continuous_weights, device=device, dtype=dtype)
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
            max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
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
                self.orientation_information_gain(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
            )
        max_ig = max(ig_scores) if ig_scores else 0.0
        dwell_time = sum(rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx)
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
        )
        pruned: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for iso, (pos, strg) in est.items():
            keep = keep_masks.get(iso)
            if keep is None or keep.size == 0:
                pruned[iso] = (pos, strg)
            else:
                pruned[iso] = (pos[keep], strg[keep])
        return pruned
