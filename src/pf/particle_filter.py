"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Tuple
from collections import deque
import os
import time

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, qmc

from measurement.model import EnvironmentConfig
from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    generate_octant_orientations,
    resolve_mu_values,
)
from measurement.source_surfaces import (
    build_surface_candidate_sources,
    project_positions_to_allowed_surfaces,
    source_surface_kinds,
)
from measurement.surface_patches import (
    SurfacePatchDictionary,
    build_surface_patch_dictionary,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.likelihood import (
    CountLikelihoodSpec,
    OBSERVATION_COUNT_VARIANCE_ADDITIONAL,
    OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL,
    count_log_likelihood,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    expected_counts_per_source,
    normalize_count_likelihood_model,
    normalize_observation_count_variance_semantics,
)
from pf.state import IsotopeState
from pf.resampling import systematic_resample, systematic_resample_count
from pf.strength_prior import StrengthPrior
from pf.structural_rj import (
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    SurfaceAdjacency,
    SurfaceSetPrior,
    add_surface_indices,
    birth_log_acceptance_ratio,
    conditional_birth_surface_log_probability,
    death_log_acceptance_ratio,
    local_position_log_acceptance_ratio,
    remove_surface_columns,
    uniform_death_index_log_probability,
)

if TYPE_CHECKING:
    import torch


def _pf_debug_timing_enabled() -> bool:
    """Return True when verbose PF phase timing should be printed."""
    return os.environ.get("PF_DEBUG_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5  # relative to N
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: dict[str, float] | None = None
    measurement_scale_by_isotope_and_pair: dict[str, dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | dict[str, float] = 0.0
    transport_model_abs_sigma: float | dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_transition_counts: float | dict[str, float] = 0.0
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
    birth_q_max: float = 3e5
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
    structural_cardinality_prior_probs: tuple[float, ...] | None = None
    ess_low: float = 0.5
    ess_high: float = 0.9
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
    mode_preserving_resample: bool = False
    mode_preserving_max_modes: int = 4
    mode_preserving_particles_per_mode: int = 2
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
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "volume"
    roughening_k: float = 0.5
    surface_rejuvenation_enable: bool = True
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_num_sources: Tuple[int, int] = (0, 3)  # inclusive range
    # Strength prior (cps@1m scale). The uniform form is used when a simulation
    # declares its source-population bounds before evaluation.
    init_strength_prior: str = "lognormal"
    init_strength_min: float = 0.0
    init_strength_max: float | None = None
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
    init_joint_position_design: str = "independent"
    init_joint_position_retries: int = 1
    init_source_min_separation_m: float = 0.0
    strength_log_sigma: float = 0.3
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    label_alignment_iters: int = 2
    label_pos_weight: float = 1.0
    label_strength_weight: float = 0.2
    label_missing_cost: float = 1e3
    label_pos_scale: float | None = None
    label_strength_scale: float | None = None
    label_enable: bool = True
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
        """Normalize likelihood semantics and reject incompatible settings."""
        self.structural_kernel_mode = (
            str(self.structural_kernel_mode).strip().lower().replace("-", "_")
        )
        if self.structural_kernel_mode not in {"heuristic", "rj_mh"}:
            raise ValueError(
                "structural_kernel_mode must be heuristic or rj_mh."
            )
        self.structural_rj_patch_spacing_m = float(
            self.structural_rj_patch_spacing_m
        )
        if (
            not np.isfinite(self.structural_rj_patch_spacing_m)
            or self.structural_rj_patch_spacing_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_patch_spacing_m must be finite and positive."
            )
        for probability_name in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
        ):
            probability = float(getattr(self, probability_name))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_name} must lie in [0, 1].")
            setattr(self, probability_name, probability)
        if (
            (
                self.structural_rj_birth_probability <= 0.0
                or self.structural_rj_death_probability <= 0.0
            )
            and self.birth_enable
            and self.structural_kernel_mode == "rj_mh"
        ):
            raise ValueError(
                "rj_mh requires positive birth and death proposal probabilities "
                "so every dimension change has a reverse move."
            )
        if self.structural_cardinality_prior_probs is not None:
            cardinality_prior = np.asarray(
                self.structural_cardinality_prior_probs,
                dtype=float,
            ).reshape(-1)
            if (
                cardinality_prior.size == 0
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain finite "
                    "positive values."
                )
            cardinality_prior /= float(np.sum(cardinality_prior))
            self.structural_cardinality_prior_probs = tuple(
                float(value) for value in cardinality_prior
            )
        if self.structural_kernel_mode == "rj_mh" and bool(self.birth_enable):
            if not self._source_prior_name_is_surface(self.source_position_prior):
                raise ValueError(
                    "rj_mh requires source_position_prior='surface'."
                )
            if self.max_sources is None or int(self.max_sources) < 1:
                raise ValueError("rj_mh requires a finite positive max_sources.")
            expected_cardinalities = int(self.max_sources) + 1
            if (
                self.structural_cardinality_prior_probs is not None
                and len(self.structural_cardinality_prior_probs)
                != expected_cardinalities
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must have "
                    "max_sources + 1 entries."
                )
            initial_lower, initial_upper = self.init_num_sources
            if (
                int(initial_lower) != 0
                or int(initial_upper) != int(self.max_sources)
            ):
                raise ValueError(
                    "rj_mh initialization must cover every cardinality from "
                    "zero through max_sources."
                )
            incompatible = {
                "cardinality_preserving_resample": bool(
                    self.cardinality_preserving_resample
                ),
                "mode_preserving_resample": bool(self.mode_preserving_resample),
                "surface_rejuvenation_enable": bool(
                    self.surface_rejuvenation_enable
                ),
                "pseudo_source_verification_enable": bool(
                    self.pseudo_source_verification_enable
                ),
                "split_prob": float(self.split_prob) > 0.0,
                "merge_prob": float(self.merge_prob) > 0.0,
                "source_detector_exclusion_m": (
                    float(self.source_detector_exclusion_m) > 0.0
                ),
                "init_source_min_separation_m": (
                    float(self.init_source_min_separation_m) > 0.0
                ),
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "rj_mh is incompatible with heuristic or state-dependent "
                    f"support settings: {', '.join(enabled)}."
                )
        semantics = normalize_observation_count_variance_semantics(
            self.observation_count_variance_semantics,
            includes_counting_noise=(
                self.observation_count_variance_includes_counting_noise
            ),
        )
        self.observation_count_variance_semantics = semantics
        self.observation_count_variance_includes_counting_noise = (
            semantics != OBSERVATION_COUNT_VARIANCE_ADDITIONAL
        )
        self.direct_spectrum_likelihood_enable = bool(
            self.direct_spectrum_likelihood_enable
        )
        self.shield_contrast_likelihood_enable = bool(
            self.shield_contrast_likelihood_enable
        )
        self.shield_view_ratio_likelihood_enable = bool(
            self.shield_view_ratio_likelihood_enable
        )
        if semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
            model = normalize_count_likelihood_model(self.count_likelihood_model)
            if model == "poisson":
                raise ValueError(
                    "complete_statistical observation variance requires gaussian "
                    "or student_t count likelihood."
                )
            # Derived shield-shape terms reuse these count observations without
            # their complete covariance, so they are inadmissible here.
            self.shield_contrast_likelihood_enable = False
            self.shield_view_ratio_likelihood_enable = False

    @staticmethod
    def _source_prior_name_is_surface(value: object) -> bool:
        """Return whether a configuration value denotes surface support."""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower().replace("-", "_") in {
            "surface",
            "surfaces",
            "surface_constrained",
        }


@dataclass
class PFConvergenceMonitor:
    """Track per-isotope convergence statistics over a sliding window."""

    window: int
    min_steps: int
    map_move_eps_m: float
    ess_ratio_high: float
    ll_improve_eps: float
    require_all: bool

    def __post_init__(self) -> None:
        """Initialize bounded convergence-history buffers."""
        self.positions: deque[NDArray[np.float64] | None] = deque(maxlen=self.window)
        self.ess_ratios: deque[float] = deque(maxlen=self.window)
        self.ll_values: deque[float] = deque(maxlen=self.window)

    def update_stats(
        self,
        step_idx: int,
        pos: NDArray[np.float64] | None,
        ess_ratio: float,
        ll_value: float,
    ) -> None:
        """Append the latest statistics to the window."""
        if step_idx < 0:
            return
        self.positions.append(pos.copy() if pos is not None else None)
        self.ess_ratios.append(float(ess_ratio))
        self.ll_values.append(float(ll_value))

    def is_converged(self, step_idx: int) -> bool:
        """Return True if all convergence criteria are satisfied."""
        if step_idx < self.min_steps:
            return False
        if len(self.positions) < self.window:
            return False
        if any(pos is None for pos in self.positions):
            return False
        pos_list = [pos for pos in self.positions if pos is not None]
        reference_shape = pos_list[0].shape
        if any(pos.shape != reference_shape for pos in pos_list):
            return False
        max_move = 0.0
        for prev, curr in zip(pos_list[:-1], pos_list[1:]):
            max_move = max(max_move, float(np.linalg.norm(curr - prev)))
        move_ok = max_move <= float(self.map_move_eps_m)
        ess_ok = min(self.ess_ratios) >= float(self.ess_ratio_high)
        ll_span = max(self.ll_values) - min(self.ll_values)
        ll_ok = ll_span <= float(self.ll_improve_eps)
        if self.require_all:
            return move_ok and ess_ok and ll_ok
        return sum([move_ok, ess_ok, ll_ok]) >= 2


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float


@dataclass(frozen=True)
class MeasurementData:
    """Bundle full-history measurement arrays for structural PF moves."""

    z_k: NDArray[np.float64]
    observation_variances: NDArray[np.float64]
    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]
    station_sequence_ids: NDArray[np.int64] | None = None
    runtime_likelihood_routes: NDArray[np.str_] | None = None
    observation_count_covariance: NDArray[np.float64] | None = None
    spectrum_counts: NDArray[np.float64] | None = None
    spectrum_response_template: NDArray[np.float64] | None = None
    spectrum_background: NDArray[np.float64] | None = None
    spectrum_variance: NDArray[np.float64] | None = None
    spectrum_variance_present: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class BirthResidualLayer:
    """Store one residual layer used for residual-driven source birth."""

    name: str
    residual: NDArray[np.float64]


class IsotopeParticleFilter:
    """Per-isotope particle filter (continuous state is the primary mode)."""

    def __init__(
        self,
        isotope: str,
        kernel: KernelPrecomputer | None,
        config: PFConfig | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: dict[str, object] | None = None,
        transport_response_model: dict[str, object] | None = None,
    ) -> None:
        """Initialize particle state, priors, and continuous measurement kernels."""
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
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
        self._surface_candidate_cache: dict[float, NDArray[np.float64]] = {}
        self._random_generator = np.random.default_rng(
            int(np.random.randint(0, np.iinfo(np.uint32).max))
        )
        self._strength_prior = self._build_strength_prior()
        self._structural_rj_surface_patches: SurfacePatchDictionary | None = None
        self._structural_rj_patch_key_to_index: dict[tuple[float, float, float], int] = (
            {}
        )
        self._structural_rj_cardinality_prior_probs = (
            self._build_structural_cardinality_prior()
        )
        self._structural_rj_cardinality_prior: CardinalityPrior | None = None
        self._structural_rj_surface_prior: SurfaceSetPrior | None = None
        self._structural_rj_surface_adjacency: SurfaceAdjacency | None = None
        self._structural_rj_move_probabilities: (
            BirthDeathMoveProbabilities | None
        ) = None
        self._structural_rj_response_cache: NDArray[np.float64] | None = None
        self._structural_rj_response_cache_signatures: (
            NDArray[np.float64] | None
        ) = None
        self._structural_rj_response_evaluation_batches = 0
        self._structural_rj_response_evaluated_cells = 0
        self._structural_rj_response_touched_mask: (
            NDArray[np.bool_] | None
        ) = None
        self._structural_rj_move_counts: dict[str, int] = {}
        if self._structural_kernel_is_exact():
            self._initialize_structural_rj_surface_support()
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        )
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        self._continuous_kernel_physics_signature = (
            self._incoming_kernel_physics_signature(kernel)
        )
        self.continuous_particles: List[IsotopeParticle] = []
        self._label_reference: IsotopeState | None = None
        self.last_ess: float | None = None
        self.last_ess_pre: float | None = None
        self.last_ess_post: float | None = None
        self.last_resample_ess = False
        self.last_resample_count = 0
        self.last_birth_count = 0
        self.last_kill_count = 0
        self.last_n_after_adapt: int | None = None
        self.last_temper_steps: list[dict[str, float]] = []
        self.last_temper_resample_count = 0
        self.last_mode_preserved_count = 0
        self.last_mode_preserving_strata_summary: dict[str, float] = {}
        self.last_mode_preserving_selected_strata: list[dict[str, object]] = []
        self.last_mode_preserving_cardinality_summary: dict[str, float] = {}
        self.last_mode_preserving_selected_cardinalities: list[dict[str, object]] = []
        self.last_mode_preserving_dynamic_spatial_summary: list[dict[str, object]] = []
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_delta_ll = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False
        self.last_birth_residual_layer = "none"
        self.last_birth_residual_layer_count = 0
        self.last_birth_structural_eligible = 0
        self.last_pseudo_source_verified = 0
        self.last_pseudo_source_failed = 0
        self.last_pseudo_source_pruned = 0
        self.last_pseudo_source_quarantined = 0
        self.last_pseudo_source_quarantine_active = 0
        self.last_pseudo_source_fail_reasons: dict[str, int] = {}
        self.last_source_event_diagnostics: list[dict[str, object]] = []
        self.last_structural_timing_s: dict[str, float] = {}
        self.last_spectrum_likelihood_route = "none"
        self._deferred_resampled_any = False
        self._deferred_ess_min: float | None = None
        self._deferred_convergence_args: (
            tuple[
                int | None,
                NDArray[np.float64],
                int,
                int,
                float,
                float,
            ]
            | None
        ) = None
        self._adapt_cooldown_remaining = 0
        self._resample_count_in_observation = 0
        self._observed_station_labels: set[tuple[float, float]] = set()
        self.is_converged = False
        self.frozen_estimate: tuple[NDArray[np.float64], NDArray[np.float64]] | None = (
            None
        )
        self.updates_skipped = 0
        self._converge_monitor = (
            PFConvergenceMonitor(
                window=int(self.config.converge_window),
                min_steps=int(self.config.converge_min_steps),
                map_move_eps_m=float(self.config.converge_map_move_eps_m),
                ess_ratio_high=float(self.config.converge_ess_ratio_high),
                ll_improve_eps=float(self.config.converge_ll_improve_eps),
                require_all=bool(self.config.converge_require_all),
            )
            if self.config.converge_enable
            else None
        )
        self._init_continuous_particles()

    def _structural_kernel_is_exact(self) -> bool:
        """Return whether the target-preserving surface RJ-MH kernel is active."""
        return bool(self.config.birth_enable) and (
            str(getattr(self.config, "structural_kernel_mode", "heuristic"))
            .strip()
            .lower()
            .replace("-", "_")
            == "rj_mh"
        )

    def _build_strength_prior(self) -> StrengthPrior:
        """Build the normalized strength prior shared by initialization and moves."""
        return StrengthPrior(
            kind=str(self.config.init_strength_prior),
            minimum=float(self.config.init_strength_min),
            maximum=(
                None
                if self.config.init_strength_max is None
                else float(self.config.init_strength_max)
            ),
            log_mean=float(self.config.init_strength_log_mean),
            log_sigma=float(self.config.init_strength_log_sigma),
        )

    def _build_structural_cardinality_prior(self) -> NDArray[np.float64]:
        """Return the normalized prior mass for cardinalities zero through max."""
        max_sources = self.config.max_sources
        if max_sources is None:
            return np.zeros(0, dtype=float)
        count = max(0, int(max_sources)) + 1
        configured = self.config.structural_cardinality_prior_probs
        if configured is None:
            return np.full(count, 1.0 / max(count, 1), dtype=float)
        probabilities = np.asarray(configured, dtype=float).reshape(-1)
        if probabilities.size != count:
            raise ValueError(
                "structural_cardinality_prior_probs must have max_sources + 1 "
                "entries."
            )
        return probabilities / float(np.sum(probabilities))

    @staticmethod
    def _surface_patch_key(position: NDArray[np.float64]) -> tuple[float, float, float]:
        """Return a stable exact-mode lookup key for one patch center."""
        values = np.asarray(position, dtype=float).reshape(3)
        rounded = np.round(values, decimals=12)
        return float(rounded[0]), float(rounded[1]), float(rounded[2])

    def _initialize_structural_rj_surface_support(self) -> None:
        """Build complete area-aware surface support for exact structural moves."""
        position_min = np.asarray(self.config.position_min, dtype=float).reshape(3)
        if not np.allclose(position_min, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "rj_mh currently requires position_min=(0, 0, 0) so the "
                "environment surface measure is unambiguous."
            )
        patches = build_surface_patch_dictionary(
            self._source_prior_environment(),
            self.obstacle_grid,
            float(self.config.structural_rj_patch_spacing_m),
            obstacle_height_m=self.obstacle_height_m,
        )
        if not patches.obstacle_surfaces_available:
            warning = patches.obstacle_geometry_warning or (
                "Obstacle component surfaces are unavailable."
            )
            raise ValueError(
                "rj_mh requires complete obstacle component geometry: "
                f"{warning}"
            )
        if patches.patch_count <= int(self.config.max_sources or 0):
            raise ValueError(
                "rj_mh surface dictionary must contain more patches than "
                "max_sources."
            )
        keys = [
            self._surface_patch_key(position)
            for position in np.asarray(patches.centers_xyz, dtype=float)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "rj_mh surface dictionary contains duplicate patch centers."
            )
        self._structural_rj_surface_patches = patches
        self._structural_rj_patch_key_to_index = {
            key: int(index) for index, key in enumerate(keys)
        }
        max_sources = int(self.config.max_sources or 0)
        self._structural_rj_surface_prior = SurfaceSetPrior(
            patches.areas_m2,
            max_cardinality=max_sources,
        )
        self._structural_rj_surface_adjacency = SurfaceAdjacency(
            dictionary_size=patches.patch_count,
            edges=patches.adjacency_edges,
        )
        self._structural_rj_cardinality_prior = CardinalityPrior(
            self._structural_rj_cardinality_prior_probs
        )
        self._structural_rj_move_probabilities = BirthDeathMoveProbabilities(
            max_cardinality=max_sources,
            birth_weight=float(self.config.structural_rj_birth_probability),
            death_weight=float(self.config.structural_rj_death_probability),
        )

    def _structural_rj_patch_indices_for_state(
        self,
        state: IsotopeState,
    ) -> NDArray[np.int64]:
        """Resolve and validate the canonical surface indices of one exact state."""
        source_count = int(state.num_sources)
        positions = np.asarray(state.positions, dtype=float).reshape(-1, 3)
        if positions.shape[0] != source_count:
            raise ValueError("rj_mh state positions must match num_sources.")
        if source_count == 0:
            return np.zeros(0, dtype=np.int64)
        try:
            indices = np.asarray(
                [
                    self._structural_rj_patch_key_to_index[
                        self._surface_patch_key(position)
                    ]
                    for position in positions
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "rj_mh state contains a position outside the finite surface "
                "dictionary."
            ) from exc
        if np.unique(indices).size != indices.size:
            raise ValueError(
                "rj_mh does not permit duplicate surface patches in one state."
            )
        return indices

    def _canonicalize_structural_rj_state(
        self,
        state: IsotopeState,
    ) -> NDArray[np.int64]:
        """Sort one exact state by surface index and return the sorted indices."""
        self._ensure_source_metadata(state)
        indices = self._structural_rj_patch_indices_for_state(state)
        if indices.size <= 1:
            return indices
        order = np.argsort(indices, kind="stable")
        if not np.array_equal(order, np.arange(indices.size)):
            state.positions = np.asarray(state.positions, dtype=float)[order]
            state.strengths = np.asarray(state.strengths, dtype=float)[order]
            state.ages = np.asarray(state.ages, dtype=int)[order]
            state.support_scores = np.asarray(
                state.support_scores,
                dtype=float,
            )[order]
            state.tentative_sources = np.asarray(
                state.tentative_sources,
                dtype=bool,
            )[order]
            state.verification_fail_streaks = np.asarray(
                state.verification_fail_streaks,
                dtype=int,
            )[order]
            indices = indices[order]
        return np.asarray(indices, dtype=np.int64)

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the continuous kernel with the filter's environment attenuation settings."""
        kernel_kwargs: dict[str, object] = {}
        orientations = getattr(self.kernel, "orientations", None)
        if orientations is not None and len(orientations) > 1:
            kernel_kwargs["orientations"] = orientations
        return ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
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
            **kernel_kwargs,
        )

    def _incoming_kernel_physics_signature(
        self,
        kernel: KernelPrecomputer | None,
    ) -> tuple[object, ...]:
        """Return canonical incoming physics that affects this isotope's kernel."""
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None)
            if kernel is not None
            else None
        )
        mu_fe, mu_pb = resolve_mu_values(
            mu_by_isotope,
            self.isotope,
            default_fe=float(shield_params.mu_fe),
            default_pb=float(shield_params.mu_pb),
        )
        incoming_orientations = (
            getattr(kernel, "orientations", None)
            if kernel is not None
            else None
        )
        orientations = (
            generate_octant_orientations()
            if incoming_orientations is None
            or len(incoming_orientations) <= 1
            else np.asarray(incoming_orientations, dtype=np.float64)
        )
        orientation_array = np.asarray(
            orientations,
            dtype=np.float64,
        )
        canonical_orientations = np.ascontiguousarray(
            np.where(orientation_array == 0.0, 0.0, orientation_array),
            dtype="<f8",
        )
        shield_signature = (
            float(shield_params.mu_pb),
            float(shield_params.mu_fe),
            float(shield_params.thickness_pb_cm),
            float(shield_params.thickness_fe_cm),
            float(shield_params.inner_radius_fe_cm),
            float(shield_params.inner_radius_pb_cm),
            max(float(shield_params.buildup_fe_coeff), 0.0),
            max(float(shield_params.buildup_pb_coeff), 0.0),
            str(shield_params.shield_geometry_model),
            bool(shield_params.use_angle_attenuation),
        )
        return (
            (float(mu_fe), float(mu_pb)),
            shield_signature,
            canonical_orientations.shape,
            canonical_orientations.tobytes(order="C"),
        )

    def _measurement_source_scale(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the isotope-specific source response scale for PF likelihoods."""
        pair_id = self._measurement_shield_pair_id(
            fe_index=fe_index,
            pb_index=pb_index,
            shield_pair_id=shield_pair_id,
        )
        pair_scales = self.config.measurement_scale_by_isotope_and_pair
        if pair_id is not None and isinstance(pair_scales, Mapping):
            isotope_pair_scales = pair_scales.get(str(self.isotope), {})
            if isinstance(isotope_pair_scales, Mapping):
                value = isotope_pair_scales.get(int(pair_id))
                if value is None:
                    value = isotope_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(self.isotope, 1.0)), 0.0)

    def _measurement_source_scale_vector(
        self,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale for each measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self._measurement_source_scale(
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _measurement_shield_pair_id(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> int | None:
        """Return the configured shield-pair id when orientation indices exist."""
        if shield_pair_id is not None:
            return int(shield_pair_id)
        if fe_index is None or pb_index is None:
            return None
        orientations = getattr(self.kernel, "orientations", None)
        num_orientations = len(orientations) if orientations is not None else 8
        return int(fe_index) * max(int(num_orientations), 1) + int(pb_index)

    @staticmethod
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

    def _reset_structural_residual_gate(self) -> None:
        """Reset birth residual diagnostics when structural updates are skipped."""
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_delta_ll = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False

    def _measurement_rows(
        self, data: MeasurementData, mask: NDArray[np.bool_]
    ) -> MeasurementData:
        """Return a measurement bundle restricted to the selected row mask."""
        row_mask = np.asarray(mask, dtype=bool).reshape(-1)
        covariance = data.observation_count_covariance
        restricted_covariance = None
        if covariance is not None:
            covariance_arr = np.asarray(covariance, dtype=float)
            expected_shape = (int(row_mask.size), int(row_mask.size))
            if covariance_arr.shape != expected_shape:
                raise ValueError("observation_count_covariance must be shaped K x K.")
            restricted_covariance = covariance_arr[np.ix_(row_mask, row_mask)]
        restricted_sequence_ids = None
        if data.station_sequence_ids is not None:
            sequence_ids = np.asarray(
                data.station_sequence_ids,
                dtype=np.int64,
            ).reshape(-1)
            if sequence_ids.size != row_mask.size:
                raise ValueError(
                    "station_sequence_ids must contain one ID per measurement."
                )
            restricted_sequence_ids = sequence_ids[row_mask]
        restricted_routes = None
        if data.runtime_likelihood_routes is not None:
            routes = np.asarray(
                data.runtime_likelihood_routes,
                dtype=str,
            ).reshape(-1)
            if routes.size != row_mask.size:
                raise ValueError(
                    "runtime_likelihood_routes must contain one route per "
                    "measurement."
                )
            restricted_routes = routes[row_mask]
        return MeasurementData(
            z_k=np.asarray(data.z_k, dtype=float)[row_mask],
            observation_variances=np.asarray(data.observation_variances, dtype=float)[
                row_mask
            ],
            detector_positions=np.asarray(data.detector_positions, dtype=float)[
                row_mask
            ],
            fe_indices=np.asarray(data.fe_indices, dtype=int)[row_mask],
            pb_indices=np.asarray(data.pb_indices, dtype=int)[row_mask],
            live_times=np.asarray(data.live_times, dtype=float)[row_mask],
            station_sequence_ids=restricted_sequence_ids,
            runtime_likelihood_routes=restricted_routes,
            observation_count_covariance=restricted_covariance,
            spectrum_counts=(
                None
                if data.spectrum_counts is None
                else np.asarray(data.spectrum_counts, dtype=float)[row_mask]
            ),
            spectrum_response_template=(
                None
                if data.spectrum_response_template is None
                else np.asarray(
                    data.spectrum_response_template,
                    dtype=float,
                )[row_mask]
            ),
            spectrum_background=(
                None
                if data.spectrum_background is None
                else np.asarray(data.spectrum_background, dtype=float)[row_mask]
            ),
            spectrum_variance=(
                None
                if data.spectrum_variance is None
                else np.asarray(data.spectrum_variance, dtype=float)[row_mask]
            ),
            spectrum_variance_present=(
                None
                if data.spectrum_variance_present is None
                else np.asarray(
                    data.spectrum_variance_present,
                    dtype=bool,
                )[row_mask]
            ),
        )

    @staticmethod
    def _optional_arrays_equal(
        first: NDArray[np.float64] | None,
        second: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether two optional arrays are both absent or exactly equal."""
        if first is None or second is None:
            return first is None and second is None
        return bool(np.array_equal(first, second))

    @staticmethod
    def _same_measurement_block(
        first: MeasurementData | None,
        second: MeasurementData | None,
    ) -> bool:
        """Return whether two bundles contain exactly the same response rows."""
        if first is second:
            return first is not None
        if first is None or second is None:
            return False
        return bool(
            np.array_equal(first.z_k, second.z_k)
            and np.array_equal(
                first.observation_variances,
                second.observation_variances,
            )
            and np.array_equal(
                first.detector_positions,
                second.detector_positions,
            )
            and np.array_equal(first.fe_indices, second.fe_indices)
            and np.array_equal(first.pb_indices, second.pb_indices)
            and np.array_equal(first.live_times, second.live_times)
            and IsotopeParticleFilter._optional_arrays_equal(
                first.station_sequence_ids,
                second.station_sequence_ids,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.runtime_likelihood_routes,
                second.runtime_likelihood_routes,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.observation_count_covariance,
                second.observation_count_covariance,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_counts,
                second.spectrum_counts,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_response_template,
                second.spectrum_response_template,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_background,
                second.spectrum_background,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_variance,
                second.spectrum_variance,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_variance_present,
                second.spectrum_variance_present,
            )
        )

    @staticmethod
    def _isotope_float_config_for(
        value: float | Mapping[str, float],
        isotope: str,
        default: float = 0.0,
    ) -> float:
        """Resolve a scalar or isotope-indexed float for one isotope."""
        if isinstance(value, Mapping):
            return max(float(value.get(str(isotope), default)), 0.0)
        return max(float(value), 0.0)

    def _isotope_float_config(
        self, value: float | dict[str, float], default: float = 0.0
    ) -> float:
        """Resolve a scalar or isotope-indexed float config value."""
        return self._isotope_float_config_for(
            value,
            self.isotope,
            default,
        )

    def _count_likelihood_kwargs(self) -> dict[str, bool | float | str]:
        """Return likelihood keyword arguments for this isotope filter."""
        return {
            "model": str(self.config.count_likelihood_model),
            "transport_model_rel_sigma": self._isotope_float_config(
                self.config.transport_model_rel_sigma,
            ),
            "transport_model_abs_sigma": self._isotope_float_config(
                self.config.transport_model_abs_sigma,
            ),
            "spectrum_count_rel_sigma": self._isotope_float_config(
                self.config.spectrum_count_rel_sigma,
            ),
            "spectrum_count_abs_sigma": self._isotope_float_config(
                self.config.spectrum_count_abs_sigma,
            ),
            "low_count_abs_sigma": self._isotope_float_config(
                self.config.low_count_abs_sigma,
            ),
            "low_count_transition_counts": self._isotope_float_config(
                self.config.low_count_transition_counts,
            ),
            "observation_count_variance_includes_counting_noise": bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            "observation_count_variance_semantics": str(
                self.config.observation_count_variance_semantics
            ),
            "student_t_df": max(float(self.config.count_likelihood_df), 1.0),
        }

    def count_likelihood_spec(self) -> CountLikelihoodSpec:
        """Return the resolved isotope-specific likelihood configuration."""
        return CountLikelihoodSpec(**self._count_likelihood_kwargs())

    def _count_log_likelihood_np(
        self,
        z_k: NDArray[np.float64],
        lambda_k: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> float:
        """Evaluate this filter's configured count log-likelihood in NumPy."""
        return count_log_likelihood(
            z_k,
            lambda_k,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _count_log_likelihood_matrix_np(
        self,
        z_k: NDArray[np.float64],
        lambda_kp: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        """Evaluate per-particle count log-likelihoods for a KxP lambda matrix."""
        z_arr = np.asarray(z_k, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != z_arr.size:
            raise ValueError("lambda_kp must have one row per measurement.")
        obs_var = np.asarray(observation_count_variance, dtype=float).reshape(-1)
        if obs_var.size == 0:
            obs_var = np.zeros(z_arr.size, dtype=float)
        elif obs_var.size == 1:
            obs_var = np.full(z_arr.size, float(obs_var[0]), dtype=float)
        elif obs_var.size != z_arr.size:
            raise ValueError(
                "observation_count_variance must be scalar or have one value per measurement."
            )
        kwargs = self._count_likelihood_kwargs()
        model = normalize_count_likelihood_model(str(kwargs["model"]))
        z_col = z_arr[:, None]
        if model == "poisson":
            return np.sum(z_col * np.log(lam) - lam, axis=0)
        variance = count_likelihood_variance(
            z_col,
            lam,
            transport_model_rel_sigma=float(kwargs["transport_model_rel_sigma"]),
            transport_model_abs_sigma=float(kwargs["transport_model_abs_sigma"]),
            spectrum_count_rel_sigma=float(kwargs["spectrum_count_rel_sigma"]),
            spectrum_count_abs_sigma=float(kwargs["spectrum_count_abs_sigma"]),
            low_count_abs_sigma=float(kwargs["low_count_abs_sigma"]),
            low_count_transition_counts=float(kwargs["low_count_transition_counts"]),
            observation_count_variance=obs_var[:, None],
            observation_count_variance_includes_counting_noise=bool(
                kwargs["observation_count_variance_includes_counting_noise"]
            ),
            observation_count_variance_semantics=str(
                kwargs["observation_count_variance_semantics"]
            ),
        )
        residual = z_col - lam
        if model == "gaussian":
            terms = -0.5 * ((residual**2) / variance + np.log(variance))
            return np.sum(terms, axis=0)
        df = max(float(kwargs["student_t_df"]), 1.0 + 1.0e-12)
        terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
        terms -= 0.5 * np.log(variance)
        return np.sum(terms, axis=0)

    def _structural_effective_variance_np(
        self,
        z_k: NDArray[np.float64],
        lambda_kp: NDArray[np.float64],
        observation_count_variance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the configured marginal count variance for structural evidence."""
        z_arr = np.asarray(z_k, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != z_arr.size:
            raise ValueError("lambda_kp must have one row per measurement.")
        obs_var = self._measurement_vector(
            observation_count_variance,
            z_arr.size,
            "observation_count_variance",
            min_value=0.0,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "poisson":
            return lam
        kwargs = self._count_likelihood_kwargs()
        return count_likelihood_variance(
            z_arr[:, None],
            lam,
            transport_model_rel_sigma=float(kwargs["transport_model_rel_sigma"]),
            transport_model_abs_sigma=float(kwargs["transport_model_abs_sigma"]),
            spectrum_count_rel_sigma=float(kwargs["spectrum_count_rel_sigma"]),
            spectrum_count_abs_sigma=float(kwargs["spectrum_count_abs_sigma"]),
            low_count_abs_sigma=float(kwargs["low_count_abs_sigma"]),
            low_count_transition_counts=float(kwargs["low_count_transition_counts"]),
            observation_count_variance=obs_var[:, None],
            observation_count_variance_includes_counting_noise=bool(
                kwargs["observation_count_variance_includes_counting_noise"]
            ),
            observation_count_variance_semantics=str(
                kwargs["observation_count_variance_semantics"]
            ),
        )

    def _structural_spectrum_log_likelihood_matrix_np(
        self,
        lambda_kp: NDArray[np.float64],
        observed_spectrum_kb: NDArray[np.float64],
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
        spectrum_variance_kb: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the runtime direct-spectrum likelihood for structural states."""
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        observed = np.asarray(observed_spectrum_kb, dtype=float)
        template = np.asarray(response_template_kb, dtype=float)
        background = np.asarray(background_spectrum_kb, dtype=float)
        if observed.ndim != 2 or observed.shape[0] != lam.shape[0]:
            raise ValueError("observed spectrum rows must match expected counts.")
        if template.shape != observed.shape or background.shape != observed.shape:
            raise ValueError(
                "spectrum template and background must match observations."
            )
        spectrum_variance = None
        if spectrum_variance_kb is not None:
            spectrum_variance = np.asarray(spectrum_variance_kb, dtype=float)
            if spectrum_variance.shape != observed.shape:
                raise ValueError("spectrum variance must match observed bins.")

        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        rel_sigma = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        abs_sigma = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        chunk_size = max(1, int(self.config.spectrum_likelihood_bin_chunk))
        result = np.zeros(int(lam.shape[1]), dtype=float)
        for start in range(0, int(observed.shape[1]), chunk_size):
            stop = min(start + chunk_size, int(observed.shape[1]))
            expected = np.maximum(
                lam[:, None, :] * template[:, start:stop, None]
                + background[:, start:stop, None],
                1.0e-12,
            )
            observed_chunk = observed[:, start:stop, None]
            if model == "poisson" and spectrum_variance is None:
                result += np.sum(
                    observed_chunk * np.log(expected) - expected,
                    axis=(0, 1),
                )
                continue
            observation_variance = (
                0.0
                if spectrum_variance is None
                else spectrum_variance[:, start:stop, None]
            )
            variance = np.maximum(
                expected
                + observation_variance
                + (float(rel_sigma) * expected) ** 2
                + float(abs_sigma) ** 2,
                1.0e-12,
            )
            residual = observed_chunk - expected
            if model == "gaussian":
                terms = -0.5 * ((residual**2) / variance + np.log(variance))
            else:
                df = max(
                    float(self.config.count_likelihood_df),
                    1.0 + 1.0e-12,
                )
                terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
                terms -= 0.5 * np.log(variance)
            result += np.sum(terms, axis=(0, 1))
        return result

    def _shield_shape_log_likelihood_batch_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        row_indices_bk: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return runtime-equivalent shield-shape evidence for station batches."""
        rows = np.asarray(row_indices_bk, dtype=np.int64)
        lam_all = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if rows.ndim != 2 or lam_all.ndim != 2:
            raise ValueError("Station rows and expected counts must be matrices.")
        block_count, view_count = rows.shape
        particle_count = int(lam_all.shape[1])
        result = np.zeros((block_count, particle_count), dtype=float)
        if block_count == 0 or particle_count == 0:
            return np.zeros(particle_count, dtype=float)

        z_bk = np.asarray(data.z_k, dtype=float)[rows]
        var_bk = np.asarray(data.observation_variances, dtype=float)[rows]
        lam_bkp = lam_all[rows, :]

        contrast_weight = max(
            float(self.config.shield_contrast_likelihood_weight),
            0.0,
        )
        contrast_min_views = max(
            int(self.config.shield_contrast_min_views),
            2,
        )
        if (
            bool(self.config.shield_contrast_likelihood_enable)
            and contrast_weight > 0.0
            and view_count >= contrast_min_views
        ):
            min_count = max(
                float(self.config.shield_contrast_min_count),
                1.0e-6,
            )
            sigma_floor = max(
                float(self.config.shield_contrast_log_sigma_floor),
                1.0e-6,
            )
            sigma_ceiling = max(
                float(self.config.shield_contrast_log_sigma_ceiling),
                sigma_floor,
            )
            df = max(
                float(self.config.shield_contrast_likelihood_df),
                1.0 + 1.0e-12,
            )
            z_safe = np.maximum(z_bk, min_count)
            lam_safe = np.maximum(lam_bkp, min_count)
            log_z = np.log(z_safe)[:, :, None]
            log_lam = np.log(lam_safe)
            log_var = np.clip(
                var_bk / np.maximum(z_safe**2, 1.0e-12) + sigma_floor**2,
                sigma_floor**2,
                sigma_ceiling**2,
            )[:, :, None]
            view_weight = np.reciprocal(log_var)
            weight_sum = np.maximum(np.sum(view_weight, axis=1, keepdims=True), 1e-12)
            observed_center = log_z - (
                np.sum(view_weight * log_z, axis=1, keepdims=True) / weight_sum
            )
            predicted_center = log_lam - (
                np.sum(view_weight * log_lam, axis=1, keepdims=True) / weight_sum
            )
            residual = observed_center - predicted_center
            terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * log_var))
            terms -= 0.5 * np.log(log_var)
            result += contrast_weight * np.sum(terms, axis=1)

        ratio_weight = max(
            float(self.config.shield_view_ratio_likelihood_weight),
            0.0,
        )
        ratio_min_views = max(
            int(self.config.shield_view_ratio_likelihood_min_views),
            2,
        )
        if (
            bool(self.config.shield_view_ratio_likelihood_enable)
            and ratio_weight > 0.0
            and view_count >= ratio_min_views
        ):
            concentration = max(
                float(self.config.shield_view_ratio_likelihood_concentration),
                1.0e-6,
            )
            min_total = max(
                float(self.config.shield_view_ratio_likelihood_min_total_count),
                0.0,
            )
            z_nonnegative = np.maximum(
                np.where(np.isfinite(z_bk), z_bk, 0.0),
                0.0,
            )
            totals = np.sum(z_nonnegative, axis=1)
            valid = np.isfinite(totals) & (totals >= min_total)
            if np.any(valid):
                lam_valid = lam_bkp[valid]
                probabilities = lam_valid / np.maximum(
                    np.sum(lam_valid, axis=1, keepdims=True),
                    1.0e-12,
                )
                alpha = np.maximum(concentration * probabilities, 1.0e-12)
                alpha0 = np.maximum(np.sum(alpha, axis=1), 1.0e-12)
                total_valid = totals[valid, None]
                ratio_ll = gammaln(alpha0) - gammaln(alpha0 + total_valid)
                ratio_ll += np.sum(
                    gammaln(alpha + z_nonnegative[valid, :, None]) - gammaln(alpha),
                    axis=1,
                )
                result[valid] += ratio_weight * ratio_ll
        return np.sum(result, axis=0)

    def _structural_shield_shape_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return shield-shape terms grouped exactly by station sequence."""
        lam = np.asarray(lambda_kp, dtype=float)
        if lam.ndim == 1:
            lam = lam[:, None]
        result = np.zeros(int(lam.shape[1]), dtype=float)
        if int(data.z_k.size) < 2 or not (
            self.config.shield_contrast_likelihood_enable
            or self.config.shield_view_ratio_likelihood_enable
        ):
            return result
        blocks_by_length = self._station_likelihood_block_rows(data)
        for block_length, rows in blocks_by_length.items():
            if int(block_length) <= 1:
                continue
            result += self._shield_shape_log_likelihood_batch_np(
                data,
                lam,
                rows,
            )
        return result

    @staticmethod
    def _contiguous_station_blocks(
        detector_positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return start indices and lengths for contiguous same-position views."""
        positions = np.asarray(detector_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            count = int(positions.shape[0]) if positions.ndim > 0 else 0
            return (
                np.arange(count, dtype=np.int64),
                np.ones(count, dtype=np.int64),
            )
        count = int(positions.shape[0])
        if count == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        rounded = np.round(positions, decimals=8)
        changes = np.any(rounded[1:] != rounded[:-1], axis=1)
        starts = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.flatnonzero(changes).astype(np.int64) + 1,
            ]
        )
        stops = np.concatenate([starts[1:], np.array([count], dtype=np.int64)])
        return starts, stops - starts

    def _station_likelihood_block_rows(
        self,
        data: MeasurementData,
    ) -> dict[int, NDArray[np.int64]]:
        """
        Return station likelihood rows batched by equal block length.

        Estimator-produced data carries explicit sequence IDs that reproduce the
        runtime update boundary exactly. Directly constructed legacy bundles
        without IDs retain the contiguous same-position fallback.
        """
        measurement_count = int(data.z_k.size)
        sequence_ids = data.station_sequence_ids
        if sequence_ids is None:
            starts, lengths = self._contiguous_station_blocks(
                data.detector_positions
            )
            labels = np.repeat(
                np.arange(starts.size, dtype=np.int64),
                lengths,
            )
        else:
            ids = np.asarray(sequence_ids, dtype=np.int64).reshape(-1)
            if ids.size != measurement_count:
                raise ValueError(
                    "station_sequence_ids must contain one ID per measurement."
                )
            if measurement_count == 0:
                return {}
            _, labels = np.unique(ids, return_inverse=True)
            labels = labels.astype(np.int64, copy=False)
        if labels.size != measurement_count:
            raise ValueError("Station likelihood block labels are inconsistent.")
        if measurement_count == 0:
            return {}
        _, inverse, lengths = np.unique(
            labels,
            return_inverse=True,
            return_counts=True,
        )
        rows_by_length: dict[int, NDArray[np.int64]] = {}
        for block_length in np.unique(lengths):
            selected_blocks = np.flatnonzero(lengths == int(block_length))
            membership = inverse[None, :] == selected_blocks[:, None]
            rows = np.nonzero(membership)[1].reshape(
                int(selected_blocks.size),
                int(block_length),
            )
            rows_by_length[int(block_length)] = rows.astype(
                np.int64,
                copy=False,
            )
        return rows_by_length

    @staticmethod
    def _regularize_station_covariance_np(
        covariance_bpkk: NDArray[np.float64],
        diagonal_bpk: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return positive-definite station covariance using the runtime policy.

        Each station is regularized independently because runtime observes one
        station sequence at a time. Particles remain batched within a station,
        matching ``_regularize_sequence_covariance_torch`` exactly.
        """
        covariance = np.asarray(covariance_bpkk, dtype=float)
        diagonal = np.asarray(diagonal_bpk, dtype=float)
        if covariance.ndim != 4 or diagonal.shape != covariance.shape[:3]:
            raise ValueError("Station covariance arrays have incompatible shapes.")
        size = int(covariance.shape[-1])
        eye = np.eye(size, dtype=float).reshape(1, size, size)
        regularized_blocks: list[NDArray[np.float64]] = []
        for block_index in range(int(covariance.shape[0])):
            block = covariance[block_index]
            block_diagonal = diagonal[block_index]
            diag_scale = np.maximum(np.mean(block_diagonal, axis=1), 1.0)
            jitter = 1.0e-9 * diag_scale
            accepted: NDArray[np.float64] | None = None
            for _ in range(6):
                candidate = block + jitter[:, None, None] * eye
                try:
                    np.linalg.cholesky(candidate)
                except np.linalg.LinAlgError:
                    jitter = jitter * 10.0
                    continue
                accepted = candidate
                break
            if accepted is None:
                fallback = np.zeros_like(block, dtype=float)
                diag_indices = np.arange(size)
                fallback[:, diag_indices, diag_indices] = np.maximum(
                    block_diagonal,
                    1.0e-12,
                )
                accepted = fallback + jitter[:, None, None] * eye
            regularized_blocks.append(accepted)
        if not regularized_blocks:
            return np.zeros_like(covariance, dtype=float)
        return np.stack(regularized_blocks, axis=0)

    def _station_covariance_log_likelihood_batch_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        row_indices_bk: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Return summed likelihoods for a batch of equal-size station blocks.

        Arrays are batched over both station blocks and PF particles. No scalar
        particle, candidate, source-slot, or shield-view loop is used.
        """
        rows = np.asarray(row_indices_bk, dtype=np.int64)
        lam_all = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if rows.ndim != 2 or lam_all.ndim != 2:
            raise ValueError("Station rows and expected counts must be matrices.")
        block_count, view_count = rows.shape
        particle_count = int(lam_all.shape[1])
        z_bk = np.asarray(data.z_k, dtype=float)[rows]
        obs_var_bk = np.asarray(data.observation_variances, dtype=float)[rows]
        lam_bkp = lam_all[rows, :]
        variance_bkp = self._structural_effective_variance_np(
            z_bk.reshape(-1),
            lam_bkp.reshape(block_count * view_count, particle_count),
            obs_var_bk.reshape(-1),
        ).reshape(block_count, view_count, particle_count)
        variance_bpk = np.transpose(variance_bkp, (0, 2, 1))
        covariance = np.zeros(
            (block_count, particle_count, view_count, view_count),
            dtype=float,
        )
        diag_indices = np.arange(view_count)
        covariance[:, :, diag_indices, diag_indices] = variance_bpk

        supplied = data.observation_count_covariance
        if supplied is not None:
            supplied_arr = np.asarray(supplied, dtype=float)
            expected_shape = (int(data.z_k.size), int(data.z_k.size))
            if supplied_arr.shape != expected_shape:
                raise ValueError("observation_count_covariance must be shaped K x K.")
            supplied_blocks = supplied_arr[rows[:, :, None], rows[:, None, :]]
            supplied_blocks = 0.5 * (
                supplied_blocks + np.swapaxes(supplied_blocks, 1, 2)
            )
            supplied_blocks[:, diag_indices, diag_indices] = 0.0
            covariance += supplied_blocks[:, None, :, :]

        fraction = max(
            float(self.config.station_view_correlated_spectrum_fraction),
            0.0,
        )
        if bool(self.config.station_view_covariance_enable) and fraction > 0.0:
            spectrum_rel = fraction * self._isotope_float_config(
                self.config.spectrum_count_rel_sigma
            )
            spectrum_abs = fraction * self._isotope_float_config(
                self.config.spectrum_count_abs_sigma
            )
            lam_bpk = np.transpose(lam_bkp, (0, 2, 1))
            common = (spectrum_rel**2) * (
                lam_bpk[:, :, :, None] * lam_bpk[:, :, None, :]
            )
            if spectrum_abs > 0.0:
                common = common + spectrum_abs**2
            common[:, :, diag_indices, diag_indices] = 0.0
            covariance += common

        covariance = self._regularize_station_covariance_np(
            covariance,
            variance_bpk,
        )
        residual = (z_bk[:, None, :] - np.transpose(lam_bkp, (0, 2, 1)))[..., None]
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, residual)
        quadratic = np.sum(whitened[..., 0] ** 2, axis=2)
        logdet = 2.0 * np.sum(
            np.log(np.diagonal(chol, axis1=2, axis2=3)),
            axis=2,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "gaussian":
            block_ll = -0.5 * (quadratic + logdet)
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            block_ll = (
                -0.5 * (df + float(view_count)) * np.log1p(quadratic / df)
                - 0.5 * logdet
            )
        return np.sum(block_ll, axis=0)

    def _structural_count_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate structural evidence using each recorded runtime route."""
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != int(data.z_k.size):
            raise ValueError("lambda_kp must have one row per measurement.")
        if data.runtime_likelihood_routes is None:
            return self._structural_single_route_log_likelihood_matrix_np(
                data,
                lam,
                use_direct_spectrum=None,
            )
        routes = np.asarray(data.runtime_likelihood_routes, dtype=str).reshape(-1)
        if routes.size != int(data.z_k.size):
            raise ValueError(
                "runtime_likelihood_routes must contain one route per measurement."
            )
        allowed_routes = np.isin(
            routes,
            np.asarray(
                ["count", "count_covariance", "direct_spectrum"],
                dtype=str,
            ),
        )
        if not np.all(allowed_routes):
            invalid = np.unique(routes[~allowed_routes]).tolist()
            raise ValueError(f"Unsupported runtime likelihood routes: {invalid}.")
        for rows in self._station_likelihood_block_rows(data).values():
            block_routes = routes[rows]
            if np.any(block_routes != block_routes[:, :1]):
                raise ValueError(
                    "Rows in one station sequence must share one runtime "
                    "likelihood route."
                )
        variance_presence = data.spectrum_variance_present
        if variance_presence is not None:
            variance_presence = np.asarray(
                variance_presence,
                dtype=bool,
            ).reshape(-1)
            if variance_presence.size != int(data.z_k.size):
                raise ValueError(
                    "spectrum_variance_present must contain one flag per "
                    "measurement."
                )
            for rows in self._station_likelihood_block_rows(data).values():
                direct_blocks = routes[rows[:, 0]] == "direct_spectrum"
                if np.any(direct_blocks):
                    block_presence = variance_presence[rows[direct_blocks]]
                    if np.any(block_presence != block_presence[:, :1]):
                        raise ValueError(
                            "Rows in one direct-spectrum station sequence must "
                            "share spectrum-variance semantics."
                        )
        result = np.zeros(int(lam.shape[1]), dtype=float)
        direct_mask = routes == "direct_spectrum"
        if np.any(direct_mask):
            if variance_presence is None:
                direct_data = self._measurement_rows(data, direct_mask)
                result += self._structural_single_route_log_likelihood_matrix_np(
                    direct_data,
                    lam[direct_mask, :],
                    use_direct_spectrum=True,
                )
            else:
                for uses_variance in (False, True):
                    direct_variance_mask = direct_mask & (
                        variance_presence == uses_variance
                    )
                    if not np.any(direct_variance_mask):
                        continue
                    direct_data = self._measurement_rows(
                        data,
                        direct_variance_mask,
                    )
                    result += (
                        self._structural_single_route_log_likelihood_matrix_np(
                            direct_data,
                            lam[direct_variance_mask, :],
                            use_direct_spectrum=True,
                            use_spectrum_variance=uses_variance,
                        )
                    )
        count_covariance_mask = routes == "count_covariance"
        if np.any(count_covariance_mask):
            count_covariance_data = self._measurement_rows(
                data,
                count_covariance_mask,
            )
            result += self._structural_single_route_log_likelihood_matrix_np(
                count_covariance_data,
                lam[count_covariance_mask, :],
                use_direct_spectrum=False,
                use_count_covariance=True,
            )
        count_mask = routes == "count"
        if np.any(count_mask):
            count_data = self._measurement_rows(data, count_mask)
            result += self._structural_single_route_log_likelihood_matrix_np(
                count_data,
                lam[count_mask, :],
                use_direct_spectrum=False,
                use_count_covariance=False,
            )
        return result

    def _structural_single_route_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        *,
        use_direct_spectrum: bool | None,
        use_spectrum_variance: bool | None = None,
        use_count_covariance: bool | None = None,
    ) -> NDArray[np.float64]:
        """
        Evaluate one homogeneous runtime likelihood route in a single batch.

        Direct spectrum payloads use the same independent-bin model as runtime.
        Otherwise same-position shield sequences use the configured multivariate
        Student-t/Gaussian count covariance.
        """
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != int(data.z_k.size):
            raise ValueError("lambda_kp must have one row per measurement.")
        spectrum_arrays = self._spectrum_update_arrays(
            data.spectrum_counts,
            data.spectrum_response_template,
            data.spectrum_background,
            (
                data.spectrum_variance
                if use_spectrum_variance is not False
                else None
            ),
            sequence_length=int(data.z_k.size),
        )
        direct_route = (
            spectrum_arrays is not None
            and self._direct_spectrum_route_admissible(
                sequence_length=int(data.z_k.size),
                observation_count_covariance=data.observation_count_covariance,
            )
            if use_direct_spectrum is None
            else bool(use_direct_spectrum)
        )
        if direct_route and spectrum_arrays is None:
            raise ValueError(
                "The recorded direct-spectrum route lacks complete spectrum arrays."
            )
        if direct_route:
            assert spectrum_arrays is not None
            observed, template, background, spectrum_variance = spectrum_arrays
            if use_spectrum_variance is True and spectrum_variance is None:
                raise ValueError(
                    "The recorded direct-spectrum route used a variance array, "
                    "but the history lacks it."
                )
            return self._structural_spectrum_log_likelihood_matrix_np(
                lam,
                observed,
                template,
                background,
                spectrum_variance,
            ) + self._structural_shield_shape_log_likelihood_matrix_np(
                data,
                lam,
            )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        shield_shape_enabled = bool(
            self.config.shield_contrast_likelihood_enable
            or self.config.shield_view_ratio_likelihood_enable
        )
        if model == "poisson" and not shield_shape_enabled:
            return self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        if use_count_covariance is None:
            covariance_enabled = model != "poisson" and (
                data.observation_count_covariance is not None
                or (
                    bool(self.config.station_view_covariance_enable)
                    and float(self.config.station_view_correlated_spectrum_fraction)
                    > 0.0
                )
            )
        else:
            covariance_enabled = model != "poisson" and bool(
                use_count_covariance
            )
        if (not covariance_enabled and not shield_shape_enabled) or int(
            data.z_k.size
        ) < 2:
            return self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )

        blocks_by_length = self._station_likelihood_block_rows(data)
        if covariance_enabled:
            result = np.zeros(int(lam.shape[1]), dtype=float)
        else:
            result = self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        single_rows = blocks_by_length.get(1)
        if covariance_enabled and single_rows is not None and single_rows.size:
            single_indices = single_rows[:, 0]
            result += self._count_log_likelihood_matrix_np(
                np.asarray(data.z_k, dtype=float)[single_indices],
                lam[single_indices, :],
                observation_count_variance=np.asarray(
                    data.observation_variances,
                    dtype=float,
                )[single_indices],
            )
        # Station programs normally share one small view count. Grouping by
        # length keeps station and particle evaluation batched while preserving
        # the product of station-level multivariate likelihoods.
        for block_length, rows in blocks_by_length.items():
            if int(block_length) <= 1:
                continue
            if covariance_enabled:
                result += self._station_covariance_log_likelihood_batch_np(
                    data,
                    lam,
                    rows,
                )
            if shield_shape_enabled:
                result += self._shield_shape_log_likelihood_batch_np(
                    data,
                    lam,
                    rows,
                )
        return result

    def _structural_count_log_likelihood_np(
        self,
        data: MeasurementData,
        lambda_k: NDArray[np.float64],
    ) -> float:
        """Return one state's count likelihood under structural PF semantics."""
        values = self._structural_count_log_likelihood_matrix_np(
            data,
            np.asarray(lambda_k, dtype=float).reshape(-1, 1),
        )
        return float(values[0]) if values.size else 0.0

    def _structural_delta_log_likelihood_remove(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
        lambda_components: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched leave-one-source-out likelihood losses."""
        total = np.asarray(lambda_total, dtype=float).reshape(-1)
        components = np.asarray(lambda_components, dtype=float)
        if components.ndim != 2 or components.shape[0] != total.size:
            return np.zeros(0, dtype=float)
        source_count = int(components.shape[1])
        if source_count == 0:
            return np.zeros(0, dtype=float)
        base_ll = self._structural_count_log_likelihood_np(data, total)
        reduced = np.maximum(total[:, None] - components, 1.0e-12)
        reduced_ll = self._structural_count_log_likelihood_matrix_np(
            data,
            reduced,
        )
        return float(base_ll) - np.asarray(reduced_ll, dtype=float)

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a discrete kernel and refresh only changed continuous physics."""
        incoming_signature = self._incoming_kernel_physics_signature(kernel)
        self.kernel = kernel
        if incoming_signature == self._continuous_kernel_physics_signature:
            return
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )
        self._continuous_kernel_physics_signature = incoming_signature
        self._structural_rj_response_cache = None
        self._structural_rj_response_cache_signatures = None

    def _source_prior_is_surface(self) -> bool:
        """Return True when source positions should be constrained to surfaces."""
        raw = getattr(self.config, "source_position_prior", "volume")
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {
            "surface",
            "surfaces",
            "surface_constrained",
            "surface-constrained",
        }

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by the source-position prior."""
        hi = np.array(self.config.position_max, dtype=float)
        if hi.shape != (3,):
            raise ValueError("position_max must be a 3-element vector.")
        if np.any(hi <= 0.0):
            raise ValueError("position_max must define positive room dimensions.")
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    def _surface_grid_positions(self, spacing: float) -> NDArray[np.float64]:
        """Return cached source-position candidates on allowed surfaces."""
        key = float(spacing)
        cached = self._surface_candidate_cache.get(key)
        if cached is not None:
            return cached
        candidates = build_surface_candidate_sources(
            self._source_prior_environment(),
            self.obstacle_grid,
            (key, key, key),
            position_min=self.config.position_min,
            position_max=self.config.position_max,
            obstacle_height_m=self.obstacle_height_m,
        )
        self._surface_candidate_cache[key] = candidates
        return candidates

    def _project_positions_to_source_prior(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Clip or project source positions according to the configured prior."""
        arr = np.asarray(positions, dtype=float)
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        clipped = np.clip(arr, lo, hi)
        if not self._source_prior_is_surface() or clipped.size == 0:
            return clipped
        projected = project_positions_to_allowed_surfaces(
            clipped,
            self._source_prior_environment(),
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        return np.clip(projected, lo, hi)

    def _sample_prior_positions(self, count: int) -> NDArray[np.float64]:
        """Sample initial source positions from the configured position prior."""
        source_count = max(0, int(count))
        if source_count <= 0:
            return np.zeros((0, 3), dtype=float)
        if self._source_prior_is_surface():
            spacing = self.config.init_grid_spacing_m
            spacing_f = 1.0 if spacing is None else max(float(spacing), 1.0e-6)
            candidates = self._surface_grid_positions(spacing_f)
            replace = candidates.shape[0] < source_count
            indices = np.random.choice(
                candidates.shape[0],
                size=source_count,
                replace=replace,
            )
            return candidates[indices].copy()
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        return lo + np.random.rand(source_count, 3) * (hi - lo)

    def _initial_grid_positions(self) -> NDArray[np.float64]:
        """Return initial grid-center positions when grid init is enabled."""
        spacing = self.config.init_grid_spacing_m
        if spacing is None:
            return np.zeros((0, 3), dtype=float)
        spacing = float(spacing)
        if spacing <= 0.0:
            return np.zeros((0, 3), dtype=float)
        if self._source_prior_is_surface():
            return self._surface_grid_positions(spacing)
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        starts = lo + spacing * 0.5
        xs = np.arange(starts[0], hi[0], spacing)
        ys = np.arange(starts[1], hi[1], spacing)
        zs = np.arange(starts[2], hi[2], spacing)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return np.zeros((0, 3), dtype=float)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        return grid.reshape(-1, 3)

    def _initial_source_count_for_particle(self, particle_index: int) -> int:
        """Return one cyclic source count as a small deterministic test oracle."""
        min_r, max_r = self._initial_source_count_bounds()
        if max_r <= min_r:
            return min_r
        span = max_r - min_r + 1
        return min_r + (int(particle_index) % span)

    def _initial_source_count_bounds(self) -> tuple[int, int]:
        """Return normalized inclusive bounds for initial source cardinality."""
        min_r, max_r = self.config.init_num_sources
        min_r = max(0, int(min_r))
        max_r = max(min_r, int(max_r))
        if self.config.max_sources is not None:
            max_r = min(max_r, max(0, int(self.config.max_sources)))
            min_r = min(min_r, max_r)
        return min_r, max_r

    def _initial_source_counts_for_particles(
        self,
        particle_count: int,
        *,
        cyclic: bool,
    ) -> NDArray[np.int64]:
        """Draw all initial source counts in one batched NumPy operation."""
        count = max(0, int(particle_count))
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        min_r, max_r = self._initial_source_count_bounds()
        if max_r <= min_r:
            return np.full(count, min_r, dtype=np.int64)
        if cyclic:
            span = max_r - min_r + 1
            return min_r + (np.arange(count, dtype=np.int64) % span)
        return np.asarray(
            np.random.randint(min_r, max_r + 1, size=count),
            dtype=np.int64,
        )

    def _initial_grid_state_positions(
        self,
        anchor_position: NDArray[np.float64],
        source_count: int,
        grid_positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one legacy grid tuple as a small scalar test oracle."""
        count = max(0, int(source_count))
        if count <= 0:
            return np.zeros((0, 3), dtype=float)
        anchor = np.asarray(anchor_position, dtype=float).reshape(1, 3)
        if count == 1:
            return anchor.copy()
        grid = np.asarray(grid_positions, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
            extra = self._sample_prior_positions(count - 1)
            return np.vstack([anchor, extra])
        replace = grid.shape[0] < count - 1
        extra_idx = np.random.choice(grid.shape[0], size=count - 1, replace=replace)
        return np.vstack([anchor, grid[extra_idx]])

    def _initial_grid_state_positions_batched(
        self,
        anchor_positions: NDArray[np.float64],
        source_counts: NDArray[np.int64],
        grid_positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched, diversity-aware source tuples for grid initialization."""
        anchors = np.asarray(anchor_positions, dtype=float).reshape(-1, 3)
        counts = np.asarray(source_counts, dtype=np.int64).reshape(-1)
        grid = np.asarray(grid_positions, dtype=float).reshape(-1, 3)
        if anchors.shape[0] != counts.size:
            raise ValueError("source_counts must match anchor_positions.")
        if grid.shape[0] == 0:
            raise ValueError("grid_positions must be non-empty.")
        if np.any(counts < 0):
            raise ValueError("source_counts must be non-negative.")
        particle_count = int(anchors.shape[0])
        max_sources = int(np.max(counts)) if counts.size else 0
        if max_sources <= 0:
            return np.zeros((particle_count, 0, 3), dtype=float)
        retry_count = max(int(self.config.init_joint_position_retries), 1)
        extra_slots = max_sources - 1
        anchor_block = np.broadcast_to(
            anchors[:, None, None, :],
            (particle_count, retry_count, 1, 3),
        )
        if extra_slots <= 0:
            return anchor_block[:, 0, :, :].copy()

        design = (
            str(self.config.init_joint_position_design)
            .strip()
            .lower()
            .replace("-", "_")
        )
        sample_shape = (particle_count, retry_count, extra_slots)
        if design == "latin_hypercube":
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
            unit = qmc.LatinHypercube(d=extra_slots, seed=seed).random(
                particle_count * retry_count
            )
            unit = np.asarray(unit, dtype=float).reshape(sample_shape)
            indices = np.minimum(
                np.floor(unit * grid.shape[0]).astype(np.int64),
                grid.shape[0] - 1,
            )
        elif design == "independent":
            indices = np.random.randint(0, grid.shape[0], size=sample_shape)
        else:
            raise ValueError(
                "init_joint_position_design must be independent or latin_hypercube."
            )
        extra_positions = grid[indices]
        tuples = np.concatenate([anchor_block, extra_positions], axis=2)

        min_separation = max(float(self.config.init_source_min_separation_m), 0.0)
        if min_separation <= 0.0 or max_sources <= 1:
            return tuples[:, 0, :, :].copy()
        selected_retry = self._select_initial_tuple_retry_indices(
            tuples,
            counts,
            min_separation_m=min_separation,
        )
        return tuples[np.arange(particle_count), selected_retry, :, :].copy()

    @staticmethod
    def _select_initial_tuple_retry_indices(
        tuple_positions: NDArray[np.float64],
        source_counts: NDArray[np.int64],
        *,
        min_separation_m: float,
    ) -> NDArray[np.int64]:
        """Select the first valid retry per particle using batched pair distances."""
        tuples = np.asarray(tuple_positions, dtype=float)
        counts = np.asarray(source_counts, dtype=np.int64).reshape(-1)
        if tuples.ndim != 4 or tuples.shape[-1] != 3:
            raise ValueError("tuple_positions must have shape P x R x S x 3.")
        particle_count, retry_count, source_slots, _ = tuples.shape
        if counts.size != particle_count:
            raise ValueError("source_counts must match tuple_positions particles.")
        if retry_count <= 0:
            raise ValueError("tuple_positions must include at least one retry.")
        if np.any(counts < 0) or np.any(counts > source_slots):
            raise ValueError("source_counts must lie within tuple source slots.")
        if source_slots <= 1 or float(min_separation_m) <= 0.0:
            return np.zeros(particle_count, dtype=np.int64)

        left_slots, right_slots = np.triu_indices(source_slots, k=1)
        pair_delta = tuples[:, :, left_slots, :] - tuples[:, :, right_slots, :]
        pair_distance_sq = np.einsum(
            "...d,...d->...",
            pair_delta,
            pair_delta,
            optimize=True,
        )
        active_pairs = right_slots[None, :] < counts[:, None]
        relevant_distance_sq = np.where(
            active_pairs[:, None, :],
            pair_distance_sq,
            np.inf,
        )
        min_distance_sq = np.min(relevant_distance_sq, axis=2)
        threshold_sq = max(float(min_separation_m), 0.0) ** 2
        valid = min_distance_sq >= threshold_sq
        selected = np.argmax(valid, axis=1).astype(np.int64, copy=False)
        no_valid = ~np.any(valid, axis=1)
        if np.any(no_valid):
            raise ValueError(
                "Unable to construct separated initial source tuples for "
                f"{int(np.count_nonzero(no_valid))} particles; increase "
                "init_joint_position_retries or revise the physically declared "
                "minimum separation."
            )
        return selected

    def _sample_initial_strengths(
        self,
        shape: int | tuple[int, ...],
    ) -> NDArray[np.float64]:
        """Draw a batched initial-strength array from the configured PF prior."""
        return np.asarray(
            self._strength_prior.sample(
                shape,
                rng=self._random_generator,
            ),
            dtype=float,
        )

    def _exact_initial_cardinality_counts(
        self,
        particle_count: int,
    ) -> NDArray[np.int64]:
        """Allocate at least one exact-mode particle to every positive K prior."""
        count = max(1, int(particle_count))
        probabilities = np.asarray(
            self._structural_rj_cardinality_prior_probs,
            dtype=float,
        )
        positive = probabilities > 0.0
        positive_count = int(np.count_nonzero(positive))
        if count < positive_count:
            raise ValueError(
                "rj_mh needs at least one initial particle for every positive "
                "cardinality-prior entry."
            )
        allocation = np.zeros(probabilities.size, dtype=np.int64)
        allocation[positive] = 1
        remaining = count - positive_count
        if remaining <= 0:
            return allocation
        expected = remaining * probabilities
        extra = np.floor(expected).astype(np.int64)
        allocation += extra
        leftover = remaining - int(np.sum(extra))
        if leftover > 0:
            fractions = expected - extra
            order = np.argsort(fractions, kind="stable")[::-1]
            allocation[order[:leftover]] += 1
        return allocation

    def _init_exact_structural_particles(self) -> None:
        """Initialize the exact PF from its K, surface-set, and strength priors."""
        patches = self._structural_rj_surface_patches
        surface_prior = self._structural_rj_surface_prior
        cardinality_prior = self._structural_rj_cardinality_prior
        if patches is None or surface_prior is None or cardinality_prior is None:
            raise RuntimeError("rj_mh surface and cardinality priors are unavailable.")
        # Exact-mode particles are Monte Carlo samples from the normalized
        # surface-set prior. The legacy grid-repeat setting must not turn the
        # finite support dictionary into one particle per patch: doing so
        # changes the configured particle budget and makes obstacle-rich scenes
        # orders of magnitude larger than requested.
        target_n = int(self.config.num_particles)
        allocation = self._exact_initial_cardinality_counts(target_n)
        particles: list[IsotopeParticle] = []
        for cardinality, cardinality_count in enumerate(allocation.tolist()):
            if cardinality_count <= 0:
                continue
            surface_sets = surface_prior.sample_rejection(
                cardinality,
                cardinality_count,
                rng=self._random_generator,
            )
            strengths = self._sample_initial_strengths(
                (cardinality_count, cardinality)
            )
            per_particle_mass = (
                float(cardinality_prior.probabilities[cardinality])
                / float(cardinality_count)
            )
            log_weight = float(np.log(per_particle_mass))
            for row in range(cardinality_count):
                patch_indices = surface_sets[row]
                positions = np.asarray(
                    patches.centers_xyz[patch_indices],
                    dtype=float,
                ).reshape(cardinality, 3)
                state = IsotopeState(
                    num_sources=cardinality,
                    positions=positions,
                    strengths=np.asarray(strengths[row], dtype=float).copy(),
                    background=self._background_level(),
                    ages=np.zeros(cardinality, dtype=int),
                    support_scores=np.zeros(cardinality, dtype=float),
                    tentative_sources=np.zeros(cardinality, dtype=bool),
                    verification_fail_streaks=np.zeros(cardinality, dtype=int),
                )
                particles.append(
                    IsotopeParticle(state=state, log_weight=log_weight)
                )
        permutation = self._random_generator.permutation(len(particles))
        self.continuous_particles = [particles[int(index)] for index in permutation]
        self.N = len(self.continuous_particles)
        self.config.num_particles = self.N
        self.config.min_particles = self.N
        self.config.max_particles = self.N

    def _init_continuous_particles(self) -> None:
        """Sample continuous positions/strengths/background from broad priors (Sec. 3.3.2)."""
        self.continuous_particles = []
        if self._structural_kernel_is_exact():
            self._init_exact_structural_particles()
            return
        grid_positions = self._initial_grid_positions()
        if grid_positions.size:
            repeat_count = max(1, int(self.config.init_grid_repeats))
            target_n = int(grid_positions.shape[0]) * repeat_count
            self.N = target_n
            self.config.num_particles = target_n
            self.config.min_particles = target_n
            self.config.max_particles = target_n
            repeated_positions = np.repeat(grid_positions, repeat_count, axis=0)
            source_counts = self._initial_source_counts_for_particles(
                target_n,
                cyclic=True,
            )
            tuple_positions = self._initial_grid_state_positions_batched(
                repeated_positions,
                source_counts,
                grid_positions,
            )
            max_initial_sources = (
                int(np.max(source_counts)) if source_counts.size else 0
            )
            strength_draws = self._sample_initial_strengths(
                (target_n, max_initial_sources)
            )
            for particle_idx in range(target_n):
                r_h = int(source_counts[particle_idx])
                if r_h > 0:
                    positions = tuple_positions[particle_idx, :r_h, :].copy()
                    strengths = strength_draws[particle_idx, :r_h].copy()
                    ages = np.zeros(r_h, dtype=int)
                    support_scores = np.zeros(r_h, dtype=float)
                    tentative_sources = np.zeros(r_h, dtype=bool)
                    verification_fail_streaks = np.zeros(r_h, dtype=int)
                else:
                    positions = np.zeros((0, 3), dtype=float)
                    strengths = np.zeros(0, dtype=float)
                    ages = np.zeros(0, dtype=int)
                    support_scores = np.zeros(0, dtype=float)
                    tentative_sources = np.zeros(0, dtype=bool)
                    verification_fail_streaks = np.zeros(0, dtype=int)
                b_h = self._background_level()
                st = IsotopeState(
                    num_sources=r_h,
                    positions=positions,
                    strengths=strengths,
                    background=b_h,
                    ages=ages,
                    support_scores=support_scores,
                    tentative_sources=tentative_sources,
                    verification_fail_streaks=verification_fail_streaks,
                )
                self.continuous_particles.append(
                    IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N)))
                )
            return
        source_counts = self._initial_source_counts_for_particles(
            self.N,
            cyclic=False,
        )
        max_initial_sources = int(np.max(source_counts)) if source_counts.size else 0
        strength_draws = self._sample_initial_strengths((self.N, max_initial_sources))
        for particle_idx in range(self.N):
            r_h = int(source_counts[particle_idx])
            if r_h > 0:
                positions = self._sample_prior_positions(r_h)
                strengths = strength_draws[particle_idx, :r_h].copy()
                ages = np.zeros(r_h, dtype=int)
                support_scores = np.zeros(r_h, dtype=float)
                tentative_sources = np.zeros(r_h, dtype=bool)
                verification_fail_streaks = np.zeros(r_h, dtype=int)
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
                ages = np.zeros(0, dtype=int)
                support_scores = np.zeros(0, dtype=float)
                tentative_sources = np.zeros(0, dtype=bool)
                verification_fail_streaks = np.zeros(0, dtype=int)
            b_h = self._background_level()
            st = IsotopeState(
                num_sources=r_h,
                positions=positions,
                strengths=strengths,
                background=b_h,
                ages=ages,
                support_scores=support_scores,
                tentative_sources=tentative_sources,
                verification_fail_streaks=verification_fail_streaks,
            )
            self.continuous_particles.append(
                IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N)))
            )

    def reset_step_stats(self) -> None:
        """Reset per-step diagnostic counters."""
        reset_step_diagnostics(self)

    def _advance_adapt_cooldown(self) -> None:
        """Decrement the adapt cooldown counter after each update."""
        if self._adapt_cooldown_remaining > 0:
            self._adapt_cooldown_remaining -= 1

    def _trigger_adapt_cooldown(self) -> None:
        """Start the adapt cooldown after a resampling event."""
        steps = max(0, int(self.config.adapt_cooldown_steps))
        if steps > 0:
            self._adapt_cooldown_remaining = max(
                self._adapt_cooldown_remaining, steps + 1
            )

    def _record_source_event(
        self,
        event: str,
        st: IsotopeState,
        source_idx: int,
        *,
        reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Record a source-slot birth, death, quarantine, or verification event."""
        self._ensure_source_metadata(st)
        record = build_source_event_record(
            event=event,
            isotope=self.isotope,
            state=st,
            source_idx=int(source_idx),
            reason=reason,
            extra=extra,
        )
        if record is None:
            return
        self.last_source_event_diagnostics.append(record)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        if not gpu_utils.torch_device_available(self.config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether this filter can use the configured torch device."""
        from pf import gpu_utils

        return bool(
            self.config.use_gpu
            and gpu_utils.torch_device_available(self.config.gpu_device)
        )

    def _ll_proxy_pair(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> float:
        """Return a Poisson log-likelihood proxy for convergence checks."""
        if not self.continuous_particles:
            return 0.0
        state = self.best_particle().state
        lam_rate = float(state.background)
        source_scale = self._measurement_source_scale(
            fe_index=fe_index,
            pb_index=pb_index,
        )
        if state.num_sources > 0:
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
                lam_rate += source_scale * float(kernel_val) * float(strength)
        lam = float(live_time_s) * lam_rate
        return self._count_log_likelihood_np(
            np.array([float(z_obs)], dtype=float),
            np.array([lam], dtype=float),
        )

    def _mmse_primary_position(self) -> NDArray[np.float64] | None:
        """Return the MMSE position for the first source slot, if available."""
        if not self.continuous_particles:
            return None
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size == 0:
            return None
        pos_stack: list[NDArray[np.float64]] = []
        w_stack: list[float] = []
        for weight, particle in zip(weights, self.continuous_particles):
            state = particle.state
            if state.num_sources > 0:
                pos_stack.append(state.positions[0])
                w_stack.append(float(weight))
        if not w_stack:
            return None
        w = np.asarray(w_stack, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.full_like(w, 1.0 / max(len(w), 1))
        else:
            w = w / w_sum
        pos_arr = np.vstack(pos_stack)
        return np.sum(w[:, None] * pos_arr, axis=0)

    def _cardinality_variance(self) -> float:
        """Return posterior variance of the active source count."""
        if not self.continuous_particles:
            return 0.0
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        counts = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=float,
        )
        mean = float(np.sum(weights * counts))
        return float(np.sum(weights * (counts - mean) ** 2))

    def _has_unverified_sources(self) -> bool:
        """Return True if posterior-supported source hypotheses are tentative."""
        if not self.continuous_particles:
            return False
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        support_mass = 0.0
        for weight, particle in zip(weights, self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            if st.num_sources <= 0:
                continue
            tentative = np.asarray(st.tentative_sources[: st.num_sources], dtype=bool)
            failed = np.asarray(
                st.verification_fail_streaks[: st.num_sources], dtype=int
            )
            if np.any(tentative | (failed > 0)):
                support_mass += float(weight)
        return support_mass > 1.0e-3

    def _convergence_state_vector(self) -> NDArray[np.float64] | None:
        """Return a cardinality-aware vector for convergence monitoring."""
        if not self.continuous_particles:
            return None
        if bool(self.config.birth_enable and self.config.use_clustered_output):
            positions, strengths = self.estimate_clustered()
        else:
            positions, strengths = self.estimate()
        if positions.size == 0:
            return None
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        pos_sorted = np.asarray(positions[order], dtype=float)
        str_sorted = np.asarray(strengths, dtype=float).reshape(-1)[order]
        strength_scale = max(float(np.max(np.abs(str_sorted))), 1.0)
        return np.concatenate([pos_sorted.reshape(-1), str_sorted / strength_scale])

    def _cluster_convergence_supported(self) -> bool:
        """
        Return True when each reported cluster is locally supported and compact.

        Isotope-level convergence can hide one stable strong source and one
        drifting weak cluster.  This guard keeps updates active until each
        output cluster has enough posterior mass nearby and, when configured, a
        bounded spatial spread.
        """
        max_spread = max(float(self.config.converge_cluster_spread_max_m), 0.0)
        min_support = max(
            float(self.config.converge_cluster_min_support_fraction),
            0.0,
        )
        if max_spread <= 0.0 and min_support <= 0.0:
            return True
        positions, _strengths = (
            self.estimate_clustered()
            if bool(self.config.birth_enable and self.config.use_clustered_output)
            else self.estimate()
        )
        if positions.size == 0:
            return False
        cluster_positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        cluster_count = int(cluster_positions.shape[0])
        support = np.zeros(cluster_count, dtype=float)
        spread_sum = np.zeros(cluster_count, dtype=float)
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        total_weight = max(float(np.sum(weights)), 1.0e-12)
        weights = weights / total_weight
        support_radius = max(
            2.0 * max(float(self.config.cluster_eps_m), 1.0e-6),
            max_spread if max_spread > 0.0 else 0.0,
        )
        for weight, particle in zip(weights, self.continuous_particles):
            st = particle.state
            if st.num_sources <= 0:
                continue
            source_positions = np.asarray(st.positions[: st.num_sources], dtype=float)
            distances = np.linalg.norm(
                source_positions[:, None, :] - cluster_positions[None, :, :],
                axis=2,
            )
            nearest = np.min(distances, axis=0)
            supported = nearest <= support_radius
            support[supported] += float(weight)
            spread_sum[supported] += float(weight) * nearest[supported] ** 2
        if min_support > 0.0 and np.any(support < min_support):
            return False
        if max_spread > 0.0:
            if np.any(support <= 0.0):
                return False
            rms = np.sqrt(spread_sum / np.maximum(support, 1.0e-12))
            if np.any(rms > max_spread):
                return False
        return True

    def _distinct_observed_station_count(self) -> int:
        """Return the number of distinct detector stations seen by this filter."""
        return int(len(self._observed_station_labels))

    def _record_observed_station(
        self,
        detector_pos: NDArray[np.float64] | None,
    ) -> None:
        """Record a detector station using rounded XY coordinates."""
        if detector_pos is None:
            return
        pos = np.asarray(detector_pos, dtype=float).reshape(-1)
        if pos.size < 2 or not np.all(np.isfinite(pos[:2])):
            return
        self._observed_station_labels.add(
            (round(float(pos[0]), 3), round(float(pos[1]), 3))
        )

    def _confirmed_source_structure(self) -> bool:
        """Return True when source structure is stable enough to preserve."""
        if self._cardinality_variance() > float(
            self.config.converge_cardinality_var_max
        ):
            return False
        if (
            bool(self.config.converge_require_no_tentative)
            and self._has_unverified_sources()
        ):
            return False
        if not self._cluster_convergence_supported():
            return False
        return True

    def _convergence_can_freeze(self) -> bool:
        """Return True when no unresolved source structure should keep updating."""
        if not self.config.converge_enable:
            return False
        min_stations = max(0, int(getattr(self.config, "converge_min_stations", 0)))
        if self._distinct_observed_station_count() < min_stations:
            return False
        if not self._confirmed_source_structure():
            return False
        return True

    def _should_skip_converged_update(self) -> bool:
        """Return True when a converged filter can safely ignore more updates."""
        if not (
            self.config.converge_enable
            and self.config.converge_freeze_updates
            and self.is_converged
        ):
            return False
        if self._convergence_can_freeze():
            return True
        self.is_converged = False
        self.frozen_estimate = None
        return False

    def _maybe_update_convergence(
        self,
        step_idx: int | None,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> None:
        """Update convergence monitor and freeze if criteria are met."""
        if not self.config.converge_enable or self._converge_monitor is None:
            return
        if step_idx is None:
            return
        if not self.continuous_particles:
            return
        summary_vec = self._convergence_state_vector()
        ess_pre = self.last_ess_pre
        if ess_pre is None:
            w = self.continuous_weights
            ess_pre = float(1.0 / max(np.sum(w**2), 1e-12)) if w.size else 0.0
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        ll_value = self._ll_proxy_pair(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )
        self._converge_monitor.update_stats(step_idx, summary_vec, ess_ratio, ll_value)
        if (
            self._converge_monitor.is_converged(step_idx)
            and self._convergence_can_freeze()
        ):
            self.is_converged = True
            self.frozen_estimate = (
                self.estimate_clustered()
                if bool(self.config.birth_enable and self.config.use_clustered_output)
                else self.estimate()
            )

    def _continuous_expected_counts_torch(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        if self.kernel is None:
            from pf import gpu_utils

            device = gpu_utils.resolve_device(self.config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
            import torch

            return torch.zeros(0, device=device, dtype=dtype)
        orient_vec = self.kernel.orientations[orient_idx]
        octant_idx = self.continuous_kernel.orient_index_from_vector(orient_vec)
        return self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=octant_idx,
            pb_index=octant_idx,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_torch(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles or self.kernel is None:
            return torch.zeros(0, device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states, device=device, dtype=dtype
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                source_scale=self._measurement_source_scale(
                    fe_index=fe_index,
                    pb_index=pb_index,
                ),
                device=device,
                dtype=dtype,
            )
        zero_backgrounds = torch.zeros(
            int(packed_positions.shape[0]),
            device=device,
            dtype=dtype,
        )
        source_counts = (
            self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=zero_backgrounds,
                mask=packed_mask,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                source_scale=self._measurement_source_scale(
                    fe_index=fe_index,
                    pb_index=pb_index,
                ),
                device=device,
                dtype=dtype,
            )
        )
        return source_counts.index_select(0, inverse) + float(live_time_s) * backgrounds

    def _continuous_expected_counts_pair_sequence_torch(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute expected counts for a same-station shield sequence in one GPU batch."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
        else:
            zero_backgrounds = torch.zeros(
                int(packed_positions.shape[0]),
                device=device,
                dtype=dtype,
            )
            source_unit_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=zero_backgrounds,
                mask=packed_mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
            unit_live_counts = source_unit_counts.index_select(1, inverse)
            unit_live_counts = unit_live_counts + backgrounds.unsqueeze(0)
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _compress_identical_packed_sources_torch(
        self,
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor | None",
    ]:
        """Return one source row plus inverse when all packed source states match."""
        import torch

        particle_count = int(positions.shape[0])
        if particle_count <= 1:
            return positions, strengths, mask, None
        if positions.ndim != 3 or strengths.ndim != 2 or mask.ndim != 2:
            return positions, strengths, mask, None
        flat = torch.cat(
            (
                positions.reshape(particle_count, -1),
                strengths.reshape(particle_count, -1),
                mask.reshape(particle_count, -1),
            ),
            dim=1,
        )
        if flat.numel() == 0:
            return positions, strengths, mask, None
        identical = bool(torch.all(flat == flat[:1]).detach().cpu().item())
        if not identical:
            return positions, strengths, mask, None
        inverse = torch.zeros(particle_count, device=positions.device, dtype=torch.long)
        return positions[:1], strengths[:1], mask[:1], inverse

    def _continuous_expected_counts_pair_sequence_torch_uncompressed(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute same-station expected counts without duplicate-state compression."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_time_s=1.0,
            source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
            device=device,
            dtype=dtype,
        )
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _log_likelihood_increment_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        observation_count_variance: float = 0.0,
    ) -> "torch.Tensor":
        """Return the per-particle count log-likelihood increment in float64."""
        import torch

        lam_t = lam_t.to(dtype=torch.float64)
        lam_t = torch.clamp(lam_t, min=1e-12)
        z = torch.as_tensor(z_obs, device=lam_t.device, dtype=torch.float64)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""}:
            return z * torch.log(lam_t) - lam_t

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = max(float(observation_count_variance), 0.0)
        variance = count_likelihood_variance_torch(
            z,
            lam_t,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )
        residual = z - lam_t
        if model == "gaussian":
            return -0.5 * ((residual**2) / variance + torch.log(variance))

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        return -0.5 * (df + 1.0) * torch.log1p(
            (residual**2) / (df * variance)
        ) - 0.5 * torch.log(variance)

    def _log_likelihood_sequence_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        observation_count_covariance: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return summed per-particle log-likelihoods for a measurement sequence."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        if lam.numel() == 0:
            return torch.zeros(0, device=lam.device, dtype=torch.float64)
        lam = torch.clamp(lam, min=1e-12)
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""}:
            count_ll = torch.sum(z * torch.log(lam) - lam, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )
        if self._sequence_covariance_enabled(
            z_arr.size,
            observation_count_covariance,
        ):
            return self._log_likelihood_sequence_covariance_gpu(
                lam,
                z_arr,
                var_arr,
                observation_count_covariance=observation_count_covariance,
            )

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        variance = count_likelihood_variance_torch(
            z,
            lam,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )
        residual = z - lam
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            count_ll = torch.sum(ll, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        ll = -0.5 * (df + 1.0) * torch.log1p(
            (residual**2) / (df * variance)
        ) - 0.5 * torch.log(variance)
        count_ll = torch.sum(ll, dim=0)
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_covariance_enabled(
        self,
        sequence_length: int,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether same-station view covariance should be evaluated."""
        if int(sequence_length) < 2:
            return False
        if observation_count_covariance is not None:
            return True
        return bool(self.config.station_view_covariance_enable) and (
            float(self.config.station_view_correlated_spectrum_fraction) > 0.0
        )

    def _spectral_bin_sequence_log_likelihood_gpu(
        self,
        expected_spectrum_kbn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods for each particle."""
        import torch

        expected = expected_spectrum_kbn.to(dtype=torch.float64)
        if expected.ndim == 2:
            expected = expected.unsqueeze(0)
        if expected.ndim != 3:
            raise ValueError("expected_spectrum_kbn must have shape K x B x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.shape != tuple(expected.shape[:2]):
            raise ValueError("observed_spectrum_kb must match K x B expected bins.")
        expected = torch.clamp(expected, min=1.0e-12)
        observed = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=expected.device,
            dtype=torch.float64,
        ).unsqueeze(-1)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""} and observation_spectrum_variance_kb is None:
            return torch.sum(observed * torch.log(expected) - expected, dim=(0, 1))

        if observation_spectrum_variance_kb is None:
            obs_var = torch.zeros_like(observed)
        else:
            obs_var_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if obs_var_arr.ndim == 1:
                obs_var_arr = obs_var_arr.reshape(1, -1)
            if obs_var_arr.shape != tuple(expected.shape[:2]):
                raise ValueError("observation_spectrum_variance_kb must match K x B.")
            obs_var = torch.as_tensor(
                np.maximum(np.where(np.isfinite(obs_var_arr), obs_var_arr, 0.0), 0.0),
                device=expected.device,
                dtype=torch.float64,
            ).unsqueeze(-1)
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        variance = torch.clamp(
            expected
            + obs_var
            + (float(spectrum_rel) * expected) ** 2
            + float(spectrum_abs) ** 2,
            min=1.0e-12,
        )
        residual = observed - expected
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            ll = -0.5 * (df + 1.0) * torch.log1p(
                (residual**2) / (df * variance)
            ) - 0.5 * torch.log(variance)
        return torch.sum(ll, dim=(0, 1))

    @classmethod
    def _direct_spectrum_likelihood_config_enabled(
        cls,
        config: Any,
        isotope: str,
    ) -> bool:
        """Return whether config permits independent-bin spectrum likelihoods.

        Complete statistical covariance from weighted transport generally has
        response-folding correlations between spectrum bins. The current
        direct-bin likelihood accepts only diagonal variance, so that semantic
        must route through isotope counts and their propagated covariance.
        Gaussian and Student-t transport discrepancy is also defined on the
        isotope count, not independently on every spectrum bin, and therefore
        must use the count likelihood.
        """
        if not bool(config.direct_spectrum_likelihood_enable):
            return False
        semantics = normalize_observation_count_variance_semantics(
            config.observation_count_variance_semantics,
            includes_counting_noise=(
                config.observation_count_variance_includes_counting_noise
            ),
        )
        if semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
            return False
        model = normalize_count_likelihood_model(
            str(config.count_likelihood_model)
        )
        if model != "poisson" and (
            cls._isotope_float_config_for(
                config.transport_model_rel_sigma,
                isotope,
            )
            > 0.0
            or cls._isotope_float_config_for(
                config.transport_model_abs_sigma,
                isotope,
            )
            > 0.0
        ):
            return False
        return True

    def _direct_spectrum_likelihood_enabled(self) -> bool:
        """Return whether this filter permits independent spectrum bins."""
        return self._direct_spectrum_likelihood_config_enabled(
            self.config,
            self.isotope,
        )

    def _direct_spectrum_route_admissible(
        self,
        *,
        sequence_length: int,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether one runtime update may use independent spectrum bins."""
        return self._direct_spectrum_likelihood_enabled() and not (
            self._sequence_covariance_enabled(
                int(sequence_length),
                observation_count_covariance,
            )
        )

    @staticmethod
    def _spectrum_update_arrays(
        observed_spectrum: NDArray[np.float64] | None,
        response_template: NDArray[np.float64] | None,
        background_spectrum: NDArray[np.float64] | None,
        spectrum_variance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
        ]
        | None
    ):
        """Return validated spectrum-bin arrays for a PF likelihood update."""
        if observed_spectrum is None or response_template is None:
            return None
        rows = max(1, int(sequence_length))
        observed = np.asarray(observed_spectrum, dtype=float)
        if observed.ndim == 1:
            observed = observed.reshape(1, -1)
        template = np.asarray(response_template, dtype=float)
        if template.ndim == 1:
            template = np.broadcast_to(template.reshape(1, -1), observed.shape)
        background = (
            np.zeros_like(observed, dtype=float)
            if background_spectrum is None
            else np.asarray(background_spectrum, dtype=float)
        )
        if background.ndim == 1:
            background = np.broadcast_to(background.reshape(1, -1), observed.shape)
        variance = None
        if spectrum_variance is not None:
            variance = np.asarray(spectrum_variance, dtype=float)
            if variance.ndim == 1:
                variance = np.broadcast_to(variance.reshape(1, -1), observed.shape)
        if observed.shape[0] != rows:
            raise ValueError("observed_spectrum row count must match sequence length.")
        if template.shape != observed.shape:
            raise ValueError("response_template must match observed spectrum bins.")
        if background.shape != observed.shape:
            raise ValueError("background_spectrum must match observed spectrum bins.")
        if variance is not None and variance.shape != observed.shape:
            raise ValueError("spectrum_variance must match observed spectrum bins.")
        observed = np.maximum(np.where(np.isfinite(observed), observed, 0.0), 0.0)
        template = np.maximum(np.where(np.isfinite(template), template, 0.0), 0.0)
        background = np.maximum(
            np.where(np.isfinite(background), background, 0.0),
            0.0,
        )
        if variance is not None:
            variance = np.maximum(np.where(np.isfinite(variance), variance, 0.0), 0.0)
        return observed, template, background, variance

    def _spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        self,
        lam_kn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods without a full KxBxN tensor."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        if lam.ndim != 2:
            raise ValueError("lam_kn must have shape K x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.ndim != 2 or observed_arr.shape[0] != int(lam.shape[0]):
            raise ValueError("observed_spectrum_kb rows must match lambda rows.")
        template_arr = np.asarray(response_template_kb, dtype=float)
        if template_arr.ndim == 1:
            template_arr = np.broadcast_to(
                template_arr.reshape(1, -1),
                observed_arr.shape,
            )
        background_arr = np.asarray(background_spectrum_kb, dtype=float)
        if background_arr.ndim == 1:
            background_arr = np.broadcast_to(
                background_arr.reshape(1, -1),
                observed_arr.shape,
            )
        if template_arr.shape != observed_arr.shape:
            raise ValueError("response_template_kb must match observed spectrum bins.")
        if background_arr.shape != observed_arr.shape:
            raise ValueError(
                "background_spectrum_kb must match observed spectrum bins."
            )
        variance_arr = None
        if observation_spectrum_variance_kb is not None:
            variance_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if variance_arr.ndim == 1:
                variance_arr = np.broadcast_to(
                    variance_arr.reshape(1, -1),
                    observed_arr.shape,
                )
            if variance_arr.shape != observed_arr.shape:
                raise ValueError("observation_spectrum_variance_kb must match bins.")
        observed_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        template_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(template_arr), template_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        background_all = torch.as_tensor(
            np.maximum(
                np.where(np.isfinite(background_arr), background_arr, 0.0),
                0.0,
            ),
            device=lam.device,
            dtype=torch.float64,
        )
        variance_all = None
        if variance_arr is not None:
            variance_all = torch.as_tensor(
                np.maximum(np.where(np.isfinite(variance_arr), variance_arr, 0.0), 0.0),
                device=lam.device,
                dtype=torch.float64,
            )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        rel_sigma = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        abs_sigma = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        chunk = max(1, int(getattr(self.config, "spectrum_likelihood_bin_chunk", 512)))
        bin_count = int(observed_all.shape[1])
        total_ll = torch.zeros(
            int(lam.shape[1]), device=lam.device, dtype=torch.float64
        )
        for start in range(0, bin_count, chunk):
            stop = min(start + chunk, bin_count)
            expected = torch.clamp(
                lam[:, None, :] * template_all[:, start:stop, None]
                + background_all[:, start:stop, None],
                min=1.0e-12,
            )
            observed = observed_all[:, start:stop, None]
            if model in {"poisson", ""} and variance_all is None:
                total_ll = total_ll + torch.sum(
                    observed * torch.log(expected) - expected,
                    dim=(0, 1),
                )
                continue
            if variance_all is None:
                obs_var = torch.zeros_like(observed)
            else:
                obs_var = variance_all[:, start:stop, None]
            variance = torch.clamp(
                expected
                + obs_var
                + (float(rel_sigma) * expected) ** 2
                + float(abs_sigma) ** 2,
                min=1.0e-12,
            )
            residual = observed - expected
            if model == "gaussian":
                ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            else:
                df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
                ll = -0.5 * (df + 1.0) * torch.log1p(
                    (residual**2) / (df * variance)
                ) - 0.5 * torch.log(variance)
            total_ll = total_ll + torch.sum(ll, dim=(0, 1))
        return total_ll

    def _expected_spectrum_sequence_torch(
        self,
        lam_kn: "torch.Tensor",
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxBxN expected spectra from particle count rates."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        template = torch.as_tensor(
            response_template_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        background = torch.as_tensor(
            background_spectrum_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        if template.ndim == 1:
            template = template.view(1, -1)
        if background.ndim == 1:
            background = background.view(1, -1)
        if template.shape[0] != lam.shape[0] or background.shape[0] != lam.shape[0]:
            raise ValueError("spectrum template rows must match lambda rows.")
        expected = lam[:, None, :] * template[:, :, None] + background[:, :, None]
        return torch.clamp(expected, min=1.0e-12)

    def _log_likelihood_sequence_covariance_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        *,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> "torch.Tensor":
        """Return a batched multivariate count likelihood for a shield sequence."""
        import torch

        lam = torch.clamp(lam_kn.to(dtype=torch.float64), min=1.0e-12)
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        variance = self._sequence_diagonal_variance_torch(lam, z, var_arr)
        covariance_nkk = torch.diag_embed(variance.T)
        covariance_nkk = covariance_nkk + self._sequence_observation_offdiag_torch(
            observation_count_covariance,
            sequence_length=z_arr.size,
            device=lam.device,
        )
        covariance_nkk = covariance_nkk + self._sequence_common_mode_offdiag_torch(
            lam,
        )
        covariance_nkk = self._regularize_sequence_covariance_torch(
            covariance_nkk,
            variance.T,
        )
        residual = (z - lam).T.unsqueeze(-1)
        chol = torch.linalg.cholesky(covariance_nkk)
        solved = torch.cholesky_solve(residual, chol)
        quad = torch.sum(residual * solved, dim=(1, 2))
        logdet = 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=1, dim2=2)),
            dim=1,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "gaussian":
            count_ll = -0.5 * (quad + logdet)
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            dim = float(z_arr.size)
            count_ll = -0.5 * (df + dim) * torch.log1p(quad / df) - 0.5 * logdet
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_diagonal_variance_torch(
        self,
        lam_kn: "torch.Tensor",
        z_kn: "torch.Tensor",
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxN diagonal variances for same-station sequence likelihood."""
        import torch

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                observation_count_variances,
                device=lam_kn.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        return count_likelihood_variance_torch(
            z_kn,
            lam_kn,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )

    def _sequence_observation_offdiag_torch(
        self,
        observation_count_covariance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
        device: "torch.device",
    ) -> "torch.Tensor":
        """Return fixed off-diagonal observation covariance for one sequence."""
        import torch

        size = int(sequence_length)
        if observation_count_covariance is None:
            return torch.zeros((1, size, size), device=device, dtype=torch.float64)
        covariance = np.asarray(observation_count_covariance, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(
                "observation_count_covariance must be shaped sequence x sequence."
            )
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, 0.0)
        return torch.as_tensor(
            covariance,
            device=device,
            dtype=torch.float64,
        ).view(1, size, size)

    def _sequence_common_mode_offdiag_torch(
        self,
        lam_kn: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return particle-wise off-diagonal covariance from common spectrum error."""
        import torch

        fraction = max(
            0.0,
            float(self.config.station_view_correlated_spectrum_fraction),
        )
        if not bool(self.config.station_view_covariance_enable) or fraction <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        spectrum_rel = fraction * self._isotope_float_config(
            self.config.spectrum_count_rel_sigma
        )
        spectrum_abs = fraction * self._isotope_float_config(
            self.config.spectrum_count_abs_sigma
        )
        if spectrum_rel <= 0.0 and spectrum_abs <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        lam_nk = lam_kn.T
        common = (float(spectrum_rel) ** 2) * (lam_nk[:, :, None] * lam_nk[:, None, :])
        if spectrum_abs > 0.0:
            common = common + float(spectrum_abs) ** 2
        diag = torch.arange(int(lam_kn.shape[0]), device=lam_kn.device)
        common[:, diag, diag] = 0.0
        return common

    def _regularize_sequence_covariance_torch(
        self,
        covariance_nkk: "torch.Tensor",
        diagonal_nk: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a positive-definite covariance batch for Cholesky solves."""
        import torch

        size = int(covariance_nkk.shape[1])
        eye = torch.eye(size, device=covariance_nkk.device, dtype=torch.float64)
        diag_scale = torch.clamp(torch.mean(diagonal_nk, dim=1), min=1.0)
        jitter = 1.0e-9 * diag_scale
        regularized = covariance_nkk
        for _ in range(6):
            candidate = regularized + jitter.view(-1, 1, 1) * eye.view(1, size, size)
            _chol, info = torch.linalg.cholesky_ex(candidate)
            if bool(torch.all(info == 0)):
                return candidate
            jitter = jitter * 10.0
            regularized = covariance_nkk
        fallback = torch.diag_embed(torch.clamp(diagonal_nk, min=1.0e-12))
        return fallback + jitter.view(-1, 1, 1) * eye.view(1, size, size)

    def _shield_shape_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return same-station shield signature likelihood terms."""
        return self._shield_contrast_sequence_log_likelihood_gpu(
            lam_kn,
            z_obs,
            observation_count_variances,
        ) + self._shield_view_ratio_sequence_log_likelihood_gpu(lam_kn, z_obs)

    def _shield_contrast_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a robust same-station shield-signature likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(getattr(self.config, "shield_contrast_likelihood_enable", False)):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_contrast_likelihood_weight", 1.0)),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_contrast_min_views", 2)),
            2,
        )
        if weight <= 0.0 or int(lam.shape[0]) < min_views or particle_count == 0:
            return zeros

        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        min_count = max(
            float(getattr(self.config, "shield_contrast_min_count", 25.0)),
            1.0e-6,
        )
        sigma_floor = max(
            float(getattr(self.config, "shield_contrast_log_sigma_floor", 0.5)),
            1.0e-6,
        )
        sigma_ceiling = max(
            float(getattr(self.config, "shield_contrast_log_sigma_ceiling", 2.0)),
            sigma_floor,
        )
        df = max(
            float(getattr(self.config, "shield_contrast_likelihood_df", 5.0)),
            1.0 + 1.0e-12,
        )

        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        z_safe = torch.clamp(z, min=min_count)
        lam_safe = torch.clamp(lam, min=min_count)
        log_z = torch.log(z_safe)
        log_lam = torch.log(lam_safe)
        log_var = obs_var / torch.clamp(z_safe**2, min=1.0e-12)
        log_var = torch.clamp(
            log_var + sigma_floor**2,
            min=sigma_floor**2,
            max=sigma_ceiling**2,
        )
        view_weight = torch.reciprocal(log_var)
        weight_sum = torch.clamp(
            torch.sum(view_weight, dim=0, keepdim=True),
            min=1e-12,
        )
        obs_center = log_z - (
            torch.sum(view_weight * log_z, dim=0, keepdim=True) / weight_sum
        )
        pred_center = (
            log_lam
            - torch.sum(
                view_weight * log_lam,
                dim=0,
                keepdim=True,
            )
            / weight_sum
        )
        residual = obs_center - pred_center
        terms = -0.5 * (df + 1.0) * torch.log1p((residual**2) / (df * log_var))
        terms -= 0.5 * torch.log(log_var)
        return float(weight) * torch.sum(terms, dim=0)

    def _shield_view_ratio_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a Dirichlet-multinomial shield-view ratio likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(getattr(self.config, "shield_view_ratio_likelihood_enable", False)):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_view_ratio_likelihood_weight", 1.0)),
            0.0,
        )
        concentration = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_concentration",
                    128.0,
                )
            ),
            1.0e-6,
        )
        min_total = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_min_total_count",
                    25.0,
                )
            ),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_view_ratio_likelihood_min_views", 2)),
            2,
        )
        if weight <= 0.0 or int(lam.shape[0]) < min_views or particle_count == 0:
            return zeros
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z_arr = np.maximum(np.where(np.isfinite(z_arr), z_arr, 0.0), 0.0)
        total = float(np.sum(z_arr))
        if total < min_total or not np.isfinite(total):
            return zeros
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        lam_safe = torch.clamp(lam, min=1.0e-12)
        lam_total = torch.clamp(torch.sum(lam_safe, dim=0, keepdim=True), min=1.0e-12)
        probabilities = lam_safe / lam_total
        alpha = torch.clamp(concentration * probabilities, min=1.0e-12)
        alpha0 = torch.clamp(torch.sum(alpha, dim=0), min=1.0e-12)
        total_t = torch.as_tensor(total, device=lam.device, dtype=torch.float64)
        ll = torch.lgamma(alpha0) - torch.lgamma(alpha0 + total_t)
        ll = ll + torch.sum(torch.lgamma(alpha + z) - torch.lgamma(alpha), dim=0)
        return float(weight) * ll

    def _normalized_log_weights_torch(self, logw: "torch.Tensor") -> "torch.Tensor":
        """Normalize log-weights using logsumexp in float64."""
        import torch

        return logw - torch.logsumexp(logw, dim=0)

    def _ess_from_logw_torch(self, logw: "torch.Tensor") -> float:
        """Return the effective sample size from normalized log-weights."""
        import torch

        w = torch.exp(logw)
        ess = 1.0 / torch.sum(w**2)
        return float(ess.detach().cpu().item())

    def _assign_logw_from_torch(self, logw: "torch.Tensor") -> None:
        """Copy log-weights from torch back into particle objects."""
        logw_cpu = logw.detach().cpu().numpy()
        for p, lw in zip(self.continuous_particles, logw_cpu):
            p.log_weight = float(lw)

    def _update_continuous_weights_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        *,
        observation_count_variance: float = 0.0,
        delta_beta: float = 1.0,
        logw_prev: "torch.Tensor | None" = None,
        ll_t: "torch.Tensor | None" = None,
        return_logw: bool = False,
    ) -> "torch.Tensor | None":
        """
        Update continuous log-weights using tempered Poisson increments.

        When return_logw is True, returns the normalized log-weights after the update.
        """
        if lam_t.numel() == 0:
            return
        logw_prev = (
            logw_prev
            if logw_prev is not None
            else self._current_log_weights_torch(lam_t.device)
        )
        ll_t = (
            ll_t
            if ll_t is not None
            else self._log_likelihood_increment_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
            )
        )
        logw = self._normalized_log_weights_torch(logw_prev + float(delta_beta) * ll_t)
        self._assign_logw_from_torch(logw)
        if return_logw:
            return logw
        return None

    def _select_delta_beta(
        self,
        logw_prev: "torch.Tensor",
        ll_t: "torch.Tensor",
        remaining: float,
        target_ess: float,
    ) -> tuple[float, "torch.Tensor", float]:
        """
        Return the largest delta_beta that keeps ESS above the target.

        Returns (delta_beta, logw_new, ess).
        """
        remaining = float(remaining)
        min_delta = max(float(self.config.min_delta_beta), 0.0)
        if remaining <= min_delta:
            logw_new = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
            ess = self._ess_from_logw_torch(logw_new)
            return remaining, logw_new, ess

        logw_full = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
        ess_full = self._ess_from_logw_torch(logw_full)
        if ess_full >= target_ess:
            return remaining, logw_full, ess_full

        logw_low = self._normalized_log_weights_torch(logw_prev + min_delta * ll_t)
        ess_low = self._ess_from_logw_torch(logw_low)
        if ess_low < target_ess:
            return min_delta, logw_low, ess_low

        low = min_delta
        high = remaining
        logw_best = logw_low
        ess_best = ess_low
        for _ in range(24):
            mid = 0.5 * (low + high)
            logw_mid = self._normalized_log_weights_torch(logw_prev + mid * ll_t)
            ess_mid = self._ess_from_logw_torch(logw_mid)
            if ess_mid >= target_ess:
                low = mid
                logw_best = logw_mid
                ess_best = ess_mid
            else:
                high = mid
        return low, logw_best, ess_best

    def _tempered_update(
        self,
        lam_fn: Callable[[], "torch.Tensor"],
        z_obs: float,
        observation_count_variance: float = 0.0,
        disable_regularize_on_resample: bool | None = None,
        roughening_scale_on_resample: float = 1.0,
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering for a single Poisson update.

        The update increments beta from 0 to 1 using delta_beta steps that
        maintain ESS above the configured target ratio when possible.

        Returns (ess_pre, resampled_any) for downstream adaptation logic.
        """

        def _ll_fn() -> "torch.Tensor":
            """Return per-particle log-likelihood increments for one count."""
            import torch

            lam_t_inner = lam_fn()
            if lam_t_inner.numel() == 0:
                return lam_t_inner.to(dtype=torch.float64)
            return self._log_likelihood_increment_gpu(
                lam_t_inner,
                z_obs,
                observation_count_variance=observation_count_variance,
            )

        return self._tempered_update_likelihood(
            ll_fn=_ll_fn,
            disable_regularize_on_resample=disable_regularize_on_resample,
            roughening_scale_on_resample=roughening_scale_on_resample,
        )

    def _tempered_update_likelihood(
        self,
        ll_fn: Callable[[], "torch.Tensor"],
        *,
        disable_regularize_on_resample: bool | None = None,
        roughening_scale_on_resample: float = 1.0,
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering to a precomputed likelihood increment.

        ``ll_fn`` is re-evaluated after a tempering resample, which keeps joint
        multi-orientation updates consistent with the newly roughened particles.
        """
        beta_total = 0.0
        steps: list[dict[str, float]] = []
        resamples = 0
        resampled_any = False
        ess_min: float | None = None
        target_ess = float(self.config.target_ess_ratio) * max(self.N, 1)
        resample_threshold = float(self.config.resample_threshold) * max(self.N, 1)
        max_resamples = max(0, int(self.config.max_resamples_per_observation))
        cooldown_steps = max(0, int(self.config.temper_resample_cooldown_steps))
        force_resample_ess = float(self.config.temper_resample_force_ratio) * max(
            self.N, 1
        )
        disable_regularize = bool(self.config.disable_regularize_on_temper_resample)
        if disable_regularize_on_resample is not None:
            disable_regularize = disable_regularize or bool(
                disable_regularize_on_resample
            )
        ll_t = ll_fn()
        if ll_t.numel() == 0:
            self.last_temper_steps = []
            self.last_temper_resample_count = 0
            return 0.0, False
        logw = self._current_log_weights_torch(ll_t.device)

        cooldown_remaining = 0
        while beta_total < 1.0 - 1e-12:
            remaining = 1.0 - beta_total
            delta_beta, logw_new, ess = self._select_delta_beta(
                logw_prev=logw,
                ll_t=ll_t,
                remaining=remaining,
                target_ess=target_ess,
            )
            logw = logw_new
            self._assign_logw_from_torch(logw)
            beta_total += delta_beta
            ess_min = ess if ess_min is None else min(ess_min, ess)
            steps.append(
                {
                    "beta_total": float(beta_total),
                    "delta_beta": float(delta_beta),
                    "ess": float(ess),
                }
            )
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            do_resample = (
                ess < resample_threshold
                and resamples < max_resamples
                and (cooldown_remaining == 0 or ess < force_resample_ess)
            )
            can_jump_without_intermediate_state_change = (
                not do_resample
                and cooldown_remaining == 0
                and delta_beta < remaining - 1.0e-12
            )
            if can_jump_without_intermediate_state_change:
                remaining_after_delta = 1.0 - beta_total
                logw_full = self._normalized_log_weights_torch(
                    logw + remaining_after_delta * ll_t
                )
                ess_full = self._ess_from_logw_torch(logw_full)
                logw = logw_full
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess_min = ess_full if ess_min is None else min(ess_min, ess_full)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining_after_delta),
                        "ess": float(ess_full),
                    }
                )
                if ess_full < resample_threshold and resamples < max_resamples:
                    self._maybe_resample_continuous(
                        disable_regularize=disable_regularize,
                        roughening_scale=roughening_scale_on_resample,
                    )
                    if self.last_resample_ess:
                        resampled_any = True
                        resamples += 1
                break
            if do_resample:
                self._maybe_resample_continuous(
                    disable_regularize=disable_regularize,
                    roughening_scale=roughening_scale_on_resample,
                )
                if self.last_resample_ess:
                    resampled_any = True
                    resamples += 1
                    cooldown_remaining = max(cooldown_remaining, cooldown_steps)
                    ll_t = ll_fn()
                    if ll_t.numel() == 0:
                        break
                    logw = self._current_log_weights_torch(ll_t.device)
            if resamples >= max_resamples and beta_total < 1.0 - 1e-12:
                remaining = 1.0 - beta_total
                logw = self._normalized_log_weights_torch(logw + remaining * ll_t)
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess = self._ess_from_logw_torch(logw)
                ess_min = ess if ess_min is None else min(ess_min, ess)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining),
                        "ess": float(ess),
                    }
                )
                break
        self.last_temper_steps = steps
        self.last_temper_resample_count = resamples
        if ess_min is None:
            ess_min = 0.0
        return float(ess_min), resampled_any

    def _continuous_expected_counts_gpu(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        lam_t = self._continuous_expected_counts_torch(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_pair_gpu(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        lam_t = self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_cpu(
        self,
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute single-orientation expected counts on CPU."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        source_scale = self._measurement_source_scale(
            fe_index=orient_idx,
            pb_index=orient_idx,
        )
        for particle_idx, particle in enumerate(self.continuous_particles):
            state = particle.state
            rate = float(state.background)
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    orient_idx=int(orient_idx),
                )
                rate += source_scale * float(kernel_val) * float(strength)
            lam[particle_idx] = float(live_time_s) * rate
        return lam

    def _continuous_expected_counts_pair_cpu(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute one Fe/Pb-pair row through the batched CPU kernel."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        counts = self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=detector_pos,
            fe_indices=np.asarray([int(fe_index)], dtype=np.int64),
            pb_indices=np.asarray([int(pb_index)], dtype=np.int64),
            live_times_s=np.asarray([float(live_time_s)], dtype=np.float64),
        )
        return counts[0]

    def _continuous_expected_counts_pair_sequence_cpu(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute all views x particles x source slots in one CPU batch."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        return self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=detector_pos,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            live_times_s=live_times_s,
        )

    def _continuous_expected_counts_pair_sequence_at_pose_cpu(
        self,
        detector_pos: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return selected-pair counts from the production kernel on CPU.

        The production packed-state kernel vectorizes shield views, particles,
        and source slots.  Running that same kernel on a CPU device avoids a
        scalar view/particle loop while retaining line, aperture, obstacle, and
        transport-response semantics.
        """
        from pf import gpu_utils
        import torch

        fe_arr = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        particle_count = len(self.continuous_particles)
        if fe_arr.size == 0 or particle_count == 0:
            return np.zeros((fe_arr.size, particle_count), dtype=np.float64)
        device = torch.device("cpu")
        dtype = torch.float64
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            [particle.state for particle in self.continuous_particles],
            device=device,
            dtype=dtype,
        )
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=np.asarray(detector_pos, dtype=float),
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
        else:
            source_unit_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=np.asarray(detector_pos, dtype=float),
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=torch.zeros(
                    int(packed_positions.shape[0]),
                    device=device,
                    dtype=dtype,
                ),
                mask=packed_mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
            unit_live_counts = source_unit_counts.index_select(1, inverse)
            unit_live_counts = unit_live_counts + backgrounds.unsqueeze(0)
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return (
            unit_live_counts.mul(live_t)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    def _continuous_expected_counts_pair_at_pose_cpu(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute one explicit-pose pair through the batched CPU kernel."""
        counts = self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=np.asarray(detector_pos, dtype=float),
            fe_indices=np.asarray([int(fe_index)], dtype=np.int64),
            pb_indices=np.asarray([int(pb_index)], dtype=np.int64),
            live_times_s=np.asarray([float(live_time_s)], dtype=np.float64),
        )
        return counts[0]

    def _continuous_expected_counts(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for each continuous particle using ContinuousKernel."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_cpu(
                pose_idx=pose_idx,
                orient_idx=orient_idx,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_gpu(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )

    def _continuous_expected_counts_pair(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using Fe/Pb octant indices (Eq. 3.41)."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_cpu(
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_pair_gpu(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_at_pose_torch(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for explicit detector position."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles:
            return torch.zeros(0, device=device, dtype=dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            [p.state for p in self.continuous_particles],
            device=device,
            dtype=dtype,
        )
        det_pos = np.asarray(detector_pos, dtype=float)
        return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=det_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self._measurement_source_scale(
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )

    def _continuous_expected_counts_pair_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for explicit detector position."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_at_pose_cpu(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        lam_t = self._continuous_expected_counts_pair_at_pose_torch(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        return lam_t.detach().cpu().numpy()

    def update_continuous_pair(
        self,
        z_obs: float,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
        defer_resample: bool = False,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Count-likelihood weight update using Fe/Pb orientation indices.

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        When ``defer_resample`` is True, structural updates are deferred to the
        caller's end-of-station finalization, but ESS/tempered resampling is
        still allowed for this posture to avoid burst-level weight collapse.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        detector_pos = (
            np.asarray(self.kernel.poses[pose_idx], dtype=float)
            if self.kernel
            else None
        )
        self._record_observed_station(detector_pos)

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            if self.config.use_gpu:
                self._gpu_enabled()
                return self._continuous_expected_counts_pair_torch(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            import torch

            return torch.as_tensor(
                self._continuous_expected_counts_pair_cpu(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                ),
                dtype=torch.float64,
                device="cpu",
            )

        candidate_spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=1,
        )
        spectrum_arrays = (
            candidate_spectrum_arrays
            if self._direct_spectrum_route_admissible(
                sequence_length=1,
                observation_count_covariance=None,
            )
            else None
        )
        if spectrum_arrays is not None:
            self.last_spectrum_likelihood_route = "direct_spectrum"
        else:
            self.last_spectrum_likelihood_route = "count"

        def _spectral_ll_fn() -> "torch.Tensor":
            """Return direct spectrum-bin likelihood increments for one view."""
            if spectrum_arrays is None:
                raise RuntimeError("spectrum_arrays must be available.")
            observed, template, background, variance = spectrum_arrays
            lam_t_inner = _lam_fn()
            return self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                lam_t_inner,
                observed,
                template,
                background,
                variance,
            )

        roughening_scale = 1.0
        disable_regularize = False
        if defer_resample:
            roughening_scale = max(
                0.0,
                float(self.config.deferred_resample_roughening_scale),
            )
            disable_regularize = roughening_scale <= 0.0
        if self.config.use_tempering and spectrum_arrays is not None:
            ess_pre, resampled_any = self._tempered_update_likelihood(
                ll_fn=_spectral_ll_fn,
                disable_regularize_on_resample=disable_regularize,
                roughening_scale_on_resample=roughening_scale,
            )
        elif self.config.use_tempering:
            debug_timing = _pf_debug_timing_enabled()
            debug_start = time.perf_counter()
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_start defer={defer_resample} "
                    f"fe={fe_index} pb={pb_index} z={float(z_obs):.6g}",
                    flush=True,
                )
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
                disable_regularize_on_resample=disable_regularize,
                roughening_scale_on_resample=roughening_scale,
            )
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"resampled={resampled_any} ess={float(ess_pre):.3f}",
                    flush=True,
                )
        elif spectrum_arrays is not None:
            ll_t = _spectral_ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous(
                disable_regularize=disable_regularize,
                roughening_scale=roughening_scale,
            )
            resampled_any = bool(self.last_resample_ess)
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous(
                disable_regularize=disable_regularize,
                roughening_scale=roughening_scale,
            )
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        if defer_resample:
            self._deferred_resampled_any = self._deferred_resampled_any or bool(
                resampled_any
            )
            if np.isfinite(float(ess_pre)):
                if self._deferred_ess_min is None:
                    self._deferred_ess_min = float(ess_pre)
                else:
                    self._deferred_ess_min = min(
                        float(self._deferred_ess_min),
                        float(ess_pre),
                    )
            if resampled_any:
                self.align_continuous_labels()
        else:
            self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
            self.align_continuous_labels()
            self._advance_adapt_cooldown()
        if detector_pos is not None:
            if defer_resample:
                self._deferred_convergence_args = (
                    step_idx,
                    np.asarray(detector_pos, dtype=float).copy(),
                    int(fe_index),
                    int(pb_index),
                    float(live_time_s),
                    float(z_obs),
                )
                return
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )

    def finalize_deferred_update(self) -> None:
        """Finalize a station whose structural updates were delayed."""
        if self._should_skip_converged_update():
            return
        if self._deferred_ess_min is not None:
            ess_pre = float(self._deferred_ess_min)
        elif self.last_ess_pre is not None:
            ess_pre = float(self.last_ess_pre)
        else:
            weights = np.asarray(self.continuous_weights, dtype=float)
            ess_pre = (
                float(1.0 / max(np.sum(weights**2), 1.0e-12)) if weights.size else 0.0
            )
        resampled_any = bool(self._deferred_resampled_any)
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        convergence_args = self._deferred_convergence_args
        self._deferred_convergence_args = None
        if convergence_args is not None:
            step_idx, detector_pos, fe_index, pb_index, live_time_s, z_obs = (
                convergence_args
            )
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )
        self._deferred_resampled_any = False
        self._deferred_ess_min = None

    def update_continuous_pair_sequence(
        self,
        z_obs: NDArray[np.float64],
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64] | None = None,
        observation_count_covariance: NDArray[np.float64] | None = None,
        step_idx: int | None = None,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Jointly update weights using a same-pose shield-orientation sequence.

        The measurement model evaluates the shield program as one station-level
        observation. When covariance is supplied or configured, same-station
        shield-view correlations are handled by a batched multivariate
        likelihood; otherwise the update reduces to the product likelihood over
        views. Updating views jointly avoids resampling, roughening, or
        birth/death moves between postures from the same physical station.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        detector_pos = (
            np.asarray(self.kernel.poses[pose_idx], dtype=float)
            if self.kernel
            else None
        )
        self._record_observed_station(detector_pos)
        z_arr = np.asarray(z_obs, dtype=float).ravel()
        fe_arr = np.asarray(fe_indices, dtype=int).ravel()
        pb_arr = np.asarray(pb_indices, dtype=int).ravel()
        live_arr = np.asarray(live_times_s, dtype=float).ravel()
        if observation_count_variances is None:
            var_arr = np.zeros_like(z_arr, dtype=float)
        else:
            var_arr = np.asarray(observation_count_variances, dtype=float).ravel()
        covariance = None
        if observation_count_covariance is not None:
            covariance = np.asarray(observation_count_covariance, dtype=float)
            if covariance.shape != (z_arr.size, z_arr.size):
                raise ValueError("observation_count_covariance must be shaped K x K.")
        if not (
            z_arr.size == fe_arr.size == pb_arr.size == live_arr.size == var_arr.size
        ):
            raise ValueError("Joint PF update arrays must have matching lengths.")
        if z_arr.size == 0:
            return
        candidate_spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=z_arr.size,
        )
        spectrum_arrays = (
            candidate_spectrum_arrays
            if self._direct_spectrum_route_admissible(
                sequence_length=int(z_arr.size),
                observation_count_covariance=covariance,
            )
            else None
        )
        if spectrum_arrays is not None:
            self.last_spectrum_likelihood_route = "direct_spectrum"
        elif self._sequence_covariance_enabled(
            z_arr.size,
            covariance,
        ):
            self.last_spectrum_likelihood_route = "count_covariance"
        else:
            self.last_spectrum_likelihood_route = "count"

        def _ll_fn() -> "torch.Tensor":
            """Return summed per-particle log-likelihood for the shield sequence."""
            if self.config.use_gpu:
                self._gpu_enabled()
                lam_kn = self._continuous_expected_counts_pair_sequence_torch(
                    pose_idx=pose_idx,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                    live_times_s=live_arr,
                )
            else:
                import torch

                lam_kn = torch.as_tensor(
                    self._continuous_expected_counts_pair_sequence_cpu(
                        pose_idx=pose_idx,
                        fe_indices=fe_arr,
                        pb_indices=pb_arr,
                        live_times_s=live_arr,
                    ),
                    dtype=torch.float64,
                    device="cpu",
                )
            if spectrum_arrays is not None:
                observed, template, background_spectrum, variance = spectrum_arrays
                spectrum_ll = (
                    self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                        lam_kn,
                        observed,
                        template,
                        background_spectrum,
                        variance,
                    )
                )
                return spectrum_ll + self._shield_shape_sequence_log_likelihood_gpu(
                    lam_kn,
                    z_arr,
                    var_arr,
                )
            return self._log_likelihood_sequence_gpu(
                lam_kn,
                z_arr,
                var_arr,
                observation_count_covariance=covariance,
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update_likelihood(
                ll_fn=_ll_fn,
            )
        else:
            ll_t = _ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
                resampled_any = False
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
                self._maybe_resample_continuous()
                resampled_any = bool(self.last_resample_ess)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        if detector_pos is not None:
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=int(fe_arr[-1]),
                pb_index=int(pb_arr[-1]),
                live_time_s=float(np.sum(live_arr)),
                z_obs=float(np.sum(z_arr)),
            )

    def update_continuous_pair_at_pose(
        self,
        z_obs: float,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
    ) -> None:
        """
        Count-likelihood weight update using explicit detector position.

        This avoids reliance on pose indices for planning-time evaluations.
        """
        if self._should_skip_converged_update():
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self.last_spectrum_likelihood_route = "count"

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            if self.config.use_gpu:
                self._gpu_enabled()
                return self._continuous_expected_counts_pair_at_pose_torch(
                    detector_pos=detector_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            import torch

            return torch.as_tensor(
                self._continuous_expected_counts_pair_at_pose_cpu(
                    detector_pos=detector_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                ),
                dtype=torch.float64,
                device="cpu",
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
            )
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous()
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()
        self._maybe_update_convergence(
            step_idx=step_idx,
            detector_pos=np.asarray(detector_pos, dtype=float),
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles], dtype=np.float64
        )
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles), dtype=float) / len(
                self.continuous_particles
            )
        return w / s

    def _normalize_continuous_log_weights(self) -> None:
        """Normalize continuous-particle log weights in place."""
        if not self.continuous_particles:
            return
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles],
            dtype=np.float64,
        )
        norm = logsumexp(logw)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, logw - norm):
            particle.log_weight = float(value)

    def _source_mode_preserving_indices(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Return particle indices that preserve distinct posterior source modes."""
        self.last_mode_preserving_dynamic_spatial_summary = []
        if not bool(self.config.mode_preserving_resample):
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)
        max_modes = max(0, int(self.config.mode_preserving_max_modes))
        per_mode = max(0, int(self.config.mode_preserving_particles_per_mode))
        radius = max(float(self.config.mode_preserving_radius_m), 1e-9)
        if max_modes <= 0 or per_mode <= 0:
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)
        positions: list[NDArray[np.float64]] = []
        scores: list[float] = []
        particle_indices: list[int] = []
        for particle_idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            if st.num_sources <= 0 or st.positions.size == 0:
                continue
            count = min(int(st.num_sources), st.positions.shape[0], st.strengths.size)
            particle_weight = float(weights[particle_idx])
            for source_idx in range(count):
                strength = max(float(st.strengths[source_idx]), 0.0)
                if strength <= float(self.config.min_strength):
                    continue
                support = 0.0
                if (
                    st.support_scores is not None
                    and source_idx < st.support_scores.size
                ):
                    support = max(float(st.support_scores[source_idx]), 0.0)
                support_boost = 1.0 + max(
                    0.0,
                    float(
                        getattr(
                            self.config,
                            "mode_preserving_support_score_weight",
                            0.0,
                        )
                    ),
                ) * np.log1p(support)
                tentative_boost = 1.0
                if (
                    st.tentative_sources is not None
                    and source_idx < st.tentative_sources.size
                    and bool(st.tentative_sources[source_idx])
                ):
                    tentative_boost = max(
                        1.0,
                        float(
                            getattr(
                                self.config,
                                "mode_preserving_tentative_boost",
                                1.0,
                            )
                        ),
                    )
                residual_boost = (
                    max(
                        1.0,
                        float(
                            getattr(
                                self.config,
                                "mode_preserving_residual_boost",
                                1.0,
                            )
                        ),
                    )
                    if bool(getattr(self, "last_birth_residual_gate_passed", False))
                    else 1.0
                )
                positions.append(np.asarray(st.positions[source_idx], dtype=float))
                scores.append(
                    particle_weight
                    * strength
                    * support_boost
                    * tentative_boost
                    * residual_boost
                )
                particle_indices.append(particle_idx)
        if not scores:
            self.last_mode_preserving_strata_summary = {}
            self.last_mode_preserving_selected_strata = []
            self.last_mode_preserving_cardinality_summary = {}
            self.last_mode_preserving_selected_cardinalities = []
            return np.zeros(0, dtype=np.int64)

        pos_arr = np.vstack(positions)
        score_arr = np.asarray(scores, dtype=float)
        particle_arr = np.asarray(particle_indices, dtype=np.int64)
        strata_enabled = (
            bool(getattr(self.config, "mode_preserving_surface_strata", True))
            and self._source_prior_is_surface()
        )
        height_bin_m = max(
            0.0,
            float(getattr(self.config, "mode_preserving_height_bin_m", 0.0)),
        )
        labels: list[tuple[str, int]] = [("all", 0)] * int(pos_arr.shape[0])
        if strata_enabled:
            kinds = source_surface_kinds(
                pos_arr,
                self._source_prior_environment(),
                self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                tolerance_m=max(1.0e-6, 0.25 * radius),
            )
            if height_bin_m > 0.0:
                height_bins = np.floor(pos_arr[:, 2] / height_bin_m).astype(np.int64)
            else:
                height_bins = np.zeros(pos_arr.shape[0], dtype=np.int64)
            labels = [
                (
                    str(kind) if kind is not None else "off_surface",
                    int(height_bin),
                )
                for kind, height_bin in zip(kinds, height_bins)
            ]
        high_surface_extra = max(
            0,
            int(
                getattr(self.config, "mode_preserving_high_surface_extra_particles", 0)
            ),
        )
        room_z = max(float(self._source_prior_environment().size_z), 1.0e-9)
        high_z_threshold = (
            float(
                np.clip(
                    float(
                        getattr(
                            self.config,
                            "mode_preserving_high_surface_z_fraction",
                            0.75,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
            * room_z
        )
        total_score = max(float(np.sum(score_arr)), 1e-300)
        strata_scores: dict[str, float] = {}
        for label, score in zip(labels, score_arr):
            key = f"{label[0]}:zbin{int(label[1])}"
            strata_scores[key] = strata_scores.get(key, 0.0) + float(score)
        self.last_mode_preserving_strata_summary = {
            key: float(value) / total_score
            for key, value in sorted(strata_scores.items())
        }
        min_score = (
            max(float(self.config.mode_preserving_min_weight_fraction), 0.0)
            * total_score
        )
        order = np.argsort(score_arr)[::-1]
        centers = np.empty((len(order), 3), dtype=float)
        cluster_scores = np.empty(len(order), dtype=float)
        cluster_labels: list[tuple[str, int]] = []
        cluster_members: list[list[int]] = []
        cluster_count = 0
        for entry_idx in order:
            pos = pos_arr[entry_idx]
            entry_score = float(score_arr[entry_idx])
            entry_label = labels[int(entry_idx)]
            if cluster_count > 0:
                distances = np.linalg.norm(centers[:cluster_count] - pos, axis=1)
                label_matches = np.asarray(
                    [
                        cluster_labels[int(cluster_idx)] == entry_label
                        for cluster_idx in range(cluster_count)
                    ],
                    dtype=bool,
                )
                matches = np.flatnonzero((distances <= radius) & label_matches)
            else:
                matches = np.zeros(0, dtype=np.int64)
            if matches.size:
                cluster_idx = int(matches[0])
                old_weight = float(cluster_scores[cluster_idx])
                weight = old_weight + entry_score
                centers[cluster_idx] = (
                    centers[cluster_idx] * old_weight + pos * entry_score
                ) / max(weight, 1.0e-300)
                cluster_scores[cluster_idx] = weight
                cluster_members[cluster_idx].append(int(entry_idx))
                continue
            centers[cluster_count] = pos
            cluster_scores[cluster_count] = entry_score
            cluster_labels.append(entry_label)
            cluster_members.append([int(entry_idx)])
            cluster_count += 1
        sorted_clusters = np.argsort(cluster_scores[:cluster_count])[::-1]
        selected_clusters: list[int] = []
        if strata_enabled:
            seen_labels: set[tuple[str, int]] = set()
            for cluster_idx in sorted_clusters:
                label = cluster_labels[int(cluster_idx)]
                if label in seen_labels:
                    continue
                selected_clusters.append(int(cluster_idx))
                seen_labels.add(label)
                if len(selected_clusters) >= max_modes:
                    break
        selected_set = set(selected_clusters)
        for cluster_idx in sorted_clusters:
            if len(selected_clusters) >= max_modes:
                break
            cluster_int = int(cluster_idx)
            if cluster_int in selected_set:
                continue
            selected_clusters.append(cluster_int)
            selected_set.add(cluster_int)

        dynamic_spatial_enabled = bool(
            getattr(
                self.config,
                "mode_preserving_dynamic_spatial_allocation",
                False,
            )
        )
        dynamic_spatial_extra = max(
            0,
            int(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_spatial_extra_particles",
                    0,
                )
            ),
        )
        dynamic_spatial_min_fraction = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_spatial_min_score_fraction",
                    0.005,
                )
            ),
        )
        protected: list[int] = []
        protected_lookup: set[int] = set()
        selected_details: list[dict[str, object]] = []
        dynamic_spatial_details: list[dict[str, object]] = []
        for cluster_idx in selected_clusters:
            cluster_int = int(cluster_idx)
            label = cluster_labels[cluster_int]
            members = cluster_members[cluster_int]
            ranked_members = sorted(
                members,
                key=lambda idx: float(score_arr[idx]),
                reverse=True,
            )
            protected_for_cluster: list[int] = []
            cluster_score = float(cluster_scores[cluster_int])
            accepted = cluster_score >= min_score
            if not accepted:
                selected_details.append(
                    {
                        "surface": str(label[0]),
                        "height_bin": int(label[1]),
                        "center": [
                            float(value) for value in centers[cluster_int].astype(float)
                        ],
                        "high_surface": bool(
                            str(label[0]) in {"ceiling", "high_wall", "obstacle_top"}
                            or float(centers[cluster_int][2]) >= high_z_threshold
                        ),
                        "score": cluster_score,
                        "score_fraction": cluster_score / total_score,
                        "member_count": int(len(members)),
                        "protected_particles": [],
                        "protected_count": 0,
                        "accepted": False,
                    }
                )
                continue
            high_surface = bool(
                str(label[0]) in {"ceiling", "high_wall", "obstacle_top"}
                or float(centers[cluster_int][2]) >= high_z_threshold
            )
            target_per_mode = per_mode + (high_surface_extra if high_surface else 0)
            added = 0
            for member_idx in ranked_members:
                particle_idx = int(particle_arr[member_idx])
                if particle_idx in protected_lookup:
                    continue
                protected.append(particle_idx)
                protected_lookup.add(particle_idx)
                protected_for_cluster.append(particle_idx)
                added += 1
                if added >= target_per_mode:
                    break
            dynamic_spatial_added: dict[int, int] = {}
            dynamic_spatial_active = bool(
                dynamic_spatial_enabled
                and dynamic_spatial_extra > 0
                and (cluster_score / total_score) >= dynamic_spatial_min_fraction
            )
            if dynamic_spatial_active:
                source_counts = sorted(
                    {
                        int(
                            self.continuous_particles[
                                int(particle_arr[member_idx])
                            ].state.num_sources
                        )
                        for member_idx in ranked_members
                    }
                )
                for source_count in source_counts:
                    added_for_count = 0
                    for member_idx in ranked_members:
                        particle_idx = int(particle_arr[member_idx])
                        particle_count = int(
                            self.continuous_particles[particle_idx].state.num_sources
                        )
                        if particle_count != int(source_count):
                            continue
                        if particle_idx in protected_lookup:
                            continue
                        protected.append(particle_idx)
                        protected_lookup.add(particle_idx)
                        protected_for_cluster.append(particle_idx)
                        added_for_count += 1
                        if added_for_count >= dynamic_spatial_extra:
                            break
                    if added_for_count > 0:
                        dynamic_spatial_added[int(source_count)] = int(added_for_count)
                if dynamic_spatial_added:
                    dynamic_spatial_details.append(
                        {
                            "surface": str(label[0]),
                            "height_bin": int(label[1]),
                            "center": [
                                float(value)
                                for value in centers[cluster_int].astype(float)
                            ],
                            "score_fraction": cluster_score / total_score,
                            "extra_per_cardinality": int(dynamic_spatial_extra),
                            "protected_by_cardinality": {
                                str(key): int(value)
                                for key, value in sorted(dynamic_spatial_added.items())
                            },
                        }
                    )
            selected_details.append(
                {
                    "surface": str(label[0]),
                    "height_bin": int(label[1]),
                    "center": [
                        float(value) for value in centers[cluster_int].astype(float)
                    ],
                    "high_surface": bool(high_surface),
                    "score": cluster_score,
                    "score_fraction": cluster_score / total_score,
                    "member_count": int(len(members)),
                    "protected_particles": protected_for_cluster,
                    "protected_count": int(len(protected_for_cluster)),
                    "accepted": True,
                    "dynamic_spatial_protected": bool(dynamic_spatial_active),
                    "dynamic_spatial_counts": {
                        str(key): int(value)
                        for key, value in sorted(dynamic_spatial_added.items())
                    },
                }
            )
        self.last_mode_preserving_dynamic_spatial_summary = dynamic_spatial_details
        cardinality_details: list[dict[str, object]] = []
        self.last_mode_preserving_cardinality_summary = {}
        self.last_mode_preserving_selected_cardinalities = []
        if bool(getattr(self.config, "mode_preserving_cardinality_strata", True)):
            min_per_cardinality = max(
                0,
                int(
                    getattr(
                        self.config,
                        "mode_preserving_min_particles_per_cardinality",
                        0,
                    )
                ),
            )
            dynamic_cardinality = bool(
                getattr(
                    self.config,
                    "mode_preserving_dynamic_cardinality_allocation",
                    False,
                )
            )
            dynamic_extra = max(
                0,
                int(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_extra_particles",
                        0,
                    )
                ),
            )
            dynamic_min_mass = max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_min_mass",
                        0.02,
                    )
                ),
            )
            dynamic_entropy_min = max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "mode_preserving_dynamic_cardinality_entropy_min",
                        0.5,
                    )
                ),
            )
            if min_per_cardinality > 0:
                particle_counts = np.asarray(
                    [
                        max(0, int(particle.state.num_sources))
                        for particle in self.continuous_particles
                    ],
                    dtype=np.int64,
                )
                particle_weights = np.asarray(weights, dtype=float).reshape(-1)
                total_mass = max(float(np.sum(particle_weights)), 1.0e-300)
                protected_lookup = set(int(value) for value in protected)
                source_counts = sorted(set(int(value) for value in particle_counts))
                mass_by_count: dict[int, float] = {}
                for source_count in source_counts:
                    member_indices = np.flatnonzero(particle_counts == source_count)
                    mass_by_count[int(source_count)] = (
                        float(np.sum(particle_weights[member_indices])) / total_mass
                    )
                mass_arr = np.asarray(list(mass_by_count.values()), dtype=float)
                positive_mass = mass_arr[mass_arr > 0.0]
                cardinality_entropy = 0.0
                if positive_mass.size > 1:
                    cardinality_entropy = float(
                        -np.sum(positive_mass * np.log(positive_mass))
                    )
                dynamic_active = bool(
                    dynamic_cardinality
                    and dynamic_extra > 0
                    and cardinality_entropy >= dynamic_entropy_min
                )
                for source_count in source_counts:
                    member_indices = np.flatnonzero(particle_counts == source_count)
                    if member_indices.size == 0:
                        continue
                    mass_fraction = mass_by_count[int(source_count)]
                    self.last_mode_preserving_cardinality_summary[
                        str(int(source_count))
                    ] = mass_fraction
                    ranked = member_indices[
                        np.argsort(particle_weights[member_indices])[::-1]
                    ]
                    protected_for_count: list[int] = []
                    target_for_count = min_per_cardinality
                    dynamically_protected = bool(
                        dynamic_active and mass_fraction >= dynamic_min_mass
                    )
                    if dynamically_protected:
                        target_for_count += dynamic_extra
                    for particle_idx in ranked:
                        idx_int = int(particle_idx)
                        if idx_int in protected_lookup:
                            continue
                        protected.append(idx_int)
                        protected_lookup.add(idx_int)
                        protected_for_count.append(idx_int)
                        if len(protected_for_count) >= target_for_count:
                            break
                    cardinality_details.append(
                        {
                            "num_sources": int(source_count),
                            "mass_fraction": mass_fraction,
                            "member_count": int(member_indices.size),
                            "dynamically_protected": bool(dynamically_protected),
                            "cardinality_entropy": cardinality_entropy,
                            "target_protected_count": int(target_for_count),
                            "protected_particles": protected_for_count,
                            "protected_count": int(len(protected_for_count)),
                        }
                    )
        self.last_mode_preserving_selected_cardinalities = cardinality_details
        self.last_mode_preserving_selected_strata = selected_details
        return np.asarray(protected, dtype=np.int64)

    def _inject_mode_preserving_indices(
        self,
        indices: NDArray[np.int64],
        protected: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Inject protected particle indices into a resampling draw."""
        idx = np.asarray(indices, dtype=np.int64).copy()
        protected_arr = np.asarray(protected, dtype=np.int64)
        if idx.size == 0 or protected_arr.size == 0:
            return idx
        valid = protected_arr[
            (protected_arr >= 0) & (protected_arr < len(self.continuous_particles))
        ]
        if valid.size == 0:
            return idx
        unique_protected = []
        for value in valid.tolist():
            if value not in unique_protected:
                unique_protected.append(int(value))
        counts = np.bincount(idx, minlength=len(self.continuous_particles))
        missing = [value for value in unique_protected if counts[value] == 0]
        if not missing:
            return idx
        replace_slots: list[int] = []
        for slot, value in enumerate(idx):
            if counts[value] > 1:
                replace_slots.append(slot)
                counts[value] -= 1
                if len(replace_slots) >= len(missing):
                    break
        if len(replace_slots) < len(missing):
            for slot, value in enumerate(idx):
                if slot in replace_slots or value in unique_protected:
                    continue
                replace_slots.append(slot)
                if len(replace_slots) >= len(missing):
                    break
        for slot, value in zip(replace_slots, missing):
            idx[slot] = int(value)
        self.last_mode_preserved_count += int(len(replace_slots))
        return idx

    def _cardinality_preserving_resample_draw(
        self,
        weights: NDArray[np.float64],
        protected_indices: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]] | None:
        """Return resampling indices and log-weights preserving source-count mass.

        Spatial source-mode protection is applied inside each cardinality group
        so that preserving K-mass does not accidentally discard distinct
        same-isotope modes during a shield burst.
        """
        if not bool(self.config.cardinality_preserving_resample):
            return None
        min_stations = max(
            0,
            int(getattr(self.config, "cardinality_preserving_min_stations", 0)),
        )
        if self._distinct_observed_station_count() < min_stations:
            return None
        if (
            bool(
                getattr(
                    self.config,
                    "cardinality_preserving_require_confirmed_structure",
                    False,
                )
            )
            and not self._confirmed_source_structure()
        ):
            return None
        n_particles = len(self.continuous_particles)
        if n_particles <= 0 or weights.size != n_particles:
            return None
        labels = np.asarray(
            [
                max(0, int(particle.state.num_sources))
                for particle in self.continuous_particles
            ],
            dtype=np.int64,
        )
        unique_labels = np.unique(labels)
        if unique_labels.size <= 1:
            return None
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, np.inf)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            return None
        w = w / total
        masses = np.array(
            [float(np.sum(w[labels == label])) for label in unique_labels]
        )
        active = masses > 0.0
        if not np.any(active):
            return None
        unique_labels = unique_labels[active]
        masses = masses[active]
        desired = masses * float(n_particles)
        counts = np.floor(desired).astype(np.int64)
        counts = np.maximum(counts, 1)
        while int(np.sum(counts)) > n_particles:
            removable = np.where(counts > 1)[0]
            if removable.size == 0:
                break
            idx = int(
                removable[np.argmin(desired[removable] - np.floor(desired[removable]))]
            )
            counts[idx] -= 1
        while int(np.sum(counts)) < n_particles:
            idx = int(np.argmax(desired - counts))
            counts[idx] += 1
        drawn: list[int] = []
        log_weights_after: list[float] = []
        protected_arr = (
            np.asarray(protected_indices, dtype=np.int64)
            if protected_indices is not None
            else np.zeros(0, dtype=np.int64)
        )
        for label, mass, count in zip(unique_labels, masses, counts):
            group_idx = np.flatnonzero(labels == label)
            if group_idx.size == 0 or count <= 0:
                continue
            local_w = w[group_idx]
            local_draw = systematic_resample_count(local_w, count=int(count))
            selected = group_idx[local_draw]
            if protected_arr.size:
                valid_protected = protected_arr[
                    (protected_arr >= 0) & (protected_arr < n_particles)
                ]
                group_protected = valid_protected[labels[valid_protected] == label]
                selected = self._inject_mode_preserving_indices(
                    selected,
                    group_protected,
                )
            drawn.extend(int(value) for value in selected.tolist())
            per_particle_weight = float(mass) / max(int(count), 1)
            log_weights_after.extend(
                [float(np.log(max(per_particle_weight, 1.0e-300)))] * int(count)
            )
        if len(drawn) != n_particles or len(log_weights_after) != n_particles:
            return None
        return np.asarray(drawn, dtype=np.int64), np.asarray(
            log_weights_after, dtype=float
        )

    def _maybe_resample_continuous(
        self,
        *,
        disable_regularize: bool = False,
        roughening_scale: float = 1.0,
    ) -> None:
        """ESS check and systematic resampling for continuous particles (Sec. 3.3.4, Eq. 3.29)."""
        w = np.asarray(self.continuous_weights, dtype=np.float64)
        self.last_mode_preserved_count = 0
        self.last_mode_preserving_strata_summary = {}
        self.last_mode_preserving_selected_strata = []
        self.last_mode_preserving_cardinality_summary = {}
        self.last_mode_preserving_selected_cardinalities = []
        if w.size == 0:
            self.last_ess = 0.0
            self.last_ess_pre = 0.0
            self.last_ess_post = 0.0
            self.last_resample_ess = False
            return
        ess = 1.0 / max(np.sum(w**2), 1e-12)
        self.last_ess = float(ess)
        self.last_ess_pre = float(ess)
        self.last_ess_post = None
        self.last_resample_ess = False
        if ess < self.config.resample_threshold * self.N:
            debug_timing = _pf_debug_timing_enabled()
            debug_start = time.perf_counter()
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=resample_start ess={float(ess):.3f} "
                    f"n={len(self.continuous_particles)}",
                    flush=True,
                )
            self.last_resample_ess = True
            self.last_resample_count += 1
            logw = np.log(np.clip(w, 1e-300, 1.0))
            protected_idx = self._source_mode_preserving_indices(w)
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=mode_protection_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"protected={int(protected_idx.size)}",
                    flush=True,
                )
            cardinality_draw = self._cardinality_preserving_resample_draw(
                w,
                protected_indices=protected_idx,
            )
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=cardinality_draw_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"used={cardinality_draw is not None}",
                    flush=True,
                )
            if cardinality_draw is None:
                idx = systematic_resample(logw)
                idx = self._inject_mode_preserving_indices(idx, protected_idx)
                log_weights_after = np.full(
                    idx.size,
                    float(-np.log(max(idx.size, 1))),
                    dtype=float,
                )
            else:
                idx, log_weights_after = cardinality_draw
            self.continuous_particles = [
                self.continuous_particles[i].state.copy() for i in idx
            ]
            self.continuous_particles = [
                IsotopeParticle(state=st, log_weight=float(log_weight))
                for st, log_weight in zip(self.continuous_particles, log_weights_after)
            ]
            post_w = np.asarray(self.continuous_weights, dtype=np.float64)
            self.last_ess_post = float(1.0 / max(np.sum(post_w**2), 1.0e-12))
            roughening_scale = max(0.0, float(roughening_scale))
            surface_rejuvenation = bool(
                getattr(self.config, "surface_rejuvenation_enable", True)
            )
            if (
                surface_rejuvenation
                and not disable_regularize
                and roughening_scale > 0.0
            ):
                mult = self._roughening_multiplier()
                sigma_pos = (
                    self._roughening_sigma_pos(len(self.continuous_particles))
                    * mult
                    * roughening_scale
                )
                self.regularize_continuous(
                    sigma_pos=sigma_pos,
                    strength_log_sigma=(
                        self.config.strength_log_sigma * mult * roughening_scale
                    ),
                    p_birth=self.config.p_birth,
                    p_kill=self.config.p_kill,
                    intensity_threshold=self.config.min_strength,
                )
            self._resample_count_in_observation += 1
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} "
                    f"phase=resample_done elapsed={time.perf_counter() - debug_start:.3f}s",
                    flush=True,
                )

    def _maybe_resample_after_structural_update(self) -> bool:
        """Resample after delayed structural moves if their weight ratios collapse ESS."""
        roughening_scale = max(
            0.0,
            float(self.config.deferred_resample_roughening_scale),
        )
        disable_regularize = roughening_scale <= 0.0
        self._maybe_resample_continuous(
            disable_regularize=disable_regularize,
            roughening_scale=roughening_scale,
        )
        resampled = bool(self.last_resample_ess)
        if resampled:
            self._trigger_adapt_cooldown()
        return resampled

    def _label_scales(
        self,
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Return (pos_scale, strength_scale) for label alignment costs."""
        if self.config.label_pos_scale is not None:
            pos_scale = float(self.config.label_pos_scale)
        else:
            span = np.array(self.config.position_max, dtype=float) - np.array(
                self.config.position_min, dtype=float
            )
            pos_scale = float(np.linalg.norm(span))
        if pos_scale <= 0.0:
            pos_scale = 1.0
        if self.config.label_strength_scale is not None:
            strength_scale = float(self.config.label_strength_scale)
        else:
            positive = ref_strengths[ref_strengths > 0]
            strength_scale = float(np.median(positive)) if positive.size else 1.0
        if strength_scale <= 0.0:
            strength_scale = 1.0
        return pos_scale, strength_scale

    def _label_cost_matrix(
        self,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
        pos_scale: float,
        strength_scale: float,
    ) -> NDArray[np.float64]:
        """Compute the label-alignment cost matrix between particle and reference sources."""
        if positions.size == 0 or ref_positions.size == 0:
            return np.zeros((positions.shape[0], ref_positions.shape[0]), dtype=float)
        if positions.shape[0] * ref_positions.shape[0] <= 64:
            pos_diff = positions[:, None, :] - ref_positions[None, :, :]
            pos_cost = np.linalg.norm(pos_diff, axis=-1) / float(pos_scale)
            str_cost = np.abs(strengths[:, None] - ref_strengths[None, :]) / float(
                strength_scale
            )
            return np.asarray(
                self.config.label_pos_weight * pos_cost
                + self.config.label_strength_weight * str_cost,
                dtype=float,
            )
        self._gpu_enabled()
        import torch
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
        ref_pos_t = torch.as_tensor(ref_positions, device=device, dtype=dtype)
        str_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        ref_str_t = torch.as_tensor(ref_strengths, device=device, dtype=dtype)
        diff = pos_t[:, None, :] - ref_pos_t[None, :, :]
        pos_cost = torch.linalg.norm(diff, dim=-1) / float(pos_scale)
        str_cost = torch.abs(str_t[:, None] - ref_str_t[None, :]) / float(
            strength_scale
        )
        cost = (
            self.config.label_pos_weight * pos_cost
            + self.config.label_strength_weight * str_cost
        )
        return cost.detach().cpu().numpy()

    def _align_particle_to_reference(
        self,
        particle: IsotopeParticle,
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
        pos_scale: float,
        strength_scale: float,
    ) -> None:
        """Reorder a particle's sources to best match the reference ordering."""
        from scipy.optimize import linear_sum_assignment

        st = particle.state
        if st.num_sources == 0 or ref_positions.size == 0:
            return
        self._ensure_source_metadata(st)
        cost = self._label_cost_matrix(
            positions=st.positions,
            strengths=st.strengths,
            ref_positions=ref_positions,
            ref_strengths=ref_strengths,
            pos_scale=pos_scale,
            strength_scale=strength_scale,
        )
        n_rows, n_cols = cost.shape
        size = max(n_rows, n_cols)
        padded = np.full(
            (size, size), float(self.config.label_missing_cost), dtype=float
        )
        padded[:n_rows, :n_cols] = cost
        row_ind, col_ind = linear_sum_assignment(padded)
        assigned = {c: r for r, c in zip(row_ind, col_ind) if r < n_rows and c < n_cols}
        ordered_pos: list[NDArray[np.float64]] = []
        ordered_str: list[float] = []
        ordered_rows: list[int] = []
        used_rows: set[int] = set()
        for ref_idx in range(n_cols):
            row = assigned.get(ref_idx)
            if row is None:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
            ordered_rows.append(row)
            used_rows.add(row)
        for row in range(n_rows):
            if row in used_rows:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
            ordered_rows.append(row)
        if ordered_pos:
            st.positions = np.vstack(ordered_pos)
            st.strengths = np.array(ordered_str, dtype=float)
            st.ages = st.ages[ordered_rows]
            st.support_scores = st.support_scores[ordered_rows]
            st.tentative_sources = st.tentative_sources[ordered_rows]
            st.verification_fail_streaks = st.verification_fail_streaks[ordered_rows]
            st.num_sources = st.positions.shape[0]

    def _reference_from_particles(
        self, ref_count: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute a weighted reference ordering from the aligned particle set."""
        if ref_count <= 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        w = self.continuous_weights
        positions = np.zeros((ref_count, 3), dtype=float)
        strengths = np.zeros(ref_count, dtype=float)
        for j in range(ref_count):
            pos_list = []
            str_list = []
            w_list = []
            for wi, p in zip(w, self.continuous_particles):
                if p.state.num_sources > j:
                    pos_list.append(p.state.positions[j])
                    str_list.append(p.state.strengths[j])
                    w_list.append(wi)
            if not w_list:
                continue
            wj = np.array(w_list, dtype=float)
            wj = wj / max(np.sum(wj), 1e-12)
            pos_arr = np.vstack(pos_list)
            str_arr = np.array(str_list, dtype=float)
            positions[j] = np.sum(wj[:, None] * pos_arr, axis=0)
            strengths[j] = float(np.sum(wj * str_arr))
        return positions, strengths

    def align_continuous_labels(self) -> None:
        """
        Align per-particle source ordering to mitigate label switching.

        Uses Hungarian assignment against a reference ordering built from the
        highest-weight particle, then refines the reference iteratively.
        """
        if self._structural_kernel_is_exact():
            for particle in self.continuous_particles:
                self._canonicalize_structural_rj_state(particle.state)
            self._label_reference = None
            return
        if not self.config.label_enable or not self.continuous_particles:
            return
        ref_state = self._label_reference or self.best_particle().state
        if ref_state.num_sources == 0:
            return
        if ref_state.num_sources == 1:
            self._label_reference = ref_state.copy()
            return
        ref_positions = ref_state.positions.copy()
        ref_strengths = ref_state.strengths.copy()
        pos_scale, strength_scale = self._label_scales(ref_positions, ref_strengths)
        for _ in range(max(1, int(self.config.label_alignment_iters))):
            for particle in self.continuous_particles:
                self._align_particle_to_reference(
                    particle=particle,
                    ref_positions=ref_positions,
                    ref_strengths=ref_strengths,
                    pos_scale=pos_scale,
                    strength_scale=strength_scale,
                )
            ref_positions, ref_strengths = self._reference_from_particles(
                ref_positions.shape[0]
            )
        self._label_reference = IsotopeState(
            num_sources=ref_positions.shape[0],
            positions=ref_positions,
            strengths=ref_strengths,
            background=0.0,
        )

    def adapt_num_particles(
        self, *, ess_pre: float | None = None, resampled: bool = False
    ) -> None:
        """
        Optional: adapt N based on variance/entropy of weights (Chapter 3.3.4).

        Uses ess_pre when provided to avoid the resampling inflation of ESS.
        Resampling or cooldown windows only allow growth.
        """
        if not self.continuous_particles:
            self.last_n_after_adapt = 0
            return
        min_particles = (
            max(1, int(self.config.min_particles))
            if self.config.min_particles is not None
            else max(1, int(self.config.num_particles))
        )
        max_particles = (
            max(1, int(self.config.max_particles))
            if self.config.max_particles is not None
            else max(1, int(self.config.num_particles))
        )
        if ess_pre is None:
            w = self.continuous_weights
            if w.size == 0:
                ess_pre = 0.0
            else:
                ess_pre = float(1.0 / max(np.sum(w**2), 1e-12))
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        allow_shrink = not resampled and self._adapt_cooldown_remaining <= 0
        if (
            ess_ratio < self.config.ess_low
            and len(self.continuous_particles) < max_particles
        ):
            grown = max(
                len(self.continuous_particles) + 1,
                int(len(self.continuous_particles) * 1.25),
            )
            target = min(max_particles, grown)
            self._resample_continuous_to(target, jitter=True)
        elif (
            allow_shrink
            and ess_ratio > self.config.ess_high
            and len(self.continuous_particles) > min_particles
        ):
            target = max(min_particles, int(len(self.continuous_particles) * 0.8))
            self._resample_continuous_to(target, jitter=False)
        self.last_n_after_adapt = int(len(self.continuous_particles))

    def _resample_continuous_to(self, target_n: int, jitter: bool = False) -> None:
        """Resample the continuous particles to a new population size."""
        target_n = max(1, int(target_n))
        self.last_resample_count += 1
        w = self.continuous_weights
        protected_idx = self._source_mode_preserving_indices(w)
        idx = np.random.choice(len(self.continuous_particles), size=target_n, p=w)
        idx = self._inject_mode_preserving_indices(idx, protected_idx)
        states = [self.continuous_particles[i].state.copy() for i in idx]
        self.continuous_particles = [
            IsotopeParticle(state=st, log_weight=float(-np.log(target_n)))
            for st in states
        ]
        self.N = target_n
        self.config.num_particles = target_n
        if jitter:
            mult = self._roughening_multiplier()
            sigma_pos = (
                self._roughening_sigma_pos(len(self.continuous_particles)) * mult
            )
            self.regularize_continuous(
                sigma_pos=sigma_pos,
                strength_log_sigma=self.config.strength_log_sigma * mult,
                p_birth=self.config.p_birth,
                p_kill=self.config.p_kill,
                intensity_threshold=self.config.min_strength,
            )
        self._resample_count_in_observation += 1

    def best_particle(self) -> IsotopeParticle:
        """Return the particle with maximum log_weight."""
        return max(self.continuous_particles, key=lambda p: p.log_weight)

    def _resize_metadata_array(
        self,
        arr: NDArray[np.float64] | NDArray[np.int64] | None,
        size: int,
        fill_value: float,
        dtype: type,
    ) -> NDArray:
        """Resize or initialize a metadata array to a target length."""
        if arr is None:
            return np.full(size, fill_value, dtype=dtype)
        arr = np.asarray(arr)
        if arr.size == size:
            return arr.astype(dtype, copy=False)
        if arr.size < size:
            pad = np.full(size - arr.size, fill_value, dtype=dtype)
            return np.concatenate([arr.astype(dtype, copy=False), pad])
        return arr[:size].astype(dtype, copy=False)

    def _ensure_source_metadata(self, st: IsotopeState) -> None:
        """Ensure per-source metadata arrays exist and match num_sources."""
        r = int(st.num_sources)
        st.ages = self._resize_metadata_array(st.ages, r, 0, int)
        st.support_scores = self._resize_metadata_array(
            st.support_scores, r, 0.0, float
        )
        st.tentative_sources = self._resize_metadata_array(
            st.tentative_sources,
            r,
            False,
            bool,
        )
        st.verification_fail_streaks = self._resize_metadata_array(
            st.verification_fail_streaks,
            r,
            0,
            int,
        )

    def _pseudo_source_fail_grace(self) -> int:
        """Return failed-verification count needed before quarantine or prune."""
        return max(
            0,
            int(self.config.pseudo_source_fail_grace_stations),
            int(self.config.source_prune_fail_grace_stations),
        )

    def _quarantined_source_mask(self, st: IsotopeState) -> NDArray[np.bool_]:
        """Return tentative sources currently quarantined by verification failures."""
        if st.num_sources <= 0:
            return np.zeros(0, dtype=bool)
        self._ensure_source_metadata(st)
        count = int(st.num_sources)
        tentative = np.asarray(st.tentative_sources[:count], dtype=bool)
        failed = np.asarray(st.verification_fail_streaks[:count], dtype=int)
        if failed.size != count:
            failed_padded = np.zeros(count, dtype=int)
            failed_padded[: min(failed.size, count)] = failed[:count]
            failed = failed_padded
        grace = self._pseudo_source_fail_grace()
        return tentative & (failed > 0) & (failed >= grace)

    def _active_source_mask(
        self,
        st: IsotopeState,
    ) -> NDArray[np.bool_]:
        """Return the mask of physical sources represented by a PF state."""
        if st.num_sources <= 0:
            return np.zeros(0, dtype=bool)
        self._ensure_source_metadata(st)
        return np.ones(int(st.num_sources), dtype=bool)

    def _apply_source_keep_mask(
        self,
        st: IsotopeState,
        keep: NDArray[np.bool_],
    ) -> int:
        """Apply a per-source keep mask and return the removed source count."""
        if st.num_sources <= 0:
            return 0
        self._ensure_source_metadata(st)
        keep_arr = np.asarray(keep, dtype=bool).ravel()[: int(st.num_sources)]
        if keep_arr.size != int(st.num_sources):
            raise ValueError("keep mask must match source count.")
        removed = int(np.count_nonzero(~keep_arr))
        if removed <= 0:
            return 0
        st.positions = st.positions[keep_arr]
        st.strengths = st.strengths[keep_arr]
        st.ages = st.ages[keep_arr]
        st.support_scores = st.support_scores[keep_arr]
        st.tentative_sources = st.tentative_sources[keep_arr]
        st.verification_fail_streaks = st.verification_fail_streaks[keep_arr]
        st.num_sources = int(st.positions.shape[0])
        return removed

    def _lambda_components(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (lambda_m, lambda_total) for a state across measurements."""
        if data.z_k.size == 0:
            return np.zeros((0, st.num_sources), dtype=float), np.zeros(0, dtype=float)
        lambda_m = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=st.positions,
            strengths=st.strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        background_counts = float(st.background) * data.live_times
        lambda_total = background_counts + np.sum(lambda_m, axis=1)
        return lambda_m, lambda_total

    def _particle_indices_by_source_count(
        self,
        particle_indices: list[int] | None = None,
    ) -> tuple[dict[int, list[int]], list[int]]:
        """Group valid particle indices by active source count for batched kernels."""
        if particle_indices is None:
            candidate_indices = range(len(self.continuous_particles))
        else:
            candidate_indices = [int(idx) for idx in particle_indices]
        grouped: dict[int, list[int]] = {}
        fallback_indices: list[int] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[idx].state
            self._ensure_source_metadata(st)
            source_count = max(0, int(st.num_sources))
            if source_count > 0 and (
                st.positions.ndim != 2
                or st.positions.shape[0] < source_count
                or st.strengths.size < source_count
            ):
                fallback_indices.append(idx)
                continue
            grouped.setdefault(source_count, []).append(idx)
        return grouped, fallback_indices

    def _lambda_components_for_particle_group(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        source_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return batched per-source and total counts for equal-cardinality particles."""
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        count = max(0, int(source_count))
        if num_meas == 0 or particle_count == 0:
            return (
                np.zeros((num_meas, particle_count, count), dtype=float),
                np.zeros((num_meas, particle_count), dtype=float),
            )
        k_tensor, background_counts, strengths = (
            self._unit_kernel_tensor_for_particle_group(
                data,
                particle_indices,
                count,
            )
        )
        lambda_m = k_tensor * strengths[None, :, :]
        lambda_total = background_counts + np.sum(lambda_m, axis=2)
        return lambda_m, lambda_total

    def _unit_kernel_tensor_for_particle_group(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        source_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Return unit-strength kernel tensor, background counts, and strengths.

        The tensor has shape K x P x S and is reused by batched structural
        evidence calculations so the geometry response is computed once per
        equal-cardinality particle group.
        """
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        count = max(0, int(source_count))
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        background_counts = data.live_times[:, None] * backgrounds[None, :]
        if count > 0 and particle_count > 0:
            strengths = np.vstack(
                [
                    np.asarray(
                        self.continuous_particles[idx].state.strengths[:count],
                        dtype=float,
                    )
                    for idx in particle_indices
                ]
            )
        else:
            strengths = np.zeros((particle_count, count), dtype=float)
        if num_meas == 0 or particle_count == 0 or count <= 0:
            return (
                np.zeros((num_meas, particle_count, count), dtype=float),
                background_counts,
                strengths,
            )
        sources = np.vstack(
            [
                np.asarray(
                    self.continuous_particles[idx].state.positions[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        unique_sources, inverse = np.unique(
            sources,
            axis=0,
            return_inverse=True,
        )
        has_duplicate_sources = unique_sources.shape[0] < sources.shape[0]
        evaluate_sources = unique_sources if has_duplicate_sources else sources
        unit_strengths = np.ones(evaluate_sources.shape[0], dtype=float)
        k_evaluated = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=evaluate_sources,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        if has_duplicate_sources:
            k_flat = np.asarray(k_evaluated, dtype=float)[:, inverse]
        else:
            k_flat = np.asarray(k_evaluated, dtype=float)
        k_tensor = np.asarray(k_flat, dtype=float).reshape(
            num_meas,
            particle_count,
            count,
        )
        return k_tensor, background_counts, strengths

    def _log_likelihood_and_delta_remove_group(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
        lambda_components: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return base likelihoods and removal losses for a batched group."""
        total = np.asarray(lambda_total, dtype=float)
        components = np.asarray(lambda_components, dtype=float)
        if components.ndim != 3 or components.shape[:2] != total.shape:
            particle_count = total.shape[1] if total.ndim == 2 else 0
            return (
                np.zeros(particle_count, dtype=float),
                np.zeros((particle_count, 0), dtype=float),
            )
        source_count = int(components.shape[2])
        if source_count <= 0:
            return (
                self._structural_count_log_likelihood_matrix_np(data, total),
                np.zeros((total.shape[1], 0), dtype=float),
            )
        particle_count = int(total.shape[1])
        base_ll = self._structural_count_log_likelihood_matrix_np(data, total)
        reduced = np.maximum(total[:, :, None] - components, 1.0e-12)
        reduced_flat = reduced.reshape(
            int(total.shape[0]),
            particle_count * source_count,
        )
        reduced_ll = self._structural_count_log_likelihood_matrix_np(
            data,
            reduced_flat,
        ).reshape(particle_count, source_count)
        return base_ll, base_ll[:, None] - reduced_ll

    def _delta_log_likelihood_remove_group(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
        lambda_components: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-particle, per-source removal support for a batched group."""
        _, delta_ll = self._log_likelihood_and_delta_remove_group(
            data,
            lambda_total,
            lambda_components,
        )
        return delta_ll

    def _compute_birth_proposal(
        self,
        data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None,
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            NDArray[np.float64],
            NDArray[np.float64],
        ]
        | None
    ):
        """
        Build residual-driven birth proposal and cached candidate responses.
        """
        if data is None or candidate_positions is None or candidate_positions.size == 0:
            return None
        if data.z_k.size == 0:
            return None
        if not self.continuous_particles:
            return None
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            return None
        topk = max(1, int(self.config.birth_topk_particles))
        order = np.argsort(weights)[::-1][:topk]
        sel_weights = weights[order]
        if np.sum(sel_weights) <= 0.0:
            sel_weights = np.ones_like(sel_weights, dtype=float)
        sel_weights = sel_weights / np.sum(sel_weights)
        layers = self._compute_birth_residual_layers(
            data=data,
            particle_indices=order,
            particle_weights=sel_weights,
        )
        self.last_birth_residual_layer_count = len(layers)
        if not layers:
            return None
        raw_layer = next((layer for layer in layers if layer.name == "raw"), None)
        raw_gate_passed_cache: bool | None = None
        if raw_layer is not None:
            raw_sum = float(np.sum(np.maximum(raw_layer.residual, 0.0)))
            raw_gate_passed = False
            if raw_sum > 0.0:
                raw_gate_passed = self._birth_residual_gate_allows(
                    raw_layer.residual,
                    data,
                )
            raw_gate_passed_cache = bool(raw_gate_passed)
        selected_layer: BirthResidualLayer | None = None
        selected_sum = 0.0
        for layer in sorted(
            layers,
            key=lambda item: float(np.sum(np.maximum(item.residual, 0.0))),
            reverse=True,
        ):
            residual_sum = float(np.sum(np.maximum(layer.residual, 0.0)))
            if residual_sum <= 0.0:
                continue
            if layer.name == "raw" and raw_gate_passed_cache is not None:
                gate_passed = raw_gate_passed_cache
            else:
                gate_passed = self._birth_residual_gate_allows(
                    layer.residual,
                    data,
                )
            if not gate_passed:
                continue
            selected_layer = layer
            selected_sum = residual_sum
            break
        if selected_layer is None:
            return None
        residual_mix = np.maximum(selected_layer.residual, 0.0)
        residual_sum = float(selected_sum)
        self.last_birth_residual_layer = str(selected_layer.name)

        base_candidates = self._project_and_deduplicate_birth_candidates(
            candidate_positions
        )
        base_candidates = self._exclude_birth_candidates_near_detectors(
            base_candidates,
            data,
        )
        if base_candidates.size == 0:
            return None

        unit_strengths = np.ones(base_candidates.shape[0], dtype=float)
        base_candidate_counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=base_candidates,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        base_support_mask = self._birth_candidate_support_mask(
            data=data,
            candidate_counts=base_candidate_counts,
            residual_mix=residual_mix,
        )
        if not np.any(base_support_mask):
            return None
        base_scores, _ = self._birth_residual_candidate_scores(
            candidate_counts=base_candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
        )
        base_supported_indices = np.flatnonzero(base_support_mask)
        num_jitter = max(0, int(self.config.birth_num_local_jitter))
        candidate_counts = base_candidate_counts[:, base_supported_indices]
        candidates = base_candidates[base_supported_indices]
        if num_jitter > 0 and base_supported_indices.size > 0:
            jitter_limit = self.config.birth_jitter_topk_candidates
            if jitter_limit is None:
                jitter_indices = base_supported_indices
            else:
                top_count = min(max(1, int(jitter_limit)), base_supported_indices.size)
                ranked = base_supported_indices[
                    np.argsort(base_scores[base_supported_indices])[::-1][:top_count]
                ]
                jitter_indices = ranked
            jitter_sigma = float(self.config.birth_candidate_jitter_sigma)
            jitter = np.random.normal(
                loc=0.0,
                scale=jitter_sigma,
                size=(jitter_indices.size, num_jitter, 3),
            )
            jittered = base_candidates[jitter_indices, None, :] + jitter
            jittered = self._project_positions_to_source_prior(jittered.reshape(-1, 3))
            jittered = self._exclude_birth_candidates_near_detectors(jittered, data)
            if jittered.size:
                jitter_counts = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=jittered,
                    strengths=np.ones(jittered.shape[0], dtype=float),
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        data.fe_indices,
                        data.pb_indices,
                    ),
                )
                candidate_counts = np.hstack([candidate_counts, jitter_counts])
                candidates = np.vstack([candidates, jittered])
                final_support_mask = self._birth_candidate_support_mask(
                    data=data,
                    candidate_counts=candidate_counts,
                    residual_mix=residual_mix,
                )
                if not np.any(final_support_mask):
                    return None
                candidate_counts = candidate_counts[:, final_support_mask]
                candidates = candidates[final_support_mask]
        scores, q_hat = self._birth_residual_candidate_scores(
            candidate_counts=candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
        )
        finite = (
            np.isfinite(scores) & np.isfinite(q_hat) & (scores > 0.0) & (q_hat > 0.0)
        )
        if not np.any(finite):
            return None
        candidate_counts = candidate_counts[:, finite]
        candidates = candidates[finite]
        scores = scores[finite]
        q_hat = q_hat[finite]
        if np.max(scores) <= 0.0:
            return None
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        q_hat = q_hat[order]
        kernel_sums = max(residual_sum, 1.0e-12) / np.maximum(q_hat, 1.0e-12)
        candidates = candidates[order]
        candidate_counts = candidate_counts[:, order]
        scores = np.maximum(scores, float(self.config.birth_min_score))
        temp = max(float(self.config.birth_softmax_temp), 1e-6)
        scaled = scores / temp
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        return probs, kernel_sums, residual_sum, candidates, candidate_counts

    def _compute_birth_residual_layers(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
    ) -> list[BirthResidualLayer]:
        """
        Return residual layers using batched per-cardinality expected counts.

        This is mathematically equivalent to the scalar residual-layer oracle,
        but it evaluates all selected particles with the same active source
        count in one expected-count kernel call.
        """
        if data.z_k.size == 0:
            return []
        indices = np.asarray(particle_indices, dtype=int).ravel()
        weights = np.asarray(particle_weights, dtype=float).ravel()
        if weights.size != indices.size:
            weights = np.ones(indices.size, dtype=float)
        valid = (indices >= 0) & (indices < len(self.continuous_particles))
        indices = indices[valid]
        weights = weights[valid]
        if indices.size == 0:
            return []
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)

        records: list[
            tuple[
                int,
                float,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.int64],
                NDArray[np.float64],
            ]
        ] = []
        groups: dict[int, list[int]] = {}
        for local_idx, particle_idx in enumerate(indices):
            st = self.continuous_particles[int(particle_idx)].state
            self._ensure_source_metadata(st)
            active_mask = self._active_source_mask(st)
            if st.num_sources > 0 and np.any(active_mask):
                active_positions = np.asarray(
                    st.positions[: st.num_sources][active_mask],
                    dtype=float,
                )
                active_strengths = np.asarray(
                    st.strengths[: st.num_sources][active_mask],
                    dtype=float,
                )
                active_indices = np.flatnonzero(active_mask).astype(np.int64)
            else:
                active_positions = np.zeros((0, 3), dtype=float)
                active_strengths = np.zeros(0, dtype=float)
                active_indices = np.zeros(0, dtype=np.int64)
            background_counts = float(st.background) * data.live_times
            record_idx = len(records)
            records.append(
                (
                    int(particle_idx),
                    float(weights[local_idx]),
                    active_positions,
                    active_strengths,
                    active_indices,
                    np.asarray(background_counts, dtype=float),
                )
            )
            groups.setdefault(int(active_positions.shape[0]), []).append(record_idx)

        lambda_by_record: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        for active_count, record_indices in groups.items():
            if active_count <= 0:
                for record_idx in record_indices:
                    background_counts = records[record_idx][5]
                    lambda_by_record[record_idx] = (
                        np.zeros((data.z_k.size, 0), dtype=float),
                        background_counts,
                    )
                continue
            stacked_positions = np.vstack(
                [records[record_idx][2] for record_idx in record_indices]
            )
            stacked_strengths = np.concatenate(
                [records[record_idx][3] for record_idx in record_indices]
            )
            lambda_flat = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=stacked_positions,
                strengths=stacked_strengths,
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            lambda_group = np.asarray(lambda_flat, dtype=float).reshape(
                int(data.z_k.size),
                int(len(record_indices)),
                int(active_count),
            )
            for local_group_idx, record_idx in enumerate(record_indices):
                background_counts = records[record_idx][5]
                lambda_m = lambda_group[:, local_group_idx, :]
                lambda_total = background_counts + np.sum(lambda_m, axis=1)
                lambda_by_record[record_idx] = (lambda_m, lambda_total)

        layer_parts: dict[str, list[NDArray[np.float64]]] = {"raw": []}
        weighted_lambda_total = np.zeros(data.z_k.size, dtype=float)
        cluster_records: list[
            tuple[NDArray[np.float64], NDArray[np.float64], float]
        ] = []
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        min_fraction = max(float(self.config.peak_suppression_min_source_fraction), 0.0)
        suppress_factor = float(
            np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0)
        )
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        for record_idx, record in enumerate(records):
            particle_idx, weight, _positions, _strengths, active_indices, _bg = record
            lambda_m, lambda_total = lambda_by_record[record_idx]
            weighted_lambda_total += weight * np.asarray(lambda_total, dtype=float)
            raw = self._clip_birth_residual(np.maximum(data.z_k - lambda_total, 0.0))
            layer_parts["raw"].append(raw * weight)
            if not allow_suppression or lambda_m.shape[1] == 0:
                continue
            st = self.continuous_particles[int(particle_idx)].state
            source_totals = np.sum(np.maximum(lambda_m, 0.0), axis=0)
            total_source = max(float(np.sum(source_totals)), 1.0e-12)
            strong_order = np.argsort(source_totals)[::-1]
            added = 0
            for source_idx in strong_order:
                if added >= max_layers - 1:
                    break
                if float(source_totals[int(source_idx)]) < min_fraction * total_source:
                    continue
                suppressed_total = (
                    lambda_total - suppress_factor * lambda_m[:, int(source_idx)]
                )
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"strong_suppressed_{added}"
                layer_parts.setdefault(layer_name, []).append(residual * weight)
                state_source_idx = int(active_indices[int(source_idx)])
                cluster_records.append(
                    (
                        np.asarray(st.positions[state_source_idx], dtype=float).copy(),
                        weight * np.asarray(lambda_m[:, int(source_idx)], dtype=float),
                        weight * float(source_totals[int(source_idx)]),
                    )
                )
                added += 1
        return self._finalize_birth_residual_layers(
            data=data,
            layer_parts=layer_parts,
            weighted_lambda_total=weighted_lambda_total,
            cluster_records=cluster_records,
        )

    def _compute_birth_residual_layers_scalar(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
    ) -> list[BirthResidualLayer]:
        """
        Return raw, source-suppressed, and cluster-suppressed residual layers.

        The raw layer is the usual positive residual after all existing sources.
        Peak-suppressed layers add back a strong source or source cluster before
        candidate ranking.  These layers are used only to propose new source
        hypotheses; the accepted move is still judged by the original full
        observation likelihood.
        """
        if data.z_k.size == 0:
            return []
        weights = np.asarray(particle_weights, dtype=float).ravel()
        particle_indices = np.asarray(particle_indices, dtype=int).ravel()
        if weights.size != particle_indices.size:
            weights = np.ones(particle_indices.size, dtype=float)
        if weights.size == 0:
            return []
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        layer_parts: dict[str, list[NDArray[np.float64]]] = {"raw": []}
        weighted_lambda_total = np.zeros(data.z_k.size, dtype=float)
        cluster_records: list[
            tuple[NDArray[np.float64], NDArray[np.float64], float]
        ] = []
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        min_fraction = max(float(self.config.peak_suppression_min_source_fraction), 0.0)
        suppress_factor = float(
            np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0)
        )
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        for local_idx, particle_idx in enumerate(particle_indices):
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            weight = float(weights[local_idx])
            st = self.continuous_particles[int(particle_idx)].state
            background_counts = float(st.background) * data.live_times
            active_mask = self._active_source_mask(st)
            if st.num_sources > 0 and np.any(active_mask):
                lambda_m = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=st.positions[: st.num_sources][active_mask],
                    strengths=st.strengths[: st.num_sources][active_mask],
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale_vector(
                        data.fe_indices,
                        data.pb_indices,
                    ),
                )
                lambda_total = background_counts + np.sum(lambda_m, axis=1)
            else:
                lambda_m = np.zeros((data.z_k.size, 0), dtype=float)
                lambda_total = background_counts
            weighted_lambda_total += weight * np.asarray(lambda_total, dtype=float)
            raw = self._clip_birth_residual(np.maximum(data.z_k - lambda_total, 0.0))
            layer_parts["raw"].append(raw * weight)
            if not allow_suppression or lambda_m.shape[1] == 0:
                continue
            source_totals = np.sum(np.maximum(lambda_m, 0.0), axis=0)
            total_source = max(float(np.sum(source_totals)), 1.0e-12)
            strong_order = np.argsort(source_totals)[::-1]
            active_indices = np.flatnonzero(active_mask)
            added = 0
            for source_idx in strong_order:
                if added >= max_layers - 1:
                    break
                if float(source_totals[int(source_idx)]) < min_fraction * total_source:
                    continue
                suppressed_total = (
                    lambda_total - suppress_factor * lambda_m[:, int(source_idx)]
                )
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"strong_suppressed_{added}"
                layer_parts.setdefault(layer_name, []).append(residual * weight)
                state_source_idx = int(active_indices[int(source_idx)])
                cluster_records.append(
                    (
                        np.asarray(st.positions[state_source_idx], dtype=float).copy(),
                        weight * np.asarray(lambda_m[:, int(source_idx)], dtype=float),
                        weight * float(source_totals[int(source_idx)]),
                    )
                )
                added += 1
        return self._finalize_birth_residual_layers(
            data=data,
            layer_parts=layer_parts,
            weighted_lambda_total=weighted_lambda_total,
            cluster_records=cluster_records,
        )

    def _finalize_birth_residual_layers(
        self,
        *,
        data: MeasurementData,
        layer_parts: dict[str, list[NDArray[np.float64]]],
        weighted_lambda_total: NDArray[np.float64],
        cluster_records: list[tuple[NDArray[np.float64], NDArray[np.float64], float]],
    ) -> list[BirthResidualLayer]:
        """Finalize weighted residual layer parts into ordered birth layers."""
        max_layers = max(1, int(self.config.residual_decomposition_max_layers))
        suppress_factor = float(
            np.clip(float(self.config.peak_suppression_factor), 0.0, 1.0)
        )
        allow_suppression = (
            bool(self.config.residual_decomposition_enable)
            and bool(self.config.peak_suppression_enable)
            and max_layers > 1
        )
        if allow_suppression and cluster_records and max_layers > 2:
            cluster_radius = max(
                float(self.config.cluster_eps_m),
                0.5 * float(self.config.birth_min_sep_m),
                1.0e-6,
            )
            cluster_components: list[NDArray[np.float64]] = []
            cluster_scores: list[float] = []
            used = np.zeros(len(cluster_records), dtype=bool)
            positions = np.vstack([record[0] for record in cluster_records])
            record_scores = np.asarray(
                [record[2] for record in cluster_records],
                dtype=float,
            )
            for seed_idx in np.argsort(record_scores)[::-1]:
                if used[int(seed_idx)]:
                    continue
                dists = np.linalg.norm(
                    positions - positions[int(seed_idx)][None, :],
                    axis=1,
                )
                members = np.flatnonzero((dists <= cluster_radius) & (~used))
                if members.size == 0:
                    continue
                used[members] = True
                component = np.sum(
                    np.vstack([cluster_records[int(idx)][1] for idx in members]),
                    axis=0,
                )
                score = float(np.sum([cluster_records[int(idx)][2] for idx in members]))
                if score <= 0.0 or float(np.sum(component)) <= 0.0:
                    continue
                cluster_components.append(component)
                cluster_scores.append(score)
            for layer_idx, cluster_idx in enumerate(np.argsort(cluster_scores)[::-1]):
                if layer_idx >= max_layers - 1:
                    break
                suppressed_total = (
                    weighted_lambda_total
                    - suppress_factor * cluster_components[int(cluster_idx)]
                )
                residual = self._clip_birth_residual(
                    np.maximum(data.z_k - suppressed_total, 0.0)
                )
                layer_name = f"leave_one_cluster_out_{layer_idx}"
                layer_parts.setdefault(layer_name, []).append(residual)
        layers: list[BirthResidualLayer] = []
        for name, parts in layer_parts.items():
            if not parts:
                continue
            stack = np.vstack(parts)
            residual = (
                np.sum(stack, axis=0)
                if bool(self.config.birth_use_weighted_topk)
                else np.mean(stack, axis=0)
            )
            if float(np.sum(np.maximum(residual, 0.0))) > 0.0:
                layers.append(BirthResidualLayer(name=name, residual=residual))
        raw_layers = [layer for layer in layers if layer.name == "raw"]
        aux_layers = [layer for layer in layers if layer.name != "raw"]
        aux_layers.sort(
            key=lambda layer: float(np.sum(np.maximum(layer.residual, 0.0))),
            reverse=True,
        )
        return raw_layers + aux_layers[: max(0, max_layers - len(raw_layers))]

    def _clip_birth_residual(
        self,
        residual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Clip residual outliers before residual-driven structural proposals."""
        clipped = np.asarray(residual, dtype=float).copy()
        clip_q = float(self.config.birth_residual_clip_quantile)
        if 0.0 < clip_q < 1.0 and clipped.size:
            clip_val = float(np.quantile(clipped, clip_q))
            clipped = np.minimum(clipped, clip_val)
        return clipped

    def _birth_residual_candidate_scores(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return shield-coded residual matching scores and fitted strengths."""
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        if not bool(self.config.birth_use_shield_coded_residual):
            sums = np.maximum(np.sum(counts, axis=0), 1.0e-12)
            residual_sum = max(float(np.sum(residual)), 0.0)
            scores = np.asarray(residual @ counts, dtype=float)
            q_hat = residual_sum / sums
            scores *= self._birth_count_distance_prior(
                candidate_counts=counts,
                residual_mix=residual,
                q_hat=q_hat,
            )
            return scores, q_hat
        variances = self._measurement_vector(
            observation_variances,
            counts.shape[0],
            "observation_variances",
            min_value=1.0e-12,
        )
        weights = 1.0 / variances
        numerator = np.sum(weights[:, None] * residual[:, None] * counts, axis=0)
        denominator = np.sum(weights[:, None] * counts * counts, axis=0)
        q_hat = np.maximum(numerator / np.maximum(denominator, 1.0e-12), 0.0)
        scores = numerator * q_hat
        scores *= self._birth_count_distance_prior(
            candidate_counts=counts,
            residual_mix=residual,
            q_hat=q_hat,
        )
        return np.asarray(scores, dtype=float), np.asarray(q_hat, dtype=float)

    def _birth_count_distance_prior(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        q_hat: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a soft proposal prior favoring high unit-response candidates."""
        response_weight = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_prior_weight", 0.0)),
        )
        strength_weight = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_strength_weight", 0.0)),
        )
        counts = np.maximum(np.asarray(candidate_counts, dtype=float), 0.0)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=float)
        num_candidates = counts.shape[1]
        if response_weight <= 0.0 and strength_weight <= 0.0:
            return np.ones(num_candidates, dtype=float)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        residual_sum = float(np.sum(residual))
        if residual_sum <= 0.0:
            return np.ones(num_candidates, dtype=float)

        eps = 1.0e-12
        prior = np.ones(num_candidates, dtype=float)
        residual_weights = residual / max(residual_sum, eps)
        unit_response = np.sum(residual_weights[:, None] * counts, axis=0)
        finite_response = np.isfinite(unit_response) & (unit_response > eps)
        log_clip = max(
            0.0,
            float(getattr(self.config, "birth_count_distance_log_clip", 3.0)),
        )

        if response_weight > 0.0 and np.any(finite_response):
            response_ref = float(np.median(unit_response[finite_response]))
            if response_ref > eps:
                log_response = np.log(np.maximum(unit_response, eps) / response_ref)
                prior *= np.exp(
                    response_weight * np.clip(log_response, -log_clip, log_clip)
                )

        q = np.maximum(np.asarray(q_hat, dtype=float).ravel(), 0.0)
        if q.size != num_candidates:
            raise ValueError("q_hat must have one value per candidate.")
        finite_q = np.isfinite(q) & (q > eps)
        if strength_weight > 0.0 and np.any(finite_q):
            q_ref = float(np.median(q[finite_q]))
            sigma = max(
                float(
                    getattr(
                        self.config,
                        "birth_count_distance_strength_sigma",
                        2.0,
                    )
                ),
                eps,
            )
            if q_ref > eps:
                log_q = np.log(np.maximum(q, eps) / q_ref)
                high_strength = np.maximum(log_q, 0.0)
                high_strength = np.clip(high_strength, 0.0, log_clip)
                prior *= np.exp(-0.5 * strength_weight * (high_strength / sigma) ** 2)

        finite_prior = np.isfinite(prior) & (prior > 0.0)
        if np.any(finite_prior):
            norm = float(np.median(prior[finite_prior]))
            if norm > eps:
                prior /= norm
        return np.clip(np.where(np.isfinite(prior), prior, 0.0), eps, 1.0e6)

    def _birth_existing_unit_response_counts(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return batched unit response columns for existing top-particle sources."""
        return self._birth_existing_unit_response_counts_batched(
            data,
            particle_indices=particle_indices,
        )

    def _birth_existing_unit_response_counts_scalar(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Scalar oracle for existing top-particle unit response columns."""
        columns: list[NDArray[np.float64]] = []
        for particle_idx in np.asarray(particle_indices, dtype=int).ravel():
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[int(particle_idx)].state
            if st.num_sources <= 0:
                continue
            active_mask = self._active_source_mask(st)
            if not np.any(active_mask):
                continue
            positions = st.positions[: st.num_sources][active_mask]
            counts = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=positions,
                strengths=np.ones(positions.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            for col_idx in range(counts.shape[1]):
                columns.append(np.asarray(counts[:, col_idx], dtype=float))
        if not columns:
            return np.zeros((data.z_k.size, 0), dtype=float)
        return np.column_stack(columns)

    def _birth_existing_unit_response_counts_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return existing-source unit responses using grouped batched kernels."""
        records: list[tuple[int, NDArray[np.float64]]] = []
        groups: dict[int, list[int]] = {}
        for particle_idx in np.asarray(particle_indices, dtype=int).ravel():
            if particle_idx < 0 or particle_idx >= len(self.continuous_particles):
                continue
            st = self.continuous_particles[int(particle_idx)].state
            if st.num_sources <= 0:
                continue
            active_mask = self._active_source_mask(st)
            if not np.any(active_mask):
                continue
            positions = np.asarray(
                st.positions[: st.num_sources][active_mask],
                dtype=float,
            )
            active_count = int(positions.shape[0])
            if active_count <= 0:
                continue
            record_idx = len(records)
            records.append((active_count, positions))
            groups.setdefault(active_count, []).append(record_idx)
        if not records:
            return np.zeros((data.z_k.size, 0), dtype=float)

        counts_by_record: dict[int, NDArray[np.float64]] = {}
        for active_count, record_indices in groups.items():
            stacked_positions = np.vstack(
                [records[record_idx][1] for record_idx in record_indices]
            )
            flat_counts = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=stacked_positions,
                strengths=np.ones(stacked_positions.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
            group_counts = np.asarray(flat_counts, dtype=float).reshape(
                int(data.z_k.size),
                int(len(record_indices)),
                int(active_count),
            )
            for local_idx, record_idx in enumerate(record_indices):
                counts_by_record[record_idx] = group_counts[:, local_idx, :]

        columns: list[NDArray[np.float64]] = []
        for record_idx in range(len(records)):
            counts = counts_by_record.get(record_idx)
            if counts is None:
                continue
            for col_idx in range(counts.shape[1]):
                columns.append(np.asarray(counts[:, col_idx], dtype=float))
        if not columns:
            return np.zeros((data.z_k.size, 0), dtype=float)
        return np.column_stack(columns)

    def _birth_residual_support_evidence(
        self,
        residual_mix: NDArray[np.float64],
        data: MeasurementData,
    ) -> tuple[
        NDArray[np.bool_],
        NDArray[np.float64],
        float,
        NDArray[np.float64],
    ]:
        """
        Return exact per-row residual support under the structural PF likelihood.

        Each column saturates one positive-residual row while retaining all
        covariance-coupled rows. Trials are evaluated in NumPy batches of at
        most 64 columns to bound direct-spectrum memory without changing the
        likelihood or statistical interpretation.
        """
        residual = np.maximum(np.asarray(residual_mix, dtype=float).reshape(-1), 0.0)
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if residual.size == 0 or residual.size != z_arr.size:
            return (
                np.zeros(z_arr.size, dtype=bool),
                np.maximum(z_arr, 1.0e-12),
                0.0,
                np.zeros(z_arr.size, dtype=float),
            )
        lambda_null = np.maximum(z_arr - residual, 1.0e-12)
        null_ll = self._structural_count_log_likelihood_np(
            data,
            lambda_null,
        )
        positive_rows = np.flatnonzero(residual > 0.0)
        delta_ll = np.zeros(z_arr.size, dtype=float)
        batch_columns = 64
        for batch_start in range(0, int(positive_rows.size), batch_columns):
            row_indices = positive_rows[
                batch_start : batch_start + batch_columns
            ]
            trials = np.repeat(
                lambda_null[:, None],
                int(row_indices.size),
                axis=1,
            )
            trials[
                row_indices,
                np.arange(int(row_indices.size), dtype=np.int64),
            ] = z_arr[row_indices]
            trial_ll = self._structural_count_log_likelihood_matrix_np(
                data,
                trials,
            )
            delta_ll[row_indices] = np.asarray(trial_ll, dtype=float) - float(
                null_ll
            )
        min_sigma = max(float(self.config.birth_residual_support_sigma), 0.0)
        min_delta_ll = 0.5 * min_sigma * min_sigma
        support_mask = (
            (residual > 0.0)
            & np.isfinite(delta_ll)
            & (delta_ll > 0.0)
            & (delta_ll >= min_delta_ll)
        )
        return support_mask, lambda_null, float(null_ll), delta_ll

    def _birth_residual_gate_allows(
        self,
        residual_mix: NDArray[np.float64],
        data: MeasurementData,
    ) -> bool:
        """
        Return True when positive residuals statistically justify a birth move.

        The null mean is reconstructed as ``z - positive_residual``. The
        alternative saturates only rows supported by exact one-row likelihood
        trials. Both support and the final likelihood-ratio gain use the same
        configured structural likelihood, including direct spectrum and
        same-station covariance.
        """
        residual = np.maximum(np.asarray(residual_mix, dtype=float).reshape(-1), 0.0)
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if residual.size == 0 or residual.size != z_arr.size:
            return False
        (
            support_mask,
            lambda_null,
            null_ll,
            _row_delta_ll,
        ) = self._birth_residual_support_evidence(
            residual,
            data,
        )
        support_count = int(np.count_nonzero(support_mask))
        distinct_supported = self._distinct_supported_view_count(
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
            support_mask,
        )
        distinct_stations = self._distinct_supported_station_count(
            data.detector_positions,
            support_mask,
        )
        lambda_saturated = lambda_null.copy()
        lambda_saturated[support_mask] = z_arr[support_mask]
        if support_count > 0:
            saturated_ll = self._structural_count_log_likelihood_np(
                data,
                lambda_saturated,
            )
            delta_ll = max(float(saturated_ll - null_ll), 0.0)
        else:
            delta_ll = 0.0
        likelihood_ratio_stat = 2.0 * delta_ll
        dof = max(support_count, 1)
        p_value = (
            float(chi2.sf(likelihood_ratio_stat, dof)) if support_count > 0 else 1.0
        )
        # Keep the historical diagnostic field as the asymptotic 2*DeltaLL
        # statistic while exposing the actual configured-likelihood gain.
        self.last_birth_residual_chi2 = likelihood_ratio_stat
        self.last_birth_residual_delta_ll = delta_ll
        self.last_birth_residual_p_value = p_value
        self.last_birth_residual_support = support_count
        self.last_birth_residual_distinct_poses = distinct_supported
        self.last_birth_residual_distinct_stations = distinct_stations
        min_support = max(1, int(self.config.birth_residual_min_support))
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
        p_threshold = float(self.config.birth_residual_gate_p_value)
        if p_threshold <= 0.0:
            passed = (
                support_count >= min_support
                and distinct_supported >= min_distinct
                and distinct_stations >= min_stations
                and delta_ll > 0.0
            )
        else:
            p_threshold = min(max(p_threshold, 0.0), 1.0)
            passed = (
                support_count >= min_support
                and distinct_supported >= min_distinct
                and distinct_stations >= min_stations
                and p_value <= p_threshold
            )
        self.last_birth_residual_gate_passed = bool(passed)
        return bool(passed)

    def _distinct_supported_view_count(
        self,
        detector_positions: NDArray[np.float64] | None,
        fe_indices: NDArray[np.int64] | None,
        pb_indices: NDArray[np.int64] | None,
        support_mask: NDArray[np.bool_],
    ) -> int:
        """Return the number of distinct pose/shield views with residual support."""
        if detector_positions is None:
            return int(np.count_nonzero(support_mask))
        positions = np.asarray(detector_positions, dtype=float)
        mask = np.asarray(support_mask, dtype=bool).ravel()
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] != mask.size
        ):
            return int(np.count_nonzero(mask))
        if not np.any(mask):
            return 0
        rounded = np.round(positions[mask], decimals=3)
        if fe_indices is None or pb_indices is None:
            return int(np.unique(rounded, axis=0).shape[0])
        fe = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe.size != mask.size or pb.size != mask.size:
            return int(np.unique(rounded, axis=0).shape[0])
        views = np.column_stack([rounded, fe[mask], pb[mask]])
        return int(np.unique(views, axis=0).shape[0])

    def _distinct_supported_station_count(
        self,
        detector_positions: NDArray[np.float64] | None,
        support_mask: NDArray[np.bool_],
    ) -> int:
        """Return the number of distinct robot stations with residual support."""
        if detector_positions is None:
            return int(np.count_nonzero(support_mask))
        positions = np.asarray(detector_positions, dtype=float)
        mask = np.asarray(support_mask, dtype=bool).ravel()
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] != mask.size
        ):
            return int(np.count_nonzero(mask))
        if not np.any(mask):
            return 0
        rounded_xy = np.round(positions[mask, :2], decimals=3)
        return int(np.unique(rounded_xy, axis=0).shape[0])

    def _source_prune_support_ready(self, data: MeasurementData) -> bool:
        """Return True when a measurement block can justify source removal."""
        if data.z_k.size == 0:
            return False
        full_support = np.ones(data.z_k.size, dtype=bool)
        distinct_views = self._distinct_supported_view_count(
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
            full_support,
        )
        distinct_stations = self._distinct_supported_station_count(
            data.detector_positions,
            full_support,
        )
        return distinct_views >= max(
            1, int(self.config.source_prune_min_distinct_views)
        ) and distinct_stations >= max(
            1, int(self.config.source_prune_min_distinct_stations)
        )

    def _source_prune_delta_threshold(self) -> float:
        """Return the leave-one-out ΔLL threshold used for source removal."""
        return float(self.config.source_prune_delta_ll_threshold)

    def _bic_model_penalty(self, measurement_count: int, parameter_count: int) -> float:
        """Return half-BIC penalty gain for removing model parameters."""
        params = max(0, int(parameter_count))
        if params <= 0:
            return 0.0
        count = max(2, int(measurement_count))
        return 0.5 * float(params) * float(np.log(count))

    def _remove_source_trial(self, st: IsotopeState, source_idx: int) -> IsotopeState:
        """Return a copy of a state with one source removed."""
        trial = st.copy()
        self._ensure_source_metadata(trial)
        if trial.num_sources <= 0:
            return trial
        keep = np.ones(int(trial.num_sources), dtype=bool)
        keep[int(source_idx)] = False
        self._apply_source_keep_mask(trial, keep)
        return trial

    def _source_prune_allowed_mask(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        lambda_m: NDArray[np.float64] | None = None,
        lambda_total: NDArray[np.float64] | None = None,
        delta_ll: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Return sources whose removal is supported across multiple stations.

        A source is removable only when leaving it out has low likelihood loss
        in enough distinct robot stations. This prevents a single shield view or
        one low-count station from deleting a weak but physically plausible
        source hypothesis.
        """
        if st.num_sources <= 0 or data.z_k.size == 0:
            return np.zeros(max(0, int(st.num_sources)), dtype=bool)
        self._ensure_source_metadata(st)
        if not self._source_prune_support_ready(data):
            return np.zeros(int(st.num_sources), dtype=bool)
        if lambda_m is None or lambda_total is None:
            lambda_m, lambda_total = self._lambda_components(st, data)
        if lambda_m.shape != (int(data.z_k.size), int(st.num_sources)):
            return np.zeros(int(st.num_sources), dtype=bool)
        allowed_loss = (
            self._bic_model_penalty(
                int(data.z_k.size),
                int(self.config.source_prune_bic_penalty_params),
            )
            + self._source_prune_delta_threshold()
        )
        if delta_ll is None or delta_ll.shape != (int(st.num_sources),):
            delta_ll = self._structural_delta_log_likelihood_remove(
                data,
                lambda_total,
                lambda_m,
            )
        station_labels = self._support_station_labels(
            data.detector_positions,
            int(data.z_k.size),
        )
        fail_counts = np.zeros(int(st.num_sources), dtype=int)
        for label in np.unique(station_labels):
            rows = station_labels == int(label)
            if not np.any(rows):
                continue
            station_allowed_loss = (
                self._bic_model_penalty(
                    int(np.count_nonzero(rows)),
                    int(self.config.source_prune_bic_penalty_params),
                )
                + self._source_prune_delta_threshold()
            )
            station_data = self._measurement_rows(data, rows)
            station_delta = self._structural_delta_log_likelihood_remove(
                station_data,
                lambda_total[rows],
                lambda_m[rows, :],
            )
            fail_counts += (station_delta <= station_allowed_loss).astype(int)
        min_stations = max(1, int(self.config.source_prune_min_distinct_stations))
        global_failed = np.asarray(delta_ll, dtype=float) <= allowed_loss
        return (fail_counts >= min_stations) & global_failed

    def _source_prune_allowed_mask_group(
        self,
        data: MeasurementData,
        lambda_m: NDArray[np.float64],
        lambda_total: NDArray[np.float64],
        *,
        delta_ll: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Return batched evidence-based removal masks for an equal-cardinality group.

        Each station iteration evaluates every particle and source slot in one
        likelihood batch. The station partition itself remains explicit because
        stations may contain different runtime likelihood routes and view counts.
        """
        components = np.asarray(lambda_m, dtype=float)
        total = np.asarray(lambda_total, dtype=float)
        if components.ndim != 3 or total.ndim != 2:
            return np.zeros((0, 0), dtype=bool)
        particle_count = int(total.shape[1])
        source_count = int(components.shape[2])
        expected_shape = (int(data.z_k.size), particle_count, source_count)
        if total.shape[0] != int(data.z_k.size) or components.shape != expected_shape:
            return np.zeros((particle_count, source_count), dtype=bool)
        if (
            particle_count <= 0
            or source_count <= 0
            or not self._source_prune_support_ready(data)
        ):
            return np.zeros((particle_count, source_count), dtype=bool)
        if delta_ll is None or np.asarray(delta_ll).shape != (
            particle_count,
            source_count,
        ):
            delta_ll = self._delta_log_likelihood_remove_group(
                data,
                total,
                components,
            )
        allowed_loss = (
            self._bic_model_penalty(
                int(data.z_k.size),
                int(self.config.source_prune_bic_penalty_params),
            )
            + self._source_prune_delta_threshold()
        )
        station_labels = self._support_station_labels(
            data.detector_positions,
            int(data.z_k.size),
        )
        fail_counts = np.zeros((particle_count, source_count), dtype=int)
        for label in np.unique(station_labels):
            rows = station_labels == int(label)
            if not np.any(rows):
                continue
            station_allowed_loss = (
                self._bic_model_penalty(
                    int(np.count_nonzero(rows)),
                    int(self.config.source_prune_bic_penalty_params),
                )
                + self._source_prune_delta_threshold()
            )
            station_data = self._measurement_rows(data, rows)
            station_delta = self._delta_log_likelihood_remove_group(
                station_data,
                total[rows, :],
                components[rows, :, :],
            )
            fail_counts += (station_delta <= station_allowed_loss).astype(int)
        min_stations = max(1, int(self.config.source_prune_min_distinct_stations))
        global_failed = np.asarray(delta_ll, dtype=float) <= allowed_loss
        return (fail_counts >= min_stations) & global_failed

    def _birth_candidate_support_mask(
        self,
        *,
        data: MeasurementData,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        min_support: int | None = None,
        min_distinct_poses: int | None = None,
        min_distinct_stations: int | None = None,
    ) -> NDArray[np.bool_]:
        """
        Return candidates whose residual explanation is coherent across views.

        A birth location is accepted only when its expected count pattern overlaps
        statistically significant positive residuals in enough measurements and
        enough distinct pose/shield views. This uses the rotating shield as an
        independent measurement primitive instead of requiring robot motion
        before a new-source hypothesis can be born.
        """
        counts = np.asarray(candidate_counts, dtype=float)
        if counts.ndim != 2 or counts.size == 0:
            return np.zeros(0, dtype=bool)
        residual = self._measurement_vector(
            residual_mix,
            counts.shape[0],
            "residual_mix",
            min_value=0.0,
            allow_scalar=False,
        )
        if int(data.z_k.size) != int(counts.shape[0]):
            raise ValueError(
                "candidate_counts must contain one row per structural measurement."
            )
        residual_support, _lambda_null, _null_ll, _row_delta_ll = (
            self._birth_residual_support_evidence(
                residual,
                data,
            )
        )
        if not np.any(residual_support):
            return np.zeros(counts.shape[1], dtype=bool)
        overlap = np.maximum(counts, 0.0) * residual[:, None]
        max_overlap = np.max(overlap, axis=0)
        fraction = float(self.config.birth_candidate_support_fraction)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        threshold = max_overlap[None, :] * fraction
        support = (overlap >= threshold) & (max_overlap[None, :] > 0.0)
        support &= residual_support[:, None]
        support_counts = np.sum(support, axis=0)
        support_floor = (
            self.config.birth_residual_min_support
            if min_support is None
            else min_support
        )
        distinct_floor = (
            self.config.birth_min_distinct_poses
            if min_distinct_poses is None
            else min_distinct_poses
        )
        station_floor = (
            self.config.birth_min_distinct_stations
            if min_distinct_stations is None
            else min_distinct_stations
        )
        support_floor = max(1, int(support_floor))
        distinct_floor = max(1, int(distinct_floor))
        station_floor = max(1, int(station_floor))
        keep = support_counts >= support_floor
        if distinct_floor <= 1:
            view_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            view_labels = self._support_view_labels(
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                support.shape[0],
            )
            distinct_counts = self._distinct_label_counts_for_support_matrix(
                support,
                view_labels,
            )
            view_keep = distinct_counts >= distinct_floor
        if station_floor <= 1:
            station_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            station_labels = self._support_station_labels(
                data.detector_positions,
                support.shape[0],
            )
            station_counts = self._distinct_label_counts_for_support_matrix(
                support,
                station_labels,
            )
            station_keep = station_counts >= station_floor
        keep &= view_keep
        keep &= station_keep
        return keep.astype(bool)

    @staticmethod
    def _distinct_label_counts_for_support_matrix(
        support: NDArray[np.bool_],
        labels: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Count distinct measurement labels supporting each candidate column."""
        support_arr = np.asarray(support, dtype=bool)
        if support_arr.ndim != 2 or support_arr.size == 0:
            return np.zeros(
                support_arr.shape[1] if support_arr.ndim == 2 else 0, dtype=int
            )
        label_arr = np.asarray(labels, dtype=int).reshape(-1)
        if label_arr.size != support_arr.shape[0]:
            return np.sum(support_arr, axis=0).astype(int)
        counts = np.zeros(support_arr.shape[1], dtype=int)
        for label in np.unique(label_arr):
            rows = label_arr == int(label)
            counts += np.any(support_arr[rows, :], axis=0).astype(int)
        return counts

    @staticmethod
    def _support_view_labels(
        detector_positions: NDArray[np.float64] | None,
        fe_indices: NDArray[np.int64] | None,
        pb_indices: NDArray[np.int64] | None,
        measurement_count: int,
    ) -> NDArray[np.int64]:
        """Return compact labels for distinct detector pose and shield views."""
        count = max(0, int(measurement_count))
        if detector_positions is None:
            return np.arange(count, dtype=int)
        positions = np.asarray(detector_positions, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] != count
        ):
            return np.arange(count, dtype=int)
        rounded = np.round(positions, decimals=3)
        if fe_indices is None or pb_indices is None:
            _, labels = np.unique(rounded, axis=0, return_inverse=True)
            return labels.astype(int, copy=False)
        fe = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe.size != count or pb.size != count:
            _, labels = np.unique(rounded, axis=0, return_inverse=True)
            return labels.astype(int, copy=False)
        views = np.column_stack([rounded, fe, pb])
        _, labels = np.unique(views, axis=0, return_inverse=True)
        return labels.astype(int, copy=False)

    @staticmethod
    def _support_station_labels(
        detector_positions: NDArray[np.float64] | None,
        measurement_count: int,
    ) -> NDArray[np.int64]:
        """Return compact labels for distinct detector station positions."""
        count = max(0, int(measurement_count))
        if detector_positions is None:
            return np.arange(count, dtype=int)
        positions = np.asarray(detector_positions, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] != count
        ):
            return np.arange(count, dtype=int)
        rounded_xy = np.round(positions[:, :2], decimals=3)
        _, labels = np.unique(rounded_xy, axis=0, return_inverse=True)
        return labels.astype(int, copy=False)

    def _exclude_birth_candidates_near_detectors(
        self,
        candidates: NDArray[np.float64],
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Remove birth candidates that are too close to measured detector poses."""
        min_sep = float(self.config.birth_detector_min_sep_m)
        if min_sep <= 0.0 or candidates.size == 0 or data.detector_positions.size == 0:
            return candidates
        diff = candidates[:, None, :] - data.detector_positions[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        keep = np.all(distances >= min_sep, axis=1)
        return candidates[keep]

    def _project_and_deduplicate_birth_candidates(
        self,
        candidates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project birth candidates to the source prior and retain first uniques."""
        candidate_arr = np.asarray(candidates, dtype=float).reshape(-1, 3)
        if candidate_arr.size == 0:
            return np.zeros((0, 3), dtype=float)
        projected = self._project_positions_to_source_prior(candidate_arr)
        _, first_indices = np.unique(
            projected,
            axis=0,
            return_index=True,
        )
        return projected[np.sort(first_indices)].copy()

    def _roughening_sigma_pos(self, num_particles: int) -> NDArray[np.float64]:
        """
        Compute per-axis roughening sigma based on the current particle count.

        Uses sigma = k * range * N^(-1/d) with clamping.
        """
        count = max(1, int(num_particles))
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        span = np.maximum(hi - lo, 0.0)
        scale = float(self.config.roughening_k) * (count ** (-1.0 / 3.0))
        sigma = scale * span
        min_sigma = float(self.config.min_sigma_pos)
        max_sigma = float(self.config.max_sigma_pos)
        if max_sigma < min_sigma:
            min_sigma, max_sigma = max_sigma, min_sigma
        return np.clip(sigma, min_sigma, max_sigma)

    def _roughening_multiplier(self) -> float:
        """Return the roughening multiplier based on resamples in this observation."""
        decay = float(self.config.roughening_decay)
        min_mult = float(self.config.roughening_min_mult)
        if decay <= 0.0:
            decay = 1.0
        if min_mult < 0.0:
            min_mult = 0.0
        count = max(0, int(self._resample_count_in_observation))
        mult = decay**count
        return max(min_mult, mult)

    def regularize_continuous(
        self,
        sigma_pos: float | NDArray[np.float64] = 0.05,
        strength_log_sigma: float | None = None,
        p_birth: float = 0.05,
        p_kill: float = 0.1,
        intensity_threshold: float = 0.05,
    ) -> None:
        """
        Apply position roughening and log-space strength jitter (Sec. 3.3.4).

        Source-cardinality moves are handled in apply_structural_moves().
        """
        if self._structural_kernel_is_exact():
            for particle in self.continuous_particles:
                self._canonicalize_structural_rj_state(particle.state)
                if not np.all(
                    self._strength_prior.in_support(particle.state.strengths)
                ):
                    raise ValueError(
                        "rj_mh state strength escaped the configured prior "
                        "support."
                    )
            return
        sigma_pos_arr = np.asarray(sigma_pos, dtype=float)
        if sigma_pos_arr.size not in (1, 3):
            raise ValueError("sigma_pos must be a scalar or a 3-element vector.")
        log_sigma = (
            float(self.config.strength_log_sigma)
            if strength_log_sigma is None
            else float(strength_log_sigma)
        )
        log_sigma = max(log_sigma, 0.0)
        for p in self.continuous_particles:
            st = p.state
            self._ensure_source_metadata(st)
            st.background = self._background_level()
            if st.positions.size:
                st.positions = st.positions + np.random.normal(
                    scale=sigma_pos_arr,
                    size=st.positions.shape,
                )
                st.positions = self._project_positions_to_source_prior(st.positions)
                if log_sigma > 0.0:
                    logq = np.log(st.strengths + 1e-12)
                    logq = logq + np.random.normal(
                        scale=log_sigma, size=st.strengths.shape
                    )
                    st.strengths = np.exp(logq)
                st.strengths = np.maximum(st.strengths, 0.0)
                st.num_sources = st.positions.shape[0]

    def _replace_particle_state_from_trial(
        self,
        target: IsotopeState,
        trial: IsotopeState,
    ) -> None:
        """Replace a particle state from an accepted structural proposal."""
        self._ensure_source_metadata(trial)
        target.positions = np.asarray(trial.positions, dtype=float).copy()
        target.strengths = np.asarray(trial.strengths, dtype=float).copy()
        target.background = float(trial.background)
        target.ages = np.asarray(trial.ages, dtype=int).copy()
        target.support_scores = np.asarray(trial.support_scores, dtype=float).copy()
        target.tentative_sources = np.asarray(
            trial.tentative_sources,
            dtype=bool,
        ).copy()
        target.verification_fail_streaks = np.asarray(
            trial.verification_fail_streaks,
            dtype=int,
        ).copy()
        target.num_sources = int(target.positions.shape[0])

    def _trial_log_likelihood(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> float:
        """Return structural PF count likelihood for one fixed state."""
        _, lambda_total = self._lambda_components(st, data)
        return self._structural_count_log_likelihood_np(
            data,
            lambda_total,
        )

    def _trial_log_likelihood_from_lambda(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
    ) -> float:
        """Return structural PF count likelihood for precomputed counts."""
        return self._structural_count_log_likelihood_np(
            data,
            np.asarray(lambda_total, dtype=float),
        )

    def _unit_response_counts_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength response columns for every source in a state."""
        if st.num_sources <= 0 or data.z_k.size == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        self._ensure_source_metadata(st)
        positions = np.asarray(st.positions[: st.num_sources], dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0:
            return np.zeros((int(data.z_k.size), 0), dtype=float)
        counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=positions,
            strengths=np.ones(positions.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        return np.asarray(counts, dtype=float)

    def _orthogonalized_residual_candidate_indices(
        self,
        ranked_candidate_indices: NDArray[np.int64],
        *,
        candidate_counts: NDArray[np.float64],
        existing_response_counts: NDArray[np.float64],
        observation_variances: NDArray[np.float64] | None,
        max_corr: float,
    ) -> NDArray[np.int64]:
        """
        Return residual-birth candidates after response-column orthogonalization.

        Matching-pursuit birth evaluates only a tiny pre-ranked candidate set,
        so the greedy Gram-Schmidt-style loop here is bounded by the configured
        top-k candidate count. The heavy response and likelihood evaluation
        remain batched; this helper only prevents multiple nearly collinear
        response columns from entering the same birth proposal set.
        """
        ranked = np.asarray(ranked_candidate_indices, dtype=int).ravel()
        if ranked.size <= 1:
            return ranked.astype(np.int64, copy=False)
        counts = np.asarray(candidate_counts, dtype=float)
        if counts.ndim != 2:
            return ranked.astype(np.int64, copy=False)
        valid_ranked = ranked[(ranked >= 0) & (ranked < counts.shape[1])]
        if valid_ranked.size <= 1:
            return valid_ranked.astype(np.int64, copy=False)
        corr_limit = float(np.clip(float(max_corr), 0.0, 1.0))
        if corr_limit >= 1.0:
            return valid_ranked.astype(np.int64, copy=False)
        if observation_variances is None:
            scale = np.ones(counts.shape[0], dtype=float)
        else:
            variances = np.asarray(observation_variances, dtype=float).reshape(-1)
            if variances.size != counts.shape[0]:
                scale = np.ones(counts.shape[0], dtype=float)
            else:
                scale = 1.0 / np.sqrt(np.maximum(variances, 1.0e-12))

        def _normalized_columns(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return variance-whitened unit-norm response columns."""
            arr = np.asarray(matrix, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != counts.shape[0] or arr.shape[1] == 0:
                return np.zeros((counts.shape[0], 0), dtype=float)
            whitened = arr * scale[:, None]
            norms = np.linalg.norm(whitened, axis=0)
            keep = norms > 1.0e-12
            if not np.any(keep):
                return np.zeros((counts.shape[0], 0), dtype=float)
            return whitened[:, keep] / norms[keep][None, :]

        basis = _normalized_columns(existing_response_counts)
        candidate_basis = _normalized_columns(counts[:, valid_ranked])
        if candidate_basis.shape[1] != valid_ranked.size:
            return valid_ranked.astype(np.int64, copy=False)
        selected: list[int] = []
        selected_columns: list[NDArray[np.float64]] = []
        for local_idx, candidate_idx in enumerate(valid_ranked.tolist()):
            column = candidate_basis[:, int(local_idx)]
            if basis.size:
                corr_existing = float(np.max(np.abs(basis.T @ column)))
                if corr_existing > corr_limit:
                    continue
            if selected_columns:
                selected_matrix = np.column_stack(selected_columns)
                corr_selected = float(np.max(np.abs(selected_matrix.T @ column)))
                if corr_selected > corr_limit:
                    continue
            selected.append(int(candidate_idx))
            selected_columns.append(column)
        if not selected:
            return valid_ranked[:1].astype(np.int64, copy=False)
        return np.asarray(selected, dtype=np.int64)

    def _best_cached_matching_pursuit_birth_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        candidates: NDArray[np.float64],
        ranked_candidate_indices: NDArray[np.int64],
        q_hat: NDArray[np.float64],
        unit_counts_existing: NDArray[np.float64],
        unit_counts_all: NDArray[np.float64],
        source_strengths: NDArray[np.float64],
        base_ll: float,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best fixed-strength matching-pursuit trial in one batch."""
        ranked = np.asarray(ranked_candidate_indices, dtype=int).ravel()
        if ranked.size == 0:
            return None, -np.inf
        candidate_positions = np.asarray(candidates, dtype=float)
        candidate_counts = np.asarray(unit_counts_all, dtype=float)
        existing_counts = np.asarray(unit_counts_existing, dtype=float)
        if candidate_positions.ndim != 2 or candidate_positions.shape[1] != 3:
            return None, -np.inf
        if candidate_counts.shape != (int(data.z_k.size), candidate_positions.shape[0]):
            return None, -np.inf
        if existing_counts.ndim != 2 or existing_counts.shape[0] != int(data.z_k.size):
            return None, -np.inf
        ranked = ranked[(ranked >= 0) & (ranked < candidate_positions.shape[0])]
        if ranked.size == 0:
            return None, -np.inf
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        trial_q = np.clip(
            np.asarray(q_hat, dtype=float).reshape(-1)[ranked], q_min, q_max
        )
        existing_count = int(existing_counts.shape[1])
        background_counts = np.asarray(data.live_times, dtype=float)[:, None] * float(
            st.background
        )
        source_prior = np.asarray(source_strengths, dtype=float).reshape(-1)
        if existing_count > 0:
            if source_prior.size != existing_count:
                raise ValueError("source_strengths must match existing source count.")
            existing_lambda = existing_counts @ source_prior
        else:
            existing_lambda = np.zeros(int(data.z_k.size), dtype=float)
        lambda_total = (
            background_counts
            + existing_lambda[:, None]
            + candidate_counts[:, ranked] * trial_q[None, :]
        )
        ll_after = self._structural_count_log_likelihood_matrix_np(
            data,
            lambda_total,
        )
        deltas = np.asarray(ll_after, dtype=float) - float(base_ll)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            return None, -np.inf
        best_local = int(np.flatnonzero(finite)[np.argmax(deltas[finite])])
        best_delta = float(deltas[best_local])
        best_candidate_idx = int(ranked[best_local])
        pos_new = self._project_positions_to_source_prior(
            candidate_positions[best_candidate_idx].reshape(1, 3)
        )[0]
        trial = st.copy()
        self._ensure_source_metadata(trial)
        trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
        trial.strengths = np.append(
            trial.strengths[: trial.num_sources],
            float(trial_q[best_local]),
        )
        trial.ages = np.append(trial.ages[: trial.num_sources], 0)
        trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
        trial.tentative_sources = np.append(
            trial.tentative_sources[: trial.num_sources],
            True,
        )
        trial.verification_fail_streaks = np.append(
            trial.verification_fail_streaks[: trial.num_sources],
            0,
        )
        trial.num_sources = int(trial.positions.shape[0])
        return trial, best_delta

    def _structural_acceptance_threshold(
        self,
        *,
        base_threshold: float,
        complexity_penalty: float,
    ) -> float:
        """Return the likelihood-gain threshold for one structural parameter jump."""
        return float(base_threshold) + max(float(complexity_penalty), 0.0)

    def _birth_complexity_penalty(
        self,
        *,
        measurement_count: int = 0,
    ) -> float:
        """Return the configured and BIC complexity penalties for one birth."""
        penalty = max(float(self.config.birth_complexity_penalty), 0.0)
        return penalty + self._bic_model_penalty(
            int(measurement_count),
            int(self.config.birth_bic_penalty_params),
        )

    def _candidate_initial_strengths(
        self,
        *,
        candidate_count: int,
        candidate_kernel_sums: NDArray[np.float64] | None,
        residual_sum: float,
    ) -> NDArray[np.float64]:
        """Return residual-scaled initial strengths for structural candidates."""
        count = max(0, int(candidate_count))
        if count <= 0:
            return np.zeros(0, dtype=float)
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        kernel_sums = (
            np.ones(count, dtype=float)
            if candidate_kernel_sums is None
            else np.asarray(candidate_kernel_sums, dtype=float).reshape(-1)[:count]
        )
        if kernel_sums.size != count:
            raise ValueError("candidate_kernel_sums must match candidate_count.")
        denom = np.maximum(kernel_sums, 1.0e-12)
        q = float(self.config.birth_alpha) * max(float(residual_sum), 0.0) / denom
        q = np.clip(q, q_min, q_max)
        return np.where(np.isfinite(q), q, q_min)

    def _best_residual_guided_split_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
        candidate_positions: NDArray[np.float64] | None,
        candidate_strengths: NDArray[np.float64] | None,
        *,
        candidate_unit_counts: NDArray[np.float64] | None = None,
        cached_existing_unit_counts: NDArray[np.float64] | None = None,
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best residual-guided split trial and its likelihood gain.

        The proposal moves a residual-derived amount of strength from one
        existing source to a new candidate, then compares the unchanged base
        state and fixed-strength proposal with the configured PF likelihood.
        """
        if not bool(self.config.split_residual_guided):
            return None, -np.inf
        if data.z_k.size == 0 or st.num_sources <= 0:
            return None, -np.inf
        if (
            self.config.max_sources is not None
            and st.num_sources >= self.config.max_sources
        ):
            return None, -np.inf
        candidates = (
            np.asarray(candidate_positions, dtype=float)
            if candidate_positions is not None
            else np.zeros((0, 3))
        )
        if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
            return None, -np.inf
        self._ensure_source_metadata(st)
        eligible = np.flatnonzero(
            (st.strengths[: st.num_sources] >= float(self.config.split_strength_min))
            & (st.ages[: st.num_sources] > int(self.config.min_age_to_split))
        )
        if eligible.size == 0:
            return None, -np.inf
        cached_existing_counts = (
            None
            if cached_existing_unit_counts is None
            else np.asarray(cached_existing_unit_counts, dtype=float)
        )
        expected_existing_shape = (int(data.z_k.size), int(st.num_sources))
        if (
            cached_existing_counts is None
            or cached_existing_counts.shape != expected_existing_shape
        ):
            existing_unit_counts = self._unit_response_counts_for_state(st, data)
        else:
            existing_unit_counts = cached_existing_counts
        if candidate_unit_counts is None:
            candidate_counts_arr = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=candidates,
                strengths=np.ones(candidates.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
        else:
            candidate_counts_arr = np.asarray(candidate_unit_counts, dtype=float)
        expected_shape = (int(data.z_k.size), int(candidates.shape[0]))
        if candidate_counts_arr.shape != expected_shape:
            raise ValueError("candidate_unit_counts must have shape K x C.")
        base_lambda = np.asarray(data.live_times, dtype=float) * float(
            st.background
        ) + existing_unit_counts @ np.asarray(
            st.strengths[: st.num_sources], dtype=float
        )
        base_ll = self._trial_log_likelihood_from_lambda(data, base_lambda)
        if not np.isfinite(base_ll):
            return None, -np.inf
        max_candidates = max(1, int(self.config.split_residual_candidate_count))
        candidate_count = min(max_candidates, candidates.shape[0])
        cand_strengths = self._candidate_initial_strengths(
            candidate_count=candidates.shape[0],
            candidate_kernel_sums=None,
            residual_sum=float(np.sum(np.maximum(data.z_k, 0.0))),
        )
        if candidate_strengths is not None:
            candidate_strengths_arr = np.asarray(
                candidate_strengths, dtype=float
            ).reshape(-1)
            if candidate_strengths_arr.size:
                copy_count = min(cand_strengths.size, candidate_strengths_arr.size)
                cand_strengths[:copy_count] = candidate_strengths_arr[:copy_count]
        ranked_sources = eligible[np.argsort(st.strengths[eligible])[::-1]]
        min_sep = max(float(self.config.birth_min_sep_m), 0.0)
        split_pairs = [
            (int(source_idx), int(cand_idx))
            for cand_idx in range(candidate_count)
            for source_idx in ranked_sources
        ][:max_candidates]
        return self._best_cached_residual_guided_split_trial_batched(
            st,
            data,
            candidates=candidates,
            candidate_strengths=cand_strengths,
            split_pairs=split_pairs,
            existing_unit_counts=existing_unit_counts,
            candidate_unit_counts=candidate_counts_arr,
            base_ll=base_ll,
            min_sep=min_sep,
        )

    def _best_cached_residual_guided_split_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        candidates: NDArray[np.float64],
        candidate_strengths: NDArray[np.float64],
        split_pairs: list[tuple[int, int]],
        existing_unit_counts: NDArray[np.float64],
        candidate_unit_counts: NDArray[np.float64],
        base_ll: float,
        min_sep: float,
        allow_parallel: bool = True,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best fixed-strength residual split trial in one batch."""
        if not split_pairs or data.z_k.size == 0:
            return None, -np.inf
        del allow_parallel
        self._ensure_source_metadata(st)
        candidates_arr = np.asarray(candidates, dtype=float)
        cand_strengths = np.asarray(candidate_strengths, dtype=float).reshape(-1)
        existing_counts = np.asarray(existing_unit_counts, dtype=float)
        candidate_counts = np.asarray(candidate_unit_counts, dtype=float)
        if existing_counts.shape != (int(data.z_k.size), int(st.num_sources)):
            return None, -np.inf
        if candidate_counts.shape[0] != int(data.z_k.size):
            return None, -np.inf
        pair_arr = np.asarray(split_pairs, dtype=np.int64).reshape(-1, 2)
        source_indices = pair_arr[:, 0]
        candidate_indices = pair_arr[:, 1]
        valid = (
            (source_indices >= 0)
            & (source_indices < int(st.num_sources))
            & (candidate_indices >= 0)
            & (candidate_indices < int(candidates_arr.shape[0]))
        )
        safe_candidate_indices = np.clip(
            candidate_indices,
            0,
            max(int(candidates_arr.shape[0]) - 1, 0),
        )
        projected = self._project_positions_to_source_prior(
            candidates_arr[safe_candidate_indices]
        )
        distances = np.linalg.norm(
            projected[:, None, :]
            - np.asarray(st.positions[: st.num_sources], dtype=float)[None, :, :],
            axis=2,
        )
        safe_source_indices = np.clip(
            source_indices,
            0,
            max(int(st.num_sources) - 1, 0),
        )
        parent_distances = distances[
            np.arange(distances.shape[0]),
            safe_source_indices,
        ]
        distances[
            np.arange(distances.shape[0]),
            safe_source_indices,
        ] = np.inf
        valid &= np.all(distances >= float(min_sep), axis=1)
        valid &= parent_distances >= 0.5 * float(min_sep)
        if not np.any(valid):
            return None, -np.inf
        source_indices = source_indices[valid]
        candidate_indices = candidate_indices[valid]
        projected = projected[valid]
        base_strengths = np.asarray(st.strengths[: st.num_sources], dtype=float)
        min_strength = max(float(self.config.min_strength), 0.0)
        available_strength = np.maximum(
            base_strengths[source_indices] - min_strength,
            0.0,
        )
        strength_valid = available_strength >= min_strength
        if not np.any(strength_valid):
            return None, -np.inf
        source_indices = source_indices[strength_valid]
        candidate_indices = candidate_indices[strength_valid]
        projected = projected[strength_valid]
        available_strength = available_strength[strength_valid]
        q_new = np.minimum(
            np.maximum(
                cand_strengths[candidate_indices],
                min_strength,
            ),
            available_strength,
        )
        keep_strength = base_strengths[source_indices] - q_new
        trial_count = int(source_indices.size)
        trial_strengths = np.repeat(
            base_strengths.reshape(1, -1),
            trial_count,
            axis=0,
        )
        trial_strengths[np.arange(trial_count), source_indices] = keep_strength
        existing_lambda = existing_counts @ trial_strengths.T
        candidate_lambda = candidate_counts[:, candidate_indices] * q_new[None, :]
        lambda_total = (
            np.asarray(data.live_times, dtype=float)[:, None] * float(st.background)
            + existing_lambda
            + candidate_lambda
        )
        ll_after = self._structural_count_log_likelihood_matrix_np(
            data,
            lambda_total,
        )
        deltas = np.asarray(ll_after, dtype=float) - float(base_ll)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            return None, -np.inf
        best_local = int(np.flatnonzero(finite)[np.argmax(deltas[finite])])
        best_delta = float(deltas[best_local])
        pos_new = projected[best_local]
        trial = st.copy()
        self._ensure_source_metadata(trial)
        trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
        trial.strengths = np.append(
            trial_strengths[best_local],
            float(q_new[best_local]),
        )
        trial.ages = np.append(trial.ages[: trial.num_sources], 0)
        trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
        trial.tentative_sources = np.append(
            trial.tentative_sources[: trial.num_sources],
            True,
        )
        trial.verification_fail_streaks = np.append(
            trial.verification_fail_streaks[: trial.num_sources],
            0,
        )
        trial.num_sources = int(trial.positions.shape[0])
        return trial, best_delta

    def _apply_matching_pursuit_births_to_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
        candidate_positions: NDArray[np.float64],
        *,
        max_new_sources: int,
        candidate_unit_counts: NDArray[np.float64] | None = None,
    ) -> int:
        """
        Add residual-supported fixed-strength sources by matching pursuit.

        Every iteration recomputes the residual for the unchanged current state,
        evaluates the candidate additions under the structural PF likelihood,
        and accepts only a proposal that pays the configured complexity penalty.
        """
        max_new = max(0, int(max_new_sources))
        if max_new <= 0 or data.z_k.size == 0:
            return 0
        candidates = np.asarray(candidate_positions, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
            return 0
        self._ensure_source_metadata(st)
        accepted = 0
        topk = max(1, int(self.config.birth_matching_pursuit_topk_candidates))
        threshold = self._structural_acceptance_threshold(
            base_threshold=float(self.config.birth_delta_ll_threshold),
            complexity_penalty=self._birth_complexity_penalty(
                measurement_count=int(data.z_k.size),
            ),
        )
        if candidate_unit_counts is None:
            unit_counts_all = expected_counts_per_source(
                kernel=self.continuous_kernel,
                isotope=self.isotope,
                detector_positions=data.detector_positions,
                sources=candidates,
                strengths=np.ones(candidates.shape[0], dtype=float),
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
                source_scale=self._measurement_source_scale_vector(
                    data.fe_indices,
                    data.pb_indices,
                ),
            )
        else:
            unit_counts_all = np.asarray(candidate_unit_counts, dtype=float)
            expected_shape = (int(data.z_k.size), int(candidates.shape[0]))
            if unit_counts_all.shape != expected_shape:
                raise ValueError("candidate_unit_counts must have shape K x C.")

        for _ in range(max_new):
            if (
                self.config.max_sources is not None
                and st.num_sources >= self.config.max_sources
            ):
                break
            unit_counts_existing = self._unit_response_counts_for_state(st, data)
            source_strengths = np.asarray(
                st.strengths[: st.num_sources],
                dtype=float,
            )
            lambda_total = (
                np.asarray(data.live_times, dtype=float) * float(st.background)
                + unit_counts_existing @ source_strengths
            )
            residual = np.maximum(
                np.asarray(data.z_k, dtype=float) - lambda_total,
                0.0,
            )
            if float(np.sum(residual)) <= 0.0:
                break

            support_mask = self._birth_candidate_support_mask(
                data=data,
                candidate_counts=unit_counts_all,
                residual_mix=residual,
            )
            active_mask = self._active_source_mask(st)
            if (
                active_mask.size == st.num_sources
                and unit_counts_existing.shape[1] == st.num_sources
            ):
                existing_counts = unit_counts_existing[:, active_mask]
            else:
                existing_counts = self._birth_existing_unit_response_counts_for_state(
                    st,
                    data,
                )
            distance_mask = np.ones(candidates.shape[0], dtype=bool)
            if st.num_sources > 0:
                distances = np.linalg.norm(
                    candidates[:, None, :] - st.positions[None, : st.num_sources, :],
                    axis=2,
                )
                distance_mask = np.min(distances, axis=1) >= float(
                    self.config.birth_min_sep_m
                )
            keep = support_mask & distance_mask
            if not np.any(keep):
                break

            scores, q_hat = self._birth_residual_candidate_scores(
                candidate_counts=unit_counts_all,
                residual_mix=residual,
                observation_variances=data.observation_variances,
            )
            valid = (
                keep
                & np.isfinite(scores)
                & np.isfinite(q_hat)
                & (scores > 0.0)
                & (q_hat > 0.0)
            )
            if not np.any(valid):
                break
            ranked = np.flatnonzero(valid)
            ranked = ranked[np.argsort(scores[ranked])[::-1][:topk]]
            if bool(self.config.birth_orthogonalize_residual_candidates):
                ranked = self._orthogonalized_residual_candidate_indices(
                    ranked.astype(np.int64, copy=False),
                    candidate_counts=unit_counts_all,
                    existing_response_counts=existing_counts,
                    observation_variances=data.observation_variances,
                    max_corr=float(self.config.birth_orthogonal_candidate_corr_max),
                )
            base_ll = self._trial_log_likelihood_from_lambda(
                data,
                lambda_total,
            )
            best_trial, best_delta = (
                self._best_cached_matching_pursuit_birth_trial_batched(
                    st,
                    data,
                    candidates=candidates,
                    ranked_candidate_indices=ranked.astype(int, copy=False),
                    q_hat=q_hat,
                    unit_counts_existing=unit_counts_existing,
                    unit_counts_all=unit_counts_all,
                    source_strengths=source_strengths,
                    base_ll=base_ll,
                )
            )
            if (
                best_trial is None
                or not np.isfinite(best_delta)
                or best_delta < threshold
            ):
                break

            old_count = int(st.num_sources)
            self._stage_new_birth_metadata(
                best_trial,
                data,
                old_count=old_count,
                delta_ll=best_delta,
            )
            for idx in range(old_count, int(best_trial.num_sources)):
                self._record_source_event(
                    "source_birth_accepted",
                    best_trial,
                    int(idx),
                    reason="matching_pursuit_birth",
                    extra={"delta_ll": float(best_delta)},
                )
            self._replace_particle_state_from_trial(st, best_trial)
            accepted += 1
        return accepted

    def _single_residual_birth_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        position: NDArray[np.float64],
        strength: float,
    ) -> tuple[IsotopeState | None, float]:
        """Return an accepted one-source birth trial under the full likelihood."""
        q_new = float(strength)
        if not np.isfinite(q_new) or q_new <= 0.0:
            return None, -np.inf
        pos_new = np.asarray(position, dtype=float).reshape(3)
        if st.num_sources > 0:
            distance = np.linalg.norm(
                st.positions[: st.num_sources] - pos_new[None, :],
                axis=1,
            )
            if np.any(distance < float(self.config.birth_min_sep_m)):
                return None, -np.inf
        trial = st.copy()
        self._ensure_source_metadata(trial)
        trial.positions = np.vstack(
            [trial.positions[: trial.num_sources], pos_new]
        )
        trial.strengths = np.append(
            trial.strengths[: trial.num_sources],
            q_new,
        )
        trial.ages = np.append(trial.ages[: trial.num_sources], 0)
        trial.support_scores = np.append(
            trial.support_scores[: trial.num_sources],
            0.0,
        )
        trial.tentative_sources = np.append(
            trial.tentative_sources[: trial.num_sources],
            True,
        )
        trial.verification_fail_streaks = np.append(
            trial.verification_fail_streaks[: trial.num_sources],
            0,
        )
        trial.num_sources = int(trial.positions.shape[0])
        delta_ll = float(
            self._trial_log_likelihood(trial, data)
            - self._trial_log_likelihood(st, data)
        )
        threshold = self._structural_acceptance_threshold(
            base_threshold=float(self.config.birth_delta_ll_threshold),
            complexity_penalty=self._birth_complexity_penalty(
                measurement_count=int(data.z_k.size),
            ),
        )
        if not np.isfinite(delta_ll) or delta_ll < threshold:
            return None, delta_ll
        return trial, delta_ll

    def _stage_new_birth_metadata(
        self,
        trial: IsotopeState,
        data: MeasurementData,
        *,
        old_count: int,
        delta_ll: float,
    ) -> None:
        """Mark single-station births as tentative and boost their support score."""
        self._ensure_source_metadata(trial)
        new_start = max(0, int(old_count))
        if int(trial.num_sources) <= new_start:
            return
        support_value = max(float(delta_ll), 0.0)
        if trial.support_scores.size < int(trial.num_sources):
            trial.support_scores = self._resize_metadata_array(
                trial.support_scores,
                int(trial.num_sources),
                0.0,
                float,
            )
        trial.support_scores[new_start : int(trial.num_sources)] = np.maximum(
            trial.support_scores[new_start : int(trial.num_sources)],
            support_value,
        )
        if not bool(
            getattr(self.config, "birth_stage_single_station_as_quarantine", True)
        ):
            return
        support = np.ones(int(data.z_k.size), dtype=bool)
        station_count = self._distinct_supported_station_count(
            data.detector_positions,
            support,
        )
        if int(station_count) > 1:
            return
        if trial.verification_fail_streaks.size < int(trial.num_sources):
            trial.verification_fail_streaks = self._resize_metadata_array(
                trial.verification_fail_streaks,
                int(trial.num_sources),
                0,
                int,
            )
        trial.tentative_sources[new_start : int(trial.num_sources)] = True
        trial.verification_fail_streaks[new_start : int(trial.num_sources)] = (
            np.maximum(
                trial.verification_fail_streaks[new_start : int(trial.num_sources)],
                1,
            )
        )

    def _birth_existing_unit_response_counts_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return unit-strength response columns for one particle state."""
        if st.num_sources <= 0:
            return np.zeros((data.z_k.size, 0), dtype=float)
        active_mask = self._active_source_mask(st)
        if not np.any(active_mask):
            return np.zeros((data.z_k.size, 0), dtype=float)
        positions = st.positions[: st.num_sources][active_mask]
        counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=positions,
            strengths=np.ones(positions.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        return np.asarray(counts, dtype=float)

    def _verify_pseudo_sources_for_state(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        suppress_prune: bool = False,
        cached_lambda_m: NDArray[np.float64] | None = None,
        cached_lambda_total: NDArray[np.float64] | None = None,
        cached_delta_ll: NDArray[np.float64] | None = None,
        cached_prune_allowed: NDArray[np.bool_] | None = None,
    ) -> bool:
        """
        Verify tentative birth sources with leave-one-out likelihood support.

        Tentative sources are kept only when the original observation block
        supports them across enough independent shield views.  This implements
        a Bai-style pseudo-source verification gate without altering transport
        or spectrum-derived counts.
        """
        if not bool(self.config.pseudo_source_verification_enable):
            return False
        if st.num_sources <= 0 or data.z_k.size == 0:
            return False
        self._ensure_source_metadata(st)
        tentative = np.asarray(st.tentative_sources[: st.num_sources], dtype=bool)
        if not np.any(tentative):
            return False
        if (
            cached_lambda_m is not None
            and cached_lambda_total is not None
            and cached_delta_ll is not None
            and cached_lambda_m.shape == (int(data.z_k.size), int(st.num_sources))
            and cached_lambda_total.shape == (int(data.z_k.size),)
            and cached_delta_ll.shape == (int(st.num_sources),)
        ):
            lambda_m = cached_lambda_m
            lambda_total = cached_lambda_total
            delta_ll = cached_delta_ll
        else:
            lambda_m, lambda_total = self._lambda_components(st, data)
            delta_ll = self._structural_delta_log_likelihood_remove(
                data,
                lambda_total,
                lambda_m,
            )
        if lambda_m.shape[1] != st.num_sources:
            return False
        variances = self._structural_effective_variance_np(
            data.z_k,
            np.asarray(lambda_total, dtype=float)[:, None],
            data.observation_variances,
        )[:, 0]
        sigma = np.sqrt(variances)
        min_delta = float(self.config.pseudo_source_min_delta_ll)
        min_views = max(1, int(self.config.pseudo_source_min_distinct_views))
        grace = self._pseudo_source_fail_grace()
        corr_max = float(np.clip(float(self.config.pseudo_source_corr_max), 0.0, 1.0))
        temporal_sep_min = max(
            0.0,
            float(getattr(self.config, "pseudo_source_temporal_sep_min", 0.0)),
        )
        keep = np.ones(st.num_sources, dtype=bool)
        changed = False
        prune_allowed: NDArray[np.bool_] | None = None
        quarantined_before = self._quarantined_source_mask(st)
        for source_idx in range(st.num_sources):
            if not bool(tentative[source_idx]):
                continue
            component = np.maximum(lambda_m[:, source_idx], 0.0)
            support_mask = component / np.maximum(sigma, 1.0e-12) >= max(
                float(self.config.birth_residual_support_sigma), 0.0
            )
            distinct_views = self._distinct_supported_view_count(
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                support_mask,
            )
            response_supported = (
                float(delta_ll[source_idx]) >= min_delta
                and int(distinct_views) >= min_views
            )
            corr_failed = False
            if st.num_sources > 1 and corr_max < 1.0:
                stronger = [
                    idx
                    for idx in range(st.num_sources)
                    if idx != source_idx
                    and float(st.strengths[idx]) >= float(st.strengths[source_idx])
                ]
                if stronger:
                    correlations = [
                        self._response_correlation(component, lambda_m[:, idx])
                        for idx in stronger
                    ]
                    corr_failed = max(correlations, default=0.0) >= corr_max
                    if corr_failed and temporal_sep_min > 0.0:
                        separations = [
                            self._temporal_response_separation(
                                component,
                                lambda_m[:, idx],
                                sigma,
                            )
                            for idx in stronger
                        ]
                        if max(separations, default=0.0) >= temporal_sep_min:
                            corr_failed = False
            if response_supported and not corr_failed:
                self._record_source_event(
                    "pseudo_source_verified",
                    st,
                    source_idx,
                    reason="delta_ll_and_distinct_views_supported",
                    extra={
                        "delta_ll": float(delta_ll[source_idx]),
                        "distinct_views": int(distinct_views),
                        "min_delta_ll": float(min_delta),
                        "min_distinct_views": int(min_views),
                    },
                )
                st.tentative_sources[source_idx] = False
                st.verification_fail_streaks[source_idx] = 0
                self.last_pseudo_source_verified += 1
                changed = True
                continue
            was_quarantined = bool(quarantined_before[source_idx])
            self.last_pseudo_source_failed += 1
            fail_reasons: list[str] = []
            if float(delta_ll[source_idx]) < min_delta:
                fail_reasons.append("insufficient_delta_ll")
            if int(distinct_views) < min_views:
                fail_reasons.append("insufficient_distinct_views")
            if corr_failed:
                fail_reasons.append("high_response_corr")
                if temporal_sep_min > 0.0:
                    fail_reasons.append("insufficient_temporal_separation")
            if float(np.sum(component)) <= 0.0:
                fail_reasons.append("low_expected_contribution")
            if int(st.verification_fail_streaks[source_idx]) < grace:
                fail_reasons.append("too_young_to_prune")
            if not fail_reasons:
                fail_reasons.append("unsupported")
            for reason in fail_reasons:
                self.last_pseudo_source_fail_reasons[reason] = (
                    int(self.last_pseudo_source_fail_reasons.get(reason, 0)) + 1
                )
            observation_limited = int(distinct_views) < min_views or bool(corr_failed)
            if observation_limited:
                self.last_pseudo_source_fail_reasons["needs_discriminative_views"] = (
                    int(
                        self.last_pseudo_source_fail_reasons.get(
                            "needs_discriminative_views",
                            0,
                        )
                    )
                    + 1
                )
                continue
            st.verification_fail_streaks[source_idx] += 1
            if int(st.verification_fail_streaks[source_idx]) < grace:
                self.last_pseudo_source_fail_reasons["too_young_to_prune"] = (
                    int(
                        self.last_pseudo_source_fail_reasons.get(
                            "too_young_to_prune",
                            0,
                        )
                    )
                    + 1
                )
            if (
                st.num_sources > 1
                and int(st.verification_fail_streaks[source_idx]) >= grace
            ):
                quarantine_enabled = bool(
                    self.config.pseudo_source_quarantine_on_suppress
                )
                if not suppress_prune and (was_quarantined or not quarantine_enabled):
                    if prune_allowed is None:
                        if (
                            cached_prune_allowed is not None
                            and cached_prune_allowed.shape == (int(st.num_sources),)
                        ):
                            prune_allowed = np.asarray(cached_prune_allowed, dtype=bool)
                        else:
                            prune_allowed = self._source_prune_allowed_mask(
                                st,
                                data,
                                lambda_m=lambda_m,
                                lambda_total=lambda_total,
                                delta_ll=delta_ll,
                            )
                prune_now = (
                    not suppress_prune
                    and prune_allowed is not None
                    and bool(prune_allowed[source_idx])
                    and (was_quarantined or not quarantine_enabled)
                )
                if quarantine_enabled and not was_quarantined:
                    self._record_source_event(
                        "pseudo_source_quarantined",
                        st,
                        source_idx,
                        reason=";".join(fail_reasons),
                        extra={
                            "delta_ll": float(delta_ll[source_idx]),
                            "distinct_views": int(distinct_views),
                            "min_delta_ll": float(min_delta),
                            "min_distinct_views": int(min_views),
                            "suppress_prune": bool(suppress_prune),
                        },
                    )
                    self.last_pseudo_source_quarantined += 1
                    changed = True
                elif prune_now:
                    keep[source_idx] = False
        quarantine_mask_after = self._quarantined_source_mask(st)
        self.last_pseudo_source_quarantine_active += int(
            np.count_nonzero(quarantine_mask_after)
        )
        if np.all(keep):
            return changed
        if np.count_nonzero(keep) == 0:
            strongest = int(np.argmax(st.strengths[: st.num_sources]))
            keep[strongest] = True
        pruned = int(np.count_nonzero(~keep))
        if pruned <= 0:
            return changed
        for idx in np.flatnonzero(~keep):
            self._record_source_event(
                "source_removed",
                st,
                int(idx),
                reason="pseudo_source_pruned",
                extra={"suppress_prune": bool(suppress_prune)},
            )
        st.positions = st.positions[keep]
        st.strengths = st.strengths[keep]
        st.ages = st.ages[keep]
        st.support_scores = st.support_scores[keep]
        st.tentative_sources = st.tentative_sources[keep]
        st.verification_fail_streaks = st.verification_fail_streaks[keep]
        st.num_sources = st.positions.shape[0]
        self.last_pseudo_source_pruned += pruned
        self.last_kill_count += pruned
        return True

    @staticmethod
    def _response_correlation(
        first: NDArray[np.float64],
        second: NDArray[np.float64],
    ) -> float:
        """Return non-negative response-pattern correlation for two sources."""
        a = np.asarray(first, dtype=float).reshape(-1)
        b = np.asarray(second, dtype=float).reshape(-1)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 0.0 or b_norm <= 0.0:
            return 0.0
        return float(np.dot(a, b) / max(a_norm * b_norm, 1.0e-12))

    @staticmethod
    def _temporal_response_separation(
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        sigma: NDArray[np.float64],
    ) -> float:
        """Return whitened temporal-code separation between two source responses."""
        a = np.asarray(first, dtype=float).reshape(-1)
        b = np.asarray(second, dtype=float).reshape(-1)
        s = np.asarray(sigma, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size or a.size != s.size:
            return 0.0
        denom = np.maximum(s, 1.0e-12)
        diff = (a - b) / denom
        value = float(np.sum(diff * diff))
        return value if np.isfinite(value) else 0.0

    def _structural_trial_worker_count(self, trial_count: int) -> int:
        """Return worker count for deterministic structural trial chunks."""
        count = max(0, int(trial_count))
        if count <= 1:
            return 1
        min_trials = max(1, int(self.config.structural_trial_parallel_min_trials))
        if count < min_trials:
            return 1
        workers = max(1, int(self.config.structural_trial_workers))
        return min(count, workers)

    @staticmethod
    def _chunk_sequence(
        values: list[Any],
        worker_count: int,
    ) -> list[list[Any]]:
        """Split values into non-empty ordered chunks for deterministic workers."""
        workers = max(1, int(worker_count))
        if workers <= 1 or len(values) <= 1:
            return [values]
        chunks: list[list[Any]] = []
        for index_array in np.array_split(np.arange(len(values)), workers):
            if index_array.size == 0:
                continue
            chunks.append([values[int(idx)] for idx in index_array])
        return chunks

    def _best_merge_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best likelihood-tested merge trial and its likelihood gain.

        Candidate pairs are selected either by spatial proximity or by nearly
        collinear response signatures over the actual measurement block. The
        merged state preserves total source strength and is scored directly by
        the configured PF count likelihood.
        """
        if data.z_k.size == 0 or st.num_sources < 2:
            return None, -np.inf
        self._ensure_source_metadata(st)
        lambda_m, _ = self._lambda_components(st, data)
        if lambda_m.shape[1] < 2:
            return None, -np.inf
        base_ll = self._trial_log_likelihood(st, data)
        if not np.isfinite(base_ll):
            return None, -np.inf
        pair_scores = self._merge_candidate_pair_scores(st, lambda_m)
        if not pair_scores:
            return None, -np.inf
        pair_scores.sort(reverse=True)
        max_pairs = max(1, int(self.config.merge_search_topk_pairs))
        return self._best_merge_trial_batched(
            st,
            data,
            pair_scores=pair_scores[:max_pairs],
            base_ll=base_ll,
        )

    def _best_merge_trial_scalar(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best merge trial using the scalar reference path."""
        if data.z_k.size == 0 or st.num_sources < 2:
            return None, -np.inf
        self._ensure_source_metadata(st)
        lambda_m, _ = self._lambda_components(st, data)
        if lambda_m.shape[1] < 2:
            return None, -np.inf
        base_ll = self._trial_log_likelihood(st, data)
        if not np.isfinite(base_ll):
            return None, -np.inf
        pair_scores = self._merge_candidate_pair_scores(st, lambda_m)
        if not pair_scores:
            return None, -np.inf
        pair_scores.sort(reverse=True)
        max_pairs = max(1, int(self.config.merge_search_topk_pairs))
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        for _, i, j in pair_scores[:max_pairs]:
            trial = self._make_merge_trial_state(st, int(i), int(j))
            ll_after = self._trial_log_likelihood(trial, data)
            delta_ll = float(ll_after - base_ll)
            if delta_ll > best_delta:
                best_delta = delta_ll
                best_trial = trial
        return best_trial, best_delta

    def _merge_candidate_pair_scores(
        self,
        st: IsotopeState,
        lambda_m: NDArray[np.float64],
    ) -> list[tuple[float, int, int]]:
        """Return sorted-eligible merge pair scores before likelihood testing."""
        corr_min = float(self.config.merge_response_corr_min)
        distance_max = max(float(self.config.merge_distance_max), 0.0)
        pair_scores: list[tuple[float, int, int]] = []
        for i in range(st.num_sources):
            for j in range(i + 1, st.num_sources):
                distance = float(np.linalg.norm(st.positions[i] - st.positions[j]))
                corr = self._response_correlation(lambda_m[:, i], lambda_m[:, j])
                close_enough = distance_max > 0.0 and distance <= distance_max
                response_redundant = corr_min > 0.0 and corr >= corr_min
                if not close_enough and not response_redundant:
                    continue
                score = corr - distance / max(distance_max, 1.0)
                pair_scores.append((float(score), i, j))
        return pair_scores

    def _make_merge_trial_state(
        self,
        st: IsotopeState,
        first_idx: int,
        second_idx: int,
    ) -> IsotopeState:
        """Return a fixed-strength merge trial state for one pair."""
        self._ensure_source_metadata(st)
        i = int(first_idx)
        j = int(second_idx)
        q1 = float(st.strengths[i])
        q2 = float(st.strengths[j])
        if q1 + q2 > 0.0:
            merged_pos = (q1 * st.positions[i] + q2 * st.positions[j]) / (q1 + q2)
        else:
            merged_pos = 0.5 * (st.positions[i] + st.positions[j])
        merged_pos = self._project_positions_to_source_prior(
            np.asarray(merged_pos, dtype=float).reshape(1, 3)
        )[0]
        keep = np.ones(st.num_sources, dtype=bool)
        keep[[i, j]] = False
        return IsotopeState(
            num_sources=int(np.count_nonzero(keep) + 1),
            positions=np.vstack([st.positions[keep], merged_pos]),
            strengths=np.append(st.strengths[keep], q1 + q2),
            background=float(st.background),
            ages=np.append(st.ages[keep], max(int(st.ages[i]), int(st.ages[j]))),
            support_scores=np.append(
                st.support_scores[keep],
                max(float(st.support_scores[i]), float(st.support_scores[j])),
            ),
            tentative_sources=np.append(
                st.tentative_sources[keep],
                bool(st.tentative_sources[i] or st.tentative_sources[j]),
            ),
            verification_fail_streaks=np.append(
                st.verification_fail_streaks[keep],
                min(
                    int(st.verification_fail_streaks[i]),
                    int(st.verification_fail_streaks[j]),
                ),
            ),
        )

    def _best_merge_trial_batched(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        pair_scores: list[tuple[float, int, int]],
        base_ll: float,
        allow_parallel: bool = True,
    ) -> tuple[IsotopeState | None, float]:
        """Return the best fixed-strength merge trial with batched responses."""
        if not pair_scores or data.z_k.size == 0:
            return None, -np.inf
        worker_count = (
            self._structural_trial_worker_count(len(pair_scores))
            if allow_parallel
            else 1
        )
        if worker_count > 1:
            chunks = self._chunk_sequence(pair_scores, worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        lambda chunk: self._best_merge_trial_batched(
                            st,
                            data,
                            pair_scores=chunk,
                            base_ll=base_ll,
                            allow_parallel=False,
                        ),
                        chunks,
                    )
                )
            best_trial: IsotopeState | None = None
            best_delta = -np.inf
            for trial, delta in results:
                if delta > best_delta:
                    best_delta = float(delta)
                    best_trial = trial
            return best_trial, best_delta
        trials = [
            self._make_merge_trial_state(st, int(i), int(j)) for _, i, j in pair_scores
        ]
        if not trials:
            return None, -np.inf
        source_count = int(trials[0].num_sources)
        if source_count <= 0 or any(
            int(trial.num_sources) != source_count for trial in trials
        ):
            return self._best_merge_trial_scalar(st, data)
        trial_count = int(len(trials))
        flat_sources = np.vstack([trial.positions[:source_count] for trial in trials])
        flat_unit_counts = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=flat_sources,
            strengths=np.ones(flat_sources.shape[0], dtype=float),
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale_vector(
                data.fe_indices,
                data.pb_indices,
            ),
        )
        unit_tensor = np.asarray(flat_unit_counts, dtype=float).reshape(
            int(data.z_k.size),
            trial_count,
            source_count,
        )
        strengths = np.vstack(
            [
                np.asarray(trial.strengths[:source_count], dtype=float)
                for trial in trials
            ]
        )
        source_counts = np.einsum(
            "kts,ts->kt",
            unit_tensor,
            strengths,
            optimize=True,
        )
        background_counts = (
            np.asarray(data.live_times, dtype=float)[:, None]
            * np.asarray(
                [float(trial.background) for trial in trials],
                dtype=float,
            )[None, :]
        )
        lambda_total = background_counts + source_counts
        ll_cached = self._structural_count_log_likelihood_matrix_np(
            data,
            lambda_total,
        )
        deltas = np.asarray(ll_cached, dtype=float) - float(base_ll)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            return None, -np.inf
        best_idx = int(np.flatnonzero(finite)[np.argmax(deltas[finite])])
        return trials[best_idx], float(deltas[best_idx])

    def _source_detector_exclusion_mask(
        self,
        st: IsotopeState,
        data: MeasurementData | None,
    ) -> NDArray[np.bool_]:
        """Return sources satisfying the declared detector-clearance prior."""
        if st.num_sources <= 0:
            return np.ones(0, dtype=bool)
        min_sep = max(float(self.config.source_detector_exclusion_m), 0.0)
        if min_sep <= 0.0 or data is None or data.detector_positions.size == 0:
            return np.ones(st.num_sources, dtype=bool)
        det = np.asarray(data.detector_positions, dtype=float)
        if det.ndim != 2 or det.shape[1] != 3:
            return np.ones(st.num_sources, dtype=bool)
        dist = np.linalg.norm(st.positions[:, None, :] - det[None, :, :], axis=2)
        return np.min(dist, axis=1) >= min_sep

    def _source_detector_exclusion_mask_group(
        self,
        source_positions: NDArray[np.float64],
        data: MeasurementData | None,
    ) -> NDArray[np.bool_]:
        """Return detector-clearance masks for a particle-by-source position batch."""
        positions = np.asarray(source_positions, dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 3:
            return np.zeros((0, 0), dtype=bool)
        particle_count, source_count = positions.shape[:2]
        min_sep = max(float(self.config.source_detector_exclusion_m), 0.0)
        if min_sep <= 0.0 or data is None or data.detector_positions.size == 0:
            return np.ones((particle_count, source_count), dtype=bool)
        detector_positions = np.asarray(data.detector_positions, dtype=float)
        if detector_positions.ndim != 2 or detector_positions.shape[1] != 3:
            return np.ones((particle_count, source_count), dtype=bool)
        offsets = (
            positions[None, :, :, :]
            - detector_positions[:, None, None, :]
        )
        squared_distances = np.sum(offsets * offsets, axis=3)
        return np.all(squared_distances >= min_sep * min_sep, axis=0)

    def _select_structural_proposal_indices(
        self,
        limit: int | None,
        *,
        require_birth_capacity: bool = False,
    ) -> set[int] | None:
        """
        Return posterior-diverse particle indices for expensive structural moves.

        Structural birth/split/merge proposals are likelihood-scored heuristic
        moves. Evaluating them for every particle can dominate runtime without
        changing transport fidelity. This selector keeps the highest posterior
        particles while reserving quota for each active source cardinality so
        low-probability multi-source hypotheses are not discarded solely by a
        global weight sort.
        """
        total = len(self.continuous_particles)
        if total <= 0:
            return set()
        if limit is None:
            return None
        max_count = int(limit)
        if max_count <= 0:
            return set()
        if require_birth_capacity and self.config.max_sources is not None:
            eligible_indices = [
                int(idx)
                for idx, particle in enumerate(self.continuous_particles)
                if int(particle.state.num_sources) < int(self.config.max_sources)
            ]
        else:
            eligible_indices = list(range(total))
        if not eligible_indices:
            return set()
        if max_count >= len(eligible_indices):
            return None if len(eligible_indices) == total else set(eligible_indices)
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != total:
            weights = np.ones(total, dtype=float) / float(total)
        finite = np.isfinite(weights)
        if not np.any(finite):
            weights = np.ones(total, dtype=float) / float(total)
        else:
            weights = np.where(finite, weights, 0.0)
        global_quota = max(1, max_count // 2)
        eligible_set = set(eligible_indices)
        order = np.asarray(
            [idx for idx in np.argsort(weights)[::-1] if int(idx) in eligible_set],
            dtype=int,
        )
        if order.size == 0:
            return set()
        selected: set[int] = set(int(idx) for idx in order[:global_quota])
        grouped: dict[int, list[int]] = {}
        for idx in eligible_indices:
            particle = self.continuous_particles[int(idx)]
            grouped.setdefault(int(particle.state.num_sources), []).append(idx)
        if grouped and len(selected) < max_count:
            group_quota = max(1, (max_count - len(selected)) // max(len(grouped), 1))
            for indices in grouped.values():
                ranked = sorted(
                    indices, key=lambda item: float(weights[item]), reverse=True
                )
                for idx in ranked[:group_quota]:
                    selected.add(int(idx))
                    if len(selected) >= max_count:
                        break
                if len(selected) >= max_count:
                    break
        if len(selected) < max_count:
            for idx in order:
                selected.add(int(idx))
                if len(selected) >= max_count:
                    break
        return selected

    def refresh_weights_from_measurements(
        self,
        data: MeasurementData | None,
        *,
        lambda_total_by_index: dict[int, NDArray[np.float64]] | None = None,
        reference_log_likelihood_by_index: dict[int, float] | None = None,
        moved_indices: set[int] | None = None,
        likelihood_unchanged_indices: set[int] | None = None,
    ) -> None:
        """
        Recompute particle weights from a measurement block after structural moves.

        Birth, death, split, and merge moves change the state dimension.  When
        those moves are proposed after a station-level resampling step, the
        modified particles must be reweighted by the same station likelihood;
        otherwise a proposal can affect the reported posterior without being
        judged by the observation that triggered it.

        ``lambda_total_by_index`` may contain exact expected-count vectors for
        particles whose states were not modified during the structural update.
        Reusing those vectors only avoids duplicate kernel evaluations; it does
        not alter the likelihood or PF update rule.

        When ``reference_log_likelihood_by_index`` and ``moved_indices`` are
        given, only moved particles are corrected by the likelihood ratio
        ``new_window_ll - old_window_ll``.  This preserves all previous
        posterior evidence already accumulated before the structural move. It
        is not a forward/reverse proposal-density, prior, or Jacobian correction,
        so it does not turn the structural heuristic into an exact RJ move.

        ``likelihood_unchanged_indices`` identifies moved particles whose
        response-defining state is exactly unchanged (for example, a tentative
        source becoming verified). Their old likelihood is reused exactly while
        retaining the same normalization and resampling schedule.
        """
        if data is None or data.z_k.size == 0 or not self.continuous_particles:
            return
        if reference_log_likelihood_by_index is not None and moved_indices is not None:
            self._refresh_moved_particle_weights_from_measurements(
                data,
                reference_log_likelihood_by_index=reference_log_likelihood_by_index,
                moved_indices=moved_indices,
                likelihood_unchanged_indices=likelihood_unchanged_indices,
            )
            return
        log_likelihoods = np.full(len(self.continuous_particles), -np.inf, dtype=float)
        cached_lambda = lambda_total_by_index or {}
        expected_shape = (int(data.z_k.size),)
        grouped, fallback_indices = self._particle_indices_by_source_count()
        for source_count, particle_indices in grouped.items():
            missing_indices: list[int] = []
            cached_indices: list[int] = []
            cached_values: list[NDArray[np.float64]] = []
            for particle_idx in particle_indices:
                cached = cached_lambda.get(int(particle_idx))
                if cached is not None and np.asarray(cached).shape == expected_shape:
                    cached_indices.append(int(particle_idx))
                    cached_values.append(np.asarray(cached, dtype=float))
                else:
                    missing_indices.append(int(particle_idx))
            if cached_indices:
                cached_matrix = np.column_stack(cached_values)
                cached_ll = self._structural_count_log_likelihood_matrix_np(
                    data,
                    cached_matrix,
                )
                log_likelihoods[np.asarray(cached_indices, dtype=int)] = cached_ll
            if not missing_indices:
                continue
            _, lambda_total = self._lambda_components_for_particle_group(
                data,
                missing_indices,
                source_count,
            )
            group_ll = self._structural_count_log_likelihood_matrix_np(
                data,
                lambda_total,
            )
            log_likelihoods[np.asarray(missing_indices, dtype=int)] = group_ll
        for idx in fallback_indices:
            cached = cached_lambda.get(int(idx))
            if cached is not None and np.asarray(cached).shape == expected_shape:
                lambda_total = np.asarray(cached, dtype=float)
            else:
                st = self.continuous_particles[idx].state
                _, lambda_total = self._lambda_components(st, data)
            log_likelihoods[idx] = self._structural_count_log_likelihood_np(
                data,
                lambda_total,
            )
        norm = logsumexp(log_likelihoods)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, log_likelihoods - norm):
            particle.log_weight = float(value)

    def _refresh_moved_particle_weights_from_measurements(
        self,
        data: MeasurementData,
        *,
        reference_log_likelihood_by_index: dict[int, float],
        moved_indices: set[int],
        likelihood_unchanged_indices: set[int] | None = None,
    ) -> None:
        """Apply station-window likelihood-ratio corrections to moved particles."""
        if data.z_k.size == 0 or not moved_indices:
            return
        valid_indices = [
            int(idx)
            for idx in sorted(moved_indices)
            if 0 <= int(idx) < len(self.continuous_particles)
        ]
        if not valid_indices:
            return
        unchanged = {
            int(idx)
            for idx in (likelihood_unchanged_indices or set())
            if int(idx) in moved_indices
        }
        changed_indices = [idx for idx in valid_indices if idx not in unchanged]
        changed_ll = (
            self._window_log_likelihoods_for_indices(data, changed_indices)
            if changed_indices
            else np.zeros(0, dtype=float)
        )
        changed_ll_by_index = {
            int(idx): float(value) for idx, value in zip(changed_indices, changed_ll)
        }
        for particle_idx in valid_indices:
            ll_old = float(
                reference_log_likelihood_by_index.get(int(particle_idx), np.nan)
            )
            ll_new = (
                ll_old
                if int(particle_idx) in unchanged
                else float(changed_ll_by_index.get(int(particle_idx), np.nan))
            )
            if not np.isfinite(ll_old) or not np.isfinite(ll_new):
                continue
            particle = self.continuous_particles[int(particle_idx)]
            particle.log_weight = float(particle.log_weight + float(ll_new) - ll_old)
        self._normalize_continuous_log_weights()

    def _window_log_likelihoods_for_indices(
        self,
        data: MeasurementData,
        indices: list[int],
    ) -> NDArray[np.float64]:
        """Return measurement-window log likelihoods for selected particles."""
        out = np.full(len(indices), -np.inf, dtype=float)
        if not indices:
            return out
        grouped: dict[int, list[tuple[int, int]]] = {}
        fallback: list[tuple[int, int]] = []
        for out_idx, particle_idx in enumerate(indices):
            st = self.continuous_particles[int(particle_idx)].state
            source_count = int(st.num_sources)
            if source_count > 0:
                grouped.setdefault(source_count, []).append(
                    (out_idx, int(particle_idx))
                )
            else:
                fallback.append((out_idx, int(particle_idx)))
        for source_count, pairs in grouped.items():
            particle_indices = [particle_idx for _, particle_idx in pairs]
            _, lambda_total = self._lambda_components_for_particle_group(
                data,
                particle_indices,
                source_count,
            )
            group_ll = self._structural_count_log_likelihood_matrix_np(
                data,
                lambda_total,
            )
            for local_idx, (out_idx, _) in enumerate(pairs):
                out[int(out_idx)] = float(group_ll[int(local_idx)])
        for out_idx, particle_idx in fallback:
            st = self.continuous_particles[int(particle_idx)].state
            _, lambda_total = self._lambda_components(st, data)
            out[int(out_idx)] = self._structural_count_log_likelihood_np(
                data,
                lambda_total,
            )
        return out

    @staticmethod
    def _weighted_quantile(
        values: NDArray[np.float64],
        weights: NDArray[np.float64],
        quantile: float,
    ) -> float:
        """Return a robust weighted quantile for finite one-dimensional samples."""
        vals = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if vals.size == 0:
            return 0.0
        if w.size != vals.size:
            w = np.ones(vals.size, dtype=float)
        finite = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
        if not np.any(finite):
            return (
                float(np.median(vals[np.isfinite(vals)]))
                if np.any(np.isfinite(vals))
                else 0.0
            )
        vals = vals[finite]
        w = w[finite]
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        cumulative = np.cumsum(w)
        total = float(cumulative[-1])
        if total <= 0.0:
            return float(np.median(vals))
        target = float(np.clip(quantile, 0.0, 1.0)) * total
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), vals.size - 1)
        return float(vals[idx])

    def estimate_clustered(
        self,
        max_k: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Estimate source positions/strengths by robust posterior clustering.

        Source-existence ordering uses posterior mass, while the reported
        position and strength use weighted medians.  This avoids high-intensity
        posterior tails dominating the displayed estimate without changing the
        PF update itself. Surface-prior summaries are projected back into the
        constrained state space before they are returned.
        """
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        positions: list[NDArray[np.float64]] = []
        weights: list[float] = []
        strengths: list[float] = []
        cont_weights = np.asarray(self.continuous_weights, dtype=float)
        for p, w in zip(self.continuous_particles, cont_weights):
            st = p.state
            if st.num_sources <= 0:
                continue
            active_mask = self._active_source_mask(st)
            for pos, q in zip(
                st.positions[: st.num_sources][active_mask],
                st.strengths[: st.num_sources][active_mask],
            ):
                positions.append(np.asarray(pos, dtype=float))
                weights.append(float(w))
                strengths.append(float(q))
        if not positions:
            return np.zeros((0, 3)), np.zeros(0)
        pos_arr = np.vstack(positions)
        w_arr = np.asarray(weights, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        pos_arr, w_arr, q_arr = self._downsample_report_points(
            pos_arr,
            w_arr,
            q_arr,
            max_points=int(self.config.cluster_report_max_points),
        )
        eps = float(self.config.cluster_eps_m)
        if eps <= 0.0:
            eps = 1e-6
        min_samples = max(1, int(self.config.cluster_min_samples))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self.estimate()
        tree = cKDTree(pos_arr)
        clusters = self._connected_position_clusters(
            tree,
            point_count=int(pos_arr.shape[0]),
            eps=eps,
            min_samples=min_samples,
            exact_max_points=int(self.config.cluster_exact_max_points),
        )
        if not clusters:
            return np.zeros((0, 3)), np.zeros(0)
        cluster_pos: list[NDArray[np.float64]] = []
        cluster_q: list[float] = []
        cluster_mass: list[float] = []
        strength_floor = max(float(self.config.min_strength), 0.0) * (1.0 + 1.0e-6)
        for members in clusters:
            member_strengths = q_arr[members]
            active = member_strengths > strength_floor
            members_for_summary = members[active] if np.any(active) else members
            w = w_arr[members_for_summary]
            if np.sum(w) <= 0.0:
                w = np.ones_like(w, dtype=float)
            w = w / np.sum(w)
            member_pos = pos_arr[members_for_summary]
            member_q = q_arr[members_for_summary]
            pos_robust = np.array(
                [
                    self._weighted_quantile(member_pos[:, dim], w, 0.5)
                    for dim in range(member_pos.shape[1])
                ],
                dtype=float,
            )
            q_robust = self._weighted_quantile(member_q, w, 0.5)
            cluster_pos.append(pos_robust)
            cluster_q.append(q_robust)
            cluster_mass.append(float(np.sum(w_arr[members_for_summary])))
        order = np.argsort(cluster_mass)[::-1]
        if max_k is None:
            max_k = self.config.max_sources
        if max_k is not None:
            order = order[: max(0, int(max_k))]
        pos_out = (
            np.vstack([cluster_pos[i] for i in order])
            if order.size
            else np.zeros((0, 3))
        )
        pos_out = self._project_positions_to_source_prior(pos_out)
        q_out = (
            np.array([cluster_q[i] for i in order], dtype=float)
            if order.size
            else np.zeros(0, dtype=float)
        )
        return pos_out, q_out

    @staticmethod
    def _downsample_report_points(
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        max_points: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return a deterministic bounded point set for report-only clustering."""
        n_points = int(positions.shape[0])
        limit = int(max_points)
        if limit <= 0 or n_points <= limit:
            return positions, weights, strengths
        finite_weights = np.asarray(weights, dtype=float)
        finite_weights = np.where(np.isfinite(finite_weights), finite_weights, 0.0)
        if finite_weights.size != n_points or np.allclose(
            finite_weights, finite_weights[0]
        ):
            idx = np.linspace(0, n_points - 1, num=limit, dtype=np.int64)
        else:
            top_count = max(1, limit // 2)
            top_idx = np.argsort(finite_weights)[::-1][:top_count]
            uniform_count = max(0, limit - top_idx.size)
            uniform_idx = (
                np.linspace(0, n_points - 1, num=uniform_count, dtype=np.int64)
                if uniform_count > 0
                else np.zeros(0, dtype=np.int64)
            )
            idx = np.unique(np.concatenate([top_idx, uniform_idx]))
            if idx.size < limit:
                missing = limit - idx.size
                fill = np.setdiff1d(
                    np.linspace(
                        0, n_points - 1, num=min(n_points, 2 * missing), dtype=np.int64
                    ),
                    idx,
                    assume_unique=False,
                )[:missing]
                idx = np.concatenate([idx, fill])
        idx = np.asarray(idx[:limit], dtype=np.int64)
        return positions[idx], weights[idx], strengths[idx]

    @staticmethod
    def _connected_position_clusters(
        tree: Any,
        *,
        point_count: int,
        eps: float,
        min_samples: int,
        exact_max_points: int = 5000,
    ) -> list[NDArray[np.int64]]:
        """Return epsilon-neighborhood connected components for source positions."""
        count = max(0, int(point_count))
        if count <= 0:
            return []
        try:
            data = np.asarray(tree.data, dtype=float)
        except AttributeError:
            data = np.zeros((count, 0), dtype=float)
        if data.shape[0] != count:
            data = np.zeros((count, 0), dtype=float)
        if int(exact_max_points) > 0 and count > int(exact_max_points):
            return IsotopeParticleFilter._grid_position_clusters(
                data,
                eps=eps,
                min_samples=min_samples,
            )
        visited = np.zeros(count, dtype=bool)
        clusters: list[NDArray[np.int64]] = []
        sample_floor = max(1, int(min_samples))
        radius = float(eps)
        for seed in range(count):
            if visited[seed]:
                continue
            visited[seed] = True
            members: list[int] = [int(seed)]
            queue: list[int] = [int(seed)]
            while queue:
                idx = queue.pop()
                if data.size:
                    neighbors = tree.query_ball_point(data[idx], r=radius)
                else:
                    neighbors = [idx]
                if not neighbors:
                    continue
                new_neighbors: list[int] = []
                for neighbor in neighbors:
                    n_idx = int(neighbor)
                    if n_idx < 0 or n_idx >= count or visited[n_idx]:
                        continue
                    visited[n_idx] = True
                    members.append(n_idx)
                    new_neighbors.append(n_idx)
                if len(members) >= count:
                    queue.clear()
                    break
                queue.extend(new_neighbors)
            if len(members) >= sample_floor:
                clusters.append(np.asarray(members, dtype=np.int64))
        return clusters

    @staticmethod
    def _grid_position_clusters(
        data: NDArray[np.float64],
        *,
        eps: float,
        min_samples: int,
    ) -> list[NDArray[np.int64]]:
        """Return scalable report clusters by connected occupied spatial cells."""
        n_points = int(data.shape[0])
        if n_points <= 0:
            return []
        if data.ndim != 2 or data.shape[1] == 0:
            members = np.arange(n_points, dtype=np.int64)
            return [members] if members.size >= max(1, int(min_samples)) else []
        cell_size = max(float(eps), 1.0e-6)
        cells = np.floor(data / cell_size).astype(np.int64, copy=False)
        cell_to_points: dict[tuple[int, ...], list[int]] = {}
        for idx, cell in enumerate(cells):
            key = tuple(int(v) for v in cell)
            cell_to_points.setdefault(key, []).append(int(idx))
        parent: dict[tuple[int, ...], tuple[int, ...]] = {
            key: key for key in cell_to_points
        }

        def find(key: tuple[int, ...]) -> tuple[int, ...]:
            """Find the representative occupied cell."""
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != key:
                nxt = parent[key]
                parent[key] = root
                key = nxt
            return root

        def union(a: tuple[int, ...], b: tuple[int, ...]) -> None:
            """Union two occupied cells."""
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        offsets = np.array(np.meshgrid(*([[-1, 0, 1]] * data.shape[1]), indexing="ij"))
        offsets = offsets.reshape(data.shape[1], -1).T
        occupied = set(cell_to_points)
        for key in list(cell_to_points):
            key_arr = np.asarray(key, dtype=np.int64)
            for offset in offsets:
                if not np.any(offset):
                    continue
                neighbor = tuple(int(v) for v in (key_arr + offset))
                if neighbor in occupied:
                    union(key, neighbor)
        grouped: dict[tuple[int, ...], list[int]] = {}
        for key, members in cell_to_points.items():
            grouped.setdefault(find(key), []).extend(members)
        sample_floor = max(1, int(min_samples))
        return [
            np.asarray(sorted(members), dtype=np.int64)
            for members in grouped.values()
            if len(members) >= sample_floor
        ]

    def _structural_rj_response_signatures(
        self,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return response-relevant row signatures for safe prefix caching."""
        measurement_count = int(data.z_k.size)
        detector_positions = np.asarray(
            data.detector_positions,
            dtype=float,
        ).reshape(measurement_count, 3)
        live_times = np.asarray(
            data.live_times,
            dtype=float,
        ).reshape(measurement_count)
        fe_indices = np.asarray(
            data.fe_indices,
            dtype=np.int64,
        ).reshape(measurement_count)
        pb_indices = np.asarray(
            data.pb_indices,
            dtype=np.int64,
        ).reshape(measurement_count)
        source_scales = self._measurement_source_scale_vector(
            fe_indices,
            pb_indices,
        )
        signatures = np.column_stack(
            [
                detector_positions,
                live_times,
                fe_indices,
                pb_indices,
                source_scales,
            ]
        ).astype(np.float64, copy=False)
        if not np.all(np.isfinite(signatures)):
            raise ValueError(
                "rj_mh response signatures must contain finite values."
            )
        return signatures

    def _structural_rj_prepare_response_cache(
        self,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return a prefix-safe lazy response cache for the supplied history."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        signatures = self._structural_rj_response_signatures(data)
        measurement_count = int(signatures.shape[0])
        patch_count = int(patches.patch_count)
        cached = self._structural_rj_response_cache
        cached_signatures = self._structural_rj_response_cache_signatures

        matching_prefix = 0
        if cached is not None and cached_signatures is not None:
            matching_prefix = min(
                measurement_count,
                int(cached_signatures.shape[0]),
            )
            prefix_matches = np.array_equal(
                signatures[:matching_prefix],
                cached_signatures[:matching_prefix],
            )
        else:
            prefix_matches = False

        if (
            cached is None
            or cached_signatures is None
            or cached.shape[1] != patch_count
            or not prefix_matches
        ):
            cached = np.full(
                (measurement_count, patch_count),
                np.nan,
                dtype=float,
            )
            cached_signatures = signatures.copy()
        elif measurement_count > int(cached.shape[0]):
            added_rows = measurement_count - int(cached.shape[0])
            cached = np.vstack(
                [
                    cached,
                    np.full((added_rows, patch_count), np.nan, dtype=float),
                ]
            )
            cached_signatures = signatures.copy()

        self._structural_rj_response_cache = cached
        self._structural_rj_response_cache_signatures = cached_signatures
        return cached[:measurement_count]

    def _structural_rj_evaluate_response_columns(
        self,
        data: MeasurementData,
        patch_indices: NDArray[np.int64],
        *,
        row_start: int,
    ) -> NDArray[np.float64]:
        """Evaluate one batched suffix for selected global surface patches."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        indices = np.unique(
            np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        )
        start = max(0, int(row_start))
        measurement_count = int(data.z_k.size)
        if start > measurement_count:
            raise ValueError("row_start exceeds the measurement history.")
        if np.any(indices < 0) or np.any(indices >= patches.patch_count):
            raise ValueError("rj_mh response request contains an invalid patch.")
        if indices.size == 0 or start == measurement_count:
            return np.zeros(
                (measurement_count - start, indices.size),
                dtype=float,
            )
        fe_indices = np.asarray(data.fe_indices, dtype=np.int64)[start:]
        pb_indices = np.asarray(data.pb_indices, dtype=np.int64)[start:]
        responses = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=np.asarray(
                data.detector_positions,
                dtype=float,
            )[start:],
            sources=np.asarray(
                patches.centers_xyz,
                dtype=float,
            )[indices],
            strengths=np.ones(indices.size, dtype=float),
            live_times=np.asarray(data.live_times, dtype=float)[start:],
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            source_scale=self._measurement_source_scale_vector(
                fe_indices,
                pb_indices,
            ),
        )
        response_array = np.asarray(responses, dtype=float)
        expected_shape = (measurement_count - start, int(indices.size))
        if response_array.shape != expected_shape:
            raise ValueError("rj_mh response columns have an unexpected shape.")
        if not np.all(np.isfinite(response_array)):
            raise ValueError("rj_mh response columns must be finite.")
        self._structural_rj_response_evaluation_batches += 1
        self._structural_rj_response_evaluated_cells += int(
            response_array.size
        )
        touched_mask = self._structural_rj_response_touched_mask
        if touched_mask is None or touched_mask.size != patches.patch_count:
            touched_mask = np.zeros(patches.patch_count, dtype=bool)
            self._structural_rj_response_touched_mask = touched_mask
        touched_mask[indices] = True
        return response_array

    def _structural_rj_ensure_response_columns(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
        patch_indices: NDArray[np.int64],
    ) -> None:
        """Fill only missing cached row suffixes for required patch columns."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        response_array = np.asarray(response_dictionary, dtype=float)
        expected_shape = (int(data.z_k.size), int(patches.patch_count))
        if response_array.shape != expected_shape:
            raise ValueError("rj_mh response dictionary has an unexpected shape.")
        required = np.unique(
            np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        )
        if required.size == 0:
            return
        if np.any(required < 0) or np.any(required >= patches.patch_count):
            raise ValueError("rj_mh response request contains an invalid patch.")

        finite = np.isfinite(response_array[:, required])
        valid_prefixes = np.sum(finite, axis=0).astype(
            np.int64,
            copy=False,
        )
        row_indices = np.arange(response_array.shape[0])[:, None]
        prefix_mask = row_indices < valid_prefixes[None, :]
        if not np.array_equal(finite, prefix_mask):
            raise ValueError(
                "rj_mh response cache contains a non-prefix validity pattern."
            )
        for valid_prefix in np.unique(valid_prefixes).tolist():
            if int(valid_prefix) >= response_array.shape[0]:
                continue
            selected = required[valid_prefixes == int(valid_prefix)]
            evaluated = self._structural_rj_evaluate_response_columns(
                data,
                selected,
                row_start=int(valid_prefix),
            )
            response_array[int(valid_prefix) :, selected] = evaluated

    def _structural_rj_response_dictionary(
        self,
        data: MeasurementData,
        patch_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return a prefix-cached response matrix, eagerly filling requested columns."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        response_dictionary = self._structural_rj_prepare_response_cache(data)
        requested = (
            np.arange(patches.patch_count, dtype=np.int64)
            if patch_indices is None
            else np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        )
        self._structural_rj_ensure_response_columns(
            data,
            response_dictionary,
            requested,
        )
        return response_dictionary

    def _structural_rj_group_arrays(
        self,
        particle_indices: NDArray[np.int64],
        cardinality: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """Return canonical patch, strength, and background arrays for one K."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        source_count = max(0, int(cardinality))
        patch_rows: list[NDArray[np.int64]] = []
        strength_rows: list[NDArray[np.float64]] = []
        backgrounds = np.empty(indices.size, dtype=float)
        for row, particle_index in enumerate(indices.tolist()):
            state = self.continuous_particles[int(particle_index)].state
            if int(state.num_sources) != source_count:
                raise ValueError("rj_mh particle group mixes cardinalities.")
            patch_rows.append(self._canonicalize_structural_rj_state(state))
            strength_rows.append(
                np.asarray(state.strengths, dtype=float).reshape(source_count)
            )
            backgrounds[row] = float(state.background)
        if source_count == 0:
            patch_sets = np.zeros((indices.size, 0), dtype=np.int64)
            strengths = np.zeros((indices.size, 0), dtype=float)
        else:
            patch_sets = np.vstack(patch_rows).astype(np.int64, copy=False)
            strengths = np.vstack(strength_rows).astype(float, copy=False)
        if strengths.size and not np.all(self._strength_prior.in_support(strengths)):
            raise ValueError(
                "rj_mh particle strength lies outside the configured prior."
            )
        return patch_sets, strengths, backgrounds

    @staticmethod
    def _structural_rj_lambda_from_arrays(
        response_dictionary: NDArray[np.float64],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
        backgrounds: NDArray[np.float64],
        live_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts for an equal-cardinality state batch."""
        responses = np.asarray(response_dictionary, dtype=float)
        patches = np.asarray(patch_sets, dtype=np.int64)
        strength_array = np.asarray(strengths, dtype=float)
        background_array = np.asarray(backgrounds, dtype=float).reshape(-1)
        if patches.ndim != 2 or strength_array.shape != patches.shape:
            raise ValueError("rj_mh patch and strength arrays must share P x K.")
        if background_array.size != patches.shape[0]:
            raise ValueError("rj_mh background array must contain P entries.")
        background_counts = (
            np.asarray(live_times, dtype=float)[:, None]
            * background_array[None, :]
        )
        if patches.shape[1] == 0:
            return background_counts
        selected = responses[:, patches]
        source_counts = np.einsum(
            "mpk,pk->mp",
            selected,
            strength_array,
            optimize=True,
        )
        return background_counts + source_counts

    def _structural_rj_group_log_likelihood(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
        backgrounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one equal-cardinality group with the shared PF likelihood."""
        self._structural_rj_ensure_response_columns(
            data,
            response_dictionary,
            patch_sets,
        )
        lambda_total = self._structural_rj_lambda_from_arrays(
            response_dictionary,
            patch_sets,
            strengths,
            backgrounds,
            data.live_times,
        )
        return self._structural_count_log_likelihood_matrix_np(data, lambda_total)

    def _sample_structural_rj_unused_indices(
        self,
        occupied_sets: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Sample area-weighted unused patches for a same-cardinality batch."""
        surface_prior = self._structural_rj_surface_prior
        if surface_prior is None:
            raise RuntimeError("rj_mh surface prior is unavailable.")
        occupied = np.asarray(occupied_sets, dtype=np.int64)
        if occupied.ndim != 2:
            raise ValueError("occupied_sets must have shape P x K.")
        if occupied.shape[1] >= surface_prior.dictionary_size:
            raise ValueError("rj_mh birth has no unused surface patch.")
        sample_count = int(occupied.shape[0])
        sampled = self._random_generator.choice(
            surface_prior.dictionary_size,
            size=sample_count,
            p=surface_prior.area_masses,
        ).astype(np.int64, copy=False)
        if occupied.shape[1] == 0 or sample_count == 0:
            return sampled
        rejected = np.any(occupied == sampled[:, None], axis=1)
        while np.any(rejected):
            sampled[rejected] = self._random_generator.choice(
                surface_prior.dictionary_size,
                size=int(np.count_nonzero(rejected)),
                p=surface_prior.area_masses,
            )
            rejected = np.any(occupied == sampled[:, None], axis=1)
        return sampled

    @staticmethod
    def _structural_rj_insert_values(
        current_sets: NDArray[np.int64],
        current_values: NDArray[np.float64],
        new_indices: NDArray[np.int64],
        new_values: NDArray[np.float64],
        *,
        dictionary_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Insert one indexed scalar per row while preserving canonical order."""
        sets = np.asarray(current_sets, dtype=np.int64)
        values = np.asarray(current_values, dtype=float)
        indices = np.asarray(new_indices, dtype=np.int64).reshape(-1)
        additions = np.asarray(new_values, dtype=float).reshape(-1)
        if values.shape != sets.shape:
            raise ValueError("current_values must match current_sets.")
        if indices.size != sets.shape[0] or additions.size != sets.shape[0]:
            raise ValueError("new_indices and new_values must contain P entries.")
        proposed_sets = add_surface_indices(
            sets,
            indices,
            dictionary_size=dictionary_size,
        )
        combined_indices = np.concatenate([sets, indices[:, None]], axis=1)
        combined_values = np.concatenate([values, additions[:, None]], axis=1)
        order = np.argsort(combined_indices, axis=1, kind="stable")
        proposed_values = np.take_along_axis(combined_values, order, axis=1)
        return proposed_sets, np.asarray(proposed_values, dtype=float)

    @staticmethod
    def _structural_rj_remove_values(
        current_sets: NDArray[np.int64],
        current_values: NDArray[np.float64],
        columns: NDArray[np.int64],
        *,
        dictionary_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Remove one indexed scalar per row from canonical state arrays."""
        sets = np.asarray(current_sets, dtype=np.int64)
        values = np.asarray(current_values, dtype=float)
        removed_columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if values.shape != sets.shape:
            raise ValueError("current_values must match current_sets.")
        if removed_columns.size != sets.shape[0]:
            raise ValueError("columns must contain one entry per row.")
        proposed_sets = remove_surface_columns(
            sets,
            removed_columns,
            dictionary_size=dictionary_size,
        )
        keep = np.arange(sets.shape[1])[None, :] != removed_columns[:, None]
        proposed_values = values[keep].reshape(sets.shape[0], sets.shape[1] - 1)
        return proposed_sets, np.asarray(proposed_values, dtype=float)

    def _commit_structural_rj_states(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
    ) -> int:
        """Commit accepted canonical state arrays after batched RJ/MH scoring."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
        sets = np.asarray(patch_sets, dtype=np.int64)
        strength_array = np.asarray(strengths, dtype=float)
        if (
            acceptance.size != indices.size
            or sets.shape[0] != indices.size
            or strength_array.shape != sets.shape
        ):
            raise ValueError("rj_mh commit arrays must share one particle axis.")
        accepted_rows = np.flatnonzero(acceptance)
        cardinality = int(sets.shape[1])
        for row in accepted_rows.tolist():
            state = self.continuous_particles[int(indices[row])].state
            state.num_sources = cardinality
            state.positions = np.asarray(
                patches.centers_xyz[sets[row]],
                dtype=float,
            ).reshape(cardinality, 3)
            state.strengths = strength_array[row].copy()
            # Background is not proposed by this kernel and must stay fixed.
            state.ages = np.zeros(cardinality, dtype=int)
            state.support_scores = np.zeros(cardinality, dtype=float)
            state.tentative_sources = np.zeros(cardinality, dtype=bool)
            state.verification_fail_streaks = np.zeros(cardinality, dtype=int)
        return int(accepted_rows.size)

    def _apply_structural_rj_birth_death(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> tuple[int, int]:
        """Apply one exact batched birth/death RJ-MH attempt per selected particle."""
        surface_prior = self._structural_rj_surface_prior
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_move_probabilities
        if (
            surface_prior is None
            or cardinality_prior is None
            or move_probabilities is None
        ):
            raise RuntimeError("rj_mh priors or move probabilities are unavailable.")
        particle_count = len(self.continuous_particles)
        attempt = self._random_generator.random(particle_count) < float(
            self.config.structural_rj_move_probability
        )
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        accepted_births = 0
        accepted_deaths = 0
        attempted_births = 0
        attempted_deaths = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            if group_indices.size == 0:
                continue
            birth_probability, _ = move_probabilities.probabilities(
                int(cardinality)
            )
            birth_move = (
                self._random_generator.random(group_indices.size)
                < float(birth_probability)
            )
            for is_birth in (True, False):
                selected_rows = np.flatnonzero(birth_move == is_birth)
                if selected_rows.size == 0:
                    continue
                selected_indices = group_indices[selected_rows]
                if is_birth:
                    attempted_births += int(selected_indices.size)
                else:
                    attempted_deaths += int(selected_indices.size)
                patch_sets, strengths, backgrounds = (
                    self._structural_rj_group_arrays(
                        selected_indices,
                        int(cardinality),
                    )
                )
                base_ll = self._structural_rj_group_log_likelihood(
                    data,
                    response_dictionary,
                    patch_sets,
                    strengths,
                    backgrounds,
                )
                if is_birth:
                    new_patch_indices = (
                        self._sample_structural_rj_unused_indices(patch_sets)
                    )
                    new_strengths = np.asarray(
                        self._strength_prior.sample(
                            selected_indices.size,
                            rng=self._random_generator,
                        ),
                        dtype=float,
                    )
                    proposed_sets, proposed_strengths = (
                        self._structural_rj_insert_values(
                            patch_sets,
                            strengths,
                            new_patch_indices,
                            new_strengths,
                            dictionary_size=surface_prior.dictionary_size,
                        )
                    )
                    proposed_ll = self._structural_rj_group_log_likelihood(
                        data,
                        response_dictionary,
                        proposed_sets,
                        proposed_strengths,
                        backgrounds,
                    )
                    position_log_proposal = (
                        conditional_birth_surface_log_probability(
                            surface_prior,
                            patch_sets,
                            new_patch_indices,
                        )
                    )
                    strength_log_density = np.asarray(
                        self._strength_prior.log_prob(new_strengths),
                        dtype=float,
                    )
                    log_ratio = birth_log_acceptance_ratio(
                        current_surface_sets=patch_sets,
                        birth_surface_indices=new_patch_indices,
                        log_likelihood_ratio=proposed_ll - base_ll,
                        surface_prior=surface_prior,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=move_probabilities,
                        log_strength_prior_density=strength_log_density,
                        log_forward_position_proposal=position_log_proposal,
                        log_forward_strength_proposal=strength_log_density,
                        log_reverse_death_index_probability=(
                            uniform_death_index_log_probability(
                                int(cardinality) + 1
                            )
                        ),
                    )
                    accepted = np.log(
                        self._random_generator.random(selected_indices.size)
                    ) < np.minimum(log_ratio, 0.0)
                    accepted_births += self._commit_structural_rj_states(
                        selected_indices,
                        accepted,
                        proposed_sets,
                        proposed_strengths,
                    )
                    for row in np.flatnonzero(accepted).tolist():
                        state = self.continuous_particles[
                            int(selected_indices[row])
                        ].state
                        source_column = int(
                            np.flatnonzero(
                                proposed_sets[row] == new_patch_indices[row]
                            )[0]
                        )
                        self._record_source_event(
                            "source_birth_accepted",
                            state,
                            source_column,
                            reason="rj_mh_birth",
                            extra={
                                "delta_ll": float(proposed_ll[row] - base_ll[row]),
                                "log_acceptance_ratio": float(log_ratio[row]),
                                "surface_patch_index": int(
                                    new_patch_indices[row]
                                ),
                            },
                        )
                    continue

                death_columns = self._random_generator.integers(
                    0,
                    int(cardinality),
                    size=selected_indices.size,
                    dtype=np.int64,
                )
                row_indices = np.arange(selected_indices.size, dtype=np.int64)
                removed_patch_indices = patch_sets[row_indices, death_columns]
                removed_strengths = strengths[row_indices, death_columns]
                proposed_sets, proposed_strengths = (
                    self._structural_rj_remove_values(
                        patch_sets,
                        strengths,
                        death_columns,
                        dictionary_size=surface_prior.dictionary_size,
                    )
                )
                proposed_ll = self._structural_rj_group_log_likelihood(
                    data,
                    response_dictionary,
                    proposed_sets,
                    proposed_strengths,
                    backgrounds,
                )
                removed_strength_log_density = np.asarray(
                    self._strength_prior.log_prob(removed_strengths),
                    dtype=float,
                )
                reverse_position_log_proposal = (
                    conditional_birth_surface_log_probability(
                        surface_prior,
                        proposed_sets,
                        removed_patch_indices,
                    )
                )
                log_ratio = death_log_acceptance_ratio(
                    current_surface_sets=patch_sets,
                    death_columns=death_columns,
                    log_likelihood_ratio=proposed_ll - base_ll,
                    surface_prior=surface_prior,
                    cardinality_prior=cardinality_prior,
                    move_probabilities=move_probabilities,
                    log_removed_strength_prior_density=(
                        removed_strength_log_density
                    ),
                    log_forward_death_index_probability=(
                        uniform_death_index_log_probability(int(cardinality))
                    ),
                    log_reverse_position_proposal=(
                        reverse_position_log_proposal
                    ),
                    log_reverse_strength_proposal=(
                        removed_strength_log_density
                    ),
                )
                accepted = np.log(
                    self._random_generator.random(selected_indices.size)
                ) < np.minimum(log_ratio, 0.0)
                for row in np.flatnonzero(accepted).tolist():
                    state = self.continuous_particles[
                        int(selected_indices[row])
                    ].state
                    self._record_source_event(
                        "source_removed",
                        state,
                        int(death_columns[row]),
                        reason="rj_mh_death",
                        extra={
                            "delta_ll": float(proposed_ll[row] - base_ll[row]),
                            "log_acceptance_ratio": float(log_ratio[row]),
                            "surface_patch_index": int(
                                removed_patch_indices[row]
                            ),
                        },
                    )
                accepted_deaths += self._commit_structural_rj_states(
                    selected_indices,
                    accepted,
                    proposed_sets,
                    proposed_strengths,
                )
        self._structural_rj_move_counts.update(
            {
                "birth_attempted": int(attempted_births),
                "birth_accepted": int(accepted_births),
                "death_attempted": int(attempted_deaths),
                "death_accepted": int(accepted_deaths),
            }
        )
        return accepted_births, accepted_deaths

    def _apply_structural_rj_position_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Relocate sources with an exact area-prior-cancelling MH proposal."""
        surface_prior = self._structural_rj_surface_prior
        if surface_prior is None:
            raise RuntimeError("rj_mh surface prior is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            relocated_strengths = strengths[rows, source_columns]
            old_patch_indices = patch_sets[rows, source_columns]
            reduced_sets, reduced_strengths = self._structural_rj_remove_values(
                patch_sets,
                strengths,
                source_columns,
                dictionary_size=surface_prior.dictionary_size,
            )
            new_patch_indices = self._sample_structural_rj_unused_indices(
                reduced_sets
            )
            proposed_sets, proposed_strengths = (
                self._structural_rj_insert_values(
                    reduced_sets,
                    reduced_strengths,
                    new_patch_indices,
                    relocated_strengths,
                    dictionary_size=surface_prior.dictionary_size,
                )
            )
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                proposed_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = proposed_ll - base_ll
            accepted = (
                np.log(self._random_generator.random(particle_indices.size))
                < np.minimum(log_ratio, 0.0)
            ) & (new_patch_indices != old_patch_indices)
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                proposed_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "global_position_attempted": int(attempted_count),
                "global_position_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_structural_rj_local_position_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Relocate sources to adjacent free patches with exact local MH."""
        surface_prior = self._structural_rj_surface_prior
        adjacency = self._structural_rj_surface_adjacency
        if surface_prior is None or adjacency is None:
            raise RuntimeError(
                "rj_mh surface prior or adjacency is unavailable."
            )
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [
                particle.state.num_sources
                for particle in self.continuous_particles
            ],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(
                self.config.structural_rj_local_position_move_probability
            )
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        movable_count = 0
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = (
                self._structural_rj_group_arrays(
                    particle_indices,
                    int(cardinality),
                )
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            old_patch_indices = patch_sets[rows, source_columns]
            relocated_strengths = strengths[rows, source_columns]
            reduced_sets, reduced_strengths = (
                self._structural_rj_remove_values(
                    patch_sets,
                    strengths,
                    source_columns,
                    dictionary_size=surface_prior.dictionary_size,
                )
            )
            (
                new_patch_indices,
                forward_degrees,
                movable,
            ) = adjacency.sample_unoccupied_neighbors(
                patch_sets,
                source_columns,
                rng=self._random_generator,
            )
            movable_count += int(np.count_nonzero(movable))
            proposed_sets, proposed_strengths = (
                self._structural_rj_insert_values(
                    reduced_sets,
                    reduced_strengths,
                    new_patch_indices,
                    relocated_strengths,
                    dictionary_size=surface_prior.dictionary_size,
                )
            )
            reverse_degrees = adjacency.available_neighbor_degrees(
                new_patch_indices,
                reduced_sets,
            )
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                proposed_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = local_position_log_acceptance_ratio(
                old_surface_indices=old_patch_indices,
                new_surface_indices=new_patch_indices,
                forward_available_degrees=forward_degrees,
                reverse_available_degrees=reverse_degrees,
                log_likelihood_ratio=proposed_ll - base_ll,
                surface_prior=surface_prior,
            )
            accepted = (
                movable
                & (new_patch_indices != old_patch_indices)
                & (
                    np.log(
                        self._random_generator.random(
                            particle_indices.size
                        )
                    )
                    < np.minimum(log_ratio, 0.0)
                )
            )
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                proposed_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "local_position_attempted": int(attempted_count),
                "local_position_movable": int(movable_count),
                "local_position_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_structural_rj_strength_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Update strengths with an exact prior-independence MH proposal."""
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_strength_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            proposed_strengths = strengths.copy()
            proposed_values = np.asarray(
                self._strength_prior.sample(
                    particle_indices.size,
                    rng=self._random_generator,
                ),
                dtype=float,
            )
            proposed_strengths[
                np.arange(particle_indices.size),
                source_columns,
            ] = proposed_values
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = proposed_ll - base_ll
            accepted = np.log(
                self._random_generator.random(particle_indices.size)
            ) < np.minimum(log_ratio, 0.0)
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                patch_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "strength_attempted": int(attempted_count),
                "strength_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_exact_structural_rj_moves(
        self,
        evidence_data: MeasurementData,
        *,
        allow_structural_birth_proposals: bool,
    ) -> None:
        """Apply target-preserving RJ/MH rejuvenation without changing PF weights."""
        structural_start = time.perf_counter()
        original_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        self._structural_rj_response_evaluation_batches = 0
        self._structural_rj_response_evaluated_cells = 0
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
        }
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        self._structural_rj_response_touched_mask = np.zeros(
            patches.patch_count,
            dtype=bool,
        )
        response_start = time.perf_counter()
        response_dictionary = self._structural_rj_response_dictionary(
            evidence_data,
            patch_indices=np.zeros(0, dtype=np.int64),
        )
        response_elapsed = time.perf_counter() - response_start
        birth_count = 0
        death_count = 0
        birth_death_elapsed = 0.0
        if allow_structural_birth_proposals:
            move_start = time.perf_counter()
            birth_count, death_count = self._apply_structural_rj_birth_death(
                evidence_data,
                response_dictionary,
            )
            birth_death_elapsed = time.perf_counter() - move_start
        position_start = time.perf_counter()
        position_count = self._apply_structural_rj_position_moves(
            evidence_data,
            response_dictionary,
        )
        position_elapsed = time.perf_counter() - position_start
        local_position_start = time.perf_counter()
        local_position_count = (
            self._apply_structural_rj_local_position_moves(
                evidence_data,
                response_dictionary,
            )
        )
        local_position_elapsed = (
            time.perf_counter() - local_position_start
        )
        strength_start = time.perf_counter()
        strength_count = self._apply_structural_rj_strength_moves(
            evidence_data,
            response_dictionary,
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
        self.last_birth_count += int(birth_count)
        self.last_kill_count += int(death_count)
        self._reset_structural_residual_gate()
        self.last_birth_residual_layer = "not_used_by_rj_mh"
        self.last_structural_timing_s = {
            "total": float(time.perf_counter() - structural_start),
            "response_dictionary": float(response_elapsed),
            "rj_birth_death": float(birth_death_elapsed),
            "rj_position": float(position_elapsed),
            "rj_global_position": float(position_elapsed),
            "rj_local_position": float(local_position_elapsed),
            "rj_strength": float(strength_elapsed),
            "rj_birth_attempted": float(
                self._structural_rj_move_counts["birth_attempted"]
            ),
            "rj_birth_accepted": float(birth_count),
            "rj_death_attempted": float(
                self._structural_rj_move_counts["death_attempted"]
            ),
            "rj_death_accepted": float(death_count),
            "rj_global_position_attempted": float(
                self._structural_rj_move_counts[
                    "global_position_attempted"
                ]
            ),
            "rj_global_position_accepted": float(position_count),
            "rj_position_attempted": float(
                self._structural_rj_move_counts[
                    "global_position_attempted"
                ]
            ),
            "rj_position_accepted": float(position_count),
            "rj_local_position_attempted": float(
                self._structural_rj_move_counts[
                    "local_position_attempted"
                ]
            ),
            "rj_local_position_movable": float(
                self._structural_rj_move_counts[
                    "local_position_movable"
                ]
            ),
            "rj_local_position_accepted": float(local_position_count),
            "rj_strength_attempted": float(
                self._structural_rj_move_counts["strength_attempted"]
            ),
            "rj_strength_accepted": float(strength_count),
            "rj_response_evaluation_batches": float(
                self._structural_rj_response_evaluation_batches
            ),
            "rj_response_evaluated_cells": float(
                self._structural_rj_response_evaluated_cells
            ),
            "rj_response_touched_columns": float(
                np.count_nonzero(self._structural_rj_response_touched_mask)
            ),
            "outer_log_weight_max_abs_diff": float(
                outer_weight_max_abs_diff
            ),
            "outer_log_weight_array_equal": float(
                outer_weight_array_equal
            ),
            "weights_preserved": float(outer_weight_array_equal),
        }
        if not outer_weight_array_equal:
            raise RuntimeError("rj_mh rejuvenation must not alter PF weights.")
        self.align_continuous_labels()

    def apply_structural_moves(
        self,
        evidence_data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None = None,
        allow_structural_birth_proposals: bool = True,
    ) -> None:
        """
        Apply evidence-based birth, death, split, and merge proposals.
        """
        if not self.continuous_particles:
            return
        if not bool(self.config.birth_enable):
            self._reset_structural_residual_gate()
            self.last_structural_timing_s = {
                "total": 0.0,
                "structural_moves_gated": 1.0,
            }
            return
        if self._structural_kernel_is_exact():
            if evidence_data is None or evidence_data.z_k.size == 0:
                self._reset_structural_residual_gate()
                self.last_structural_timing_s = {
                    "total": 0.0,
                    "rj_mh_no_evidence": 1.0,
                    "weights_preserved": 1.0,
                }
                return
            self._apply_exact_structural_rj_moves(
                evidence_data,
                allow_structural_birth_proposals=(
                    allow_structural_birth_proposals
                ),
            )
            return
        timing: dict[str, float] = {
            "total": 0.0,
            "cache": 0.0,
            "birth": 0.0,
            "prune": 0.0,
            "pseudo": 0.0,
            "split": 0.0,
            "merge": 0.0,
            "refresh_weights": 0.0,
            "label": 0.0,
        }
        structural_start = time.perf_counter()
        if evidence_data is not None and evidence_data.z_k.size == 0:
            evidence_data = None
        support_data = evidence_data
        birth_data = evidence_data
        structural_data = evidence_data
        if structural_data is None or structural_data.z_k.size == 0:
            self._reset_structural_residual_gate()
            return
        proposal_enabled = bool(
            allow_structural_birth_proposals and self.config.birth_enable
        )
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
        regular_structural_support_ready = True
        structural_distinct_count: int | None = None
        structural_station_count: int | None = None
        if min_distinct > 1 or min_stations > 1:
            full_support = np.ones(structural_data.z_k.size, dtype=bool)
            distinct_count = self._distinct_supported_view_count(
                structural_data.detector_positions,
                structural_data.fe_indices,
                structural_data.pb_indices,
                full_support,
            )
            station_count = self._distinct_supported_station_count(
                structural_data.detector_positions,
                full_support,
            )
            structural_distinct_count = int(distinct_count)
            structural_station_count = int(station_count)
            self.last_birth_residual_distinct_poses = int(distinct_count)
            self.last_birth_residual_distinct_stations = int(station_count)
            regular_structural_support_ready = (
                distinct_count >= min_distinct and station_count >= min_stations
            )
            if distinct_count < min_distinct or station_count < min_stations:
                self._reset_structural_residual_gate()
                self.last_birth_residual_distinct_poses = int(distinct_count)
                self.last_birth_residual_distinct_stations = int(station_count)
        if self.config.max_sources is None:
            birth_capacity_available = True
        else:
            max_sources = int(self.config.max_sources)
            birth_capacity_available = any(
                int(particle.state.num_sources) < max_sources
                for particle in self.continuous_particles
            )
        birth_proposal = None
        if proposal_enabled and birth_capacity_available:
            if bool(regular_structural_support_ready):
                birth_start = time.perf_counter()
                birth_proposal = self._compute_birth_proposal(
                    birth_data,
                    candidate_positions,
                )
                timing["birth"] += time.perf_counter() - birth_start
            else:
                self._reset_structural_residual_gate()
                if structural_distinct_count is not None:
                    self.last_birth_residual_distinct_poses = int(
                        structural_distinct_count
                    )
                if structural_station_count is not None:
                    self.last_birth_residual_distinct_stations = int(
                        structural_station_count
                    )
        if birth_proposal is not None:
            if len(birth_proposal) == 4:
                birth_probs, birth_kernel_sums, residual_sum, birth_candidates = (
                    birth_proposal
                )
                birth_candidate_counts = None
            else:
                (
                    birth_probs,
                    birth_kernel_sums,
                    residual_sum,
                    birth_candidates,
                    birth_candidate_counts,
                ) = birth_proposal
        else:
            birth_probs = None
            birth_kernel_sums = None
            residual_sum = 0.0
            birth_candidates = None
            birth_candidate_counts = None
        residual_birth_gate_active = (
            birth_proposal is not None
            and bool(self.last_birth_residual_gate_passed)
            and residual_sum > 0.0
        )
        split_candidate_strengths = (
            self._candidate_initial_strengths(
                candidate_count=birth_candidates.shape[0],
                candidate_kernel_sums=birth_kernel_sums,
                residual_sum=residual_sum,
            )
            if birth_candidates is not None and birth_kernel_sums is not None
            else None
        )
        proposal_data = None
        if birth_data is not None and birth_data.z_k.size:
            proposal_data = birth_data
        elif support_data is not None and support_data.z_k.size:
            proposal_data = support_data
        proposal_matches_support = self._same_measurement_block(
            proposal_data,
            support_data,
        )
        evidence_data = proposal_data
        split_candidates_for_trial = birth_candidates
        split_candidate_counts_for_trial = birth_candidate_counts
        split_candidate_strengths_for_trial = split_candidate_strengths
        max_births = self.config.birth_max_per_update
        births_remaining = None if max_births is None else max(0, int(max_births))
        any_moved = False
        moved_indices: set[int] = set()
        likelihood_unchanged_indices: set[int] = set()
        refresh_reference_ll: dict[int, float] = {}
        has_support_data = support_data is not None and support_data.z_k.size > 0
        support_cache: dict[
            int,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.bool_],
                NDArray[np.float64] | None,
                float,
                NDArray[np.bool_],
            ],
        ] = {}
        structural_proposal_indices: set[int] | None = None
        topk_structural = self.config.structural_proposal_topk_particles
        if topk_structural is not None:
            structural_proposal_indices = self._select_structural_proposal_indices(
                int(topk_structural),
            )
        if residual_birth_gate_active and bool(
            self.config.birth_residual_expand_structural_particles
        ):
            structural_proposal_indices = self._select_structural_proposal_indices(
                self.config.birth_residual_expanded_structural_topk_particles,
                require_birth_capacity=True,
            )
            if structural_proposal_indices is not None:
                self.last_birth_structural_eligible = len(structural_proposal_indices)
        if has_support_data:
            cache_start = time.perf_counter()
            grouped, fallback_indices = self._particle_indices_by_source_count()
            for source_count, particle_indices in grouped.items():
                if source_count <= 0 or not particle_indices:
                    continue
                (
                    k_tensor_group,
                    background_counts_group,
                    strengths_group,
                ) = self._unit_kernel_tensor_for_particle_group(
                    support_data,
                    particle_indices,
                    source_count,
                )
                lambda_m_group = k_tensor_group * strengths_group[None, :, :]
                lambda_total_group = background_counts_group + np.sum(
                    lambda_m_group,
                    axis=2,
                )
                base_ll_group, delta_ll_group = (
                    self._log_likelihood_and_delta_remove_group(
                        support_data,
                        lambda_total_group,
                        lambda_m_group,
                    )
                )
                prune_allowed_group = self._source_prune_allowed_mask_group(
                    support_data,
                    lambda_m_group,
                    lambda_total_group,
                    delta_ll=delta_ll_group,
                )
                source_positions_group = np.stack(
                    [
                        np.asarray(
                            self.continuous_particles[idx].state.positions[
                                :source_count
                            ],
                            dtype=float,
                        )
                        for idx in particle_indices
                    ],
                    axis=0,
                )
                detector_clear_group = (
                    self._source_detector_exclusion_mask_group(
                        source_positions_group,
                        structural_data,
                    )
                )
                for row_idx, particle_idx in enumerate(particle_indices):
                    support_cache[int(particle_idx)] = (
                        lambda_m_group[:, row_idx, :],
                        lambda_total_group[:, row_idx],
                        delta_ll_group[row_idx],
                        prune_allowed_group[row_idx],
                        k_tensor_group[:, row_idx, :],
                        float(base_ll_group[row_idx]),
                        detector_clear_group[row_idx],
                    )
            for particle_idx in fallback_indices:
                st = self.continuous_particles[particle_idx].state
                if st.num_sources <= 0:
                    continue
                lambda_m, lambda_total = self._lambda_components(st, support_data)
                delta_ll = self._structural_delta_log_likelihood_remove(
                    support_data,
                    lambda_total,
                    lambda_m,
                )
                prune_allowed = self._source_prune_allowed_mask(
                    st,
                    support_data,
                    lambda_m=lambda_m,
                    lambda_total=lambda_total,
                    delta_ll=delta_ll,
                )
                support_cache[int(particle_idx)] = (
                    lambda_m,
                    lambda_total,
                    delta_ll,
                    prune_allowed,
                    None,
                    self._structural_count_log_likelihood_np(
                        support_data,
                        lambda_total,
                    ),
                    self._source_detector_exclusion_mask(
                        st,
                        structural_data,
                    ),
                )
            timing["cache"] += time.perf_counter() - cache_start

        for particle_idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            allow_structural_proposal = (
                structural_proposal_indices is None
                or int(particle_idx) in structural_proposal_indices
            )
            has_support = has_support_data
            moved = False
            if st.num_sources > 0:
                st.ages = st.ages + 1
            lambda_m = None
            lambda_total = None
            cached_prune_allowed = None
            cached_unit_counts = None
            cached_base_log_likelihood = None
            cached_detector_clear = None
            pseudo_response_snapshot: (
                tuple[
                    NDArray[np.float64],
                    NDArray[np.float64],
                    float,
                    NDArray[np.bool_],
                ]
                | None
            ) = None
            if has_support and st.num_sources > 0:
                cached_support = support_cache.get(int(particle_idx))
                if (
                    cached_support is not None
                    and cached_support[2].size == st.num_sources
                    and cached_support[3].size == st.num_sources
                ):
                    (
                        lambda_m,
                        lambda_total,
                        delta_ll,
                        cached_prune_allowed,
                        cached_unit_counts,
                        cached_base_log_likelihood,
                        cached_detector_clear,
                    ) = cached_support
                else:
                    lambda_m, lambda_total = self._lambda_components(st, support_data)
                    delta_ll = self._structural_delta_log_likelihood_remove(
                        support_data,
                        lambda_total,
                        lambda_m,
                    )
                alpha = float(self.config.support_ema_alpha)
                st.support_scores = (1.0 - alpha) * st.support_scores + alpha * delta_ll
            if evidence_data is not None and evidence_data.z_k.size > 0:
                if (
                    proposal_matches_support
                    and cached_base_log_likelihood is not None
                    and np.asarray(lambda_total).shape == (int(evidence_data.z_k.size),)
                ):
                    refresh_reference_ll[int(particle_idx)] = float(
                        cached_base_log_likelihood
                    )
                else:
                    refresh_reference_ll[int(particle_idx)] = (
                        self._trial_log_likelihood(
                            st,
                            evidence_data,
                        )
                    )
            if has_support and st.num_sources > 0:
                pseudo_start = time.perf_counter()
                pseudo_response_snapshot = (
                    np.asarray(st.positions[: st.num_sources], dtype=float).copy(),
                    np.asarray(st.strengths[: st.num_sources], dtype=float).copy(),
                    float(st.background),
                    self._active_source_mask(st).copy(),
                )
                pseudo_moved = self._verify_pseudo_sources_for_state(
                    st,
                    support_data,
                    suppress_prune=False,
                    cached_lambda_m=lambda_m,
                    cached_lambda_total=lambda_total,
                    cached_delta_ll=delta_ll,
                    cached_prune_allowed=cached_prune_allowed,
                )
                timing["pseudo"] += time.perf_counter() - pseudo_start
                moved = moved or pseudo_moved
                if lambda_m is not None and lambda_m.shape[1] != st.num_sources:
                    lambda_m = None
                    lambda_total = None
                    delta_ll = None
                    cached_prune_allowed = None
                    cached_unit_counts = None
                    cached_base_log_likelihood = None
                    cached_detector_clear = None
            if st.num_sources > 0 and has_support:
                prune_start = time.perf_counter()
                kill_mask = np.ones(st.num_sources, dtype=bool)
                if (
                    lambda_m is None
                    or lambda_total is None
                    or lambda_m.shape
                    != (int(support_data.z_k.size), int(st.num_sources))
                ):
                    lambda_m, lambda_total = self._lambda_components(
                        st,
                        support_data,
                    )
                if delta_ll is None or np.asarray(delta_ll).shape != (
                    int(st.num_sources),
                ):
                    delta_ll = self._structural_delta_log_likelihood_remove(
                        support_data,
                        lambda_total,
                        lambda_m,
                    )
                if cached_prune_allowed is not None and cached_prune_allowed.shape == (
                    int(st.num_sources),
                ):
                    prune_allowed = np.asarray(cached_prune_allowed, dtype=bool)
                else:
                    prune_allowed = self._source_prune_allowed_mask(
                        st,
                        support_data,
                        lambda_m=lambda_m,
                        lambda_total=lambda_total,
                        delta_ll=delta_ll,
                    )
                if (
                    cached_detector_clear is not None
                    and cached_detector_clear.shape == (int(st.num_sources),)
                ):
                    physical_prior_violation = ~np.asarray(
                        cached_detector_clear,
                        dtype=bool,
                    )
                else:
                    physical_prior_violation = (
                        ~self._source_detector_exclusion_mask(
                            st,
                            structural_data,
                        )
                    )
                evidence_candidates = np.flatnonzero(prune_allowed)
                candidate_indices = np.zeros(0, dtype=np.int64)
                if evidence_candidates.size and np.random.rand() < float(
                    self.config.p_kill
                ):
                    physical_evidence_candidates = evidence_candidates[
                        physical_prior_violation[evidence_candidates]
                    ]
                    candidate_indices = (
                        physical_evidence_candidates
                        if physical_evidence_candidates.size
                        else evidence_candidates
                    )
                if candidate_indices.size:
                    candidate_losses = np.asarray(delta_ll, dtype=float)[
                        candidate_indices
                    ]
                    remove_idx = int(
                        candidate_indices[int(np.argmin(candidate_losses))]
                    )
                    kill_mask[remove_idx] = False
                if not np.all(kill_mask):
                    remove_idx = int(np.flatnonzero(~kill_mask)[0])
                    self.last_kill_count += 1
                    self._record_source_event(
                        "source_removed",
                        st,
                        remove_idx,
                        reason=(
                            "leave_one_out_evidence_physical_prior_violation"
                            if bool(physical_prior_violation[remove_idx])
                            else "leave_one_out_evidence"
                        ),
                        extra={
                            "delta_ll_loss": float(delta_ll[remove_idx]),
                            "complexity_gain": float(
                                self._bic_model_penalty(
                                    int(support_data.z_k.size),
                                    int(self.config.source_prune_bic_penalty_params),
                                )
                            ),
                        },
                    )
                    st.positions = st.positions[kill_mask]
                    st.strengths = st.strengths[kill_mask]
                    st.ages = st.ages[kill_mask]
                    st.support_scores = st.support_scores[kill_mask]
                    st.tentative_sources = st.tentative_sources[kill_mask]
                    st.verification_fail_streaks = st.verification_fail_streaks[
                        kill_mask
                    ]
                    st.num_sources = st.positions.shape[0]
                    moved = True
                timing["prune"] += time.perf_counter() - prune_start

            can_try_split = (
                proposal_enabled
                and allow_structural_proposal
                and st.num_sources > 0
                and proposal_data is not None
                and proposal_data.z_k.size
            )
            if can_try_split:
                split_start = time.perf_counter()
                split_moved = False
                if (
                    self.config.max_sources is None
                    or st.num_sources < self.config.max_sources
                ):
                    try_residual_split = np.random.rand() < float(
                        self.config.split_prob
                    )
                    if try_residual_split:
                        split_trial, split_delta = (
                            self._best_residual_guided_split_trial(
                                st,
                                proposal_data,
                                split_candidates_for_trial,
                                split_candidate_strengths_for_trial,
                                candidate_unit_counts=split_candidate_counts_for_trial,
                                cached_existing_unit_counts=(
                                    cached_unit_counts
                                    if proposal_matches_support
                                    else None
                                ),
                            )
                        )
                        if (
                            split_trial is not None
                            and split_delta
                            >= self._structural_acceptance_threshold(
                                base_threshold=float(
                                    self.config.split_delta_ll_threshold
                                ),
                                complexity_penalty=float(
                                    self.config.split_complexity_penalty
                                ),
                            )
                        ):
                            old_count = int(st.num_sources)
                            for idx in range(old_count, int(split_trial.num_sources)):
                                self._record_source_event(
                                    "source_birth_accepted",
                                    split_trial,
                                    int(idx),
                                    reason="residual_guided_split",
                                    extra={"delta_ll": float(split_delta)},
                                )
                            self._replace_particle_state_from_trial(st, split_trial)
                            split_moved = True
                            moved = True
                    if not split_moved and np.random.rand() < float(
                        self.config.split_prob
                    ):
                        candidates = np.where(
                            st.strengths >= float(self.config.split_strength_min)
                        )[0]
                        if candidates.size > 0:
                            idx = int(np.random.choice(candidates))
                            if st.ages[idx] > int(self.config.min_age_to_split):
                                split_lambda_m, split_lambda_total = (
                                    self._lambda_components(
                                        st,
                                        proposal_data,
                                    )
                                )
                                delta = np.random.normal(
                                    scale=float(self.config.split_position_sigma),
                                    size=3,
                                )
                                split_positions = (
                                    self._project_positions_to_source_prior(
                                        np.vstack(
                                            [
                                                st.positions[idx] + delta,
                                                st.positions[idx] - delta,
                                            ]
                                        )
                                    )
                                )
                                s1 = split_positions[0]
                                s2 = split_positions[1]
                                if np.linalg.norm(s1 - s2) >= 0.5 * float(
                                    self.config.birth_min_sep_m
                                ):
                                    u_min = float(self.config.split_strength_min_frac)
                                    u_max = float(self.config.split_strength_max_frac)
                                    u_low, u_high = (
                                        (u_min, u_max)
                                        if u_min <= u_max
                                        else (u_max, u_min)
                                    )
                                    u = np.random.uniform(u_low, u_high)
                                    q1 = float(st.strengths[idx]) * float(u)
                                    q2 = float(st.strengths[idx]) * float(1.0 - u)
                                    lam_new = expected_counts_per_source(
                                        kernel=self.continuous_kernel,
                                        isotope=self.isotope,
                                        detector_positions=proposal_data.detector_positions,
                                        sources=np.vstack([s1, s2]),
                                        strengths=np.array([q1, q2], dtype=float),
                                        live_times=proposal_data.live_times,
                                        fe_indices=proposal_data.fe_indices,
                                        pb_indices=proposal_data.pb_indices,
                                        source_scale=self._measurement_source_scale_vector(
                                            proposal_data.fe_indices,
                                            proposal_data.pb_indices,
                                        ),
                                    )
                                    lambda_new = (
                                        split_lambda_total
                                        - split_lambda_m[:, idx]
                                        + np.sum(lam_new, axis=1)
                                    )
                                    delta_ll = self._structural_count_log_likelihood_np(
                                        proposal_data,
                                        lambda_new,
                                    ) - self._structural_count_log_likelihood_np(
                                        proposal_data,
                                        split_lambda_total,
                                    )
                                    split_threshold = (
                                        self._structural_acceptance_threshold(
                                            base_threshold=float(
                                                self.config.split_delta_ll_threshold
                                            ),
                                            complexity_penalty=float(
                                                self.config.split_complexity_penalty
                                            ),
                                        )
                                    )
                                    if (
                                        delta_ll >= split_threshold
                                        and np.log(np.random.rand()) < delta_ll
                                    ):
                                        self._record_source_event(
                                            "source_removed",
                                            st,
                                            int(idx),
                                            reason="random_split_replaced_parent",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        st.positions = np.vstack(
                                            [
                                                st.positions[:idx],
                                                st.positions[idx + 1 :],
                                                s1,
                                                s2,
                                            ]
                                        )
                                        st.strengths = np.concatenate(
                                            [
                                                st.strengths[:idx],
                                                st.strengths[idx + 1 :],
                                                [q1, q2],
                                            ]
                                        )
                                        st.ages = np.concatenate(
                                            [st.ages[:idx], st.ages[idx + 1 :], [0, 0]]
                                        )
                                        st.support_scores = np.concatenate(
                                            [
                                                st.support_scores[:idx],
                                                st.support_scores[idx + 1 :],
                                                [0.0, 0.0],
                                            ]
                                        )
                                        st.tentative_sources = np.concatenate(
                                            [
                                                st.tentative_sources[:idx],
                                                st.tentative_sources[idx + 1 :],
                                                [True, True],
                                            ]
                                        )
                                        st.verification_fail_streaks = np.concatenate(
                                            [
                                                st.verification_fail_streaks[:idx],
                                                st.verification_fail_streaks[idx + 1 :],
                                                [0, 0],
                                            ]
                                        )
                                        st.num_sources = st.positions.shape[0]
                                        self._record_source_event(
                                            "source_birth_accepted",
                                            st,
                                            int(st.num_sources - 2),
                                            reason="random_split_child",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        self._record_source_event(
                                            "source_birth_accepted",
                                            st,
                                            int(st.num_sources - 1),
                                            reason="random_split_child",
                                            extra={"delta_ll": float(delta_ll)},
                                        )
                                        moved = True
                timing["split"] += time.perf_counter() - split_start

            if (
                proposal_enabled
                and allow_structural_proposal
                and st.num_sources >= 2
                and proposal_data is not None
                and proposal_data.z_k.size
                and np.random.rand() < float(self.config.merge_prob)
            ):
                merge_start = time.perf_counter()
                merge_trial, merge_delta = self._best_merge_trial(st, proposal_data)
                timing["merge"] += time.perf_counter() - merge_start
                if merge_trial is not None and merge_delta >= float(
                    self.config.merge_delta_ll_threshold
                ):
                    for idx in range(int(st.num_sources)):
                        self._record_source_event(
                            "source_merge_accepted",
                            st,
                            int(idx),
                            reason="merge_replaced_particle_state",
                            extra={
                                "delta_ll": float(merge_delta),
                                "merged_source_count": int(merge_trial.num_sources),
                            },
                        )
                    self._replace_particle_state_from_trial(st, merge_trial)
                    moved = True

            if (
                proposal_enabled
                and allow_structural_proposal
                and birth_probs is not None
                and birth_kernel_sums is not None
                and birth_candidates is not None
                and residual_sum > 0.0
                and (births_remaining is None or births_remaining > 0)
                and (
                    self.config.max_sources is None
                    or st.num_sources < self.config.max_sources
                )
                and np.random.rand() < float(self.config.p_birth)
            ):
                birth_moved = False
                mp_limit = max(
                    1, int(self.config.birth_matching_pursuit_max_new_sources)
                )
                if mp_limit > 1 and proposal_data is not None:
                    if births_remaining is None:
                        max_new = mp_limit
                    else:
                        max_new = min(mp_limit, max(0, int(births_remaining)))
                    birth_mp_start = time.perf_counter()
                    accepted_births = self._apply_matching_pursuit_births_to_state(
                        st,
                        proposal_data,
                        birth_candidates,
                        max_new_sources=max_new,
                        candidate_unit_counts=birth_candidate_counts,
                    )
                    timing["birth"] += time.perf_counter() - birth_mp_start
                    if accepted_births > 0:
                        self.last_birth_count += int(accepted_births)
                        if births_remaining is not None:
                            births_remaining -= int(accepted_births)
                        birth_moved = True
                        moved = True
                if birth_moved:
                    pass
                else:
                    idx = int(np.random.choice(len(birth_probs), p=birth_probs))
                    denom = float(birth_kernel_sums[idx])
                    q_new = 0.0
                    if denom > 0.0:
                        q_new = (
                            float(self.config.birth_alpha)
                            * residual_sum
                            / denom
                        )
                    q_min = float(self.config.birth_q_min)
                    q_max = float(self.config.birth_q_max)
                    if q_max < q_min:
                        q_min, q_max = q_max, q_min
                    if q_new > 0.0:
                        q_new = float(np.clip(q_new, q_min, q_max))
                    birth_ll_start = time.perf_counter()
                    trial, delta_ll = self._single_residual_birth_trial(
                        st,
                        proposal_data,
                        position=birth_candidates[idx],
                        strength=q_new,
                    )
                    timing["birth"] += time.perf_counter() - birth_ll_start
                    if trial is not None:
                        self._stage_new_birth_metadata(
                            trial,
                            proposal_data,
                            old_count=int(st.num_sources),
                            delta_ll=delta_ll,
                        )
                        self._record_source_event(
                            "source_birth_accepted",
                            trial,
                            int(trial.num_sources - 1),
                            reason="single_residual_birth",
                            extra={"delta_ll": float(delta_ll)},
                        )
                        self._replace_particle_state_from_trial(st, trial)
                        self.last_birth_count += 1
                        if births_remaining is not None:
                            births_remaining -= 1
                        moved = True

            if moved:
                moved_indices.add(int(particle_idx))
                if pseudo_response_snapshot is not None:
                    (
                        positions_before,
                        strengths_before,
                        background_before,
                        active_mask_before,
                    ) = pseudo_response_snapshot
                    active_mask_after = self._active_source_mask(st)
                    if (
                        np.array_equal(
                            np.asarray(st.positions[: st.num_sources], dtype=float),
                            positions_before,
                        )
                        and np.array_equal(
                            np.asarray(st.strengths[: st.num_sources], dtype=float),
                            strengths_before,
                        )
                        and float(st.background) == background_before
                        and np.array_equal(active_mask_after, active_mask_before)
                    ):
                        likelihood_unchanged_indices.add(int(particle_idx))
            any_moved = any_moved or moved

        if any_moved and evidence_data is not None:
            refresh_start = time.perf_counter()
            refresh_lambda_cache = None
            if proposal_matches_support and support_cache:
                refresh_lambda_cache = {
                    int(particle_idx): np.asarray(cached[1], dtype=float)
                    for particle_idx, cached in support_cache.items()
                    if int(particle_idx) not in moved_indices
                    and np.asarray(cached[1]).shape == (int(evidence_data.z_k.size),)
                }
            self.refresh_weights_from_measurements(
                evidence_data,
                lambda_total_by_index=refresh_lambda_cache,
                reference_log_likelihood_by_index=refresh_reference_ll,
                moved_indices=moved_indices,
                likelihood_unchanged_indices=likelihood_unchanged_indices,
            )
            timing["refresh_weights"] += time.perf_counter() - refresh_start
            self._maybe_resample_after_structural_update()
        label_start = time.perf_counter()
        self.align_continuous_labels()
        timing["label"] += time.perf_counter() - label_start
        timing["total"] = time.perf_counter() - structural_start
        self.last_structural_timing_s = {
            key: float(value)
            for key, value in timing.items()
            if float(value) > 0.0 or key == "total"
        }
        if not proposal_enabled:
            self.last_structural_timing_s["birth_proposals_gated"] = 1.0

    def _background_level(self) -> float:
        """Resolve per-isotope background level."""
        level = self.config.background_level
        if isinstance(level, dict):
            return float(level.get(self.isotope, 0.0))
        return float(level)

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return a continuous posterior point estimate over positions and strengths.

        The unconstrained position estimate is the MMSE posterior mean. When the
        source prior is surface-constrained, its nearest feasible projection is
        returned as the constrained squared-error Bayes action.
        """
        if (
            self.config.converge_enable
            and self.is_converged
            and self.frozen_estimate is not None
            and self._convergence_can_freeze()
        ):
            frozen_positions, frozen_strengths = self.frozen_estimate
            return (
                self._project_positions_to_source_prior(frozen_positions),
                np.asarray(frozen_strengths, dtype=float).copy(),
            )
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        if not self._can_use_gpu():
            states = [p.state for p in self.continuous_particles]
            weights = np.asarray(self.continuous_weights, dtype=float)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                weights = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights = weights / weight_sum
            max_sources = max((state.num_sources for state in states), default=0)
            positions = np.zeros((max_sources, 3), dtype=float)
            strengths = np.zeros(max_sources, dtype=float)
            for source_idx in range(max_sources):
                source_weights = []
                source_positions = []
                source_strengths = []
                for weight, state in zip(weights, states):
                    if state.num_sources > source_idx:
                        source_weights.append(float(weight))
                        source_positions.append(state.positions[source_idx])
                        source_strengths.append(float(state.strengths[source_idx]))
                if not source_weights:
                    continue
                w_arr = np.asarray(source_weights, dtype=float)
                w_arr = w_arr / max(float(np.sum(w_arr)), 1e-12)
                pos_arr = np.vstack(source_positions)
                q_arr = np.asarray(source_strengths, dtype=float)
                positions[source_idx] = np.sum(w_arr[:, None] * pos_arr, axis=0)
                strengths[source_idx] = float(np.sum(w_arr * q_arr))
            active = strengths > 0.0
            return (
                self._project_positions_to_source_prior(positions[active]),
                strengths[active],
            )
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        states = [p.state for p in self.continuous_particles]
        positions_t, strengths_t, _, mask_t = gpu_utils.pack_states(
            states, device=device, dtype=dtype
        )
        weights = torch.as_tensor(self.continuous_weights, device=device, dtype=dtype)
        weight_sum = torch.sum(weights)
        if float(weight_sum) <= 0.0:
            weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
        else:
            weights = weights / weight_sum
        w_mask = weights[:, None] * mask_t
        w_sum = torch.sum(w_mask, dim=0)
        w_sum_safe = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        pos_mean = (
            torch.sum(w_mask[:, :, None] * positions_t, dim=0) / w_sum_safe[:, None]
        )
        str_mean = torch.sum(w_mask * strengths_t, dim=0) / w_sum_safe
        positions = pos_mean.detach().cpu().numpy()
        strengths = str_mean.detach().cpu().numpy()
        # Trim zero-strength slots.
        mask = strengths > 0
        positions = self._project_positions_to_source_prior(positions[mask])
        strengths = strengths[mask]
        return positions, strengths
