"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from pf.likelihood import (
    count_log_likelihood,
    delta_log_likelihood_remove,
    delta_log_likelihood_update,
    expected_counts_per_source,
)
from pf.state import IsotopeState
from pf.resampling import systematic_resample


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    max_sources: int | None = None
    resample_threshold: float = 0.5  # relative to N
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: dict[str, float] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | dict[str, float] = 0.0
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
    adapt_cooldown_steps: int = 0
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    roughening_k: float = 0.5
    min_sigma_pos: float = 0.05
    max_sigma_pos: float = 1.5
    roughening_decay: float = 0.5
    roughening_min_mult: float = 0.25
    init_num_sources: Tuple[int, int] = (0, 3)  # inclusive range
    # Strength prior (cps@1m scale). Defaults cover ~1e3–1e5 cps via log-normal.
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
    init_grid_spacing_m: float | None = None
    init_grid_repeats: int = 1
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
    """Bundle measurement arrays for birth/death and split/merge proposals."""

    z_k: NDArray[np.float64]
    observation_variances: NDArray[np.float64]
    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]


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
        detector_aperture_samples: int = 1,
    ) -> None:
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = float(obstacle_height_m)
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = max(float(obstacle_buildup_coeff), 0.0)
        self.detector_radius_m = max(float(detector_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        mu_by_isotope = getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        shield_params = getattr(kernel, "shield_params", ShieldParams()) if kernel is not None else ShieldParams()
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
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
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        self._adapt_cooldown_remaining = 0
        self._resample_count_in_observation = 0
        self.is_converged = False
        self.frozen_estimate: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
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

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the continuous kernel with the filter's environment attenuation settings."""
        return ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
        )

    def _measurement_source_scale(self) -> float:
        """Return the isotope-specific source response scale for PF likelihoods."""
        scales = self.config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(self.isotope, 1.0)), 0.0)

    def _isotope_float_config(self, value: float | dict[str, float], default: float = 0.0) -> float:
        """Resolve a scalar or isotope-indexed float config value."""
        if isinstance(value, dict):
            return max(float(value.get(self.isotope, default)), 0.0)
        return max(float(value), 0.0)

    def _count_likelihood_kwargs(self) -> dict[str, float | str]:
        """Return likelihood keyword arguments for this isotope filter."""
        return {
            "model": str(self.config.count_likelihood_model),
            "transport_model_rel_sigma": self._isotope_float_config(
                self.config.transport_model_rel_sigma,
            ),
            "spectrum_count_rel_sigma": self._isotope_float_config(
                self.config.spectrum_count_rel_sigma,
            ),
            "spectrum_count_abs_sigma": self._isotope_float_config(
                self.config.spectrum_count_abs_sigma,
            ),
            "student_t_df": max(float(self.config.count_likelihood_df), 1.0),
        }

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
        if obs_var.size != z_arr.size:
            obs_var = np.resize(obs_var, z_arr.size)
        kwargs = self._count_likelihood_kwargs()
        model = str(kwargs["model"])
        z_col = z_arr[:, None]
        if model == "poisson":
            return np.sum(z_col * np.log(lam) - lam, axis=0)
        transport_rel = float(kwargs["transport_model_rel_sigma"])
        spectrum_rel = float(kwargs["spectrum_count_rel_sigma"])
        spectrum_abs = float(kwargs["spectrum_count_abs_sigma"])
        scale_ref = np.maximum(np.maximum(z_col, 0.0), lam)
        variance = (
            lam
            + (transport_rel * lam) ** 2
            + (spectrum_rel * scale_ref) ** 2
            + spectrum_abs**2
            + np.maximum(obs_var[:, None], 0.0)
        )
        variance = np.maximum(variance, 1.0e-12)
        residual = z_col - lam
        if model == "gaussian":
            terms = -0.5 * ((residual**2) / variance + np.log(variance))
            return np.sum(terms, axis=0)
        df = max(float(kwargs["student_t_df"]), 1.0 + 1.0e-12)
        terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
        terms -= 0.5 * np.log(variance)
        return np.sum(terms, axis=0)

    def _signal_bearing_refit_data(
        self,
        data: MeasurementData,
    ) -> MeasurementData | None:
        """
        Return measurements suitable for conditional strength refitting.

        Censored low-signal observations are still valid PF likelihood updates,
        but using only those upper bounds in a non-negative strength projection
        collapses source rates to the numerical floor.  The deterministic
        strength refit therefore uses only recent measurements with either a
        minimum positive count or a minimum count SNR.
        """
        z_arr = np.asarray(data.z_k, dtype=float).reshape(-1)
        if z_arr.size == 0:
            return None
        variances = np.maximum(
            np.asarray(data.observation_variances, dtype=float).reshape(-1),
            1.0,
        )
        if variances.size != z_arr.size:
            variances = np.resize(variances, z_arr.size)
        min_count = max(float(self.config.conditional_strength_refit_min_count), 0.0)
        min_snr = max(float(self.config.conditional_strength_refit_min_snr), 0.0)
        if min_count <= 0.0 and min_snr <= 0.0:
            return data
        snr = np.divide(
            np.maximum(z_arr, 0.0),
            np.sqrt(variances),
            out=np.zeros_like(z_arr, dtype=float),
            where=variances > 0.0,
        )
        finite = np.isfinite(z_arr) & np.isfinite(variances)
        mask = finite & ((z_arr >= min_count) | (snr >= min_snr))
        if not np.any(mask):
            return None
        return MeasurementData(
            z_k=data.z_k[mask],
            observation_variances=data.observation_variances[mask],
            detector_positions=data.detector_positions[mask],
            fe_indices=data.fe_indices[mask],
            pb_indices=data.pb_indices[mask],
            live_times=data.live_times[mask],
        )

    def _strength_refit_prior_precision(
        self,
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return local quadratic prior precision for strength MAP refits."""
        weight = max(float(self.config.conditional_strength_refit_prior_weight), 0.0)
        if weight <= 0.0:
            return np.zeros_like(np.asarray(strengths, dtype=float), dtype=float)
        rel_sigma = max(
            float(self.config.conditional_strength_refit_prior_rel_sigma),
            1.0e-6,
        )
        floor = max(float(self.config.min_strength), 1.0)
        strengths_arr = np.asarray(strengths, dtype=float)
        scale = np.maximum(np.abs(strengths_arr), floor)
        sigma = rel_sigma * scale
        precision = weight / np.maximum(sigma * sigma, 1.0e-12)
        inactive = np.abs(strengths_arr) <= max(float(self.config.min_strength), 0.0) * (
            1.0 + 1.0e-6
        )
        return np.where(inactive, 0.0, precision)

    def _strength_refit_prior_log_ratio(
        self,
        prior_mean: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> float:
        """
        Return the local strength-prior log-density ratio after a MAP refit.

        The deterministic strength refit is a proposal that moves particle
        strengths while fixed source positions are kept.  When particle weights
        are corrected by a profile likelihood, the same local Gaussian strength
        prior used by the MAP solve must also be included; otherwise particles
        with poor geometry can survive by making an unpenalized jump to an
        extreme source rate.
        """
        prior = np.asarray(prior_mean, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.size == 0 or posterior.size == 0:
            return 0.0
        size = min(prior.size, posterior.size)
        prior = prior[:size]
        posterior = posterior[:size]
        precision = self._strength_refit_prior_precision(prior)
        if precision.size != size:
            precision = np.resize(precision, size)
        delta = posterior - prior
        return float(-0.5 * np.sum(precision * delta * delta))

    def _strength_refit_prior_log_ratio_batched(
        self,
        prior_mean: NDArray[np.float64],
        posterior_strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-particle local strength-prior log-density ratios."""
        prior = np.asarray(prior_mean, dtype=float)
        posterior = np.asarray(posterior_strengths, dtype=float)
        if prior.size == 0 or posterior.size == 0:
            return np.zeros(prior.shape[0] if prior.ndim else 0, dtype=float)
        if prior.shape != posterior.shape:
            rows = min(prior.shape[0], posterior.shape[0])
            cols = min(prior.shape[1], posterior.shape[1])
            prior = prior[:rows, :cols]
            posterior = posterior[:rows, :cols]
        precision = self._strength_refit_prior_precision(prior)
        if precision.shape != prior.shape:
            precision = np.resize(precision, prior.shape)
        delta = posterior - prior
        return -0.5 * np.sum(precision * delta * delta, axis=1)

    def _delta_log_likelihood_remove(
        self,
        z_k: NDArray[np.float64],
        lambda_total: NDArray[np.float64],
        lambda_m: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        """Return per-source support using the configured count likelihood."""
        return delta_log_likelihood_remove(
            z_k,
            lambda_total,
            lambda_m,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _delta_log_likelihood_update(
        self,
        z_k: NDArray[np.float64],
        lambda_old: NDArray[np.float64],
        lambda_new: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> float:
        """Return proposal support using the configured count likelihood."""
        return delta_log_likelihood_update(
            z_k,
            lambda_old,
            lambda_new,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _obstacle_gpu_kwargs(self) -> dict[str, object]:
        """Return obstacle attenuation kwargs for GPU expected-count kernels."""
        return self.continuous_kernel.obstacle_gpu_kwargs(self.isotope)

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a kernel and refresh the continuous-kernel configuration."""
        self.kernel = kernel
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )

    def _initial_grid_positions(self) -> NDArray[np.float64]:
        """Return initial grid-center positions when grid init is enabled."""
        spacing = self.config.init_grid_spacing_m
        if spacing is None:
            return np.zeros((0, 3), dtype=float)
        spacing = float(spacing)
        if spacing <= 0.0:
            return np.zeros((0, 3), dtype=float)
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
        """Return an initial source count that respects the configured prior range."""
        min_r, max_r = self.config.init_num_sources
        min_r = max(0, int(min_r))
        max_r = max(min_r, int(max_r))
        if self.config.max_sources is not None:
            max_r = min(max_r, max(0, int(self.config.max_sources)))
            min_r = min(min_r, max_r)
        if max_r <= min_r:
            return min_r
        span = max_r - min_r + 1
        return min_r + (int(particle_index) % span)

    def _initial_grid_state_positions(
        self,
        anchor_position: NDArray[np.float64],
        source_count: int,
        grid_positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return initial source positions for one grid-anchored particle."""
        count = max(0, int(source_count))
        if count <= 0:
            return np.zeros((0, 3), dtype=float)
        anchor = np.asarray(anchor_position, dtype=float).reshape(1, 3)
        if count == 1:
            return anchor.copy()
        grid = np.asarray(grid_positions, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
            lo = np.array(self.config.position_min, dtype=float)
            hi = np.array(self.config.position_max, dtype=float)
            extra = lo + np.random.rand(count - 1, 3) * (hi - lo)
            return np.vstack([anchor, extra])
        replace = grid.shape[0] < count - 1
        extra_idx = np.random.choice(grid.shape[0], size=count - 1, replace=replace)
        return np.vstack([anchor, grid[extra_idx]])

    def _init_continuous_particles(self) -> None:
        """Sample continuous positions/strengths/background from broad priors (Sec. 3.3.2)."""
        self.continuous_particles = []
        grid_positions = self._initial_grid_positions()
        if grid_positions.size:
            repeat_count = max(1, int(self.config.init_grid_repeats))
            target_n = int(grid_positions.shape[0]) * repeat_count
            self.N = target_n
            self.config.num_particles = target_n
            self.config.min_particles = target_n
            self.config.max_particles = target_n
            repeated_positions = np.repeat(grid_positions, repeat_count, axis=0)
            for particle_idx, pos in enumerate(repeated_positions):
                r_h = self._initial_source_count_for_particle(particle_idx)
                if r_h > 0:
                    positions = self._initial_grid_state_positions(
                        anchor_position=pos,
                        source_count=r_h,
                        grid_positions=grid_positions,
                    )
                    strengths = np.random.lognormal(
                        mean=self.config.init_strength_log_mean,
                        sigma=self.config.init_strength_log_sigma,
                        size=r_h,
                    )
                    ages = np.zeros(r_h, dtype=int)
                    low_q_streaks = np.zeros(r_h, dtype=int)
                    support_scores = np.zeros(r_h, dtype=float)
                else:
                    positions = np.zeros((0, 3), dtype=float)
                    strengths = np.zeros(0, dtype=float)
                    ages = np.zeros(0, dtype=int)
                    low_q_streaks = np.zeros(0, dtype=int)
                    support_scores = np.zeros(0, dtype=float)
                b_h = self._background_level()
                st = IsotopeState(
                    num_sources=r_h,
                    positions=positions,
                    strengths=strengths,
                    background=b_h,
                    ages=ages,
                    low_q_streaks=low_q_streaks,
                    support_scores=support_scores,
                )
                self.continuous_particles.append(
                    IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N)))
                )
            return
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        min_r, max_r = self.config.init_num_sources
        for _ in range(self.N):
            r_h = int(np.random.randint(min_r, max_r + 1))
            if self.config.max_sources is not None and self.config.max_sources > 0:
                r_h = min(r_h, self.config.max_sources)
            if r_h > 0:
                positions = lo + np.random.rand(r_h, 3) * (hi - lo)
                strengths = np.random.lognormal(
                    mean=self.config.init_strength_log_mean, sigma=self.config.init_strength_log_sigma, size=r_h
                )
                ages = np.zeros(r_h, dtype=int)
                low_q_streaks = np.zeros(r_h, dtype=int)
                support_scores = np.zeros(r_h, dtype=float)
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
                ages = np.zeros(0, dtype=int)
                low_q_streaks = np.zeros(0, dtype=int)
                support_scores = np.zeros(0, dtype=float)
            b_h = self._background_level()
            st = IsotopeState(
                num_sources=r_h,
                positions=positions,
                strengths=strengths,
                background=b_h,
                ages=ages,
                low_q_streaks=low_q_streaks,
                support_scores=support_scores,
            )
            self.continuous_particles.append(IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N))))

    def reset_step_stats(self) -> None:
        """Reset per-step diagnostic counters."""
        self.last_ess = None
        self.last_ess_pre = None
        self.last_ess_post = None
        self.last_resample_ess = False
        self.last_resample_count = 0
        self.last_birth_count = 0
        self.last_kill_count = 0
        self.last_n_after_adapt = None
        self.last_temper_steps = []
        self.last_temper_resample_count = 0
        self.last_birth_residual_chi2 = 0.0
        self.last_birth_residual_p_value = 1.0
        self.last_birth_residual_support = 0
        self.last_birth_residual_distinct_poses = 0
        self.last_birth_residual_distinct_stations = 0
        self.last_birth_residual_gate_passed = False
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        self._resample_count_in_observation = 0

    def _advance_adapt_cooldown(self) -> None:
        """Decrement the adapt cooldown counter after each update."""
        if self._adapt_cooldown_remaining > 0:
            self._adapt_cooldown_remaining -= 1

    def _trigger_adapt_cooldown(self) -> None:
        """Start the adapt cooldown after a resampling event."""
        steps = max(0, int(self.config.adapt_cooldown_steps))
        if steps > 0:
            self._adapt_cooldown_remaining = max(self._adapt_cooldown_remaining, steps + 1)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        if not gpu_utils.torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

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
        source_scale = self._measurement_source_scale()
        if state.num_sources > 0:
            for pos, strength in zip(state.positions[:state.num_sources], state.strengths[:state.num_sources]):
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
        mmse_pos = self._mmse_primary_position()
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
        self._converge_monitor.update_stats(step_idx, mmse_pos, ess_ratio, ll_value)
        if self._converge_monitor.is_converged(step_idx):
            self.is_converged = True
            self.frozen_estimate = self.estimate()

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
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
        mu_fe, mu_pb = self.continuous_kernel._mu_values(isotope=self.isotope)
        shield_params = self.continuous_kernel.shield_params
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        return gpu_utils.expected_counts_pair_torch(
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
            source_scale=self._measurement_source_scale(),
            detector_radius_m=self.continuous_kernel.detector_radius_m,
            detector_aperture_samples=self.continuous_kernel.detector_aperture_samples,
            buildup_fe_coeff=shield_params.buildup_fe_coeff,
            buildup_pb_coeff=shield_params.buildup_pb_coeff,
            **self._obstacle_gpu_kwargs(),
        )

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
        model = str(self.config.count_likelihood_model).strip().lower()
        if model in {"poisson", ""}:
            return z * torch.log(lam_t) - lam_t
        if model == "normal":
            model = "gaussian"
        if model in {"robust", "robust_gaussian", "t"}:
            model = "student_t"
        if model not in {"gaussian", "student_t"}:
            raise ValueError(f"Unknown count likelihood model: {self.config.count_likelihood_model}")

        transport_rel = self._isotope_float_config(self.config.transport_model_rel_sigma)
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        obs_var = max(float(observation_count_variance), 0.0)
        z_nonnegative = torch.clamp(z, min=0.0)
        scale_ref = torch.maximum(lam_t, z_nonnegative)
        variance = (
            lam_t
            + (float(transport_rel) * lam_t) ** 2
            + (float(spectrum_rel) * scale_ref) ** 2
            + float(spectrum_abs) ** 2
            + obs_var
        )
        variance = torch.clamp(variance, min=1e-12)
        residual = z - lam_t
        if model == "gaussian":
            return -0.5 * ((residual**2) / variance + torch.log(variance))

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        return -0.5 * (df + 1.0) * torch.log1p((residual**2) / (df * variance)) - 0.5 * torch.log(variance)

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
        logw_prev = logw_prev if logw_prev is not None else self._current_log_weights_torch(lam_t.device)
        ll_t = ll_t if ll_t is not None else self._log_likelihood_increment_gpu(
            lam_t,
            z_obs,
            observation_count_variance=observation_count_variance,
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

        return self._tempered_update_likelihood(ll_fn=_ll_fn)

    def _tempered_update_likelihood(
        self,
        ll_fn: Callable[[], "torch.Tensor"],
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
        force_resample_ess = float(self.config.temper_resample_force_ratio) * max(self.N, 1)
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
            if do_resample:
                self._maybe_resample_continuous(
                    disable_regularize=bool(self.config.disable_regularize_on_temper_resample),
                )
                if self.last_resample_ess:
                    resampled_any = True
                    resamples += 1
                    cooldown_remaining = max(cooldown_remaining, cooldown_steps)
                    ll_t = ll_fn()
                    if ll_t.numel() == 0:
                        break
                    logw = self._current_log_weights_torch(ll_t.device)
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
            pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts(self, pose_idx: int, orient_idx: int, live_time_s: float) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for each continuous particle using ContinuousKernel."""
        self._gpu_enabled()
        return self._continuous_expected_counts_gpu(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )

    def _continuous_expected_counts_pair(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using Fe/Pb octant indices (Eq. 3.41)."""
        self._gpu_enabled()
        return self._continuous_expected_counts_pair_gpu(
            pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
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
        mu_fe, mu_pb = self.continuous_kernel._mu_values(isotope=self.isotope)
        shield_params = self.continuous_kernel.shield_params
        det_pos = np.asarray(detector_pos, dtype=float)
        return gpu_utils.expected_counts_pair_torch(
            detector_pos=det_pos,
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
            source_scale=self._measurement_source_scale(),
            detector_radius_m=self.continuous_kernel.detector_radius_m,
            detector_aperture_samples=self.continuous_kernel.detector_aperture_samples,
            buildup_fe_coeff=shield_params.buildup_fe_coeff,
            buildup_pb_coeff=shield_params.buildup_pb_coeff,
            **self._obstacle_gpu_kwargs(),
        )

    def _continuous_expected_counts_pair_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for explicit detector position."""
        self._gpu_enabled()
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
    ) -> None:
        """
        Count-likelihood weight update using Fe/Pb orientation indices.

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        When ``defer_resample`` is True, only log-weights are updated; resampling,
        roughening, particle-count adaptation, and birth/death are left to the
        caller's end-of-station finalization.
        """
        if self.config.converge_enable and self.is_converged:
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            return self._continuous_expected_counts_pair_torch(
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )

        if defer_resample:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            ess_pre = 0.0 if logw is None else self._ess_from_logw_torch(logw)
            self.last_ess = float(ess_pre)
            self.last_ess_pre = float(ess_pre)
            self.last_ess_post = None
            self.last_resample_ess = False
            resampled_any = False
        elif self.config.use_tempering:
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
        if not defer_resample:
            if resampled_any:
                self._trigger_adapt_cooldown()
            self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
            self.align_continuous_labels()
            self._advance_adapt_cooldown()
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float) if self.kernel else None
        if detector_pos is not None:
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )

    def finalize_deferred_update(self) -> None:
        """Apply deferred resampling, adaptation, and label alignment after a station."""
        if self.config.converge_enable and self.is_converged:
            return
        prior_ess = self.last_ess_pre
        self._maybe_resample_continuous()
        resampled_any = bool(self.last_resample_ess)
        ess_pre = float(self.last_ess_pre if self.last_ess_pre is not None else 0.0)
        if prior_ess is not None:
            ess_pre = float(prior_ess)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self.align_continuous_labels()
        self._advance_adapt_cooldown()

    def update_continuous_pair_sequence(
        self,
        z_obs: NDArray[np.float64],
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64] | None = None,
        step_idx: int | None = None,
    ) -> None:
        """
        Jointly update weights using a same-pose shield-orientation sequence.

        The measurement model is the product of conditionally independent count
        likelihoods for the shield program. Updating them jointly avoids
        resampling, roughening, or birth/death moves between postures from the
        same physical station.
        """
        if self.config.converge_enable and self.is_converged:
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()
        z_arr = np.asarray(z_obs, dtype=float).ravel()
        fe_arr = np.asarray(fe_indices, dtype=int).ravel()
        pb_arr = np.asarray(pb_indices, dtype=int).ravel()
        live_arr = np.asarray(live_times_s, dtype=float).ravel()
        if observation_count_variances is None:
            var_arr = np.zeros_like(z_arr, dtype=float)
        else:
            var_arr = np.asarray(observation_count_variances, dtype=float).ravel()
        if not (
            z_arr.size == fe_arr.size == pb_arr.size == live_arr.size == var_arr.size
        ):
            raise ValueError("Joint PF update arrays must have matching lengths.")
        if z_arr.size == 0:
            return

        def _ll_fn() -> "torch.Tensor":
            """Return summed per-particle log-likelihood for the shield sequence."""
            ll_total = None
            for z_val, fe_index, pb_index, live_time_s, variance in zip(
                z_arr,
                fe_arr,
                pb_arr,
                live_arr,
                var_arr,
            ):
                lam_t = self._continuous_expected_counts_pair_torch(
                    pose_idx=pose_idx,
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                    live_time_s=float(live_time_s),
                )
                ll_t = self._log_likelihood_increment_gpu(
                    lam_t,
                    float(z_val),
                    observation_count_variance=float(variance),
                )
                ll_total = ll_t if ll_total is None else ll_total + ll_t
            if ll_total is None:
                import torch

                from pf import gpu_utils

                device = gpu_utils.resolve_device(self.config.gpu_device)
                return torch.zeros(0, device=device, dtype=torch.float64)
            return ll_total

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update_likelihood(ll_fn=_ll_fn)
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
        if self.kernel is not None:
            detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
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
        if self.config.converge_enable and self.is_converged:
            self.updates_skipped += 1
            return
        self.reset_step_stats()
        self._gpu_enabled()

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            return self._continuous_expected_counts_pair_at_pose_torch(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
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
        logw = np.asarray([p.log_weight for p in self.continuous_particles], dtype=np.float64)
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles), dtype=float) / len(self.continuous_particles)
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

    def _maybe_resample_continuous(self, *, disable_regularize: bool = False) -> None:
        """ESS check and systematic resampling for continuous particles (Sec. 3.3.4, Eq. 3.29)."""
        w = np.asarray(self.continuous_weights, dtype=np.float64)
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
            self.last_resample_ess = True
            self.last_resample_count += 1
            logw = np.log(np.clip(w, 1e-300, 1.0))
            idx = systematic_resample(logw)
            self.continuous_particles = [self.continuous_particles[i].state.copy() for i in idx]
            # reset weights to uniform
            self.continuous_particles = [
                IsotopeParticle(state=st, log_weight=float(-np.log(self.N))) for st in self.continuous_particles
            ]
            self.last_ess_post = float(len(self.continuous_particles))
            if not disable_regularize:
                mult = self._roughening_multiplier()
                sigma_pos = self._roughening_sigma_pos(len(self.continuous_particles)) * mult
                self.regularize_continuous(
                    sigma_pos=sigma_pos,
                    strength_log_sigma=self.config.strength_log_sigma * mult,
                    p_birth=self.config.p_birth,
                    p_kill=self.config.p_kill,
                    intensity_threshold=self.config.min_strength,
                )
            self._resample_count_in_observation += 1

    def _label_scales(
        self,
        ref_positions: NDArray[np.float64],
        ref_strengths: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Return (pos_scale, strength_scale) for label alignment costs."""
        if self.config.label_pos_scale is not None:
            pos_scale = float(self.config.label_pos_scale)
        else:
            span = np.array(self.config.position_max, dtype=float) - np.array(self.config.position_min, dtype=float)
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
            str_cost = (
                np.abs(strengths[:, None] - ref_strengths[None, :])
                / float(strength_scale)
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
        str_cost = torch.abs(str_t[:, None] - ref_str_t[None, :]) / float(strength_scale)
        cost = self.config.label_pos_weight * pos_cost + self.config.label_strength_weight * str_cost
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
        padded = np.full((size, size), float(self.config.label_missing_cost), dtype=float)
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
            st.low_q_streaks = st.low_q_streaks[ordered_rows]
            st.support_scores = st.support_scores[ordered_rows]
            st.num_sources = st.positions.shape[0]

    def _reference_from_particles(self, ref_count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            ref_positions, ref_strengths = self._reference_from_particles(ref_positions.shape[0])
        self._label_reference = IsotopeState(
            num_sources=ref_positions.shape[0],
            positions=ref_positions,
            strengths=ref_strengths,
            background=0.0,
        )

    def adapt_num_particles(self, *, ess_pre: float | None = None, resampled: bool = False) -> None:
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
        if ess_ratio < self.config.ess_low and len(self.continuous_particles) < max_particles:
            grown = max(len(self.continuous_particles) + 1, int(len(self.continuous_particles) * 1.25))
            target = min(max_particles, grown)
            self._resample_continuous_to(target, jitter=True)
        elif allow_shrink and ess_ratio > self.config.ess_high and len(self.continuous_particles) > min_particles:
            target = max(min_particles, int(len(self.continuous_particles) * 0.8))
            self._resample_continuous_to(target, jitter=False)
        self.last_n_after_adapt = int(len(self.continuous_particles))

    def _resample_continuous_to(self, target_n: int, jitter: bool = False) -> None:
        """Resample the continuous particles to a new population size."""
        target_n = max(1, int(target_n))
        self.last_resample_count += 1
        w = self.continuous_weights
        idx = np.random.choice(len(self.continuous_particles), size=target_n, p=w)
        states = [self.continuous_particles[i].state.copy() for i in idx]
        self.continuous_particles = [
            IsotopeParticle(state=st, log_weight=float(-np.log(target_n))) for st in states
        ]
        self.N = target_n
        self.config.num_particles = target_n
        if jitter:
            mult = self._roughening_multiplier()
            sigma_pos = self._roughening_sigma_pos(len(self.continuous_particles)) * mult
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
        st.low_q_streaks = self._resize_metadata_array(st.low_q_streaks, r, 0, int)
        st.support_scores = self._resize_metadata_array(st.support_scores, r, 0.0, float)

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
            source_scale=self._measurement_source_scale(),
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
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        background_counts = data.live_times[:, None] * backgrounds[None, :]
        if count <= 0:
            return np.zeros((num_meas, particle_count, 0), dtype=float), background_counts
        sources = np.vstack(
            [
                np.asarray(
                    self.continuous_particles[idx].state.positions[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        strengths = np.concatenate(
            [
                np.asarray(
                    self.continuous_particles[idx].state.strengths[:count],
                    dtype=float,
                )
                for idx in particle_indices
            ]
        )
        lambda_flat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=sources,
            strengths=strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale(),
        )
        lambda_m = np.asarray(lambda_flat, dtype=float).reshape(
            num_meas,
            particle_count,
            count,
        )
        lambda_total = background_counts + np.sum(lambda_m, axis=2)
        return lambda_m, lambda_total

    def _delta_log_likelihood_remove_group(
        self,
        data: MeasurementData,
        lambda_total: NDArray[np.float64],
        lambda_components: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-particle, per-source removal support for a batched group."""
        total = np.asarray(lambda_total, dtype=float)
        components = np.asarray(lambda_components, dtype=float)
        if components.ndim != 3 or components.shape[:2] != total.shape:
            return np.zeros((total.shape[1] if total.ndim == 2 else 0, 0), dtype=float)
        source_count = int(components.shape[2])
        if source_count <= 0:
            return np.zeros((total.shape[1], 0), dtype=float)
        model = str(self._count_likelihood_kwargs()["model"]).strip().lower()
        if model in {"poisson", ""}:
            total_safe = np.maximum(total, 1.0e-12)
            ratio = np.clip(
                components / total_safe[:, :, None],
                0.0,
                1.0 - 1.0e-12,
            )
            terms = (
                np.asarray(data.z_k, dtype=float)[:, None, None] * (-np.log1p(-ratio))
                - components
            )
            return np.sum(terms, axis=0)
        base_ll = self._count_log_likelihood_matrix_np(
            data.z_k,
            total,
            observation_count_variance=data.observation_variances,
        )
        deltas = np.zeros((total.shape[1], source_count), dtype=float)
        for source_idx in range(source_count):
            reduced = np.maximum(total - components[:, :, source_idx], 1.0e-12)
            reduced_ll = self._count_log_likelihood_matrix_np(
                data.z_k,
                reduced,
                observation_count_variance=data.observation_variances,
            )
            deltas[:, source_idx] = base_ll - reduced_ll
        return deltas

    def _compute_birth_proposal(
        self,
        data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]] | None:
        """
        Build residual-driven birth proposal (probabilities, kernel_sums, residual_sum, candidates).
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
        residuals: list[NDArray[np.float64]] = []
        for idx, p_idx in enumerate(order):
            st = self.continuous_particles[int(p_idx)].state
            background_counts = float(st.background) * data.live_times
            if st.num_sources > 0:
                lambda_m = expected_counts_per_source(
                    kernel=self.continuous_kernel,
                    isotope=self.isotope,
                    detector_positions=data.detector_positions,
                    sources=st.positions,
                    strengths=st.strengths,
                    live_times=data.live_times,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                    source_scale=self._measurement_source_scale(),
                )
                lambda_total = background_counts + np.sum(lambda_m, axis=1)
            else:
                lambda_total = background_counts
            residual = np.maximum(data.z_k - lambda_total, 0.0)
            clip_q = float(self.config.birth_residual_clip_quantile)
            if 0.0 < clip_q < 1.0 and residual.size:
                clip_val = float(np.quantile(residual, clip_q))
                residual = np.minimum(residual, clip_val)
            if bool(self.config.birth_use_weighted_topk):
                residuals.append(residual * float(sel_weights[idx]))
            else:
                residuals.append(residual)
        if not residuals:
            return None
        residual_stack = np.vstack(residuals)
        if bool(self.config.birth_use_weighted_topk):
            residual_mix = np.sum(residual_stack, axis=0)
        else:
            residual_mix = np.mean(residual_stack, axis=0)
        residual_sum = float(np.sum(residual_mix))
        if residual_sum <= 0.0:
            return None
        if not self._birth_residual_gate_allows(
            residual_mix,
            data.observation_variances,
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
        ):
            return None
        if not self._birth_residual_survives_strength_refit(
            data=data,
            particle_indices=order,
            particle_weights=sel_weights,
            residual_sum_before=residual_sum,
        ):
            return None

        base_candidates = self._exclude_birth_candidates_near_detectors(
            candidate_positions.copy(),
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
            source_scale=1.0,
        )
        base_support_mask = self._birth_candidate_support_mask(
            candidate_counts=base_candidate_counts,
            residual_mix=residual_mix,
            observation_variances=data.observation_variances,
            detector_positions=data.detector_positions,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
        )
        if not np.any(base_support_mask):
            return None
        base_scores = np.asarray(residual_mix @ base_candidate_counts, dtype=float)
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
            lo = np.array(self.config.position_min, dtype=float)
            hi = np.array(self.config.position_max, dtype=float)
            jittered = np.clip(jittered, lo, hi).reshape(-1, 3)
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
                    source_scale=1.0,
                )
                candidate_counts = np.hstack([candidate_counts, jitter_counts])
                candidates = np.vstack([candidates, jittered])
                final_support_mask = self._birth_candidate_support_mask(
                    candidate_counts=candidate_counts,
                    residual_mix=residual_mix,
                    observation_variances=data.observation_variances,
                    detector_positions=data.detector_positions,
                    fe_indices=data.fe_indices,
                    pb_indices=data.pb_indices,
                )
                if not np.any(final_support_mask):
                    return None
                candidate_counts = candidate_counts[:, final_support_mask]
                candidates = candidates[final_support_mask]
        kernel_sums = np.sum(candidate_counts, axis=0)
        scores = np.asarray(residual_mix @ candidate_counts, dtype=float)
        if np.max(scores) <= 0.0:
            return None
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        kernel_sums = kernel_sums[order]
        candidates = candidates[order]
        scores = np.maximum(scores, float(self.config.birth_min_score))
        temp = max(float(self.config.birth_softmax_temp), 1e-6)
        scaled = scores / temp
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        return probs, kernel_sums, residual_sum, candidates

    def _birth_residual_survives_strength_refit(
        self,
        *,
        data: MeasurementData,
        particle_indices: NDArray[np.int64],
        particle_weights: NDArray[np.float64],
        residual_sum_before: float,
    ) -> bool:
        """
        Return True when residual evidence remains after refitting existing sources.

        A positive residual should create a new source only when it cannot be
        explained by re-estimating the strengths of sources already present in
        high-posterior particles. This makes the birth move a residual-support
        test rather than a response to a stale strength estimate.
        """
        self.last_birth_residual_refit_fraction = 1.0
        self.last_birth_residual_refit_gate_passed = True
        if not bool(self.config.birth_refit_residual_gate):
            return True
        before = max(float(residual_sum_before), 1.0e-12)
        if data.z_k.size == 0:
            return False
        residuals: list[NDArray[np.float64]] = []
        weights = np.asarray(particle_weights, dtype=float).ravel()
        if weights.size == 0 or float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(particle_indices), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        for local_idx, particle_idx in enumerate(np.asarray(particle_indices, dtype=int).ravel()):
            st = self.continuous_particles[int(particle_idx)].state.copy()
            self._ensure_source_metadata(st)
            self._refit_strengths_for_particle(
                st,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            _, lambda_total = self._lambda_components(st, data)
            residual = np.maximum(data.z_k - lambda_total, 0.0)
            clip_q = float(self.config.birth_residual_clip_quantile)
            if 0.0 < clip_q < 1.0 and residual.size:
                clip_val = float(np.quantile(residual, clip_q))
                residual = np.minimum(residual, clip_val)
            residuals.append(residual * float(weights[local_idx]))
        if not residuals:
            self.last_birth_residual_refit_fraction = 0.0
            self.last_birth_residual_refit_gate_passed = False
            return False
        residual_after = np.sum(np.vstack(residuals), axis=0)
        after_sum = float(np.sum(residual_after))
        fraction = after_sum / before
        self.last_birth_residual_refit_fraction = float(fraction)
        min_fraction = max(float(self.config.birth_refit_residual_min_fraction), 0.0)
        passed = fraction >= min_fraction and self._birth_residual_gate_allows(
            residual_after,
            data.observation_variances,
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
        )
        self.last_birth_residual_refit_gate_passed = bool(passed)
        return bool(passed)

    def _birth_residual_gate_allows(
        self,
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        detector_positions: NDArray[np.float64] | None = None,
        fe_indices: NDArray[np.int64] | None = None,
        pb_indices: NDArray[np.int64] | None = None,
    ) -> bool:
        """
        Return True when positive residuals statistically justify a birth move.

        The gate tests whether the posterior residual that cannot be explained by
        the current particles is large relative to the spectrum/PF observation
        variance. It prevents a single low-confidence spectrum decomposition
        outlier from creating persistent false-positive sources.
        """
        residual = np.maximum(np.asarray(residual_mix, dtype=float), 0.0)
        variances = np.maximum(np.asarray(observation_variances, dtype=float), 1.0e-12)
        if residual.size == 0:
            return False
        if variances.size != residual.size:
            variances = np.resize(variances, residual.size)
            variances = np.maximum(variances, 1.0e-12)
        sigma = np.sqrt(variances)
        z_score = residual / np.maximum(sigma, 1.0e-12)
        min_sigma = max(float(self.config.birth_residual_support_sigma), 0.0)
        support_mask = z_score >= min_sigma
        support_count = int(np.count_nonzero(support_mask))
        distinct_supported = self._distinct_supported_view_count(
            detector_positions,
            fe_indices,
            pb_indices,
            support_mask,
        )
        distinct_stations = self._distinct_supported_station_count(
            detector_positions,
            support_mask,
        )
        chi2_stat = float(np.sum((residual[support_mask] ** 2) / variances[support_mask]))
        dof = max(support_count, 1)
        p_value = float(chi2.sf(chi2_stat, dof)) if support_count > 0 else 1.0
        self.last_birth_residual_chi2 = chi2_stat
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
                and chi2_stat > 0.0
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
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != mask.size:
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
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != mask.size:
            return int(np.count_nonzero(mask))
        if not np.any(mask):
            return 0
        rounded_xy = np.round(positions[mask, :2], decimals=3)
        return int(np.unique(rounded_xy, axis=0).shape[0])

    def _birth_candidate_support_mask(
        self,
        *,
        candidate_counts: NDArray[np.float64],
        residual_mix: NDArray[np.float64],
        observation_variances: NDArray[np.float64],
        detector_positions: NDArray[np.float64],
        fe_indices: NDArray[np.int64] | None = None,
        pb_indices: NDArray[np.int64] | None = None,
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
        residual = np.maximum(np.asarray(residual_mix, dtype=float).ravel(), 0.0)
        variances = np.maximum(
            np.asarray(observation_variances, dtype=float).ravel(),
            1.0e-12,
        )
        if residual.size != counts.shape[0]:
            residual = np.resize(residual, counts.shape[0])
        if variances.size != counts.shape[0]:
            variances = np.resize(variances, counts.shape[0])
        sigma = np.sqrt(variances)
        z_score = residual / np.maximum(sigma, 1.0e-12)
        residual_support = z_score >= max(float(self.config.birth_residual_support_sigma), 0.0)
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
        min_support = max(1, int(self.config.birth_residual_min_support))
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
        keep = support_counts >= min_support
        if min_distinct <= 1:
            view_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            view_labels = self._support_view_labels(
                detector_positions,
                fe_indices,
                pb_indices,
                support.shape[0],
            )
            distinct_counts = self._distinct_label_counts_for_support_matrix(
                support,
                view_labels,
            )
            view_keep = distinct_counts >= min_distinct
        if min_stations <= 1:
            station_keep = np.ones(counts.shape[1], dtype=bool)
        else:
            station_labels = self._support_station_labels(
                detector_positions,
                support.shape[0],
            )
            station_counts = self._distinct_label_counts_for_support_matrix(
                support,
                station_labels,
            )
            station_keep = station_counts >= min_stations
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
            return np.zeros(support_arr.shape[1] if support_arr.ndim == 2 else 0, dtype=int)
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
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != count:
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
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != count:
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

        Birth/death moves are handled in apply_birth_death().
        """
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
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
                st.positions = st.positions + np.random.normal(scale=sigma_pos_arr, size=st.positions.shape)
                st.positions = np.clip(st.positions, lo, hi)
                if log_sigma > 0.0:
                    logq = np.log(st.strengths + 1e-12)
                    logq = logq + np.random.normal(scale=log_sigma, size=st.strengths.shape)
                    st.strengths = np.exp(logq)
                st.strengths = np.maximum(st.strengths, 0.0)
                st.num_sources = st.positions.shape[0]

    def _refit_strengths_for_particle(
        self,
        st: IsotopeState,
        data: MeasurementData,
        *,
        iters: int,
        eps: float,
    ) -> float:
        """
        Refit per-source strengths using coordinate ascent with fixed positions.

        This stabilizes source intensities after birth/kill/split/merge moves.
        """
        if st.num_sources <= 0 or data.z_k.size == 0:
            return 0.0
        num_sources = int(st.num_sources)
        num_meas = int(data.z_k.size)
        k_mat = np.zeros((num_meas, num_sources), dtype=float)
        source_scale = self._measurement_source_scale()
        for j in range(num_sources):
            pos = st.positions[j]
            for k in range(num_meas):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=data.detector_positions[k],
                    source_pos=pos,
                    fe_index=int(data.fe_indices[k]),
                    pb_index=int(data.pb_indices[k]),
                )
                k_mat[k, j] = float(data.live_times[k]) * source_scale * kernel_val
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        strengths = np.asarray(st.strengths, dtype=float)
        background_counts = float(st.background) * data.live_times
        lambda_before = background_counts + k_mat @ strengths
        ll_before = self._count_log_likelihood_np(
            data.z_k,
            lambda_before,
            observation_count_variance=data.observation_variances,
        )
        obs_weights = 1.0 / np.maximum(
            np.asarray(data.observation_variances, dtype=float),
            1.0,
        )
        prior_mean = strengths.copy()
        prior_precision = self._strength_refit_prior_precision(prior_mean)
        gram = (k_mat.T * obs_weights[None, :]) @ k_mat
        rhs = (k_mat.T * obs_weights[None, :]) @ (data.z_k - background_counts)
        try:
            gram = gram + np.diag(prior_precision)
            rhs = rhs + prior_precision * prior_mean
            direct = np.linalg.solve(
                gram + np.eye(num_sources, dtype=float) * float(eps),
                rhs,
            )
            strengths = np.clip(np.maximum(direct, 0.0), q_min, q_max)
        except np.linalg.LinAlgError:
            pass
        lambda_total = background_counts + k_mat @ strengths
        for _ in range(max(1, int(iters))):
            for j in range(num_sources):
                k_col = k_mat[:, j]
                denom = float(
                    np.sum(obs_weights * k_col * k_col)
                    + prior_precision[j]
                    + float(eps)
                )
                if denom <= 0.0:
                    strengths[j] = 0.0
                    continue
                residual = data.z_k - (lambda_total - strengths[j] * k_col)
                numer = float(
                    np.sum(obs_weights * residual * k_col)
                    + prior_precision[j] * prior_mean[j]
                )
                q_new = max(0.0, numer / denom)
                q_new = float(np.clip(q_new, q_min, q_max))
                if q_new != strengths[j]:
                    lambda_total = lambda_total - strengths[j] * k_col + q_new * k_col
                    strengths[j] = q_new
        st.strengths = strengths
        st.num_sources = st.positions.shape[0]
        ll_after = self._count_log_likelihood_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )
        prior_ratio = self._strength_refit_prior_log_ratio(prior_mean, strengths)
        return float(ll_after - ll_before + prior_ratio)

    def refit_strengths_for_particles(
        self,
        data: MeasurementData | None,
        *,
        iters: int | None = None,
        eps: float | None = None,
    ) -> None:
        """
        Refit all particle strengths conditioned on their sampled positions.

        This is a Rao-Blackwellized-style deterministic update: the PF keeps the
        nonlinear source positions in particles, while source rates are projected
        to the non-negative weighted least-squares optimum for recent
        spectrum-derived counts.
        """
        if data is None or data.z_k.size == 0:
            return
        if not bool(self.config.conditional_strength_refit):
            return
        data = self._signal_bearing_refit_data(data)
        if data is None or data.z_k.size == 0:
            return
        max_iters = (
            int(self.config.conditional_strength_refit_iters)
            if iters is None
            else int(iters)
        )
        eps_value = float(self.config.refit_eps if eps is None else eps)
        reweight = bool(self.config.conditional_strength_refit_reweight)
        corrections = np.zeros(len(self.continuous_particles), dtype=float)
        grouped: dict[int, list[int]] = {}
        fallback_indices: list[int] = []
        for idx, particle in enumerate(self.continuous_particles):
            st = particle.state
            self._ensure_source_metadata(st)
            source_count = int(st.num_sources)
            if source_count <= 0:
                continue
            if st.positions.shape[0] < source_count or st.strengths.size < source_count:
                fallback_indices.append(idx)
                continue
            grouped.setdefault(source_count, []).append(idx)
        for source_count, particle_indices in grouped.items():
            group_corrections = self._refit_fixed_source_count_particles_batched(
                data,
                particle_indices=particle_indices,
                source_count=source_count,
                iters=max_iters,
                eps=eps_value,
                compute_loglike_correction=reweight,
            )
            if reweight and group_corrections.size == len(particle_indices):
                corrections[np.asarray(particle_indices, dtype=int)] = group_corrections
        for idx in fallback_indices:
            st = self.continuous_particles[idx].state
            correction = self._refit_strengths_for_particle(
                st,
                data,
                iters=max_iters,
                eps=eps_value,
            )
            if reweight:
                corrections[int(idx)] = float(correction)
            self._prune_floor_sources_after_refit(st, data)
        if reweight and corrections.size:
            clip = max(float(self.config.conditional_strength_refit_reweight_clip), 0.0)
            if clip > 0.0:
                corrections = np.clip(corrections, -clip, clip)
            for particle, delta in zip(self.continuous_particles, corrections):
                particle.log_weight += float(delta)
            self._normalize_continuous_log_weights()
        self.align_continuous_labels()

    def _refit_particle_indices_batched(
        self,
        data: MeasurementData,
        particle_indices: list[int],
        *,
        iters: int,
        eps: float,
    ) -> None:
        """Refit selected particles by grouped cardinality after structural moves."""
        if not particle_indices or data.z_k.size == 0:
            return
        grouped, fallback_indices = self._particle_indices_by_source_count(particle_indices)
        for source_count, group_indices in grouped.items():
            if source_count <= 0 or not group_indices:
                continue
            self._refit_fixed_source_count_particles_batched(
                data,
                particle_indices=group_indices,
                source_count=source_count,
                iters=iters,
                eps=eps,
                compute_loglike_correction=False,
            )
        for idx in fallback_indices:
            st = self.continuous_particles[idx].state
            self._refit_strengths_for_particle(
                st,
                data,
                iters=iters,
                eps=eps,
            )
            self._prune_floor_sources_after_refit(st, data)

    def _refit_single_source_particles_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: list[int],
        eps: float,
    ) -> None:
        """Refit one-source particle strengths with a batched kernel evaluation."""
        self._refit_fixed_source_count_particles_batched(
            data,
            particle_indices=particle_indices,
            source_count=1,
            iters=1,
            eps=eps,
        )

    def _refit_fixed_source_count_particles_batched(
        self,
        data: MeasurementData,
        *,
        particle_indices: list[int],
        source_count: int,
        iters: int,
        eps: float,
        compute_loglike_correction: bool = False,
    ) -> NDArray[np.float64]:
        """Refit equal-cardinality particles using one batched kernel evaluation."""
        if not particle_indices:
            return np.zeros(0, dtype=float)
        count = max(1, int(source_count))
        sources = np.vstack(
            [
                self.continuous_particles[idx].state.positions[:count]
                for idx in particle_indices
            ]
        )
        unit_strengths = np.ones(sources.shape[0], dtype=float)
        k_flat = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=data.detector_positions,
            sources=sources,
            strengths=unit_strengths,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=self._measurement_source_scale(),
        )
        num_meas = int(data.z_k.size)
        particle_count = int(len(particle_indices))
        k_tensor = k_flat.reshape(num_meas, particle_count, count)
        obs_weights = 1.0 / np.maximum(
            np.asarray(data.observation_variances, dtype=float),
            1.0,
        )
        q_min = max(float(self.config.min_strength), 0.0)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        z_arr = np.asarray(data.z_k, dtype=float)
        live_arr = np.asarray(data.live_times, dtype=float)
        backgrounds = np.asarray(
            [
                float(self.continuous_particles[idx].state.background)
                for idx in particle_indices
            ],
            dtype=float,
        )
        strengths = np.vstack(
            [
                np.asarray(self.continuous_particles[idx].state.strengths[:count], dtype=float)
                for idx in particle_indices
            ]
        )
        background_counts = live_arr[:, None] * backgrounds[None, :]
        strengths_before = strengths.copy()
        prior_mean = strengths_before.copy()
        prior_precision = self._strength_refit_prior_precision(prior_mean)
        lambda_before = background_counts + np.einsum(
            "kps,ps->kp",
            k_tensor,
            strengths_before,
        )
        gram = np.einsum("kps,kpt,k->pst", k_tensor, k_tensor, obs_weights)
        rhs = np.einsum(
            "kps,kp,k->ps",
            k_tensor,
            z_arr[:, None] - background_counts,
            obs_weights,
        )
        prior_diag = np.zeros_like(gram, dtype=float)
        diag_idx = np.arange(count)
        prior_diag[:, diag_idx, diag_idx] = prior_precision
        rhs = rhs + prior_precision * prior_mean
        eye = np.eye(count, dtype=float)[None, :, :] * float(eps)
        try:
            direct = np.linalg.solve(gram + prior_diag + eye, rhs[:, :, None])[:, :, 0]
            strengths = np.clip(np.maximum(direct, 0.0), q_min, q_max)
        except np.linalg.LinAlgError:
            pass
        lambda_total = background_counts + np.einsum("kps,ps->kp", k_tensor, strengths)
        weight_col = obs_weights[:, None]
        z_col = z_arr[:, None]
        for _ in range(max(1, int(iters))):
            for source_idx in range(count):
                k_col = k_tensor[:, :, source_idx]
                prior_col = prior_precision[:, source_idx]
                denom = (
                    np.sum(weight_col * k_col * k_col, axis=0)
                    + prior_col
                    + float(eps)
                )
                old_strength = strengths[:, source_idx].copy()
                residual = z_col - (lambda_total - old_strength[None, :] * k_col)
                numer = (
                    np.sum(weight_col * residual * k_col, axis=0)
                    + prior_col * prior_mean[:, source_idx]
                )
                q_new = np.divide(
                    numer,
                    denom,
                    out=np.zeros_like(numer, dtype=float),
                    where=denom > 0.0,
                )
                q_new = np.clip(np.maximum(q_new, 0.0), q_min, q_max)
                lambda_total += (q_new - old_strength)[None, :] * k_col
                strengths[:, source_idx] = q_new
        expected_source_counts = np.sum(k_tensor * strengths[None, :, :], axis=0)
        for row_idx, particle_idx in enumerate(particle_indices):
            st = self.continuous_particles[particle_idx].state
            st.strengths[:count] = strengths[row_idx]
            st.num_sources = st.positions.shape[0]
            self._prune_floor_sources_by_expected_counts(
                st,
                expected_source_counts[row_idx],
            )
        if not compute_loglike_correction:
            return np.zeros(len(particle_indices), dtype=float)
        lambda_after = background_counts + np.einsum("kps,ps->kp", k_tensor, strengths)
        ll_before = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_before,
            observation_count_variance=data.observation_variances,
        )
        ll_after = self._count_log_likelihood_matrix_np(
            data.z_k,
            lambda_after,
            observation_count_variance=data.observation_variances,
        )
        prior_ratio = self._strength_refit_prior_log_ratio_batched(
            prior_mean,
            strengths,
        )
        return np.asarray(ll_after - ll_before + prior_ratio, dtype=float)

    def _prune_floor_sources_after_refit(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> None:
        """Prune min-clamped sources using expected counts from fresh data."""
        if st.num_sources <= 1 or data.z_k.size == 0:
            return
        self._ensure_source_metadata(st)
        lambda_m, _ = self._lambda_components(st, data)
        if lambda_m.size == 0:
            return
        expected_counts = np.sum(np.maximum(lambda_m, 0.0), axis=0)
        self._prune_floor_sources_by_expected_counts(st, expected_counts)

    def _prune_floor_sources_by_expected_counts(
        self,
        st: IsotopeState,
        expected_counts: NDArray[np.float64],
    ) -> None:
        """Prune min-clamped sources with negligible model support."""
        if st.num_sources <= 1:
            return
        if (
            float(self.config.weak_source_prune_min_expected_count) <= 0.0
            and float(self.config.weak_source_prune_min_fraction) <= 0.0
        ):
            return
        self._ensure_source_metadata(st)
        expected_counts = np.asarray(expected_counts, dtype=float).ravel()[
            : st.num_sources
        ]
        if expected_counts.size != st.num_sources:
            return
        total = float(np.sum(expected_counts))
        fraction = expected_counts / max(total, 1.0e-12)
        min_expected = max(float(self.config.weak_source_prune_min_expected_count), 0.0)
        min_fraction = max(float(self.config.weak_source_prune_min_fraction), 0.0)
        strength_floor = max(float(self.config.min_strength), 0.0)
        at_floor = (
            st.strengths[: st.num_sources] <= strength_floor * (1.0 + 1.0e-6)
        )
        weak_count = (
            expected_counts < min_expected
            if min_expected > 0.0
            else np.zeros(st.num_sources, dtype=bool)
        )
        weak_fraction = (
            fraction < min_fraction
            if min_fraction > 0.0
            else np.zeros(st.num_sources, dtype=bool)
        )
        drop = at_floor & (weak_count | weak_fraction)
        if not np.any(drop):
            return
        if np.count_nonzero(~drop) == 0:
            keep_idx = int(np.argmax(expected_counts))
            drop[keep_idx] = False
        if not np.any(drop):
            return
        keep = ~drop
        self.last_kill_count += int(np.count_nonzero(drop))
        st.positions = st.positions[keep]
        st.strengths = st.strengths[keep]
        st.ages = st.ages[keep]
        st.low_q_streaks = st.low_q_streaks[keep]
        st.support_scores = st.support_scores[keep]
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
        target.low_q_streaks = np.asarray(trial.low_q_streaks, dtype=int).copy()
        target.support_scores = np.asarray(trial.support_scores, dtype=float).copy()
        target.num_sources = int(target.positions.shape[0])

    def _trial_log_likelihood(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> float:
        """Return the configured count log-likelihood for one fixed state."""
        _, lambda_total = self._lambda_components(st, data)
        return self._count_log_likelihood_np(
            data.z_k,
            lambda_total,
            observation_count_variance=data.observation_variances,
        )

    def _structural_acceptance_threshold(
        self,
        *,
        base_threshold: float,
        complexity_penalty: float,
    ) -> float:
        """Return the likelihood-gain threshold for one structural parameter jump."""
        return float(base_threshold) + max(float(complexity_penalty), 0.0)

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
            kernel_sums = np.resize(kernel_sums, count)
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
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best residual-guided split trial and its likelihood gain.

        The proposal adds a residual-supported candidate as a split-off
        component from an existing high-strength source, then refits all
        strengths jointly with fixed positions. This is a general RJ-style
        structural move: it is accepted only when the observed count vector is
        better explained by two same-isotope components than by the original
        component.
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
        base_ll = self._trial_log_likelihood(st, data)
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
            candidate_strengths_arr = np.asarray(candidate_strengths, dtype=float).reshape(-1)
            if candidate_strengths_arr.size:
                copy_count = min(cand_strengths.size, candidate_strengths_arr.size)
                cand_strengths[:copy_count] = candidate_strengths_arr[:copy_count]
        ranked_sources = eligible[np.argsort(st.strengths[eligible])[::-1]]
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        min_sep = max(float(self.config.birth_min_sep_m), 0.0)
        for source_idx in ranked_sources:
            for cand_idx in range(candidate_count):
                pos_new = np.clip(candidates[cand_idx], lo, hi)
                if st.num_sources > 0:
                    dists = np.linalg.norm(
                        st.positions[: st.num_sources] - pos_new[None, :],
                        axis=1,
                    )
                    dists[int(source_idx)] = np.inf
                    if np.any(dists < min_sep):
                        continue
                    if np.linalg.norm(st.positions[int(source_idx)] - pos_new) < 0.5 * min_sep:
                        continue
                trial = st.copy()
                self._ensure_source_metadata(trial)
                q_new = max(float(cand_strengths[cand_idx]), float(self.config.min_strength))
                keep_strength = max(
                    float(trial.strengths[int(source_idx)]) - q_new,
                    float(self.config.min_strength),
                )
                trial.strengths[int(source_idx)] = keep_strength
                trial.positions = np.vstack([trial.positions[: trial.num_sources], pos_new])
                trial.strengths = np.append(trial.strengths[: trial.num_sources], q_new)
                trial.ages = np.append(trial.ages[: trial.num_sources], 0)
                trial.low_q_streaks = np.append(trial.low_q_streaks[: trial.num_sources], 0)
                trial.support_scores = np.append(trial.support_scores[: trial.num_sources], 0.0)
                trial.num_sources = int(trial.positions.shape[0])
                self._refit_strengths_for_particle(
                    trial,
                    data,
                    iters=max(1, int(self.config.refit_iters)),
                    eps=float(self.config.refit_eps),
                )
                self._prune_floor_sources_after_refit(trial, data)
                if trial.num_sources <= st.num_sources:
                    continue
                ll_after = self._trial_log_likelihood(trial, data)
                delta_ll = float(ll_after - base_ll)
                if delta_ll > best_delta:
                    best_delta = delta_ll
                    best_trial = trial
        return best_trial, best_delta

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

    def _best_merge_trial(
        self,
        st: IsotopeState,
        data: MeasurementData,
    ) -> tuple[IsotopeState | None, float]:
        """
        Return the best likelihood-tested merge trial and its likelihood gain.

        Candidate pairs are selected either by spatial proximity or by nearly
        collinear response signatures over the actual measurement block. The
        merged state is accepted only if the joint refit does not reduce the
        configured count likelihood beyond ``merge_delta_ll_threshold``.
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
        if not pair_scores:
            return None, -np.inf
        pair_scores.sort(reverse=True)
        max_pairs = max(1, int(self.config.merge_search_topk_pairs))
        best_trial: IsotopeState | None = None
        best_delta = -np.inf
        for _, i, j in pair_scores[:max_pairs]:
            q1 = float(st.strengths[i])
            q2 = float(st.strengths[j])
            if q1 + q2 > 0.0:
                merged_pos = (q1 * st.positions[i] + q2 * st.positions[j]) / (q1 + q2)
            else:
                merged_pos = 0.5 * (st.positions[i] + st.positions[j])
            keep = np.ones(st.num_sources, dtype=bool)
            keep[[i, j]] = False
            trial = IsotopeState(
                num_sources=int(np.count_nonzero(keep) + 1),
                positions=np.vstack([st.positions[keep], merged_pos]),
                strengths=np.append(st.strengths[keep], q1 + q2),
                background=float(st.background),
                ages=np.append(st.ages[keep], max(int(st.ages[i]), int(st.ages[j]))),
                low_q_streaks=np.append(
                    st.low_q_streaks[keep],
                    min(int(st.low_q_streaks[i]), int(st.low_q_streaks[j])),
                ),
                support_scores=np.append(
                    st.support_scores[keep],
                    max(float(st.support_scores[i]), float(st.support_scores[j])),
                ),
            )
            self._refit_strengths_for_particle(
                trial,
                data,
                iters=max(1, int(self.config.refit_iters)),
                eps=float(self.config.refit_eps),
            )
            self._prune_floor_sources_after_refit(trial, data)
            ll_after = self._trial_log_likelihood(trial, data)
            delta_ll = float(ll_after - base_ll)
            if delta_ll > best_delta:
                best_delta = delta_ll
                best_trial = trial
        return best_trial, best_delta

    def _source_detector_exclusion_mask(
        self,
        st: IsotopeState,
        data: MeasurementData | None,
    ) -> NDArray[np.bool_]:
        """Return source mask enforcing that sources cannot occupy detector poses."""
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

    def refresh_weights_from_measurements(self, data: MeasurementData | None) -> None:
        """
        Recompute particle weights from a measurement block after structural moves.

        Birth, death, split, and merge moves change the state dimension.  When
        those moves are proposed after a station-level resampling step, the
        modified particles must be reweighted by the same station likelihood;
        otherwise a proposal can affect the reported posterior without being
        judged by the observation that triggered it.
        """
        if data is None or data.z_k.size == 0 or not self.continuous_particles:
            return
        log_likelihoods = np.full(len(self.continuous_particles), -np.inf, dtype=float)
        grouped, fallback_indices = self._particle_indices_by_source_count()
        for source_count, particle_indices in grouped.items():
            _, lambda_total = self._lambda_components_for_particle_group(
                data,
                particle_indices,
                source_count,
            )
            group_ll = self._count_log_likelihood_matrix_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
            log_likelihoods[np.asarray(particle_indices, dtype=int)] = group_ll
        for idx in fallback_indices:
            st = self.continuous_particles[idx].state
            _, lambda_total = self._lambda_components(st, data)
            log_likelihoods[idx] = self._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        norm = logsumexp(log_likelihoods)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, log_likelihoods - norm):
            particle.log_weight = float(value)

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
            return float(np.median(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 0.0
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

    def estimate_clustered(self, max_k: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Estimate source positions/strengths by robust posterior clustering.

        Source-existence ordering uses posterior mass, while the reported
        position and strength use weighted medians.  This avoids high-intensity
        posterior tails dominating the displayed estimate without changing the
        PF update itself.
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
            for pos, q in zip(st.positions, st.strengths):
                positions.append(np.asarray(pos, dtype=float))
                weights.append(float(w))
                strengths.append(float(q))
        if not positions:
            return np.zeros((0, 3)), np.zeros(0)
        pos_arr = np.vstack(positions)
        w_arr = np.asarray(weights, dtype=float)
        q_arr = np.asarray(strengths, dtype=float)
        eps = float(self.config.cluster_eps_m)
        if eps <= 0.0:
            eps = 1e-6
        min_samples = max(1, int(self.config.cluster_min_samples))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self.estimate()
        tree = cKDTree(pos_arr)
        assigned = np.zeros(pos_arr.shape[0], dtype=bool)
        clusters: list[NDArray[np.int64]] = []
        for idx in range(pos_arr.shape[0]):
            if assigned[idx]:
                continue
            queue = [idx]
            members: list[int] = []
            while queue:
                j = queue.pop()
                if assigned[j]:
                    continue
                assigned[j] = True
                members.append(j)
                neighbors = tree.query_ball_point(pos_arr[j], r=eps)
                for n in neighbors:
                    if not assigned[n]:
                        queue.append(int(n))
            if len(members) >= min_samples:
                clusters.append(np.array(members, dtype=int))
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
        pos_out = np.vstack([cluster_pos[i] for i in order]) if order.size else np.zeros((0, 3))
        q_out = np.array([cluster_q[i] for i in order], dtype=float) if order.size else np.zeros(0, dtype=float)
        return pos_out, q_out

    def apply_birth_death(
        self,
        support_data: MeasurementData | None,
        birth_data: MeasurementData | None,
        candidate_positions: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Apply hysteretic death, residual-driven birth, and split/merge proposals.
        """
        if not self.continuous_particles:
            return
        if not bool(self.config.birth_enable):
            return
        structural_data = birth_data if birth_data is not None else support_data
        if structural_data is None or structural_data.z_k.size == 0:
            return
        structural_min = max(float(self.config.structural_update_min_counts), 0.0)
        if structural_min > 0.0 and float(np.max(structural_data.z_k)) < structural_min:
            self.last_birth_residual_chi2 = 0.0
            self.last_birth_residual_p_value = 1.0
            self.last_birth_residual_support = 0
            self.last_birth_residual_distinct_poses = 0
            self.last_birth_residual_distinct_stations = 0
            self.last_birth_residual_gate_passed = False
            return
        min_distinct = max(1, int(self.config.birth_min_distinct_poses))
        min_stations = max(1, int(self.config.birth_min_distinct_stations))
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
            if distinct_count < min_distinct or station_count < min_stations:
                self.last_birth_residual_chi2 = 0.0
                self.last_birth_residual_p_value = 1.0
                self.last_birth_residual_support = 0
                self.last_birth_residual_distinct_poses = int(distinct_count)
                self.last_birth_residual_distinct_stations = int(station_count)
                self.last_birth_residual_gate_passed = False
                return
        birth_proposal = self._compute_birth_proposal(birth_data, candidate_positions)
        if birth_proposal is not None:
            birth_probs, birth_kernel_sums, residual_sum, birth_candidates = birth_proposal
        else:
            birth_probs = None
            birth_kernel_sums = None
            residual_sum = 0.0
            birth_candidates = None
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
        refit_data = proposal_data
        max_births = self.config.birth_max_per_update
        births_remaining = (
            None if max_births is None else max(0, int(max_births))
        )
        any_moved = False
        moved_refit_indices: list[int] = []
        has_support_data = support_data is not None and support_data.z_k.size > 0
        support_cache: dict[
            int,
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        structural_proposal_indices: set[int] | None = None
        topk_structural = self.config.structural_proposal_topk_particles
        if topk_structural is not None:
            topk_count = int(topk_structural)
            if topk_count > 0 and topk_count < len(self.continuous_particles):
                weights = np.asarray(self.continuous_weights, dtype=float)
                structural_proposal_indices = set(
                    int(idx) for idx in np.argsort(weights)[::-1][:topk_count]
                )
        if has_support_data:
            grouped, fallback_indices = self._particle_indices_by_source_count()
            for source_count, particle_indices in grouped.items():
                if source_count <= 0 or not particle_indices:
                    continue
                lambda_m_group, lambda_total_group = self._lambda_components_for_particle_group(
                    support_data,
                    particle_indices,
                    source_count,
                )
                delta_ll_group = self._delta_log_likelihood_remove_group(
                    support_data,
                    lambda_total_group,
                    lambda_m_group,
                )
                for row_idx, particle_idx in enumerate(particle_indices):
                    support_cache[int(particle_idx)] = (
                        lambda_m_group[:, row_idx, :],
                        lambda_total_group[:, row_idx],
                        delta_ll_group[row_idx],
                    )
            for particle_idx in fallback_indices:
                st = self.continuous_particles[particle_idx].state
                if st.num_sources <= 0:
                    continue
                lambda_m, lambda_total = self._lambda_components(st, support_data)
                delta_ll = self._delta_log_likelihood_remove(
                    support_data.z_k,
                    lambda_total,
                    lambda_m,
                    observation_count_variance=support_data.observation_variances,
                )
                support_cache[int(particle_idx)] = (lambda_m, lambda_total, delta_ll)

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
                below = st.strengths < float(self.config.min_strength)
                st.low_q_streaks[below] += 1
                st.low_q_streaks[~below] = 0
            lambda_m = None
            lambda_total = None
            if has_support and st.num_sources > 0:
                cached_support = support_cache.get(int(particle_idx))
                if cached_support is not None and cached_support[2].size == st.num_sources:
                    lambda_m, lambda_total, delta_ll = cached_support
                else:
                    lambda_m, lambda_total = self._lambda_components(st, support_data)
                    delta_ll = self._delta_log_likelihood_remove(
                        support_data.z_k,
                        lambda_total,
                        lambda_m,
                        observation_count_variance=support_data.observation_variances,
                    )
                alpha = float(self.config.support_ema_alpha)
                st.support_scores = (1.0 - alpha) * st.support_scores + alpha * delta_ll
            if st.num_sources > 0 and has_support:
                kill_mask = np.ones(st.num_sources, dtype=bool)
                exclusion_mask = self._source_detector_exclusion_mask(st, structural_data)
                kill_mask[~exclusion_mask] = False
                q_min = float(self.config.min_strength)
                if q_min <= 0.0:
                    q_min = float(self.config.birth_q_min)
                deterministic = (st.low_q_streaks >= int(self.config.death_low_q_streak)) & (
                    st.strengths < q_min
                )
                kill_mask[deterministic] = False
                kill_candidates = (st.low_q_streaks >= int(self.config.death_low_q_streak)) & (
                    st.support_scores < float(self.config.death_delta_ll_threshold)
                )
                for idx, do_kill in enumerate(kill_candidates):
                    if kill_mask[idx] and do_kill and np.random.rand() < float(self.config.p_kill):
                        kill_mask[idx] = False
                if not np.all(kill_mask):
                    self.last_kill_count += int(np.sum(~kill_mask))
                    st.positions = st.positions[kill_mask]
                    st.strengths = st.strengths[kill_mask]
                    st.ages = st.ages[kill_mask]
                    st.low_q_streaks = st.low_q_streaks[kill_mask]
                    st.support_scores = st.support_scores[kill_mask]
                    st.num_sources = st.positions.shape[0]
                    moved = True
                if self.config.max_sources is not None and st.num_sources > self.config.max_sources:
                    over = int(st.num_sources - self.config.max_sources)
                    if over > 0:
                        drop = np.argsort(st.support_scores)[:over]
                        keep = np.ones(st.num_sources, dtype=bool)
                        keep[drop] = False
                        st.positions = st.positions[keep]
                        st.strengths = st.strengths[keep]
                        st.ages = st.ages[keep]
                        st.low_q_streaks = st.low_q_streaks[keep]
                        st.support_scores = st.support_scores[keep]
                        st.num_sources = st.positions.shape[0]
                        moved = True

            can_try_split = (
                allow_structural_proposal
                and st.num_sources > 0
                and proposal_data is not None
                and proposal_data.z_k.size
            )
            if can_try_split:
                split_moved = False
                if self.config.max_sources is None or st.num_sources < self.config.max_sources:
                    try_residual_split = bool(self.config.split_residual_always_try) or (
                        np.random.rand() < float(self.config.split_prob)
                    )
                    if try_residual_split:
                        split_trial, split_delta = self._best_residual_guided_split_trial(
                            st,
                            proposal_data,
                            birth_candidates,
                            split_candidate_strengths,
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
                            self._replace_particle_state_from_trial(st, split_trial)
                            split_moved = True
                            moved = True
                    if not split_moved and np.random.rand() < float(self.config.split_prob):
                        candidates = np.where(
                            st.strengths >= float(self.config.split_strength_min)
                        )[0]
                        if candidates.size > 0:
                            idx = int(np.random.choice(candidates))
                            if st.ages[idx] > int(self.config.min_age_to_split):
                                split_lambda_m, split_lambda_total = self._lambda_components(
                                    st,
                                    proposal_data,
                                )
                                delta = np.random.normal(
                                    scale=float(self.config.split_position_sigma),
                                    size=3,
                                )
                                lo = np.array(self.config.position_min, dtype=float)
                                hi = np.array(self.config.position_max, dtype=float)
                                s1 = np.clip(st.positions[idx] + delta, lo, hi)
                                s2 = np.clip(st.positions[idx] - delta, lo, hi)
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
                                        source_scale=self._measurement_source_scale(),
                                    )
                                    lambda_new = (
                                        split_lambda_total
                                        - split_lambda_m[:, idx]
                                        + np.sum(lam_new, axis=1)
                                    )
                                    delta_ll = self._delta_log_likelihood_update(
                                        proposal_data.z_k,
                                        split_lambda_total,
                                        lambda_new,
                                        observation_count_variance=(
                                            proposal_data.observation_variances
                                        ),
                                    )
                                    split_threshold = self._structural_acceptance_threshold(
                                        base_threshold=float(
                                            self.config.split_delta_ll_threshold
                                        ),
                                        complexity_penalty=float(
                                            self.config.split_complexity_penalty
                                        ),
                                    )
                                    if delta_ll >= split_threshold and np.log(
                                        np.random.rand()
                                    ) < delta_ll:
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
                                        st.low_q_streaks = np.concatenate(
                                            [
                                                st.low_q_streaks[:idx],
                                                st.low_q_streaks[idx + 1 :],
                                                [0, 0],
                                            ]
                                        )
                                        st.support_scores = np.concatenate(
                                            [
                                                st.support_scores[:idx],
                                                st.support_scores[idx + 1 :],
                                                [0.0, 0.0],
                                            ]
                                        )
                                        st.num_sources = st.positions.shape[0]
                                        moved = True

            if (
                allow_structural_proposal
                and st.num_sources >= 2
                and proposal_data is not None
                and proposal_data.z_k.size
                and np.random.rand() < float(self.config.merge_prob)
            ):
                merge_trial, merge_delta = self._best_merge_trial(st, proposal_data)
                if (
                    merge_trial is not None
                    and merge_delta >= float(self.config.merge_delta_ll_threshold)
                ):
                    self._replace_particle_state_from_trial(st, merge_trial)
                    moved = True

            if (
                birth_probs is not None
                and birth_kernel_sums is not None
                and birth_candidates is not None
                and residual_sum > 0.0
                and (births_remaining is None or births_remaining > 0)
                and np.random.rand() < float(self.config.p_birth)
            ):
                if (
                    self.config.max_sources is not None
                    and st.num_sources >= self.config.max_sources
                ):
                    continue
                idx = int(np.random.choice(len(birth_probs), p=birth_probs))
                denom = float(birth_kernel_sums[idx])
                if denom <= 0.0:
                    continue
                q_new = float(self.config.birth_alpha) * residual_sum / max(denom, 1e-12)
                if q_new <= 0.0:
                    continue
                q_min = float(self.config.birth_q_min)
                q_max = float(self.config.birth_q_max)
                if q_max < q_min:
                    q_min, q_max = q_max, q_min
                q_new = float(np.clip(q_new, q_min, q_max))
                pos_new = birth_candidates[idx]
                if st.num_sources > 0:
                    dist = np.linalg.norm(st.positions - pos_new[None, :], axis=1)
                    if np.any(dist < float(self.config.birth_min_sep_m)):
                        continue
                trial = st.copy()
                self._ensure_source_metadata(trial)
                trial.positions = np.vstack(
                    [trial.positions[: trial.num_sources], pos_new]
                )
                trial.strengths = np.append(trial.strengths[: trial.num_sources], q_new)
                trial.ages = np.append(trial.ages[: trial.num_sources], 0)
                trial.low_q_streaks = np.append(
                    trial.low_q_streaks[: trial.num_sources],
                    0,
                )
                trial.support_scores = np.append(
                    trial.support_scores[: trial.num_sources],
                    0.0,
                )
                trial.num_sources = int(trial.positions.shape[0])
                base_ll = self._trial_log_likelihood(st, proposal_data)
                self._refit_strengths_for_particle(
                    trial,
                    proposal_data,
                    iters=max(1, int(self.config.refit_iters)),
                    eps=float(self.config.refit_eps),
                )
                self._prune_floor_sources_after_refit(trial, proposal_data)
                if trial.num_sources <= st.num_sources:
                    continue
                delta_ll = float(
                    self._trial_log_likelihood(trial, proposal_data) - base_ll
                )
                birth_threshold = self._structural_acceptance_threshold(
                    base_threshold=float(self.config.birth_delta_ll_threshold),
                    complexity_penalty=float(self.config.birth_complexity_penalty),
                )
                if not np.isfinite(delta_ll) or delta_ll < birth_threshold:
                    continue
                self._replace_particle_state_from_trial(st, trial)
                self.last_birth_count += 1
                if births_remaining is not None:
                    births_remaining -= 1
                moved = True

            if moved and refit_data is not None and bool(self.config.refit_after_moves):
                moved_refit_indices.append(int(particle_idx))
            any_moved = any_moved or moved

        if moved_refit_indices and refit_data is not None and bool(self.config.refit_after_moves):
            self._refit_particle_indices_batched(
                refit_data,
                moved_refit_indices,
                iters=int(self.config.refit_iters),
                eps=float(self.config.refit_eps),
            )
        if any_moved and refit_data is not None:
            self.refresh_weights_from_measurements(refit_data)
        self.align_continuous_labels()

    def _background_level(self) -> float:
        """Resolve per-isotope background level."""
        level = self.config.background_level
        if isinstance(level, dict):
            return float(level.get(self.isotope, 0.0))
        return float(level)

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Continuous MMSE estimate over positions/strengths using continuous_particles.
        """
        if self.config.converge_enable and self.is_converged and self.frozen_estimate is not None:
            return self.frozen_estimate
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        self._gpu_enabled()
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        states = [p.state for p in self.continuous_particles]
        positions_t, strengths_t, _, mask_t = gpu_utils.pack_states(states, device=device, dtype=dtype)
        weights = torch.as_tensor(self.continuous_weights, device=device, dtype=dtype)
        weight_sum = torch.sum(weights)
        if float(weight_sum) <= 0.0:
            weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
        else:
            weights = weights / weight_sum
        w_mask = weights[:, None] * mask_t
        w_sum = torch.sum(w_mask, dim=0)
        w_sum_safe = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
        pos_mean = torch.sum(w_mask[:, :, None] * positions_t, dim=0) / w_sum_safe[:, None]
        str_mean = torch.sum(w_mask * strengths_t, dim=0) / w_sum_safe
        positions = pos_mean.detach().cpu().numpy()
        strengths = str_mean.detach().cpu().numpy()
        # Trim zero-strength slots.
        mask = strengths > 0
        positions = positions[mask]
        strengths = strengths[mask]
        return positions, strengths
