"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
from collections import deque

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from pf.likelihood import delta_log_likelihood_remove, delta_log_likelihood_update, expected_counts_per_source
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
    birth_candidate_jitter_sigma: float = 0.5
    birth_num_local_jitter: int = 8
    birth_alpha: float = 0.2
    birth_q_max: float = 3e5
    birth_q_min: float = 1e2
    birth_residual_clip_quantile: float = 0.95
    refit_after_moves: bool = True
    refit_iters: int = 3
    refit_eps: float = 1e-12
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
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
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
    converge_window: int = 10
    converge_map_move_eps_m: float = 0.2
    converge_ess_ratio_high: float = 0.9
    converge_ll_improve_eps: float = 0.0
    converge_min_steps: int = 20
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
        map_pos: NDArray[np.float64] | None,
        ess_ratio: float,
        ll_value: float,
    ) -> None:
        """Append the latest statistics to the window."""
        if step_idx < 0:
            return
        self.positions.append(map_pos.copy() if map_pos is not None else None)
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
    ) -> None:
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        mu_by_isotope = getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        shield_params = getattr(kernel, "shield_params", ShieldParams()) if kernel is not None else ShieldParams()
        self.continuous_kernel = ContinuousKernel(
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

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a kernel and refresh the continuous-kernel configuration."""
        self.kernel = kernel
        self.continuous_kernel = ContinuousKernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )

    def _init_continuous_particles(self) -> None:
        """Sample continuous positions/strengths/background from broad priors (Sec. 3.3.2)."""
        self.continuous_particles = []
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
        if state.num_sources > 0:
            for pos, strength in zip(state.positions[:state.num_sources], state.strengths[:state.num_sources]):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
                lam_rate += float(kernel_val) * float(strength)
        lam = float(live_time_s) * lam_rate
        eps = 1e-12
        return float(z_obs * np.log(lam + eps) - lam)

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
        best_state = self.best_particle().state
        map_pos = best_state.positions[0].copy() if best_state.num_sources > 0 else None
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
        self._converge_monitor.update_stats(step_idx, map_pos, ess_ratio, ll_value)
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
            use_angle_attenuation=shield_params.use_angle_attenuation,
            live_time_s=live_time_s,
            device=device,
            dtype=dtype,
        )

    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _log_likelihood_increment_gpu(self, lam_t: "torch.Tensor", z_obs: float) -> "torch.Tensor":
        """Return the per-particle Poisson log-likelihood increment in float64."""
        import torch

        lam_t = lam_t.to(dtype=torch.float64)
        lam_t = torch.clamp(lam_t, min=1e-12)
        z = torch.as_tensor(z_obs, device=lam_t.device, dtype=torch.float64)
        return z * torch.log(lam_t) - lam_t

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
        ll_t = ll_t if ll_t is not None else self._log_likelihood_increment_gpu(lam_t, z_obs)
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
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering for a single Poisson update.

        The update increments beta from 0 to 1 using delta_beta steps that
        maintain ESS above the configured target ratio when possible.

        Returns (ess_pre, resampled_any) for downstream adaptation logic.
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
        lam_t = lam_fn()
        if lam_t.numel() == 0:
            self.last_temper_steps = []
            self.last_temper_resample_count = 0
            return 0.0, False
        logw = self._current_log_weights_torch(lam_t.device)
        ll_t = self._log_likelihood_increment_gpu(lam_t, z_obs)

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
                    lam_t = lam_fn()
                    if lam_t.numel() == 0:
                        break
                    logw = self._current_log_weights_torch(lam_t.device)
                    ll_t = self._log_likelihood_increment_gpu(lam_t, z_obs)
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
            use_angle_attenuation=shield_params.use_angle_attenuation,
            live_time_s=live_time_s,
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
        step_idx: int | None = None,
    ) -> None:
        """
        Poisson log-weight update using Fe/Pb orientation indices (Eq. 3.41–3.44).

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
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

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update(lam_fn=_lam_fn, z_obs=z_obs)
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(lam_t, z_obs, return_logw=True)
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

    def update_continuous_pair_at_pose(
        self,
        z_obs: float,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        step_idx: int | None = None,
    ) -> None:
        """
        Poisson log-weight update using explicit detector position.

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
            ess_pre, resampled_any = self._tempered_update(lam_fn=_lam_fn, z_obs=z_obs)
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(lam_t, z_obs, return_logw=True)
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
        self._gpu_enabled()
        import torch
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
        ref_pos_t = torch.as_tensor(ref_positions, device=device, dtype=dtype)
        str_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        ref_str_t = torch.as_tensor(ref_strengths, device=device, dtype=dtype)
        if pos_t.numel() == 0 or ref_pos_t.numel() == 0:
            return np.zeros((positions.shape[0], ref_positions.shape[0]), dtype=float)
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
        )
        background_counts = float(st.background) * data.live_times
        lambda_total = background_counts + np.sum(lambda_m, axis=1)
        return lambda_m, lambda_total

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

        num_jitter = max(0, int(self.config.birth_num_local_jitter))
        if num_jitter > 0:
            jitter_sigma = float(self.config.birth_candidate_jitter_sigma)
            jitter = np.random.normal(
                loc=0.0,
                scale=jitter_sigma,
                size=(candidate_positions.shape[0], num_jitter, 3),
            )
            jittered = candidate_positions[:, None, :] + jitter
            lo = np.array(self.config.position_min, dtype=float)
            hi = np.array(self.config.position_max, dtype=float)
            jittered = np.clip(jittered, lo, hi)
            candidates = np.vstack([candidate_positions, jittered.reshape(-1, 3)])
        else:
            candidates = candidate_positions.copy()

        scores = np.zeros(candidates.shape[0], dtype=float)
        kernel_sums = np.zeros(candidates.shape[0], dtype=float)
        for j, pos in enumerate(candidates):
            k_sum = 0.0
            s_sum = 0.0
            for k in range(data.z_k.size):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=data.detector_positions[k],
                    source_pos=pos,
                    fe_index=int(data.fe_indices[k]),
                    pb_index=int(data.pb_indices[k]),
                )
                contrib = float(data.live_times[k]) * kernel_val
                k_sum += contrib
                s_sum += residual_mix[k] * contrib
            kernel_sums[j] = k_sum
            scores[j] = s_sum
        if np.max(scores) <= 0.0:
            return None
        scores = np.maximum(scores, float(self.config.birth_min_score))
        temp = max(float(self.config.birth_softmax_temp), 1e-6)
        scaled = scores / temp
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        return probs, kernel_sums, residual_sum, candidates

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
    ) -> None:
        """
        Refit per-source strengths using coordinate ascent with fixed positions.

        This stabilizes source intensities after birth/kill/split/merge moves.
        """
        if st.num_sources <= 0 or data.z_k.size == 0:
            return
        num_sources = int(st.num_sources)
        num_meas = int(data.z_k.size)
        k_mat = np.zeros((num_meas, num_sources), dtype=float)
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
                k_mat[k, j] = float(data.live_times[k]) * kernel_val
        q_min = float(self.config.birth_q_min)
        q_max = float(self.config.birth_q_max)
        if q_max < q_min:
            q_min, q_max = q_max, q_min
        strengths = np.asarray(st.strengths, dtype=float)
        background_counts = float(st.background) * data.live_times
        lambda_total = background_counts + k_mat @ strengths
        for _ in range(max(1, int(iters))):
            for j in range(num_sources):
                k_col = k_mat[:, j]
                denom = float(np.sum(k_col * k_col) + float(eps))
                if denom <= 0.0:
                    strengths[j] = 0.0
                    continue
                residual = data.z_k - (lambda_total - strengths[j] * k_col)
                numer = float(np.sum(residual * k_col))
                q_new = max(0.0, numer / denom)
                q_new = float(np.clip(q_new, q_min, q_max))
                if q_new != strengths[j]:
                    lambda_total = lambda_total - strengths[j] * k_col + q_new * k_col
                    strengths[j] = q_new
        st.strengths = strengths
        st.num_sources = st.positions.shape[0]

    def estimate_clustered(self, max_k: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Estimate source positions/strengths by clustering weighted particle sources.
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
        cluster_strength: list[float] = []
        for members in clusters:
            w = w_arr[members]
            if np.sum(w) <= 0.0:
                w = np.ones_like(w, dtype=float)
            w = w / np.sum(w)
            pos_mean = np.sum(w[:, None] * pos_arr[members], axis=0)
            q_weighted = float(np.sum(w * q_arr[members]))
            cluster_pos.append(pos_mean)
            cluster_q.append(q_weighted)
            cluster_strength.append(float(np.sum(w_arr[members] * q_arr[members])))
        order = np.argsort(cluster_strength)[::-1]
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
        birth_proposal = self._compute_birth_proposal(birth_data, candidate_positions)
        if birth_proposal is not None:
            birth_probs, birth_kernel_sums, residual_sum, birth_candidates = birth_proposal
        else:
            birth_probs = None
            birth_kernel_sums = None
            residual_sum = 0.0
            birth_candidates = None
        refit_data = None
        if support_data is not None and support_data.z_k.size:
            refit_data = support_data
        elif birth_data is not None and birth_data.z_k.size:
            refit_data = birth_data

        for particle in self.continuous_particles:
            st = particle.state
            self._ensure_source_metadata(st)
            has_support = support_data is not None and support_data.z_k.size > 0
            moved = False
            if st.num_sources > 0:
                st.ages = st.ages + 1
                below = st.strengths < float(self.config.min_strength)
                st.low_q_streaks[below] += 1
                st.low_q_streaks[~below] = 0
            lambda_m = None
            lambda_total = None
            if has_support and st.num_sources > 0:
                lambda_m, lambda_total = self._lambda_components(st, support_data)
                delta_ll = delta_log_likelihood_remove(
                    support_data.z_k,
                    lambda_total,
                    lambda_m,
                )
                alpha = float(self.config.support_ema_alpha)
                st.support_scores = (1.0 - alpha) * st.support_scores + alpha * delta_ll
            if st.num_sources > 0 and has_support:
                kill_mask = np.ones(st.num_sources, dtype=bool)
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

            if (
                st.num_sources > 0
                and support_data is not None
                and support_data.z_k.size
                and np.random.rand() < float(self.config.split_prob)
            ):
                if self.config.max_sources is not None and st.num_sources >= self.config.max_sources:
                    continue
                candidates = np.where(st.strengths >= float(self.config.split_strength_min))[0]
                if candidates.size > 0:
                    idx = int(np.random.choice(candidates))
                    if st.ages[idx] <= int(self.config.min_age_to_split):
                        continue
                    if lambda_total is None or lambda_m is None:
                        lambda_m, lambda_total = self._lambda_components(st, support_data)
                    delta = np.random.normal(scale=float(self.config.split_position_sigma), size=3)
                    lo = np.array(self.config.position_min, dtype=float)
                    hi = np.array(self.config.position_max, dtype=float)
                    s1 = np.clip(st.positions[idx] + delta, lo, hi)
                    s2 = np.clip(st.positions[idx] - delta, lo, hi)
                    if np.linalg.norm(s1 - s2) < 0.5 * float(self.config.birth_min_sep_m):
                        continue
                    u_min = float(self.config.split_strength_min_frac)
                    u_max = float(self.config.split_strength_max_frac)
                    u_low, u_high = (u_min, u_max) if u_min <= u_max else (u_max, u_min)
                    u = np.random.uniform(u_low, u_high)
                    q1 = float(st.strengths[idx]) * float(u)
                    q2 = float(st.strengths[idx]) * float(1.0 - u)
                    lam_new = expected_counts_per_source(
                        kernel=self.continuous_kernel,
                        isotope=self.isotope,
                        detector_positions=support_data.detector_positions,
                        sources=np.vstack([s1, s2]),
                        strengths=np.array([q1, q2], dtype=float),
                        live_times=support_data.live_times,
                        fe_indices=support_data.fe_indices,
                        pb_indices=support_data.pb_indices,
                    )
                    lambda_new = lambda_total - lambda_m[:, idx] + np.sum(lam_new, axis=1)
                    delta_ll = delta_log_likelihood_update(
                        support_data.z_k,
                        lambda_total,
                        lambda_new,
                    )
                    if delta_ll >= float(self.config.split_delta_ll_threshold):
                        if np.log(np.random.rand()) < delta_ll:
                            st.positions = np.vstack([st.positions[:idx], st.positions[idx + 1 :], s1, s2])
                            st.strengths = np.concatenate(
                                [st.strengths[:idx], st.strengths[idx + 1 :], [q1, q2]]
                            )
                            st.ages = np.concatenate([st.ages[:idx], st.ages[idx + 1 :], [0, 0]])
                            st.low_q_streaks = np.concatenate(
                                [st.low_q_streaks[:idx], st.low_q_streaks[idx + 1 :], [0, 0]]
                            )
                            st.support_scores = np.concatenate(
                                [st.support_scores[:idx], st.support_scores[idx + 1 :], [0.0, 0.0]]
                            )
                            st.num_sources = st.positions.shape[0]
                            moved = True

            if (
                st.num_sources >= 2
                and support_data is not None
                and support_data.z_k.size
                and np.random.rand() < float(self.config.merge_prob)
            ):
                pos = st.positions
                diff = pos[:, None, :] - pos[None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                dist = np.where(np.eye(dist.shape[0], dtype=bool), np.inf, dist)
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if dist[i, j] <= float(self.config.merge_distance_max):
                    if (
                        lambda_total is None
                        or lambda_m is None
                        or lambda_m.shape[1] != st.num_sources
                    ):
                        lambda_m, lambda_total = self._lambda_components(st, support_data)
                    q1 = float(st.strengths[i])
                    q2 = float(st.strengths[j])
                    if q1 + q2 > 0.0:
                        merged_pos = (q1 * st.positions[i] + q2 * st.positions[j]) / (q1 + q2)
                    else:
                        merged_pos = 0.5 * (st.positions[i] + st.positions[j])
                    merged_strength = q1 + q2
                    lam_merge = expected_counts_per_source(
                        kernel=self.continuous_kernel,
                        isotope=self.isotope,
                        detector_positions=support_data.detector_positions,
                        sources=np.array([merged_pos]),
                        strengths=np.array([merged_strength], dtype=float),
                        live_times=support_data.live_times,
                        fe_indices=support_data.fe_indices,
                        pb_indices=support_data.pb_indices,
                    )
                    lambda_new = lambda_total - lambda_m[:, i] - lambda_m[:, j] + lam_merge[:, 0]
                    delta_ll = delta_log_likelihood_update(
                        support_data.z_k,
                        lambda_total,
                        lambda_new,
                    )
                    if delta_ll >= float(self.config.merge_delta_ll_threshold):
                        if np.log(np.random.rand()) < delta_ll:
                            keep = np.ones(st.num_sources, dtype=bool)
                            keep[[i, j]] = False
                            st.positions = np.vstack([st.positions[keep], merged_pos])
                            st.strengths = np.append(st.strengths[keep], merged_strength)
                            st.ages = np.append(st.ages[keep], max(int(st.ages[i]), int(st.ages[j])))
                            st.low_q_streaks = np.append(
                                st.low_q_streaks[keep], min(int(st.low_q_streaks[i]), int(st.low_q_streaks[j]))
                            )
                            st.support_scores = np.append(
                                st.support_scores[keep], max(float(st.support_scores[i]), float(st.support_scores[j]))
                            )
                            st.num_sources = st.positions.shape[0]
                            moved = True

            if (
                birth_probs is not None
                and birth_kernel_sums is not None
                and birth_candidates is not None
                and residual_sum > 0.0
                and np.random.rand() < float(self.config.p_birth)
            ):
                if self.config.max_sources is not None and st.num_sources >= self.config.max_sources:
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
                st.positions = np.vstack([st.positions, pos_new])
                st.strengths = np.append(st.strengths, q_new)
                st.ages = np.append(st.ages, 0)
                st.low_q_streaks = np.append(st.low_q_streaks, 0)
                st.support_scores = np.append(st.support_scores, 0.0)
                st.num_sources = st.positions.shape[0]
                self.last_birth_count += 1
                moved = True

            if moved and refit_data is not None and bool(self.config.refit_after_moves):
                self._refit_strengths_for_particle(
                    st,
                    refit_data,
                    iters=int(self.config.refit_iters),
                    eps=float(self.config.refit_eps),
                )

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
