"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
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
    ess_low: float = 0.5
    ess_high: float = 0.9
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (0, 3)  # inclusive range
    # Strength prior (cps@1m scale). Defaults cover ~1e3–1e5 cps via log-normal.
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
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


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float


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
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
            b_h = self._background_level()
            st = IsotopeState(num_sources=r_h, positions=positions, strengths=strengths, background=b_h)
            self.continuous_particles.append(IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N))))

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        if not gpu_utils.torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

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

    def _update_continuous_weights_gpu(self, lam_t: "torch.Tensor", z_obs: float) -> None:
        """Update continuous log-weights using torch on the configured device."""
        if lam_t.numel() == 0:
            return
        import torch

        logw = torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=lam_t.device,
            dtype=lam_t.dtype,
        )
        logw = logw + z_obs * torch.log(lam_t + 1e-12) - lam_t
        logw = logw - torch.max(logw)
        w = torch.exp(logw)
        w = w / torch.sum(w)
        w_cpu = w.detach().cpu().numpy()
        for p, wi in zip(self.continuous_particles, w_cpu):
            p.log_weight = float(np.log(wi + 1e-20))

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

    def update_continuous_pair(self, z_obs: float, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float) -> None:
        """
        Poisson log-weight update using Fe/Pb orientation indices (Eq. 3.41–3.44).

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        """
        self._gpu_enabled()
        lam_t = self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        self._update_continuous_weights_gpu(lam_t=lam_t, z_obs=z_obs)
        self._maybe_resample_continuous()
        self.adapt_num_particles()
        self.align_continuous_labels()

    def update_continuous_pair_at_pose(
        self,
        z_obs: float,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> None:
        """
        Poisson log-weight update using explicit detector position.

        This avoids reliance on pose indices for planning-time evaluations.
        """
        self._gpu_enabled()
        lam_t = self._continuous_expected_counts_pair_at_pose_torch(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        self._update_continuous_weights_gpu(lam_t=lam_t, z_obs=z_obs)
        self._maybe_resample_continuous()
        self.adapt_num_particles()
        self.align_continuous_labels()

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        w = np.exp([p.log_weight for p in self.continuous_particles])
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles)) / len(self.continuous_particles)
        return w / s

    def _maybe_resample_continuous(self) -> None:
        """ESS check and systematic resampling for continuous particles (Sec. 3.3.4, Eq. 3.29)."""
        w = self.continuous_weights
        ess = 1.0 / np.sum(w**2)
        if ess < self.config.resample_threshold * self.N:
            idx = systematic_resample(np.log(w))
            self.continuous_particles = [self.continuous_particles[i].state.copy() for i in idx]
            # reset weights to uniform
            self.continuous_particles = [
                IsotopeParticle(state=st, log_weight=float(-np.log(self.N))) for st in self.continuous_particles
            ]
            self.regularize_continuous(
                sigma_pos=self.config.position_sigma,
                sigma_int=self.config.strength_sigma,
                p_birth=self.config.p_birth,
                p_kill=self.config.p_kill,
                intensity_threshold=self.config.min_strength,
            )

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
        used_rows: set[int] = set()
        for ref_idx in range(n_cols):
            row = assigned.get(ref_idx)
            if row is None:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
            used_rows.add(row)
        for row in range(n_rows):
            if row in used_rows:
                continue
            ordered_pos.append(st.positions[row])
            ordered_str.append(float(st.strengths[row]))
        if ordered_pos:
            st.positions = np.vstack(ordered_pos)
            st.strengths = np.array(ordered_str, dtype=float)
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

    def adapt_num_particles(self) -> None:
        """
        Optional: adapt N based on variance/entropy of weights (Chapter 3.3.4).
        """
        if not self.continuous_particles:
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
        w = self.continuous_weights
        ess = 1.0 / np.sum(w**2)
        ess_ratio = ess / max(len(w), 1)
        if ess_ratio < self.config.ess_low and len(w) < max_particles:
            target = min(max_particles, max(len(w) + 1, int(len(w) * 1.25)))
            self._resample_continuous_to(target, jitter=True)
        elif ess_ratio > self.config.ess_high and len(w) > min_particles:
            target = max(min_particles, int(len(w) * 0.8))
            self._resample_continuous_to(target, jitter=False)

    def _resample_continuous_to(self, target_n: int, jitter: bool = False) -> None:
        """Resample the continuous particles to a new population size."""
        target_n = max(1, int(target_n))
        w = self.continuous_weights
        idx = np.random.choice(len(self.continuous_particles), size=target_n, p=w)
        states = [self.continuous_particles[i].state.copy() for i in idx]
        self.continuous_particles = [
            IsotopeParticle(state=st, log_weight=float(-np.log(target_n))) for st in states
        ]
        self.N = target_n
        self.config.num_particles = target_n
        if jitter:
            self.regularize_continuous(
                sigma_pos=self.config.position_sigma,
                sigma_int=self.config.strength_sigma,
                p_birth=self.config.p_birth,
                p_kill=self.config.p_kill,
                intensity_threshold=self.config.min_strength,
            )

    def best_particle(self) -> IsotopeParticle:
        """Return the particle with maximum log_weight."""
        return max(self.continuous_particles, key=lambda p: p.log_weight)

    def regularize_continuous(
        self,
        sigma_pos: float = 0.05,
        sigma_int: float = 0.05,
        p_birth: float = 0.05,
        p_kill: float = 0.1,
        intensity_threshold: float = 0.05,
    ) -> None:
        """
        Apply small Gaussian jitter to positions/strengths and simple birth/death moves (Sec. 3.3.4).

        - positions: s <- s + N(0, sigma_pos^2 I)
        - strengths: q <- max(q + N(0, sigma_int^2), intensity_threshold)
        - delete sources with q < intensity_threshold with prob p_kill
        - with prob p_birth, add a new source uniformly in workspace with small initial strength
        """
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        max_sources = self.config.max_sources
        for p in self.continuous_particles:
            st = p.state
            st.background = self._background_level()
            if st.positions.size:
                st.positions = st.positions + np.random.normal(scale=sigma_pos, size=st.positions.shape)
                st.positions = np.clip(st.positions, lo, hi)
                st.strengths = st.strengths + np.random.normal(scale=sigma_int, size=st.strengths.shape)
                st.strengths = np.maximum(st.strengths, 0.0)
                # kill weak sources
                mask = np.ones(st.num_sources, dtype=bool)
                for i, q in enumerate(st.strengths):
                    if q < intensity_threshold and np.random.rand() < p_kill:
                        mask[i] = False
                st.positions = st.positions[mask]
                st.strengths = np.maximum(st.strengths[mask], intensity_threshold)
                st.num_sources = st.positions.shape[0]
            # birth
            if np.random.rand() < p_birth:
                if max_sources is not None and st.num_sources >= max_sources:
                    continue
                new_pos = lo + np.random.rand(3) * (hi - lo)
                new_strength = float(np.abs(np.random.normal(loc=0.1, scale=0.05)))
                st.positions = np.vstack([st.positions, new_pos])
                st.strengths = np.append(st.strengths, new_strength)
                st.num_sources = st.positions.shape[0]

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
