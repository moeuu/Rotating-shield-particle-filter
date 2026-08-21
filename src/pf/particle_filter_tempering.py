"""GPU transport, weight normalization, and adaptive tempering for the PF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pf.particle_types import TorchLineTransportComponents

if TYPE_CHECKING:
    import torch


class TemperingIncrementRequiresRejuvenation(RuntimeError):
    """Signal that no configured positive beta increment preserves target ESS."""


class ParticleTemperingMixin:
    """Provide batched transport, weight, ESS, and tempering algorithms."""

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        gpu_utils.require_torch_compute_device(
            str(self.config.gpu_device),
            str(self.config.gpu_dtype),
        )
        return True

    def _can_use_gpu(self) -> bool:
        """Select explicit NumPy mode or require the configured torch device."""
        if not self.config.use_gpu:
            return False
        return self._gpu_enabled()

    def _continuous_expected_line_transport_components_pair_sequence_torch(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        positive_line_indices: NDArray[np.int64],
    ) -> TorchLineTransportComponents:
        """Return batched source-resolved line-rate transport components.

        Total and uncollided arrays have shape particle x view x source-slot x
        line and contain strength-scaled rates before branching fractions and
        live time.  Geometry features share that shape and are independent of
        source strength.  The full-spectrum model is the only component that
        converts rates to finite-live-time event counts.
        """
        from pf import gpu_utils
        import torch

        fe_arr = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size and fe_arr.size > 0):
            raise ValueError(
                "Fe, Pb, and live-time arrays must have one common positive view count."
            )
        if np.any(~np.isfinite(live_arr)) or np.any(live_arr <= 0.0):
            raise ValueError("Full-spectrum live times must be positive.")
        if (
            line_indices.size == 0
            or np.unique(line_indices).size != line_indices.size
            or np.any(line_indices < 0)
        ):
            raise ValueError(
                "positive_line_indices must be nonempty, unique, and nonnegative."
            )
        if self.kernel is None:
            raise RuntimeError("Continuous line transport requires PF poses.")
        self.validate_continuous_surface_states()
        device = (
            gpu_utils.resolve_device(self.config.gpu_device)
            if self.config.use_gpu
            else torch.device("cpu")
        )
        (
            positions,
            strengths,
            mask,
            chart_ids,
            _surface_uv,
        ) = self._packed_continuous_surface_state_arrays()
        particle_count = int(positions.shape[0])
        slot_count = int(mask.shape[1])
        view_count = int(fe_arr.size)
        line_count = int(line_indices.size)
        output_shape = (
            particle_count,
            view_count,
            slot_count,
            line_count,
        )
        arrays = [np.zeros(output_shape, dtype=np.float64) for _ in range(6)]
        if np.any(mask):
            active_positions = self._surface_transport_positions(
                positions[mask],
                chart_ids=chart_ids[mask],
            )
            active_strengths = strengths[mask]
            particle_ids, source_slots = np.nonzero(mask)
            unique_positions, inverse = np.unique(
                active_positions,
                axis=0,
                return_inverse=True,
            )
            detector_position = np.asarray(
                self.kernel.poses[int(pose_idx)],
                dtype=np.float64,
            )
            components = (
                self.continuous_kernel
                .line_transport_components_pair_program_for_detectors(
                    isotope=self.isotope,
                    detector_positions=detector_position.reshape(1, 3),
                    sources=unique_positions,
                    fe_indices=fe_arr.reshape(1, view_count),
                    pb_indices=pb_arr.reshape(1, view_count),
                    positive_line_indices=line_indices,
                )
            )
            component_total = np.asarray(
                components.total_kernel,
                dtype=np.float64,
            )[0]
            component_uncollided = np.asarray(
                components.uncollided_kernel,
                dtype=np.float64,
            )[0]
            total_active = np.transpose(
                component_total[:, inverse, :],
                (1, 0, 2),
            )
            uncollided_active = np.transpose(
                component_uncollided[:, inverse, :],
                (1, 0, 2),
            )
            rate_scale = active_strengths[:, None, None]
            arrays[0][particle_ids, :, source_slots, :] = total_active * rate_scale
            arrays[1][particle_ids, :, source_slots, :] = uncollided_active * rate_scale
            for output, values in zip(
                arrays[2:],
                (
                    components.tau_fe,
                    components.tau_pb,
                    components.tau_obstacle,
                    components.distance_m,
                ),
            ):
                component_values = np.asarray(
                    values,
                    dtype=np.float64,
                )[0]
                output[particle_ids, :, source_slots, :] = np.transpose(
                    component_values[:, inverse, :],
                    (1, 0, 2),
                )
        tensors = [
            torch.as_tensor(value, dtype=torch.float64, device=device)
            for value in arrays
        ]
        return TorchLineTransportComponents(*tensors)

    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _normalized_log_weights_torch(self, logw: "torch.Tensor") -> "torch.Tensor":
        """Normalize valid log-weights or fail before posterior corruption."""
        import torch

        if logw.ndim != 1 or int(logw.numel()) <= 0:
            raise ValueError("Particle log weights must be a nonempty vector.")
        if bool(torch.any(torch.isnan(logw)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(logw) & (logw > 0.0)).detach().cpu().item()
        ):
            raise RuntimeError("Particle log weights contain NaN or positive infinity.")
        if not bool(torch.any(torch.isfinite(logw)).detach().cpu().item()):
            raise RuntimeError(
                "All particle log weights are negative infinity; posterior "
                "normalization is undefined."
            )
        finite_max = torch.max(logw)
        if not bool(torch.isfinite(finite_max).detach().cpu().item()):
            raise RuntimeError("Particle log-weight normalizer is non-finite.")
        shifted = logw - finite_max
        shifted_normalizer = torch.logsumexp(shifted, dim=0)
        if not bool(torch.isfinite(shifted_normalizer).detach().cpu().item()):
            raise RuntimeError("Particle log-weight normalizer is non-finite.")
        normalized = shifted - shifted_normalizer
        if bool(torch.any(torch.isnan(normalized)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(normalized) & (normalized > 0.0))
            .detach()
            .cpu()
            .item()
        ):
            raise RuntimeError("Normalized particle log weights are invalid.")
        return normalized

    def _ess_from_logw_torch(self, logw: "torch.Tensor") -> float:
        """Return the effective sample size from normalized log-weights."""
        import torch

        if logw.ndim != 1 or int(logw.numel()) <= 0:
            raise ValueError("ESS requires a nonempty normalized log-weight vector.")
        if (
            bool(torch.any(torch.isnan(logw)).detach().cpu().item())
            or bool(torch.any(torch.isinf(logw) & (logw > 0.0)).detach().cpu().item())
            or not bool(torch.any(torch.isfinite(logw)).detach().cpu().item())
        ):
            raise RuntimeError("ESS received invalid particle log weights.")
        log_normalizer = float(torch.logsumexp(logw, dim=0).detach().cpu().item())
        if not np.isfinite(log_normalizer) or not np.isclose(
            log_normalizer,
            0.0,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("ESS requires already normalized log weights.")
        w = torch.exp(logw)
        denominator = float(torch.sum(w**2).detach().cpu().item())
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("ESS denominator must be finite and positive.")
        ess = 1.0 / denominator
        if (
            not np.isfinite(ess)
            or not 1.0 - 1.0e-9 <= ess <= int(logw.numel()) + 1.0e-9
        ):
            raise RuntimeError("Effective sample size lies outside its support.")
        return float(ess)

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
        import torch

        remaining = float(remaining)
        target_ess = float(target_ess)
        if (
            logw_prev.ndim != 1
            or ll_t.ndim != 1
            or tuple(logw_prev.shape) != tuple(ll_t.shape)
            or int(logw_prev.numel()) <= 0
        ):
            raise ValueError(
                "Tempering requires aligned nonempty log-weight and likelihood vectors."
            )
        if not np.isfinite(remaining) or not 0.0 < remaining <= 1.0:
            raise ValueError("Tempering remaining beta must lie in (0, 1].")
        if (
            not np.isfinite(target_ess)
            or target_ess <= 0.0
            or target_ess > int(logw_prev.numel()) + 1.0e-9
        ):
            raise ValueError("Tempering target ESS lies outside particle support.")
        self._ess_from_logw_torch(logw_prev)
        if bool(torch.any(torch.isnan(ll_t)).detach().cpu().item()) or bool(
            torch.any(torch.isinf(ll_t) & (ll_t > 0.0)).detach().cpu().item()
        ):
            raise RuntimeError(
                "Tempering likelihood contains NaN or positive infinity."
            )
        if not bool(torch.any(torch.isfinite(ll_t)).detach().cpu().item()):
            raise RuntimeError(
                "All particle likelihoods are negative infinity; the "
                "observation is impossible under the PF model."
            )
        min_delta = float(self.config.min_delta_beta)
        if remaining <= min_delta:
            logw_new = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
            ess = self._ess_from_logw_torch(logw_new)
            if ess + 1.0e-9 < target_ess:
                raise TemperingIncrementRequiresRejuvenation(
                    "The final configured tempering increment would violate "
                    "the target ESS and requires rejuvenation at the current "
                    "intermediate target."
                )
            return remaining, logw_new, ess

        logw_full = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
        ess_full = self._ess_from_logw_torch(logw_full)
        if ess_full >= target_ess:
            return remaining, logw_full, ess_full

        logw_low = self._normalized_log_weights_torch(logw_prev + min_delta * ll_t)
        ess_low = self._ess_from_logw_torch(logw_low)
        if ess_low < target_ess:
            raise TemperingIncrementRequiresRejuvenation(
                "No configured positive tempering increment preserves target "
                "ESS before rejuvenation at the current intermediate target."
            )

        low = min_delta
        high = remaining
        logw_best = logw_low
        ess_best = ess_low
        for _ in range(48):
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

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles], dtype=np.float64
        )
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        if np.any(np.isnan(logw)) or np.any(np.isposinf(logw)):
            raise RuntimeError("Particle log weights contain NaN or positive infinity.")
        finite = np.isfinite(logw)
        if not np.any(finite):
            raise RuntimeError(
                "All particle log weights are negative infinity; posterior is invalid."
            )
        normalized = np.zeros(logw.size, dtype=np.float64)
        shifted = logw[finite] - float(np.max(logw[finite]))
        normalized[finite] = np.exp(shifted)
        total = float(np.sum(normalized))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "Particle weights do not have a finite positive normalization."
            )
        return normalized / total


__all__ = ["ParticleTemperingMixin", "TemperingIncrementRequiresRejuvenation"]
