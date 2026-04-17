"""Likelihood helpers and per-source expected-count utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel


def poisson_log_likelihood(z_k: NDArray[np.float64], lambda_k: NDArray[np.float64], epsilon: float = 1e-12) -> float:
    """
    Return the Poisson log-likelihood sum_k [z_k * log(lambda_k) - lambda_k] (constants omitted).
    """
    lambda_safe = np.maximum(lambda_k, float(epsilon))
    return float(np.sum(z_k * np.log(lambda_safe) - lambda_safe))


def delta_log_likelihood_remove(
    z_k: NDArray[np.float64],
    lambda_total: NDArray[np.float64],
    lambda_m: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Compute ΔLL when removing each source m using a stable log1p formulation.

    lambda_m must be shaped (K, M). Returns a vector (M,).
    """
    if lambda_m.ndim != 2:
        raise ValueError("lambda_m must be a (K, M) array.")
    lambda_total_safe = np.maximum(lambda_total, float(epsilon))
    ratio = lambda_m / lambda_total_safe[:, None]
    ratio = np.clip(ratio, 0.0, 1.0 - float(epsilon))
    log_term = -np.log1p(-ratio)
    delta_ll = np.sum(z_k[:, None] * log_term - lambda_m, axis=0)
    return delta_ll


def delta_log_likelihood_update(
    z_k: NDArray[np.float64],
    lambda_old: NDArray[np.float64],
    lambda_new: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> float:
    """
    Compute ΔLL for replacing lambda_old with lambda_new across measurements.
    """
    lambda_old_safe = np.maximum(lambda_old, float(epsilon))
    lambda_new_safe = np.maximum(lambda_new, float(epsilon))
    return float(np.sum(z_k * (np.log(lambda_new_safe) - np.log(lambda_old_safe)) - (lambda_new - lambda_old)))


def expected_counts_per_source(
    kernel: ContinuousKernel,
    isotope: str,
    detector_positions: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    live_times: NDArray[np.float64],
    fe_indices: NDArray[np.int64] | None = None,
    pb_indices: NDArray[np.int64] | None = None,
    orient_indices: NDArray[np.int64] | None = None,
    source_scale: float = 1.0,
) -> NDArray[np.float64]:
    """
    Return per-source expected counts Λ_{k,m} for each measurement k.

    Supports either paired Fe/Pb indices or single orientation indices.
    source_scale maps ideal source counts into the measurement domain while
    leaving the additive background term to the caller.
    """
    if sources.size == 0:
        return np.zeros((len(live_times), 0), dtype=float)
    if sources.shape[0] != strengths.shape[0]:
        raise ValueError("sources and strengths must have matching length.")
    num_meas = int(len(live_times))
    num_sources = int(sources.shape[0])
    lambda_m = np.zeros((num_meas, num_sources), dtype=float)
    for k in range(num_meas):
        det = detector_positions[k]
        live_time = float(live_times[k])
        for m in range(num_sources):
            if fe_indices is not None and pb_indices is not None:
                kernel_val = kernel.kernel_value_pair(
                    isotope=isotope,
                    detector_pos=det,
                    source_pos=sources[m],
                    fe_index=int(fe_indices[k]),
                    pb_index=int(pb_indices[k]),
                )
            elif orient_indices is not None:
                kernel_val = kernel.kernel_value(
                    isotope=isotope,
                    detector_pos=det,
                    source_pos=sources[m],
                    orient_idx=int(orient_indices[k]),
                )
            else:
                raise ValueError("Either fe_indices/pb_indices or orient_indices must be provided.")
            lambda_m[k, m] = live_time * max(float(source_scale), 0.0) * kernel_val * float(strengths[m])
    return lambda_m
