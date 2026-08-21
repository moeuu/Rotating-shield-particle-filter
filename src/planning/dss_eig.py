"""Batched exact-EIG kernels and memory scheduling for DSS-PP."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from pf.estimator import RotatingShieldPFEstimator
from pf.full_spectrum import validate_full_spectrum_model
from planning.dss_modes import _normalise_weights
from planning.dss_types import _JointProgramSpectrumComponents
from planning.shield_programs import ShieldProgram


def _program_pair_id_matrix(
    programs: Sequence[ShieldProgram],
) -> NDArray[np.int64]:
    """Return a padded pair-id matrix for a set of shield programs."""
    if not programs:
        return np.zeros((0, 0), dtype=np.int64)
    pair_rows = tuple(
        np.asarray(program.pair_ids, dtype=np.int64) for program in programs
    )
    lengths = np.fromiter(
        (row.size for row in pair_rows),
        dtype=np.int64,
        count=len(pair_rows),
    )
    max_length = int(np.max(lengths, initial=0))
    if max_length <= 0:
        return np.zeros((len(programs), 0), dtype=np.int64)
    matrix = np.zeros((len(programs), max_length), dtype=np.int64)
    total_values = int(np.sum(lengths))
    flat_values = np.concatenate(pair_rows)
    row_indices = np.repeat(
        np.arange(len(programs), dtype=np.int64),
        lengths,
    )
    starts = np.cumsum(
        np.concatenate((np.zeros(1, dtype=np.int64), lengths[:-1])),
        dtype=np.int64,
    )
    row_starts = np.repeat(starts, lengths)
    column_indices = np.arange(total_values, dtype=np.int64) - row_starts
    matrix[row_indices, column_indices] = flat_values
    return matrix


def _program_view_mask(
    programs: Sequence[ShieldProgram],
    *,
    max_length: int,
) -> NDArray[np.bool_]:
    """Return a mask selecting the physical views in padded programs."""
    if max_length <= 0:
        return np.zeros((len(programs), 0), dtype=bool)
    lengths = np.asarray([len(program.pair_ids) for program in programs], dtype=int)
    return np.arange(max_length, dtype=int)[None, :] < lengths[:, None]


def _finite_sphere_geometric_terms_batched(
    detector_positions: NDArray[np.float64],
    source_positions: NDArray[np.float64],
    *,
    detector_radius_m: float,
) -> NDArray[np.float64]:
    """Return finite-sphere detector geometry for batched positions."""
    detectors = np.asarray(detector_positions, dtype=float)
    sources = np.asarray(source_positions, dtype=float)
    if detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detector_positions must be shaped (D, 3).")
    if sources.ndim < 2 or sources.shape[-1] != 3:
        raise ValueError("source_positions must end in a three-vector dimension.")
    source_shape = sources.shape[:-1]
    delta = detectors.reshape((detectors.shape[0],) + (1,) * len(source_shape) + (3,))
    delta = delta - sources.reshape((1,) + source_shape + (3,))
    distance = np.linalg.norm(delta, axis=-1)
    radius = float(detector_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("detector_radius_m must be finite and nonnegative.")
    if radius <= 0.0:
        scale = np.zeros_like(distance, dtype=float)
        positive = np.square(distance) > 1.0e-12
        scale[positive] = 1.0 / np.square(distance[positive])
        return scale
    effective_distance = np.maximum(distance, radius)
    ratio = np.clip(
        radius / np.maximum(effective_distance, 1.0e-12),
        0.0,
        1.0,
    )
    fraction = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - np.square(ratio), 0.0)))
    reference_distance = max(1.0, radius)
    reference_ratio = min(radius / reference_distance, 1.0)
    reference_fraction = max(
        0.5 * (1.0 - float(np.sqrt(max(1.0 - reference_ratio * reference_ratio, 0.0)))),
        1.0e-12,
    )
    scale = fraction / reference_fraction
    return np.where(distance > 1.0e-12, scale, 0.0)


def _information_gain_from_log_likelihood(
    log_likelihood_psn: NDArray[np.float64],
    weights_n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return program mutual information from sampled particle likelihoods."""
    log_likelihood = np.asarray(log_likelihood_psn, dtype=float)
    weights = _normalise_weights(np.asarray(weights_n, dtype=float))
    if log_likelihood.ndim != 3:
        raise ValueError(
            "log_likelihood_psn must be shaped (program, sample, particle)."
        )
    if weights.shape != (log_likelihood.shape[2],):
        raise ValueError("weights_n must match the particle dimension.")
    if np.any(np.isnan(log_likelihood)) or np.any(np.isposinf(log_likelihood)):
        raise ValueError("Program likelihoods may be finite or minus infinity only.")
    positive_prior = weights > 0.0
    if not np.any(positive_prior):
        raise ValueError("Program EIG requires positive posterior mass.")
    active_likelihood = log_likelihood[:, :, positive_prior]
    active_weights = weights[positive_prior]
    log_prior = np.log(active_weights)[None, None, :]
    log_joint = active_likelihood + log_prior
    log_evidence = logsumexp(log_joint, axis=2, keepdims=True)
    if np.any(~np.isfinite(log_evidence)):
        raise RuntimeError(
            "A predictive DSS observation is outside every positive-mass PF state."
        )
    posterior = np.exp(log_joint - log_evidence)
    kl_terms = np.zeros_like(posterior)
    np.multiply(
        posterior,
        active_likelihood - log_evidence,
        out=kl_terms,
        where=posterior > 0.0,
    )
    kl_samples = np.sum(kl_terms, axis=2)
    information_gain = np.mean(kl_samples, axis=1)
    if np.any(~np.isfinite(information_gain)):
        raise ValueError("Program mutual information must be finite.")
    numerical_tolerance = 1.0e-10
    if np.any(information_gain < -numerical_tolerance):
        raise RuntimeError(
            "Program mutual information became materially negative; the "
            "joint likelihood or posterior weights are inconsistent."
        )
    return np.maximum(information_gain, 0.0)


def _finite_sample_information_gain_upper_bound(
    weights_n: NDArray[np.float64],
) -> float:
    """Bound every sampled posterior KL by the smallest positive prior mass.

    The entropy of the prior bounds the *expected* mutual information, but it
    does not bound a finite Monte Carlo average: an unusually diagnostic draw
    from a rare particle can have KL larger than the prior entropy. For any
    posterior supported on the positive-prior particles,
    ``KL(q || p) <= -log(min(p))``. This looser bound is therefore safe for the
    actual finite-sample EIG objective used by adaptive action expansion.
    """
    weights = _normalise_weights(np.asarray(weights_n, dtype=np.float64))
    positive = weights[weights > 0.0]
    if positive.size == 0:
        raise ValueError("Program EIG requires positive posterior mass.")
    return float(-np.log(np.min(positive)))


def _joint_program_action_layout(
    programs_by_pose: Sequence[Sequence[ShieldProgram]],
) -> tuple[
    list[ShieldProgram],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.int64],
]:
    """Return one flattened action table for all candidate poses."""
    counts = np.asarray(
        [len(programs) for programs in programs_by_pose],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    flattened = [program for programs in programs_by_pose for program in programs]
    pose_indices = np.repeat(
        np.arange(len(programs_by_pose), dtype=np.int64),
        counts,
    )
    pair_ids = _program_pair_id_matrix(flattened)
    view_mask = _program_view_mask(
        flattened,
        max_length=int(pair_ids.shape[1]) if pair_ids.ndim == 2 else 0,
    )
    return flattened, pose_indices, pair_ids, view_mask, offsets


def _full_spectrum_information_gain(
    estimator: RotatingShieldPFEstimator,
    components: _JointProgramSpectrumComponents,
    particle_weights: NDArray[np.float64],
    *,
    sample_count: int,
    rng: np.random.Generator,
    use_gpu: bool,
    gpu_device: str,
    latent_particle_indices: NDArray[np.int64] | None = None,
    action_seeds_a: NDArray[np.int64] | None = None,
    action_chunk_size: int | None = None,
    state_chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Estimate full-spectrum mutual information with bounded action scheduling.

    Transport and cross-likelihood tensors are batched. The generative model
    schedules one canonically seeded predictive draw stream per action. Caller
    batching changes only the execution schedule, not any action's physics,
    likelihood, posterior sample, or random stream.
    """
    model = validate_full_spectrum_model(estimator.full_spectrum_generative_model)
    if str(model.contract_hash_sha256) != str(components.contract_hash_sha256):
        raise RuntimeError("DSS spectrum components use a different model hash.")
    total = np.asarray(components.total_pnvsl, dtype=np.float64)
    uncollided = np.asarray(components.uncollided_pnvsl, dtype=np.float64)
    features = np.asarray(components.features_pnvslf, dtype=np.float64)
    live_times = np.asarray(components.live_times_v, dtype=np.float64)
    if (
        total.ndim != 5
        or uncollided.shape != total.shape
        or features.shape != total.shape + (4,)
        or live_times.shape != (total.shape[2],)
    ):
        raise ValueError("DSS full-spectrum component shapes are inconsistent.")
    action_count, particle_count = total.shape[:2]
    action_seeds = None
    if action_seeds_a is not None:
        action_seeds = np.asarray(action_seeds_a)
        if (
            action_seeds.ndim != 1
            or action_seeds.shape != (action_count,)
            or not np.issubdtype(action_seeds.dtype, np.integer)
        ):
            raise ValueError(
                "action_seeds_a must contain one integer seed per DSS action."
            )
    weights = _normalise_weights(np.asarray(particle_weights, dtype=np.float64))
    if weights.shape != (particle_count,):
        raise ValueError("DSS particle weights do not match spectrum states.")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or int(sample_count) <= 0
    ):
        raise ValueError("sample_count must be a positive integer.")
    resolved_sample_count = int(sample_count)
    if latent_particle_indices is None:
        latent_indices = rng.choice(
            particle_count,
            size=resolved_sample_count,
            replace=True,
            p=weights,
        )
    else:
        latent_indices = np.asarray(
            latent_particle_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            latent_indices.shape != (resolved_sample_count,)
            or np.any(latent_indices < 0)
            or np.any(latent_indices >= particle_count)
        ):
            raise ValueError(
                "DSS latent_particle_indices must contain one valid common "
                "posterior-particle index per predictive sample."
            )
    truth_total = total[:, latent_indices]
    truth_uncollided = uncollided[:, latent_indices]
    truth_features = features[:, latent_indices]
    predictive = np.asarray(
        model.sample_predictive_numpy(
            truth_total,
            truth_uncollided,
            truth_features,
            live_times,
            sample_count=1,
            rng=rng,
            action_seeds_a=action_seeds,
        ),
        dtype=np.float64,
    )
    expected_predictive_shape = (
        action_count,
        resolved_sample_count,
        1,
        int(total.shape[2]),
        int(np.asarray(model.energy_axis_keV).size),
    )
    if predictive.shape != expected_predictive_shape:
        raise RuntimeError(
            "Full-spectrum predictive sampler returned an invalid DSS shape."
        )
    observations = np.ascontiguousarray(predictive[:, :, 0])
    if bool(use_gpu):
        import torch

        cross_likelihood = getattr(model, "cross_log_likelihood_torch", None)
        if not callable(cross_likelihood):
            raise RuntimeError(
                "GPU DSS requires vectorized full-spectrum Torch cross likelihood."
            )
        device = torch.device(str(gpu_device))
        log_likelihood = np.asarray(
            cross_likelihood(
                torch.as_tensor(
                    observations,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(total, dtype=torch.float64, device=device),
                torch.as_tensor(uncollided, dtype=torch.float64, device=device),
                torch.as_tensor(features, dtype=torch.float64, device=device),
                torch.as_tensor(live_times, dtype=torch.float64, device=device),
                action_chunk_size=action_chunk_size,
                state_chunk_size=state_chunk_size,
            )
            .detach()
            .cpu()
            .numpy(),
            dtype=np.float64,
        )
    else:
        cross_likelihood = getattr(model, "cross_log_likelihood_numpy", None)
        if not callable(cross_likelihood):
            raise RuntimeError(
                "DSS requires vectorized full-spectrum cross likelihoods."
            )
        log_likelihood = np.asarray(
            cross_likelihood(
                observations,
                total,
                uncollided,
                features,
                live_times,
                action_chunk_size=action_chunk_size,
                state_chunk_size=state_chunk_size,
            ),
            dtype=np.float64,
        )
    expected_log_shape = (
        action_count,
        resolved_sample_count,
        particle_count,
    )
    if log_likelihood.shape != expected_log_shape:
        raise RuntimeError(
            "Full-spectrum cross likelihood returned an invalid DSS shape."
        )
    return _information_gain_from_log_likelihood(log_likelihood, weights)


def _dss_eig_state_chunk_size(
    model: object,
    *,
    action_count: int,
    particle_count: int,
    sample_count: int,
    source_slot_count: int,
    view_count: int,
    memory_budget_bytes: int,
) -> int:
    """Return the largest power-of-two state chunk within half the budget."""
    estimator = getattr(
        model,
        "estimate_cross_likelihood_working_set_bytes",
        None,
    )
    if not callable(estimator):
        raise RuntimeError(
            "The full-spectrum model must publish an exact likelihood "
            "working-set estimate for DSS batching."
        )
    upper = 1 << (int(particle_count).bit_length() - 1)
    target_workspace_bytes = max(1, int(memory_budget_bytes) // 2)
    state_chunk_size = int(upper)
    while state_chunk_size > 1:
        working_set_bytes = int(
            estimator(
                num_actions=int(action_count),
                num_samples=int(sample_count),
                num_particles=int(particle_count),
                num_isotopes=int(source_slot_count),
                num_views=int(view_count),
                action_chunk_size=1,
                state_chunk_size=int(state_chunk_size),
                dtype_bytes=np.dtype(np.float64).itemsize,
            )
        )
        if working_set_bytes <= target_workspace_bytes:
            return int(state_chunk_size)
        state_chunk_size //= 2
    return 1


def _dss_eig_likelihood_action_chunk_size(
    model: object,
    *,
    action_count: int,
    particle_count: int,
    sample_count: int,
    source_slot_count: int,
    view_count: int,
    state_chunk_size: int,
    memory_budget_bytes: int,
) -> int:
    """Return a batched likelihood action chunk within half the budget."""
    estimator = getattr(
        model,
        "estimate_cross_likelihood_working_set_bytes",
        None,
    )
    if not callable(estimator):
        raise RuntimeError(
            "The full-spectrum model must publish an exact likelihood "
            "working-set estimate for DSS batching."
        )
    if int(action_count) <= 0:
        raise ValueError("DSS likelihood action_count must be positive.")
    upper = 1 << (int(action_count).bit_length() - 1)
    target_workspace_bytes = max(1, int(memory_budget_bytes) // 2)
    action_chunk_size = int(upper)
    while action_chunk_size > 1:
        working_set_bytes = int(
            estimator(
                num_actions=int(action_count),
                num_samples=int(sample_count),
                num_particles=int(particle_count),
                num_isotopes=int(source_slot_count),
                num_views=int(view_count),
                action_chunk_size=int(action_chunk_size),
                state_chunk_size=int(state_chunk_size),
                dtype_bytes=np.dtype(np.float64).itemsize,
            )
        )
        if working_set_bytes <= target_workspace_bytes:
            return int(action_chunk_size)
        action_chunk_size //= 2
    return 1


def _dss_eig_action_batch_size(
    model: object,
    *,
    action_count: int,
    particle_count: int,
    sample_count: int,
    source_slot_count: int,
    view_count: int,
    line_count: int,
    feature_count: int,
    memory_budget_bytes: int,
    state_chunk_size: int | None = None,
    diagnostics: dict[str, int] | None = None,
) -> int:
    """Return a conservative action batch using the model workspace contract."""
    counts = {
        "action_count": action_count,
        "particle_count": particle_count,
        "sample_count": sample_count,
        "source_slot_count": source_slot_count,
        "view_count": view_count,
        "line_count": line_count,
        "feature_count": feature_count,
        "memory_budget_bytes": memory_budget_bytes,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in counts.values()
    ):
        raise ValueError("DSS EIG batch dimensions and memory budget must be positive.")
    estimator = getattr(
        model,
        "estimate_cross_likelihood_working_set_bytes",
        None,
    )
    if not callable(estimator):
        raise RuntimeError(
            "The full-spectrum model must publish an exact likelihood "
            "working-set estimate for DSS batching."
        )
    model_working_set = int(
        estimator(
            num_actions=int(action_count),
            num_samples=int(sample_count),
            num_particles=int(particle_count),
            num_isotopes=int(source_slot_count),
            num_views=int(view_count),
            state_chunk_size=state_chunk_size,
            dtype_bytes=np.dtype(np.float64).itemsize,
        )
    )
    if model_working_set <= 0:
        raise RuntimeError(
            "The full-spectrum likelihood returned an invalid working-set estimate."
        )
    energy_axis = np.asarray(getattr(model, "energy_axis_keV", ()))
    if energy_axis.ndim != 1 or energy_axis.size <= 0:
        raise RuntimeError("The full-spectrum model has no valid energy axis.")
    float_bytes = np.dtype(np.float64).itemsize
    transport_per_action = (
        int(particle_count)
        * int(view_count)
        * int(source_slot_count)
        * int(line_count)
        * (2 + int(feature_count))
        * float_bytes
    )
    predictive_per_action = (
        int(sample_count) * int(view_count) * int(energy_axis.size) * float_bytes
    )
    likelihood_output_per_action = int(sample_count) * int(particle_count) * float_bytes
    # Account for NumPy storage, Torch device copies, and allocator overlap.
    persistent_per_action = 3 * (
        transport_per_action + predictive_per_action + likelihood_output_per_action
    )
    available_for_actions = int(memory_budget_bytes) - model_working_set
    if available_for_actions < persistent_per_action:
        raise MemoryError(
            "DSS EIG memory budget cannot hold the model workspace and one "
            "action without violating the declared limit."
        )
    selected_batch_size = min(
        int(action_count),
        int(available_for_actions // persistent_per_action),
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "requested_action_count": int(action_count),
                "particle_count": int(particle_count),
                "sample_count": int(sample_count),
                "source_slot_count": int(source_slot_count),
                "view_count": int(view_count),
                "line_count": int(line_count),
                "feature_count": int(feature_count),
                "energy_bin_count": int(energy_axis.size),
                "memory_budget_bytes": int(memory_budget_bytes),
                "state_chunk_size": (
                    int(particle_count)
                    if state_chunk_size is None
                    else int(state_chunk_size)
                ),
                "model_working_set_bytes": int(model_working_set),
                "transport_per_action_bytes": int(transport_per_action),
                "predictive_per_action_bytes": int(predictive_per_action),
                "likelihood_output_per_action_bytes": int(likelihood_output_per_action),
                "persistent_per_action_bytes": int(persistent_per_action),
                "available_for_actions_bytes": int(available_for_actions),
                "initial_action_batch_size": int(selected_batch_size),
            }
        )
    return int(selected_batch_size)


def _is_dss_eig_memory_error(error: BaseException) -> bool:
    """Return whether an exception represents host or accelerator exhaustion."""
    if isinstance(error, MemoryError):
        return True
    error_type = type(error)
    if error_type.__name__ == "OutOfMemoryError" and error_type.__module__.startswith(
        "torch"
    ):
        return True
    return "out of memory" in str(error).lower()


def _release_dss_gpu_cache() -> None:
    """Release unused Torch cache blocks after a recoverable DSS OOM."""
    try:
        import torch
    except ImportError:
        return
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def _dss_accelerator_memory_snapshot(
    *,
    use_gpu: bool,
    gpu_device: str,
) -> dict[str, object]:
    """Return read-only accelerator memory diagnostics for exact DSS EIG."""
    if not bool(use_gpu):
        return {
            "enabled": False,
            "device": "cpu",
        }
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU DSS EIG was requested but Torch is unavailable."
        ) from error
    device = torch.device(str(gpu_device))
    if device.type != "cuda":
        return {
            "enabled": True,
            "device": str(device),
            "cuda": False,
        }
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("GPU DSS EIG was requested but CUDA is unavailable.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "enabled": True,
        "device": str(device),
        "cuda": True,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
