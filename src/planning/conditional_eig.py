"""Prepare shared-runtime arbitrary-subset likelihood caches for DSS search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from pf.full_spectrum import (
    PreparedSubsetCrossLikelihood,
    SubsetCrossLikelihoodFullSpectrumModel,
    TorchPredictiveFullSpectrumModel,
)
from pf.randomness import named_random_generator, named_stream_seed
from planning.dss_modes import _normalise_weights
from planning.dss_types import (
    _DeviceJointProgramSpectrumComponents,
    _JointProgramSpectrumComponents,
)


@dataclass(frozen=True, slots=True)
class PreparedConditionalObservation:
    """Store one opaque likelihood cache and reproducibility metadata."""

    cache: PreparedSubsetCrossLikelihood
    latent_particle_indices_q: NDArray[np.int64]
    action_seeds_a: NDArray[np.int64]
    sample_count: int
    action_count: int
    pair_count: int


def prepare_conditional_observation_cache(
    estimator: RotatingShieldPFEstimator,
    components: (
        _JointProgramSpectrumComponents
        | _DeviceJointProgramSpectrumComponents
    ),
    particle_weights_n: NDArray[np.float64],
    detector_positions_a3: NDArray[np.float64],
    *,
    sample_count: int,
    eig_call_seed: int,
    stream_name: str,
) -> PreparedConditionalObservation:
    """Generate all-pair virtual observations and prepare an exact GPU cache.

    The component view axis must contain every physical Fe/Pb pair exactly
    once in pair-ID order. One latent PF state and every station-shared
    nuisance draw are shared across all 64 virtual views of a pose. Only a
    later selected subset enters the opaque runtime likelihood.
    """
    import torch

    model = estimator.authenticated_full_spectrum_model()
    if not isinstance(model, TorchPredictiveFullSpectrumModel):
        raise RuntimeError(
            "Conditional DSS requires the runtime Torch predictive sampler."
        )
    if not isinstance(model, SubsetCrossLikelihoodFullSpectrumModel):
        raise RuntimeError(
            "Conditional DSS requires the runtime arbitrary-subset likelihood "
            "cache API."
        )
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or int(sample_count) <= 0
    ):
        raise ValueError("sample_count must be a positive integer.")
    if (
        isinstance(eig_call_seed, bool)
        or not isinstance(eig_call_seed, (int, np.integer))
        or int(eig_call_seed) < 0
    ):
        raise ValueError("eig_call_seed must be a nonnegative integer.")
    if not isinstance(stream_name, str) or not stream_name:
        raise ValueError("stream_name must be a nonempty string.")

    detectors = np.asarray(detector_positions_a3, dtype=np.float64)
    if (
        detectors.ndim != 2
        or detectors.shape[1] != 3
        or np.any(~np.isfinite(detectors))
    ):
        raise ValueError("detector_positions_a3 must be finite and shaped (A, 3).")
    device_components = isinstance(
        components,
        _DeviceJointProgramSpectrumComponents,
    )
    if device_components:
        total = components.total_pnvsl
        uncollided = components.uncollided_pnvsl
        features = components.features_pnvslf
        live_times = components.live_times_v
        device = total.device
    else:
        device = torch.device(
            str(estimator.pf_config.gpu_device)
            if bool(estimator.pf_config.use_gpu)
            else "cpu"
        )
        total = torch.as_tensor(
            components.total_pnvsl,
            device=device,
            dtype=torch.float64,
        )
        uncollided = torch.as_tensor(
            components.uncollided_pnvsl,
            device=device,
            dtype=torch.float64,
        )
        features = torch.as_tensor(
            components.features_pnvslf,
            device=device,
            dtype=torch.float64,
        )
        live_times = torch.as_tensor(
            components.live_times_v,
            device=device,
            dtype=torch.float64,
        )
    if (
        total.ndim != 5
        or tuple(uncollided.shape) != tuple(total.shape)
        or tuple(features.shape)
        != tuple(total.shape) + (len(tuple(model.transport_feature_order)),)
        or tuple(live_times.shape) != (int(total.shape[2]),)
        or int(total.shape[0]) != int(detectors.shape[0])
    ):
        raise ValueError("All-pair conditional component shapes are inconsistent.")
    action_count = int(total.shape[0])
    particle_count = int(total.shape[1])
    pair_count = int(estimator.num_orientations) ** 2
    if int(total.shape[2]) != pair_count:
        raise ValueError(
            "Conditional components must contain every shield pair in pair-ID "
            "order."
        )
    weights = _normalise_weights(
        np.asarray(particle_weights_n, dtype=np.float64)
    )
    if weights.shape != (particle_count,):
        raise ValueError("particle_weights_n must align with conditional states.")

    snapshot_index = len(estimator.measurements)
    latent_rng = named_random_generator(
        int(eig_call_seed),
        "dss_pp",
        "conditional_all_pairs",
        int(snapshot_index),
        str(stream_name),
        "common_latent_particles",
    )
    latent_indices = latent_rng.choice(
        particle_count,
        size=int(sample_count),
        replace=True,
        p=weights,
    ).astype(np.int64, copy=False)
    action_seeds = np.asarray(
        [
            named_stream_seed(
                int(eig_call_seed),
                "dss_pp",
                "conditional_all_pairs",
                int(snapshot_index),
                str(stream_name),
                "canonical_pose",
                *(float(value).hex() for value in detector),
            )
            & ((1 << 63) - 1)
            for detector in detectors
        ],
        dtype=np.int64,
    )
    latent_index = torch.as_tensor(
        latent_indices,
        device=device,
        dtype=torch.long,
    )
    predictive = model.sample_predictive_torch(
        total[:, latent_index],
        uncollided[:, latent_index],
        features[:, latent_index],
        live_times,
        sample_count=1,
        action_seeds_a=action_seeds,
    )
    expected_predictive_shape = (
        action_count,
        int(sample_count),
        1,
        pair_count,
        int(np.asarray(model.energy_axis_keV).size),
    )
    if (
        not torch.is_tensor(predictive)
        or tuple(predictive.shape) != expected_predictive_shape
        or predictive.device != device
        or predictive.dtype != torch.int64
        or bool(torch.any(predictive < 0).item())
    ):
        raise RuntimeError(
            "Runtime predictive sampling returned invalid all-pair observations."
        )
    observations = predictive[:, :, 0].to(dtype=torch.float64).contiguous()
    cache = model.prepare_subset_cross_likelihood_torch(
        observations,
        total,
        uncollided,
        features,
        live_times,
        action_chunk_size=(
            1 if particle_count >= 256 else min(action_count, 32)
        ),
        sample_chunk_size=min(int(sample_count), 10),
        state_chunk_size=min(particle_count, 128),
        view_chunk_size=min(pair_count, 8),
    )
    if (
        int(cache.action_count) != action_count
        or int(cache.sample_count) != int(sample_count)
        or int(cache.state_count) != particle_count
        or int(cache.view_count) != pair_count
        or getattr(cache, "device", device) != device
        or getattr(cache, "dtype", torch.float64) != torch.float64
    ):
        raise RuntimeError("Runtime prepared an inconsistent subset cache.")
    return PreparedConditionalObservation(
        cache=cache,
        latent_particle_indices_q=latent_indices,
        action_seeds_a=action_seeds,
        sample_count=int(sample_count),
        action_count=action_count,
        pair_count=pair_count,
    )


__all__ = [
    "PreparedConditionalObservation",
    "prepare_conditional_observation_cache",
]
