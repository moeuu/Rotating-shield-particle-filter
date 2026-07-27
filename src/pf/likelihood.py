"""Likelihood helpers and per-source expected-count utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel

if TYPE_CHECKING:
    import torch


DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL = "student_t"
DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA = 0.10
DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA = 5.0
DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA = 0.05
DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA = 5.0
DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA = 20.0
DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS = 100.0
DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF = 5.0

OBSERVATION_COUNT_VARIANCE_ADDITIONAL = "additional"
OBSERVATION_COUNT_VARIANCE_COUNTING_NOISE_INCLUSIVE = "counting_noise_inclusive"
OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL = "complete_statistical"


def normalize_observation_count_variance_semantics(
    semantics: str,
) -> str:
    """Return canonical observation-count variance semantics.

    ``additional`` means that the supplied variance excludes counting noise. The
    ``counting_noise_inclusive`` mode replaces the observed plug-in Poisson
    component with the candidate-dependent expected count. In contrast,
    ``complete_statistical`` declares the supplied covariance to be the full
    statistical covariance of the observation, including weighted-transport
    and detector counting noise, so no extra Poisson term may be added.
    """
    normalized = str(semantics).strip().lower()
    allowed = {
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL,
        OBSERVATION_COUNT_VARIANCE_COUNTING_NOISE_INCLUSIVE,
        OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL,
    }
    if normalized not in allowed:
        raise ValueError(
            "observation_count_variance_semantics must be additional, "
            "counting_noise_inclusive, or complete_statistical."
        )
    return normalized


def normalize_count_likelihood_model(model: str) -> str:
    """Return a canonical count likelihood model name."""
    normalized = str(model).strip().lower()
    if normalized not in {"poisson", "gaussian", "student_t"}:
        raise ValueError(f"Unknown count likelihood model: {model}")
    return normalized


@dataclass(frozen=True)
class CountLikelihoodSpec:
    """Store one normalized count-likelihood configuration."""

    model: str = "poisson"
    transport_model_rel_sigma: float = 0.0
    transport_model_abs_sigma: float = 0.0
    spectrum_count_rel_sigma: float = 0.0
    spectrum_count_abs_sigma: float = 0.0
    low_count_abs_sigma: float = 0.0
    low_count_transition_counts: float = 0.0
    observation_count_variance_semantics: str = (
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL
    )
    student_t_df: float = 5.0

    def __post_init__(self) -> None:
        """Normalize canonical names and numeric likelihood inputs."""
        object.__setattr__(
            self,
            "model",
            normalize_count_likelihood_model(self.model),
        )
        semantics = normalize_observation_count_variance_semantics(
            self.observation_count_variance_semantics,
        )
        object.__setattr__(self, "observation_count_variance_semantics", semantics)
        if (
            semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL
            and self.model == "poisson"
        ):
            raise ValueError(
                "complete_statistical observation variance requires gaussian "
                "or student_t count likelihood; Poisson would ignore the supplied "
                "covariance."
            )
        for field_name in (
            "transport_model_rel_sigma",
            "transport_model_abs_sigma",
            "spectrum_count_rel_sigma",
            "spectrum_count_abs_sigma",
            "low_count_abs_sigma",
            "low_count_transition_counts",
            "student_t_df",
        ):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))


def count_likelihood_variance(
    z_k: NDArray[np.float64],
    lambda_k: NDArray[np.float64],
    *,
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    observation_count_variance_semantics: str = (
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL
    ),
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Return observation variance for count likelihoods with model discrepancy.

    The variance includes Poisson counting noise plus multiplicative uncertainty
    from transport-model mismatch and spectrum-count extraction error. This is
    the likelihood-side place to absorb Geant4 effects that are intentionally
    not represented in the fast PF kernel, such as scatter, build-up, and
    photopeak-decomposition residuals. When the propagated extraction variance
    already contains counting noise, its plug-in ``max(z, 1)`` component is
    removed before the particle-dependent ``lambda`` term is added. With
    ``complete_statistical``, the supplied variance is authoritative and no
    particle-dependent Poisson term is added.
    """
    z_arr = np.maximum(np.asarray(z_k, dtype=float), 0.0)
    lam_arr = np.maximum(np.asarray(lambda_k, dtype=float), float(epsilon))
    transport_rel = max(float(transport_model_rel_sigma), 0.0)
    transport_abs = max(float(transport_model_abs_sigma), 0.0)
    spectrum_rel = max(float(spectrum_count_rel_sigma), 0.0)
    spectrum_abs = max(float(spectrum_count_abs_sigma), 0.0)
    low_count_abs = max(float(low_count_abs_sigma), 0.0)
    low_count_transition = max(float(low_count_transition_counts), 0.0)
    obs_var = np.maximum(np.asarray(observation_count_variance, dtype=float), 0.0)
    semantics = normalize_observation_count_variance_semantics(
        observation_count_variance_semantics,
    )
    poisson_variance = lam_arr
    if semantics == OBSERVATION_COUNT_VARIANCE_COUNTING_NOISE_INCLUSIVE:
        # Response-regression covariance already propagates the Poisson
        # statistics of the observed spectrum.  Retain only its variance in
        # excess of the plug-in source-equivalent Poisson term; ``lam_arr``
        # below supplies that term once for every candidate particle.
        obs_var = np.maximum(obs_var - np.maximum(z_arr, 1.0), 0.0)
    elif semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
        # Weighted-history sum-w2 covariance is already the full observation
        # sampling covariance. Adding ``lam_arr`` would count detector/source
        # statistics a second time and make uncertainty particle-dependent.
        poisson_variance = np.zeros_like(lam_arr, dtype=float)
    scale_ref = np.maximum(z_arr, lam_arr)
    low_count_weight = 0.0
    if low_count_abs > 0.0 and low_count_transition > 0.0:
        low_count_weight = low_count_transition / (scale_ref + low_count_transition)
    variance = (
        poisson_variance
        + (transport_rel * lam_arr) ** 2
        + transport_abs**2
        + (spectrum_rel * scale_ref) ** 2
        + spectrum_abs**2
        + (low_count_abs * low_count_weight) ** 2
        + obs_var
    )
    return np.maximum(variance, float(epsilon))


def count_likelihood_variance_torch(
    z_k: "torch.Tensor",
    lambda_k: "torch.Tensor",
    *,
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | "torch.Tensor" = 0.0,
    observation_count_variance_semantics: str = (
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL
    ),
    epsilon: float = 1e-12,
) -> "torch.Tensor":
    """Return torch observation variance equivalent to count_likelihood_variance."""
    import torch

    if torch.is_tensor(lambda_k):
        lam_arr = lambda_k.to(dtype=torch.float64)
    else:
        lam_arr = torch.as_tensor(lambda_k, dtype=torch.float64)
    lam_arr = torch.clamp(lam_arr, min=float(epsilon))
    z_arr = torch.clamp(
        torch.as_tensor(z_k, device=lam_arr.device, dtype=torch.float64),
        min=0.0,
    )
    obs_var = torch.clamp(
        torch.as_tensor(
            observation_count_variance,
            device=lam_arr.device,
            dtype=torch.float64,
        ),
        min=0.0,
    )
    semantics = normalize_observation_count_variance_semantics(
        observation_count_variance_semantics,
    )
    poisson_variance = lam_arr
    if semantics == OBSERVATION_COUNT_VARIANCE_COUNTING_NOISE_INCLUSIVE:
        obs_var = torch.clamp(
            obs_var - torch.clamp(z_arr, min=1.0),
            min=0.0,
        )
    elif semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
        poisson_variance = torch.zeros_like(lam_arr)
    transport_rel = max(float(transport_model_rel_sigma), 0.0)
    transport_abs = max(float(transport_model_abs_sigma), 0.0)
    spectrum_rel = max(float(spectrum_count_rel_sigma), 0.0)
    spectrum_abs = max(float(spectrum_count_abs_sigma), 0.0)
    low_count_abs = max(float(low_count_abs_sigma), 0.0)
    low_count_transition = max(float(low_count_transition_counts), 0.0)
    scale_ref = torch.maximum(z_arr, lam_arr)
    low_count_weight = torch.zeros_like(scale_ref)
    if low_count_abs > 0.0 and low_count_transition > 0.0:
        low_count_weight = low_count_transition / (scale_ref + low_count_transition)
    variance = (
        poisson_variance
        + (transport_rel * lam_arr) ** 2
        + transport_abs**2
        + (spectrum_rel * scale_ref) ** 2
        + spectrum_abs**2
        + (low_count_abs * low_count_weight) ** 2
        + obs_var
    )
    return torch.clamp(variance, min=float(epsilon))


def predictive_count_likelihood_variance(
    posterior_mean_lambda: NDArray[np.float64],
    *,
    spec: CountLikelihoodSpec,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Return the plug-in likelihood variance at a posterior-mean count rate.

    The future observation is unknown during planning, so the observation-side
    scale reference is set to the posterior mean itself. For a Poisson model,
    model-discrepancy and extraction variances are intentionally ignored just
    as they are in the Poisson likelihood.
    """
    mean_lambda = np.maximum(
        np.asarray(posterior_mean_lambda, dtype=float),
        float(epsilon),
    )
    if spec.model == "poisson":
        return mean_lambda
    return count_likelihood_variance(
        mean_lambda,
        mean_lambda,
        transport_model_rel_sigma=spec.transport_model_rel_sigma,
        transport_model_abs_sigma=spec.transport_model_abs_sigma,
        spectrum_count_rel_sigma=spec.spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spec.spectrum_count_abs_sigma,
        low_count_abs_sigma=spec.low_count_abs_sigma,
        low_count_transition_counts=spec.low_count_transition_counts,
        observation_count_variance=observation_count_variance,
        observation_count_variance_semantics=(
            spec.observation_count_variance_semantics
        ),
        epsilon=epsilon,
    )


def predictive_count_likelihood_variance_torch(
    posterior_mean_lambda: "torch.Tensor",
    *,
    spec: CountLikelihoodSpec,
    observation_count_variance: float | "torch.Tensor" = 0.0,
    epsilon: float = 1e-12,
) -> "torch.Tensor":
    """Return the Torch equivalent of predictive_count_likelihood_variance."""
    import torch

    if torch.is_tensor(posterior_mean_lambda):
        mean_lambda = posterior_mean_lambda.to(dtype=torch.float64)
    else:
        mean_lambda = torch.as_tensor(posterior_mean_lambda, dtype=torch.float64)
    mean_lambda = torch.clamp(mean_lambda, min=float(epsilon))
    if spec.model == "poisson":
        return mean_lambda
    return count_likelihood_variance_torch(
        mean_lambda,
        mean_lambda,
        transport_model_rel_sigma=spec.transport_model_rel_sigma,
        transport_model_abs_sigma=spec.transport_model_abs_sigma,
        spectrum_count_rel_sigma=spec.spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spec.spectrum_count_abs_sigma,
        low_count_abs_sigma=spec.low_count_abs_sigma,
        low_count_transition_counts=spec.low_count_transition_counts,
        observation_count_variance=observation_count_variance,
        observation_count_variance_semantics=(
            spec.observation_count_variance_semantics
        ),
        epsilon=epsilon,
    )


def count_log_likelihood_terms_np(
    z_k: NDArray[np.float64],
    lambda_k: NDArray[np.float64],
    *,
    spec: CountLikelihoodSpec,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Return broadcast count-log-likelihood terms without summing dimensions."""
    z_arr = np.asarray(z_k, dtype=float)
    lam_arr = np.maximum(np.asarray(lambda_k, dtype=float), float(epsilon))
    if spec.model == "poisson":
        return np.asarray(z_arr * np.log(lam_arr) - lam_arr, dtype=float)

    variance = count_likelihood_variance(
        z_arr,
        lam_arr,
        transport_model_rel_sigma=spec.transport_model_rel_sigma,
        transport_model_abs_sigma=spec.transport_model_abs_sigma,
        spectrum_count_rel_sigma=spec.spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spec.spectrum_count_abs_sigma,
        low_count_abs_sigma=spec.low_count_abs_sigma,
        low_count_transition_counts=spec.low_count_transition_counts,
        observation_count_variance=observation_count_variance,
        observation_count_variance_semantics=(
            spec.observation_count_variance_semantics
        ),
        epsilon=epsilon,
    )
    residual = z_arr - lam_arr
    if spec.model == "gaussian":
        terms = -0.5 * ((residual**2) / variance + np.log(variance))
        return np.asarray(terms, dtype=float)

    df = max(float(spec.student_t_df), 1.0 + float(epsilon))
    terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
    terms -= 0.5 * np.log(variance)
    return np.asarray(terms, dtype=float)


def count_log_likelihood_terms_torch(
    z_k: "torch.Tensor",
    lambda_k: "torch.Tensor",
    *,
    spec: CountLikelihoodSpec,
    observation_count_variance: float | "torch.Tensor" = 0.0,
    epsilon: float = 1e-12,
) -> "torch.Tensor":
    """Return Torch terms equivalent to count_log_likelihood_terms_np."""
    import torch

    if torch.is_tensor(lambda_k):
        lam_arr = lambda_k.to(dtype=torch.float64)
    else:
        lam_arr = torch.as_tensor(lambda_k, dtype=torch.float64)
    lam_arr = torch.clamp(lam_arr, min=float(epsilon))
    z_arr = torch.as_tensor(z_k, device=lam_arr.device, dtype=torch.float64)
    if spec.model == "poisson":
        return z_arr * torch.log(lam_arr) - lam_arr

    variance = count_likelihood_variance_torch(
        z_arr,
        lam_arr,
        transport_model_rel_sigma=spec.transport_model_rel_sigma,
        transport_model_abs_sigma=spec.transport_model_abs_sigma,
        spectrum_count_rel_sigma=spec.spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spec.spectrum_count_abs_sigma,
        low_count_abs_sigma=spec.low_count_abs_sigma,
        low_count_transition_counts=spec.low_count_transition_counts,
        observation_count_variance=observation_count_variance,
        observation_count_variance_semantics=(
            spec.observation_count_variance_semantics
        ),
        epsilon=epsilon,
    )
    residual = z_arr - lam_arr
    if spec.model == "gaussian":
        return -0.5 * ((residual**2) / variance + torch.log(variance))

    df = max(float(spec.student_t_df), 1.0 + float(epsilon))
    return -0.5 * (df + 1.0) * torch.log1p(
        (residual**2) / (df * variance)
    ) - 0.5 * torch.log(variance)


def count_log_likelihood(
    z_k: NDArray[np.float64],
    lambda_k: NDArray[np.float64],
    *,
    model: str = "poisson",
    transport_model_rel_sigma: float = 0.0,
    transport_model_abs_sigma: float = 0.0,
    spectrum_count_rel_sigma: float = 0.0,
    spectrum_count_abs_sigma: float = 0.0,
    low_count_abs_sigma: float = 0.0,
    low_count_transition_counts: float = 0.0,
    observation_count_variance: float | NDArray[np.float64] = 0.0,
    observation_count_variance_semantics: str = (
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL
    ),
    student_t_df: float = 5.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Return a count log-likelihood with optional model-discrepancy variance.

    ``poisson`` preserves the original PF likelihood. ``gaussian`` and
    ``student_t`` allow non-integer spectrum-decomposition counts and inflate
    the variance for unmodelled Geant4 transport effects.
    """
    spec = CountLikelihoodSpec(
        model=model,
        transport_model_rel_sigma=transport_model_rel_sigma,
        transport_model_abs_sigma=transport_model_abs_sigma,
        spectrum_count_rel_sigma=spectrum_count_rel_sigma,
        spectrum_count_abs_sigma=spectrum_count_abs_sigma,
        low_count_abs_sigma=low_count_abs_sigma,
        low_count_transition_counts=low_count_transition_counts,
        observation_count_variance_semantics=observation_count_variance_semantics,
        student_t_df=student_t_df,
    )
    terms = count_log_likelihood_terms_np(
        z_k,
        lambda_k,
        spec=spec,
        observation_count_variance=observation_count_variance,
        epsilon=epsilon,
    )
    return float(np.sum(terms))


def expected_counts_per_source(
    kernel: ContinuousKernel,
    isotope: str,
    detector_positions: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    live_times: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
    source_scale: float | NDArray[np.float64] = 1.0,
) -> NDArray[np.float64]:
    """
    Return per-source expected counts Λ_{k,m} for each measurement k.

    source_scale maps ideal source counts into the measurement domain while
    leaving the additive background term to the caller. It may be a scalar or a
    length-``num_meas`` vector for shield-pair-conditioned calibration.
    """
    detector_arr = np.asarray(detector_positions, dtype=float)
    source_arr = np.asarray(sources, dtype=float)
    strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
    live_arr = np.asarray(live_times, dtype=float).reshape(-1)
    fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
    pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
    num_meas = int(live_arr.size)
    if detector_arr.shape != (num_meas, 3):
        raise ValueError("detector_positions must be shaped num_measurements x 3.")
    if fe_arr.size != num_meas or pb_arr.size != num_meas:
        raise ValueError(
            "fe_indices and pb_indices must contain one value per measurement."
        )
    if source_arr.size == 0:
        return np.zeros((num_meas, 0), dtype=float)
    source_arr = source_arr.reshape(-1, 3)
    if source_arr.shape[0] != strength_arr.shape[0]:
        raise ValueError("sources and strengths must have matching length.")
    num_sources = int(source_arr.shape[0])
    scale_arr = _source_scale_vector(source_scale, num_meas)
    values = kernel.kernel_values_selected_pairs_for_detectors(
        isotope=isotope,
        detector_positions=detector_arr,
        sources=source_arr,
        fe_indices=fe_arr,
        pb_indices=pb_arr,
    )
    values_arr = np.asarray(values, dtype=float)
    if values_arr.shape != (num_meas, num_sources):
        raise ValueError(
            "Batched selected-pair kernel returned shape "
            f"{values_arr.shape}, expected {(num_meas, num_sources)}."
        )
    scale = live_arr * scale_arr
    return scale[:, None] * values_arr * strength_arr[None, :]


def _source_scale_vector(
    source_scale: float | NDArray[np.float64],
    num_meas: int,
) -> NDArray[np.float64]:
    """Return a non-negative source-response scale per measurement."""
    arr = np.asarray(source_scale, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.ones(max(int(num_meas), 0), dtype=float)
    if arr.size == 1:
        return np.full(max(int(num_meas), 0), max(float(arr[0]), 0.0), dtype=float)
    if arr.size != int(num_meas):
        raise ValueError("source_scale vector must match the number of measurements.")
    return np.maximum(arr.astype(float, copy=False), 0.0)
