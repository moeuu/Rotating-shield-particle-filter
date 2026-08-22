"""Contracts for the production joint full-spectrum PF observation model."""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY = (
    "full_spectrum_contract_hash_sha256"
)


@runtime_checkable
class FullSpectrumGenerativeModel(Protocol):
    """Define the fail-closed full-spectrum model shared by PF and planning.

    Implementations own the complete observation distribution.  In
    particular, the PF must not add a second Poisson term, a projected
    isotope-count covariance, or another likelihood derived from the same
    spectrum.
    """

    @property
    def runtime_ready(self) -> bool:
        """Return whether training-only contracts authorize runtime use."""

    @property
    def production_ready(self) -> bool:
        """Return whether independent holdout gates approved formal release."""

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable model, response, and energy-axis digest."""

    @property
    def energy_axis_keV(self) -> NDArray[np.float64]:
        """Return the exact analysis-spectrum bin axis."""

    @property
    def line_identity(self) -> tuple[Mapping[str, object], ...]:
        """Return the global positive transport-line order."""

    @property
    def transport_feature_order(self) -> tuple[str, ...]:
        """Return the final-axis order of geometry-conditioned line features."""

    def require_runtime_ready(self) -> None:
        """Raise unless the immutable training-only model is runtime-ready."""

    def require_production_ready(self) -> None:
        """Raise unless the immutable model passed every production gate."""

    def log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one joint full-spectrum log likelihood per particle."""

    def log_likelihood_torch(
        self,
        observed_spectrum_vb: object,
        total_line_contributions_nvsl: object,
        uncollided_line_contributions_nvsl: object,
        transport_features_nvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch-equivalent joint log likelihood per particle."""

    def prefix_log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return exact shared-latent likelihoods for every view prefix."""

    def prefix_log_likelihood_torch(
        self,
        observed_spectrum_vb: object,
        total_line_contributions_nvsl: object,
        uncollided_line_contributions_nvsl: object,
        transport_features_nvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return Torch shared-latent likelihoods for every view prefix."""

    def cross_log_likelihood_numpy(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return batched action/sample/state full-spectrum log likelihoods."""

    def cross_log_likelihood_torch(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> object:
        """Return Torch-equivalent batched action/sample/state likelihoods."""

    def predict_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected analysis spectra ending in view/bin axes."""

    def predict_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return Torch expected analysis spectra ending in view/bin axes."""

    def sample_predictive_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        sample_count: int,
        rng: np.random.Generator,
        action_seeds_a: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Draw exact future spectra shaped state x sample x view x bin."""

    def posterior_predictive_innovation_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        particle_weights_n: NDArray[np.float64],
        *,
        confidence: float,
    ) -> Mapping[str, float | int | bool | None]:
        """Return a model-native calibrated posterior innovation diagnostic."""

    def birth_proposal_log_scores_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        candidate_total_line_contributions_gvsl: NDArray[np.float64],
        candidate_uncollided_line_contributions_gvsl: NDArray[np.float64],
        candidate_transport_features_gvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        target_line_mask_l: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return deterministic proposal-only scores for target-line candidates."""

    def birth_proposal_log_scores_torch(
        self,
        observed_spectrum_vb: object,
        candidate_total_line_contributions_gvsl: object,
        candidate_uncollided_line_contributions_gvsl: object,
        candidate_transport_features_gvslf: object,
        live_times_s_v: object,
        *,
        target_line_mask_l: object,
    ) -> object:
        """Return Torch-equivalent deterministic proposal-only scores."""

    def manifest_payload(self) -> Mapping[str, object]:
        """Return immutable model and validation provenance."""


@runtime_checkable
class TorchPredictiveFullSpectrumModel(Protocol):
    """Define the optional device-resident predictive sampling capability."""

    def sample_predictive_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
        *,
        sample_count: int,
        generator: object | None = None,
        action_seeds_a: object | None = None,
    ) -> object:
        """Draw exact integer spectra without leaving the Torch device."""


def validate_full_spectrum_model(
    model: object,
) -> FullSpectrumGenerativeModel:
    """Return a structurally complete, training-approved runtime model."""
    if not isinstance(model, FullSpectrumGenerativeModel):
        raise TypeError(
            "Pure PF requires a FullSpectrumGenerativeModel implementing the "
            "shared NumPy/Torch likelihood, predictive sampler, and manifest."
        )
    model.require_runtime_ready()
    runtime_ready = model.runtime_ready
    if type(runtime_ready) is not bool or runtime_ready is not True:
        raise RuntimeError(
            "Full-spectrum model reported runtime_ready=False after its "
            "training-only runtime gate."
        )
    contract_hash = model.contract_hash_sha256
    if (
        type(contract_hash) is not str
        or
        len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash)
    ):
        raise ValueError("Full-spectrum model contract hash must be SHA-256.")
    energy_axis = np.asarray(model.energy_axis_keV, dtype=np.float64)
    if (
        energy_axis.ndim != 1
        or energy_axis.size == 0
        or np.any(~np.isfinite(energy_axis))
        or np.any(np.diff(energy_axis) <= 0.0)
    ):
        raise ValueError(
            "Full-spectrum model energy axis must be finite and strictly "
            "increasing."
        )
    line_identity = tuple(model.line_identity)
    if not line_identity:
        raise ValueError("Full-spectrum model requires positive transport lines.")
    feature_order = tuple(model.transport_feature_order)
    if (
        any(type(value) is not str or not value for value in feature_order)
        or feature_order
        != ("tau_fe", "tau_pb", "tau_obstacle", "distance_m")
    ):
        raise ValueError(
            "Full-spectrum transport features must use the canonical order "
            "(tau_fe, tau_pb, tau_obstacle, distance_m)."
        )
    return model


def validate_observed_spectrum(
    spectrum_vb: NDArray[np.float64],
    *,
    expected_bin_count: int,
) -> NDArray[np.float64]:
    """Return an unweighted integer-count view-major analysis spectrum.

    The production likelihood models unit-weight detected events.  Fractional,
    efficiency-corrected, or variance-reduced spectra have different sampling
    laws and must remain in explicitly diagnostic replay paths.
    """
    if isinstance(expected_bin_count, (bool, np.bool_)) or not isinstance(
        expected_bin_count,
        Integral,
    ):
        raise TypeError("expected_bin_count must be an integer.")
    bin_count = int(expected_bin_count)
    if bin_count <= 0:
        raise ValueError("expected_bin_count must be positive.")
    raw_spectrum = np.asarray(spectrum_vb)
    if raw_spectrum.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(
            "Observed full spectra must contain JSON numbers, not values "
            "coercible to numbers."
        )
    spectrum = np.asarray(raw_spectrum, dtype=np.float64)
    if (
        spectrum.ndim != 2
        or int(spectrum.shape[1]) != bin_count
        or int(spectrum.shape[0]) == 0
        or np.any(~np.isfinite(spectrum))
        or np.any(spectrum < 0.0)
    ):
        raise ValueError(
            "Observed full spectra must be finite, nonnegative, nonempty, and "
            "shaped view x model-energy-bin."
        )
    if np.any(spectrum > float(2**53)) or np.any(
        spectrum != np.rint(spectrum)
    ):
        raise ValueError(
            "Production PF spectra must contain exact unit-weight integer event "
            "counts; weighted, corrected, and fractional spectra are unsupported."
        )
    return np.ascontiguousarray(spectrum)
