"""Tests for complete statistical covariance in PF count likelihoods."""

from __future__ import annotations

import numpy as np
import pytest

from pf.estimator import RotatingShieldPFConfig
from pf.likelihood import (
    CountLikelihoodSpec,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    delta_log_likelihood_update,
    normalize_observation_count_variance_semantics,
)
from pf.particle_filter import IsotopeParticle, IsotopeParticleFilter, PFConfig
from pf.state import IsotopeState


def test_complete_statistical_variance_does_not_add_candidate_poisson_noise() -> None:
    """A complete supplied variance must remain independent of candidate lambda."""
    z_obs = np.asarray([[100.0]], dtype=float)
    lambdas = np.asarray([[80.0, 120.0]], dtype=float)
    supplied = np.asarray([[5_000.0]], dtype=float)

    actual = count_likelihood_variance(
        z_obs,
        lambdas,
        observation_count_variance=supplied,
        observation_count_variance_semantics="complete_statistical",
    )

    np.testing.assert_array_equal(actual, np.full_like(lambdas, 5_000.0))


def test_complete_statistical_variance_numpy_torch_equivalence() -> None:
    """Torch and NumPy paths must share complete covariance semantics exactly."""
    torch = pytest.importorskip("torch")
    z_obs = np.asarray([[100.0], [20.0]], dtype=float)
    lambdas = np.asarray([[80.0, 120.0], [15.0, 30.0]], dtype=float)
    supplied = np.asarray([[5_000.0], [700.0]], dtype=float)
    kwargs = {
        "transport_model_rel_sigma": 0.1,
        "spectrum_count_abs_sigma": 3.0,
        "observation_count_variance_semantics": "complete_statistical",
    }

    expected = count_likelihood_variance(
        z_obs,
        lambdas,
        observation_count_variance=supplied,
        **kwargs,
    )
    actual = count_likelihood_variance_torch(
        torch.as_tensor(z_obs, dtype=torch.float64),
        torch.as_tensor(lambdas, dtype=torch.float64),
        observation_count_variance=torch.as_tensor(
            supplied,
            dtype=torch.float64,
        ),
        **kwargs,
    )

    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_legacy_boolean_maps_to_counting_noise_inclusive_semantics() -> None:
    """The old boolean must keep its plug-in counting-noise behavior."""
    assert (
        normalize_observation_count_variance_semantics(
            "",
            includes_counting_noise=True,
        )
        == "counting_noise_inclusive"
    )
    config = PFConfig(
        count_likelihood_model="gaussian",
        observation_count_variance_includes_counting_noise=True,
    )
    assert config.observation_count_variance_semantics == "counting_noise_inclusive"


def test_complete_covariance_rejects_poisson_likelihood() -> None:
    """Poisson likelihood must not silently discard a supplied full covariance."""
    with pytest.raises(ValueError, match="complete_statistical.*requires"):
        CountLikelihoodSpec(
            model="poisson",
            observation_count_variance_semantics="complete_statistical",
        )
    with pytest.raises(ValueError, match="complete_statistical.*requires"):
        PFConfig(
            count_likelihood_model="poisson",
            observation_count_variance_semantics="complete_statistical",
        )
    with pytest.raises(ValueError, match="complete_statistical.*requires"):
        delta_log_likelihood_update(
            np.asarray([10.0], dtype=float),
            np.asarray([9.0], dtype=float),
            np.asarray([10.0], dtype=float),
            model="poisson",
            observation_count_variance=np.asarray([500.0], dtype=float),
            observation_count_variance_semantics="complete_statistical",
        )


@pytest.mark.parametrize("config_type", [PFConfig, RotatingShieldPFConfig])
def test_complete_covariance_disables_derived_count_likelihoods(
    config_type: type[PFConfig] | type[RotatingShieldPFConfig],
) -> None:
    """Derived shield terms must not reuse completely modelled count data."""
    config = config_type(
        count_likelihood_model="student_t",
        observation_count_variance_semantics="complete_statistical",
        shield_contrast_likelihood_enable=True,
        shield_view_ratio_likelihood_enable=True,
    )

    assert config.shield_contrast_likelihood_enable is False
    assert config.shield_view_ratio_likelihood_enable is False


@pytest.mark.parametrize("config_type", [PFConfig, RotatingShieldPFConfig])
def test_additional_variance_keeps_derived_count_likelihoods(
    config_type: type[PFConfig] | type[RotatingShieldPFConfig],
) -> None:
    """Standard variance semantics must retain configured shield auxiliaries."""
    config = config_type(
        shield_contrast_likelihood_enable=True,
        shield_view_ratio_likelihood_enable=True,
    )

    assert config.shield_contrast_likelihood_enable is True
    assert config.shield_view_ratio_likelihood_enable is True


def test_complete_covariance_routes_weighted_spectrum_to_count_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Correlated weighted spectra must bypass the independent-bin likelihood."""
    torch = pytest.importorskip("torch")
    config = PFConfig(
        num_particles=2,
        count_likelihood_model="gaussian",
        observation_count_variance_semantics="complete_statistical",
        use_gpu=True,
        use_tempering=False,
        resample_threshold=0.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.asarray([0.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.orientations = [np.asarray([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=dummy_kernel,
        config=config,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        )
        for _ in range(2)
    ]

    def fake_gpu_enabled() -> bool:
        """Pretend that the torch backend is available for the path test."""
        return True

    def fake_counts(**_kwargs: object) -> "torch.Tensor":
        """Return count predictions favoring the first particle."""
        return torch.as_tensor([[10.0, 30.0]], dtype=torch.float64)

    def forbidden_spectrum(*_args: object, **_kwargs: object) -> "torch.Tensor":
        """Fail if the diagonal direct-spectrum helper is reached."""
        raise AssertionError("direct spectrum likelihood was invoked")

    def noop() -> None:
        """Skip non-likelihood side effects in this path test."""
        return None

    def noop_kwargs(**_kwargs: object) -> None:
        """Skip keyword-only non-likelihood side effects in this path test."""
        return None

    monkeypatch.setattr(filt, "_gpu_enabled", fake_gpu_enabled)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(
        filt,
        "_spectral_bin_sequence_log_likelihood_from_lambda_gpu",
        forbidden_spectrum,
    )
    monkeypatch.setattr(filt, "_maybe_resample_continuous", noop)
    monkeypatch.setattr(filt, "align_continuous_labels", noop)
    monkeypatch.setattr(filt, "_advance_adapt_cooldown", noop)
    monkeypatch.setattr(filt, "adapt_num_particles", noop_kwargs)
    monkeypatch.setattr(filt, "_maybe_update_convergence", noop_kwargs)

    filt.update_continuous_pair_sequence(
        z_obs=np.asarray([10.0], dtype=float),
        pose_idx=0,
        fe_indices=np.asarray([0], dtype=int),
        pb_indices=np.asarray([0], dtype=int),
        live_times_s=np.asarray([1.0], dtype=float),
        observation_count_variances=np.asarray([500.0], dtype=float),
        spectrum_counts=np.asarray([[0.0, 30.0]], dtype=float),
        spectrum_response_template=np.asarray([[0.0, 1.0]], dtype=float),
        spectrum_background=np.zeros((1, 2), dtype=float),
        spectrum_variance=np.asarray([[500.0, 500.0]], dtype=float),
    )

    assert filt.last_spectrum_likelihood_route == "count_covariance"
    assert filt.continuous_weights[0] > filt.continuous_weights[1]


def test_standard_defaults_keep_direct_spectrum_likelihood_enabled() -> None:
    """Standard unweighted defaults must retain the existing spectrum path."""
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(num_particles=1),
    )

    assert filt.config.observation_count_variance_semantics == "additional"
    assert filt._direct_spectrum_likelihood_enabled() is True
