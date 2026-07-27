"""Tests for complete statistical covariance in PF count likelihoods."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pf.estimator import RotatingShieldPFConfig
from pf.likelihood import (
    CountLikelihoodSpec,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    normalize_observation_count_variance_semantics,
)
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    MeasurementData,
    PFConfig,
)
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


@pytest.mark.parametrize(
    "semantics",
    [
        "",
        "auto",
        "legacy",
        "excludes_counting_noise",
        "extra",
        "includes_counting_noise",
        "legacy_includes_counting_noise",
        "complete",
        "full_statistical",
        "transport_and_counting",
    ],
)
def test_observation_variance_semantics_rejects_legacy_aliases(
    semantics: str,
) -> None:
    """Only the three canonical observation-variance semantics are accepted."""
    with pytest.raises(ValueError, match="observation_count_variance_semantics"):
        normalize_observation_count_variance_semantics(semantics)
    with pytest.raises(ValueError, match="observation_count_variance_semantics"):
        CountLikelihoodSpec(
            model="gaussian",
            observation_count_variance_semantics=semantics,
        )
    with pytest.raises(ValueError, match="observation_count_variance_semantics"):
        PFConfig(
            count_likelihood_model="gaussian",
            observation_count_variance_semantics=semantics,
        )


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


def test_station_likelihood_blocks_require_explicit_sequence_ids() -> None:
    """Structural likelihood blocks must never be inferred from detector positions."""
    parameter = inspect.signature(MeasurementData).parameters[
        "station_sequence_ids"
    ]
    assert parameter.default is inspect.Parameter.empty
    route_parameter = inspect.signature(MeasurementData).parameters[
        "runtime_likelihood_routes"
    ]
    assert route_parameter.default is inspect.Parameter.empty
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(0, 0),
            use_gpu=False,
        ),
    )
    data = MeasurementData(
        z_k=np.ones(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=np.zeros((2, 3), dtype=float),
        fe_indices=np.zeros(2, dtype=np.int64),
        pb_indices=np.zeros(2, dtype=np.int64),
        live_times=np.ones(2, dtype=float),
        station_sequence_ids=np.asarray([0, 1], dtype=np.int64),
        runtime_likelihood_routes=np.asarray(
            ["count", "count"],
            dtype="<U16",
        ),
    )

    blocks = filt._station_likelihood_block_rows(data)
    np.testing.assert_array_equal(blocks[1], [[0], [1]])


@pytest.mark.parametrize("likelihood_model", ["gaussian", "student_t"])
def test_exact_structural_likelihood_matches_runtime_covariance_sequence(
    likelihood_model: str,
) -> None:
    """Exact RJ evidence must use the same block covariance as PF updates."""
    torch = pytest.importorskip("torch")
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=1,
            max_sources=2,
            variable_cardinality=False,
            init_num_sources=(0, 0),
            count_likelihood_model=likelihood_model,
            count_likelihood_df=6.0,
            transport_model_rel_sigma=0.08,
            transport_model_abs_sigma=1.5,
            spectrum_count_rel_sigma=0.12,
            spectrum_count_abs_sigma=0.7,
            low_count_abs_sigma=0.5,
            low_count_transition_counts=25.0,
            station_view_covariance_enable=True,
            station_view_correlated_spectrum_fraction=0.4,
            shield_contrast_likelihood_enable=True,
            shield_contrast_likelihood_weight=0.3,
            shield_contrast_min_count=1.0,
            shield_view_ratio_likelihood_enable=True,
            shield_view_ratio_likelihood_weight=0.2,
            shield_view_ratio_likelihood_min_total_count=1.0,
            use_gpu=False,
        ),
    )
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = np.asarray(
        [[0.0, 0.20, 0.10], [0.20, 0.0, 0.15], [0.10, 0.15, 0.0]],
        dtype=float,
    )
    covariance[3:, 3:] = np.asarray(
        [[0.0, 0.12, 0.08], [0.12, 0.0, 0.11], [0.08, 0.11, 0.0]],
        dtype=float,
    )
    z_values = np.asarray(
        [72.0, 48.0, 31.0, 26.0, 39.0, 57.0],
        dtype=float,
    )
    data = MeasurementData(
        z_k=z_values,
        observation_variances=np.asarray(
            [4.0, 5.0, 3.5, 2.5, 4.5, 6.0],
            dtype=float,
        ),
        detector_positions=np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.0, 0.5],
            ],
            dtype=float,
        ),
        fe_indices=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        pb_indices=np.asarray([2, 1, 0, 2, 1, 0], dtype=np.int64),
        live_times=np.ones(6, dtype=float),
        station_sequence_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        runtime_likelihood_routes=np.asarray(
            ["count_covariance"] * 6,
            dtype="<U20",
        ),
        observation_count_covariance=covariance,
    )
    expected_counts = np.asarray(
        [
            [68.0, 75.0, 58.0, 83.0],
            [51.0, 44.0, 55.0, 39.0],
            [28.0, 35.0, 24.0, 42.0],
            [29.0, 22.0, 34.0, 18.0],
            [36.0, 43.0, 31.0, 49.0],
            [61.0, 52.0, 67.0, 46.0],
        ],
        dtype=float,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(
        data,
        expected_counts,
    )
    runtime = torch.zeros(expected_counts.shape[1], dtype=torch.float64)
    for rows in (np.arange(3), np.arange(3, 6)):
        runtime += filt._log_likelihood_sequence_gpu(
            torch.as_tensor(expected_counts[rows], dtype=torch.float64),
            data.z_k[rows],
            data.observation_variances[rows],
            runtime_likelihood_route="count_covariance",
            observation_count_covariance=covariance[np.ix_(rows, rows)],
        )

    np.testing.assert_allclose(
        structural,
        runtime.numpy(),
        rtol=1.0e-11,
        atol=1.0e-10,
    )


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


def test_complete_covariance_uses_count_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete statistical variance must use the count likelihood."""
    torch = pytest.importorskip("torch")
    config = PFConfig(
        num_particles=2,
        variable_cardinality=False,
        init_num_sources=(0, 0),
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

    def noop() -> None:
        """Skip non-likelihood side effects in this path test."""
        return None

    monkeypatch.setattr(filt, "_gpu_enabled", fake_gpu_enabled)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(filt, "_maybe_resample_continuous", noop)

    filt.update_continuous_pair_sequence(
        z_obs=np.asarray([10.0], dtype=float),
        pose_idx=0,
        fe_indices=np.asarray([0], dtype=int),
        pb_indices=np.asarray([0], dtype=int),
        live_times_s=np.asarray([1.0], dtype=float),
        runtime_likelihood_route="count",
        observation_count_variances=np.asarray([500.0], dtype=float),
    )

    assert filt.last_runtime_likelihood_route == "count"
    assert filt.continuous_weights[0] > filt.continuous_weights[1]


def test_ral_uncertainty_uses_count_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAL transport and station uncertainty must use batched count covariance."""
    torch = pytest.importorskip("torch")
    config = PFConfig(
        num_particles=2,
        variable_cardinality=False,
        init_num_sources=(0, 0),
        count_likelihood_model="student_t",
        count_likelihood_df=5.0,
        transport_model_rel_sigma=0.1,
        transport_model_abs_sigma=5.0,
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=5.0,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=1.0,
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
    expected_counts = torch.as_tensor(
        [[12.0, 30.0], [18.0, 45.0]],
        dtype=torch.float64,
    )
    z_obs = np.asarray([13.0, 20.0], dtype=float)
    count_variances = np.asarray([4.0, 6.0], dtype=float)
    count_covariance = np.asarray(
        [[0.0, 2.0], [2.0, 0.0]],
        dtype=float,
    )
    expected_ll = filt._log_likelihood_sequence_gpu(
        expected_counts,
        z_obs,
        count_variances,
        runtime_likelihood_route="count_covariance",
        observation_count_covariance=count_covariance,
    )
    expected_weights = torch.softmax(expected_ll, dim=0).numpy()

    def fake_gpu_enabled() -> bool:
        """Pretend that the torch backend is available for the route test."""
        return True

    def fake_counts(**_kwargs: object) -> "torch.Tensor":
        """Return deterministic batched particle count predictions."""
        return expected_counts

    def noop() -> None:
        """Skip non-likelihood side effects in this route test."""
        return None

    monkeypatch.setattr(filt, "_gpu_enabled", fake_gpu_enabled)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(filt, "_maybe_resample_continuous", noop)

    filt.update_continuous_pair_sequence(
        z_obs=z_obs,
        pose_idx=0,
        fe_indices=np.asarray([0, 0], dtype=int),
        pb_indices=np.asarray([0, 0], dtype=int),
        live_times_s=np.ones(2, dtype=float),
        runtime_likelihood_route="count_covariance",
        observation_count_variances=count_variances,
        observation_count_covariance=count_covariance,
    )

    assert filt.last_runtime_likelihood_route == "count_covariance"
    np.testing.assert_allclose(
        filt.continuous_weights,
        expected_weights,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
