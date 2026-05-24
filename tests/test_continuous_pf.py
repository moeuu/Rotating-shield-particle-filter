"""Basic tests for continuous measurement model and PF scaffold."""

import numpy as np
import pytest

from measurement.continuous_kernels import expected_counts_single_isotope
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import is_allowed_source_surface_position
from pf.likelihood import (
    count_likelihood_variance,
    count_log_likelihood,
    normalize_count_likelihood_model,
)
from pf.particle_filter import IsotopeParticleFilter, PFConfig, IsotopeParticle
from pf.state import IsotopeState


def test_geometric_scaling_inverse_square() -> None:
    """Expected counts should follow inverse-square scaling without shielding."""
    src = np.array([[0.0, 0.0, 0.0]])
    strength = np.array([10.0])
    d1 = 1.0
    d2 = 2.0
    lam1 = expected_counts_single_isotope(
        detector_position=np.array([d1, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam2 = expected_counts_single_isotope(
        detector_position=np.array([d2, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    assert np.allclose(lam1 / lam2, (d2**2) / (d1**2), rtol=1e-6)


def test_shield_attenuation_factor_both_materials() -> None:
    """When both Fe and Pb block, expected counts should follow exp(-mu*L)."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strength = np.array([5.0])
    lam_free = expected_counts_single_isotope(
        detector_position=det,
        RFe=np.array([1.0, 1.0, 1.0]),
        RPb=np.array([1.0, 1.0, 1.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam_blocked = expected_counts_single_isotope(
        detector_position=det,
        RFe=np.array([-1.0, -1.0, -1.0]),
        RPb=np.array([-1.0, -1.0, -1.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    from measurement.kernels import ShieldParams

    shield_params = ShieldParams()
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(lam_blocked, expected_ratio * lam_free, rtol=1e-6)


def test_poisson_weight_update_prefers_higher_lambda() -> None:
    """Weight update should favor particle with higher expected Λ for given z."""
    cfg = PFConfig(num_particles=2)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    # Override continuous particles with deterministic states
    p_hi = IsotopeState(num_sources=1, positions=np.array([[1.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0)
    p_lo = IsotopeState(num_sources=1, positions=np.array([[5.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0)
    pf.continuous_particles = [
        IsotopeParticle(state=p_hi, log_weight=np.log(0.5)),
        IsotopeParticle(state=p_lo, log_weight=np.log(0.5)),
    ]
    pf.kernel = dummy_kernel
    z_obs = 1.0
    pf.update_continuous_pair(z_obs=z_obs, pose_idx=0, fe_index=0, pb_index=0, live_time_s=1.0)
    weights = pf.continuous_weights
    assert weights[0] > weights[1]


def test_deferred_pair_update_allows_scaled_roughening() -> None:
    """Deferred station updates should resample with small roughening, not freeze positions."""
    cfg = PFConfig(
        num_particles=2,
        use_tempering=True,
        deferred_resample_roughening_scale=0.15,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    captured: dict[str, float | bool] = {}

    def fake_tempered_update(
        *,
        lam_fn,
        z_obs,
        observation_count_variance=0.0,
        disable_regularize_on_resample=None,
        roughening_scale_on_resample=1.0,
    ):
        """Capture resampling options passed by the deferred update path."""
        captured["disable_regularize"] = bool(disable_regularize_on_resample)
        captured["roughening_scale"] = float(roughening_scale_on_resample)
        return 1.0, True

    pf._tempered_update = fake_tempered_update
    pf.update_continuous_pair(
        z_obs=1.0,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        defer_resample=True,
    )

    assert captured["disable_regularize"] is False
    assert captured["roughening_scale"] == 0.15


def test_surface_position_prior_initializes_and_roughens_on_surfaces() -> None:
    """Surface source prior should keep PF particles on known source surfaces."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )
    cfg = PFConfig(
        num_particles=4,
        use_gpu=False,
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_grid_repeats=1,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        source_position_prior="surface",
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1

    pf = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=dummy_kernel,
        config=cfg,
        obstacle_grid=grid,
        obstacle_height_m=1.0,
    )
    pf.regularize_continuous(
        sigma_pos=np.array([0.2, 0.2, 0.2]),
        strength_log_sigma=0.0,
    )

    assert pf.continuous_particles
    for particle in pf.continuous_particles:
        for position in particle.state.positions:
            assert is_allowed_source_surface_position(
                position,
                env,
                grid,
                obstacle_height_m=1.0,
            )


def test_student_t_count_likelihood_softens_model_mismatch() -> None:
    """Robust count likelihood should not over-trust a simplified transport kernel."""
    z_obs = np.array([100.0], dtype=float)
    lambda_good = np.array([100.0], dtype=float)
    lambda_mismatch = np.array([50.0], dtype=float)

    poisson_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="poisson",
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="poisson",
    )
    robust_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_rel_sigma=0.4,
        spectrum_count_rel_sigma=0.2,
        spectrum_count_abs_sigma=5.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_rel_sigma=0.4,
        spectrum_count_rel_sigma=0.2,
        spectrum_count_abs_sigma=5.0,
        student_t_df=5.0,
    )

    assert robust_gap > 0.0
    assert robust_gap < poisson_gap


def test_observation_count_variance_softens_spectrum_unfolding_update() -> None:
    """Spectrum decomposition variance should reduce overconfident count updates."""
    z_obs = np.array([100.0], dtype=float)
    lambda_good = np.array([100.0], dtype=float)
    lambda_mismatch = np.array([60.0], dtype=float)

    certain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        observation_count_variance=0.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        observation_count_variance=0.0,
        student_t_df=5.0,
    )
    uncertain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        observation_count_variance=400.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        observation_count_variance=400.0,
        student_t_df=5.0,
    )

    assert uncertain_gap > 0.0
    assert uncertain_gap < certain_gap


def test_matrix_count_likelihood_normal_alias_matches_scalar_gaussian() -> None:
    """The batched NumPy likelihood path should normalize model aliases."""
    cfg = PFConfig(
        num_particles=1,
        count_likelihood_model="normal",
        transport_model_rel_sigma=0.15,
        spectrum_count_abs_sigma=2.0,
        count_likelihood_df=3.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    z_obs = np.array([42.0, 18.0], dtype=float)
    lambda_kp = np.array(
        [
            [39.0, 23.0],
            [21.0, 12.0],
        ],
        dtype=float,
    )
    obs_var = np.array([4.0, 9.0], dtype=float)

    actual = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=obs_var,
    )
    expected = np.array(
        [
            count_log_likelihood(
                z_obs,
                lambda_kp[:, idx],
                model="normal",
                transport_model_rel_sigma=0.15,
                spectrum_count_abs_sigma=2.0,
                observation_count_variance=obs_var,
                student_t_df=3.0,
            )
            for idx in range(lambda_kp.shape[1])
        ],
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_matrix_count_likelihood_rejects_unknown_model() -> None:
    """The batched likelihood path should not silently reinterpret bad models."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="not_a_model")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)

    with pytest.raises(ValueError, match="Unknown count likelihood model"):
        filt._count_log_likelihood_matrix_np(
            np.array([1.0], dtype=float),
            np.array([[1.0]], dtype=float),
        )


def test_matrix_count_likelihood_scalar_variance_broadcasts() -> None:
    """Scalar unfolding variance should apply to every batched measurement."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="gaussian")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    z_obs = np.array([12.0, 18.0, 25.0], dtype=float)
    lambda_kp = np.array(
        [
            [11.0, 14.0],
            [17.0, 20.0],
            [23.0, 26.0],
        ],
        dtype=float,
    )

    scalar = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=4.0,
    )
    repeated = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=np.full(z_obs.shape, 4.0, dtype=float),
    )

    np.testing.assert_allclose(scalar, repeated, rtol=1.0e-12, atol=1.0e-12)


def test_matrix_count_likelihood_rejects_mismatched_variance() -> None:
    """Batched likelihoods should reject ambiguous observation variance shapes."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="gaussian")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)

    with pytest.raises(ValueError, match="observation_count_variance"):
        filt._count_log_likelihood_matrix_np(
            np.array([12.0, 18.0, 25.0], dtype=float),
            np.array([[11.0], [17.0], [23.0]], dtype=float),
            observation_count_variance=np.array([4.0, 9.0], dtype=float),
        )


def test_count_likelihood_aliases_are_consistent_across_gpu_increment() -> None:
    """Scalar, matrix, and torch increments should agree on model aliases."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        count_likelihood_model="normal",
        transport_model_rel_sigma=0.2,
        spectrum_count_abs_sigma=1.5,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    lam = torch.as_tensor([37.0, 41.0], dtype=torch.float64)

    actual = filt._log_likelihood_increment_gpu(
        lam,
        z_obs=39.0,
        observation_count_variance=4.0,
    ).detach().cpu().numpy()
    expected = np.array(
        [
            count_log_likelihood(
                np.array([39.0], dtype=float),
                np.array([value], dtype=float),
                model="gaussian",
                transport_model_rel_sigma=0.2,
                spectrum_count_abs_sigma=1.5,
                observation_count_variance=4.0,
            )
            for value in (37.0, 41.0)
        ],
        dtype=float,
    )

    assert normalize_count_likelihood_model("") == "poisson"
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_low_count_variance_floor_decays_for_informative_counts() -> None:
    """Low-count uncertainty should protect weak spectra without weakening high counts."""
    z_obs = np.array([5.0, 5000.0], dtype=float)
    lambda_obs = np.array([4.0, 5100.0], dtype=float)

    base_variance = count_likelihood_variance(z_obs, lambda_obs)
    robust_variance = count_likelihood_variance(
        z_obs,
        lambda_obs,
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
    )

    assert robust_variance[0] - base_variance[0] > 300.0
    assert robust_variance[1] - base_variance[1] < 110.0


def test_transport_absolute_floor_softens_low_count_model_mismatch() -> None:
    """Absolute transport mismatch should prevent low counts from over-pruning particles."""
    z_obs = np.array([8.0], dtype=float)
    lambda_good = np.array([8.0], dtype=float)
    lambda_mismatch = np.array([20.0], dtype=float)

    certain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_abs_sigma=0.0,
        student_t_df=3.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_abs_sigma=0.0,
        student_t_df=3.0,
    )
    uncertain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
        student_t_df=3.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
        student_t_df=3.0,
    )

    assert uncertain_gap > 0.0
    assert uncertain_gap < certain_gap


def test_resampling_increases_neff() -> None:
    """Highly skewed weights should be flattened after resampling."""
    cfg = PFConfig(num_particles=3)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    pf.continuous_particles = [
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.99)),
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.005)),
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.005)),
    ]
    before = 1.0 / np.sum(pf.continuous_weights**2)
    pf._maybe_resample_continuous()
    after = 1.0 / np.sum(pf.continuous_weights**2)
    assert after > before
    assert np.allclose(pf.continuous_weights, np.ones(3) / 3, rtol=1e-3)
