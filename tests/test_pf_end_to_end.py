"""簡易なPFエンドツーエンド動作を確認するスモークテスト。"""

import numpy as np

from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from measurement.kernels import ShieldParams
from spectrum.pipeline import SpectralDecomposer
from measurement.model import EnvironmentConfig, PointSource


def test_pf_estimator_runs_one_step():
    """単一測定でPFが更新できることを確認する。"""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=RotatingShieldPFConfig(num_particles=50, max_sources=1),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    # PF observation should come from spectrum unfolding (Sec. 2.5.7)
    decomposer = SpectralDecomposer()
    env = EnvironmentConfig(detector_position=(0.5, 0.0, 0.0))
    sources = [PointSource("Cs-137", position=(0.0, 0.0, 0.0), intensity_cps_1m=20.0)]
    spectrum, _ = decomposer.simulate_spectrum(sources=sources, environment=env, acquisition_time=1.0, rng=np.random.default_rng(0))
    z_k = decomposer.isotope_counts(spectrum)
    est.update(z_k=z_k, pose_idx=0, orient_idx=0, live_time_s=1.0)
    estimates = est.estimates()
    assert "Cs-137" in estimates
    positions, strengths = estimates["Cs-137"]
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
