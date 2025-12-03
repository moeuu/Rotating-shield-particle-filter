"""簡易なPFエンドツーエンド動作を確認するスモークテスト。"""

import numpy as np

from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from measurement.kernels import ShieldParams


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
    z_k = {"Cs-137": 5.0}
    est.update(z_k=z_k, pose_idx=0, orient_idx=0, live_time_s=1.0)
    estimates = est.estimates()
    assert "Cs-137" in estimates
    positions, strengths = estimates["Cs-137"]
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
