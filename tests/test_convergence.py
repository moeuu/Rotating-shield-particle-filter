"""Convergence criteria tests for RotatingShieldPFEstimator (Sec. 3.5–3.6)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import IsotopeState
from pf.particle_filter import IsotopeParticle


def _build_stable_estimator(strength: float = 10.0) -> RotatingShieldPFEstimator:
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        birth_enable=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    # populate history with identical estimates to simulate stabilization
    est.history_estimates = [est.estimates(), est.estimates()]
    return est


def test_should_stop_shield_rotation_when_stable() -> None:
    """Stable posterior with zero IG and low uncertainty should trigger stop."""
    est = _build_stable_estimator()
    assert est.should_stop_shield_rotation(
        pose_idx=0, ig_threshold=1e-6, change_tol=1e-6, uncertainty_tol=1e-6, live_time_s=1.0
    )


def test_should_not_stop_when_uncertain() -> None:
    """High variance across particles keeps exploration active."""
    est = _build_stable_estimator()
    filt = est.filters["Cs-137"]
    # Inject variance in strengths to raise U
    filt.continuous_particles[0].state.strengths = np.array([1.0])
    filt.continuous_particles[1].state.strengths = np.array([10.0])
    est.history_estimates = [est.estimates(), est.estimates()]
    assert not est.should_stop_exploration(
        ig_threshold=1e-6, change_tol=1e-6, uncertainty_tol=1e-3, live_time_s=1.0
    )


def test_convergence_diagnostic_never_skips_pf_measurement(monkeypatch) -> None:
    """A converged diagnostic flag must not bypass a later likelihood update."""
    estimator = _build_stable_estimator()
    filt = estimator.filters["Cs-137"]
    filt.config.converge_enable = True
    filt.is_converged = True
    update_calls: list[float] = []

    def fake_tempered_update(**kwargs: object) -> tuple[float, bool]:
        """Record that the likelihood update path was entered."""
        update_calls.append(float(kwargs["z_obs"]))
        return float(filt.N), False

    def fake_convergence_update(**_kwargs: object) -> None:
        """Avoid changing the injected diagnostic flag during this check."""

    monkeypatch.setattr(filt, "_tempered_update", fake_tempered_update)
    monkeypatch.setattr(filt, "_maybe_update_convergence", fake_convergence_update)

    filt.update_continuous_pair(
        z_obs=7.0,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert update_calls == [7.0]
    assert not hasattr(filt, "updates_skipped")
