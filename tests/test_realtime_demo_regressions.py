"""Regression coverage for isotope locking and missing-measurement handling."""

from __future__ import annotations

import numpy as np
import pytest

from pf.estimator import MeasurementRecord, RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.mixing import prune_spurious_sources_continuous
from realtime_demo import run_live_pf
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer


def test_demo_expected_counts_keeps_all_isotopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure demo measurements keep all isotope keys after detection locking."""
    import realtime_demo

    class _DummyViz:
        """Minimal visualizer stub for fast regression testing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the stub visualizer."""
            return None

        def update(self, frame: object) -> None:
            """Ignore frame updates in tests."""
            return None

        def save_final(self, path: str) -> None:
            """Skip saving final snapshots in tests."""
            return None

        def save_estimates_only(self, path: str) -> None:
            """Skip saving estimate snapshots in tests."""
            return None

    def _fake_update_pair(
        self: RotatingShieldPFEstimator,
        z_k: dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> None:
        """Append a lightweight measurement record without GPU updates."""
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
            )
        )

    def _fake_estimates(self: RotatingShieldPFEstimator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([1.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in ANALYSIS_ISOTOPES}

    def _fake_sim(self: SpectralDecomposer, *args: object, **kwargs: object) -> tuple[np.ndarray, None]:
        """Return a zero spectrum to avoid heavy simulation work."""
        return np.zeros_like(self.energy_axis, dtype=float), None

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic counts and a stable detection set."""
        counts = {iso: 10.0 for iso in ANALYSIS_ISOTOPES}
        return counts, {"Cs-137"}

    def _fake_ig_grid(
        estimator: RotatingShieldPFEstimator,
        rot_mats: list[np.ndarray],
        *,
        pose_idx: int,
        live_time_s: float,
        planning_isotopes: list[str] | None = None,
    ) -> np.ndarray:
        """Return a zero IG grid to bypass heavy IG evaluation."""
        size = len(rot_mats)
        return np.zeros((size, size), dtype=float)

    def _fake_expected_counts(*args: object, **kwargs: object) -> dict[str, float]:
        """Return deterministic expected counts for all analysis isotopes."""
        return {iso: 5.0 for iso in ANALYSIS_ISOTOPES}

    def _fake_frame(*args: object, **kwargs: object) -> dict[str, object]:
        """Return an empty frame placeholder."""
        return {}

    def _fake_best_orientation(*args: object, **kwargs: object) -> tuple[int, float]:
        """Return a fixed orientation with positive IG."""
        return 0, 1.0

    def _fake_candidate_poses(*args: object, **kwargs: object) -> np.ndarray:
        """Return two deterministic candidate poses."""
        return np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]], dtype=float)

    def _fake_next_pose(*args: object, **kwargs: object) -> int:
        """Select the first candidate pose."""
        return 0

    def _fake_gpu_enabled(self: RotatingShieldPFEstimator) -> bool:
        """Pretend GPU is disabled to avoid CUDA checks in tests."""
        return False

    monkeypatch.setattr(realtime_demo, "RealTimePFVisualizer", _DummyViz)
    monkeypatch.setattr(realtime_demo, "build_frame_from_pf", _fake_frame)
    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", _fake_ig_grid)
    monkeypatch.setattr(realtime_demo, "_expected_counts", _fake_expected_counts)
    monkeypatch.setattr(realtime_demo, "select_best_orientation", _fake_best_orientation)
    monkeypatch.setattr(
        realtime_demo,
        "generate_candidate_poses",
        _fake_candidate_poses,
    )
    monkeypatch.setattr(realtime_demo, "select_next_pose_from_candidates", _fake_next_pose)
    monkeypatch.setattr(SpectralDecomposer, "simulate_spectrum", _fake_sim)
    monkeypatch.setattr(SpectralDecomposer, "isotope_counts_with_detection", _fake_counts)
    monkeypatch.setattr(RotatingShieldPFEstimator, "update_pair", _fake_update_pair)
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _fake_estimates)
    monkeypatch.setattr(RotatingShieldPFEstimator, "_gpu_enabled", _fake_gpu_enabled)

    estimator = run_live_pf(
        live=False,
        max_steps=None,
        max_poses=2,
        count_mode="expected",
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_consecutive=1,
        detect_min_steps=1,
        min_peaks_by_isotope={"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
        ig_threshold_mode="absolute",
        ig_threshold_min=0.0,
        obstacle_layout_path=None,
        pf_config_overrides={"orientation_k": 1},
        save_outputs=False,
        return_state=True,
    )
    assert estimator is not None
    assert len(estimator.measurements) >= 2
    assert len(estimator.poses) >= 2
    for rec in estimator.measurements:
        for iso in ANALYSIS_ISOTOPES:
            assert iso in rec.z_k
    estimates = estimator.estimates()
    for iso in ANALYSIS_ISOTOPES:
        positions, strengths = estimates.get(iso, (np.zeros((0, 3)), np.zeros(0)))
        assert positions.size > 0
        assert strengths.size > 0


def test_prune_missing_isotope_does_not_zero_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing isotope keys should not be treated as zero-count measurements."""
    isotopes = ["Cs-137", "Co-60"]
    pf_conf = RotatingShieldPFConfig(num_particles=4, min_particles=4, max_particles=4, use_gpu=False)
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=np.zeros((1, 3), dtype=float),
        shield_normals=None,
        mu_by_isotope=None,
        pf_config=pf_conf,
    )
    estimator.poses = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    estimator.measurements = [
        MeasurementRecord(
            z_k={"Cs-137": 10.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
        ),
        MeasurementRecord(
            z_k={"Cs-137": 12.0, "Co-60": 5.0},
            pose_idx=1,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
        ),
    ]

    def _fake_estimates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([100.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in isotopes}

    monkeypatch.setattr(estimator, "estimates", _fake_estimates)
    keep_masks = prune_spurious_sources_continuous(
        estimator,
        method="deltaLL",
        params={"deltaLL_min": 1e9},
        min_support=2,
    )
    assert keep_masks["Co-60"].size == 1
    assert keep_masks["Co-60"].all()
