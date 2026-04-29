"""Regression coverage for isotope locking and missing-measurement handling."""

from __future__ import annotations

import numpy as np
import pytest

from pf.estimator import MeasurementRecord, RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.mixing import prune_spurious_sources_continuous
from realtime_demo import (
    ADAPTIVE_STEP_ID_STRIDE,
    _acquire_spectrum_observation,
    run_live_pf,
)
from sim import SimulationCommand, SimulationObservation
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer


def test_demo_spectrum_counts_keep_all_isotopes(monkeypatch: pytest.MonkeyPatch) -> None:
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
        z_variance_k: dict[str, float] | None = None,
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
                z_variance_k=z_variance_k,
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
        self.last_count_variances = {iso: 2.0 for iso in ANALYSIS_ISOTOPES}
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
        planning_isotope_args.append(planning_isotopes)
        size = len(rot_mats)
        return np.zeros((size, size), dtype=float)

    def _fake_frame(*args: object, **kwargs: object) -> dict[str, object]:
        """Return an empty frame placeholder."""
        return {}

    def _fake_candidate_poses(*args: object, **kwargs: object) -> np.ndarray:
        """Return two deterministic candidate poses."""
        return np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]], dtype=float)

    def _fake_next_pose(*args: object, **kwargs: object) -> int:
        """Select the candidate that requires travel from the initial pose."""
        return 1

    def _fake_gpu_enabled(self: RotatingShieldPFEstimator) -> bool:
        """Pretend GPU is disabled to avoid CUDA checks in tests."""
        return False

    def _forbidden_restrict(
        self: RotatingShieldPFEstimator,
        active_isotopes: list[str],
    ) -> None:
        """Fail if runtime detection tries to remove isotope filters."""
        raise AssertionError("Detection must not restrict runtime isotopes.")

    planning_isotope_args: list[list[str] | None] = []
    monkeypatch.setattr(realtime_demo, "RealTimePFVisualizer", _DummyViz)
    monkeypatch.setattr(realtime_demo, "build_frame_from_pf", _fake_frame)
    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", _fake_ig_grid)
    monkeypatch.setattr(
        realtime_demo,
        "DETECT_CONSECUTIVE_BY_ISOTOPE",
        {"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
    )
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
    monkeypatch.setattr(RotatingShieldPFEstimator, "restrict_isotopes", _forbidden_restrict)

    estimator = run_live_pf(
        live=False,
        max_steps=None,
        max_poses=2,
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
        nominal_motion_speed_m_s=1.0,
        rotation_overhead_s=2.0,
    )
    assert estimator is not None
    assert len(estimator.measurements) >= 2
    assert len(estimator.poses) >= 2
    metrics = estimator.mission_metrics
    assert metrics["total_measurements"] >= 2
    assert metrics["total_motion_distance_m"] == pytest.approx(np.sqrt(2.0))
    assert metrics["total_travel_time_s"] == pytest.approx(np.sqrt(2.0))
    assert metrics["total_shield_actuation_time_s"] == pytest.approx(
        metrics["total_measurements"] * 2.0
    )
    assert metrics["total_mission_time_s"] == pytest.approx(
        metrics["total_live_time_s"]
        + metrics["total_travel_time_s"]
        + metrics["total_shield_actuation_time_s"]
    )
    assert metrics["estimated_end_to_end_time_s"] == pytest.approx(
        metrics["total_mission_time_s"]
    )
    assert metrics["num_motion_segments"] == 1
    assert len(metrics["path_segments"]) == 1
    assert metrics["path_segments"][0]["travel_time_s"] == pytest.approx(np.sqrt(2.0))
    assert metrics["mean_orientation_selection_time_s"] >= 0.0
    assert metrics["mean_pf_update_time_s"] >= 0.0
    for rec in estimator.measurements:
        for iso in ANALYSIS_ISOTOPES:
            assert iso in rec.z_k
            assert rec.z_variance_k is not None
            assert rec.z_variance_k[iso] == pytest.approx(2.0)
    assert planning_isotope_args
    assert all(value is None for value in planning_isotope_args)
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


def test_adaptive_dwell_chunks_stop_at_ready_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive dwell should stop after accumulated isotope counts are usable."""
    decomposer = SpectralDecomposer()
    commands: list[SimulationCommand] = []

    class _FakeRuntime:
        """Return deterministic spectra proportional to requested dwell time."""

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Record the command and return one non-zero spectrum bin."""
            commands.append(command)
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            spectrum = np.zeros_like(energy, dtype=float)
            spectrum[0] = float(command.dwell_time_s) * 60.0
            spectrum_variance = np.full_like(energy, float(command.dwell_time_s) * 25.0)
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=spectrum.tolist(),
                energy_bin_edges_keV=np.concatenate(
                    [energy, [energy[-1] + step]]
                ).tolist(),
                metadata={
                    "backend": "fake",
                    "weighted_transport": True,
                    "spectrum_count_variance": spectrum_variance.tolist(),
                },
            )

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a Cs-137 count equal to total accumulated spectrum counts."""
        count = float(np.sum(spectrum))
        detected = {"Cs-137"} if count > 0.0 else set()
        return {"Cs-137": count}, detected

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    def _fake_variance_floor(
        self: SpectralDecomposer,
        spectrum_variance: np.ndarray,
        *,
        isotopes: list[str],
    ) -> dict[str, float]:
        """Return a deterministic weighted-MC variance floor for the test."""
        assert float(np.sum(spectrum_variance)) > 0.0
        return {"Cs-137": 1000.0}

    monkeypatch.setattr(
        SpectralDecomposer,
        "estimate_count_variances_from_spectrum_variance",
        _fake_variance_floor,
    )
    observation, actual_live, counts, variances, detected, reason, chunks = (
        _acquire_spectrum_observation(
            simulation_runtime=_FakeRuntime(),
            decomposer=decomposer,
            step_id=7,
            pose_xyz=np.array([1.0, 2.0, 0.5], dtype=float),
            fe_idx=3,
            pb_idx=4,
            live_time_s=30.0,
            travel_time_s=5.0,
            shield_actuation_time_s=2.0,
            adaptive_dwell=True,
            adaptive_dwell_chunk_s=2.0,
            adaptive_min_dwell_s=2.0,
            adaptive_ready_min_counts=200.0,
            adaptive_ready_min_isotopes=1,
            spectrum_count_method="photopeak_nnls",
            detect_threshold_abs=0.0,
            detect_threshold_rel=0.0,
            detect_threshold_rel_by_isotope={},
            min_peaks_by_isotope=None,
        )
    )

    assert actual_live == pytest.approx(4.0)
    assert counts["Cs-137"] == pytest.approx(240.0)
    assert variances["Cs-137"] == pytest.approx(1000.0)
    assert detected == {"Cs-137"}
    assert reason == "detected_isotope_counts_ready"
    assert chunks == 2
    assert observation.step_id == 7
    assert observation.metadata["adaptive_dwell_chunks"] == 2
    assert "adaptive_dwell_count_variance_by_isotope" in observation.metadata
    assert observation.metadata["spectrum_count_variance_total"] > 0.0
    assert commands[0].step_id == 7 * ADAPTIVE_STEP_ID_STRIDE
    assert commands[1].step_id == 7 * ADAPTIVE_STEP_ID_STRIDE + 1
    assert commands[0].travel_time_s == pytest.approx(5.0)
    assert commands[1].travel_time_s == pytest.approx(0.0)
    assert commands[0].shield_actuation_time_s == pytest.approx(2.0)
    assert commands[1].shield_actuation_time_s == pytest.approx(0.0)
