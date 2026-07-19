"""Regression tests for simulation fidelity shortcuts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from realtime_demo import run_live_pf
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig
from spectrum.runtime_config import spectrum_config_from_runtime_config
from spectrum.runtime_counts import RuntimeCountExtractor


def test_codex_instructions_define_simulation_fidelity_guardrails() -> None:
    """Codex project instructions should forbid low-fidelity runtime shortcuts."""
    root = Path(__file__).resolve().parents[1]
    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    policy_text = (root / "docs" / "simulation_fidelity_policy.md").read_text(
        encoding="utf-8"
    )
    normalized_agents_text = " ".join(agents_text.split())
    required_phrases = (
        "Simulation fidelity policy",
        "Runtime simulation fidelity must take priority over speed shortcuts.",
        "CUI mode means \"run without the Isaac Sim GUI.\"",
        "surrogate transport",
        "expected-count observations",
        "weighted or capped Geant4 histories",
        "peak-window or full-spectrum-continuum runtime count extraction",
        "Full simulation",
        "uv run python main.py --full-simulation",
    )

    for phrase in required_phrases:
        assert phrase in normalized_agents_text
    assert "Prohibited Runtime Shortcuts" in policy_text
    assert "Allowed Performance Work" in policy_text


def test_codex_instructions_require_parallel_first_heavy_runtime_work() -> None:
    """Codex project instructions should require parallel-first heavy code."""
    root = Path(__file__).resolve().parents[1]
    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    parallel_policy = (root / "docs" / "compute_parallelism_policy.md").read_text(
        encoding="utf-8"
    )
    fidelity_policy = (root / "docs" / "simulation_fidelity_policy.md").read_text(
        encoding="utf-8"
    )
    combined = " ".join((agents_text + "\n" + parallel_policy).split())
    required_phrases = (
        "Compute parallelism policy",
        "batched, GPU, Geant4-threaded, or process-parallel",
        "Do not add scalar Python runtime loops over particles",
        "Parallelization must preserve the same physics",
        "serial-vs-parallel equivalence test",
    )

    for phrase in required_phrases:
        assert phrase in combined
    assert "docs/compute_parallelism_policy.md" in fidelity_policy


def test_demo_entrypoints_do_not_expose_expected_count_bypasses() -> None:
    """Demo entrypoints should feed PF updates from spectra, not expected counts."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = (
        root / "main.py",
        root / "src" / "realtime_demo.py",
        root / "src" / "baselines" / "legacy_no_shield" / "cli.py",
        root / "src" / "baselines" / "legacy_no_shield" / "realtime_demo.py",
    )
    forbidden_tokens = (
        "--count",
        "count_mode",
        "_detect_isotopes_from_expected",
        "def _expected_counts(",
    )

    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_runtime_count_extraction_defaults_to_response_poisson() -> None:
    """Runtime count extraction should standardize on response_poisson."""
    assert RuntimeCountExtractor.STANDARD_METHOD == "response_poisson"
    assert RuntimeCountExtractor.validate_count_method("response_poisson") == (
        "response_poisson"
    )


def test_runtime_count_extraction_preserves_isotope_count_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime count extraction should keep response-Poisson count covariance."""
    decomposer = SpectralDecomposer()
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return correlated isotope counts from a response-Poisson fit."""
        self.last_count_variances = {"Cs-137": 25.0, "Co-60": 100.0}
        self.last_count_covariance = {
            "Cs-137": {"Cs-137": 25.0, "Co-60": -25.0},
            "Co-60": {"Cs-137": -25.0, "Co-60": 100.0},
        }
        self.last_response_poisson_diagnostics = {"status": "ok"}
        return {"Cs-137": 50.0, "Co-60": 20.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=10.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.covariance is not None
    assert result.covariance["Cs-137"]["Cs-137"] == pytest.approx(
        result.variances["Cs-137"]
    )
    assert result.covariance["Co-60"]["Co-60"] == pytest.approx(
        result.variances["Co-60"]
    )
    corr = result.covariance["Cs-137"]["Co-60"] / np.sqrt(
        result.variances["Cs-137"] * result.variances["Co-60"]
    )
    assert corr == pytest.approx(-0.5)


def test_runtime_count_extraction_symmetrizes_reciprocal_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime covariance must remain exactly symmetric after numerical drift."""
    decomposer = SpectralDecomposer()
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a covariance with realistic reciprocal roundoff drift."""
        self.last_count_variances = {"Cs-137": 25.0, "Co-60": 100.0}
        self.last_count_covariance = {
            "Cs-137": {"Cs-137": 25.0, "Co-60": -25.0},
            "Co-60": {"Cs-137": -25.0001, "Co-60": 100.0},
        }
        self.last_response_poisson_diagnostics = {"status": "ok"}
        return {"Cs-137": 50.0, "Co-60": 20.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=10.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.covariance is not None
    forward = result.covariance["Cs-137"]["Co-60"]
    reverse = result.covariance["Co-60"]["Cs-137"]
    assert forward == reverse
    assert forward == pytest.approx(-25.00005)


def test_runtime_covariance_does_not_rescale_formal_correlation_to_added_floors(
) -> None:
    """Independent runtime floors must not inflate formal cross-covariance."""
    decomposer = SpectralDecomposer()
    decomposer.last_count_covariance = {
        "Cs-137": {"Cs-137": 25.0, "Co-60": -25.0},
        "Co-60": {"Cs-137": -25.0, "Co-60": 100.0},
    }
    covariance = RuntimeCountExtractor(
        decomposer
    )._count_covariance_with_runtime_variances(
        {"Cs-137": 100.0, "Co-60": 100.0},
        {"Cs-137": 10000.0, "Co-60": 40000.0},
    )

    assert covariance["Cs-137"]["Cs-137"] == pytest.approx(10000.0)
    assert covariance["Co-60"]["Co-60"] == pytest.approx(40000.0)
    assert covariance["Cs-137"]["Co-60"] == pytest.approx(-25.0)
    assert covariance["Co-60"]["Cs-137"] == pytest.approx(-25.0)


def test_runtime_covariance_ceiling_scales_complete_formal_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capping formal diagonals must preserve their fitted correlation."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_count_variance_ceiling_enable=True,
            response_poisson_count_variance_max_rel_sigma=0.15,
            response_poisson_count_variance_max_abs_sigma=0.0,
        )
    )
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a high-variance correlated formal regression result."""
        del spectrum, live_time_s, kwargs
        self.last_count_variances = {"Cs-137": 1.0e9, "Co-60": 4.0e9}
        self.last_count_covariance = {
            "Cs-137": {"Cs-137": 1.0e9, "Co-60": -1.0e9},
            "Co-60": {"Cs-137": -1.0e9, "Co-60": 4.0e9},
        }
        self.last_response_poisson_diagnostics = {"status": "ok"}
        return {"Cs-137": 1000.0, "Co-60": 2000.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )
    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=10.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.covariance is not None
    row_var = result.covariance["Cs-137"]["Cs-137"]
    col_var = result.covariance["Co-60"]["Co-60"]
    correlation = result.covariance["Cs-137"]["Co-60"] / np.sqrt(
        row_var * col_var
    )
    assert row_var == pytest.approx((0.15 * 1000.0) ** 2)
    assert col_var == pytest.approx((0.15 * 2000.0) ** 2)
    assert correlation == pytest.approx(-0.5)


def test_global_response_chi2_does_not_inflate_every_isotope_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-spectrum fit residuals are not isotope-count error calibration."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_diagnostic_variance_enable=True,
            response_poisson_global_diagnostic_variance_enable=False,
        )
    )
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return stable isotope counts with only a poor global fit statistic."""
        self.last_count_variances = {"Cs-137": 25.0, "Co-60": 36.0}
        self.last_count_covariance = {}
        self.last_response_poisson_diagnostics = {
            "reduced_chi2": 500.0,
            "design_condition_number": 1.0,
            "fisher_condition_number": 1.0,
        }
        return {"Cs-137": 1000.0, "Co-60": 800.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.variances == pytest.approx({"Cs-137": 25.0, "Co-60": 36.0})
    components = decomposer.last_response_poisson_diagnostics[
        "runtime_variance_components"
    ]
    assert components["Cs-137"]["final_variance"] == pytest.approx(25.0)
    assert components["Co-60"]["final_variance"] == pytest.approx(36.0)


def test_global_response_chi2_variance_remains_explicitly_opt_in() -> None:
    """Legacy global diagnostic inflation should require an explicit setting."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_global_diagnostic_variance_enable=True,
            response_poisson_diagnostic_reduced_chi2_threshold=2.0,
            response_poisson_diagnostic_reduced_chi2_scale=0.5,
        )
    )
    extractor = RuntimeCountExtractor(decomposer)

    sigma = extractor._diagnostic_relative_sigma({"reduced_chi2": 10.0})

    assert sigma == pytest.approx(1.0)


def test_response_diagnostics_inflate_runtime_count_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreliable response-regression diagnostics should soften PF observations."""
    config = SpectrumConfig(
        response_poisson_diagnostic_reduced_chi2_threshold=2.0,
        response_poisson_diagnostic_reduced_chi2_scale=0.5,
        response_poisson_crosstalk_corr_threshold=0.85,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return counts with diagnostics resembling crosstalk-prone unfolding."""
        self.last_count_variances = {"Cs-137": 1.0, "Co-60": 1.0}
        self.last_response_poisson_diagnostics = {
            "reduced_chi2": 10.0,
            "design_condition_number": 1.0,
            "fisher_condition_number": 1.0,
            "coefficient_correlation_by_isotope": {
                "Cs-137": 0.95,
                "Co-60": 0.1,
            },
            "low_snr_photopeak_suppression": {
                "Co-60": {"reason": "retained_poisson"},
            },
        }
        return {"Cs-137": 100.0, "Co-60": 0.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.counts["Cs-137"] == pytest.approx(100.0)
    assert result.variances["Cs-137"] > 1000.0
    assert result.variances["Co-60"] > 1.0
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert "runtime_diagnostic_variance_floor" in diagnostics
    assert "transport_detected_counts_Cs-137" not in diagnostics


def test_crosstalk_guard_disagreement_inflates_runtime_count_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Photopeak/response disagreement should become PF observation covariance."""
    config = SpectrumConfig(
        response_poisson_crosstalk_min_rel_sigma=0.25,
        response_poisson_crosstalk_variance_scale=1.0,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return counts with an unresolved crosstalk guard diagnostic."""
        self.last_count_variances = {"Cs-137": 1.0, "Co-60": 1.0}
        self.last_response_poisson_diagnostics = {
            "reduced_chi2": 1.0,
            "design_condition_number": 1.0,
            "fisher_condition_number": 1.0,
            "crosstalk_count_guard": {
                "Cs-137": {
                    "reason": "combined_crosstalk_photopeak_log_blend",
                    "adjust_count": False,
                    "poisson_count": 100.0,
                    "photopeak_count": 25.0,
                    "disagreement_fraction": 0.75,
                    "combined_crosstalk_weight": 0.8,
                    "blend_weight": 0.0,
                }
            },
        }
        return {"Cs-137": 100.0, "Co-60": 1000.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.counts["Cs-137"] == pytest.approx(100.0)
    assert result.variances["Cs-137"] > 1000.0
    assert result.variances["Co-60"] == pytest.approx(1.0)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["runtime_diagnostic_variance_floor"]["Cs-137"] > 1000.0


def test_diagnostic_variance_floor_is_recorded_after_formal_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard disagreement should be a stage after the formal covariance cap."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=True,
        response_poisson_count_variance_ceiling_enable=True,
        response_poisson_count_variance_max_rel_sigma=0.15,
        response_poisson_count_variance_max_abs_sigma=40.0,
        response_poisson_count_variance_preserve_diagnostic_floors=False,
        response_poisson_count_variance_preserve_guard_floors=False,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a tail disagreement with a large guard variance."""
        self.last_count_variances = {"Cs-137": 1.0e9}
        self.last_response_poisson_diagnostics = {
            "reduced_chi2": 1.0,
            "design_condition_number": 1.0,
            "fisher_condition_number": 1.0,
            "crosstalk_count_guard": {
                "Cs-137": {
                    "reason": "combined_crosstalk_photopeak_log_blend",
                    "adjust_count": False,
                    "poisson_count": 100000.0,
                    "photopeak_count": 1000.0,
                    "disagreement_fraction": 0.99,
                    "combined_crosstalk_weight": 1.0,
                    "guarded_variance": 1.0e8,
                }
            },
        }
        return {"Cs-137": 100000.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    naive_ceiling = (0.15 * 100000.0) ** 2
    diagnostic_floor = (0.99 * 100000.0) ** 2
    assert result.variances["Cs-137"] > naive_ceiling
    assert result.variances["Cs-137"] == pytest.approx(diagnostic_floor)
    diagnostics = decomposer.last_response_poisson_diagnostics
    components = diagnostics["runtime_variance_components"]["Cs-137"]
    assert components["formal_after_ceiling_variance"] <= naive_ceiling
    assert components["formal_after_ceiling_variance"] == pytest.approx(
        naive_ceiling
    )
    assert components["isotope_diagnostic_variance"] == pytest.approx(
        diagnostic_floor
    )
    assert components["isotope_diagnostic_increment"] == pytest.approx(
        diagnostic_floor - components["transport_statistical_variance"]
    )


def test_low_snr_photo_count_disagreement_survives_runtime_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-SNR suppression should not become an overconfident PF observation."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=False,
        response_poisson_count_variance_ceiling_enable=True,
        response_poisson_count_variance_max_rel_sigma=0.15,
        response_poisson_count_variance_max_abs_sigma=40.0,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a suppressed low-SNR count with a large hidden disagreement."""
        self.last_count_variances = {"Co-60": 3.0e7}
        self.last_response_poisson_diagnostics = {
            "low_snr_photopeak_suppression": {
                "Co-60": {
                    "suppressed": True,
                    "reason": "missing_expected_photopeaks",
                    "poisson_count": 6500.0,
                    "photo_count": 1100.0,
                    "photo_snr": 3.0,
                }
            }
        }
        return {"Co-60": 1100.0}, {"Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    naive_ceiling = (0.15 * 1100.0) ** 2
    disagreement_floor = (6500.0 - 1100.0) ** 2
    assert result.variances["Co-60"] > naive_ceiling
    assert result.variances["Co-60"] >= disagreement_floor
    assert result.variances["Co-60"] == pytest.approx(3.0e7)


def test_low_snr_threshold_variance_survives_runtime_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-SNR zero estimates should retain threshold uncertainty in PF."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=False,
        response_poisson_count_variance_ceiling_enable=True,
        response_poisson_count_variance_max_rel_sigma=0.15,
        response_poisson_count_variance_max_abs_sigma=40.0,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a zero low-SNR estimate with a threshold variance."""
        self.last_count_variances = {"Cs-137": 50000.0}
        self.last_response_poisson_diagnostics = {
            "low_snr_photopeak_suppression": {
                "Cs-137": {
                    "suppressed": False,
                    "reason": "zero_poisson_photopeak_fused",
                    "poisson_count": 0.0,
                    "photo_count": 0.0,
                    "photo_snr": 0.0,
                }
            }
        }
        return {"Cs-137": 0.0}, set()

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result.variances["Cs-137"] == pytest.approx(50000.0)


def test_response_truth_calibration_is_not_runtime_count_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime count extraction should ignore truth-fit calibration payloads."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=False,
    )
    setattr(
        config,
        "response_poisson_truth_calibration",
        {
            "enabled": True,
            "feature_names": ["log_count"],
            "neighbor_count": 1.0,
            "distance_epsilon": 0.05,
            "fallback_scale_by_isotope": {"Cs-137": 1.0},
            "groups": {
                "Cs-137|10": {
                    "n": 1.0,
                    "feature_mean": [float(np.log(100.0))],
                    "feature_std": [1.0],
                    "features": [[0.0]],
                    "log_scales": [float(np.log(1.25))],
                }
            },
        },
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return stable counts with intentionally small base variances."""
        self.last_count_variances = {"Cs-137": 16.0}
        self.last_response_poisson_diagnostics = {
            "coefficients": {"Cs-137": 100.0},
            "photopeak_counts": {"Cs-137": 100.0},
            "reduced_chi2": 1.0,
        }
        return {"Cs-137": 100.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        transport_metadata={
            "fe_orientation_index": 1,
            "pb_orientation_index": 2,
            "shield_num_orientations": 8,
        },
    )

    assert result.counts["Cs-137"] == pytest.approx(100.0)
    assert result.variances["Cs-137"] == pytest.approx(16.0)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert "runtime_response_truth_calibration" not in diagnostics


def test_runtime_config_ignores_response_truth_calibration_keys() -> None:
    """Runtime config conversion should not expose truth-fit count calibration."""
    config = spectrum_config_from_runtime_config(
        {
            "response_poisson_truth_calibration": {"enabled": True},
            "response_poisson_truth_calibration_path": "unused.json",
        }
    )

    assert not hasattr(config, "response_poisson_truth_calibration")
    assert not hasattr(config, "response_poisson_truth_calibration_path")


def test_shield_systematics_inflate_runtime_count_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shielded spectra should carry a configurable response-model variance floor."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=False,
        response_poisson_shield_systematic_variance_enable=True,
        response_poisson_shield_systematic_rel_sigma=0.2,
        response_poisson_shield_systematic_anchor_pair_ids=(0,),
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return stable counts with intentionally small base variances."""
        self.last_count_variances = {"Cs-137": 1.0}
        self.last_response_poisson_diagnostics = {"status": "ok"}
        return {"Cs-137": 100.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        transport_metadata={
            "fe_orientation_index": 1,
            "pb_orientation_index": 2,
            "shield_num_orientations": 8,
            "shield_thickness_scale": 1.0,
        },
    )

    assert result.variances["Cs-137"] == pytest.approx(400.0)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["runtime_shield_systematic_pair_id"] == 10
    assert diagnostics["runtime_shield_systematic_anchor_pair"] is False


def test_shield_systematics_skip_zero_thickness_no_shield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-shield baselines should not receive shield-model mismatch inflation."""
    config = SpectrumConfig(
        response_poisson_diagnostic_variance_enable=False,
        response_poisson_shield_systematic_variance_enable=True,
        response_poisson_shield_systematic_rel_sigma=0.2,
    )
    decomposer = SpectralDecomposer(config)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return stable counts with intentionally small base variances."""
        self.last_count_variances = {"Cs-137": 9.0}
        self.last_response_poisson_diagnostics = {"status": "ok"}
        return {"Cs-137": 100.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = RuntimeCountExtractor(decomposer).extract(
        spectrum,
        live_time_s=30.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        transport_metadata={
            "shield_pair_id": 10,
            "shield_thickness_scale": 0.0,
        },
    )

    assert result.variances["Cs-137"] == pytest.approx(9.0)
    assert "runtime_shield_systematic_variance_floor" not in (
        decomposer.last_response_poisson_diagnostics
    )


def test_runtime_rejects_peak_window_count_method(tmp_path: Path) -> None:
    """Runtime simulations should reject lower-fidelity peak-window counting."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "peak_window"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_runtime_rejects_photopeak_nnls_count_method(tmp_path: Path) -> None:
    """Runtime simulations should keep photopeak NNLS out of PF ingestion."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "photopeak_nnls"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_runtime_rejects_full_spectrum_response_count_method(tmp_path: Path) -> None:
    """Runtime simulations should reject continuum-fitting count extraction."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "response_matrix"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_wall_absorber_is_explicit_native_transport_mode() -> None:
    """Wall absorption should be an explicit mode, not a hidden shortcut."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(
        encoding="utf-8"
    )

    assert 'transport_mode) != "absorber"' in source
    assert "TransportSteppingAction" in source
    assert "SetTrackStatus(fStopAndKill)" in source
    assert 'result.metadata["absorbing_volume_count"]' in source
