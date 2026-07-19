"""スペクトル分解パイプラインの基本挙動を検証するテスト。"""

import numpy as np
import pytest

from measurement.model import EnvironmentConfig, PointSource
from spectrum import pipeline
from spectrum.pipeline import IsotopeCountEstimate, SpectralDecomposer, SpectrumConfig
from spectrum.runtime_counts import RuntimeCountExtractor


def _response_integral(decomposer: SpectralDecomposer, isotope: str) -> float:
    """Return the recorded-count integral for one isotope response column."""
    index = decomposer.isotope_names.index(isotope)
    return float(np.sum(decomposer.response_matrix[:, index]))


def test_spectral_decomposition_recovers_sources():
    """Cs-137, Co-60, Eu-154の合成スペクトルを分解して強度を近似的に復元する。"""
    # バックグラウンドの影響を避けるため一時的にオフにする
    original_bg = pipeline.BACKGROUND_COUNTS_PER_SECOND
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    decomposer = SpectralDecomposer()
    env = EnvironmentConfig()
    sources = [
        PointSource("Cs-137", position=(2.0, 2.0, 2.0), intensity_cps_1m=20000.0),
        PointSource("Co-60", position=(8.0, 5.0, 2.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(1.0, 10.0, 1.0), intensity_cps_1m=20000.0),
    ]
    spectrum, effective = decomposer.simulate_spectrum(
        sources, environment=env, acquisition_time=2.0, rng=None
    )
    estimates = decomposer.decompose(spectrum)
    pipeline.BACKGROUND_COUNTS_PER_SECOND = original_bg

    for iso in ["Cs-137", "Co-60", "Eu-154"]:
        assert iso in estimates
        assert np.isclose(estimates[iso], effective[iso], rtol=0.05, atol=1e-6)


def test_response_poisson_fuses_visible_photopeak_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible photopeak evidence should rescue a zero-boundary response fit."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_photopeak_fusion=True,
            response_poisson_low_snr_photopeak_anchor=True,
        )
    )
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return a high-SNR Cs photopeak estimate for fusion testing."""
        decomposer.last_count_variances = {isotope: 1.0 for isotope in isotopes}
        return {isotope: (50.0 if isotope == "Cs-137" else 0.0) for isotope in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Cs-137", "Co-60"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(50.0, rel=1e-3)
    assert estimates["Cs-137"].variance > estimates["Cs-137"].counts
    assert (
        estimates["Cs-137"].method
        == "response_poisson_photopeak_fused_source_equivalent"
    )
    assert estimates["Co-60"].counts == pytest.approx(0.0)


def test_response_poisson_default_does_not_fuse_photopeak_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default response-Poisson counts should not be replaced by photopeak evidence."""
    decomposer = SpectralDecomposer(SpectrumConfig())
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return high-SNR photopeak evidence that must not alter default counts."""
        decomposer.last_count_variances = {isotope: 1.0 for isotope in isotopes}
        return {isotope: (50.0 if isotope == "Cs-137" else 0.0) for isotope in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Cs-137", "Co-60"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(0.0)
    assert estimates["Cs-137"].method == "response_poisson_source_equivalent"
    assert estimates["Co-60"].counts == pytest.approx(0.0)


def test_runtime_response_poisson_variance_ceiling_preserves_count() -> None:
    """Runtime variance ceiling should cap covariance without changing counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_count_variance_ceiling_enable=True,
            response_poisson_count_variance_max_rel_sigma=0.15,
            response_poisson_count_variance_max_abs_sigma=40.0,
        )
    )
    extractor = RuntimeCountExtractor(decomposer)

    capped = extractor._apply_response_poisson_variance_ceiling(
        {"Cs-137": 1000.0, "Co-60": 5.0},
        {"Cs-137": 1.0e9, "Co-60": 1.0e9},
    )

    assert capped["Cs-137"] == pytest.approx((0.15 * 1000.0) ** 2)
    assert capped["Co-60"] == pytest.approx(40.0**2)
    diagnostics = decomposer.last_response_poisson_diagnostics
    assert diagnostics["runtime_variance_ceiling"]["Cs-137"]["count"] == (
        pytest.approx(1000.0)
    )


def test_runtime_count_extractor_forwards_transport_covariance_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime extraction must send variance provenance into response fitting."""
    decomposer = SpectralDecomposer(SpectrumConfig())
    extractor = RuntimeCountExtractor(decomposer)
    captured: dict[str, object] = {}

    def _fake_counts(
        _spectrum: np.ndarray,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Capture the response-regression keyword arguments."""
        captured.update(kwargs)
        counts = {"Cs-137": 100.0, "Co-60": 50.0, "Eu-154": 25.0}
        decomposer.last_count_variances = dict(counts)
        decomposer.last_count_covariance = {
            isotope: {other: (value if isotope == other else 0.0) for other in counts}
            for isotope, value in counts.items()
        }
        decomposer.last_count_covariance_semantics = "inverse_fisher_poisson"
        return counts, set(counts)

    monkeypatch.setattr(decomposer, "isotope_counts_with_detection", _fake_counts)
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)
    variance = 3.0 * spectrum
    transport_spectrum = 2.0 * spectrum
    metadata = {"spectrum_variance_semantics": "test"}

    extractor.extract(
        spectrum,
        live_time_s=1.0,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        spectrum_variance=variance,
        transport_metadata=metadata,
        transport_spectrum=transport_spectrum,
    )

    assert captured["spectrum_variance"] is variance
    assert captured["transport_metadata"] is metadata
    assert captured["transport_spectrum"] is transport_spectrum


def test_photopeak_channel_estimates_preserve_line_semantics() -> None:
    """Diagnostic photopeak channels should expose source and line-split counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            normalize_line_intensities=True,
            response_efficiency_model="unit",
            response_continuum_to_peak=0.0,
            response_backscatter_fraction=0.0,
            photopeak_background_order=0,
        )
    )
    source_count = 2000.0
    co_index = decomposer.isotope_names.index("Co-60")
    spectrum = source_count * decomposer.response_matrix[:, co_index]

    channels = decomposer.compute_photopeak_channel_estimates(
        spectrum,
        live_time_s=1.0,
        isotopes=["Co-60"],
    )

    assert len(channels) == 2
    for channel in channels:
        assert channel.isotope == "Co-60"
        assert channel.source_equivalent_counts == pytest.approx(
            source_count,
            rel=1e-3,
        )
        assert channel.line_weight == pytest.approx(0.5)
        assert channel.line_equivalent_counts == pytest.approx(
            0.5 * source_count,
            rel=1e-3,
        )


def test_photopeak_channel_estimates_do_not_update_count_variances() -> None:
    """Diagnostic channel extraction should not mutate isotope count variances."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            normalize_line_intensities=True,
            response_efficiency_model="unit",
            response_continuum_to_peak=0.0,
            response_backscatter_fraction=0.0,
        )
    )
    decomposer.last_count_variances = {"Cs-137": 123.0}
    cs_index = decomposer.isotope_names.index("Cs-137")
    spectrum = 1000.0 * decomposer.response_matrix[:, cs_index]

    channels = decomposer.compute_photopeak_channel_estimates(
        spectrum,
        live_time_s=1.0,
        isotopes=["Cs-137"],
    )

    assert len(channels) == 1
    assert decomposer.last_count_variances == {"Cs-137": 123.0}


def test_response_poisson_keeps_low_snr_photopeak_with_large_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-SNR photopeak evidence should remain available with high variance."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(response_poisson_photopeak_fusion=True)
    )
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return weak Cs evidence below the nominal fusion SNR."""
        decomposer.last_count_variances = {isotope: 100.0 for isotope in isotopes}
        return {isotope: (10.0 if isotope == "Cs-137" else 0.0) for isotope in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Cs-137", "Co-60"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(10.0)
    assert estimates["Cs-137"].variance >= 400.0
    assert (
        estimates["Cs-137"].method
        == "response_poisson_photopeak_fused_source_equivalent"
    )
    assert estimates["Co-60"].counts == pytest.approx(0.0)


def test_response_poisson_keeps_full_spectrum_count_for_weak_photopeak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak photopeak evidence must not suppress a full-spectrum response fit."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_photopeak_fusion=True,
            response_poisson_low_snr_photopeak_anchor=True,
        )
    )
    no_fusion_decomposer = SpectralDecomposer(
        SpectrumConfig(response_poisson_photopeak_fusion=False)
    )
    isotope = "Eu-154"
    isotope_index = decomposer.isotope_names.index(isotope)
    spectrum = 80.0 * decomposer.response_matrix[:, isotope_index]
    reference = no_fusion_decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=[isotope],
    )[isotope]

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return a weak local photopeak estimate below the fusion SNR."""
        decomposer.last_count_variances = {name: 100.0 for name in isotopes}
        return {name: (5.0 if name == isotope else 0.0) for name in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=[isotope],
    )

    assert estimates[isotope].counts == pytest.approx(reference.counts, rel=1e-3)
    assert estimates[isotope].variance >= estimates[isotope].counts
    assert (
        estimates[isotope].method
        == "response_poisson_low_snr_photopeak_retained_source_equivalent"
    )


def test_response_poisson_suppresses_low_support_continuum_crosstalk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak low-SNR continuum-only components should not become isotope counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_photopeak_fusion=True,
            response_poisson_low_snr_photopeak_anchor=True,
            response_poisson_low_snr_suppress_count=True,
        )
    )
    cs_index = decomposer.isotope_names.index("Cs-137")
    eu_index = decomposer.isotope_names.index("Eu-154")
    spectrum = (
        10000.0 * decomposer.response_matrix[:, cs_index]
        + 200.0 * decomposer.response_matrix[:, eu_index]
    )

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return no local Eu photopeak evidence despite a continuum coefficient."""
        decomposer.last_count_variances = {
            name: (100.0 if name == "Cs-137" else 1.0) for name in isotopes
        }
        return {name: (10000.0 if name == "Cs-137" else 0.0) for name in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Cs-137", "Eu-154"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(10000.0, rel=0.05)
    assert estimates["Eu-154"].counts == pytest.approx(0.0)
    assert (
        estimates["Eu-154"].method
        == "response_poisson_low_snr_photopeak_suppressed_source_equivalent"
    )


def test_response_poisson_low_snr_guard_runs_without_photopeak_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-SNR crosstalk suppression should not depend on photopeak fusion."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_photopeak_fusion=False,
            response_poisson_low_snr_photopeak_anchor=True,
            response_poisson_low_snr_suppress_count=True,
            response_poisson_low_snr_suppress_fraction=0.15,
        )
    )
    co_index = decomposer.isotope_names.index("Co-60")
    eu_index = decomposer.isotope_names.index("Eu-154")
    spectrum = (
        100000.0 * decomposer.response_matrix[:, co_index]
        + 10000.0 * decomposer.response_matrix[:, eu_index]
    )

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return no Eu photopeak evidence despite a continuum coefficient."""
        decomposer.last_count_variances = {
            name: (100.0 if name == "Co-60" else 1.0) for name in isotopes
        }
        return {name: (100000.0 if name == "Co-60" else 0.0) for name in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Co-60", "Eu-154"],
    )
    diagnostics = decomposer.last_response_poisson_diagnostics

    assert estimates["Co-60"].counts == pytest.approx(100000.0, rel=0.05)
    assert estimates["Eu-154"].counts == pytest.approx(0.0)
    assert (
        estimates["Eu-154"].method
        == "response_poisson_low_snr_photopeak_suppressed_source_equivalent"
    )
    assert diagnostics["low_snr_photopeak_suppression"]["Eu-154"]["suppressed"] is True


def test_response_poisson_low_snr_guard_retains_partial_photopeak_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial photopeak support should not be treated as missing peaks."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_photopeak_fusion=False,
            response_poisson_low_snr_photopeak_anchor=True,
            response_poisson_low_snr_suppress_count=True,
            response_poisson_low_snr_suppress_fraction=0.15,
            response_poisson_low_snr_suppress_photo_to_poisson_ratio=0.2,
        )
    )
    co_index = decomposer.isotope_names.index("Co-60")
    eu_index = decomposer.isotope_names.index("Eu-154")
    spectrum = (
        50000.0 * decomposer.response_matrix[:, co_index]
        + 1000000.0 * decomposer.response_matrix[:, eu_index]
    )

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return low-SNR but non-missing Co photopeak support."""
        decomposer.last_count_variances = {
            name: (1.0e8 if name == "Co-60" else 1.0) for name in isotopes
        }
        return {name: (25000.0 if name == "Co-60" else 1000000.0) for name in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Co-60", "Eu-154"],
    )
    diagnostics = decomposer.last_response_poisson_diagnostics

    assert estimates["Co-60"].counts == pytest.approx(50000.0, rel=0.05)
    assert estimates["Co-60"].method.endswith("source_equivalent")
    assert diagnostics["low_snr_photopeak_suppression"]["Co-60"]["suppressed"] is False
    assert (
        diagnostics["low_snr_photopeak_suppression"]["Co-60"]["photo_to_poisson_ratio"]
        > 0.2
    )


def test_response_poisson_retains_low_snr_full_spectrum_count_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime counts should keep weak full-spectrum evidence with high variance."""
    decomposer = SpectralDecomposer(SpectrumConfig())
    cs_index = decomposer.isotope_names.index("Cs-137")
    eu_index = decomposer.isotope_names.index("Eu-154")
    spectrum = (
        10000.0 * decomposer.response_matrix[:, cs_index]
        + 200.0 * decomposer.response_matrix[:, eu_index]
    )

    def _fake_photopeak_counts(
        _spectrum: np.ndarray,
        *,
        isotopes: list[str],
        **_: object,
    ) -> dict[str, float]:
        """Return no local Eu photopeak evidence despite a continuum coefficient."""
        decomposer.last_count_variances = {
            name: (100.0 if name == "Cs-137" else 1.0) for name in isotopes
        }
        return {name: (10000.0 if name == "Cs-137" else 0.0) for name in isotopes}

    monkeypatch.setattr(
        decomposer,
        "compute_photopeak_nnls_counts",
        _fake_photopeak_counts,
    )

    estimates = decomposer.compute_response_poisson_estimates(
        spectrum,
        isotopes=["Cs-137", "Eu-154"],
    )

    assert estimates["Eu-154"].counts == pytest.approx(200.0, rel=0.1)
    assert estimates["Eu-154"].variance > estimates["Eu-154"].counts
    assert estimates["Eu-154"].method == "response_poisson_source_equivalent"


def test_response_poisson_count_guard_blends_high_chi2_crosstalk() -> None:
    """High-SNR photopeak disagreement should robustly blend crosstalk counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=1.0,
            response_poisson_crosstalk_count_guard_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=1000.0,
            variance=1000.0,
            method="response_poisson",
        )
    }
    variances = {"Cs-137": 1000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 100.0},
        {"Cs-137": 25.0},
        reduced_chi2=10.0,
        requested=["Cs-137"],
    )
    assert 100.0 < estimates["Cs-137"].counts < 250.0
    assert 0.0 < diagnostics["Cs-137"]["blend_weight"] < 1.0
    assert diagnostics["Cs-137"]["disagreement_fraction"] == pytest.approx(0.9)
    assert estimates["Cs-137"].variance > estimates["Cs-137"].counts
    assert estimates["Cs-137"].method == "response_poisson_photopeak_crosstalk_blend"
    assert "Cs-137" in diagnostics


def test_response_poisson_count_guard_uses_high_snr_photopeak_support() -> None:
    """High-chi2 crosstalk guard should favor photopeaks for large mismatches."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=1.0,
            response_poisson_crosstalk_count_guard_ratio=1.5,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=50000.0,
            variance=50000.0,
            method="response_poisson",
        )
    }
    variances = {"Co-60": 50000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Co-60": 25000.0},
        {"Co-60": 100.0},
        reduced_chi2=10.0,
        requested=["Co-60"],
    )
    assert 25000.0 <= estimates["Co-60"].counts < 0.5 * (50000.0 + 25000.0)
    assert estimates["Co-60"].method == "response_poisson_photopeak_crosstalk_blend"
    assert estimates["Co-60"].variance > estimates["Co-60"].counts
    assert (
        diagnostics["Co-60"]["reason"] == "high_chi2_full_response_photopeak_log_blend"
    )


def test_response_poisson_count_guard_keeps_dominant_channel_count() -> None:
    """Photopeak disagreement should not move the dominant isotope count."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=1.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_weak_channel_fraction=0.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=100000.0,
            variance=100000.0,
            method="response_poisson",
        ),
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=12000.0,
            variance=12000.0,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 100000.0, "Co-60": 12000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 50000.0, "Co-60": 6000.0},
        {"Cs-137": 100.0, "Co-60": 100.0},
        reduced_chi2=10.0,
        requested=["Cs-137", "Co-60"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(100000.0)
    assert estimates["Co-60"].counts < 12000.0
    assert diagnostics["Cs-137"]["weak_channel"] is False
    assert diagnostics["Co-60"]["weak_channel"] is True
    assert (
        estimates["Cs-137"].method == "response_poisson_photopeak_crosstalk_uncertain"
    )
    assert estimates["Co-60"].method == "response_poisson_photopeak_crosstalk_blend"


def test_response_poisson_count_guard_can_adjust_high_chi2_dominant_count() -> None:
    """Configured high-chi2 guard should adjust dominant over-response counts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=1.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_weak_channel_fraction=0.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
            response_poisson_crosstalk_count_guard_adjust_high_chi2_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=100000.0,
            variance=100000.0,
            method="response_poisson",
        ),
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=12000.0,
            variance=12000.0,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 100000.0, "Co-60": 12000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 50000.0, "Co-60": 6000.0},
        {"Cs-137": 100.0, "Co-60": 100.0},
        reduced_chi2=10.0,
        requested=["Cs-137", "Co-60"],
    )

    assert estimates["Cs-137"].counts < 100000.0
    assert estimates["Co-60"].counts < 12000.0
    assert diagnostics["Cs-137"]["weak_channel"] is False
    assert diagnostics["Cs-137"]["high_chi2"] is True
    assert diagnostics["Cs-137"]["adjust_high_chi2_count"] is True
    assert estimates["Cs-137"].method == "response_poisson_photopeak_crosstalk_blend"


def test_response_poisson_count_guard_handles_low_chi2_dominant_crosstalk() -> None:
    """A weak buried isotope should be guarded even with a low global chi2."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_weak_channel_fraction=0.5,
            response_poisson_crosstalk_count_guard_dominance_ratio=20.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=1_600_000.0,
            variance=1_600_000.0,
            method="response_poisson",
        ),
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=15_000.0,
            variance=15_000.0,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 1_600_000.0, "Eu-154": 15_000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 1_550_000.0, "Eu-154": 11_200.0},
        {"Cs-137": 1_550_000.0, "Eu-154": 500.0},
        reduced_chi2=1.5,
        requested=["Cs-137", "Eu-154"],
    )

    assert estimates["Cs-137"].counts == pytest.approx(1_600_000.0)
    assert 11_200.0 < estimates["Eu-154"].counts < 13_000.0
    assert (
        diagnostics["Eu-154"]["reason"]
        == "dominant_channel_crosstalk_photopeak_log_blend"
    )
    assert diagnostics["Eu-154"]["adjust_count"] is True
    assert diagnostics["Eu-154"]["dominant_crosstalk"] is True
    assert diagnostics["Eu-154"]["high_chi2"] is False


def test_response_poisson_count_guard_log_blends_extreme_ratio() -> None:
    """Extreme mismatch should not replace the count with photopeak evidence."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_extreme_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=67564.1379,
            variance=67564.1379,
            method="response_poisson",
        )
    }
    variances = {"Eu-154": 67564.1379}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Eu-154": 26461.8856},
        {"Eu-154": 9203323.8539},
        reduced_chi2=19.3009,
        requested=["Eu-154"],
    )

    assert 26461.8856 < estimates["Eu-154"].counts < 67564.1379
    disagreement_var = (67564.1379 - 26461.8856) ** 2
    assert estimates["Eu-154"].variance >= disagreement_var * (1.0 - 1e-6)
    assert 0.0 < diagnostics["Eu-154"]["blend_weight"] < 1.0
    assert (
        diagnostics["Eu-154"]["reason"]
        == "high_chi2_extreme_full_response_photopeak_log_blend"
    )


def test_response_poisson_count_guard_compounds_extreme_evidence() -> None:
    """Extreme high-chi2 count disagreement should combine guard evidence."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_extreme_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=89893.5293,
            variance=89893.5293,
            method="response_poisson",
        )
    }
    variances = {"Eu-154": 89893.5293}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Eu-154": 34244.7210},
        {"Eu-154": 5822665.3238},
        reduced_chi2=44.7236,
        requested=["Eu-154"],
    )

    assert 34244.7210 < estimates["Eu-154"].counts < 40000.0
    assert diagnostics["Eu-154"]["blend_weight"] > 0.8
    assert diagnostics["Eu-154"]["ratio_photo_weight"] > 0.7
    assert diagnostics["Eu-154"]["chi2_mismatch_weight"] > 0.6


def test_response_poisson_count_guard_boosts_high_chi2_ratio_mismatch() -> None:
    """High-chi2 ratio mismatch should strengthen photopeak count support."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=30000.0,
            variance=30000.0,
            method="response_poisson",
        )
    }
    variances = {"Eu-154": 30000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Eu-154": 20000.0},
        {"Eu-154": 4_000_000.0},
        reduced_chi2=10.0,
        requested=["Eu-154"],
    )

    assert diagnostics["Eu-154"]["high_chi2_ratio_boost"] > 0.0
    assert diagnostics["Eu-154"]["blend_weight"] > 0.6
    assert 20000.0 < estimates["Eu-154"].counts < 23500.0


def test_response_poisson_count_guard_boosts_extreme_dominant_crosstalk() -> None:
    """Extreme high-chi2 weak-channel crosstalk should overcome the SNR cap."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_extreme_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_dominance_ratio=20.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=1_960_000.0,
            variance=1_960_000.0,
            method="response_poisson",
        ),
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=46_600.0,
            variance=46_600.0,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 1_960_000.0, "Eu-154": 46_600.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 1_900_000.0, "Eu-154": 18_200.0},
        {"Cs-137": 1_900_000.0, "Eu-154": 22_325_000.0},
        reduced_chi2=29.0,
        requested=["Cs-137", "Eu-154"],
    )

    assert diagnostics["Eu-154"]["extreme_dominant_boost"] > 0.0
    assert (
        diagnostics["Eu-154"]["blend_weight"] > diagnostics["Eu-154"]["snr_reliability"]
    )
    assert 18_200.0 < estimates["Eu-154"].counts < 22_000.0
    assert estimates["Cs-137"].counts == pytest.approx(1_960_000.0)


def test_response_poisson_count_guard_floors_dominant_crosstalk_collapse() -> None:
    """Dominant-channel crosstalk should not collapse weak counts to zero."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_extreme_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_dominance_ratio=20.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=3_200_000.0,
            variance=3_200_000.0,
            method="response_poisson",
        ),
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=47_000.0,
            variance=47_000.0,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 3_200_000.0, "Co-60": 47_000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 3_100_000.0, "Co-60": 1.0e-10},
        {"Cs-137": 3_100_000.0, "Co-60": 1.0},
        reduced_chi2=170.0,
        requested=["Cs-137", "Co-60"],
    )

    floor = 47_000.0 / np.sqrt(20.0)
    assert estimates["Co-60"].counts == pytest.approx(floor)
    assert diagnostics["Co-60"]["crosstalk_count_floor"] == pytest.approx(floor)
    assert diagnostics["Co-60"]["dominant_crosstalk"] is True
    assert estimates["Co-60"].variance >= (47_000.0 - floor) ** 2


def test_response_poisson_count_guard_limits_modest_dominance_blend() -> None:
    """Dominance should trigger crosstalk checks without dominating blend size."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_extreme_ratio=2.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_dominance_ratio=20.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=591_643.1458,
            variance=591_643.1458,
            method="response_poisson",
        ),
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=9_113.0139,
            variance=9_113.0139,
            method="response_poisson",
        ),
    }
    variances = {"Cs-137": 591_643.1458, "Co-60": 9_113.0139}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 591_643.1458, "Co-60": 7_554.1773},
        {"Cs-137": 1.0, "Co-60": 58_438.4146},
        reduced_chi2=6.2666,
        requested=["Cs-137", "Co-60"],
    )

    assert diagnostics["Co-60"]["dominant_crosstalk"] is True
    assert diagnostics["Co-60"]["poisson_to_photopeak_ratio"] < 1.25
    assert diagnostics["Co-60"]["dominance_blend_weight"] < 0.15
    assert diagnostics["Co-60"]["blend_weight"] < 0.5
    assert 8_300.0 < estimates["Co-60"].counts < 9_113.0139


def test_response_poisson_count_guard_combines_subthreshold_evidence() -> None:
    """Near-threshold weak-channel crosstalk should inflate uncertainty only."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_dominance_ratio=20.0,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=360826.8026,
            variance=360826.8026,
            method="response_poisson",
        ),
        "Eu-154": IsotopeCountEstimate(
            isotope="Eu-154",
            counts=18581.8103,
            variance=18581.8103,
            method="response_poisson",
        ),
    }
    variances = {"Co-60": 360826.8026, "Eu-154": 18581.8103}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Co-60": 360826.8026, "Eu-154": 10873.0886},
        {"Co-60": 1.0, "Eu-154": 520787.8019},
        reduced_chi2=3.782965,
        requested=["Co-60", "Eu-154"],
    )

    assert estimates["Eu-154"].counts == pytest.approx(18581.8103)
    assert estimates["Eu-154"].variance >= (18581.8103 - 10873.0886) ** 2
    assert diagnostics["Eu-154"]["high_chi2"] is False
    assert diagnostics["Eu-154"]["dominant_crosstalk"] is False
    assert diagnostics["Eu-154"]["combined_crosstalk"] is True
    assert diagnostics["Eu-154"]["adjust_count"] is False
    assert diagnostics["Eu-154"]["count_adjustable_crosstalk"] is False
    assert diagnostics["Eu-154"]["combined_crosstalk_weight"] > 0.8
    assert diagnostics["Eu-154"]["reason"] == "combined_crosstalk_photopeak_log_blend"
    assert (
        estimates["Eu-154"].method == "response_poisson_photopeak_crosstalk_uncertain"
    )


def test_response_poisson_count_guard_uses_chi2_mismatch_weight() -> None:
    """A severe full-spectrum mismatch should strongly downweight crosstalk."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.5,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=233196.7576,
            variance=233196.7576,
            method="response_poisson",
        )
    }
    variances = {"Cs-137": 233196.7576}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 139775.6243},
        {"Cs-137": 139775.6243},
        reduced_chi2=259.996,
        requested=["Cs-137"],
    )

    assert 139775.6243 <= estimates["Cs-137"].counts < 160000.0
    assert diagnostics["Cs-137"]["blend_weight"] > 0.85
    assert diagnostics["Cs-137"]["chi2_mismatch_weight"] > 0.85
    assert estimates["Cs-137"].method == "response_poisson_photopeak_crosstalk_blend"


def test_response_poisson_count_guard_partially_blends_near_threshold() -> None:
    """Moderate crosstalk disagreements should keep information from both fits."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=1.0,
            response_poisson_crosstalk_count_guard_ratio=1.2,
            response_poisson_crosstalk_count_guard_photo_snr=1.0,
            response_poisson_crosstalk_count_guard_adjust_count=True,
        )
    )
    estimates = {
        "Cs-137": IsotopeCountEstimate(
            isotope="Cs-137",
            counts=140.0,
            variance=140.0,
            method="response_poisson",
        )
    }
    variances = {"Cs-137": 140.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Cs-137": 100.0},
        {"Cs-137": 25.0},
        reduced_chi2=10.0,
        requested=["Cs-137"],
    )

    assert 100.0 < estimates["Cs-137"].counts < 140.0
    assert 0.0 < diagnostics["Cs-137"]["blend_weight"] < 1.0
    assert estimates["Cs-137"].variance > estimates["Cs-137"].counts


def test_response_poisson_count_guard_blends_high_chi2_underallocation() -> None:
    """High-SNR photopeaks should partially correct full-response undercounts."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_underallocation_count_guard_ratio=1.05,
            response_poisson_underallocation_count_guard_photo_snr=8.0,
        )
    )
    estimates = {
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=1000.0,
            variance=1000.0,
            method="response_poisson",
        )
    }
    variances = {"Co-60": 1000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Co-60": 1200.0},
        {"Co-60": 100.0},
        reduced_chi2=12.0,
        requested=["Co-60"],
    )

    assert 1000.0 < estimates["Co-60"].counts < 1200.0
    assert estimates["Co-60"].variance >= (1200.0 - 1000.0) ** 2
    assert (
        estimates["Co-60"].method == "response_poisson_photopeak_underallocation_blend"
    )
    assert (
        diagnostics["Co-60"]["reason"]
        == "high_chi2_photopeak_underallocation_log_blend"
    )
    assert diagnostics["Co-60"]["underallocation"] is True
    assert diagnostics["Co-60"]["snr_reliability"] == pytest.approx(1.0)


def test_response_poisson_count_guard_soft_weights_subthreshold_underallocation() -> (
    None
):
    """Subthreshold photopeak support should move high-chi2 undercounts softly."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_underallocation_count_guard_ratio=1.05,
            response_poisson_underallocation_count_guard_photo_snr=8.0,
        )
    )
    estimates = {
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=1000.0,
            variance=1000.0,
            method="response_poisson",
        )
    }
    variances = {"Co-60": 1000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Co-60": 1200.0},
        {"Co-60": 90000.0},
        reduced_chi2=12.0,
        requested=["Co-60"],
    )

    assert 1000.0 < estimates["Co-60"].counts < 1100.0
    assert diagnostics["Co-60"]["underallocation"] is True
    assert diagnostics["Co-60"]["snr_reliability"] == pytest.approx(0.5)


def test_response_poisson_count_guard_requires_high_chi2_for_underallocation() -> None:
    """Low residual fits should not be moved upward by photopeak disagreement."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(
            response_poisson_crosstalk_count_guard_reduced_chi2=4.0,
            response_poisson_underallocation_count_guard_ratio=1.05,
            response_poisson_underallocation_count_guard_photo_snr=8.0,
        )
    )
    estimates = {
        "Co-60": IsotopeCountEstimate(
            isotope="Co-60",
            counts=1000.0,
            variance=1000.0,
            method="response_poisson",
        )
    }
    variances = {"Co-60": 1000.0}

    diagnostics = decomposer._apply_response_poisson_count_guard(
        estimates,
        variances,
        {"Co-60": 1200.0},
        {"Co-60": 100.0},
        reduced_chi2=1.5,
        requested=["Co-60"],
    )

    assert estimates["Co-60"].counts == pytest.approx(1000.0)
    assert diagnostics == {}
