"""スペクトル分解パイプラインの基本挙動を検証するテスト。"""

import numpy as np
import pytest

from measurement.model import EnvironmentConfig, PointSource
from spectrum import pipeline
from spectrum.pipeline import SpectralDecomposer


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
    spectrum, effective = decomposer.simulate_spectrum(sources, environment=env, acquisition_time=2.0, rng=None)
    estimates = decomposer.decompose(spectrum)
    pipeline.BACKGROUND_COUNTS_PER_SECOND = original_bg

    for iso in ["Cs-137", "Co-60", "Eu-154"]:
        assert iso in estimates
        assert np.isclose(estimates[iso], effective[iso], rtol=0.05, atol=1e-6)


def test_response_poisson_fuses_visible_photopeak_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible photopeak evidence should rescue a zero-boundary response fit."""
    decomposer = SpectralDecomposer()
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
    assert estimates["Cs-137"].method == "response_poisson_photopeak_fused_source_equivalent"
    assert estimates["Co-60"].counts == pytest.approx(0.0)


def test_response_poisson_keeps_low_snr_photopeak_with_large_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-SNR photopeak evidence should remain available with high variance."""
    decomposer = SpectralDecomposer()
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
    assert estimates["Cs-137"].method == "response_poisson_photopeak_fused_source_equivalent"
    assert estimates["Co-60"].counts == pytest.approx(0.0)
