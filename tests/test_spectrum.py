"""スペクトル分解パイプラインの基本挙動を検証するテスト。"""

import numpy as np

from measurement.model import EnvironmentConfig, PointSource
from spectrum import pipeline
from spectrum.pipeline import SpectralDecomposer


def test_spectral_decomposition_recovers_sources():
    """Cs-137, Co-60, Eu-155の合成スペクトルを分解して強度を近似的に復元する。"""
    # バックグラウンドの影響を避けるため一時的にオフにする
    original_bg = pipeline.BACKGROUND_COUNTS_PER_SECOND
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    decomposer = SpectralDecomposer()
    env = EnvironmentConfig()
    sources = [
        PointSource("Cs-137", position=(2.0, 2.0, 2.0), intensity_cps_1m=5.0),
        PointSource("Co-60", position=(8.0, 5.0, 2.0), intensity_cps_1m=3.0),
        PointSource("Eu-155", position=(1.0, 10.0, 1.0), intensity_cps_1m=7.0),
    ]
    spectrum, effective = decomposer.simulate_spectrum(sources, environment=env, acquisition_time=2.0, rng=None)
    estimates = decomposer.decompose(spectrum)
    pipeline.BACKGROUND_COUNTS_PER_SECOND = original_bg

    for iso in ["Cs-137", "Co-60", "Eu-155"]:
        assert iso in estimates
        assert np.isclose(estimates[iso], effective[iso], rtol=0.05, atol=1e-6)
