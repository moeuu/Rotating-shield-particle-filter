"""デフォルトパラメータで品質基準を満たすことを確認する回帰テスト。"""

import numpy as np

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
from spectrum.tuning import evaluate_spectrum_quality


def test_default_parameters_produce_good_spectrum():
    """デフォルト設定でスペクトル品質指標が合格することを確認する。"""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    sources = [
        PointSource("Cs-137", position=(5.3, 10.0, 5.0), intensity_cps_1m=20000.0),
        PointSource("Co-60", position=(4.7, 10.6, 5.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(5.0, 9.4, 4.6), intensity_cps_1m=20000.0),
    ]

    decomposer = SpectralDecomposer()
    rng = np.random.default_rng(0)
    acquisition_time = 1.0
    loops = 60
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    for _ in range(loops):
        loop_spectrum, _ = decomposer.simulate_spectrum(
            sources,
            environment=env,
            acquisition_time=acquisition_time,
            rng=rng,
            dead_time_s=0.0,
        )
        spectrum += loop_spectrum

    quality = evaluate_spectrum_quality(decomposer.energy_axis, spectrum)
    # CeBr3右肩下がりの形状を許容：低エネルギー帯が高く、中高エネルギーが順に減少
    assert quality.mean_L >= quality.mean_M >= quality.mean_H
    assert quality.peak_prominence["Cs-137_662"] > 1.0
