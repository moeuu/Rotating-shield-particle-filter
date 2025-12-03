"""バックグラウンド形状と低エネルギー優位性のテスト。"""

import numpy as np

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
from spectrum.response_matrix import default_background_shape


def test_background_shape_basic():
    """バックグラウンド形状が閾値以下で0となり低エネルギーに重みがあることを確認。"""
    energy_axis = np.linspace(0.0, 1500.0, 1501)
    bg = default_background_shape(energy_axis)
    assert bg.shape == energy_axis.shape
    assert float(bg[energy_axis < 30.0].max()) == 0.0

    low_band = bg[(energy_axis >= 80.0) & (energy_axis <= 200.0)].mean()
    high_band = bg[(energy_axis >= 800.0) & (energy_axis <= 1500.0)].mean()
    assert low_band > high_band


def test_background_makes_low_energy_dominant():
    """合成スペクトルで低エネルギーの平均が中エネルギーより大きいことを確認。"""
    dec = SpectralDecomposer()
    env = EnvironmentConfig()
    sources = [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=20.0),
        PointSource("Co-60", position=(5.0, 11.0, 5.0), intensity_cps_1m=20.0),
    ]
    spectrum, _ = dec.simulate_spectrum(
        sources, environment=env, acquisition_time=20.0, rng=None, dead_time_s=0.0
    )
    energy_axis = dec.energy_axis
    mean_80_200 = spectrum[(energy_axis >= 80.0) & (energy_axis < 200.0)].mean()
    mean_400_800 = spectrum[(energy_axis >= 400.0) & (energy_axis < 800.0)].mean()
    assert mean_80_200 > mean_400_800
