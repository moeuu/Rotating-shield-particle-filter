"""コンプトン端と連続成分の基本挙動を検証するテスト。"""

import numpy as np

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
from spectrum.response_matrix import compton_continuum, compton_edge


def test_compton_edge_values():
    """コンプトン端計算が既知値に近いことを確認する。"""
    assert abs(compton_edge(662.0) - 478.0) < 10.0
    assert abs(compton_edge(352.0) - 204.0) < 10.0
    assert compton_edge(1332.0) > compton_edge(662.0)


def test_compton_continuum_properties():
    """連続成分の形状と面積が期待通りであることを確認する。"""
    energy_axis = np.arange(0.0, 1501.0, 1.0)
    bin_width_keV = 1.0
    peak_area = 10.0
    continuum_to_peak = 3.0
    cont = compton_continuum(
        energy_axis,
        e_gamma_keV=662.0,
        bin_width_keV=bin_width_keV,
        peak_area=peak_area,
        continuum_to_peak=continuum_to_peak,
    )
    assert cont.shape == energy_axis.shape
    area_expected = continuum_to_peak * peak_area
    area_actual = cont.sum() * bin_width_keV
    assert abs(area_actual - area_expected) / area_expected < 0.10

    edge = compton_edge(662.0)
    left_sum = cont[energy_axis < 0.5 * edge].sum()
    right_sum = cont[(energy_axis >= 0.5 * edge) & (energy_axis < edge)].sum()
    assert left_sum > right_sum


def test_continuum_dominates_low_energy_mean():
    """合成スペクトルで低エネルギー平均が高エネルギー平均より大きいことを確認する。"""
    dec = SpectralDecomposer()
    env = EnvironmentConfig()
    sources = [
        PointSource("Cs-137", position=(5.0, 10.0, 5.0), intensity_cps_1m=20.0),
        PointSource("Co-60", position=(5.0, 11.0, 5.0), intensity_cps_1m=20.0),
        PointSource("Eu-155", position=(5.0, 9.0, 5.0), intensity_cps_1m=20.0),
    ]
    spectrum, _ = dec.simulate_spectrum(sources, environment=env, acquisition_time=10.0, rng=None, dead_time_s=0.0)
    energy_axis = dec.energy_axis
    low_mask = (energy_axis >= 50.0) & (energy_axis < 200.0)
    high_mask = (energy_axis >= 1000.0) & (energy_axis < 1500.0)
    low_mean = spectrum[low_mask].mean()
    high_mean = spectrum[high_mask].mean()
    assert low_mean > high_mean
