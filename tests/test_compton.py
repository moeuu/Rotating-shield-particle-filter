"""Direct detector-response tests for Compton physics helpers."""

import numpy as np

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
