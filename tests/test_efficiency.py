"""エネルギー依存検出効率の挙動を確認するテスト。"""

import numpy as np

from spectrum.response_matrix import energy_dependent_efficiency


def test_efficiency_low_energy_threshold():
    """40 keV未満ではほぼ0となることを確認する。"""
    val = energy_dependent_efficiency(20.0)
    assert val == 0.0


def test_efficiency_high_vs_low_energy():
    """100 keV付近の効率が1 MeVより大きいことを確認する。"""
    eff_100 = energy_dependent_efficiency(100.0)
    eff_1000 = energy_dependent_efficiency(1000.0)
    assert eff_100 > eff_1000


def test_efficiency_array_input():
    """配列入力でもクリップされ期待通りの形状を返す。"""
    energies = np.array([0.0, 50.0, 100.0])
    eff = energy_dependent_efficiency(energies)
    assert eff.shape == energies.shape
    assert eff[0] == 0.0
    assert eff[1] > 0.0


def test_efficiency_ratios_and_order():
    """CeBr3らしい比率と単調性を確認する。"""
    e1 = energy_dependent_efficiency(59.5)
    e2 = energy_dependent_efficiency(662.0)
    e3 = energy_dependent_efficiency(1332.0)
    assert e1 > e2 > e3
    ratio = e1 / e3
    assert 3.5 < ratio < 6.0
    # 高エネルギーで減少が続くこと
    assert energy_dependent_efficiency(500.0) > energy_dependent_efficiency(1332.0) > energy_dependent_efficiency(2000.0)
