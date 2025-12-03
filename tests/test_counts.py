"""同位体別カウント系列生成の基本挙動を検証する。"""

import numpy as np

from counts.isotope_sequence import build_isotope_count_sequence
from spectrum.library import Nuclide, NuclideLine


def test_isotope_count_sequence_weights_and_windows():
    """強度比とウィンドウ積分が正しく反映されることを確認する。"""
    energy_axis = np.arange(0.0, 11.0, 1.0)
    spectra = [
        np.array([0, 0, 10, 0, 0, 20, 0, 0, 0, 0, 0], dtype=float),
        np.array([0, 0, 5, 0, 0, 10, 0, 0, 0, 0, 0], dtype=float),
    ]
    library = {
        "TestIso": Nuclide(
            name="TestIso",
            lines=[
                NuclideLine(energy_keV=2.0, intensity=1.0),
                NuclideLine(energy_keV=5.0, intensity=1.0),
            ],
            representative_energy_keV=2.0,
        )
    }
    names, counts = build_isotope_count_sequence(
        spectra,
        energy_axis_keV=energy_axis,
        library=library,
        live_time_s=1.0,
        dead_time_s=0.0,
        window_keV=0.5,
        smooth_sigma_bins=None,
        subtract_baseline=False,
    )
    assert names == ["TestIso"]
    # 各ピークを半分ずつ寄与させた合計
    np.testing.assert_allclose(counts[:, 0], [15.0, 7.5])
