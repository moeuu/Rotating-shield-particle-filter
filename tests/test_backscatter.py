"""バックスキャターピークの挙動を確認するテスト。"""

import numpy as np

from spectrum.library import Nuclide, NuclideLine
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    backscatter_energy,
    build_response_matrix,
    energy_dependent_efficiency,
    default_resolution,
)


def test_backscatter_energy_values():
    """代表的なエネルギーでのバックスキャター位置を確認する。"""
    assert abs(backscatter_energy(352.0) - 148.0) < 10.0
    assert abs(backscatter_energy(662.0) - 184.0) < 10.0
    assert abs(backscatter_energy(1332.0) - 214.0) < 10.0
    assert backscatter_energy(1332.0) > backscatter_energy(662.0)


def test_backscatter_peak_creates_bump():
    """応答行列にバックスキャターピークが反映されることを確認する。"""
    energy_axis = np.arange(0.0, 1501.0, 1.0)
    library = {
        "Test": Nuclide(
            name="Test",
            lines=[NuclideLine(energy_keV=662.0, intensity=1.0)],
            representative_energy_keV=662.0,
        )
    }
    response = build_response_matrix(
        energy_axis,
        library,
        resolution_fn=default_resolution(),
        efficiency_fn=energy_dependent_efficiency,
        bin_width_keV=1.0,
    )[:, 0]
    idx_back = np.argmin(np.abs(energy_axis - backscatter_energy(662.0)))
    window = 5
    back_val = response[max(0, idx_back - window) : idx_back + window + 1].mean()

    mid_mask = (energy_axis > 250.0) & (energy_axis < 400.0)
    mid_val = response[mid_mask].mean()

    assert back_val > mid_val
