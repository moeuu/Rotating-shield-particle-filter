"""検出器応答行列の生成と背景スペクトルモデルを扱うモジュール。"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
from numpy.typing import NDArray

from spectrum.library import Nuclide, NuclideLine


def gaussian_peak(energy_axis: NDArray[np.float64], center: float, sigma: float) -> NDArray[np.float64]:
    """与えられた中心と分散で正規分布形状のピークを返す。"""
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return norm * np.exp(-0.5 * ((energy_axis - center) / sigma) ** 2)


def default_resolution(a: float = 0.8, b: float = 1.5) -> Callable[[float], float]:
    """エネルギー分解能σ(E)=a*sqrt(E)+bを返す関数を生成する。"""

    def sigma(energy_keV: float) -> float:
        return a * np.sqrt(energy_keV) + b

    return sigma


def constant_efficiency(value: float = 1.0) -> Callable[[float], float]:
    """エネルギーによらず一定の検出効率を返す関数を生成する。"""

    def eff(_: float) -> float:
        return value

    return eff


def build_response_matrix(
    energy_axis: NDArray[np.float64],
    library: Dict[str, Nuclide],
    resolution_fn: Callable[[float], float],
    efficiency_fn: Callable[[float], float],
    bin_width_keV: float | None = None,
) -> NDArray[np.float64]:
    """
    核種ライブラリに基づいて応答行列を構築する。

    行はエネルギービン、列は核種を表し、各核種の単位強度に対する期待カウントを格納する。
    """
    if bin_width_keV is None:
        if energy_axis.size < 2:
            raise ValueError("energy_axis must contain at least two points to infer bin width")
        bin_width_keV = float(energy_axis[1] - energy_axis[0])
    num_bins = energy_axis.size
    num_iso = len(library)
    matrix = np.zeros((num_bins, num_iso), dtype=float)
    for col_idx, nuclide in enumerate(library.values()):
        matrix[:, col_idx] = _nuclide_response(
            energy_axis, nuclide.lines, resolution_fn, efficiency_fn, bin_width_keV
        )
    return matrix


def _nuclide_response(
    energy_axis: NDArray[np.float64],
    lines: Iterable[NuclideLine],
    resolution_fn: Callable[[float], float],
    efficiency_fn: Callable[[float], float],
    bin_width_keV: float,
) -> NDArray[np.float64]:
    """単一核種のピークを重ねた応答を計算する。"""
    response = np.zeros_like(energy_axis, dtype=float)
    for line in lines:
        sigma = resolution_fn(line.energy_keV)
        peak = gaussian_peak(energy_axis, center=line.energy_keV, sigma=sigma)
        response += line.intensity * efficiency_fn(line.energy_keV) * peak * bin_width_keV
    return response
