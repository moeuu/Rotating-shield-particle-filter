"""検出器応答行列の生成と背景スペクトルモデルを扱うモジュール。"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
from numpy.typing import NDArray

from spectrum.library import Nuclide, NuclideLine

# 電子静止エネルギー
ME_C2_KEV = 511.0  # keV
# バックスキャターピークの強度比
BACKSCATTER_FRACTION = 0.15


def gaussian_peak(energy_axis: NDArray[np.float64], center: float, sigma: float) -> NDArray[np.float64]:
    """与えられた中心と分散で正規分布形状のピークを返す。"""
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return norm * np.exp(-0.5 * ((energy_axis - center) / sigma) ** 2)


def compton_edge(e_gamma_keV: float) -> float:
    """
    単一コンプトン散乱における最大エネルギー付与量（コンプトン端）を返す。

    E_edge = E_gamma * (1 - 1 / (1 + 2 * E_gamma / 511 keV))
    """
    return float(e_gamma_keV * (1.0 - 1.0 / (1.0 + 2.0 * e_gamma_keV / ME_C2_KEV)))


def backscatter_energy(e_gamma_keV: float) -> float:
    """
    180度バックスキャター後のガンマ線エネルギーを返す。

    E_back = E_gamma / (1 + 2 E_gamma / 511 keV)
    """
    return float(e_gamma_keV / (1.0 + 2.0 * e_gamma_keV / ME_C2_KEV))


def compton_continuum(
    energy_axis: NDArray[np.float64],
    e_gamma_keV: float,
    bin_width_keV: float,
    peak_area: float,
    continuum_to_peak: float = 3.0,
    shape_power: float = 2.0,
) -> NDArray[np.float64]:
    """
    単一ガンマ線に対する簡易コンプトン連続成分を生成する。

    - 0 < E < Compton端のみ非ゼロ
    - 高エネルギー側へ単調減少
    - 総面積が continuum_to_peak * peak_area となるよう正規化
    """
    edge = compton_edge(e_gamma_keV)
    cont = np.zeros_like(energy_axis, dtype=float)
    mask = (energy_axis > 0.0) & (energy_axis < edge)
    if not np.any(mask) or peak_area <= 0.0:
        return cont
    x = energy_axis[mask] / edge
    base = (1.0 - x) ** shape_power
    norm = base.sum() * bin_width_keV
    if norm <= 0:
        return cont
    scale = (continuum_to_peak * peak_area) / norm
    cont[mask] = base * scale
    return cont


def default_background_shape(energy_axis_keV: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    エネルギー依存の簡易バックグラウンド形状を返す（最大値で正規化）。

    - 低エネルギー閾値未満は0
    - 100 keV付近で膨らみを持ち、高エネルギーに向けて減衰
    """
    E = np.asarray(energy_axis_keV, dtype=float)
    E_thr = 40.0
    E_scale = 350.0
    bump_center = 110.0
    bump_sigma = 40.0

    base = np.exp(-E / E_scale)
    bump = np.exp(-0.5 * ((E - bump_center) / bump_sigma) ** 2)
    bg = base + 0.8 * bump
    bg[E < E_thr] = 0.0
    if bg.max() > 0:
        bg = bg / bg.max()
    return bg


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


def energy_dependent_efficiency(e_keV: np.ndarray | float) -> np.ndarray:
    """
    NaI(Tl)/CeBr3を模した簡易エネルギー依存効率。

    - 低エネルギー閾値（40 keV未満）でほぼ0
    - 80–200 keV付近が高効率
    - 数MeVに向けて徐々に低下
    """
    e = np.asarray(e_keV, dtype=float)
    x = np.maximum(e, 40.0) / 300.0
    eff = x ** (-0.4)
    eff = np.clip(eff, 0.1, 1.0)
    eff = np.where(e < 40.0, 0.0, eff)
    # スカラー入力時はスカラーで返す
    if eff.shape == ():
        return float(eff)
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
        peak_area = peak.sum() * bin_width_keV
        cont = compton_continuum(
            energy_axis,
            e_gamma_keV=line.energy_keV,
            bin_width_keV=bin_width_keV,
            peak_area=peak_area,
            continuum_to_peak=3.0,
        )
        eff = efficiency_fn(line.energy_keV)
        peak *= eff
        cont *= eff
        response += line.intensity * peak * bin_width_keV
        response += line.intensity * cont
        # バックスキャターピーク（高エネルギーラインのみ）
        if line.energy_keV > 200.0:
            e_back = backscatter_energy(line.energy_keV)
            sigma_back = resolution_fn(e_back)
            back = gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
            back_norm = back.sum() * bin_width_keV
            if back_norm > 0:
                area_back = BACKSCATTER_FRACTION * peak_area
                back *= area_back / back_norm
                back *= energy_dependent_efficiency(e_back)
                response += line.intensity * back
    return response
