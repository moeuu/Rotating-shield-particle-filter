"""ピーク重畳を考慮した簡易ストリッピングと強度推定を行うモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from spectrum.library import Nuclide


@dataclass(frozen=True)
class Peak:
    """検出されたピークのエネルギーと面積を保持する。"""

    energy_keV: float
    area: float


def reference_line(nuclide: Nuclide) -> float:
    """最強度のラインエネルギーを参照線として返す。"""
    if not nuclide.lines:
        raise ValueError("ライン情報が存在しません")
    return max(nuclide.lines, key=lambda l: l.intensity).energy_keV


def intensity_ratios(nuclide: Nuclide) -> Dict[float, float]:
    """
    参照線に対する各ラインの強度比を計算する。

    Returns:
        {line_energy_keV: ratio}
    """
    ref_energy = reference_line(nuclide)
    ref_intensity = max(l.intensity for l in nuclide.lines)
    ratios: Dict[float, float] = {}
    for line in nuclide.lines:
        ratios[line.energy_keV] = line.intensity / ref_intensity if ref_intensity > 0 else 0.0
    return ratios


def match_peaks_to_library(peaks: Iterable[Peak], library: Dict[str, Nuclide], tolerance_keV: float = 5.0) -> Dict[str, Peak]:
    """
    検出ピークをライブラリの参照線にマッチングする。

    Args:
        peaks: 検出ピーク群。
        library: 核種ライブラリ。
        tolerance_keV: エネルギー差許容幅。

    Returns:
        {isotope: matched_peak}
    """
    matches: Dict[str, Peak] = {}
    for iso, nuclide in library.items():
        ref_energy = reference_line(nuclide)
        closest: Peak | None = None
        min_diff = tolerance_keV
        for pk in peaks:
            diff = abs(pk.energy_keV - ref_energy)
            if diff <= min_diff:
                min_diff = diff
                closest = pk
        if closest is not None:
            matches[iso] = closest
    return matches


def strip_overlaps(
    peaks: Iterable[Peak],
    library: Dict[str, Nuclide],
    tolerance_keV: float = 5.0,
) -> Tuple[Dict[str, float], List[Peak]]:
    """
    ライブラリの強度比を用いて重畳ピークをストリップし、核種ごとの参照線面積を推定する。

    Args:
        peaks: 検出ピーク群。
        library: 核種ライブラリ。
        tolerance_keV: マッチング許容幅。

    Returns:
        (核種ごとの参照線面積, ストリップ後ピーク)
    """
    peak_list = list(peaks)
    matches = match_peaks_to_library(peak_list, library, tolerance_keV=tolerance_keV)
    stripped_peaks: List[Peak] = peak_list.copy()
    ref_areas: Dict[str, float] = {}

    for iso, ref_peak in matches.items():
        ref_areas[iso] = ref_peak.area
        ratios = intensity_ratios(library[iso])
        for energy, ratio in ratios.items():
            if energy == reference_line(library[iso]):
                continue
            # 該当ピークを探索し、予測寄与を差し引く
            target = _find_peak(stripped_peaks, energy, tolerance_keV)
            if target is None:
                continue
            expected = ref_peak.area * ratio
            updated_area = max(target.area - expected, 0.0)
            stripped_peaks = _replace_peak(stripped_peaks, target, updated_area)
    return ref_areas, stripped_peaks


def _find_peak(peaks: List[Peak], energy: float, tolerance_keV: float) -> Peak | None:
    """エネルギー近傍のピークを検索する。"""
    closest: Peak | None = None
    min_diff = tolerance_keV
    for pk in peaks:
        diff = abs(pk.energy_keV - energy)
        if diff <= min_diff:
            min_diff = diff
            closest = pk
    return closest


def _replace_peak(peaks: List[Peak], target: Peak, new_area: float) -> List[Peak]:
    """ピークリスト中の対象ピークの面積を更新する。"""
    updated: List[Peak] = []
    for pk in peaks:
        if pk is target:
            updated.append(Peak(energy_keV=pk.energy_keV, area=new_area))
        else:
            updated.append(pk)
    return updated
