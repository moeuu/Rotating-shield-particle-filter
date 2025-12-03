"""エネルギー校正多項式の推定と適用を行うモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationModel:
    """校正式係数を保持し、チャネルとエネルギーを相互変換するモデル。"""

    coefficients: Sequence[float]

    def channel_to_energy(self, channels: NDArray[np.float64]) -> NDArray[np.float64]:
        """チャネル配列をエネルギー軸（keV）に変換する。"""
        poly = np.poly1d(self.coefficients)
        return poly(channels)

    def energy_to_channel(self, energies: NDArray[np.float64]) -> NDArray[np.float64]:
        """エネルギー配列をチャネルに変換する。"""
        # 逆変換は一意ではないため、近似的にルートを探す
        # 低次多項式想定なので簡易ニュートン法を用いる
        energies = np.asarray(energies, dtype=float)
        channels = np.zeros_like(energies, dtype=float)
        for _ in range(10):
            f = np.polyval(self.coefficients, channels) - energies
            df = np.polyval(np.polyder(self.coefficients), channels)
            df = np.where(df == 0, 1e-9, df)
            channels -= f / df
        return channels


def fit_polynomial_calibration(
    reference_peaks: Iterable[Tuple[float, float]],
    order: int = 2,
) -> CalibrationModel:
    """
    既知ピークのチャネルとエネルギーを用いて校正式をフィットする。

    Args:
        reference_peaks: (channel, energy_keV) の反復可能オブジェクト。
        order: 多項式の次数。

    Returns:
        CalibrationModel: 校正式モデル。
    """
    refs = np.asarray(reference_peaks, dtype=float)
    if refs.shape[0] < order + 1:
        raise ValueError("十分な基準ピークがありません")
    channels = refs[:, 0]
    energies = refs[:, 1]
    coeffs = np.polyfit(channels, energies, order)
    return CalibrationModel(coefficients=coeffs)


def apply_calibration(model: CalibrationModel, num_channels: int) -> NDArray[np.float64]:
    """
    校正式を用いてエネルギー軸を生成する。

    Args:
        model: CalibrationModel。
        num_channels: チャネル数。

    Returns:
        keV単位のエネルギー軸配列。
    """
    channels = np.arange(num_channels, dtype=float)
    return model.channel_to_energy(channels)
