"""非指向性検出器による計測モデルと環境設定を表現するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    """計測環境のサイズと検出器位置を保持する設定クラス。"""

    size_x: float = 10.0
    size_y: float = 20.0
    size_z: float = 10.0
    detector_position: Tuple[float, float, float] | None = None

    def detector(self) -> np.ndarray:
        """検出器位置を返す。未指定の場合は環境中央を返す。"""
        if self.detector_position is None:
            return np.array([self.size_x / 2.0, self.size_y / 2.0, self.size_z / 2.0])
        return np.array(self.detector_position, dtype=float)


@dataclass(frozen=True)
class PointSource:
    """点放射線源を表すデータ構造。"""

    isotope: str
    position: Tuple[float, float, float]
    strength: float

    def position_array(self) -> np.ndarray:
        """位置をnumpy配列として返す。"""
        return np.array(self.position, dtype=float)


def inverse_square_scale(detector: np.ndarray, source: PointSource) -> float:
    """
    逆二乗則による単一点源の幾何スケール係数を返す。

    検出器までの距離に基づき1/(4πd^2)を計算する。
    """
    distance = np.linalg.norm(detector - source.position_array())
    if distance == 0:
        # 同位置は非現実的なので小さな距離でクリップする
        distance = 1e-6
    return 1.0 / (4.0 * np.pi * distance**2)
