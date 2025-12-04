"""1/8球殻状の鉛・鉄シールド形状と遮蔽判定・減衰計算を扱うモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

# 8オクタントを表す符号付き単位ベクトル（(+,+,+), (+,+,-), ...）
OCTANT_NORMALS: NDArray[np.float64] = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=float,
)
OCTANT_NORMALS /= np.linalg.norm(OCTANT_NORMALS, axis=1, keepdims=True)


def cartesian_to_spherical(vec: NDArray[np.float64]) -> Tuple[float, float, float]:
    """
    デカルト座標ベクトルを球座標 (r, theta, phi) に変換する。

    theta: z軸からの極角 [0, π]、phi: x軸からxy平面への方位角 [0, 2π)。
    """
    x, y, z = vec
    r = float(np.linalg.norm(vec))
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = float(np.arccos(z / r))
    phi = float(np.arctan2(y, x) % (2 * np.pi))
    return r, theta, phi


def shield_blocks_radiation(direction: NDArray[np.float64], shield_normal: NDArray[np.float64], tol: float = 1e-6) -> bool:
    """
    方向ベクトルが指定オクタントのシールドに向いているか簡易判定する。

    direction: 源→検出器（または検出器→源）方向のベクトル
    shield_normal: オクタント法線（OCTANT_NORMALSの1本）
    """
    if np.linalg.norm(direction) == 0:
        return False
    dir_unit = direction / np.linalg.norm(direction)
    # 符号が一致する方向のみを遮蔽する（1/8球殻）
    sign_dir = np.sign(np.where(np.abs(dir_unit) < tol, 0.0, dir_unit))
    sign_shield = np.sign(shield_normal)
    return bool(np.all(sign_dir == sign_shield))


@dataclass(frozen=True)
class SphericalOctantShield:
    """1/8球殻シールドの単純モデル。"""

    mu_cm_inv: float  # 材質ごとの線減弱係数（代表値、1/cm）
    thickness_cm: float = 2.0  # 殻の厚み
    inner_radius_cm: float = 5.0  # シールド内側半径（検出器中心）

    def orientations(self) -> NDArray[np.float64]:
        """8オクタントの法線ベクトルを返す。"""
        return OCTANT_NORMALS.copy()

    def path_length_cm(self, direction: NDArray[np.float64], shield_normal: NDArray[np.float64]) -> float:
        """
        指定方向でシールドを貫通する有効経路長を返す。

        1/8球殻に向かう場合のみ cos成分で厚みを割った距離を返す。
        """
        if not shield_blocks_radiation(direction, shield_normal):
            return 0.0
        dir_unit = direction / (np.linalg.norm(direction) + 1e-9)
        cos_theta = float(np.clip(np.dot(dir_unit, shield_normal), 0.0, 1.0))
        if cos_theta <= 0.0:
            return 0.0
        return float(self.thickness_cm / cos_theta)

    def attenuation_factor(self, direction: NDArray[np.float64], shield_normal: NDArray[np.float64]) -> float:
        """
        exp(-μ·L) による単純な減衰係数を返す。

        direction: 源→検出器方向ベクトル
        shield_normal: 選択したオクタント法線
        """
        path = self.path_length_cm(direction, shield_normal)
        if path <= 0.0:
            return 1.0
        return float(np.exp(-self.mu_cm_inv * path))


def lead_shield(thickness_cm: float = 2.0, inner_radius_cm: float = 5.0) -> SphericalOctantShield:
    """
    鉛シールドを生成する。

    mu_cm_inv は代表値 0.7 1/cm を使用。
    """
    return SphericalOctantShield(mu_cm_inv=0.7, thickness_cm=thickness_cm, inner_radius_cm=inner_radius_cm)


def iron_shield(thickness_cm: float = 2.0, inner_radius_cm: float = 5.0) -> SphericalOctantShield:
    """
    鉄シールドを生成する。

    mu_cm_inv は代表値 0.5 1/cm を使用。
    """
    return SphericalOctantShield(mu_cm_inv=0.5, thickness_cm=thickness_cm, inner_radius_cm=inner_radius_cm)


def _angle_in_range(angle: float, low: float, high: float, tol: float = 1e-6) -> bool:
    """角度が指定範囲[low, high)にあるか（wrap-aroundなし）。"""
    return (angle + tol) >= low and (angle - tol) < high


def generate_octant_orientations() -> NDArray[np.float64]:
    """
    8オクタントのシールド方位を返すヘルパ（鉛・鉄で共通使用）。

    KernelPrecomputerなどで orientation 行列として渡すことを想定。
    """
    return OCTANT_NORMALS.copy()


def octant_index_from_normal(normal: NDArray[np.float64], tol: float = 1e-6) -> int:
    """
    法線ベクトルから最も近いオクタントインデックスを取得する。

    符号が不明瞭な軸ベクトル ([1,0,0] など) も扱えるよう、ドット積最大のオクタントを選択する。
    """
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(n) == 0:
        raise ValueError("normal must be non-zero")
    n_unit = n / np.linalg.norm(n)
    dots = OCTANT_NORMALS @ n_unit
    return int(np.argmax(dots))


class OctantShield:
    """
    1/8球殻シールドの幾何判定を担当するクラス。

    detector_positionとsource_positionからベクトル v を計算し、球座標 (θ, φ) を用いて
    指定オクタント（octant_index 0..7）に含まれるかを判定する。
    - θ: 極角 [0, π]（z軸からの角度）
    - φ: 方位角 [0, 2π)（x軸からxy平面への角度）
    """

    def __init__(self, octant_normals: NDArray[np.float64] | None = None) -> None:
        self.octant_normals = octant_normals if octant_normals is not None else OCTANT_NORMALS
        # θ, φ 範囲をオクタントごとに定義
        self.theta_phi_ranges = [
            ((0.0, np.pi / 2.0), (0.0, np.pi / 2.0)),  # + + +
            ((np.pi / 2.0, np.pi), (0.0, np.pi / 2.0)),  # + + -
            ((0.0, np.pi / 2.0), (3.0 * np.pi / 2.0, 2.0 * np.pi)),  # + - +
            ((np.pi / 2.0, np.pi), (3.0 * np.pi / 2.0, 2.0 * np.pi)),  # + - -
            ((0.0, np.pi / 2.0), (np.pi / 2.0, np.pi)),  # - + +
            ((np.pi / 2.0, np.pi), (np.pi / 2.0, np.pi)),  # - + -
            ((0.0, np.pi / 2.0), (np.pi, 3.0 * np.pi / 2.0)),  # - - +
            ((np.pi / 2.0, np.pi), (np.pi, 3.0 * np.pi / 2.0)),  # - - -
        ]

    def blocks_ray(self, source_position: NDArray[np.float64], detector_position: NDArray[np.float64], octant_index: int) -> bool:
        """
        源→検出器のレイが指定オクタントシールドを通過するかを判定する。

        - v = detector - source
        - (θ, φ) を計算し、octant_indexの角度範囲に入れば遮蔽される。
        方向の符号一致判定（shield_blocks_radiation相当）と整合するように定義。
        """
        if octant_index < 0 or octant_index >= len(self.theta_phi_ranges):
            raise ValueError("octant_index must be in [0, 7]")
        v = np.asarray(detector_position, dtype=float) - np.asarray(source_position, dtype=float)
        r, theta, phi = cartesian_to_spherical(v)
        if r == 0.0:
            return False
        (theta_low, theta_high), (phi_low, phi_high) = self.theta_phi_ranges[octant_index]
        in_theta = _angle_in_range(theta, theta_low, theta_high)
        in_phi = _angle_in_range(phi, phi_low, phi_high)
        return bool(in_theta and in_phi)
