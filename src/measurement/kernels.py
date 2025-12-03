"""幾何減衰と遮蔽減衰を組み合わせたカーネル計算を提供するモジュール（Chapter 3, Sec. 3.2/3.4）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShieldParams:
    """軽量シールドの物性と厚み設定。"""

    mu_pb: float = 0.7  # 1/cm at representative energies
    mu_fe: float = 0.5  # 1/cm at representative energies
    thickness_pb_cm: float = 2.0
    thickness_fe_cm: float = 2.0


def _attenuation_factor(
    unit_vec: NDArray[np.float64],
    shield_normal: NDArray[np.float64],
    params: ShieldParams,
    mu_energy_scale: float,
) -> float:
    """
    シールド法線と入射方向の内積に比例した透過厚みを仮定した簡易減衰係数。

    unit_vec: 源→検出器方向の単位ベクトル
    shield_normal: シールド面の法線（遮蔽方向）
    mu_energy_scale: エネルギー依存μをあらかじめ組み込んだスカラー
    """
    # シールドに向かう成分のみ遮蔽すると仮定
    cos_theta = np.clip(np.dot(unit_vec, shield_normal), 0.0, 1.0)
    path_length = (params.thickness_pb_cm + params.thickness_fe_cm) * cos_theta
    return float(np.exp(-mu_energy_scale * path_length))


class KernelPrecomputer:
    """
    測定姿勢・遮蔽方位・候補源位置に対する幾何+遮蔽カーネルを事前計算する。
    Sec. 3.2, 3.4に基づく簡易モデル。
    """

    def __init__(
        self,
        candidate_sources: NDArray[np.float64],
        poses: NDArray[np.float64],
        orientations: NDArray[np.float64],
        shield_params: ShieldParams,
        mu_by_isotope: Dict[str, float],
    ) -> None:
        """
        Args:
            candidate_sources: (J,3) 配列の候補源位置。
            poses: (K,3) 配列の検出器姿勢。
            orientations: (R,3) 配列のシールド法線ベクトル。
            shield_params: ShieldParams。
            mu_by_isotope: 核種ごとの線減弱係数（1/cm）代表値。
        """
        self.sources = candidate_sources
        self.poses = poses
        self.orientations = orientations
        self.shield_params = shield_params
        self.mu_by_isotope = mu_by_isotope
        self.num_sources = candidate_sources.shape[0]
        self.num_poses = poses.shape[0]
        self.num_orient = orientations.shape[0]

    def geometric_term(self, pose: NDArray[np.float64], source: NDArray[np.float64]) -> float:
        """逆二乗の幾何項 1/(4πd^2)"""
        d = np.linalg.norm(pose - source)
        if d == 0:
            d = 1e-6
        return float(1.0 / (4.0 * np.pi * d**2))

    def kernel(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
    ) -> NDArray[np.float64]:
        """
        単位強度源に対する期待計数カーネル (J,) を返す。

        Includes geometric term and orientation-dependent attenuation.
        """
        mu = self.mu_by_isotope.get(isotope, 0.0)
        pose = self.poses[pose_idx]
        shield_normal = self.orientations[orient_idx]
        kernels = np.zeros(self.num_sources, dtype=float)
        for j, src in enumerate(self.sources):
            vec = pose - src
            dist = np.linalg.norm(vec)
            if dist == 0:
                dist = 1e-6
            unit_vec = vec / dist
            geom = 1.0 / (4.0 * np.pi * dist**2)
            att = _attenuation_factor(unit_vec, shield_normal, self.shield_params, mu_energy_scale=mu)
            kernels[j] = geom * att
        return kernels

    def expected_counts(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
        source_strengths: NDArray[np.float64],
        background: float = 0.0,
        live_time_s: float = 1.0,
    ) -> float:
        """源強度ベクトルから期待計数を計算する。"""
        kvec = self.kernel(isotope, pose_idx, orient_idx)
        return float(live_time_s * (np.dot(kvec, source_strengths) + background))
