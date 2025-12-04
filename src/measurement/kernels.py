"""幾何減衰と遮蔽減衰を組み合わせたカーネル計算を提供するモジュール（Chapter 3, Sec. 3.2/3.4）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import OctantShield, octant_index_from_normal


@dataclass(frozen=True)
class ShieldParams:
    """軽量シールドの物性と厚み設定。"""

    mu_pb: float = 0.7  # 1/cm at representative energies
    mu_fe: float = 0.5  # 1/cm at representative energies
    thickness_pb_cm: float = 2.0
    thickness_fe_cm: float = 2.0


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
        self.octant_shield = OctantShield()

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

        Includes geometric term and simple orientation-dependent attenuation:
        if the ray falls into the current octant shield, apply factor 0.1 (−90%),
        otherwise factor 1.0.
        """
        pose = self.poses[pose_idx]
        kernels = np.zeros(self.num_sources, dtype=float)
        oct_idx = octant_index_from_normal(self.orientations[orient_idx])
        for j, src in enumerate(self.sources):
            vec = pose - src
            dist = np.linalg.norm(vec)
            if dist == 0:
                dist = 1e-6
            unit_vec = vec / dist
            geom = 1.0 / (4.0 * np.pi * dist**2)
            # Lead/iron扱いを簡略化：blocks_rayがTrueなら0.1、それ以外は1.0
            blocked_lead = self.octant_shield.blocks_ray(src, pose, octant_index=oct_idx)
            blocked_iron = self.octant_shield.blocks_ray(src, pose, octant_index=oct_idx)
            att = 0.1 if (blocked_lead or blocked_iron) else 1.0
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
        """
        源強度ベクトルから期待計数を計算する（スペクトル生成用の内部ヘルパ）。

        PF観測は常にスペクトル展開後の同位体別カウントであり、本関数は直接PF入力を
        生成しないことに注意（コメント目的のみ）。
        """
        kvec = self.kernel(isotope, pose_idx, orient_idx)
        return float(live_time_s * (np.dot(kvec, source_strengths) + background))
