"""Legacy grid-based kernel calculations (Chapter 3, Sec. 3.2/3.4).

Note: This module uses discrete candidate_sources indices. New continuous PF code
uses measurement.continuous_kernels instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import OctantShield, octant_index_from_normal, path_length_cm, resolve_mu_values


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
    Sec. 3.2, 3.4に基づくモデル。
    """

    def __init__(
        self,
        candidate_sources: NDArray[np.float64],
        poses: NDArray[np.float64],
        orientations: NDArray[np.float64],
        shield_params: ShieldParams,
        mu_by_isotope: Dict[str, object],
    ) -> None:
        """
        Args:
            candidate_sources: (J,3) 配列の候補源位置。
            poses: (K,3) 配列の検出器姿勢。
            orientations: (R,3) 配列のシールド法線ベクトル。
            shield_params: ShieldParams。
            mu_by_isotope: 核種ごとの線減弱係数（1/cm）。float, (fe, pb), or {"fe","pb"}.
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

        Includes geometric term and exponential attenuation based on shield thickness
        and per-isotope linear attenuation coefficients.
        """
        pose = self.poses[pose_idx]
        kernels = np.zeros(self.num_sources, dtype=float)
        oct_idx = octant_index_from_normal(self.orientations[orient_idx])
        mu_fe, mu_pb = resolve_mu_values(
            self.mu_by_isotope, isotope, default_fe=self.shield_params.mu_fe, default_pb=self.shield_params.mu_pb
        )
        normal = self.orientations[orient_idx]
        for j, src in enumerate(self.sources):
            vec = pose - src
            dist = np.linalg.norm(vec)
            if dist == 0:
                dist = 1e-6
            geom = 1.0 / (4.0 * np.pi * dist**2)
            blocked = self.octant_shield.blocks_ray(
                detector_position=pose, source_position=src, octant_index=oct_idx
            )
            L_fe = path_length_cm(vec, normal, self.shield_params.thickness_fe_cm, blocked=blocked)
            L_pb = path_length_cm(vec, normal, self.shield_params.thickness_pb_cm, blocked=blocked)
            att = float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))
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
        源強度ベクトルから期待計数を計算する（スペクトル生成やテスト専用のヘルパ）。

        PFの観測値は常にスペクトル展開後の同位体別カウント（Sec. 2.5.7）であり、
        本関数はPF重み更新に直接使われないことに注意。
        """
        kvec = self.kernel(isotope, pose_idx, orient_idx)
        return float(live_time_s * (np.dot(kvec, source_strengths) + background))
