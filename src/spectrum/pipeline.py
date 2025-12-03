"""スペクトル生成と分解の簡易パイプラインを提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from measurement.model import EnvironmentConfig, PointSource, inverse_square_scale
from spectrum.library import Nuclide, default_library
from spectrum.response_matrix import (
    build_response_matrix,
    constant_efficiency,
    default_resolution,
)


@dataclass
class SpectrumConfig:
    """スペクトル生成と分解に用いる基本設定。"""

    energy_min_keV: float = 0.0
    energy_max_keV: float = 1500.0
    bin_width_keV: float = 1.0
    resolution_a: float = 0.8
    resolution_b: float = 1.5

    def energy_axis(self) -> NDArray[np.float64]:
        """エネルギー軸を返す。"""
        return np.arange(self.energy_min_keV, self.energy_max_keV + self.bin_width_keV, self.bin_width_keV)


class SpectralDecomposer:
    """Chapter 2のピークベース手法を簡略化したスペクトル分解器。"""

    def __init__(
        self,
        spectrum_config: SpectrumConfig | None = None,
        library: Dict[str, Nuclide] | None = None,
    ) -> None:
        """分解に必要な応答行列と設定を初期化する。"""
        self.config = spectrum_config or SpectrumConfig()
        self.library = library or default_library()
        self.energy_axis = self.config.energy_axis()
        self.resolution_fn = default_resolution(a=self.config.resolution_a, b=self.config.resolution_b)
        self.efficiency_fn = constant_efficiency()
        self.response_matrix = build_response_matrix(
            self.energy_axis,
            self.library,
            resolution_fn=self.resolution_fn,
            efficiency_fn=self.efficiency_fn,
        )
        self.isotope_names = list(self.library.keys())

    def simulate_spectrum(
        self,
        sources: Iterable[PointSource],
        environment: EnvironmentConfig | None = None,
        acquisition_time: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, float]]:
        """
        点源と環境設定に基づき合成スペクトルを生成する。

        戻り値はスペクトル配列と、幾何減衰込みの実効強度辞書。
        """
        env = environment or EnvironmentConfig()
        detector = env.detector()
        spectrum = np.zeros_like(self.energy_axis, dtype=float)
        effective_strengths: Dict[str, float] = {name: 0.0 for name in self.isotope_names}
        for source in sources:
            if source.isotope not in self.library:
                continue
            geom = inverse_square_scale(detector, source)
            effective_strength = source.strength * geom
            col_idx = self.isotope_names.index(source.isotope)
            contribution = acquisition_time * effective_strength
            spectrum += contribution * self.response_matrix[:, col_idx]
            effective_strengths[source.isotope] += contribution

        if rng is not None:
            spectrum = rng.poisson(spectrum)
        return spectrum, effective_strengths

    def decompose(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """観測スペクトルを非負値最小二乗で分解し、核種ごとの強度を返す。"""
        # 非負制約付き最小二乗で活動度を推定
        activities, _ = nnls(self.response_matrix, spectrum)
        return {name: act for name, act in zip(self.isotope_names, activities)}

    def isotope_counts(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """分解結果を使ってPFに渡しやすい同位体別カウントを返す。"""
        activities = self.decompose(spectrum)
        return activities
