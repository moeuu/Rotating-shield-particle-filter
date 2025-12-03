"""スペクトル生成と分解の簡易パイプラインを提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource, inverse_square_scale
from spectrum.library import Nuclide, default_library
from spectrum.response_matrix import (
    build_response_matrix,
    default_background_shape,
    default_resolution,
)
from spectrum.smoothing import gaussian_smooth
from spectrum.baseline import asymmetric_least_squares
from spectrum.dead_time import non_paralyzable_correction
from spectrum.activity_estimation import estimate_activities
from spectrum.decomposition import Peak, strip_overlaps
from spectrum.peak_detection import detect_peaks

# バックグラウンド強度（counts/s）
BACKGROUND_COUNTS_PER_SECOND = 50.0

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
        # エネルギー依存効率に切り替え
        from spectrum.response_matrix import energy_dependent_efficiency

        self.efficiency_fn = energy_dependent_efficiency
        self._background_shape = default_background_shape(self.energy_axis)
        self.response_matrix = build_response_matrix(
            self.energy_axis,
            self.library,
            resolution_fn=self.resolution_fn,
            efficiency_fn=self.efficiency_fn,
            bin_width_keV=self.config.bin_width_keV,
        )
        self.isotope_names = list(self.library.keys())

    def simulate_spectrum(
        self,
        sources: Iterable[PointSource],
        environment: EnvironmentConfig | None = None,
        acquisition_time: float = 1.0,
        rng: np.random.Generator | None = None,
        dead_time_s: float = 0.0,
    ) -> Tuple[NDArray[np.float64], Dict[str, float]]:
        """
        点源と環境設定に基づき合成スペクトルを生成する。

        戻り値はスペクトル配列と、幾何減衰込みの実効強度辞書。
        """
        env = environment or EnvironmentConfig()
        detector = env.detector()
        expected = np.zeros_like(self.energy_axis, dtype=float)
        effective_strengths: Dict[str, float] = {name: 0.0 for name in self.isotope_names}
        for source in sources:
            if source.isotope not in self.library:
                continue
            geom = inverse_square_scale(detector, source)
            effective_strength = source.intensity_cps_1m * geom
            col_idx = self.isotope_names.index(source.isotope)
            contribution = acquisition_time * effective_strength
            expected += contribution * self.response_matrix[:, col_idx]
            effective_strengths[source.isotope] += contribution

        # バックグラウンドを加算
        if BACKGROUND_COUNTS_PER_SECOND > 0.0:
            total_bg_counts = BACKGROUND_COUNTS_PER_SECOND * acquisition_time
            expected += self._background_shape * total_bg_counts

        noisy = rng.poisson(expected) if rng is not None else expected
        corrected = non_paralyzable_correction(noisy, dead_time_s=dead_time_s)
        return corrected, effective_strengths

    def preprocess(self, spectrum: NDArray[np.float64]) -> NDArray[np.float64]:
        """平滑化とベースライン補正を適用してピーク検出を安定化させる。"""
        smoothed = gaussian_smooth(spectrum, sigma_bins=1.0)
        baseline = asymmetric_least_squares(smoothed, lam=1e4, p=0.01, niter=10)
        corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
        return corrected

    def decompose(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """観測スペクトルを非負値最小二乗で分解し、核種ごとの強度を返す。"""
        return estimate_activities(self.response_matrix, spectrum, self.isotope_names)

    def isotope_counts(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """分解結果を使ってPFに渡しやすい同位体別カウントを返す。"""
        return self.decompose(spectrum)

    def identify_by_peaks(
        self,
        spectrum: NDArray[np.float64],
        tolerance_keV: float = 5.0,
    ) -> Dict[str, float]:
        """
        ピーク検出とストリッピングに基づき核種ごとの参照ピーク面積を推定する。

        低カウント環境でピークベース同定を行いたい場合に使用する。
        """
        corrected = self.preprocess(spectrum)
        peak_indices = detect_peaks(corrected, prominence=0.05, distance=5)
        peaks: list[Peak] = []
        for idx in peak_indices:
            energy = self.energy_axis[idx]
            area = corrected[idx]
            peaks.append(Peak(energy_keV=float(energy), area=float(area)))
        ref_areas, _ = strip_overlaps(peaks, self.library, tolerance_keV=tolerance_keV)
        return ref_areas
