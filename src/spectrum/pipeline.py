"""スペクトル生成と分解の簡易パイプラインを提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource, inverse_square_scale
from measurement.shielding import OctantShield, octant_index_from_normal
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
BACKGROUND_RATE_CPS = 3.0
# 互換性のための別名
BACKGROUND_COUNTS_PER_SECOND = BACKGROUND_RATE_CPS
# ALS基線推定のデフォルトパラメータ
BASELINE_LAM = 1e5
BASELINE_P = 0.01
BASELINE_NITER = 10

@dataclass
class SpectrumConfig:
    """スペクトル生成と分解に用いる基本設定。"""

    energy_min_keV: float = 0.0
    energy_max_keV: float = 1500.0
    bin_width_keV: float = 2.0
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
        self.resolution_fn = default_resolution()
        # エネルギー依存効率（CeBr3想定）
        from spectrum.response_matrix import cebr3_efficiency

        self.efficiency_fn = cebr3_efficiency
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
        shield_orientation: NDArray[np.float64] | None = None,
        octant_shield: OctantShield | None = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, float]]:
        """
        点源と環境設定に基づき合成スペクトルを生成する。

        戻り値はスペクトル配列と、幾何減衰込みの実効強度辞書。

        Shielding (Sec. 3.4–3.5): if shield_orientation/octant_shield are provided,
        the line-of-sight is tested via blocks_ray and a 0.1 attenuation factor is
        applied to the source contribution to reflect attenuated photopeaks.
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
            atten = 1.0
            if octant_shield is not None and shield_orientation is not None:
                oct_idx = octant_index_from_normal(np.asarray(shield_orientation))
                if octant_shield.blocks_ray(detector_position=detector, source_position=source.position_array(), octant_index=oct_idx):
                    atten = 0.1
            col_idx = self.isotope_names.index(source.isotope)
            contribution = acquisition_time * effective_strength
            expected += atten * contribution * self.response_matrix[:, col_idx]
            effective_strengths[source.isotope] += atten * contribution

        # バックグラウンドを加算
        # エイリアスのどちらを更新しても反映されるように値を解決
        background_rate = BACKGROUND_RATE_CPS
        if BACKGROUND_COUNTS_PER_SECOND != BACKGROUND_RATE_CPS:
            background_rate = BACKGROUND_COUNTS_PER_SECOND
        if background_rate > 0.0:
            total_bg_counts = background_rate * acquisition_time
            expected += self._background_shape * total_bg_counts

        noisy = rng.poisson(expected) if rng is not None else expected
        corrected = non_paralyzable_correction(noisy, dead_time_s=dead_time_s)
        return corrected, effective_strengths

    def preprocess(self, spectrum: NDArray[np.float64]) -> NDArray[np.float64]:
        """平滑化とベースライン補正を適用してピーク検出を安定化させる。"""
        smoothed = gaussian_smooth(spectrum, sigma_bins=2.0)
        baseline = asymmetric_least_squares(
            smoothed,
            lam=BASELINE_LAM,
            p=BASELINE_P,
            niter=BASELINE_NITER,
        )
        corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
        return corrected

    def decompose(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """観測スペクトルを非負値最小二乗で分解し、核種ごとの強度を返す。"""
        return estimate_activities(self.response_matrix, spectrum, self.isotope_names)

    def isotope_counts(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """分解結果を使ってPFに渡しやすい同位体別カウントを返す。"""
        return self.decompose(spectrum)

    @staticmethod
    def debug_baseline(
        energy_axis: NDArray[np.float64],
        raw: NDArray[np.float64],
        smoothed: NDArray[np.float64],
        baseline: NDArray[np.float64],
        corrected: NDArray[np.float64],
        title: str = "Baseline Debug",
    ) -> None:
        """基線推定の挙動を可視化するためのデバッグ用プロット。"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(energy_axis, raw, label="Raw")
        plt.plot(energy_axis, smoothed, label="Smoothed")
        plt.plot(energy_axis, baseline, label="Baseline")
        plt.plot(energy_axis, corrected, label="Corrected")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.title(title)
        plt.legend()
        plt.show()

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
