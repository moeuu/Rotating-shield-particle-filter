"""パラメータチューニング用の参照スペクトル生成ヘルパー。"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource
from spectrum.pipeline import SpectralDecomposer
import spectrum.response_matrix as response_matrix
import spectrum.pipeline as pipeline
import matplotlib.pyplot as plt


def _standard_sources() -> list[PointSource]:
    """main.py と同じ標準点源セットを返す。"""
    return [
        PointSource("Cs-137", position=(5.3, 10.0, 5.0), intensity_cps_1m=20000.0),
        PointSource("Co-60", position=(4.7, 10.6, 5.0), intensity_cps_1m=20000.0),
        PointSource("Eu-154", position=(5.0, 9.4, 4.6), intensity_cps_1m=20000.0),
    ]


def simulate_reference_spectrum(
    continuum_to_peak: float,
    backscatter_fraction: float,
    background_rate_cps: float,
    rng_seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    標準シナリオでスペクトルを生成し、(energy_axis, spectrum)を返す。

    - Cs-137 / Co-60 / Eu-154 (各20 cps@1m)
    - 環境: 10 x 20 x 10 m で中心付近の検出器
    - 120秒相当の計測（acquisition_time=120）
    """
    # 現在値を保存
    prev_cont = response_matrix.COMPTON_CONTINUUM_TO_PEAK
    prev_back = response_matrix.BACKSCATTER_FRACTION
    prev_bg_rate = pipeline.BACKGROUND_RATE_CPS
    prev_bg_rate_alias = pipeline.BACKGROUND_COUNTS_PER_SECOND
    try:
        # パラメータを上書き
        response_matrix.COMPTON_CONTINUUM_TO_PEAK = continuum_to_peak
        response_matrix.BACKSCATTER_FRACTION = backscatter_fraction
        pipeline.BACKGROUND_RATE_CPS = background_rate_cps
        pipeline.BACKGROUND_COUNTS_PER_SECOND = background_rate_cps

        decomposer = SpectralDecomposer()
        env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
        rng = np.random.default_rng(rng_seed)
        spectrum, _ = decomposer.simulate_spectrum(
            _standard_sources(),
            environment=env,
            acquisition_time=120.0,
            rng=rng,
            dead_time_s=0.0,
        )
        return decomposer.energy_axis, spectrum
    finally:
        # パラメータを元に戻す
        response_matrix.COMPTON_CONTINUUM_TO_PEAK = prev_cont
        response_matrix.BACKSCATTER_FRACTION = prev_back
        pipeline.BACKGROUND_RATE_CPS = prev_bg_rate
        pipeline.BACKGROUND_COUNTS_PER_SECOND = prev_bg_rate_alias


@dataclass
class SpectrumQuality:
    """スペクトル品質の簡易指標。"""

    passes: bool
    mean_L: float
    mean_M: float
    mean_H: float
    global_max_energy_keV: float
    peak_prominence: Dict[str, float]


def peak_contrast(
    energy_keV: NDArray[np.float64],
    counts: NDArray[np.float64],
    E0: float,
    peak_half_width_keV: float = 10.0,
    sideband_inner_keV: float = 30.0,
    sideband_outer_keV: float = 60.0,
) -> float:
    """
    指定エネルギー周辺のピーク顕著度を算出する。
    contrast = peak_max / sideband_mean
    """
    E = energy_keV
    C = counts
    peak_mask = (E >= E0 - peak_half_width_keV) & (E <= E0 + peak_half_width_keV)
    side_mask = ((E >= E0 - sideband_outer_keV) & (E <= E0 - sideband_inner_keV)) | (
        (E >= E0 + sideband_inner_keV) & (E <= E0 + sideband_outer_keV)
    )
    if not np.any(peak_mask):
        return 0.0
    peak_max = float(C[peak_mask].max())
    side_vals = C[side_mask]
    side_mean = float(side_vals.mean()) if side_vals.size > 0 else 0.0
    if side_mean <= 0.0:
        return 0.0
    return peak_max / side_mean


def evaluate_spectrum_quality(
    energy_keV: NDArray[np.float64],
    spectrum: NDArray[np.float64],
) -> SpectrumQuality:
    """
    CeBr3想定の右肩下がり形状と主ピーク顕著度にもとづきスペクトル品質を評価する。
    """
    E = energy_keV
    C = spectrum

    mask_L = (E >= 80.0) & (E <= 200.0)
    mask_M = (E >= 400.0) & (E <= 800.0)
    mask_H = (E >= 1200.0) & (E <= 1600.0)

    mean_L = float(C[mask_L].mean())
    mean_M = float(C[mask_M].mean())
    mean_H = float(C[mask_H].mean())

    cond_shape_1 = mean_L >= 0.6 * mean_M
    cond_shape_2 = mean_M >= 1.2 * mean_H

    idx_max = int(np.argmax(C))
    global_max_energy_keV = float(E[idx_max])
    cond_max = (80.0 <= global_max_energy_keV <= 200.0) or (mean_L >= 0.4 * mean_M)

    prom: Dict[str, float] = {}
    prom["Eu-154_1275"] = peak_contrast(E, C, 1274.5, peak_half_width_keV=15.0, sideband_inner_keV=40.0, sideband_outer_keV=90.0)
    prom["Cs-137_662"] = peak_contrast(E, C, 662.0, peak_half_width_keV=12.0, sideband_inner_keV=40.0, sideband_outer_keV=90.0)
    prom["Co-60_1173"] = peak_contrast(E, C, 1173.0, peak_half_width_keV=15.0, sideband_inner_keV=50.0, sideband_outer_keV=100.0)
    prom["Co-60_1332"] = peak_contrast(E, C, 1332.0, peak_half_width_keV=15.0, sideband_inner_keV=50.0, sideband_outer_keV=100.0)

    cond_peak_eu = prom["Eu-154_1275"] >= 1.2
    cond_peak_cs137 = prom["Cs-137_662"] >= 1.5
    cond_peak_co60_1 = prom["Co-60_1173"] >= 1.3
    cond_peak_co60_2 = prom["Co-60_1332"] >= 1.3

    passes = (
        cond_shape_1
        and cond_shape_2
        and cond_max
        and cond_peak_eu
        and cond_peak_cs137
        and cond_peak_co60_1
        and cond_peak_co60_2
    )

    return SpectrumQuality(
        passes=passes,
        mean_L=mean_L,
        mean_M=mean_M,
        mean_H=mean_H,
        global_max_energy_keV=global_max_energy_keV,
        peak_prominence=prom,
    )


def grid_search_parameters(rng_seed: int = 0) -> tuple[float, float, float, SpectrumQuality]:
    """
    簡易グリッドサーチでパラメータを探索し、ベストな組み合わせを返す。
    """
    continuum_grid = [0.4, 0.7, 1.0, 1.5]
    backscatter_grid = [0.03, 0.05, 0.08, 0.10]
    background_grid = [0.0, 5.0, 10.0, 15.0]

    best_quality: SpectrumQuality | None = None
    best_params: tuple[float, float, float] | None = None

    for c in continuum_grid:
        for b in backscatter_grid:
            for bg in background_grid:
                E, spectrum = simulate_reference_spectrum(
                    continuum_to_peak=c,
                    backscatter_fraction=b,
                    background_rate_cps=bg,
                    rng_seed=rng_seed,
                )
                quality = evaluate_spectrum_quality(E, spectrum)
                if quality.passes:
                    return c, b, bg, quality
                # スコアリングでベストを記録
                shape_score = (quality.mean_L / max(quality.mean_M, 1e-6)) + (
                    quality.mean_M / max(quality.mean_H, 1e-6)
                )
                min_prom = min(quality.peak_prominence.values()) if quality.peak_prominence else 0.0
                score = shape_score + 0.5 * min_prom
                if best_quality is None:
                    best_quality = quality
                    best_params = (c, b, bg, score)
                else:
                    _, _, _, best_score = best_params  # type: ignore
                    if score > best_score:
                        best_quality = quality
                        best_params = (c, b, bg, score)

    if best_quality is None or best_params is None:
        # フォールバック（あり得ないはずだが念のため）
        return continuum_grid[0], backscatter_grid[0], background_grid[0], evaluate_spectrum_quality(
            *simulate_reference_spectrum(
                continuum_to_peak=continuum_grid[0],
                backscatter_fraction=backscatter_grid[0],
                background_rate_cps=background_grid[0],
                rng_seed=0,
            )
        )
    c_best, b_best, bg_best, _ = best_params  # type: ignore
    return c_best, b_best, bg_best, best_quality


def print_grid_search_result() -> None:
    """グリッドサーチ結果を人が読みやすく出力する。"""
    c, b, bg, q = grid_search_parameters()
    print("Best parameters:")
    print(f"  continuum_to_peak:   {c}")
    print(f"  backscatter_fraction:{b}")
    print(f"  background_rate_cps: {bg}")
    print("Quality metrics:")
    print(f"  passes: {q.passes}")
    print(f"  mean_L: {q.mean_L:.3f}, mean_M: {q.mean_M:.3f}, mean_H: {q.mean_H:.3f}")
    print(f"  global_max_energy_keV: {q.global_max_energy_keV:.1f}")
    for key, val in q.peak_prominence.items():
        print(f"  peak {key}: {val:.3f}")


def get_best_parameters() -> tuple[float, float, float, SpectrumQuality]:
    """デフォルトグリッドとrng_seed=0でグリッドサーチを実行しベストを返す。"""
    return grid_search_parameters(rng_seed=0)


def plot_cebr3_reference_spectrum() -> None:
    """
    チューニング済みデフォルトでの参照スペクトルを描画・保存する。

    - 662, 1332 keV付近にラベルを付与
    - results/spectrum/cebr3_reference.png に保存
    """
    c, b, bg, _ = get_best_parameters()
    E, spec = simulate_reference_spectrum(
        continuum_to_peak=c, backscatter_fraction=b, background_rate_cps=bg, rng_seed=0
    )
    from pathlib import Path
    results_dir = Path(__file__).resolve().parents[2] / "results" / "spectrum"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(E, spec, label="CeBr3-like reference")
    for energy, label in [(662.0, "Cs-137"), (1332.0, "Co-60"), (1173.0, "Co-60")]:
        ax.axvline(energy, color="r", linestyle="--", alpha=0.5)
        ax.text(energy + 5, max(spec) * 0.05, label, rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("CeBr3-like reference spectrum")
    ax.legend()
    fig.tight_layout()
    out_path = results_dir / "cebr3_reference.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved CeBr3 reference spectrum to {out_path}")


if __name__ == "__main__":
    print_grid_search_result()
