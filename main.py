"""Demo script to simulate and decompose a gamma spectrum from three point sources.

Detector assumptions: 2"x2" CeBr3 scintillator with CeBr3-like energy resolution and
right-shoulder-down continuum; includes weak intrinsic/background continuum.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np

# Ensure matplotlib uses a non-interactive backend before importing pyplot.
import matplotlib

matplotlib.use("Agg")

# Ensure src/ is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_DIR = ROOT / "results" / "spectrum"

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import OctantShield, generate_octant_orientations
from spectrum.pipeline import SpectralDecomposer
from spectrum.library import default_library
from spectrum.tuning import evaluate_spectrum_quality
from spectrum.smoothing import gaussian_smooth
from spectrum.baseline import asymmetric_least_squares
from counts.isotope_sequence import build_isotope_count_sequence
import matplotlib.pyplot as plt


def fill_peak_area(
    ax: plt.Axes,
    energy_axis: np.ndarray,
    spectrum: np.ndarray,
    peak_energy_keV: float,
    window_keV: float = 40.0,
    alpha: float = 0.3,
) -> None:
    """
    Visually highlight the net peak area around a known photopeak.

    - Select a symmetric energy window [E0 - window_keV, E0 + window_keV].
    - Within that window, define a simple linear baseline between the two
      end points of the spectrum.
    - Shade the area between this baseline and the spectrum curve.
    - Do not modify the data; this is visualization only.
    """
    mask = (energy_axis >= peak_energy_keV - window_keV) & (energy_axis <= peak_energy_keV + window_keV)
    if not np.any(mask):
        return
    x = energy_axis[mask]
    y = spectrum[mask]
    x0, x1 = x[0], x[-1]
    y0, y1 = y[0], y[-1]
    baseline = y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-9)
    upper = np.maximum(y, baseline)
    ax.fill_between(x, baseline, upper, alpha=alpha)


def main() -> None:
    """Simulate a spectrum and print decomposition results."""
    parser = argparse.ArgumentParser(description="Simulate gamma spectrum demo.")
    parser.add_argument("--compton-continuum-to-peak", type=float, default=None, help="Continuum-to-peak area ratio")
    parser.add_argument("--als-lambda", type=float, default=None, help="ALS baseline lambda")
    parser.add_argument("--als-p", type=float, default=None, help="ALS baseline asymmetry p")
    parser.add_argument("--als-iters", type=int, default=None, help="ALS baseline iterations")
    args = parser.parse_args()

    from spectrum import response_matrix
    from spectrum import pipeline as pipeline_mod

    if args.compton_continuum_to_peak is not None:
        response_matrix.COMPTON_CONTINUUM_TO_PEAK = args.compton_continuum_to_peak
    if args.als_lambda is not None:
        pipeline_mod.BASELINE_LAM = args.als_lambda
    if args.als_p is not None:
        pipeline_mod.BASELINE_P = args.als_p
    if args.als_iters is not None:
        pipeline_mod.BASELINE_NITER = args.als_iters
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0, detector_position=(5.0, 10.0, 5.0))
    acquisition_time = 3.0
    loops = 40
    dead_time_s = 2e-8
    detector_pos = env.detector()
    sources = [
        PointSource("Cs-137", position=(8.0, 12.0, 8.0), intensity_cps_1m=20000.0),
        # Place Co-60 so the (-,-,-) octant also blocks it
        PointSource("Co-60", position=(7.0, 13.0, 6.0), intensity_cps_1m=20000.0),
        # Place Eu-154 so it is also blocked (all components negative toward detector)
        PointSource("Eu-154", position=(7.0, 12.5, 7.0), intensity_cps_1m=20000.0),
    ]
    decomposer = SpectralDecomposer()
    energy_axis = decomposer.energy_axis
    rng = np.random.default_rng(42)
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    spectrum_blocked = np.zeros_like(decomposer.energy_axis, dtype=float)
    spectra_series: list[np.ndarray] = []
    effective_loop = None
    octant_shield = OctantShield()
    orientations = generate_octant_orientations()
    # Block Cs-137 only: vector from Cs (8,12,8) to detector (5,10,5) is negative in all axes
    blocking_orient = orientations[7]
    for _ in range(loops):
        loop_spectrum, loop_effective = decomposer.simulate_spectrum(
            sources, environment=env, acquisition_time=acquisition_time, rng=rng, dead_time_s=dead_time_s
        )
        loop_spectrum_blocked, _ = decomposer.simulate_spectrum(
            sources,
            environment=env,
            acquisition_time=acquisition_time,
            rng=rng,
            dead_time_s=dead_time_s,
            shield_orientation=blocking_orient,
            octant_shield=octant_shield,
        )
        spectrum += loop_spectrum
        spectrum_blocked += loop_spectrum_blocked
        spectra_series.append(loop_spectrum)
        effective_loop = loop_effective
    effective = {k: v * loops for k, v in (effective_loop or {}).items()}
    estimates = decomposer.decompose(spectrum)
    quality = evaluate_spectrum_quality(decomposer.energy_axis, spectrum)

    print("=== Simulation configuration ===")
    print(f"Environment (m): {env.size_x} x {env.size_y} x {env.size_z}")
    print(f"Detector position: {detector_pos.tolist()}")
    for src in sources:
        print(f"  {src.isotope} @ {src.position} -> {src.intensity_cps_1m:.0f} cps at 1 m")
    print(f"Shield orientation (blocking): {blocking_orient.tolist()}")
    for src in sources:
        blocked = octant_shield.blocks_ray(np.array(src.position), detector_pos, octant_index=7)
        print(f"  Shield blocks {src.isotope}: {blocked}")
    print(f"Acquisition: {loops} loops x {acquisition_time:.1f} s = {loops*acquisition_time:.1f} s")
    print(f"Dead time (non-paralyzable): {dead_time_s:.1e} s")

    # Isotope-wise counts per 2.5.7
    iso_names, iso_counts = build_isotope_count_sequence(
        spectra_series,
        energy_axis_keV=decomposer.energy_axis,
        library=default_library(),
        live_time_s=[acquisition_time] * len(spectra_series),
        dead_time_s=0.0,
        window_keV=5.0,
        smooth_sigma_bins=1.0,
        subtract_baseline=True,
    )
    total_counts_per_iso = iso_counts.sum(axis=0)
    print("\n=== Isotope-wise counts (Eq. 2.51–2.53 aggregation) ===")
    for name, total in zip(iso_names, total_counts_per_iso):
        print(f"  {name}: {total:.3f}")

    # 収集スペクトルの統計量を簡易確認
    total_counts = float(spectrum.sum())
    cs_window = (energy_axis >= 652.0) & (energy_axis <= 672.0)
    cs_counts = float(spectrum[cs_window].sum())
    print(f"\nTotal counts (raw): {total_counts:.1f}")
    print(f"Counts in Cs-137 peak window (652–672 keV, raw): {cs_counts:.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Apply smoothing and baseline correction for plotting/CSV
    smoothed = gaussian_smooth(spectrum, sigma_bins=2.0)
    smoothed_blocked = gaussian_smooth(spectrum_blocked, sigma_bins=2.0)
    baseline = asymmetric_least_squares(
        smoothed,
        lam=pipeline_mod.BASELINE_LAM,
        p=pipeline_mod.BASELINE_P,
        niter=pipeline_mod.BASELINE_NITER,
    )
    baseline_blocked = asymmetric_least_squares(
        smoothed_blocked,
        lam=pipeline_mod.BASELINE_LAM,
        p=pipeline_mod.BASELINE_P,
        niter=pipeline_mod.BASELINE_NITER,
    )
    processed = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
    processed_blocked = np.clip(smoothed_blocked - baseline_blocked, a_min=0.0, a_max=None)

    output = np.column_stack([energy_axis, processed])
    out_path = RESULTS_DIR / "spectrum.csv"
    np.savetxt(out_path, output, delimiter=",", header="energy_keV,counts", comments="")
    print(f"\nSpectrum saved to: {out_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_axis, processed, label="Unshielded (processed)")
    ax.plot(energy_axis, processed_blocked, label="Shielded (Cs-137 blocked)", alpha=0.8)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Simulated gamma spectrum (with shielding comparison)")
    # Mark key photopeaks for readability
    peak_definitions = [
        (662.0, "Cs-137"),
        (1173.0, "Co-60"),
        (1332.0, "Co-60"),
        (723.0, "Eu-154"),
        (873.0, "Eu-154"),
        (996.0, "Eu-154"),
        (1275.0, "Eu-154"),
        (1408.0, "Eu-154"),
    ]
    ymin, ymax = ax.get_ylim()
    for e, label in peak_definitions:
        ax.axvline(e, linestyle="--", alpha=0.5)
        ax.text(e, ymax * 0.95, label, rotation=90, va="top", ha="center", fontsize=8)
        fill_peak_area(ax, energy_axis, processed, peak_energy_keV=e, window_keV=40.0, alpha=0.15)
        fill_peak_area(ax, energy_axis, processed_blocked, peak_energy_keV=e, window_keV=40.0, alpha=0.15)
    ax.legend()
    img_path = RESULTS_DIR / "spectrum.png"
    fig.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Spectrum image saved to: {img_path}")

    # 基線確認用のデバッグ図を保存
    fig_dbg, ax_dbg = plt.subplots(figsize=(10, 5))
    ax_dbg.plot(energy_axis, spectrum, label="Raw")
    ax_dbg.plot(energy_axis, spectrum_blocked, label="Raw shielded")
    ax_dbg.plot(energy_axis, smoothed, label="Smoothed")
    ax_dbg.plot(energy_axis, smoothed_blocked, label="Smoothed shielded")
    ax_dbg.plot(energy_axis, baseline, label="Baseline")
    ax_dbg.plot(energy_axis, baseline_blocked, label="Baseline shielded")
    ax_dbg.plot(energy_axis, processed, label="Corrected")
    ax_dbg.plot(energy_axis, processed_blocked, label="Corrected shielded")
    ax_dbg.set_xlabel("Energy (keV)")
    ax_dbg.set_ylabel("Counts")
    ax_dbg.set_title("Baseline debug")
    ax_dbg.legend()
    debug_path = RESULTS_DIR / "spectrum_debug.png"
    fig_dbg.tight_layout()
    fig_dbg.savefig(debug_path, dpi=150)
    plt.close(fig_dbg)
    print(f"Spectrum debug image saved to: {debug_path}")

    # Raw spectra plots: unshielded and shielded positions
    fig_raw, ax_raw = plt.subplots(figsize=(10, 5))
    ax_raw.plot(energy_axis, spectrum, label="Unshielded spectrum (raw)")
    ax_raw.plot(energy_axis, spectrum_blocked, label="Shielded spectrum (raw, Cs-137 attenuated)")
    ax_raw.set_xlabel("Energy (keV)")
    ax_raw.set_ylabel("Counts")
    ax_raw.set_title("Gamma spectra (raw)")
    ax_raw.legend()
    img_path_raw = RESULTS_DIR / "spectrum_raw.png"
    fig_raw.tight_layout()
    fig_raw.savefig(img_path_raw, dpi=150)
    plt.close(fig_raw)
    print(f"Raw spectra image saved to: {img_path_raw}")

    # Shielded-only raw for quick inspection
    fig_atten, ax_atten = plt.subplots(figsize=(10, 5))
    ax_atten.plot(energy_axis, spectrum_blocked, label="Shielded spectrum (raw)")
    ax_atten.set_xlabel("Energy (keV)")
    ax_atten.set_ylabel("Counts")
    ax_atten.set_title("Shielded gamma spectrum (raw, Cs-137 attenuated)")
    ax_atten.legend()
    img_path_atten = RESULTS_DIR / "spectrum_atten.png"
    fig_atten.tight_layout()
    fig_atten.savefig(img_path_atten, dpi=150)
    plt.close(fig_atten)
    print(f"Shielded spectrum image saved to: {img_path_atten}")


if __name__ == "__main__":
    main()
