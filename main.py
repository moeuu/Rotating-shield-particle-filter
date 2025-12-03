"""Demo script to simulate and decompose a gamma spectrum from three point sources."""

from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

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
from spectrum.pipeline import SpectralDecomposer
import matplotlib.pyplot as plt


def main() -> None:
    """Simulate a spectrum and print decomposition results."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=10.0)
    acquisition_time = 1.0
    loops = 120
    sources = [
        PointSource("Cs-137", position=(5.3, 10.0, 5.0), intensity_cps_1m=20.0),
        PointSource("Co-60", position=(4.7, 10.6, 5.0), intensity_cps_1m=20.0),
        PointSource("Eu-155", position=(5.0, 9.4, 4.6), intensity_cps_1m=20.0),
    ]
    decomposer = SpectralDecomposer()
    rng = np.random.default_rng(42)
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    effective_loop = None
    for _ in range(loops):
        loop_spectrum, loop_effective = decomposer.simulate_spectrum(
            sources, environment=env, acquisition_time=acquisition_time, rng=rng, dead_time_s=0.0
        )
        spectrum += loop_spectrum
        effective_loop = loop_effective
    effective = {k: v * loops for k, v in (effective_loop or {}).items()}
    estimates = decomposer.decompose(spectrum)

    print("=== Simulation configuration ===")
    print(f"Environment (m): {env.size_x} x {env.size_y} x {env.size_z}")
    for src in sources:
        print(f"  {src.isotope} @ {src.position} -> 20 cps at 1 m")
    print(f"Acquisition: {loops} loops x {acquisition_time:.1f} s = {loops*acquisition_time:.1f} s")
    print("\n=== Effective counts (geometry + acquisition time) ===")
    for iso, val in effective.items():
        print(f"  {iso}: {val:.3f} counts")

    print("\n=== Decomposition (estimated activities, arbitrary units) ===")
    for iso, val in estimates.items():
        print(f"  {iso}: {val:.3f}")

    print("\n=== Peak-based identification (reference peak areas) ===")
    peak_based = decomposer.identify_by_peaks(spectrum)
    for iso, val in peak_based.items():
        print(f"  {iso}: {val:.3f}")

    print("\n=== Sample spectrum (first 20 bins) ===")
    print(np.array2string(spectrum[:20], precision=3, separator=", "))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    energy_axis = decomposer.energy_axis
    output = np.column_stack([energy_axis, spectrum])
    out_path = RESULTS_DIR / "spectrum.csv"
    np.savetxt(out_path, output, delimiter=",", header="energy_keV,counts", comments="")
    print(f"\nSpectrum saved to: {out_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_axis, spectrum, label="Simulated spectrum")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Simulated gamma spectrum")
    ax.legend()
    img_path = RESULTS_DIR / "spectrum.png"
    fig.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Spectrum image saved to: {img_path}")


if __name__ == "__main__":
    main()
