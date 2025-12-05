"""Verify shielding attenuates unfolded isotope-wise counts (IAS19_EIJIMORITA)."""

import numpy as np
import pytest

from pathlib import Path

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import OctantShield, generate_octant_orientations
from spectrum.pipeline import SpectralDecomposer


def test_shielding_reduces_unfolded_counts() -> None:
    """Blocked rays should yield smaller unfolded counts; print all combos for traceability."""
    rng = np.random.default_rng(0)
    decomposer = SpectralDecomposer()
    shield = OctantShield()
    orientations = generate_octant_orientations()

    detector = np.array([2.0, 2.0, 2.0])
    env = EnvironmentConfig(detector_position=tuple(detector.tolist()))
    # 13 positions x 8 orientations ≈ 100 combinations
    source_positions = rng.uniform(0.5, 3.5, size=(13, 3))

    import spectrum.pipeline as pl

    bg_backup = (pl.BACKGROUND_RATE_CPS, pl.BACKGROUND_COUNTS_PER_SECOND)
    pl.BACKGROUND_RATE_CPS = 0.0
    pl.BACKGROUND_COUNTS_PER_SECOND = 0.0

    outputs: list[str] = []
    log_path = Path("results") / "shielding_unfolding_cases.log"
    try:
        for src_pos in source_positions:
                for orient_idx, orient in enumerate(orientations):
                    blocked = shield.blocks_ray(detector_position=detector, source_position=src_pos, octant_index=orient_idx)
                src = PointSource("Cs-137", position=tuple(src_pos.tolist()), intensity_cps_1m=500.0)
                spectrum_unshielded, _ = decomposer.simulate_spectrum(
                    sources=[src],
                    environment=env,
                    acquisition_time=1.0,
                    rng=None,
                    shield_orientation=None,
                    octant_shield=None,
                )
                spectrum_shielded, _ = decomposer.simulate_spectrum(
                    sources=[src],
                    environment=env,
                    acquisition_time=1.0,
                    rng=None,
                    shield_orientation=orient,
                    octant_shield=shield,
                )
                z_unshielded = decomposer.isotope_counts(spectrum_unshielded)
                z_shielded = decomposer.isotope_counts(spectrum_shielded)
                outputs.append(
                    f"src={src_pos.tolist()}, det={detector.tolist()}, orient_idx={orient_idx}, "
                    f"blocked={blocked}, z_free={z_unshielded['Cs-137']:.3f}, z_blocked={z_shielded['Cs-137']:.3f}"
                )
                if blocked:
                    assert z_shielded["Cs-137"] < z_unshielded["Cs-137"]
                else:
                    assert np.isclose(z_shielded["Cs-137"], z_unshielded["Cs-137"], rtol=1e-6)
    finally:
        pl.BACKGROUND_RATE_CPS, pl.BACKGROUND_COUNTS_PER_SECOND = bg_backup
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(outputs), encoding="utf-8")
        for line in outputs:
            print(line)
        print(f"Shielding/unfolding combinations logged to: {log_path}")
