"""CeBr3-specific resolution helper tests (Step 1)."""

import numpy as np

from spectrum.response_matrix import default_resolution
from spectrum.pipeline import SpectralDecomposer


def test_cebr3_resolution_matches_expected_fwhm_ratios() -> None:
    """Check FWHM ratios at reference energies fall in CeBr3 ranges."""
    sigma = default_resolution()
    energies = np.array([122.0, 661.7, 1332.5], dtype=float)
    fwhm = 2.355 * np.array([sigma(e) for e in energies])

    ratios = fwhm / energies
    assert 0.07 < ratios[0] < 0.10  # ~8-10% at 122 keV
    assert 0.035 < ratios[1] < 0.05  # ~4% at 662 keV
    assert 0.025 < ratios[2] < 0.04  # ~3% at 1332 keV


def test_cebr3_resolution_improves_with_energy() -> None:
    """Resolution should monotonically improve (relative FWHM decreases)."""
    sigma = default_resolution()
    energies = np.array([122.0, 661.7, 1332.5], dtype=float)
    fwhm = 2.355 * np.array([sigma(e) for e in energies])
    ratios = fwhm / energies
    assert ratios[2] < ratios[1] < ratios[0]


def test_pipeline_uses_cebr3_like_default_resolution() -> None:
    """The runtime spectrum pipeline should use the same CeBr3-like resolution."""
    decomposer = SpectralDecomposer()
    energies = np.array([122.0, 661.7, 1332.5], dtype=float)
    fwhm = 2.355 * np.array([decomposer.resolution_fn(e) for e in energies])
    ratios = fwhm / energies

    assert 0.07 < ratios[0] < 0.10
    assert 0.035 < ratios[1] < 0.05
    assert 0.025 < ratios[2] < 0.04
