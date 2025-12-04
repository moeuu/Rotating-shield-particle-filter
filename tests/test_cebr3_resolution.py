"""CeBr3-specific resolution helper tests (Step 1)."""

import numpy as np

from spectrum.response_matrix import default_resolution


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
