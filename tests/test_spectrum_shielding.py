"""Spectrum-level shielding attenuation tests."""

import numpy as np

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import OctantShield, generate_octant_orientations
from measurement.kernels import ShieldParams
from spectrum.pipeline import SpectralDecomposer


def test_spectrum_attenuates_when_shield_blocks() -> None:
    """
    When the shield blocks the line-of-sight, the full spectrum is attenuated by exp(-mu*L).

    This ensures photopeaks (and unfolded isotope counts) decrease, consistent with
    the physical model (Sec. 3.4–3.5 shielding effect).
    """
    # Remove background to isolate shielding effect
    import spectrum.pipeline as pl

    bg_backup = (pl.BACKGROUND_RATE_CPS, pl.BACKGROUND_COUNTS_PER_SECOND)
    pl.BACKGROUND_RATE_CPS = 0.0
    pl.BACKGROUND_COUNTS_PER_SECOND = 0.0
    env = EnvironmentConfig(detector_position=(2.0, 2.0, 2.0))
    source = PointSource("Cs-137", position=(1.0, 1.0, 1.0), intensity_cps_1m=1000.0)
    decomposer = SpectralDecomposer()
    orientations = generate_octant_orientations()
    shield = OctantShield()

    spectrum_blocked, _ = decomposer.simulate_spectrum(
        sources=[source],
        environment=env,
        acquisition_time=1.0,
        rng=None,
        shield_orientation=orientations[0],  # + + +
        octant_shield=shield,
    )
    spectrum_free, _ = decomposer.simulate_spectrum(
        sources=[source],
        environment=env,
        acquisition_time=1.0,
        rng=None,
        shield_orientation=orientations[7],  # - - -
        octant_shield=shield,
    )
    ratio = spectrum_blocked.sum() / spectrum_free.sum()
    pl.BACKGROUND_RATE_CPS, pl.BACKGROUND_COUNTS_PER_SECOND = bg_backup
    shield_params = ShieldParams()
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(ratio, expected_ratio, rtol=1e-2)
