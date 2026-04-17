"""Tests for response-matrix spectrum unmixing."""

import pytest

from measurement.continuous_kernels import ContinuousKernel, geometric_term
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    generate_octant_orientations,
    mu_by_isotope_from_tvl_mm,
)
from spectrum.pipeline import SpectralDecomposer


def test_response_model_counts_match_shield_theory_for_mixed_python_spectrum() -> None:
    """Full-response NNLS should recover theory for mixed shielded Python spectra."""
    import spectrum.pipeline as pipeline

    background_backup = (pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND)
    pipeline.BACKGROUND_RATE_CPS = 0.0
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    try:
        isotopes = ["Co-60", "Cs-137", "Eu-154"]
        sources = [
            PointSource("Cs-137", position=(5.0, 5.0, 4.5), intensity_cps_1m=50000.0),
            PointSource("Co-60", position=(6.0, 6.0, 5.5), intensity_cps_1m=30000.0),
            PointSource("Eu-154", position=(7.0, 7.0, 6.5), intensity_cps_1m=30000.0),
        ]
        env = EnvironmentConfig(detector_position=(1.0, 1.0, 0.5))
        dwell_time_s = 30.0
        fe_index = 7
        pb_index = 7
        orientations = generate_octant_orientations()
        mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
        shield_params = ShieldParams()
        kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=False,
        )
        decomposer = SpectralDecomposer()

        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=dwell_time_s,
            rng=None,
            fe_shield_orientation=orientations[fe_index],
            pb_shield_orientation=orientations[pb_index],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        counts = decomposer.compute_response_model_counts(spectrum, isotopes=isotopes)

        detector = env.detector()
        theory = {isotope: 0.0 for isotope in isotopes}
        for source in sources:
            attenuation = kernel.attenuation_factor_pair(
                source.isotope,
                source.position_array(),
                detector,
                fe_index,
                pb_index,
            )
            theory[source.isotope] += (
                dwell_time_s
                * source.intensity_cps_1m
                * geometric_term(detector, source.position_array())
                * attenuation
            )

        for isotope in isotopes:
            assert counts[isotope] == pytest.approx(theory[isotope], rel=1e-10, abs=1e-8)
    finally:
        pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND = background_backup


def test_peak_window_counts_are_not_conservative_for_mixed_spectra() -> None:
    """Peak-window counts should expose cross-talk in mixed shielded spectra."""
    import spectrum.pipeline as pipeline

    background_backup = (pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND)
    pipeline.BACKGROUND_RATE_CPS = 0.0
    pipeline.BACKGROUND_COUNTS_PER_SECOND = 0.0
    try:
        isotopes = ["Co-60", "Cs-137", "Eu-154"]
        sources = [
            PointSource("Cs-137", position=(5.0, 5.0, 4.5), intensity_cps_1m=50000.0),
            PointSource("Co-60", position=(6.0, 6.0, 5.5), intensity_cps_1m=30000.0),
            PointSource("Eu-154", position=(7.0, 7.0, 6.5), intensity_cps_1m=30000.0),
        ]
        env = EnvironmentConfig(detector_position=(1.0, 1.0, 0.5))
        orientations = generate_octant_orientations()
        mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes)
        shield_params = ShieldParams()
        decomposer = SpectralDecomposer()

        free_spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=30.0,
            rng=None,
            fe_shield_orientation=orientations[0],
            pb_shield_orientation=orientations[0],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        blocked_spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=30.0,
            rng=None,
            fe_shield_orientation=orientations[7],
            pb_shield_orientation=orientations[7],
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        free_peak = decomposer.compute_isotope_counts_thesis(
            free_spectrum,
            live_time_s=30.0,
            isotopes=isotopes,
        )
        blocked_peak = decomposer.compute_isotope_counts_thesis(
            blocked_spectrum,
            live_time_s=30.0,
            isotopes=isotopes,
        )
        free_response = decomposer.compute_response_model_counts(free_spectrum, isotopes=isotopes)
        blocked_response = decomposer.compute_response_model_counts(blocked_spectrum, isotopes=isotopes)

        peak_ratio = blocked_peak["Eu-154"] / free_peak["Eu-154"]
        response_ratio = blocked_response["Eu-154"] / free_response["Eu-154"]

        assert peak_ratio > response_ratio * 1.2
    finally:
        pipeline.BACKGROUND_RATE_CPS, pipeline.BACKGROUND_COUNTS_PER_SECOND = background_backup
