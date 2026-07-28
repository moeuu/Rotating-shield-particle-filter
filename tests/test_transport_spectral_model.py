"""Tests for the shared geometry-conditioned full-spectrum model."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy import stats

import spectrum.transport_spectral as transport_spectral
from spectrum.response_matrix import (
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    build_native_geant4_detector_response_matrix,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    GeometryConditionedSpectralModel,
    MARK_CONCENTRATION_GRID,
    RATE_SCALE_HALF_WIDTH_GRID,
    VALIDATION_SCENARIO_IDS,
    full_spectrum_acceptance_contract_payload,
    geometry_conditioned_model_from_runtime_config,
    nonparalyzable_count_log_probability_numpy,
    nonparalyzable_count_log_probability_torch,
    rate_scale_mixture_for_half_width,
    sample_nonparalyzable_counts_numpy,
)
from tests.pure_pf_test_support import approved_full_spectrum_model


def _model(
    *,
    dead_time_tau_s: float = 5.813e-9,
    background_rate_cps: float = 5.0,
) -> GeometryConditionedSpectralModel:
    """Return an unapproved physical model for deterministic unit tests."""
    return GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=float(dead_time_tau_s),
        background_rate_cps=float(background_rate_cps),
    )


def _native_response_column_reference(
    input_index: int,
    *,
    bin_count: int = 851,
    bin_width_keV: float = 2.0,
) -> np.ndarray:
    """Independently reproduce one C++ detector-response probability column."""
    output_energy = np.arange(bin_count, dtype=np.float64) * bin_width_keV
    incident_energy = float(input_index) * bin_width_keV
    if incident_energy <= 0.0:
        result = np.zeros(bin_count, dtype=np.float64)
        result[0] = 1.0
        return result
    sigma = max(0.5 * np.sqrt(incident_energy) - 1.5, 0.5)
    peak = np.exp(
        -0.5 * np.square((output_energy - incident_energy) / sigma)
    )
    peak /= np.sum(peak)
    compton_edge = incident_energy * (
        1.0 - 1.0 / (1.0 + 2.0 * incident_energy / 511.0)
    )
    continuum_tau = max(compton_edge / 3.0, 1.0e-12)
    continuum = np.where(
        output_energy <= compton_edge,
        np.exp(-output_energy / continuum_tau),
        0.0,
    )
    continuum /= np.sum(continuum)
    backscatter_energy = incident_energy / (
        1.0 + 2.0 * incident_energy / 511.0
    )
    backscatter_sigma = max(
        0.5 * np.sqrt(backscatter_energy) - 1.5,
        0.5,
    )
    backscatter = np.exp(
        -0.5
        * np.square(
            (output_energy - backscatter_energy) / backscatter_sigma
        )
    )
    backscatter /= np.sum(backscatter)
    result = (peak + 2.0 * continuum + 0.03 * backscatter) / 3.03
    return result / np.sum(result)


def test_native_detector_response_matches_cpp_probability_contract() -> None:
    """The Python mean operator must equal the C++ marking probabilities."""
    axis = np.arange(851, dtype=np.float64) * 2.0
    operator = build_native_geant4_detector_response_matrix(axis, 2.0)
    assert np.allclose(np.sum(operator, axis=0), 1.0, rtol=0.0, atol=2e-15)
    for input_index in (0, 1, 331, 586, 666, 798, 850):
        assert np.allclose(
            operator[:, input_index],
            _native_response_column_reference(input_index),
            rtol=2e-15,
            atol=2e-15,
        )
    assert (
        NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        == "0ab48601e0965c2c9ea973505523558bf"
        "4dd5d394129397c3d4079c143787ae5"
    )


def test_total_below_uncollided_fails_closed_numpy_and_torch() -> None:
    """An impossible incident-count decomposition must never be normalized."""
    model = _model(dead_time_tau_s=0.0, background_rate_cps=0.0)
    line_count = len(model.line_identity)
    total = np.zeros((2, 1, 3, line_count), dtype=np.float64)
    total[0, 0, 0, 0] = 7.0
    total[0, 0, 1, 4] = 11.0
    total[1, 0, 2, 8] = 13.0
    uncollided = 3.0 * total
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    live_times = np.array([30.0], dtype=np.float64)
    with pytest.raises(ValueError, match="cannot exceed total"):
        model._pre_dead_time_mean_numpy(
            total,
            uncollided,
            features,
            live_times,
        )
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cannot exceed total"):
        model._pre_dead_time_mean_torch(
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )


def test_spectrum_mean_and_cross_likelihood_match_torch() -> None:
    """CPU and Torch must agree for batched source-resolved spectra."""
    torch = pytest.importorskip("torch")
    model = _model()
    rng = np.random.default_rng(42)
    particle_count = 4
    view_count = 3
    source_count = 5
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        100.0,
        (particle_count, view_count, source_count, line_count),
    )
    uncollided = total * rng.uniform(0.25, 1.0, total.shape)
    features = rng.uniform(0.0, 2.0, total.shape + (4,))
    live_times = np.array([30.0, 20.0, 10.0], dtype=np.float64)
    mean_numpy = model.predict_mean_numpy(
        total,
        uncollided,
        features,
        live_times,
    )
    mean_torch = (
        model.predict_mean_torch(
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(mean_numpy, mean_torch, rtol=2e-11, atol=2e-9)
    observations = np.rint(mean_numpy[:2]).astype(np.float64)
    log_numpy = model.cross_log_likelihood_numpy(
        observations,
        total,
        uncollided,
        features,
        live_times,
    )
    log_torch = (
        model.cross_log_likelihood_torch(
            torch.as_tensor(observations, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert log_numpy.shape == (2, particle_count)
    assert np.allclose(log_numpy, log_torch, rtol=1e-10, atol=1e-7)


def test_zero_total_marks_are_exactly_neutral_numpy_torch_and_cross() -> None:
    """A zero-count spectrum must contribute only its renewal total term."""
    torch = pytest.importorskip("torch")
    hierarchical = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=0.0,
        rate_scale_nodes_j=(0.9, 1.0, 1.1),
        rate_scale_weights_j=(0.25, 0.5, 0.25),
        mark_concentration_source=100.0,
    )
    plain = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=0.0,
        rate_scale_nodes_j=(0.9, 1.0, 1.1),
        rate_scale_weights_j=(0.25, 0.5, 0.25),
        mark_concentration_source=None,
    )
    line_count = len(hierarchical.line_identity)
    total = np.zeros((2, 3, 2, 1, line_count), dtype=np.float64)
    total[..., 3] = np.asarray(
        (5.0, 25.0, 100.0),
        dtype=np.float64,
    )[np.newaxis, :, np.newaxis, np.newaxis]
    uncollided = total.copy()
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    features[..., 3] = 2.0
    observed = np.zeros((2, 4, 2, 851), dtype=np.float64)
    live = np.asarray((0.01, 30.0), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        numpy_cross = hierarchical.cross_log_likelihood_numpy(
            observed,
            total,
            uncollided,
            features,
            live,
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
        numpy_plain = plain.cross_log_likelihood_numpy(
            observed,
            total,
            uncollided,
            features,
            live,
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
    numpy_single = hierarchical.log_likelihood_numpy(
        observed[0, 0],
        total[0],
        uncollided[0],
        features[0],
        live,
    )
    torch_cross = (
        hierarchical.cross_log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
        .detach()
        .cpu()
        .numpy()
    )
    torch_single = (
        hierarchical.log_likelihood_torch(
            torch.as_tensor(observed[0, 0], dtype=torch.float64),
            torch.as_tensor(total[0], dtype=torch.float64),
            torch.as_tensor(uncollided[0], dtype=torch.float64),
            torch.as_tensor(features[0], dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.isfinite(numpy_cross))
    assert np.array_equal(numpy_cross, numpy_plain)
    assert np.allclose(numpy_cross, torch_cross, rtol=0.0, atol=1.0e-10)
    assert np.array_equal(numpy_single, numpy_cross[0, 0])
    assert np.allclose(numpy_single, torch_single, rtol=0.0, atol=1.0e-10)


def test_background_and_source_share_one_pre_dead_time_total() -> None:
    """Background is added once before one dead-time transform of total rate."""
    tau = 2.0e-3
    background_rate = 7.0
    model = _model(
        dead_time_tau_s=tau,
        background_rate_cps=background_rate,
    )
    line_count = len(model.line_identity)
    live_times = np.asarray([3.0, 5.0], dtype=np.float64)
    source_rates = np.asarray([20.0, 40.0], dtype=np.float64)
    total = np.zeros((1, 2, 1, line_count), dtype=np.float64)
    total[0, :, 0, 0] = source_rates
    features = np.zeros(total.shape + (4,), dtype=np.float64)

    predicted = model.predict_mean_numpy(
        total,
        total,
        features,
        live_times,
    )
    predicted_total = np.sum(predicted[0], axis=-1)
    pre_dead_time_rate = source_rates + background_rate
    expected_total = (
        pre_dead_time_rate
        * live_times
        / (1.0 + pre_dead_time_rate * tau)
    )

    np.testing.assert_allclose(
        predicted_total,
        expected_total,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    manifest = model.manifest_payload()
    assert manifest["dead_time_application_count"] == 1
    assert manifest["background_semantics"] == (
        "independent_pre_dead_time_pulse_rate_added_once"
    )


def test_cross_likelihood_chunking_matches_unchunked_numpy_and_torch() -> None:
    """Action, predictive-sample, and state chunks must preserve exact scores."""
    torch = pytest.importorskip("torch")
    nodes, weights = rate_scale_mixture_for_half_width(0.10)
    model = GeometryConditionedSpectralModel.standard_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=4.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=300.0,
    )
    rng = np.random.default_rng(9284)
    action_count = 2
    state_count = 11
    sample_count = 5
    view_count = 2
    source_count = 2
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        10.0,
        (
            action_count,
            state_count,
            view_count,
            source_count,
            line_count,
        ),
    )
    uncollided = total * rng.uniform(0.2, 1.0, total.shape)
    features = rng.uniform(0.0, 1.0, total.shape + (4,))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    observations = model.sample_predictive_numpy(
        total[:, :sample_count],
        uncollided[:, :sample_count],
        features[:, :sample_count],
        live_times,
        sample_count=1,
        rng=rng,
    )[:, :, 0].astype(np.float64)
    oracle_numpy = model._cross_log_likelihood_numpy_unchunked(
        observations,
        total,
        uncollided,
        features,
        live_times,
    )
    chunked_numpy = model.cross_log_likelihood_numpy(
        observations,
        total,
        uncollided,
        features,
        live_times,
        action_chunk_size=1,
        sample_chunk_size=2,
        state_chunk_size=3,
    )
    np.testing.assert_allclose(
        chunked_numpy,
        oracle_numpy,
        rtol=2.0e-12,
        atol=2.0e-9,
    )
    oracle_torch = model._cross_log_likelihood_torch_unchunked(
        torch.as_tensor(observations, dtype=torch.float64),
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(live_times, dtype=torch.float64),
    )
    chunked_torch = model.cross_log_likelihood_torch(
        torch.as_tensor(observations, dtype=torch.float64),
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(live_times, dtype=torch.float64),
        action_chunk_size=1,
        sample_chunk_size=2,
        state_chunk_size=3,
    )
    np.testing.assert_allclose(
        chunked_torch.detach().cpu().numpy(),
        oracle_torch.detach().cpu().numpy(),
        rtol=2.0e-12,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        chunked_torch.detach().cpu().numpy(),
        chunked_numpy,
        rtol=1.0e-10,
        atol=1.0e-7,
    )


def test_cross_likelihood_working_set_estimate_tracks_chunk_sizes() -> None:
    """The memory estimate must include and respond to all three chunk axes."""
    model = _model()
    unchunked = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=8,
        num_samples=50,
        num_particles=512,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=8,
        sample_chunk_size=50,
        state_chunk_size=512,
    )
    bounded = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=8,
        num_samples=50,
        num_particles=512,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=8,
    )
    assert bounded < unchunked
    assert bounded > 0


def test_birth_proposal_score_is_finite_target_only_and_matches_torch() -> None:
    """Proposal matched-filter scores must not invoke the target likelihood."""
    torch = pytest.importorskip("torch")
    model = _model()
    line_count = len(model.line_identity)
    mask = np.asarray(
        [item["isotope"] == "Cs-137" for item in model.line_identity],
        dtype=np.bool_,
    )
    candidate_count = 6
    total = np.zeros((candidate_count, 2, 1, line_count), dtype=np.float64)
    total[:, :, 0, mask] = np.linspace(10.0, 60.0, candidate_count)[
        :, None, None
    ]
    uncollided = 0.8 * total
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    live_times = np.array([30.0, 30.0], dtype=np.float64)
    observed = np.rint(
        model.predict_mean_numpy(
            total[3:4],
            uncollided[3:4],
            features[3:4],
            live_times,
        )[0]
    )
    numpy_scores = model.birth_proposal_log_scores_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        target_line_mask_l=mask,
    )
    torch_scores = (
        model.birth_proposal_log_scores_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
            target_line_mask_l=torch.as_tensor(mask),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert numpy_scores.shape == (candidate_count,)
    assert np.all(np.isfinite(numpy_scores))
    assert np.allclose(numpy_scores, torch_scores, rtol=1e-10, atol=1e-8)
    assert int(np.argmax(numpy_scores)) in (2, 3, 4)


def test_torch_production_paths_reject_float32() -> None:
    """PF/DSS spectral calculations must fail closed outside float64."""
    torch = pytest.importorskip("torch")
    model = _model()
    line_count = len(model.line_identity)
    total = torch.zeros((1, 1, 1, line_count), dtype=torch.float32)
    features = torch.zeros(total.shape + (4,), dtype=torch.float32)
    with pytest.raises(TypeError, match="float64"):
        model.predict_mean_torch(
            total,
            total,
            features,
            torch.ones(1, dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="float64"):
        nonparalyzable_count_log_probability_torch(
            torch.zeros(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            dead_time_tau_s=1e-4,
        )


def test_renewal_likelihood_has_exact_poisson_zero_dead_time_limit() -> None:
    """At tau=0 the CPU and Torch laws must be the exact Poisson law."""
    torch = pytest.importorskip("torch")
    counts = np.arange(0, 31, dtype=np.float64)
    rates = np.full(counts.shape, 7.25, dtype=np.float64)
    live_times = np.full(counts.shape, 2.5, dtype=np.float64)
    expected = stats.poisson.logpmf(counts, rates * live_times)
    actual = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        live_times,
        dead_time_tau_s=0.0,
    )
    actual_torch = (
        nonparalyzable_count_log_probability_torch(
            torch.as_tensor(counts, dtype=torch.float64),
            torch.as_tensor(rates, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
            dead_time_tau_s=0.0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.allclose(actual_torch, expected, rtol=1e-13, atol=1e-13)


def test_renewal_probability_normalizes_and_matches_torch() -> None:
    """The finite-support renewal law must normalize on CPU and Torch."""
    torch = pytest.importorskip("torch")
    rate = 20.0
    live_time = 1.0
    tau = 0.02
    counts = np.arange(0, int(np.floor(live_time / tau)) + 2)
    rates = np.full(counts.shape, rate, dtype=np.float64)
    times = np.full(counts.shape, live_time, dtype=np.float64)
    log_probability = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        times,
        dead_time_tau_s=tau,
    )
    torch_probability = (
        nonparalyzable_count_log_probability_torch(
            torch.as_tensor(counts, dtype=torch.float64),
            torch.as_tensor(rates, dtype=torch.float64),
            torch.as_tensor(times, dtype=torch.float64),
            dead_time_tau_s=tau,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.isclose(np.sum(np.exp(log_probability)), 1.0, atol=2e-13)
    assert np.allclose(log_probability, torch_probability, atol=2e-11)


def _high_precision_renewal_log_probability(
    count: int,
    rate_cps: float,
    live_time_s: float,
    dead_time_tau_s: float,
) -> float:
    """Return an independent positive-term renewal oracle using mpmath."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        m = mp.mpf(int(count))
        rate = mp.mpf(str(rate_cps))
        live_time = mp.mpf(str(live_time_s))
        tau = mp.mpf(str(dead_time_tau_s))
        first = rate * max(live_time - (m - 1) * tau, mp.mpf(0))
        second = rate * max(live_time - m * tau, mp.mpf(0))
        width = first - second
        log_terms = []

        def _log_gamma_density(argument: object) -> object:
            """Return the shape-m unit-scale gamma log density."""
            if argument == 0:
                return mp.mpf(0) if m == 1 else mp.ninf
            return (
                (m - 1) * mp.log(argument)
                - argument
                - mp.loggamma(m)
            )

        if width > 0:
            mode = min(max(m - 1, second), first)
            log_scale = _log_gamma_density(mode)
            scaled_interval = width * mp.quad(
                lambda unit: mp.exp(
                    _log_gamma_density(second + width * unit) - log_scale
                ),
                [0, 1],
            )
            log_terms.append(log_scale + mp.log(scaled_interval))
        if second > 0:
            log_terms.append(
                m * mp.log(second)
                - second
                - mp.loggamma(m + 1)
            )
        if not log_terms:
            return -np.inf
        maximum = max(log_terms)
        return float(
            maximum
            + mp.log(
                sum(mp.exp(value - maximum) for value in log_terms)
            )
        )


def test_renewal_extreme_tails_match_positive_high_precision_oracle() -> None:
    """Physically possible high/low count tails must remain finite and exact."""
    cases = (
        (9_000, 1_500.0, 2.0, 5.813e-9),
        (9_001, 1_500.0, 2.0, 5.813e-9),
        (1_000, 5_000.0, 1.0, 5.813e-9),
        (500, 5.0, 1.0, 1.0e-3),
        (501, 5.0, 1.0, 1.0e-3),
    )
    first_group_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([case[0] for case in cases[:3]], dtype=np.float64),
        np.asarray([case[1] for case in cases[:3]], dtype=np.float64),
        np.asarray([case[2] for case in cases[:3]], dtype=np.float64),
        dead_time_tau_s=cases[0][3],
    )
    first_group_expected = np.asarray(
        [
            _high_precision_renewal_log_probability(*case)
            for case in cases[:3]
        ],
        dtype=np.float64,
    )
    second_group_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([case[0] for case in cases[3:]], dtype=np.float64),
        np.asarray([case[1] for case in cases[3:]], dtype=np.float64),
        np.asarray([case[2] for case in cases[3:]], dtype=np.float64),
        dead_time_tau_s=cases[3][3],
    )
    second_group_expected = np.asarray(
        [
            _high_precision_renewal_log_probability(*case)
            for case in cases[3:]
        ],
        dtype=np.float64,
    )
    saturation_case = (100_000, 1.0, 1.0, 1.0e-5)
    saturation_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([saturation_case[0]], dtype=np.float64),
        np.asarray([saturation_case[1]], dtype=np.float64),
        np.asarray([saturation_case[2]], dtype=np.float64),
        dead_time_tau_s=saturation_case[3],
    )[0]
    saturation_expected = _high_precision_renewal_log_probability(
        *saturation_case
    )
    assert np.all(np.isfinite(first_group_actual))
    assert np.all(np.isfinite(second_group_actual))
    assert np.allclose(
        first_group_actual,
        first_group_expected,
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.allclose(
        second_group_actual,
        second_group_expected,
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.isclose(
        saturation_actual,
        saturation_expected,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_renewal_extreme_tail_numpy_torch_cpu_cuda_equivalence() -> None:
    """Tail recovery must agree on NumPy, Torch CPU, and available CUDA."""
    torch = pytest.importorskip("torch")
    counts = np.asarray([500.0, 501.0, 9_000.0, 9_001.0])
    rates = np.asarray([5.0, 5.0, 1_500.0, 1_500.0])
    live_times = np.asarray([1.0, 1.0, 2.0, 2.0])
    tau = 5.813e-9
    numpy_log = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        live_times,
        dead_time_tau_s=tau,
    )

    def _torch_result(
        device: str,
        input_counts: np.ndarray,
        input_rates: np.ndarray,
        input_live_times: np.ndarray,
        input_tau: float,
    ) -> np.ndarray:
        """Evaluate the renewal tail on one Torch device."""
        return (
            nonparalyzable_count_log_probability_torch(
                torch.as_tensor(
                    input_counts,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    input_rates,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    input_live_times,
                    dtype=torch.float64,
                    device=device,
                ),
                dead_time_tau_s=input_tau,
            )
            .detach()
            .cpu()
            .numpy()
        )

    assert np.allclose(
        _torch_result("cpu", counts, rates, live_times, tau),
        numpy_log,
        rtol=0.0,
        atol=2.0e-10,
    )
    saturation_counts = np.asarray([100_000.0])
    saturation_rates = np.asarray([1.0])
    saturation_live_times = np.asarray([1.0])
    saturation_tau = 1.0e-5
    saturation_numpy = nonparalyzable_count_log_probability_numpy(
        saturation_counts,
        saturation_rates,
        saturation_live_times,
        dead_time_tau_s=saturation_tau,
    )
    assert np.allclose(
        _torch_result(
            "cpu",
            saturation_counts,
            saturation_rates,
            saturation_live_times,
            saturation_tau,
        ),
        saturation_numpy,
        rtol=0.0,
        atol=1.0e-8,
    )
    if torch.cuda.is_available():
        assert np.allclose(
            _torch_result("cuda", counts, rates, live_times, tau),
            numpy_log,
            rtol=0.0,
            atol=2.0e-10,
        )
        assert np.allclose(
            _torch_result(
                "cuda",
                saturation_counts,
                saturation_rates,
                saturation_live_times,
                saturation_tau,
            ),
            saturation_numpy,
            rtol=0.0,
            atol=1.0e-8,
        )


def test_renewal_log_tail_iteration_limit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconverged lower-series and upper-fraction tails must never be used."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        transport_spectral,
        "RENEWAL_LOG_GAMMA_MAX_ITERATIONS",
        1,
    )
    lower_inputs = (
        np.asarray([500.0]),
        np.asarray([5.0]),
        np.asarray([1.0]),
        1.0e-3,
    )
    upper_inputs = (
        np.asarray([1_000.0]),
        np.asarray([5_000.0]),
        np.asarray([1.0]),
        5.813e-9,
    )
    with pytest.raises(RuntimeError, match="Lower regularized-gamma"):
        nonparalyzable_count_log_probability_numpy(
            lower_inputs[0],
            lower_inputs[1],
            lower_inputs[2],
            dead_time_tau_s=lower_inputs[3],
        )
    with pytest.raises(RuntimeError, match="upper regularized-gamma"):
        nonparalyzable_count_log_probability_torch(
            torch.as_tensor(upper_inputs[0], dtype=torch.float64),
            torch.as_tensor(upper_inputs[1], dtype=torch.float64),
            torch.as_tensor(upper_inputs[2], dtype=torch.float64),
            dead_time_tau_s=upper_inputs[3],
        )


def test_vectorized_renewal_sampler_matches_exact_mean_and_fano() -> None:
    """Inverse-CDF samples must reproduce exact renewal mean and variance."""
    rate = 45.0
    live_time = 1.0
    tau = 0.008
    support = np.arange(0, int(np.floor(live_time / tau)) + 2)
    log_probability = nonparalyzable_count_log_probability_numpy(
        support,
        np.full(support.shape, rate, dtype=np.float64),
        np.full(support.shape, live_time, dtype=np.float64),
        dead_time_tau_s=tau,
    )
    probability = np.exp(log_probability)
    exact_mean = float(np.sum(support * probability))
    exact_variance = float(
        np.sum(np.square(support - exact_mean) * probability)
    )
    samples = sample_nonparalyzable_counts_numpy(
        np.array([rate], dtype=np.float64),
        np.array([live_time], dtype=np.float64),
        dead_time_tau_s=tau,
        sample_count=40_000,
        rng=np.random.default_rng(20260727),
    )[0]
    sampled_mean = float(np.mean(samples))
    sampled_variance = float(np.var(samples))
    assert abs(sampled_mean - exact_mean) < 0.04
    assert abs(
        sampled_variance / sampled_mean
        - exact_variance / exact_mean
    ) < 0.015


def test_renewal_sampler_handles_zero_rate_and_near_saturation() -> None:
    """Inverse-CDF bracketing must cover zero and almost saturated regimes."""
    samples = sample_nonparalyzable_counts_numpy(
        np.array([0.0, 1.0e6], dtype=np.float64),
        np.array([1.0, 0.01], dtype=np.float64),
        dead_time_tau_s=1.0e-4,
        sample_count=1_000,
        rng=np.random.default_rng(12),
    )
    assert np.all(samples[0] == 0)
    assert np.all(samples[1] <= 101)
    assert float(np.mean(samples[1])) > 95.0
    poisson_samples = sample_nonparalyzable_counts_numpy(
        np.array([1.0e4], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        dead_time_tau_s=0.0,
        sample_count=100,
        rng=np.random.default_rng(13),
    )
    assert poisson_samples.shape == (1, 100)


def test_predictive_sampler_preserves_renewal_total_and_mark_shape() -> None:
    """Predictive spectra must contain exact integer conditional marks."""
    model = _model(dead_time_tau_s=1.0e-4, background_rate_cps=0.0)
    line_count = len(model.line_identity)
    total = np.zeros((2, 1, 1, line_count), dtype=np.float64)
    total[:, 0, 0, 0] = np.array([500.0, 800.0])
    uncollided = total.copy()
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    samples = model.sample_predictive_numpy(
        total,
        uncollided,
        features,
        np.array([1.0], dtype=np.float64),
        sample_count=2_000,
        rng=np.random.default_rng(987),
    )
    assert samples.shape == (2, 2_000, 1, 851)
    assert np.all(samples >= 0.0)
    assert np.all(samples == np.floor(samples))
    total_samples = np.sum(samples, axis=-1)
    assert np.all(total_samples <= np.array([500.0, 800.0])[:, None, None] * 2)


def test_model_fails_closed_without_independent_holdout() -> None:
    """A unit-constructed model must never authorize production runtime."""
    model = _model()
    assert model.production_ready is False
    with pytest.raises(RuntimeError, match="all-64 holdout"):
        model.require_production_ready()
    assert model.manifest_payload()["production_ready"] is False


def test_acceptance_contract_file_is_predeclared_and_hash_stable() -> None:
    """The versioned JSON must exactly match the in-code holdout contract."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "validation"
        / "full_spectrum_acceptance_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == full_spectrum_acceptance_contract_payload()
    assert len(FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256) == 64


def test_validation_manifest_and_line_identity_are_immutable_snapshots() -> None:
    """External mutation must not authorize an already-created model."""
    validation = {
        "schema_version": 1,
        "all_passed": False,
    }
    model = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        validation_manifest=validation,
    )
    original_hash = model.contract_hash_sha256
    validation["all_passed"] = True
    assert model.production_ready is False
    assert model.contract_hash_sha256 == original_hash
    returned_line = dict(model.line_identity[0])
    returned_line["energy_keV"] = -1.0
    assert float(model.line_identity[0]["energy_keV"]) > 0.0


def test_hierarchical_likelihood_matches_torch_and_background_is_exact() -> None:
    """The calibrated mixture must match Torch and leave K=0 marks exact."""
    torch = pytest.importorskip("torch")
    nodes, weights = rate_scale_mixture_for_half_width(0.20)
    model = GeometryConditionedSpectralModel.standard_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=7.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=300.0,
    )
    plain = GeometryConditionedSpectralModel.standard_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=7.0,
    )
    rng = np.random.default_rng(1729)
    line_count = len(model.line_identity)
    total = rng.uniform(0.0, 25.0, (3, 2, 2, line_count))
    uncollided = 0.7 * total
    features = rng.uniform(0.0, 1.0, total.shape + (4,))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    observed = model.sample_predictive_numpy(
        total[:1],
        uncollided[:1],
        features[:1],
        live_times,
        sample_count=2,
        rng=rng,
    )[0]
    numpy_log = model.cross_log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
    )
    torch_log = (
        model.cross_log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(numpy_log, torch_log, rtol=1e-10, atol=1e-7)

    zero = np.zeros((2, 2, 1, line_count), dtype=np.float64)
    zero_features = np.zeros(zero.shape + (4,), dtype=np.float64)
    background_observed = plain.sample_predictive_numpy(
        zero[:1],
        zero[:1],
        zero_features[:1],
        live_times,
        sample_count=3,
        rng=rng,
    )[0]
    hierarchical_background = model.cross_log_likelihood_numpy(
        background_observed,
        zero,
        zero,
        zero_features,
        live_times,
    )
    exact_background = plain.cross_log_likelihood_numpy(
        background_observed,
        zero,
        zero,
        zero_features,
        live_times,
    )
    assert np.allclose(
        hierarchical_background,
        exact_background,
        rtol=0.0,
        atol=2e-12,
    )


def test_predictive_sampler_uses_one_rate_scale_for_all_station_views() -> None:
    """Station-shared scale draws must induce cross-view total covariance."""
    nodes = (0.5, 1.0, 1.5)
    weights = (0.25, 0.50, 0.25)
    model = GeometryConditionedSpectralModel.standard_native(
        ("Cs-137",),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=1_000.0,
    )
    line_count = len(model.line_identity)
    total = np.zeros((1, 2, 1, line_count), dtype=np.float64)
    total[:, :, :, 0] = 100.0
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    samples = model.sample_predictive_numpy(
        total,
        total,
        features,
        np.ones(2, dtype=np.float64),
        sample_count=1_500,
        rng=np.random.default_rng(82),
    )
    totals = np.sum(samples[0], axis=-1)
    correlation = float(np.corrcoef(totals[:, 0], totals[:, 1])[0, 1])
    assert samples.dtype == np.int64
    assert correlation > 0.75


def test_predictive_action_seeds_are_invariant_to_action_batch_width() -> None:
    """Canonical action substreams must survive arbitrary outer batching."""
    model = GeometryConditionedSpectralModel.standard_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=6.0,
        mark_concentration_source=300.0,
    )
    rng = np.random.default_rng(951)
    action_count = 5
    state_count = 3
    view_count = 2
    source_count = 2
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        20.0,
        (
            action_count,
            state_count,
            view_count,
            source_count,
            line_count,
        ),
    )
    uncollided = 0.6 * total
    features = rng.uniform(0.0, 1.0, total.shape + (4,))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    action_seeds = np.asarray([19, 23, 29, 31, 37], dtype=np.int64)
    all_actions = model.sample_predictive_numpy(
        total,
        uncollided,
        features,
        live_times,
        sample_count=4,
        rng=np.random.default_rng(1),
        action_seeds_a=action_seeds,
    )
    split_actions = np.concatenate(
        [
            model.sample_predictive_numpy(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times,
                sample_count=4,
                rng=np.random.default_rng(999 + start),
                action_seeds_a=action_seeds[start:stop],
            )
            for start, stop in ((0, 2), (2, 3), (3, 5))
        ],
        axis=0,
    )
    assert np.array_equal(all_actions, split_actions)


def test_discrepancy_manifest_is_training_only_and_manifest_is_schema_three() -> None:
    """Production provenance must bind global training-only discrepancy fit."""
    width = RATE_SCALE_HALF_WIDTH_GRID[2]
    concentration = MARK_CONCENTRATION_GRID[3]
    nodes, weights = rate_scale_mixture_for_half_width(width)
    digest = "a" * 64
    training = {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "pair_ids_by_scene": {
            str(seed): list(range(64))
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        },
        "artifact_sha256_by_scene": {
            str(seed): digest
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        },
        "rate_scale_family": (
            "station_shared_three_node_symmetric_mean_one"
        ),
        "mark_family": "source_fraction_dirichlet_multinomial",
        "selection_objective": (
            "maximum_joint_training_log_predictive_density"
        ),
        "selected_rate_scale_half_width": width,
        "selected_mark_concentration_source": concentration,
        "candidate_count": (
            len(RATE_SCALE_HALF_WIDTH_GRID)
            * len(MARK_CONCENTRATION_GRID)
        ),
        "selected_training_log_predictive_density": -123.0,
        "selection_artifact_sha256": "b" * 64,
        "selection_completed": True,
    }
    model = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=concentration,
        discrepancy_training_manifest=training,
    )
    manifest = model.manifest_payload()
    assert model.discrepancy_training_ready is True
    assert manifest["schema_version"] == 3
    assert manifest["bin_width_keV"] == 2.0
    assert manifest["line_identity"] == [
        dict(item) for item in model.line_identity
    ]
    assert manifest["source_rate_semantics"] == (
        "pre_dead_time_detector_pulse_rate_at_1m"
    )
    assert manifest["discrepancy_training_ready"] is True
    assert manifest["rate_scale_mixture"]["weighted_mean"] == 1.0


def test_runtime_factory_reconstructs_and_authenticates_schema_three() -> None:
    """Live/replay construction must reject every manifest identity mismatch."""
    model = approved_full_spectrum_model()
    manifest = model.manifest_payload()
    runtime = {
        "source_rate_model": "detector_cps_1m",
        "energy_min_keV": manifest["energy_min_keV"],
        "energy_max_keV": manifest["energy_max_keV"],
        "bin_width_keV": manifest["bin_width_keV"],
        "energy_bin_count": manifest["energy_bin_count"],
        "background_rate_cps": manifest["background_rate_cps"],
        "dead_time_tau_s": manifest["dead_time_tau_s"],
        "full_spectrum_generative_model": manifest,
        "full_spectrum_contract_hash_sha256": (
            model.contract_hash_sha256
        ),
    }
    reconstructed = geometry_conditioned_model_from_runtime_config(runtime)
    assert reconstructed.production_ready is True
    assert reconstructed.contract_hash_sha256 == model.contract_hash_sha256
    corrupted = json.loads(json.dumps(runtime))
    corrupted["full_spectrum_generative_model"]["background_rate_cps"] = 6.0
    with pytest.raises(ValueError, match="reconstruct"):
        geometry_conditioned_model_from_runtime_config(corrupted)
    corrupted = json.loads(json.dumps(runtime))
    corrupted["full_spectrum_contract_hash_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        geometry_conditioned_model_from_runtime_config(corrupted)


@pytest.mark.parametrize(
    "field_name",
    (
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
        "energy_bin_count",
        "background_rate_cps",
        "dead_time_tau_s",
    ),
)
def test_runtime_factory_rejects_numeric_string_scalars(
    field_name: str,
) -> None:
    """Runtime identity scalars must retain their exact JSON numeric types."""
    model = approved_full_spectrum_model()
    manifest = model.manifest_payload()
    runtime = {
        "source_rate_model": "detector_cps_1m",
        "energy_min_keV": manifest["energy_min_keV"],
        "energy_max_keV": manifest["energy_max_keV"],
        "bin_width_keV": manifest["bin_width_keV"],
        "energy_bin_count": manifest["energy_bin_count"],
        "background_rate_cps": manifest["background_rate_cps"],
        "dead_time_tau_s": manifest["dead_time_tau_s"],
        "full_spectrum_generative_model": manifest,
        "full_spectrum_contract_hash_sha256": model.contract_hash_sha256,
    }
    runtime[field_name] = str(runtime[field_name])

    with pytest.raises(ValueError, match=field_name):
        geometry_conditioned_model_from_runtime_config(runtime)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("dead_time_tau_s",), "5.813e-09"),
        (("background_rate_cps",), "5.0"),
        (("mark_concentration_source",), "300.0"),
        (("rate_scale_mixture", "nodes", 0), "0.9"),
        (("rate_scale_mixture", "weights", 0), "0.25"),
        (("line_identity", 0, "isotope"), 137),
    ),
)
def test_manifest_factory_rejects_scalar_type_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """File-backed model fields must not acquire meaning through coercion."""
    payload = json.loads(
        json.dumps(approved_full_spectrum_model().manifest_payload())
    )
    target: object = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        GeometryConditionedSpectralModel.from_manifest_payload(payload)


@pytest.mark.parametrize("field_name", ("value", "threshold"))
def test_validation_metrics_reject_numeric_strings(field_name: str) -> None:
    """Validation evidence must retain exact JSON-number semantics."""
    approved = approved_full_spectrum_model()
    validation = approved.manifest_payload()["validation"]
    metric_id = next(iter(validation["metrics"]))
    validation["metrics"][metric_id][field_name] = str(
        validation["metrics"][metric_id][field_name]
    )
    candidate = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=approved.dead_time_tau_s,
        background_rate_cps=approved.background_rate_cps,
        rate_scale_nodes_j=approved.rate_scale_nodes_j,
        rate_scale_weights_j=approved.rate_scale_weights_j,
        mark_concentration_source=approved.mark_concentration_source,
        discrepancy_training_manifest=approved.discrepancy_training_manifest,
        validation_manifest=validation,
        additive_scatter_response=approved.additive_scatter_response,
    )

    assert candidate.production_ready is False


def test_runtime_factory_authenticates_file_backed_schema_three(
    tmp_path: Path,
) -> None:
    """File-backed production assets must match both byte and model hashes."""
    model = approved_full_spectrum_model()
    manifest = model.manifest_payload()
    raw_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    asset_path = tmp_path / "approved.json"
    asset_path.write_bytes(raw_bytes)
    runtime = {
        "source_rate_model": "detector_cps_1m",
        "energy_min_keV": manifest["energy_min_keV"],
        "energy_max_keV": manifest["energy_max_keV"],
        "bin_width_keV": manifest["bin_width_keV"],
        "energy_bin_count": manifest["energy_bin_count"],
        "background_rate_cps": manifest["background_rate_cps"],
        "dead_time_tau_s": manifest["dead_time_tau_s"],
        "full_spectrum_generative_model_path": asset_path.name,
        "full_spectrum_generative_model_file_sha256": (
            hashlib.sha256(raw_bytes).hexdigest()
        ),
        "full_spectrum_contract_hash_sha256": (
            model.contract_hash_sha256
        ),
    }
    reconstructed = geometry_conditioned_model_from_runtime_config(
        runtime,
        run_root=tmp_path,
    )
    assert reconstructed.contract_hash_sha256 == model.contract_hash_sha256
    runtime["full_spectrum_generative_model_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file SHA-256"):
        geometry_conditioned_model_from_runtime_config(
            runtime,
            run_root=tmp_path,
        )


def test_runtime_factory_rejects_duplicate_model_asset_keys(
    tmp_path: Path,
) -> None:
    """A pinned asset must not acquire last-key-wins model semantics."""
    model = approved_full_spectrum_model()
    manifest = model.manifest_payload()
    assert manifest["schema_version"] == 3
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_bytes = (
        '{"schema_version":3,' + canonical.removeprefix("{")
    ).encode()
    asset_path = tmp_path / "duplicate.json"
    asset_path.write_bytes(raw_bytes)
    runtime = {
        "source_rate_model": "detector_cps_1m",
        "energy_min_keV": manifest["energy_min_keV"],
        "energy_max_keV": manifest["energy_max_keV"],
        "bin_width_keV": manifest["bin_width_keV"],
        "energy_bin_count": manifest["energy_bin_count"],
        "background_rate_cps": manifest["background_rate_cps"],
        "dead_time_tau_s": manifest["dead_time_tau_s"],
        "full_spectrum_generative_model_path": asset_path.name,
        "full_spectrum_generative_model_file_sha256": (
            hashlib.sha256(raw_bytes).hexdigest()
        ),
        "full_spectrum_contract_hash_sha256": model.contract_hash_sha256,
    }

    with pytest.raises(ValueError, match="canonical JSON"):
        geometry_conditioned_model_from_runtime_config(
            runtime,
            run_root=tmp_path,
        )
