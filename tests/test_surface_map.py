"""Tests for PF-independent regularized Poisson surface-map reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from pf.surface_map import (
    SurfaceMapConfig,
    _flatten_response,
    aggregate_contiguous_poisson_bins,
    contiguous_poisson_bin_aggregation,
    evaluate_surface_map_objective,
    fit_surface_map_poisson,
)


def test_surface_map_recovers_piecewise_smooth_density() -> None:
    """Batched Poisson L1+TV fitting should recover a synthetic surface map."""
    response = np.asarray(
        [
            [1.0, 0.1, 0.05],
            [0.8, 0.2, 0.1],
            [0.1, 0.9, 0.2],
            [0.2, 0.8, 0.1],
            [0.1, 0.2, 1.0],
            [0.05, 0.1, 0.8],
        ],
        dtype=float,
    )
    areas = np.asarray([1.0, 1.0, 0.5], dtype=float)
    truth_density = np.asarray([40.0, 40.0, 8.0], dtype=float)
    background = np.full(response.shape[0], 2.0, dtype=float)
    observed = background + response @ (areas * truth_density)

    result = fit_surface_map_poisson(
        observed,
        response,
        areas,
        adjacency_edges=np.asarray([[0, 1], [1, 2]], dtype=int),
        adjacency_weights=np.ones(2, dtype=float),
        background=background,
        config=SurfaceMapConfig(
            l1_weight=1.0e-3,
            tv_weight=2.0e-3,
            max_iterations=5000,
            tolerance=2.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.densities_cps_1m_m2[:, 0] == pytest.approx(
        truth_density,
        rel=0.02,
        abs=0.15,
    )
    assert result.integrated_strengths_cps_1m[:, 0] == pytest.approx(
        areas * truth_density,
        rel=0.02,
        abs=0.15,
    )
    assert result.deviance < 1.0e-2
    assert result.kkt_residual < 1.0e-4


def test_contiguous_poisson_aggregation_preserves_full_spectrum_mean() -> None:
    """Shared grouping must sum every count and model-mean bin exactly once."""
    observed = np.asarray(
        [[2.0, 4.0, 1.0, 3.0, 5.0, 7.0, 6.0], [1.0, 0.0, 2.0, 8.0, 4.0, 3.0, 9.0]],
        dtype=float,
    )
    background = np.asarray(
        [[0.1, 0.2, 0.1, 0.3, 0.2, 0.4, 0.1], [0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.4]],
        dtype=float,
    )
    response = np.arange(2 * 7 * 3 * 2, dtype=float).reshape(2, 7, 3, 2) / 100.0
    strengths = np.asarray([[3.0, 2.0], [1.0, 4.0], [2.5, 0.5]], dtype=float)
    expected = background + np.einsum("mbci,ci->mb", response, strengths)
    aggregation = contiguous_poisson_bin_aggregation(7, 3)

    aggregated_observed = aggregate_contiguous_poisson_bins(
        observed,
        aggregation,
        axis=1,
    )
    aggregated_background = aggregate_contiguous_poisson_bins(
        background,
        aggregation,
        axis=1,
    )
    aggregated_response = aggregate_contiguous_poisson_bins(
        response,
        aggregation,
        axis=1,
    )
    aggregated_expected = aggregated_background + np.einsum(
        "mgci,ci->mg",
        aggregated_response,
        strengths,
    )

    np.testing.assert_array_equal(aggregation.group_starts, [0, 2, 4])
    np.testing.assert_array_equal(aggregation.group_ends, [2, 4, 7])
    np.testing.assert_array_equal(aggregation.group_widths, [2, 2, 3])
    np.testing.assert_allclose(
        aggregated_expected,
        aggregate_contiguous_poisson_bins(expected, aggregation, axis=1),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(aggregated_observed, axis=1),
        np.sum(observed, axis=1),
    )


def test_surface_map_zero_signal_stays_zero() -> None:
    """A zero-count observation should not create a regularized surface source."""
    result = fit_surface_map_poisson(
        np.zeros(3, dtype=float),
        np.eye(3, dtype=float),
        np.ones(3, dtype=float),
        adjacency_edges=np.asarray([[0, 1], [1, 2]], dtype=int),
        config=SurfaceMapConfig(l1_weight=0.1, tv_weight=0.2),
    )

    assert result.converged is True
    assert np.array_equal(result.densities_cps_1m_m2, np.zeros((3, 1)))
    assert np.array_equal(result.integrated_strengths_cps_1m, np.zeros((3, 1)))
    assert result.deviance == pytest.approx(0.0, abs=1.0e-10)


def test_surface_map_area_semantics_separate_density_and_strength() -> None:
    """Equal integrated sources on unequal patches should have inverse-area density."""
    result = fit_surface_map_poisson(
        np.asarray([41.0, 41.0], dtype=float),
        np.eye(2, dtype=float),
        np.asarray([2.0, 0.5], dtype=float),
        background=1.0,
        config=SurfaceMapConfig(
            max_iterations=4000,
            tolerance=1.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.integrated_strengths_cps_1m[:, 0] == pytest.approx(
        [40.0, 40.0],
        rel=1.0e-5,
    )
    assert result.densities_cps_1m_m2[:, 0] == pytest.approx(
        [20.0, 80.0],
        rel=1.0e-5,
    )


def test_surface_map_profiles_non_negative_nuisance_without_fake_source() -> None:
    """An unpenalized nuisance basis should absorb common leakage instead of a source."""
    source_response = np.asarray([[1.0], [0.8], [0.4], [0.2]], dtype=float)
    nuisance_response = source_response.copy()
    observed = 1.0 + nuisance_response[:, 0] * 100.0

    result = fit_surface_map_poisson(
        observed,
        source_response,
        np.ones(1, dtype=float),
        background=1.0,
        nuisance_response=nuisance_response,
        config=SurfaceMapConfig(
            l1_weight=1.0,
            max_iterations=4000,
            tolerance=1.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.densities_cps_1m_m2[0, 0] == pytest.approx(0.0, abs=1.0e-5)
    assert result.nuisance_coefficients == pytest.approx([100.0], rel=1.0e-5)
    assert result.deviance < 1.0e-8


def test_surface_map_tensor_batch_matches_flattened_batch() -> None:
    """Spectrum-tensor fitting should equal the same batched flattened problem."""
    response = np.asarray(
        [
            [
                [[1.0, 0.1], [0.2, 0.0]],
                [[0.8, 0.2], [0.1, 0.1]],
                [[0.2, 0.7], [0.0, 0.2]],
            ],
            [
                [[0.4, 0.1], [0.8, 0.0]],
                [[0.1, 0.2], [0.7, 0.1]],
                [[0.0, 0.4], [0.2, 0.9]],
            ],
        ],
        dtype=float,
    )
    areas = np.asarray([2.0, 0.5], dtype=float)
    density = np.asarray([[12.0, 4.0], [3.0, 18.0]], dtype=float)
    nuisance_response = np.asarray(
        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.2]],
        dtype=float,
    )
    nuisance_coefficient = 7.0
    background = np.full((2, 3), 1.5, dtype=float)
    observed = (
        background
        + np.einsum("mbci,ci->mb", response, density * areas[:, None])
        + nuisance_response * nuisance_coefficient
    )
    config = SurfaceMapConfig(
        l1_weight=1.0e-3,
        tv_weight=2.0e-3,
        nuisance_l2_weight=1.0e-4,
        max_iterations=2500,
        tolerance=1.0e-7,
        objective_tolerance=1.0e-8,
    )
    common_kwargs = {
        "patch_areas_m2": areas,
        "adjacency_edges": np.asarray([[0, 1], [1, 0]], dtype=int),
        "adjacency_weights": np.asarray([0.75, 0.75], dtype=float),
        "background": background,
        "nuisance_response": nuisance_response[..., None],
        "config": config,
    }

    tensor_result = fit_surface_map_poisson(
        observed,
        response,
        **common_kwargs,
    )
    flat_result = fit_surface_map_poisson(
        observed.reshape(-1),
        response.reshape(observed.size, 2, 2),
        patch_areas_m2=areas,
        adjacency_edges=common_kwargs["adjacency_edges"],
        adjacency_weights=common_kwargs["adjacency_weights"],
        background=background.reshape(-1),
        nuisance_response=nuisance_response.reshape(-1, 1),
        config=config,
    )

    assert tensor_result.densities_cps_1m_m2 == pytest.approx(
        flat_result.densities_cps_1m_m2,
        rel=1.0e-11,
        abs=1.0e-11,
    )
    assert tensor_result.nuisance_coefficients == pytest.approx(
        flat_result.nuisance_coefficients,
        rel=1.0e-11,
        abs=1.0e-11,
    )
    assert tensor_result.objective == pytest.approx(flat_result.objective, rel=1.0e-12)


def test_surface_map_objective_matches_manual_oracle() -> None:
    """The public objective should match a direct Poisson, L1, TV, and nuisance oracle."""
    observed = np.asarray([7.0, 11.0], dtype=float)
    response = np.asarray(
        [
            [[1.0, 0.5], [0.2, 0.1]],
            [[0.1, 0.3], [0.8, 0.4]],
        ],
        dtype=float,
    )
    areas = np.asarray([2.0, 0.5], dtype=float)
    density = np.asarray([[3.0, 1.0], [2.0, 4.0]], dtype=float)
    nuisance_response = np.asarray([[0.2], [0.4]], dtype=float)
    nuisance = np.asarray([2.5], dtype=float)
    background = np.asarray([1.0, 1.5], dtype=float)
    config = SurfaceMapConfig(
        l1_weight=0.3,
        tv_weight=0.7,
        nuisance_l1_weight=0.2,
        nuisance_l2_weight=0.1,
    )

    objective = evaluate_surface_map_objective(
        observed,
        response,
        areas,
        density,
        adjacency_edges=np.asarray([[0, 1]], dtype=int),
        adjacency_weights=np.asarray([1.5], dtype=float),
        background=background,
        nuisance_response=nuisance_response,
        nuisance_coefficients=nuisance,
        config=config,
    )

    expected = (
        background
        + response.reshape(2, -1) @ (density * areas[:, None]).reshape(-1)
        + nuisance_response[:, 0] * nuisance[0]
    )
    poisson_nll = float(np.sum(expected - observed * np.log(expected)))
    l1_penalty = 0.3 * float(np.sum(density * areas[:, None]))
    tv_penalty = 0.7 * 1.5 * float(np.sum(np.abs(density[1] - density[0])))
    nuisance_penalty = 0.2 * nuisance[0] + 0.5 * 0.1 * nuisance[0] ** 2

    assert objective.poisson_nll == pytest.approx(poisson_nll)
    assert objective.l1_penalty == pytest.approx(l1_penalty)
    assert objective.tv_penalty == pytest.approx(tv_penalty)
    assert objective.nuisance_penalty == pytest.approx(nuisance_penalty)
    assert objective.total == pytest.approx(
        poisson_nll + l1_penalty + tv_penalty + nuisance_penalty
    )


def test_non_negative_response_flattening_keeps_a_view() -> None:
    """Validation should not duplicate an already non-negative response tensor."""
    response = np.arange(24, dtype=float).reshape(2, 3, 4)

    matrix, isotope_count = _flatten_response((2,), response, patch_count=3)

    assert isotope_count == 4
    assert matrix.shape == (2, 12)
    assert np.shares_memory(matrix, response)
