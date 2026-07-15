"""Tests for all-history sparse Poisson evidence scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pf.sparse_evidence import (
    SparsePoissonEvidenceConfig,
    fit_joint_sparse_poisson_evidence,
    fit_sparse_poisson_evidence,
    fit_sparse_poisson_spectral_evidence,
    joint_sparse_poisson_evidence_to_diagnostics,
    refine_sparse_poisson_evidence_offgrid,
    sparse_poisson_ambiguity_diagnostics,
    sparse_poisson_evidence_to_diagnostics,
)
from planning.dss_pp import DSSPPConfig, _cardinality_evidence_gap_pressure


def test_sparse_poisson_evidence_selects_two_sources() -> None:
    """BIC evidence should recover two independent response columns."""
    response = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.1],
            [1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.1, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    background = np.full(response.shape[0], 2.0, dtype=float)
    counts = background + response[:, [0, 2]] @ np.asarray([90.0, 70.0])

    evidence = fit_sparse_poisson_evidence(
        counts,
        response,
        background=background,
        config=SparsePoissonEvidenceConfig(
            max_sources=4,
            parameter_count_per_source=2,
            holdout_stride=3,
        ),
    )

    assert evidence.available is True
    assert evidence.selected_count == 2
    assert set(evidence.selected_indices) == {0, 2}
    assert evidence.bic_gap_to_simpler > 0.0
    assert evidence.bic_margin_to_runner_up > 0.0
    assert np.isfinite(evidence.heldout_deviance_by_count[2])


def test_sparse_poisson_evidence_rejects_duplicate_column() -> None:
    """Correlation pruning and BIC should avoid double-counting duplicate columns."""
    base = np.asarray([1.0, 0.8, 1.2, 0.9, 1.1, 0.7], dtype=float)
    response = np.column_stack(
        [
            base,
            1.05 * base,
            np.asarray([0.0, 0.1, 0.0, 0.2, 0.0, 0.1], dtype=float),
        ]
    )
    counts = 3.0 + response[:, 0] * 120.0

    evidence = fit_sparse_poisson_evidence(
        counts,
        response,
        background=3.0,
        config=SparsePoissonEvidenceConfig(
            max_sources=3,
            correlation_prune_threshold=0.99,
            parameter_count_per_source=2,
        ),
    )

    assert evidence.selected_count == 1
    assert evidence.selected_indices in ((0,), (1,))
    assert evidence.selected_max_response_correlation == pytest.approx(0.0)


def test_sparse_poisson_diagnostics_are_json_safe() -> None:
    """Evidence diagnostics should serialize without NumPy objects."""
    response = np.eye(3, dtype=float)
    counts = np.asarray([20.0, 0.0, 0.0], dtype=float)

    evidence = fit_sparse_poisson_evidence(
        counts,
        response,
        config=SparsePoissonEvidenceConfig(max_sources=2),
    )
    payload = sparse_poisson_evidence_to_diagnostics(evidence)

    json.dumps(payload, allow_nan=False)
    assert payload["method"] == "all_history_sparse_poisson"
    assert payload["selected_count"] == 1


def test_sparse_poisson_spectral_evidence_uses_bin_tensor_metadata() -> None:
    """Spectrum-bin evidence should select candidate/isotope response columns."""
    response = np.zeros((2, 3, 2, 2), dtype=float)
    response[:, :, 0, 0] = np.asarray([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    response[:, :, 1, 1] = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]])
    response[:, :, 0, 1] = np.asarray([[0.0, 0.4, 0.0], [0.0, 0.3, 0.0]])
    response[:, :, 1, 0] = np.asarray([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])
    background = np.full((2, 3), 2.0, dtype=float)
    spectra = background + response[:, :, 0, 0] * 90.0 + response[:, :, 1, 1] * 70.0

    evidence = fit_sparse_poisson_spectral_evidence(
        spectra,
        response,
        background_spectrum=background,
        isotope_names=("Cs-137", "Eu-154"),
        config=SparsePoissonEvidenceConfig(
            max_sources=3,
            parameter_count_per_source=1,
            holdout_stride=2,
        ),
    )

    assert evidence.available is True
    assert evidence.method == "spectral_bin_sparse_poisson"
    assert evidence.selected_count == 2
    selected = {(item["candidate_index"], item["isotope"]) for item in evidence.selected_column_metadata}
    assert selected == {(0, "Cs-137"), (1, "Eu-154")}


def test_sparse_poisson_evidence_profiles_low_dimensional_nuisance() -> None:
    """Nuisance basis columns should absorb common leakage instead of fake sources."""
    response = np.asarray([[1.0], [0.9], [0.0], [0.0]], dtype=float)
    nuisance = np.asarray([[1.0], [0.9], [0.0], [0.0]], dtype=float)
    counts = 3.0 + nuisance[:, 0] * 80.0

    evidence = fit_sparse_poisson_evidence(
        counts,
        response,
        background=3.0,
        nuisance_response_matrix=nuisance,
        config=SparsePoissonEvidenceConfig(
            max_sources=1,
            parameter_count_per_source=2,
            nuisance_parameter_count=1,
        ),
    )

    assert evidence.selected_count == 0
    assert evidence.selected_nuisance_strengths[0] == pytest.approx(80.0, rel=0.2)


def test_joint_sparse_poisson_evidence_selects_multi_isotope_cardinality() -> None:
    """Joint evidence should compare isotope cardinalities in one sparse model."""
    cs_response = np.asarray(
        [[1.0, 0.0], [0.8, 0.0], [0.0, 0.1], [0.0, 0.0]],
        dtype=float,
    )
    co_response = np.asarray(
        [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.0, 0.9]],
        dtype=float,
    )
    counts = 2.0 + cs_response[:, 0] * 100.0 + co_response[:, 1] * 75.0

    evidence = fit_joint_sparse_poisson_evidence(
        counts,
        {"Cs-137": cs_response, "Co-60": co_response},
        max_sources_by_isotope={"Cs-137": 1, "Co-60": 1},
        background=2.0,
        config=SparsePoissonEvidenceConfig(
            max_sources=2,
            parameter_count_per_source=1,
        ),
    )
    payload = joint_sparse_poisson_evidence_to_diagnostics(evidence)

    assert evidence.available is True
    assert evidence.selected_counts_by_isotope == {"Cs-137": 1, "Co-60": 1}
    assert payload["method"] == "joint_multi_isotope_sparse_poisson"
    assert payload["bic_margin_to_runner_up"] > 0.0


def test_joint_sparse_poisson_prefilter_preserves_original_indices() -> None:
    """Joint candidate prefiltering should report original dictionary indices."""
    cs_response = np.asarray(
        [
            [0.1, 0.2, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.1, 0.2, 0.9],
            [0.0, 0.1, 0.0, 0.0, 0.8],
        ],
        dtype=float,
    )
    co_response = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    counts = 1.0 + cs_response[:, 4] * 80.0

    evidence = fit_joint_sparse_poisson_evidence(
        counts,
        {"Cs-137": cs_response, "Co-60": co_response},
        max_sources_by_isotope={"Cs-137": 1, "Co-60": 0},
        background=1.0,
        config=SparsePoissonEvidenceConfig(
            max_sources=1,
            candidate_limit=2,
            parameter_count_per_source=1,
        ),
    )

    assert evidence.available is True
    assert evidence.selected_counts_by_isotope["Cs-137"] == 1
    assert evidence.selected_indices_by_isotope["Cs-137"] == (4,)


def test_offgrid_refinement_improves_profile_likelihood_position() -> None:
    """Off-grid refinement should improve selected grid positions continuously."""
    detector_x = np.asarray([-1.0, -0.25, 0.5, 1.0], dtype=float)
    truth_x = 0.2

    def response_at_positions(positions: np.ndarray) -> np.ndarray:
        """Return a batched one-dimensional surface response."""
        x = np.asarray(positions, dtype=float).reshape(-1)
        return np.exp(-((detector_x[:, None] - x[None, :]) ** 2) / 0.25)

    counts = 1.0 + response_at_positions(np.asarray([[truth_x]], dtype=float))[:, 0] * 120.0

    refined = refine_sparse_poisson_evidence_offgrid(
        counts,
        np.asarray([[0.65]], dtype=float),
        response_at_positions,
        background=1.0,
        bounds=((-1.0, 1.0),),
        config=SparsePoissonEvidenceConfig(max_sources=1, parameter_count_per_source=2),
        max_iter=80,
    )

    assert refined.available is True
    assert abs(refined.positions[0][0] - truth_x) < abs(0.65 - truth_x)
    assert refined.improvement_log_likelihood > 0.0


def test_sparse_poisson_ambiguity_reports_correlated_interval() -> None:
    """Highly correlated response columns should produce an ambiguity cluster."""
    base = np.asarray([1.0, 0.8, 1.2, 0.9], dtype=float)
    response = np.column_stack([base, 1.01 * base, np.asarray([0.0, 0.0, 1.0, 1.0])])
    counts = 2.0 + base * 60.0
    positions = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [3.0, 0.0, 0.0]])
    evidence = fit_sparse_poisson_evidence(
        counts,
        response,
        background=2.0,
        config=SparsePoissonEvidenceConfig(
            max_sources=1,
            correlation_prune_threshold=0.999,
        ),
    )

    clusters = sparse_poisson_ambiguity_diagnostics(
        evidence,
        response,
        candidate_positions=positions,
        correlation_threshold=0.99,
    )

    assert clusters
    assert clusters[0]["identifiable"] is False
    assert clusters[0]["position_max"][0] == pytest.approx(0.2)


def test_cardinality_evidence_pressure_uses_unresolved_gap() -> None:
    """DSS-PP pressure should grow when sparse evidence gaps are below target."""

    class _Estimator:
        """Minimal estimator exposing sparse evidence diagnostics."""

        def sparse_poisson_evidence_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return one unresolved sparse-evidence payload."""
            return {
                "Cs-137": {
                    "available": True,
                    "selected_count": 2,
                    "model_order_ready": False,
                    "criterion_margin_to_runner_up": 2.0,
                    "criterion_margin_to_simpler": 3.0,
                    "bic_gap_to_next_count": 4.0,
                }
            }

    pressure = _cardinality_evidence_gap_pressure(
        _Estimator(),  # type: ignore[arg-type]
        DSSPPConfig(cardinality_evidence_gap_target=10.0),
    )

    assert pressure == pytest.approx(0.8)


def test_cardinality_evidence_pressure_keeps_unready_high_gap() -> None:
    """DSS-PP pressure should not vanish when readiness is explicitly unresolved."""

    class _Estimator:
        """Minimal estimator exposing unready high-gap sparse evidence."""

        def sparse_poisson_evidence_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return high-gap evidence that still lacks model-order readiness."""
            return {
                "Cs-137": {
                    "available": True,
                    "selected_count": 2,
                    "model_order_ready": False,
                    "criterion_margin_to_runner_up": 20.0,
                    "criterion_margin_to_simpler": 30.0,
                    "bic_gap_to_next_count": 25.0,
                }
            }

    pressure = _cardinality_evidence_gap_pressure(
        _Estimator(),  # type: ignore[arg-type]
        DSSPPConfig(cardinality_evidence_gap_target=10.0),
    )

    assert pressure == pytest.approx(1.0)
