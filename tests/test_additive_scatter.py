"""Tests for the physical additive noncollided transport response."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID,
    AdditiveNoncollidedTransportResponse,
    fit_additive_noncollided_transport_response,
    physical_scatter_basis_numpy,
    physical_scatter_basis_torch,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
)


def _training_manifest() -> dict[str, object]:
    """Return deterministic designated-training provenance for unit tests."""
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "artifact_sha256_by_scene": {
            "2026072701": "a" * 64,
            "2026072702": "b" * 64,
            "2026072703": "c" * 64,
        },
        "pair_ids_by_scene": {
            "2026072701": list(range(64)),
            "2026072702": list(range(64)),
            "2026072703": list(range(64)),
        },
        "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
        "selection_objective": (
            "leave_one_training_scene_out_weighted_log1p_mse"
        ),
    }


def _response() -> AdditiveNoncollidedTransportResponse:
    """Return a valid nonzero additive response for deterministic tests."""
    manifest = _training_manifest()
    manifest.update(
        {
            "fit_sample_count": 210,
            "loso_scene_ids": [
                str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            ],
            "candidate_validation_scores": {
                format(value, ".12g"): (
                    0.5 if value == 0.1 else 1.0 + float(value)
                )
                for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            },
            "selected_validation_score": 0.5,
            "selected_ridge_lambda": 0.1,
            "selection_completed": True,
        }
    )
    return AdditiveNoncollidedTransportResponse(
        coefficients=(0.8, 0.5, 0.4, 0.2, 0.1, 0.05, 0.025),
        ridge_lambda=0.1,
        training_manifest=manifest,
    )


def test_physical_scatter_basis_matches_torch() -> None:
    """The batched NumPy and Torch physical bases must be equivalent."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(418)
    shape = (3, 4, 5)
    tau_fe = rng.uniform(0.0, 4.0, shape)
    tau_pb = rng.uniform(0.0, 5.0, shape)
    tau_obstacle = rng.uniform(0.0, 3.0, shape)
    tau_obstacle_compton = tau_obstacle * rng.uniform(0.1, 0.9, shape)
    distance = rng.uniform(0.2, 30.0, shape)
    energy = rng.uniform(250.0, 1600.0, shape)
    mu_fe = rng.uniform(0.35, 1.5, shape)
    mu_pb = rng.uniform(0.6, 3.0, shape)
    numpy_basis = physical_scatter_basis_numpy(
        tau_fe=tau_fe,
        tau_pb=tau_pb,
        tau_obstacle=tau_obstacle,
        tau_obstacle_compton=tau_obstacle_compton,
        distance_m=distance,
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
    )
    torch_basis = physical_scatter_basis_torch(
        tau_fe=torch.as_tensor(tau_fe, dtype=torch.float64),
        tau_pb=torch.as_tensor(tau_pb, dtype=torch.float64),
        tau_obstacle=torch.as_tensor(tau_obstacle, dtype=torch.float64),
        tau_obstacle_compton=torch.as_tensor(
            tau_obstacle_compton,
            dtype=torch.float64,
        ),
        distance_m=torch.as_tensor(distance, dtype=torch.float64),
        energy_keV=torch.as_tensor(energy, dtype=torch.float64),
        mu_fe_cm_inv=torch.as_tensor(mu_fe, dtype=torch.float64),
        mu_pb_cm_inv=torch.as_tensor(mu_pb, dtype=torch.float64),
    )
    assert numpy_basis.shape == shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
    np.testing.assert_allclose(
        torch_basis.detach().cpu().numpy(),
        numpy_basis,
        rtol=2.0e-13,
        atol=2.0e-15,
    )


def test_additive_kernel_is_nonnegative_and_cpu_torch_equivalent() -> None:
    """The response must add nonnegative scatter without changing direct counts."""
    torch = pytest.importorskip("torch")
    response = _response()
    rng = np.random.default_rng(67)
    unattenuated = rng.uniform(0.1, 5.0, (2, 3, 4))
    uncollided = unattenuated * rng.uniform(0.0, 1.0, unattenuated.shape)
    basis = rng.uniform(
        0.0,
        0.5,
        unattenuated.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),),
    )
    numpy_total = response.total_kernel_numpy(
        unattenuated,
        uncollided,
        basis,
    )
    torch_total = response.total_kernel_torch(
        torch.as_tensor(unattenuated, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(basis, dtype=torch.float64),
    )
    assert np.all(numpy_total >= uncollided)
    assert np.any(numpy_total > uncollided)
    np.testing.assert_allclose(
        torch_total.detach().cpu().numpy(),
        numpy_total,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_fit_uses_only_declared_training_arrays_and_global_coefficients() -> None:
    """Training-scene cross validation must recover one global nonzero model."""
    rng = np.random.default_rng(82)
    sample_count = 210
    features = rng.uniform(
        0.0,
        0.5,
        (sample_count, len(ADDITIVE_SCATTER_FEATURE_ORDER)),
    )
    true_coefficients = np.asarray(
        (0.6, 0.4, 0.2, 0.1, 0.05, 0.03, 0.01),
        dtype=np.float64,
    )
    targets = features @ true_coefficients
    scene_ids = np.repeat(
        ("2026072701", "2026072702", "2026072703"),
        sample_count // 3,
    )
    response = fit_additive_noncollided_transport_response(
        features,
        targets,
        np.ones(sample_count, dtype=np.float64),
        scene_ids,
        training_manifest=_training_manifest(),
    )
    assert np.all(np.asarray(response.coefficients) >= 0.0)
    assert np.any(np.asarray(response.coefficients) > 0.0)
    assert "holdout" not in response.to_payload()["training"]
    np.testing.assert_allclose(
        features @ np.asarray(response.coefficients),
        targets,
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_payload_authentication_rejects_legacy_pair_categorical_model() -> None:
    """Old pair-indexed multiplicative calibration must not enter production."""
    response = _response()
    payload = response.to_payload()
    assert AdditiveNoncollidedTransportResponse.from_payload(
        payload
    ).to_payload() == payload
    legacy = {
        "schema_version": 3,
        "model": "legacy_pair_categorical_regression",
        "scale_by_pair": {"0": 2.0},
    }
    with pytest.raises(ValueError, match="schema"):
        AdditiveNoncollidedTransportResponse.from_payload(legacy)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("schema_version",), True),
        (
            ("artifact_sha256_by_scene", "2026072701"),
            int("a" * 64, 16),
        ),
        (("pair_ids_by_scene", "2026072701", 1), True),
        (("fit_sample_count",), "210"),
        (("fit_sample_count",), True),
        (("selected_validation_score",), "0.5"),
        (("selected_validation_score",), True),
        (("selected_ridge_lambda",), "0.1"),
        (("selected_ridge_lambda",), True),
        (("candidate_validation_scores", "0.1"), "0.5"),
        (("candidate_validation_scores", "0.1"), True),
    ),
)
def test_training_manifest_rejects_json_scalar_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """Training evidence must preserve exact JSON scalar types."""
    response = _response()
    manifest = copy.deepcopy(dict(response.training_manifest))
    target: object = manifest
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="training provenance"):
        AdditiveNoncollidedTransportResponse(
            coefficients=response.coefficients,
            ridge_lambda=response.ridge_lambda,
            training_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("schema_version",), True),
        (("coefficients", 0), "0.8"),
        (("coefficients", 0), True),
        (("selected_ridge_lambda",), "0.1"),
        (("selected_ridge_lambda",), True),
        (("ridge_lambda_grid", 5), True),
        (("contract_hash_sha256",), int("a" * 64, 16)),
    ),
)
def test_payload_rejects_json_scalar_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """External response payloads must reject strings and booleans as numbers."""
    payload = copy.deepcopy(_response().to_payload())
    target: object = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="schema"):
        AdditiveNoncollidedTransportResponse.from_payload(payload)


@pytest.mark.parametrize(
    "tamper",
    ("holdout_seed", "missing_pair", "extra_key"),
)
def test_training_provenance_rejects_leakage_and_schema_tampering(
    tamper: str,
) -> None:
    """Only the designated training all-64 LOSO fit may enter production."""
    manifest = copy.deepcopy(dict(_response().training_manifest))
    if tamper == "holdout_seed":
        manifest["training_scene_seeds"][-1] = 2026072791
    elif tamper == "missing_pair":
        manifest["pair_ids_by_scene"]["2026072701"] = list(range(63))
    else:
        manifest["holdout_metrics"] = {"score": 0.0}

    with pytest.raises(ValueError, match="training provenance"):
        AdditiveNoncollidedTransportResponse(
            coefficients=(0.8, 0.5, 0.4, 0.2, 0.1, 0.05, 0.025),
            ridge_lambda=0.1,
            training_manifest=manifest,
        )


@pytest.mark.parametrize(
    "basis_case",
    ("air", "shield", "obstacle"),
)
def test_nonzero_physical_opportunities_cannot_silently_return_zero_scatter(
    basis_case: str,
) -> None:
    """Air, shield, and obstacle opportunities must produce positive scatter."""
    response = _response()
    shape = (1,)
    tau_fe = np.zeros(shape)
    tau_pb = np.zeros(shape)
    tau_obstacle = np.zeros(shape)
    tau_obstacle_compton = np.zeros(shape)
    distance = np.zeros(shape)
    if basis_case == "air":
        distance[...] = 10.0
    elif basis_case == "shield":
        tau_fe[...] = 1.0
    else:
        tau_obstacle[...] = 1.0
        tau_obstacle_compton[...] = 0.5
    basis = physical_scatter_basis_numpy(
        tau_fe=tau_fe,
        tau_pb=tau_pb,
        tau_obstacle=tau_obstacle,
        tau_obstacle_compton=tau_obstacle_compton,
        distance_m=distance,
        energy_keV=np.full(shape, 662.0),
        mu_fe_cm_inv=np.full(shape, 0.58),
        mu_pb_cm_inv=np.full(shape, 1.29),
    )
    assert float(response.scatter_fraction_numpy(basis)[0]) > 0.0
