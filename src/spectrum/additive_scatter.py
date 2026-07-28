"""Physical additive noncollided-response model for full-spectrum inference.

The model predicts detector-entry scatter before detector-response marking,
background injection, and electronics dead time.  It is deliberately additive:

``total = uncollided + unattenuated_geometric * scatter_fraction``.

Every learned coefficient is global and nonnegative.  The feature basis uses
only known interaction probabilities and contains no shield-pair, scene-seed,
source-index, or isotope categorical term.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


ELECTRON_REST_ENERGY_KEV = 510.99895
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
AIR_DENSITY_G_CM3 = 1.225e-3
AIR_EFFECTIVE_Z_OVER_A = 0.49919
IRON_DENSITY_G_CM3 = 7.874
IRON_Z_OVER_A = 26.0 / 55.845
LEAD_DENSITY_G_CM3 = 11.34
LEAD_Z_OVER_A = 82.0 / 207.2
ELEMENT_Z_AND_ATOMIC_MASS = MappingProxyType(
    {
        "H": (1.0, 1.00794),
        "C": (6.0, 12.0107),
        "N": (7.0, 14.0067),
        "O": (8.0, 15.9994),
        "Al": (13.0, 26.9815385),
        "Si": (14.0, 28.0855),
        "Ar": (18.0, 39.948),
        "Ca": (20.0, 40.078),
        "Cr": (24.0, 51.9961),
        "Fe": (26.0, 55.845),
        "Ni": (28.0, 58.6934),
        "Pb": (82.0, 207.2),
    }
)

ADDITIVE_SCATTER_FEATURE_ORDER = (
    "fe_single_compton_probability",
    "pb_single_compton_probability",
    "obstacle_single_compton_probability",
    "air_single_compton_probability",
    "fe_pb_interaction_probability",
    "shield_obstacle_interaction_probability",
    "material_air_interaction_probability",
)
ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID = (
    0.0,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
)
ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS = (
    "pre_dead_time_raw_incident_gamma_source_line_entry_class"
)
ADDITIVE_SCATTER_TARGET_SEMANTICS = (
    "interacted_plus_secondary_pre_dead_time_incident_counts_divided_by_"
    "unattenuated_geometric_line_counts"
)
ADDITIVE_SCATTER_MODEL_ID = "additive_noncollided_transport_response_v1"
_ADDITIVE_SCATTER_TRAINING_BASE_KEYS = frozenset(
    {
        "schema_version",
        "acceptance_contract_sha256",
        "training_scene_seeds",
        "scenario_ids",
        "pair_ids_by_scene",
        "artifact_sha256_by_scene",
        "label_space",
        "selection_objective",
    }
)
_ADDITIVE_SCATTER_TRAINING_FIT_KEYS = frozenset(
    {
        "fit_sample_count",
        "loso_scene_ids",
        "candidate_validation_scores",
        "selected_validation_score",
        "selected_ridge_lambda",
        "selection_completed",
    }
)


def _canonical_json_sha256(payload: object) -> str:
    """Return a canonical lowercase SHA-256 for one JSON-compatible payload."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef" for character in value
        )
    )


def _is_finite_json_number(value: object) -> bool:
    """Return whether one value is a finite JSON number, excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )


def _strict_json_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting scalar coercion."""
    if not _is_finite_json_number(value):
        raise TypeError(f"{field_name} must be a finite JSON number.")
    return float(value)


def _training_manifest_base_ready(
    manifest: Mapping[str, object],
) -> bool:
    """Validate immutable training-only scatter provenance."""
    from spectrum.transport_spectral import (
        DESIGNATED_TRAINING_SCENE_SEEDS,
        FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
        VALIDATION_SCENARIO_IDS,
    )

    if set(manifest).difference(
        _ADDITIVE_SCATTER_TRAINING_FIT_KEYS
    ) != _ADDITIVE_SCATTER_TRAINING_BASE_KEYS:
        return False
    training_seeds = manifest.get("training_scene_seeds")
    scenarios = manifest.get("scenario_ids")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("acceptance_contract_sha256")
        != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        or not isinstance(training_seeds, list)
        or any(type(seed) is not int for seed in training_seeds)
        or tuple(training_seeds) != DESIGNATED_TRAINING_SCENE_SEEDS
        or not isinstance(scenarios, list)
        or any(type(scenario) is not str for scenario in scenarios)
        or tuple(scenarios) != VALIDATION_SCENARIO_IDS
        or manifest.get("label_space")
        != ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
        or manifest.get("selection_objective")
        != "leave_one_training_scene_out_weighted_log1p_mse"
    ):
        return False
    seed_keys = {
        str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS
    }
    pair_ids = manifest.get("pair_ids_by_scene")
    artifact_hashes = manifest.get("artifact_sha256_by_scene")
    return bool(
        isinstance(pair_ids, Mapping)
        and set(pair_ids) == seed_keys
        and all(
            isinstance(pair_ids[str(seed)], list)
            and all(
                type(pair_id) is int
                for pair_id in pair_ids[str(seed)]
            )
            and tuple(pair_ids[str(seed)]) == tuple(range(64))
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        )
        and isinstance(artifact_hashes, Mapping)
        and set(artifact_hashes) == seed_keys
        and all(
            _is_lower_sha256(artifact_hashes[str(seed)])
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        )
    )


def _training_manifest_fit_ready(
    manifest: Mapping[str, object],
    *,
    selected_ridge_lambda: float,
) -> bool:
    """Validate LOSO selection outputs without accepting holdout provenance."""
    from spectrum.transport_spectral import DESIGNATED_TRAINING_SCENE_SEEDS

    if set(manifest) != (
        _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
        | _ADDITIVE_SCATTER_TRAINING_FIT_KEYS
    ):
        return False
    if not _training_manifest_base_ready(manifest):
        return False
    scores = manifest.get("candidate_validation_scores")
    expected_score_keys = {
        format(value, ".12g")
        for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
    }
    raw_fit_sample_count = manifest.get("fit_sample_count")
    raw_selected_score = manifest.get("selected_validation_score")
    raw_manifest_ridge = manifest.get("selected_ridge_lambda")
    loso_scene_ids = manifest.get("loso_scene_ids")
    if (
        type(raw_fit_sample_count) is not int
        or not _is_finite_json_number(raw_selected_score)
        or not _is_finite_json_number(raw_manifest_ridge)
    ):
        return False
    fit_sample_count = raw_fit_sample_count
    selected_score = float(raw_selected_score)
    manifest_ridge = float(raw_manifest_ridge)
    if (
        fit_sample_count < len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or not isinstance(loso_scene_ids, list)
        or any(type(scene_id) is not str for scene_id in loso_scene_ids)
        or tuple(loso_scene_ids)
        != tuple(str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS)
        or not isinstance(scores, Mapping)
        or set(scores) != expected_score_keys
        or any(type(key) is not str for key in scores)
        or any(
            not _is_finite_json_number(value) or float(value) < 0.0
            for value in scores.values()
        )
        or not np.isfinite(selected_score)
        or selected_score < 0.0
        or manifest_ridge != float(selected_ridge_lambda)
        or manifest.get("selection_completed") is not True
    ):
        return False
    minimum_score = min(float(value) for value in scores.values())
    tied_ridges = [
        ridge
        for ridge in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
        if float(scores[format(ridge, ".12g")])
        <= minimum_score + 1.0e-12
    ]
    return bool(
        np.isclose(selected_score, minimum_score, rtol=0.0, atol=0.0)
        and manifest_ridge == max(tied_ridges)
    )


def klein_nishina_total_cross_section_cm2(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
) -> NDArray[np.float64]:
    """Return the total Klein-Nishina cross section per electron."""
    energy = np.asarray(energy_keV, dtype=np.float64)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("Gamma-line energies must be finite and positive.")
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    log_term = np.log1p(2.0 * alpha)
    bracket = (
        (1.0 + alpha)
        / np.square(alpha)
        * (
            2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)
            - log_term / alpha
        )
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / np.square(1.0 + 2.0 * alpha)
    )
    return (
        2.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * np.maximum(bracket, 0.0)
    )


def material_compton_fraction_numpy(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
    total_mu_cm_inv: NDArray[np.float64] | Sequence[float] | float,
    *,
    density_g_cm3: float,
    z_over_a: float,
) -> NDArray[np.float64]:
    """Return the known Compton-to-total linear attenuation fraction."""
    energy, total_mu = np.broadcast_arrays(
        np.asarray(energy_keV, dtype=np.float64),
        np.asarray(total_mu_cm_inv, dtype=np.float64),
    )
    density = float(density_g_cm3)
    electron_ratio = float(z_over_a)
    if (
        np.any(~np.isfinite(total_mu))
        or np.any(total_mu < 0.0)
        or not np.isfinite(density)
        or density <= 0.0
        or not np.isfinite(electron_ratio)
        or electron_ratio <= 0.0
    ):
        raise ValueError("Material attenuation inputs must be physical.")
    compton_mu = (
        density
        * AVOGADRO_CONSTANT_MOL_INV
        * electron_ratio
        * klein_nishina_total_cross_section_cm2(energy)
    )
    return np.clip(
        np.divide(
            compton_mu,
            np.maximum(total_mu, np.finfo(np.float64).tiny),
        ),
        0.0,
        1.0,
    )


def composition_effective_z_over_a(
    composition_by_mass: Mapping[str, float],
) -> float:
    """Return the electron-per-atomic-mass ratio of a mass composition."""
    if not isinstance(composition_by_mass, Mapping) or not composition_by_mass:
        raise ValueError("Material composition must be a nonempty mapping.")
    normalized: dict[str, float] = {}
    for element, raw_weight in composition_by_mass.items():
        symbol = str(element)
        if symbol not in ELEMENT_Z_AND_ATOMIC_MASS:
            raise KeyError(
                f"No atomic-number contract exists for element {symbol!r}."
            )
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Material mass fractions must be nonnegative.")
        normalized[symbol] = weight
    total = float(sum(normalized.values()))
    if total <= 0.0:
        raise ValueError("Material mass fractions must have positive mass.")
    return float(
        sum(
            (weight / total)
            * ELEMENT_Z_AND_ATOMIC_MASS[element][0]
            / ELEMENT_Z_AND_ATOMIC_MASS[element][1]
            for element, weight in normalized.items()
        )
    )


def material_compton_mu_cm_inv_numpy(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
    *,
    density_g_cm3: float,
    composition_by_mass: Mapping[str, float],
) -> NDArray[np.float64]:
    """Return physical Compton linear attenuation for a known composition."""
    density = float(density_g_cm3)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Material density must be finite and positive.")
    return (
        density
        * AVOGADRO_CONSTANT_MOL_INV
        * composition_effective_z_over_a(composition_by_mass)
        * klein_nishina_total_cross_section_cm2(energy_keV)
    )


def physical_scatter_basis_numpy(
    *,
    tau_fe: NDArray[np.float64],
    tau_pb: NDArray[np.float64],
    tau_obstacle: NDArray[np.float64],
    tau_obstacle_compton: NDArray[np.float64],
    distance_m: NDArray[np.float64],
    energy_keV: NDArray[np.float64],
    mu_fe_cm_inv: NDArray[np.float64],
    mu_pb_cm_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the batched physical scatter-opportunity feature basis."""
    (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    ) = np.broadcast_arrays(
        np.asarray(tau_fe, dtype=np.float64),
        np.asarray(tau_pb, dtype=np.float64),
        np.asarray(tau_obstacle, dtype=np.float64),
        np.asarray(tau_obstacle_compton, dtype=np.float64),
        np.asarray(distance_m, dtype=np.float64),
        np.asarray(energy_keV, dtype=np.float64),
        np.asarray(mu_fe_cm_inv, dtype=np.float64),
        np.asarray(mu_pb_cm_inv, dtype=np.float64),
    )
    arrays = (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    )
    if any(np.any(~np.isfinite(array)) for array in arrays) or any(
        np.any(array < 0.0) for array in arrays
    ):
        raise ValueError("Scatter basis inputs must be finite and nonnegative.")
    if np.any(energy <= 0.0):
        raise ValueError("Scatter basis line energies must be positive.")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(obstacle_compton_tau > obstacle_tau * (1.0 + tolerance)):
        raise ValueError(
            "Obstacle Compton optical depth cannot exceed total optical depth."
        )
    fe_fraction = material_compton_fraction_numpy(
        energy,
        mu_fe,
        density_g_cm3=IRON_DENSITY_G_CM3,
        z_over_a=IRON_Z_OVER_A,
    )
    pb_fraction = material_compton_fraction_numpy(
        energy,
        mu_pb,
        density_g_cm3=LEAD_DENSITY_G_CM3,
        z_over_a=LEAD_Z_OVER_A,
    )
    obstacle_fraction = np.divide(
        obstacle_compton_tau,
        np.maximum(obstacle_tau, np.finfo(np.float64).tiny),
        out=np.zeros_like(obstacle_tau),
        where=obstacle_tau > 0.0,
    )
    air_mu = (
        AIR_DENSITY_G_CM3
        * AVOGADRO_CONSTANT_MOL_INV
        * AIR_EFFECTIVE_Z_OVER_A
        * klein_nishina_total_cross_section_cm2(energy)
    )
    p_fe = -np.expm1(-fe_tau) * fe_fraction
    p_pb = -np.expm1(-pb_tau) * pb_fraction
    p_obstacle = -np.expm1(-obstacle_tau) * np.clip(
        obstacle_fraction,
        0.0,
        1.0,
    )
    p_air = -np.expm1(-distance * 100.0 * air_mu)
    p_shield = p_fe + p_pb
    p_material = p_shield + p_obstacle
    return np.stack(
        (
            p_fe,
            p_pb,
            p_obstacle,
            p_air,
            p_fe * p_pb,
            p_shield * p_obstacle,
            p_material * p_air,
        ),
        axis=-1,
    )


def physical_scatter_basis_torch(
    *,
    tau_fe: object,
    tau_pb: object,
    tau_obstacle: object,
    tau_obstacle_compton: object,
    distance_m: object,
    energy_keV: object,
    mu_fe_cm_inv: object,
    mu_pb_cm_inv: object,
) -> object:
    """Return the Torch equivalent of the physical scatter feature basis."""
    import torch

    fe_tau = torch.as_tensor(tau_fe)
    if fe_tau.dtype != torch.float64:
        raise TypeError("Production scatter evaluation requires torch.float64.")
    tensors = [
        torch.as_tensor(value, device=fe_tau.device, dtype=fe_tau.dtype)
        for value in (
            tau_pb,
            tau_obstacle,
            tau_obstacle_compton,
            distance_m,
            energy_keV,
            mu_fe_cm_inv,
            mu_pb_cm_inv,
        )
    ]
    (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    ) = torch.broadcast_tensors(fe_tau, *tensors)
    values = (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    )
    if any(bool(torch.any(~torch.isfinite(value))) for value in values) or any(
        bool(torch.any(value < 0.0)) for value in values
    ):
        raise ValueError("Torch scatter basis inputs must be physical.")
    if bool(torch.any(energy <= 0.0)):
        raise ValueError("Torch scatter line energies must be positive.")
    tolerance = 64.0 * torch.finfo(fe_tau.dtype).eps
    if bool(
        torch.any(
            obstacle_compton_tau > obstacle_tau * (1.0 + tolerance)
        )
    ):
        raise ValueError(
            "Obstacle Compton optical depth cannot exceed total optical depth."
        )

    def _material_fraction(
        total_mu: object,
        *,
        density_g_cm3: float,
        z_over_a: float,
    ) -> object:
        """Return a material Compton fraction on the active Torch device."""
        alpha = energy / ELECTRON_REST_ENERGY_KEV
        log_term = torch.log1p(2.0 * alpha)
        bracket = (
            (1.0 + alpha)
            / torch.square(alpha)
            * (
                2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)
                - log_term / alpha
            )
            + log_term / (2.0 * alpha)
            - (1.0 + 3.0 * alpha) / torch.square(1.0 + 2.0 * alpha)
        )
        sigma = (
            2.0
            * np.pi
            * CLASSICAL_ELECTRON_RADIUS_CM**2
            * torch.clamp(bracket, min=0.0)
        )
        compton_mu = (
            float(density_g_cm3)
            * AVOGADRO_CONSTANT_MOL_INV
            * float(z_over_a)
            * sigma
        )
        return torch.clamp(
            compton_mu / torch.clamp(total_mu, min=torch.finfo(fe_tau.dtype).tiny),
            min=0.0,
            max=1.0,
        )

    fe_fraction = _material_fraction(
        mu_fe,
        density_g_cm3=IRON_DENSITY_G_CM3,
        z_over_a=IRON_Z_OVER_A,
    )
    pb_fraction = _material_fraction(
        mu_pb,
        density_g_cm3=LEAD_DENSITY_G_CM3,
        z_over_a=LEAD_Z_OVER_A,
    )
    obstacle_fraction = torch.where(
        obstacle_tau > 0.0,
        obstacle_compton_tau
        / torch.clamp(obstacle_tau, min=torch.finfo(fe_tau.dtype).tiny),
        torch.zeros_like(obstacle_tau),
    )
    air_fraction = _material_fraction(
        torch.ones_like(energy),
        density_g_cm3=AIR_DENSITY_G_CM3,
        z_over_a=AIR_EFFECTIVE_Z_OVER_A,
    )
    air_mu = air_fraction
    p_fe = -torch.expm1(-fe_tau) * fe_fraction
    p_pb = -torch.expm1(-pb_tau) * pb_fraction
    p_obstacle = -torch.expm1(-obstacle_tau) * torch.clamp(
        obstacle_fraction,
        min=0.0,
        max=1.0,
    )
    p_air = -torch.expm1(-distance * 100.0 * air_mu)
    p_shield = p_fe + p_pb
    p_material = p_shield + p_obstacle
    return torch.stack(
        (
            p_fe,
            p_pb,
            p_obstacle,
            p_air,
            p_fe * p_pb,
            p_shield * p_obstacle,
            p_material * p_air,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class AdditiveNoncollidedTransportResponse:
    """Store one authenticated global nonnegative additive scatter response."""

    coefficients: tuple[float, ...]
    ridge_lambda: float
    training_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate and freeze model coefficients and training provenance."""
        coefficients = tuple(float(value) for value in self.coefficients)
        ridge_lambda = float(self.ridge_lambda)
        if (
            len(coefficients) != len(ADDITIVE_SCATTER_FEATURE_ORDER)
            or any(not np.isfinite(value) or value < 0.0 for value in coefficients)
            or not any(value > 0.0 for value in coefficients)
            or ridge_lambda not in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
        ):
            raise ValueError(
                "Additive scatter coefficients must be nonnegative, nonzero, "
                "and use the predeclared ridge grid."
            )
        if not isinstance(self.training_manifest, Mapping):
            raise TypeError("Scatter training provenance must be a mapping.")
        manifest = json.loads(
            json.dumps(
                dict(self.training_manifest),
                sort_keys=True,
                allow_nan=False,
            )
        )
        if not _training_manifest_fit_ready(
            manifest,
            selected_ridge_lambda=ridge_lambda,
        ):
            raise ValueError(
                "Additive scatter training provenance must exactly bind the "
                "designated training scenes, every shield pair and scenario, "
                "and the predeclared LOSO ridge selection."
            )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "ridge_lambda", ridge_lambda)
        object.__setattr__(
            self,
            "training_manifest",
            MappingProxyType(manifest),
        )
        object.__setattr__(
            self,
            "_contract_hash_sha256",
            _canonical_json_sha256(self._contract_payload()),
        )

    def _contract_payload(self) -> dict[str, object]:
        """Return fields that define the immutable fitted response."""
        return {
            "model": ADDITIVE_SCATTER_MODEL_ID,
            "feature_order": list(ADDITIVE_SCATTER_FEATURE_ORDER),
            "coefficients": list(self.coefficients),
            "fit_family": "nonnegative_ridge_nnls",
            "ridge_lambda_grid": list(ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID),
            "selected_ridge_lambda": float(self.ridge_lambda),
            "selection_objective": (
                "leave_one_training_scene_out_weighted_log1p_mse"
            ),
            "tie_break": "largest_ridge_lambda_within_1e-12",
            "coefficient_scope": (
                "one_global_vector_all_scenes_pairs_sources_isotopes_lines"
            ),
            "incident_label_semantics": (
                ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
            ),
            "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
            "kernel_equation": (
                "total=uncollided+unattenuated_geometric*max(0,basis@coefficients)"
            ),
            "training": dict(self.training_manifest),
        }

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable fitted response contract hash."""
        return str(self._contract_hash_sha256)

    @property
    def training_ready(self) -> bool:
        """Return whether strict training-only provenance authenticates the fit."""
        return _training_manifest_fit_ready(
            self.training_manifest,
            selected_ridge_lambda=self.ridge_lambda,
        )

    def to_payload(self) -> dict[str, object]:
        """Return an authenticated JSON-compatible response payload."""
        payload = {
            "schema_version": 1,
            **self._contract_payload(),
            "contract_hash_sha256": self.contract_hash_sha256,
        }
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AdditiveNoncollidedTransportResponse":
        """Reconstruct and byte-authenticate one additive response payload."""
        raw_coefficients = (
            payload.get("coefficients")
            if isinstance(payload, Mapping)
            else None
        )
        raw_ridge = (
            payload.get("selected_ridge_lambda")
            if isinstance(payload, Mapping)
            else None
        )
        raw_ridge_grid = (
            payload.get("ridge_lambda_grid")
            if isinstance(payload, Mapping)
            else None
        )
        if (
            not isinstance(payload, Mapping)
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("model") != ADDITIVE_SCATTER_MODEL_ID
            or tuple(payload.get("feature_order", ()))
            != ADDITIVE_SCATTER_FEATURE_ORDER
            or payload.get("fit_family") != "nonnegative_ridge_nnls"
            or not isinstance(raw_ridge_grid, list)
            or any(
                not _is_finite_json_number(value)
                for value in raw_ridge_grid
            )
            or tuple(raw_ridge_grid)
            != ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            or not isinstance(raw_coefficients, list)
            or any(
                not _is_finite_json_number(value)
                for value in raw_coefficients
            )
            or not _is_finite_json_number(raw_ridge)
            or payload.get("incident_label_semantics")
            != ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
            or payload.get("target_semantics")
            != ADDITIVE_SCATTER_TARGET_SEMANTICS
            or not isinstance(payload.get("training"), Mapping)
            or not _is_lower_sha256(payload.get("contract_hash_sha256"))
        ):
            raise ValueError("Additive scatter response schema is invalid.")
        model = cls(
            coefficients=tuple(
                _strict_json_number(
                    value,
                    field_name=f"coefficients[{index}]",
                )
                for index, value in enumerate(raw_coefficients)
            ),
            ridge_lambda=_strict_json_number(
                raw_ridge,
                field_name="selected_ridge_lambda",
            ),
            training_manifest=payload["training"],
        )
        if model.to_payload() != dict(payload):
            raise ValueError(
                "Additive scatter response does not exactly reconstruct its "
                "declared contract."
            )
        return model

    def scatter_fraction_numpy(
        self,
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched nonnegative scatter fractions using NumPy."""
        basis = np.asarray(feature_basis, dtype=np.float64)
        if (
            basis.ndim < 1
            or basis.shape[-1] != len(self.coefficients)
            or np.any(~np.isfinite(basis))
            or np.any(basis < 0.0)
        ):
            raise ValueError("Scatter feature basis is invalid.")
        return np.maximum(
            np.einsum(
                "...f,f->...",
                basis,
                np.asarray(self.coefficients, dtype=np.float64),
                optimize=True,
            ),
            0.0,
        )

    def scatter_fraction_torch(self, feature_basis: object) -> object:
        """Return batched nonnegative scatter fractions using Torch."""
        import torch

        basis = torch.as_tensor(feature_basis)
        if basis.dtype != torch.float64:
            raise TypeError("Production scatter evaluation requires torch.float64.")
        if (
            basis.ndim < 1
            or basis.shape[-1] != len(self.coefficients)
            or bool(torch.any(~torch.isfinite(basis)))
            or bool(torch.any(basis < 0.0))
        ):
            raise ValueError("Torch scatter feature basis is invalid.")
        coefficients = torch.as_tensor(
            self.coefficients,
            device=basis.device,
            dtype=basis.dtype,
        )
        return torch.clamp(
            torch.einsum("...f,f->...", basis, coefficients),
            min=0.0,
        )

    def total_kernel_numpy(
        self,
        unattenuated_geometric_kernel: NDArray[np.float64],
        uncollided_kernel: NDArray[np.float64],
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the additive total detector-entry kernel using NumPy."""
        unattenuated = np.asarray(
            unattenuated_geometric_kernel,
            dtype=np.float64,
        )
        uncollided = np.asarray(uncollided_kernel, dtype=np.float64)
        if (
            unattenuated.shape != uncollided.shape
            or feature_basis.shape != unattenuated.shape + (len(self.coefficients),)
            or np.any(~np.isfinite(unattenuated))
            or np.any(unattenuated < 0.0)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
        ):
            raise ValueError("Additive scatter kernel inputs are invalid.")
        return uncollided + unattenuated * self.scatter_fraction_numpy(
            feature_basis
        )

    def total_kernel_torch(
        self,
        unattenuated_geometric_kernel: object,
        uncollided_kernel: object,
        feature_basis: object,
    ) -> object:
        """Return the additive total detector-entry kernel using Torch."""
        import torch

        unattenuated = torch.as_tensor(unattenuated_geometric_kernel)
        if unattenuated.dtype != torch.float64:
            raise TypeError("Production scatter evaluation requires torch.float64.")
        uncollided = torch.as_tensor(
            uncollided_kernel,
            device=unattenuated.device,
            dtype=unattenuated.dtype,
        )
        basis = torch.as_tensor(
            feature_basis,
            device=unattenuated.device,
            dtype=unattenuated.dtype,
        )
        if (
            unattenuated.shape != uncollided.shape
            or basis.shape
            != unattenuated.shape + (len(self.coefficients),)
            or bool(torch.any(~torch.isfinite(unattenuated)))
            or bool(torch.any(unattenuated < 0.0))
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(uncollided < 0.0))
        ):
            raise ValueError("Torch additive scatter kernel inputs are invalid.")
        return (
            uncollided
            + unattenuated * self.scatter_fraction_torch(basis)
        )


def _fit_nonnegative_ridge(
    features_nf: NDArray[np.float64],
    targets_n: NDArray[np.float64],
    weights_n: NDArray[np.float64],
    *,
    ridge_lambda: float,
) -> NDArray[np.float64]:
    """Fit one weighted nonnegative ridge model with an augmented NNLS solve."""
    features = np.asarray(features_nf, dtype=np.float64)
    targets = np.asarray(targets_n, dtype=np.float64)
    weights = np.asarray(weights_n, dtype=np.float64)
    root_weights = np.sqrt(weights)
    weighted_features = features * root_weights[:, np.newaxis]
    weighted_targets = targets * root_weights
    ridge = float(ridge_lambda)
    if ridge > 0.0:
        weighted_features = np.concatenate(
            (
                weighted_features,
                np.sqrt(ridge)
                * np.eye(features.shape[1], dtype=np.float64),
            ),
            axis=0,
        )
        weighted_targets = np.concatenate(
            (
                weighted_targets,
                np.zeros(features.shape[1], dtype=np.float64),
            )
        )
    coefficients, _ = nnls(weighted_features, weighted_targets)
    return np.asarray(coefficients, dtype=np.float64)


def fit_additive_noncollided_transport_response(
    feature_basis_nf: NDArray[np.float64],
    target_scatter_fraction_n: NDArray[np.float64],
    sample_weights_n: NDArray[np.float64],
    training_scene_ids_n: Sequence[object],
    *,
    training_manifest: Mapping[str, object],
) -> AdditiveNoncollidedTransportResponse:
    """Fit the predeclared global model using training scenes only.

    Ridge selection uses leave-one-training-scene-out weighted log1p mean
    squared error.  Holdout observations are intentionally not accepted by this
    API, so they cannot influence coefficients or regularization selection.
    """
    if (
        not isinstance(training_manifest, Mapping)
        or set(training_manifest)
        != _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
        or not _training_manifest_base_ready(training_manifest)
    ):
        raise ValueError(
            "Scatter fitting requires the exact designated-training manifest."
        )
    features = np.asarray(feature_basis_nf, dtype=np.float64)
    targets = np.asarray(target_scatter_fraction_n, dtype=np.float64).reshape(-1)
    weights = np.asarray(sample_weights_n, dtype=np.float64).reshape(-1)
    scene_ids = np.asarray(
        [str(value) for value in training_scene_ids_n],
        dtype=object,
    ).reshape(-1)
    if (
        features.ndim != 2
        or features.shape[1] != len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or targets.shape != (features.shape[0],)
        or weights.shape != targets.shape
        or scene_ids.shape != targets.shape
        or features.shape[0] < len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or np.unique(scene_ids).size < 2
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
        or np.any(~np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Additive scatter training arrays are invalid.")
    unique_scenes = np.unique(scene_ids)
    candidate_scores: list[tuple[float, float]] = []
    log_targets = np.log1p(targets)
    for ridge_lambda in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID:
        squared_error_sum = 0.0
        weight_sum = 0.0
        for held_out_scene in unique_scenes:
            validation_mask = scene_ids == held_out_scene
            training_mask = ~validation_mask
            coefficients = _fit_nonnegative_ridge(
                features[training_mask],
                targets[training_mask],
                weights[training_mask],
                ridge_lambda=float(ridge_lambda),
            )
            predictions = np.maximum(
                features[validation_mask] @ coefficients,
                0.0,
            )
            errors = np.square(
                np.log1p(predictions) - log_targets[validation_mask]
            )
            squared_error_sum += float(
                np.sum(weights[validation_mask] * errors)
            )
            weight_sum += float(np.sum(weights[validation_mask]))
        candidate_scores.append(
            (
                squared_error_sum / max(weight_sum, np.finfo(np.float64).tiny),
                float(ridge_lambda),
            )
        )
    best_score = min(score for score, _ in candidate_scores)
    tied = [
        ridge
        for score, ridge in candidate_scores
        if score <= best_score + 1.0e-12
    ]
    selected_lambda = max(tied)
    coefficients = _fit_nonnegative_ridge(
        features,
        targets,
        weights,
        ridge_lambda=selected_lambda,
    )
    if not np.any(coefficients > 0.0):
        raise RuntimeError(
            "Designated training produced a zero-scatter model; production "
            "approval is forbidden."
        )
    provenance = dict(training_manifest)
    provenance.update(
        {
            "fit_sample_count": int(features.shape[0]),
            "loso_scene_ids": [
                str(value) for value in unique_scenes.tolist()
            ],
            "candidate_validation_scores": {
                format(ridge, ".12g"): float(score)
                for score, ridge in candidate_scores
            },
            "selected_validation_score": float(best_score),
            "selected_ridge_lambda": float(selected_lambda),
            "selection_completed": True,
        }
    )
    return AdditiveNoncollidedTransportResponse(
        coefficients=tuple(float(value) for value in coefficients),
        ridge_lambda=float(selected_lambda),
        training_manifest=provenance,
    )
