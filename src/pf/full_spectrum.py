"""Contracts for the production joint full-spectrum PF observation model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY = "full_spectrum_contract_hash_sha256"
FULL_SPECTRUM_MODEL_SCHEMA_VERSION = 7
FULL_SPECTRUM_MODEL_ID = "geometry_conditioned_full_spectrum"
DETECTOR_IMPACT_PHASE_COUNT = 8
DETECTOR_IMPACT_FEATURE_ORDER = tuple(
    f"uncollided_impact_fraction_{index}"
    for index in range(DETECTOR_IMPACT_PHASE_COUNT)
)
TRANSPORT_FEATURE_ORDER = (
    "tau_fe",
    "tau_pb",
    "tau_obstacle",
    "tau_obstacle_compton",
    "distance_m",
    *DETECTOR_IMPACT_FEATURE_ORDER,
)
DETECTOR_GREEN_OPERATOR_ID = "isotope_independent_full_detector_green_operator_v3"
DETECTOR_GREEN_BOUNDARY_STATE = (
    "normalized_impact_parameter_at_detector_housing_entry_v1"
)
DETECTOR_GREEN_FINITE_MC_UNCERTAINTY = "pulse_plus_no_pulse_categorical_covariance_v1"
DETECTOR_GREEN_PHASE_CONDITIONING = (
    "transport_resolved_direct_impact_and_detector_cone_scatter_joint_state_v3"
)
DETECTOR_CONE_SCATTER_RESPONSE_ID = (
    "detector_cone_joint_energy_impact_single_compton_v1"
)
CATALOG_SOURCE_RATE_NORMALIZATION = (
    "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
)
CATALOG_LINE_FIELDS = frozenset(
    {
        "isotope",
        "transport_line_index",
        "energy_keV",
        "branching_weight",
        "raw_bin_index",
        "raw_bin_energy_keV",
        "mu_fe_cm_inv",
        "mu_pb_cm_inv",
    }
)


@dataclass(frozen=True)
class CatalogTransportLine:
    """Represent one authenticated catalog line consumed by PF transport."""

    global_column: int
    isotope: str
    transport_line_index: int
    energy_keV: float
    branching_weight: float
    raw_bin_index: int
    raw_bin_energy_keV: float
    mu_fe_cm_inv: float
    mu_pb_cm_inv: float


@dataclass(frozen=True)
class CatalogIsotopeLineLayout:
    """Hold the immutable global-to-catalog line mapping for one isotope."""

    isotope: str
    global_columns: tuple[int, ...]
    transport_line_indices: tuple[int, ...]
    energies_keV: tuple[float, ...]
    branching_weights: tuple[float, ...]
    mu_fe_cm_inv: tuple[float, ...]
    mu_pb_cm_inv: tuple[float, ...]


def _strict_sha256(value: object, *, field_name: str) -> str:
    """Return one strict lowercase SHA-256 digest."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _strict_line_number(value: object, *, field_name: str) -> float:
    """Return a finite non-boolean catalog-line number without coercion."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Catalog line {field_name} must be a finite number.")
    return float(value)


def _strict_line_index(value: object, *, field_name: str) -> int:
    """Return a nonnegative catalog-line integer without truncation."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"Catalog line {field_name} must be a nonnegative integer.")
    return int(value)


@runtime_checkable
class FullSpectrumGenerativeModel(Protocol):
    """Define the fail-closed full-spectrum model shared by PF and planning.

    Implementations own the complete observation distribution.  In
    particular, the PF must not add a second Poisson term, a projected
    isotope-count covariance, or another likelihood derived from the same
    spectrum.
    """

    @property
    def runtime_ready(self) -> bool:
        """Return whether explicit nonproduction contracts authorize use."""

    @property
    def production_ready(self) -> bool:
        """Return whether independent all-pair validation approved release."""

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable model, response, and energy-axis digest."""

    @property
    def energy_axis_keV(self) -> NDArray[np.float64]:
        """Return the exact analysis-spectrum bin axis."""

    @property
    def line_identity(self) -> tuple[Mapping[str, object], ...]:
        """Return the global positive transport-line order."""

    @property
    def transport_feature_order(self) -> tuple[str, ...]:
        """Return the final-axis order of geometry-conditioned line features."""

    @property
    def detector_impact_parameter_edges_fraction(self) -> NDArray[np.float64]:
        """Return the authenticated detector-impact partition."""

    @property
    def detector_target_radius_m(self) -> float:
        """Return the detector housing radius used by the Green operator."""

    def require_runtime_ready(self) -> None:
        """Raise unless immutable nonproduction construction is complete."""

    def require_production_ready(self) -> None:
        """Raise unless the immutable model passed every production gate."""

    def log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one joint full-spectrum log likelihood per particle."""

    def log_likelihood_torch(
        self,
        observed_spectrum_vb: object,
        total_line_contributions_nvsl: object,
        uncollided_line_contributions_nvsl: object,
        transport_features_nvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch-equivalent joint log likelihood per particle."""

    def cross_log_likelihood_numpy(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return batched action/sample/state full-spectrum log likelihoods."""

    def cross_log_likelihood_torch(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> object:
        """Return Torch-equivalent batched action/sample/state likelihoods."""

    def cross_log_likelihood_replace_slots_torch(
        self,
        observed_spectra_xqvb: object,
        accepted_total_line_contributions_xNvsl: object,
        accepted_uncollided_line_contributions_xNvsl: object,
        accepted_transport_features_xNvslf: object,
        replacement_total_line_contributions_xnvrl: object,
        replacement_uncollided_line_contributions_xnvrl: object,
        replacement_transport_features_xnvrlf: object,
        live_times_s_v: object,
        *,
        particle_indices_n: object,
        slot_start: int,
        slot_stop: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> object:
        """Return exact Torch likelihoods for one replaced source-slot block."""

    def predict_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected analysis spectra ending in view/bin axes."""

    def predict_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return Torch expected analysis spectra ending in view/bin axes."""

    def sample_predictive_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        sample_count: int,
        rng: np.random.Generator,
        action_seeds_a: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Draw exact future spectra shaped state x sample x view x bin."""

    def posterior_predictive_innovation_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        particle_weights_n: NDArray[np.float64],
        *,
        confidence: float,
    ) -> Mapping[str, float | int | bool | None]:
        """Return a model-native calibrated posterior innovation diagnostic."""

    def birth_proposal_log_scores_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        candidate_total_line_contributions_gvsl: NDArray[np.float64],
        candidate_uncollided_line_contributions_gvsl: NDArray[np.float64],
        candidate_transport_features_gvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        target_line_mask_l: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return deterministic proposal-only scores for target-line candidates."""

    def birth_proposal_log_scores_torch(
        self,
        observed_spectrum_vb: object,
        candidate_total_line_contributions_gvsl: object,
        candidate_uncollided_line_contributions_gvsl: object,
        candidate_transport_features_gvslf: object,
        live_times_s_v: object,
        *,
        target_line_mask_l: object,
    ) -> object:
        """Return Torch-equivalent deterministic proposal-only scores."""

    def manifest_payload(self) -> Mapping[str, object]:
        """Return immutable model and validation provenance."""


@runtime_checkable
class TorchPredictiveFullSpectrumModel(Protocol):
    """Define the optional device-resident predictive sampling capability."""

    def sample_predictive_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
        *,
        sample_count: int,
        generator: object | None = None,
        action_seeds_a: object | None = None,
    ) -> object:
        """Draw exact integer spectra without leaving the Torch device."""


@runtime_checkable
class PreparedSubsetCrossLikelihood(Protocol):
    """Define an opaque arbitrary-view likelihood cache owned by runtime."""

    @property
    def action_count(self) -> int:
        """Return the number of aligned detector-pose actions."""

    @property
    def sample_count(self) -> int:
        """Return the number of predictive observations per action."""

    @property
    def state_count(self) -> int:
        """Return the number of PF states compared with each observation."""

    @property
    def view_count(self) -> int:
        """Return the number of available shield-pair views."""

    def evaluate(self, subset_pair_indices: object) -> object:
        """Return action/candidate/sample/state likelihoods for subsets."""


@runtime_checkable
class SubsetCrossLikelihoodFullSpectrumModel(Protocol):
    """Define runtime-owned exact likelihood preparation for arbitrary views."""

    def prepare_subset_cross_likelihood_numpy(
        self,
        observed_spectra_aqvb: NDArray[np.float64],
        total_line_contributions_anvsl: NDArray[np.float64],
        uncollided_line_contributions_anvsl: NDArray[np.float64],
        transport_features_anvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
    ) -> PreparedSubsetCrossLikelihood:
        """Prepare an exact host cache for arbitrary view subsets."""

    def prepare_subset_cross_likelihood_torch(
        self,
        observed_spectra_aqvb: object,
        total_line_contributions_anvsl: object,
        uncollided_line_contributions_anvsl: object,
        transport_features_anvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
    ) -> PreparedSubsetCrossLikelihood:
        """Prepare an exact device cache for arbitrary view subsets."""


@runtime_checkable
class SubsetCrossLikelihoodMemoryModel(Protocol):
    """Define the runtime-owned arbitrary-subset memory estimate."""

    def estimate_subset_cross_likelihood_working_set_bytes(
        self,
        *,
        num_actions: int,
        num_samples: int,
        num_particles: int,
        num_source_slots: int,
        num_views: int,
        num_candidates: int,
        subset_size: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
        dtype_bytes: int = 8,
    ) -> int:
        """Return resident-cache plus peak candidate workspace bytes."""


def validated_catalog_transport_lines(
    model: FullSpectrumGenerativeModel,
    *,
    energy_axis_keV: NDArray[np.float64] | None = None,
) -> tuple[CatalogTransportLine, ...]:
    """Validate and return catalog-authenticated model transport lines.

    The PF consumes these rows as external physical inputs.  It never infers
    line energies or branching fractions from an observed spectrum.
    """
    axis = np.asarray(
        model.energy_axis_keV if energy_axis_keV is None else energy_axis_keV,
        dtype=np.float64,
    )
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("Catalog line validation requires a nonempty energy axis.")
    bin_widths = np.diff(axis)
    bin_width = float(bin_widths[0])
    if (
        np.any(~np.isfinite(axis))
        or bin_width <= 0.0
        or not np.allclose(bin_widths, bin_width, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "Catalog lines require one finite uniformly spaced energy axis."
        )
    raw_rows = tuple(model.line_identity)
    if not raw_rows:
        raise ValueError("Full-spectrum model requires positive catalog lines.")
    lines: list[CatalogTransportLine] = []
    for global_column, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or set(row) != CATALOG_LINE_FIELDS:
            raise ValueError(
                "Catalog line schema is incompatible; missing, unknown, and "
                "legacy fields are forbidden."
            )
        isotope = row["isotope"]
        if type(isotope) is not str or not isotope:
            raise ValueError("Catalog line isotope must be a nonempty string.")
        local_index = _strict_line_index(
            row["transport_line_index"],
            field_name="transport_line_index",
        )
        energy = _strict_line_number(row["energy_keV"], field_name="energy_keV")
        branching = _strict_line_number(
            row["branching_weight"],
            field_name="branching_weight",
        )
        raw_bin_index = _strict_line_index(
            row["raw_bin_index"],
            field_name="raw_bin_index",
        )
        raw_bin_energy = _strict_line_number(
            row["raw_bin_energy_keV"],
            field_name="raw_bin_energy_keV",
        )
        mu_fe = _strict_line_number(
            row["mu_fe_cm_inv"],
            field_name="mu_fe_cm_inv",
        )
        mu_pb = _strict_line_number(
            row["mu_pb_cm_inv"],
            field_name="mu_pb_cm_inv",
        )
        expected_raw_bin = int(math.floor((energy - float(axis[0])) / bin_width))
        if (
            energy <= float(axis[0])
            or energy > float(axis[-1])
            or branching <= 0.0
            or branching > 1.0
            or mu_fe <= 0.0
            or mu_pb <= 0.0
            or raw_bin_index >= axis.size
            or raw_bin_index != expected_raw_bin
            or raw_bin_energy != float(axis[raw_bin_index])
        ):
            raise ValueError(
                f"Catalog line contract is physically inconsistent for {isotope!r}."
            )
        lines.append(
            CatalogTransportLine(
                global_column=global_column,
                isotope=isotope,
                transport_line_index=local_index,
                energy_keV=energy,
                branching_weight=branching,
                raw_bin_index=raw_bin_index,
                raw_bin_energy_keV=raw_bin_energy,
                mu_fe_cm_inv=mu_fe,
                mu_pb_cm_inv=mu_pb,
            )
        )
    isotope_order = tuple(dict.fromkeys(line.isotope for line in lines))
    if isotope_order != tuple(sorted(isotope_order)):
        raise ValueError("Catalog isotope blocks must use canonical sorted order.")
    for isotope in isotope_order:
        isotope_lines = tuple(line for line in lines if line.isotope == isotope)
        if (
            tuple(line.global_column for line in isotope_lines)
            != tuple(
                range(
                    isotope_lines[0].global_column,
                    isotope_lines[0].global_column + len(isotope_lines),
                )
            )
            or tuple(line.transport_line_index for line in isotope_lines)
            != tuple(range(len(isotope_lines)))
            or len({line.energy_keV for line in isotope_lines}) != len(isotope_lines)
            or not math.isclose(
                math.fsum(line.branching_weight for line in isotope_lines),
                1.0,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(
                f"Catalog line order or branching normalization is invalid for "
                f"{isotope!r}."
            )
    return tuple(lines)


def catalog_line_layout_by_isotope(
    model: FullSpectrumGenerativeModel,
    isotopes: tuple[str, ...],
) -> dict[str, CatalogIsotopeLineLayout]:
    """Return strict catalog-line layouts for the requested PF isotopes."""
    if (
        not isotopes
        or any(type(isotope) is not str or not isotope for isotope in isotopes)
        or len(set(isotopes)) != len(isotopes)
    ):
        raise ValueError("PF isotope order must contain unique nonempty strings.")
    lines = validated_catalog_transport_lines(model)
    result: dict[str, CatalogIsotopeLineLayout] = {}
    for isotope in isotopes:
        selected = tuple(line for line in lines if line.isotope == isotope)
        if not selected:
            raise ValueError(
                f"Full-spectrum model has no catalog line for {isotope!r}."
            )
        result[isotope] = CatalogIsotopeLineLayout(
            isotope=isotope,
            global_columns=tuple(line.global_column for line in selected),
            transport_line_indices=tuple(
                line.transport_line_index for line in selected
            ),
            energies_keV=tuple(line.energy_keV for line in selected),
            branching_weights=tuple(line.branching_weight for line in selected),
            mu_fe_cm_inv=tuple(line.mu_fe_cm_inv for line in selected),
            mu_pb_cm_inv=tuple(line.mu_pb_cm_inv for line in selected),
        )
    return result


def _validate_schema_v7_manifest(
    model: FullSpectrumGenerativeModel,
    *,
    lines: tuple[CatalogTransportLine, ...],
) -> None:
    """Validate the PF-facing physics-only schema-v7 manifest contract."""
    payload = model.manifest_payload()
    if not isinstance(payload, Mapping):
        raise TypeError("Full-spectrum model manifest must be a mapping.")
    required_fields = {
        "schema_version",
        "model",
        "contract_hash_sha256",
        "runtime_ready",
        "production_ready",
        "energy_bin_count",
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
        "transport_feature_order",
        "line_identity",
        "source_rate_semantics",
        "source_rate_green_normalization",
        "additive_noncollided_transport_response",
        "rate_scale_mixture",
        "physical_component_discrepancy",
        "mark_concentration_source",
        "mark_model",
        "scatter_shape",
        "higher_order_scatter_mean",
        "detector_cone_scatter_response",
        "detector_green_operator_id",
        "detector_green_operator_contract_sha256",
        "detector_green_operator_binary_sha256",
        "detector_green_boundary_state",
        "detector_green_phase_conditioning",
        "detector_green_finite_mc_uncertainty",
    }
    missing = required_fields - set(payload)
    if missing:
        raise ValueError(
            f"Full-spectrum schema-v7 manifest is missing fields: {sorted(missing)}."
        )
    forbidden_legacy_fields = {
        "native_response_contract_sha256",
        "detector_response_contract_sha256",
        "detector_response_validation",
        "discrepancy_training",
        "discrepancy_training_manifest_sha256",
        "low_rank_spectral_mean_correction",
        "count_discrepancy_concentration",
        "count_discrepancy_scope",
        "mark_concentration_multi_isotope",
        "maximum_scatter_order",
    }
    legacy = forbidden_legacy_fields & set(payload)
    if legacy:
        raise ValueError(
            f"Production full-spectrum manifest contains retired fields: "
            f"{sorted(legacy)}."
        )
    axis = np.asarray(model.energy_axis_keV, dtype=np.float64)
    manifest_lines = payload["line_identity"]
    property_lines = [
        {
            "isotope": line.isotope,
            "transport_line_index": line.transport_line_index,
            "energy_keV": line.energy_keV,
            "branching_weight": line.branching_weight,
            "raw_bin_index": line.raw_bin_index,
            "raw_bin_energy_keV": line.raw_bin_energy_keV,
            "mu_fe_cm_inv": line.mu_fe_cm_inv,
            "mu_pb_cm_inv": line.mu_pb_cm_inv,
        }
        for line in lines
    ]
    response = payload["additive_noncollided_transport_response"]
    mixture = payload["rate_scale_mixture"]
    discrepancy = payload["physical_component_discrepancy"]
    detector_cone_scatter = payload["detector_cone_scatter_response"]
    expected_discrepancy_fields = {
        "schema_version",
        "model",
        "count_scope",
        "count_uncollided_concentration",
        "count_scatter_concentration",
        "mark_uncollided_concentration",
        "mark_scatter_concentration",
        "mark_background_group_concentration",
        "mark_background_within_concentration",
        "fraction_contract",
        "provenance",
        "mark_latent_model",
        "mark_latent_scope",
        "mark_latent_factorization",
        "photopeak_partition_contract",
        "continuum_partition_contract",
        "component_covariance_contract",
        "detector_green_finite_mc_contract",
        "background_mark_contract",
        "count_uncollided_relative_standard_uncertainty",
        "count_scatter_relative_standard_uncertainty",
        "mark_uncollided_probability_standard_uncertainty",
        "mark_scatter_probability_standard_uncertainty",
        "mark_background_group_probability_standard_uncertainty",
        "mark_background_within_probability_standard_uncertainty",
        "higher_order_scatter_nuisance",
        "obstacle_material_contract_sha256",
        "transport_physics_table_contract_sha256",
    }
    if (
        payload["schema_version"] != FULL_SPECTRUM_MODEL_SCHEMA_VERSION
        or payload["model"] != FULL_SPECTRUM_MODEL_ID
        or payload["contract_hash_sha256"] != model.contract_hash_sha256
        or payload["runtime_ready"] is not model.runtime_ready
        or payload["production_ready"] is not model.production_ready
        or payload["energy_bin_count"] != int(axis.size)
        or payload["energy_min_keV"] != float(axis[0])
        or payload["energy_max_keV"] != float(axis[-1])
        or payload["bin_width_keV"] != float(axis[1] - axis[0])
        or payload["transport_feature_order"] != list(TRANSPORT_FEATURE_ORDER)
        or not isinstance(manifest_lines, list)
        or manifest_lines != property_lines
        or payload["source_rate_semantics"] != "pre_dead_time_detector_pulse_rate_at_1m"
        or payload["source_rate_green_normalization"]
        != CATALOG_SOURCE_RATE_NORMALIZATION
        or not isinstance(response, Mapping)
        or response.get("model") != "physics_only_detector_cone_transport_response_v2"
        or not isinstance(mixture, Mapping)
        or dict(mixture)
        != {
            "scope": "station_shared_source_only",
            "nodes": [1.0],
            "weights": [1.0],
            "weighted_mean": 1.0,
        }
        or payload["mark_concentration_source"] is not None
        or not isinstance(discrepancy, Mapping)
        or set(discrepancy) != expected_discrepancy_fields
        or discrepancy.get("schema_version") != 5
        or discrepancy.get("model")
        != "uncollided_scatter_background_component_latents_v2"
        or discrepancy.get("count_scope") != "view_independent"
        or discrepancy.get("count_uncollided_concentration") != 2500.0
        or discrepancy.get("count_scatter_concentration") != 4.0
        or discrepancy.get("mark_uncollided_concentration") != 9999.0
        or discrepancy.get("mark_scatter_concentration")
        != 23.999999999999996
        or discrepancy.get("mark_background_group_concentration")
        != 23.999999999999996
        or discrepancy.get("mark_background_within_concentration") != 9999.0
        or discrepancy.get("provenance") != "physics_only_uncertainty_budget_v1"
        or discrepancy.get("mark_latent_model")
        != "component_dirichlet_tree_hierarchical"
        or discrepancy.get("mark_background_group_concentration")
        != discrepancy.get("mark_scatter_concentration")
        or discrepancy.get("mark_background_within_concentration")
        != discrepancy.get("mark_uncollided_concentration")
        or discrepancy.get("mark_latent_scope")
        != "station_view_component_energy_partition_tree"
        or discrepancy.get("mark_latent_factorization")
        != (
            "beta_binomial_balanced_partition_tree_plus_leaf_"
            "dirichlet_multinomial"
        )
        or discrepancy.get("component_covariance_contract")
        != "direct_scatter_background_moment_propagation_v1"
        or discrepancy.get("detector_green_finite_mc_contract")
        != "pulse_plus_no_pulse_categorical_covariance_all_tree_levels_v1"
        or discrepancy.get("background_mark_contract")
        != "isotope_independent_group_and_within_group_dirichlet_v1"
        or payload["mark_model"]
        != "component_background_source_dirichlet_tree_hierarchical"
        or payload["scatter_shape"] != DETECTOR_CONE_SCATTER_RESPONSE_ID
        or payload["higher_order_scatter_mean"]
        != "excluded_positive_nuisance_owned_by_likelihood"
        or not isinstance(detector_cone_scatter, Mapping)
        or set(detector_cone_scatter)
        != {
            "response",
            "quadrature_order",
            "distance_domain_m",
            "distance_nodes_m",
            "distance_interpolation",
            "single_scatter_conditioning",
            "higher_order_scatter_mean",
            "contract_hash_sha256",
        }
        or detector_cone_scatter.get("response")
        != DETECTOR_CONE_SCATTER_RESPONSE_ID
        or detector_cone_scatter.get("distance_interpolation")
        != "piecewise_linear_in_log_distance"
        or detector_cone_scatter.get("single_scatter_conditioning")
        != (
            "klein_nishina_energy_and_detector_impact_jointly_conditioned_"
            "on_housing_intersection"
        )
        or detector_cone_scatter.get("higher_order_scatter_mean") != "excluded"
        or payload["detector_green_operator_id"] != DETECTOR_GREEN_OPERATOR_ID
        or payload["detector_green_boundary_state"] != DETECTOR_GREEN_BOUNDARY_STATE
        or payload["detector_green_phase_conditioning"]
        != DETECTOR_GREEN_PHASE_CONDITIONING
        or payload["detector_green_finite_mc_uncertainty"]
        != DETECTOR_GREEN_FINITE_MC_UNCERTAINTY
    ):
        raise ValueError(
            "Production full-spectrum manifest is not the canonical "
            "physics-only detector-Green schema-v7 contract."
        )
    _strict_sha256(
        discrepancy["obstacle_material_contract_sha256"],
        field_name=(
            "physical_component_discrepancy.obstacle_material_contract_sha256"
        ),
    )
    _strict_sha256(
        discrepancy["transport_physics_table_contract_sha256"],
        field_name=(
            "physical_component_discrepancy."
            "transport_physics_table_contract_sha256"
        ),
    )
    _strict_sha256(
        detector_cone_scatter["contract_hash_sha256"],
        field_name="detector_cone_scatter_response.contract_hash_sha256",
    )
    distance_nodes = np.asarray(
        detector_cone_scatter["distance_nodes_m"],
        dtype=np.float64,
    )
    distance_domain = np.asarray(
        detector_cone_scatter["distance_domain_m"],
        dtype=np.float64,
    )
    if (
        type(detector_cone_scatter["quadrature_order"]) is not int
        or detector_cone_scatter["quadrature_order"] < 8
        or distance_nodes.ndim != 1
        or distance_nodes.size < 3
        or np.any(~np.isfinite(distance_nodes))
        or np.any(distance_nodes <= 0.0)
        or np.any(np.diff(distance_nodes) <= 0.0)
        or distance_domain.shape != (2,)
        or not np.array_equal(distance_domain, distance_nodes[[0, -1]])
    ):
        raise ValueError("Detector-cone scatter interpolation contract is invalid.")
    _strict_sha256(
        payload["detector_green_operator_contract_sha256"],
        field_name="detector_green_operator_contract_sha256",
    )
    _strict_sha256(
        payload["detector_green_operator_binary_sha256"],
        field_name="detector_green_operator_binary_sha256",
    )


def _full_spectrum_model_protocol(
    model: object,
) -> FullSpectrumGenerativeModel:
    """Return a model implementing the complete full-spectrum protocol."""
    if not isinstance(model, FullSpectrumGenerativeModel):
        raise TypeError(
            "Pure PF requires a FullSpectrumGenerativeModel implementing the "
            "shared NumPy/Torch likelihood, predictive sampler, and manifest."
        )
    return model


def _validate_full_spectrum_contract(
    model: FullSpectrumGenerativeModel,
) -> FullSpectrumGenerativeModel:
    """Validate immutable structure shared by training and production use."""
    _strict_sha256(
        model.contract_hash_sha256,
        field_name="Full-spectrum model contract hash",
    )
    energy_axis = np.asarray(model.energy_axis_keV, dtype=np.float64)
    expected_axis = np.arange(851, dtype=np.float64) * 2.0
    if (
        energy_axis.ndim != 1
        or not np.array_equal(energy_axis, expected_axis)
        or np.any(~np.isfinite(energy_axis))
    ):
        raise ValueError(
            "Full-spectrum schema-v7 energy axis must be the exact 0--1700 "
            "keV, 2 keV-bin runtime axis."
        )
    lines = validated_catalog_transport_lines(
        model,
        energy_axis_keV=energy_axis,
    )
    feature_order = tuple(model.transport_feature_order)
    if any(
        type(value) is not str or not value for value in feature_order
    ) or feature_order != TRANSPORT_FEATURE_ORDER:
        raise ValueError(
            "Full-spectrum transport features must use the canonical "
            "phase-resolved order."
        )
    impact_edges = np.asarray(
        model.detector_impact_parameter_edges_fraction,
        dtype=np.float64,
    )
    detector_radius_m = float(model.detector_target_radius_m)
    if (
        impact_edges.shape != (DETECTOR_IMPACT_PHASE_COUNT + 1,)
        or np.any(~np.isfinite(impact_edges))
        or impact_edges[0] != 0.0
        or impact_edges[-1] != 1.0
        or np.any(np.diff(impact_edges) <= 0.0)
        or not np.isfinite(detector_radius_m)
        or detector_radius_m <= 0.0
    ):
        raise ValueError(
            "Full-spectrum detector Green geometry contract is invalid."
        )
    _validate_schema_v7_manifest(model, lines=lines)
    return model


def validate_nonproduction_full_spectrum_model(
    model: object,
) -> FullSpectrumGenerativeModel:
    """Validate a model for explicit non-production construction or validation."""
    validated = _full_spectrum_model_protocol(model)
    validated.require_runtime_ready()
    runtime_ready = validated.runtime_ready
    if type(runtime_ready) is not bool or runtime_ready is not True:
        raise RuntimeError(
            "Full-spectrum model reported runtime_ready=False after its "
            "explicit nonproduction runtime gate."
        )
    return _validate_full_spectrum_contract(validated)


def validate_full_spectrum_model(
    model: object,
) -> FullSpectrumGenerativeModel:
    """Validate an independently approved model for production PF use."""
    validated = validate_nonproduction_full_spectrum_model(model)
    validated.require_production_ready()
    production_ready = validated.production_ready
    if type(production_ready) is not bool or production_ready is not True:
        raise RuntimeError(
            "Full-spectrum model reported production_ready=False after its "
            "independent all-64 validation gate."
        )
    return validated


def validate_observed_spectrum(
    spectrum_vb: NDArray[np.float64],
    *,
    expected_bin_count: int,
) -> NDArray[np.float64]:
    """Return an unweighted integer-count view-major analysis spectrum.

    The production likelihood models unit-weight detected events.  Fractional,
    efficiency-corrected, or variance-reduced spectra have different sampling
    laws and must remain in explicitly diagnostic analysis paths.
    """
    if isinstance(expected_bin_count, (bool, np.bool_)) or not isinstance(
        expected_bin_count,
        Integral,
    ):
        raise TypeError("expected_bin_count must be an integer.")
    bin_count = int(expected_bin_count)
    if bin_count <= 0:
        raise ValueError("expected_bin_count must be positive.")
    raw_spectrum = np.asarray(spectrum_vb)
    if raw_spectrum.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(
            "Observed full spectra must contain JSON numbers, not values "
            "coercible to numbers."
        )
    spectrum = np.asarray(raw_spectrum, dtype=np.float64)
    if (
        spectrum.ndim != 2
        or int(spectrum.shape[1]) != bin_count
        or int(spectrum.shape[0]) == 0
        or np.any(~np.isfinite(spectrum))
        or np.any(spectrum < 0.0)
    ):
        raise ValueError(
            "Observed full spectra must be finite, nonnegative, nonempty, and "
            "shaped view x model-energy-bin."
        )
    if np.any(spectrum > float(2**53)) or np.any(spectrum != np.rint(spectrum)):
        raise ValueError(
            "Production PF spectra must contain exact unit-weight integer event "
            "counts; weighted, corrected, and fractional spectra are unsupported."
        )
    return np.ascontiguousarray(spectrum)
