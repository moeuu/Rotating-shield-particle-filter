"""Scientific-variant and PF-only posterior aggregation tests."""

from __future__ import annotations

import json
import time
from dataclasses import MISSING, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
import pytest

import pf.estimator_rejuvenation as pf_rejuvenation_module
import pf.estimator_reporting as pf_reporting_module
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kind
from measurement.surface_charts import build_surface_chart_geometry
from runtime.experiment_profiles import STANDARD_EXPERIMENT_PROFILE
from pf.estimator import (
    JointStationObservation,
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
    _stratified_categorical_draws,
    _stratified_joint_cardinality_draws,
)
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    JointRowIdentity,
    StructuralGeometryBatch,
)
from pf.particle_filter_tempering import TemperingIncrementRequiresRejuvenation
from pf.posterior import (
    PFPointEstimate,
    PFPosteriorSnapshot,
    PFSourceMode,
    _surface_mode_medoid_coordinates_batched,
    align_surface_modes_batched,
    posterior_point_estimate_from_states,
)
from pf.posterior_uncertainty import posterior_mode_uncertainty_batched
from pf.profiles import (
    PURE_PF_SCHEMA_VERSION,
    EstimatorProfile,
    apply_profile_to_config,
    enforce_pure_runtime_settings,
    production_compute_backend_values,
    production_pf_config_values,
    resolve_estimator_profile,
    resolve_structural_transition_provenance,
)
from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    EXPLICIT_CARDINALITY_PRIOR_POLICY,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    ContinuousStrengthProposal,
    ContinuousSurfacePositionProposal,
)
from pf.pure_estimator import PurePFEstimator
from pf.state import IsotopeState
from pf.strength_prior import (
    STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY,
    BoundedUniformStrengthPriorTestConfig,
    ShiftedGammaStrengthPriorConfig,
)
from measurement.surface_atlas import ContinuousSurfaceAtlas
from sim.runtime import load_runtime_config
from pure_pf_test_support import approved_full_spectrum_model


def _production_live_settings() -> dict[str, object]:
    """Load a fresh copy of the complete production schema-v2 settings."""
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "configs/pf/pf_strict_3d.json").read_text(encoding="utf-8")
    )


def _sufficient_mixing_diagnostics() -> dict[str, float]:
    """Return explicit evidence satisfying the adaptive mixing contract."""
    return {
        "state_change_weight_mass": 1.0,
        "surface_position_esjd_m2": 1.0,
        "log_strength_esjd": 1.0,
        "ordinary_boundary_weight_mass": 0.0,
        "ordinary_boundary_escape_weight_mass": 0.0,
        "k_transition_weight_mass": 0.0,
    }


def _pure_pf_provenance(
    *,
    measurement_log_sha256: str,
    random_seed: int = 0,
    measurement_log_schema_version: object = 2,
) -> dict[str, object]:
    """Return explicit immutable provenance for a unit-test estimator."""
    return {
        "measurement_log_schema_version": measurement_log_schema_version,
        "config_hash": "a" * 64,
        "resolved_config_hash": "f" * 64,
        "measurement_log_sha256": measurement_log_sha256,
        "random_seed": random_seed,
    }


def test_joint_strength_target_batched_matches_scalar_cache_scaling() -> None:
    """Strength-only batching must equal per-row cached-column scaling."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(joint_strength_block_batch_size=2)
    estimator.isotopes = ("Cs-137",)
    estimator.filters = {"Cs-137": object()}
    estimator._active_joint_tempering_prefix_count = None
    total = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 2, 2, 1)
    estimator._joint_structural_transport_cache = (
        total.copy(),
        0.5 * total,
        np.zeros((3, 2, 2, 1, 1), dtype=np.float64),
    )
    estimator._joint_history_log_likelihood_numpy = lambda **kwargs: np.sum(
        kwargs["total_nvsl"],
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    rows = np.asarray([0, 2], dtype=np.int64)
    scale = np.asarray([[2.0, 0.5], [0.25, 3.0]], dtype=np.float64)

    batched = estimator._joint_strength_block_target(
        [SimpleNamespace(fe_indices=np.asarray([0, 1], dtype=np.int64))],
        particle_indices=rows,
        scale_ps=scale,
        target_beta=1.0,
    )
    scalar = np.asarray(
        [
            np.sum(total[row] * scale[index][None, :, None])
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=0.0)


def test_joint_strength_grid_target_matches_expanded_scalar_oracle() -> None:
    """Fixed-geometry grid batching must match repeated scalar target rows."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(joint_strength_block_batch_size=2)
    estimator.isotopes = ["Cs-137"]
    estimator.filters = {}
    estimator._active_joint_tempering_prefix_count = 1
    stations = (SimpleNamespace(fe_indices=np.asarray([0, 1])),)
    estimator._active_joint_station_history = stations
    total = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 2, 2, 1)
    estimator._joint_structural_transport_cache = (
        total.copy(),
        0.5 * total,
        np.zeros((3, 2, 2, 1, 4), dtype=np.float64),
    )
    estimator._validate_joint_structural_geometry = lambda *_: None
    estimator._full_spectrum_model = lambda: SimpleNamespace(
        line_identity=("Cs-137:661.7",),
        transport_feature_order=(
            "tau_fe",
            "tau_pb",
            "tau_obstacle",
            "distance_m",
        ),
    )
    estimator._joint_line_layout = lambda: {
        "Cs-137": (
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0.4], dtype=np.float64),
        )
    }

    transport_row_counts: list[int] = []

    def _unit_components(**kwargs: object) -> tuple[np.ndarray, ...]:
        """Return deterministic unit transport for each unique geometry."""
        positions = np.asarray(kwargs["positions_s3"], dtype=np.float64)
        transport_row_counts.append(int(positions.shape[0]))
        values = positions[:, 0] + 2.0 * positions[:, 1] + 1.0
        unit = np.broadcast_to(
            values[None, :, None],
            (2, values.size, 1),
        ).copy()
        zeros = np.zeros_like(unit)
        return unit, 0.25 * unit, zeros, zeros, zeros, zeros

    estimator._joint_cached_continuous_unit_components = _unit_components

    def _likelihood(**kwargs: object) -> np.ndarray:
        """Return a deterministic nonlinear row target for equivalence."""
        total_values = np.asarray(kwargs["total_nvsl"], dtype=np.float64)
        uncollided_values = np.asarray(
            kwargs["uncollided_nvsl"],
            dtype=np.float64,
        )
        return np.sum(
            np.log1p(total_values) + 0.2 * uncollided_values,
            axis=(1, 2, 3),
            dtype=np.float64,
        )

    estimator._joint_history_log_likelihood_numpy = _likelihood
    filt = SimpleNamespace(
        isotope="Cs-137",
        config=SimpleNamespace(hard_max_sources=2),
    )
    rows = np.asarray([0, 2], dtype=np.int64)
    positions = np.asarray(
        [
            [[1.0, 0.5, 0.0], [2.0, 0.25, 0.0]],
            [[1.5, 0.75, 0.0], [2.5, 0.5, 0.0]],
        ],
        dtype=np.float64,
    )
    charts = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    strengths = np.asarray(
        [
            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
            [[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        ],
        dtype=np.float64,
    )
    geometry = SimpleNamespace(row_count=2)

    batched = estimator._joint_structural_strength_grid_target_evaluator(
        filt=filt,
        data=geometry,
        positions_pks=positions,
        chart_ids_pk=charts,
        strengths_pgk=strengths,
        particle_indices=rows,
        target_beta=0.75,
        tempering_start_row=0,
    )
    grid_count = int(strengths.shape[1])
    expanded = estimator._joint_structural_target_evaluator(
        filt=filt,
        data=geometry,
        positions_pks=np.repeat(positions, grid_count, axis=0),
        chart_ids_pk=np.repeat(charts, grid_count, axis=0),
        strengths_pk=strengths.reshape(-1, strengths.shape[2]),
        particle_indices=np.repeat(rows, grid_count),
        target_beta=0.75,
        tempering_start_row=0,
    ).reshape(rows.size, grid_count)

    np.testing.assert_allclose(batched, expanded, rtol=1.0e-14, atol=1.0e-14)
    assert transport_row_counts == [
        rows.size * positions.shape[1],
        rows.size * grid_count * positions.shape[1],
    ]


def test_joint_strength_grid_gpu_matches_expanded_scalar_oracle() -> None:
    """CUDA grid broadcasting must preserve the expanded exact target."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the device-resident equivalence test.")
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(joint_strength_block_batch_size=1)
    estimator.isotopes = ["Cs-137"]
    estimator.filters = {}
    estimator._active_joint_tempering_prefix_count = 1
    stations = (SimpleNamespace(fe_indices=np.asarray([0, 1])),)
    estimator._active_joint_station_history = stations
    positions_all = np.asarray(
        [
            [[1.0, 0.5, 0.0], [2.0, 0.25, 0.0]],
            [[1.25, 0.5, 0.0], [2.25, 0.25, 0.0]],
            [[1.5, 0.75, 0.0], [2.5, 0.5, 0.0]],
        ],
        dtype=np.float64,
    )
    charts_all = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    accepted_strengths = np.asarray(
        [[2.0, 3.0], [2.5, 3.5], [3.0, 4.0]],
        dtype=np.float64,
    )
    unit = positions_all[:, None, :, 0] + np.asarray([1.0, 1.5])[None, :, None]
    total = unit[..., None] * accepted_strengths[:, None, :, None]
    uncollided = 0.3 * total
    features = np.broadcast_to(
        np.asarray([0.1, 0.2, 0.3, 1.5], dtype=np.float64),
        total.shape + (4,),
    ).copy()
    device = torch.device("cuda")
    estimator._joint_structural_transport_cache = tuple(
        torch.as_tensor(value, dtype=torch.float64, device=device)
        for value in (total, uncollided, features)
    )
    estimator._validate_joint_structural_geometry = lambda *_: None
    estimator._full_spectrum_model = lambda: SimpleNamespace(
        line_identity=("Cs-137:661.7",),
        transport_feature_order=(
            "tau_fe",
            "tau_pb",
            "tau_obstacle",
            "distance_m",
        ),
    )
    estimator._joint_line_layout = lambda: {
        "Cs-137": (
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0.4], dtype=np.float64),
        )
    }
    estimator._joint_structural_station_geometry_shards = lambda _: (None,)
    estimator.last_joint_structural_unit_cache_hits = 0
    estimator.last_joint_structural_unit_cache_misses = 0

    def _likelihood(**kwargs: object) -> object:
        """Return a nonlinear CUDA target for numerical equivalence."""
        total_values = kwargs["total_nvsl"]
        uncollided_values = kwargs["uncollided_nvsl"]
        beta = float(kwargs["target_beta"])
        return beta * torch.sum(
            torch.log1p(total_values) + 0.2 * uncollided_values,
            dim=(1, 2, 3),
        )

    estimator._joint_history_log_likelihood_torch = _likelihood
    transport_row_counts: list[int] = []

    def _device_components(
        _: object,
        requested_positions: np.ndarray,
        __: np.ndarray,
        **___: object,
    ) -> object:
        """Return deterministic device-resident unit transport components."""
        requested = np.asarray(requested_positions, dtype=np.float64)
        transport_row_counts.append(int(requested.shape[0]))
        values = requested[:, 0] + 2.0 * requested[:, 1] + 1.0
        component = np.broadcast_to(
            values[None, :, None],
            (2, values.size, 1),
        ).copy()
        component_tensor = torch.as_tensor(
            component,
            dtype=torch.float64,
            device=device,
        )
        return SimpleNamespace(
            total_kernel=component_tensor,
            uncollided_kernel=0.25 * component_tensor,
            tau_fe=0.1 * component_tensor,
            tau_pb=0.2 * component_tensor,
            tau_obstacle=0.3 * component_tensor,
            distance_m=1.5 * component_tensor,
        )

    filt = SimpleNamespace(
        isotope="Cs-137",
        config=SimpleNamespace(hard_max_sources=2),
        _packed_continuous_surface_state_arrays=lambda: (
            positions_all,
            accepted_strengths,
            np.ones_like(accepted_strengths, dtype=np.bool_),
            charts_all,
            np.zeros(positions_all.shape[:2] + (2,), dtype=np.float64),
        ),
        _continuous_rj_line_transport_component_columns=_device_components,
    )
    rows = np.asarray([0, 2], dtype=np.int64)
    positions = positions_all[rows].copy()
    positions[:, 1, 0] += 0.125
    charts = charts_all[rows]
    strengths = np.asarray(
        [
            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
            [[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        ],
        dtype=np.float64,
    )
    geometry = SimpleNamespace(row_count=2)

    batched = estimator._joint_structural_strength_grid_target_evaluator(
        filt=filt,
        data=geometry,
        positions_pks=positions,
        chart_ids_pk=charts,
        strengths_pgk=strengths,
        particle_indices=rows,
        target_beta=0.75,
        tempering_start_row=0,
    )
    grid_count = int(strengths.shape[1])
    expanded = estimator._joint_structural_target_evaluator(
        filt=filt,
        data=geometry,
        positions_pks=np.repeat(positions, grid_count, axis=0),
        chart_ids_pk=np.repeat(charts, grid_count, axis=0),
        strengths_pk=strengths.reshape(-1, strengths.shape[2]),
        particle_indices=np.repeat(rows, grid_count),
        target_beta=0.75,
        tempering_start_row=0,
    ).reshape(rows.size, grid_count)

    np.testing.assert_allclose(batched, expanded, rtol=1.0e-13, atol=1.0e-13)
    assert transport_row_counts == [1, 1, rows.size * grid_count]


def test_measurement_record_requires_canonical_runtime_metadata() -> None:
    """PF history records must not expose legacy orientation fallbacks."""
    record_fields = {field.name: field for field in fields(MeasurementRecord)}

    assert "orient_idx" not in record_fields
    for field_name in (
        "spectrum_counts_b",
        "fe_index",
        "pb_index",
        "detector_position_xyz_m",
        "station_sequence_id",
        "station_view_index",
        "generative_contract_hash_sha256",
    ):
        field = record_fields[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING


def _valid_posterior_snapshot() -> PFPosteriorSnapshot:
    """Return one complete strict posterior snapshot for contract tests."""
    return PFPosteriorSnapshot(
        estimator_variant="pf_strict",
        isotopes={
            "Cs-137": PFPointEstimate(
                map_cardinality=0,
                cardinality_distribution={0: 1.0},
                selected_stratum_mass=1.0,
                modes=(),
            )
        },
        planner_belief_sources=("joint_pf_particles",),
        repository_commit="a" * 40,
        measurement_log_schema_version=2,
        config_hash="b" * 64,
        resolved_config_hash="c" * 64,
        measurement_log_sha256="d" * 64,
        random_seed=0,
        profile_capability_map={"posterior_reporting_only": True},
        record_count=0,
        structural_transition_provenance={
            "posterior_semantics": "exact_continuous_surface_rj_smc",
            "structural_kernel_exact_rj": True,
            "structural_kernel_family": "continuous_surface_exact_rj",
            "structural_kernel_target_preserving": True,
            "structural_moves_enabled": True,
            "reversible_jump_mcmc_used": True,
            "support_domain": "environment_surface",
            "variable_cardinality": True,
            "birth_death_moves_enabled": True,
            "within_cardinality_moves_enabled": True,
            "within_cardinality_kernel_exact_mh": True,
        },
        structural_model_manifest={
            "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
            "support_domain": "environment_surface",
            "strength_prior": {
                "minimum_cps_1m": 300_000.0,
                "maximum_cps_1m": 2_000_000.0,
            },
        },
    )


def test_posterior_snapshot_accepts_required_measurement_log() -> None:
    """The posterior must serialize its required MeasurementLog identity."""
    snapshot = _valid_posterior_snapshot()

    payload = snapshot.to_dict()

    assert payload["schema_version"] == 2
    assert payload["pure_pf_schema_version"] == PURE_PF_SCHEMA_VERSION
    assert payload["provenance"]["measurement_log_schema_version"] == 2
    assert payload["provenance"]["measurement_log_sha256"] == "d" * 64
    assert payload["estimator_profile"] == "pf_strict"
    assert "estimator_variant" not in payload
    assert "measurement_log_sha256" not in payload
    assert "resolved_config_hash" not in payload
    assert "resolved_config_sha256" not in payload
    assert "structural_transition_provenance" not in payload


def test_source_mode_uses_only_confidence_explicit_uncertainty_fields() -> None:
    """Posterior modes must not emit the ambiguous legacy uncertainty aliases."""
    mode = PFSourceMode(
        label_index=0,
        position_medoid_xyz=(0.0, 0.0, 0.0),
        position_covariance_xyz=((1.0, 0.0, 0.0),) * 3,
        credible_radius_95_m=1.0,
        strength_representative_cps_1m=1.0,
        strength_mean_cps_1m=1.0,
        strength_median_cps_1m=1.0,
        strength_credible_interval_95_cps_1m=(0.5, 1.5),
        posterior_mass=1.0,
    )

    payload = mode.to_dict()

    assert payload["credible_radius_95_m"] == 1.0
    assert payload["strength_credible_interval_95_cps_1m"] == [0.5, 1.5]
    assert "credible_radius_m" not in payload
    assert "strength_credible_interval_cps_1m" not in payload


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("estimator_variant", 1),
        ("repository_commit", "not-a-commit"),
        ("config_hash", "b" * 63),
        ("resolved_config_hash", 123),
        ("measurement_log_sha256", "g" * 64),
        ("random_seed", "0"),
        ("random_seed", True),
        ("record_count", "0"),
        ("record_count", -1),
        ("planner_belief_sources", ["joint_pf_particles"]),
    ),
)
def test_posterior_snapshot_rejects_coerced_provenance_scalars(
    field_name: str,
    invalid_value: object,
) -> None:
    """Snapshot provenance must reject values that only look valid after casting."""
    snapshot = replace(
        _valid_posterior_snapshot(),
        **{field_name: invalid_value},
    )

    with pytest.raises(ValueError):
        snapshot.to_dict()


@pytest.mark.parametrize("invalid_value", ("false", 1, np.bool_(True)))
def test_posterior_snapshot_rejects_truthy_structural_boolean_substitutes(
    invalid_value: object,
) -> None:
    """A truthy string or integer must not claim exact target preservation."""
    provenance = dict(_valid_posterior_snapshot().structural_transition_provenance)
    provenance["structural_kernel_target_preserving"] = invalid_value
    snapshot = replace(
        _valid_posterior_snapshot(),
        structural_transition_provenance=provenance,
    )

    with pytest.raises(ValueError, match="JSON boolean"):
        snapshot.to_dict()


@pytest.mark.parametrize("invalid_value", ("false", 1, np.bool_(True)))
def test_posterior_snapshot_rejects_truthy_capability_substitutes(
    invalid_value: object,
) -> None:
    """Profile capabilities must remain exact booleans at the report boundary."""
    snapshot = replace(
        _valid_posterior_snapshot(),
        profile_capability_map={"posterior_reporting_only": invalid_value},
    )

    with pytest.raises(ValueError, match="JSON boolean"):
        snapshot.to_dict()


@pytest.mark.parametrize(
    "estimate",
    (
        PFPointEstimate(
            map_cardinality="0",  # type: ignore[arg-type]
            cardinality_distribution={0: 1.0},
            selected_stratum_mass=1.0,
            modes=(),
        ),
        PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution={0: "1.0"},  # type: ignore[dict-item]
            selected_stratum_mass=1.0,
            modes=(),
        ),
        PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution={0: 1.0},
            selected_stratum_mass="1.0",  # type: ignore[arg-type]
            modes=(),
        ),
    ),
)
def test_posterior_snapshot_rejects_coerced_point_estimate_probabilities(
    estimate: PFPointEstimate,
) -> None:
    """The final artifact must not cast textual posterior mass into evidence."""
    snapshot = replace(
        _valid_posterior_snapshot(),
        isotopes={"Cs-137": estimate},
    )

    with pytest.raises(ValueError):
        snapshot.to_dict()


def _exact_rj_config(**overrides: object) -> RotatingShieldPFConfig:
    """Build an exact finite-surface RJ-MH config for focused tests."""
    values: dict[str, object] = {
        "estimator_profile": "pf_strict",
        "max_sources": 5,
        "init_num_sources": (0, 5),
        "strength_prior": BoundedUniformStrengthPriorTestConfig(
            minimum_cps_1m=1.0,
            maximum_cps_1m=2_000_000.0,
        ),
    }
    values.update(overrides)
    return RotatingShieldPFConfig(**values)


def _surface_state(
    particle_filter: object,
    positions: np.ndarray,
    strengths: np.ndarray,
) -> IsotopeState:
    """Build one authoritative state from exact on-surface test positions."""
    position_array = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    strength_array = np.asarray(strengths, dtype=np.float64).reshape(-1)
    chart_ids, surface_uv = particle_filter.structural_surface_chart_coordinates(
        position_array
    )
    return IsotopeState(
        num_sources=int(position_array.shape[0]),
        strengths=strength_array,
        surface_chart_ids=chart_ids,
        surface_uv=surface_uv,
    )


def _joint_row_identity_estimator(
    *,
    particle_count: int = 4,
    random_seed: int = 73,
) -> PurePFEstimator:
    """Build a small initialized joint PF for row-identity tests."""
    isotopes = ("Co-60", "Cs-137")
    estimator = PurePFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=particle_count,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(2.0, 2.0, 2.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(
            measurement_log_sha256="e" * 64,
            random_seed=random_seed,
        ),
    )
    estimator.add_measurement_pose(np.asarray([1.0, 1.0, 1.0], dtype=np.float64))
    estimator._ensure_kernel_cache()
    return estimator


def test_joint_row_identity_rejects_uniform_weight_isotope_permutation() -> None:
    """A row permutation must fail before it changes joint K posterior mass."""
    estimator = _joint_row_identity_estimator()
    cardinalities = {
        "Co-60": (0, 0, 1, 1),
        "Cs-137": (0, 0, 1, 1),
    }
    for isotope, isotope_cardinalities in cardinalities.items():
        filt = estimator.filters[isotope]
        for row, cardinality in enumerate(isotope_cardinalities):
            positions = (
                np.asarray([[float(row % 2), 1.0, 0.0]], dtype=np.float64)
                if cardinality
                else np.empty((0, 3), dtype=np.float64)
            )
            strengths = (
                np.asarray([100.0 + row], dtype=np.float64)
                if cardinality
                else np.empty(0, dtype=np.float64)
            )
            filt.continuous_particles[row].state = _surface_state(
                filt,
                positions,
                strengths,
            )
    assert estimator.posterior_joint_cardinality_distribution() == (
        pytest.approx({(0, 0): 0.5, (1, 1): 0.5})
    )

    cs_particles = estimator.filters["Cs-137"].continuous_particles
    estimator.filters["Cs-137"].continuous_particles = [
        cs_particles[index] for index in (2, 3, 0, 1)
    ]

    with pytest.raises(RuntimeError, match="Joint row identity"):
        estimator.posterior_joint_cardinality_distribution()


def test_joint_row_identity_initialization_is_shared_unique_and_immutable() -> None:
    """Initial joint rows must share one authenticated positional identity."""
    first = _joint_row_identity_estimator(random_seed=81)
    repeated = _joint_row_identity_estimator(random_seed=81)
    identity_vectors: list[tuple[JointRowIdentity, ...]] = []
    for isotope in first.joint_isotope_order():
        identities = tuple(
            particle.joint_row_identity
            for particle in first.filters[isotope].continuous_particles
        )
        assert all(isinstance(identity, JointRowIdentity) for identity in identities)
        identity_vectors.append(identities)
    assert identity_vectors[0] == identity_vectors[1]
    assert len({identity.row_sha256 for identity in identity_vectors[0]}) == 4
    assert [identity.ordinal for identity in identity_vectors[0]] == list(range(4))
    assert {identity.generation for identity in identity_vectors[0]} == {0}
    repeated_identities = tuple(
        particle.joint_row_identity
        for particle in repeated.filters["Co-60"].continuous_particles
    )
    assert repeated_identities == identity_vectors[0]
    particle = first.filters["Co-60"].continuous_particles[0]
    with pytest.raises(AttributeError, match="immutable"):
        particle.joint_row_identity = identity_vectors[1][0]
    object.__setattr__(
        identity_vectors[0][0],
        "row_sha256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="digest"):
        first._assert_joint_particle_alignment()


def test_joint_resample_creates_unique_children_for_duplicate_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate ancestors need distinct new row IDs shared by all isotopes."""
    estimator = _joint_row_identity_estimator()
    parent_identities = tuple(
        particle.joint_row_identity
        for particle in estimator.filters["Co-60"].continuous_particles
    )
    indices = np.asarray([1, 1, 3, 1], dtype=np.int64)

    def _duplicate_resample(
        log_weights: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a fixed ancestor vector containing duplicate rows."""
        assert log_weights.shape == (4,)
        assert isinstance(rng, np.random.Generator)
        return indices.copy()

    monkeypatch.setattr(
        pf_rejuvenation_module,
        "systematic_resample",
        _duplicate_resample,
    )
    result = estimator._resample_joint_particles(
        np.full(4, -np.log(4.0), dtype=np.float64)
    )

    np.testing.assert_array_equal(result, indices)
    isotope_identities = {
        isotope: tuple(
            particle.joint_row_identity
            for particle in estimator.filters[isotope].continuous_particles
        )
        for isotope in estimator.joint_isotope_order()
    }
    assert isotope_identities["Co-60"] == isotope_identities["Cs-137"]
    children = isotope_identities["Co-60"]
    assert len({identity.row_sha256 for identity in children}) == 4
    assert [identity.ordinal for identity in children] == list(range(4))
    assert {identity.generation for identity in children} == {1}
    assert [identity.parent_row_sha256 for identity in children] == [
        parent_identities[int(index)].row_sha256 for index in indices
    ]
    duplicated_states = estimator.filters["Co-60"].continuous_particles
    assert duplicated_states[0].state is not duplicated_states[1].state
    estimator._assert_joint_particle_alignment()
    checkpoint = estimator.serialized_state()
    assert all(
        identity.row_sha256.encode("ascii") in checkpoint for identity in children
    )


def _stable_fixed_k_estimator() -> RotatingShieldPFEstimator:
    """Build a fixed-K pure PF with a degenerate stable posterior."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=float,
        ),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=True,
            gpu_device="cpu",
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    particle_filter = estimator.filters["Cs-137"]
    for particle in particle_filter.continuous_particles:
        particle.state = _surface_state(
            particle_filter,
            np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            np.asarray([10.0], dtype=float),
        )
        particle.log_weight = float(np.log(0.5))
    return estimator


def test_exact_posterior_summary_is_cached_per_state_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated report consumers must share one exact intrinsic medoid."""
    estimator = _stable_fixed_k_estimator()
    estimator._invalidate_posterior_summary_cache()
    original = pf_reporting_module.posterior_point_estimate_from_states
    calls: list[int] = []

    def counted_report(*args: object, **kwargs: object) -> PFPointEstimate:
        """Count exact report evaluations while preserving their result."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pf_reporting_module,
        "posterior_point_estimate_from_states",
        counted_report,
    )
    first = estimator.estimates()
    second = estimator.estimates()

    assert len(calls) == 1
    np.testing.assert_array_equal(first["Cs-137"][0], second["Cs-137"][0])
    np.testing.assert_array_equal(first["Cs-137"][1], second["Cs-137"][1])


@pytest.mark.parametrize(
    "isotopes",
    (
        (),
        ("Cs-137", "Cs-137"),
        ("Cs-137", ""),
        ("Cs-137", 137),
    ),
)
def test_joint_estimator_rejects_invalid_isotope_identity(
    isotopes: tuple[object, ...],
) -> None:
    """Duplicate or coerced isotope identities cannot define a joint target."""
    with pytest.raises(ValueError, match="unique nonempty strings"):
        RotatingShieldPFEstimator(
            isotopes=isotopes,
            surface_diagnostic_points=np.asarray(
                [[0.0, 0.0, 0.0]],
                dtype=float,
            ),
            shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            mu_by_isotope={"Cs-137": 0.5},
            pf_config=RotatingShieldPFConfig(
                num_particles=2,
                max_sources=1,
                variable_cardinality=False,
                init_num_sources=(1, 1),
                use_gpu=True,
                gpu_device="cpu",
            ),
            full_spectrum_generative_model=approved_full_spectrum_model(),
        )


@pytest.mark.parametrize(
    "profile",
    ["pf_strict", EstimatorProfile.PF_STRICT],
)
def test_only_strict_profile_is_supported(
    profile: EstimatorProfile | str,
) -> None:
    """The canonical name must resolve to the single strict PF profile."""
    resolved_profile, capabilities = resolve_estimator_profile(profile)

    assert resolved_profile is EstimatorProfile.PF_STRICT
    assert capabilities.sequential_updates_only is True
    assert capabilities.posterior_reporting_only is True
    assert capabilities.likelihood_consistent_structural_evidence is True


@pytest.mark.parametrize("profile", [None, "strict", "pure_pf", "pf_only"])
def test_profile_aliases_are_not_part_of_the_runtime_schema(
    profile: object,
) -> None:
    """Only the explicit canonical profile belongs to the runtime schema."""
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        resolve_estimator_profile(profile)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "schema_version",
    [
        None,
        0,
        1,
        True,
        "2",
    ],
)
def test_runtime_requires_exact_pure_pf_schema_version(
    schema_version: object,
) -> None:
    """Runtime configuration must explicitly select pure-PF schema version 2."""
    payload = _production_live_settings()
    if schema_version is not None:
        payload["pure_pf_schema_version"] = schema_version
    else:
        del payload["pure_pf_schema_version"]
    with pytest.raises(ValueError, match="pure_pf_schema_version=2|missing"):
        enforce_pure_runtime_settings(payload)


def test_runtime_accepts_the_positive_pure_pf_schema() -> None:
    """The versioned schema must preserve the canonical strict profile."""
    payload = _production_live_settings()
    resolved = enforce_pure_runtime_settings(payload)

    assert resolved["pure_pf_schema_version"] == PURE_PF_SCHEMA_VERSION
    assert resolved["estimator_profile"] == "pf_strict"
    assert "variable_cardinality" not in resolved
    assert "init_num_sources" not in resolved
    assert "structural_cardinality_prior_policy" not in resolved
    assert "structural_cardinality_prior_probs" not in resolved
    assert "joint_guided_initialization" not in resolved
    assert "gpu_dtype" not in resolved
    assert resolved["strength_prior"] == {
        "kind": "shifted_gamma",
        "minimum_cps_1m": 300_000.0,
        "shape": 2.0,
        "scale_cps_1m": 425_000.0,
    }


def test_production_strength_prior_rejects_retired_maximum_fields() -> None:
    """No top-level or nested ignored maximum may enter production."""
    payload = _production_live_settings()
    payload["strength_prior_max_cps_1m"] = 2_000_000.0
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)

    payload = _production_live_settings()
    strength_prior = payload["strength_prior"]
    assert isinstance(strength_prior, dict)
    strength_prior["maximum_cps_1m"] = 2_000_000.0
    with pytest.raises(ValueError, match="strength_prior.*unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


def test_production_strength_prior_rejects_test_only_bounded_family() -> None:
    """The bounded test oracle must be unreachable from production live."""
    payload = _production_live_settings()
    payload["strength_prior"] = {
        "kind": "bounded_uniform_test_only",
        "minimum_cps_1m": 1.0,
        "maximum_cps_1m": 2.0,
    }

    with pytest.raises(ValueError, match="strength_prior"):
        enforce_pure_runtime_settings(payload)


def test_runtime_schema_requires_the_canonical_profile() -> None:
    """The version marker must be paired with the canonical estimator profile."""
    payload = _production_live_settings()
    del payload["estimator_profile"]
    with pytest.raises(ValueError, match="missing=.*estimator_profile"):
        enforce_pure_runtime_settings(payload)


def test_explicit_profile_cannot_override_runtime_schema() -> None:
    """A caller profile must not replace an invalid logged profile."""
    payload = _production_live_settings()
    payload["estimator_profile"] = "removed-profile"
    with pytest.raises(ValueError, match="only 'pf_strict' is available"):
        enforce_pure_runtime_settings(
            payload,
            profile="pf_strict",
        )


def test_runtime_schema_rejects_unknown_pf_settings() -> None:
    """Unknown PF-prefixed settings must fail instead of becoming no-ops."""
    payload = _production_live_settings()
    payload["pf_unknown_transition"] = True
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "structural_rj_mov_probability",
        "structural_cardinality_prior_probability",
    ],
)
def test_runtime_schema_rejects_unknown_structural_settings(
    field_name: str,
) -> None:
    """Typos in exact-RJ controls must fail before a runtime starts."""
    payload = _production_live_settings()
    payload[field_name] = 1.0
    with pytest.raises(ValueError, match="structural"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("max_sources", None),
        ("max_sources", 0),
        ("hard_max_sources", None),
        ("num_particles", 6),
    ],
)
def test_runtime_schema_rejects_malformed_cardinality_settings(
    field_name: str,
    field_value: object,
) -> None:
    """Cardinality controls must not be silently coerced or clamped."""
    payload = _production_live_settings()
    payload[field_name] = field_value
    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("structural_rj_local_position_move_probability", 0.0),
        ("structural_rj_split_merge_probability", 0.0),
        ("structural_rj_multi_component_probability", 0.0),
        ("joint_strength_block_probability", 0.0),
        ("joint_cross_isotope_state_block_probability", 0.0),
        ("joint_rejuvenation_min_state_change_weight_mass", 0.0),
        ("joint_rejuvenation_min_surface_esjd_m2", 0.0),
        ("joint_rejuvenation_min_log_strength_esjd", 0.0),
        ("joint_rejuvenation_min_k_transition_weight_mass", 0.0),
        ("joint_rejuvenation_boundary_mass_threshold", 1.0),
    ),
)
def test_production_schema_rejects_disabled_pf_kernel_or_gate(
    field_name: str,
    field_value: object,
) -> None:
    """Production settings cannot retain subordinate no-op kernel controls."""
    payload = _production_live_settings()
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("structural_rj_birth_probability", 0.25),
        ("structural_rj_split_probability", 0.25),
        ("structural_rj_position_proposal_prior_weight", 1.0),
        ("structural_rj_strength_proposal_prior_weight", 1.0),
        ("structural_rj_split_global_position_probability", 1.0),
        ("structural_rj_merge_uniform_pair_probability", 1.0),
        ("joint_guided_initialization_prior_row_probability", 1.0),
    ),
)
def test_production_schema_rejects_noncanonical_mixture_weights(
    field_name: str,
    field_value: object,
) -> None:
    """Production mixtures must retain both declared branches and exact mass."""
    payload = _production_live_settings()
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(payload)


def test_production_schema_rejects_inactive_capacity_tail() -> None:
    """The fixed geometric tail must occupy real cardinality support."""
    payload = _production_live_settings()
    payload["hard_max_sources"] = payload["max_sources"]

    with pytest.raises(ValueError, match="hard_max_sources"):
        enforce_pure_runtime_settings(payload)


def test_production_schema_rejects_silently_capped_multi_component_group() -> None:
    """A configured RJ group size may not be clamped to estimator capacity."""
    payload = _production_live_settings()
    payload["structural_rj_multi_component_max_group_size"] = 9

    with pytest.raises(ValueError, match="multi_component_max_group_size"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    "field_name",
    (
        "variable_cardinality",
        "init_num_sources",
        "structural_cardinality_prior_policy",
        "structural_cardinality_prior_probs",
        "joint_guided_initialization",
        "gpu_dtype",
        "use_gpu",
        "gpu_device",
    ),
)
def test_runtime_schema_rejects_fixed_internal_pf_settings(field_name: str) -> None:
    """Production callers cannot override invariants owned by the live builder."""
    payload = _production_live_settings()
    payload[field_name] = None

    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    "backend",
    (
        {"kind": "cuda_float64", "device": " cuda "},
        {"kind": "cuda_float64", "device": "CUDA"},
        {"kind": "cuda_float64", "device": "cuda:00"},
        {"kind": "numpy_float64"},
        {"kind": "numpy_float64", "device": "cpu"},
        {"kind": "unknown"},
    ),
)
def test_runtime_schema_rejects_noncanonical_compute_backend(
    backend: dict[str, object],
) -> None:
    """The backend union must reject aliases and inactive device fields."""
    payload = _production_live_settings()
    payload["compute_backend"] = backend

    with pytest.raises(ValueError, match="compute_backend"):
        enforce_pure_runtime_settings(payload)


def test_runtime_schema_rejects_serving_a_disabled_cui() -> None:
    """A disabled renderer cannot silently ignore an enabled HTTP server."""
    payload = _production_live_settings()
    payload["cui_split_view"] = False

    with pytest.raises(ValueError, match="cui_split_view_serve requires"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize("renderer_enabled", (False, True))
def test_runtime_schema_accepts_explicit_nonserving_cui_state(
    renderer_enabled: bool,
) -> None:
    """A non-serving CUI must represent every network control as null."""
    payload = _production_live_settings()
    payload["cui_split_view"] = renderer_enabled
    payload["cui_split_view_serve"] = False
    payload["cui_split_view_host"] = None
    payload["cui_split_view_port"] = None
    payload["cui_split_view_public_host"] = None
    if not renderer_enabled:
        payload["cui_split_view_save_step_history"] = False
        payload["cui_split_view_max_particles_per_isotope"] = None

    result = enforce_pure_runtime_settings(payload)

    assert result["cui_split_view"] is renderer_enabled
    assert result["cui_split_view_serve"] is False
    assert result["cui_split_view_host"] is None
    assert result["cui_split_view_port"] is None
    assert result["cui_split_view_public_host"] is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("cui_split_view_host", "127.0.0.1"),
        ("cui_split_view_port", 8877),
        ("cui_split_view_public_host", "127.0.0.1"),
    ),
)
def test_runtime_schema_rejects_nonserving_cui_network_settings(
    field_name: str,
    field_value: object,
) -> None:
    """Network settings cannot remain dormant behind serve=false."""
    payload = _production_live_settings()
    payload["cui_split_view_serve"] = False
    payload["cui_split_view_host"] = None
    payload["cui_split_view_port"] = None
    payload["cui_split_view_public_host"] = None
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("cui_split_view_save_step_history", True),
        ("cui_split_view_max_particles_per_isotope", 1),
    ),
)
def test_runtime_schema_rejects_disabled_cui_renderer_settings(
    field_name: str,
    field_value: object,
) -> None:
    """Disabled renderers cannot retain ignored visualization settings."""
    payload = _production_live_settings()
    payload["cui_split_view"] = False
    payload["cui_split_view_serve"] = False
    payload["cui_split_view_host"] = None
    payload["cui_split_view_port"] = None
    payload["cui_split_view_public_host"] = None
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("cui_split_view_host", "http://127.0.0.1"),
        ("cui_split_view_public_host", "bad/path"),
        ("cui_split_view_public_host", "auto"),
    ),
)
def test_runtime_schema_rejects_invalid_or_automatic_cui_hosts(
    field_name: str,
    field_value: str,
) -> None:
    """CUI host errors must fail before runtime connection or server startup."""
    payload = _production_live_settings()
    payload[field_name] = field_value

    with pytest.raises(ValueError, match="CUI|cui"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize("field_value", (None, True, 0, 65536))
def test_runtime_schema_rejects_invalid_serving_cui_port(
    field_value: object,
) -> None:
    """The serving CUI port must be an explicit valid TCP port integer."""
    payload = _production_live_settings()
    payload["cui_split_view_port"] = field_value

    with pytest.raises(ValueError, match="cui_split_view_port"):
        enforce_pure_runtime_settings(payload)


def test_runtime_schema_rejects_inert_visual_particle_cap() -> None:
    """A CUI cap larger than the PF population must not be accepted as active."""
    payload = _production_live_settings()
    payload["cui_split_view_max_particles_per_isotope"] = 4097

    with pytest.raises(ValueError, match="must not exceed num_particles"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    "retired_key",
    [
        "adaptive_mission_stop",
        "birth_enable",
        "candidate_verification_queue_enable",
        "calibration_count_method",
        "continuous_surface_chart_max_edge_m",
        "converge_min_ess_ratio",
        "converge_cardinality_var_max",
        "credible_surface_radius_threshold_m",
        "count_likelihood_model",
        "delayed_resample_update",
        "detector_height_sampling_mode",
        "history_estimate_interval",
        "init_num_sources_min",
        "joint_observation_update",
        "measurement_pose_clearance_enabled",
        "path_planner",
        "pose_selection_workers",
        "python_worker_count",
        "roughening_k",
        "sparse_poisson_evidence_enable",
        "spectrum_likelihood_bin_chunk",
        "structural_rj_patch_spacing_m",
        "surface_observability_diagnostic_candidates",
        "refit_after_moves",
        "response_poisson_count_variance_ceiling_enable",
        "spectrum_count_method",
    ],
)
def test_runtime_schema_rejects_retired_estimator_settings(
    retired_key: str,
) -> None:
    """Deleted estimator generations must not survive as silent no-op keys."""
    payload = _production_live_settings()
    payload[retired_key] = True
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


@pytest.mark.parametrize(
    "retired_key",
    [
        "adaptive_program_length_enable",
        "beam_width",
        "global_surface_rescue_mode_weight",
        "horizon",
        "one_step_guard_enable",
        "one_step_guard_score_abs_margin",
        "one_step_guard_score_rel_margin",
        "one_step_guard_use_gpu",
        "recovery_isotope_mode_weight_multiplier",
        "residual_program_length",
        "same_isotope_direct_separation_guard",
        "typo_weight",
    ],
)
def test_runtime_schema_rejects_retired_dss_settings(
    retired_key: str,
) -> None:
    """Deleted DSS rescue and heuristic settings must fail closed."""
    payload = _production_live_settings()
    payload["dss_pp"][retired_key] = True
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


def test_runtime_schema_rejects_retired_remaining_measurement_block() -> None:
    """The physically deleted remaining-measurement module must fail closed."""
    payload = _production_live_settings()
    payload["remaining_measurement_estimate"] = {"enabled": True}
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


def test_runtime_schema_rejects_deleted_pair_categorical_response() -> None:
    """The deleted pair-categorical count-response path must fail closed."""
    payload = _production_live_settings()
    payload["pf_transport_response_model"] = {
        "enabled": True,
        "model": "log_tau_regression_v1",
        "feature_semantics": "canonical",
        "by_isotope": {
            "Cs-137": {"tau_coefficients": {"shield_tau": 0.1}}
        },
    }
    with pytest.raises(ValueError, match="unknown_or_retired"):
        enforce_pure_runtime_settings(payload)


def test_fixed_k_provenance_declares_target_preserving_mh_kernel() -> None:
    """Fixed-K provenance must declare exact within-cardinality MH moves."""
    fixed_config = RotatingShieldPFConfig(
        estimator_profile="pf_strict",
        init_num_sources=(3, 3),
        variable_cardinality=False,
    )
    fixed_capabilities = apply_profile_to_config(fixed_config)
    fixed = resolve_structural_transition_provenance(
        fixed_config,
        capabilities=fixed_capabilities,
    ).to_dict()

    assert fixed["posterior_semantics"] == (
        "fixed_cardinality_sequential_particle_filter_with_"
        "target_preserving_mh_rejuvenation"
    )
    assert fixed["structural_kernel_family"] == (
        "fixed_cardinality_surface_position_strength_mh"
    )
    assert fixed["structural_moves_enabled"] is True
    assert fixed["variable_cardinality"] is False
    assert fixed["birth_death_moves_enabled"] is False
    assert fixed["within_cardinality_moves_enabled"] is True
    assert fixed["within_cardinality_kernel_exact_mh"] is True
    assert fixed["structural_kernel_target_preserving"] is True
    assert fixed["structural_kernel_exact_rj"] is False
    assert fixed["reversible_jump_mcmc_used"] is False
    assert fixed["structural_evidence_uses_pf_likelihood"] is True


def test_exact_rj_provenance_declares_target_preserving_pf_kernel() -> None:
    """Exact RJ-MH mode must report continuous-surface posterior semantics."""
    config = _exact_rj_config(
        init_num_sources=(0, 5),
        variable_cardinality=True,
    )
    capabilities = apply_profile_to_config(config)
    provenance = resolve_structural_transition_provenance(
        config,
        capabilities=capabilities,
    ).to_dict()

    assert provenance["posterior_semantics"] == (
        "sequential_particle_filter_with_target_preserving_rj_mh_rejuvenation"
    )
    assert provenance["structural_kernel_family"] == (
        "continuous_surface_birth_death_split_merge_rj_mh"
    )
    assert provenance["structural_moves_enabled"] is True
    assert provenance["variable_cardinality"] is True
    assert provenance["birth_death_moves_enabled"] is True
    assert provenance["within_cardinality_moves_enabled"] is True
    assert provenance["within_cardinality_kernel_exact_mh"] is True
    assert provenance["structural_kernel_target_preserving"] is True
    assert provenance["structural_kernel_exact_rj"] is True
    assert provenance["reversible_jump_mcmc_used"] is True
    assert provenance["structural_evidence_uses_pf_likelihood"] is True


def test_surface_credible_radius_does_not_collapse_on_a_broad_plane() -> None:
    """Broad floor support must not look converged because its 3-D determinant is zero."""
    estimator = _stable_fixed_k_estimator()
    particle_filter = estimator.filters["Cs-137"]
    particle_filter.continuous_particles[0].state = _surface_state(
        particle_filter,
        np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        np.asarray([10.0], dtype=float),
    )
    particle_filter.continuous_particles[1].state = _surface_state(
        particle_filter,
        np.asarray([[5.0, 0.0, 0.0]], dtype=float),
        np.asarray([10.0], dtype=float),
    )
    radii = estimator.credible_surface_radii()
    diagnostics = estimator.posterior_convergence_diagnostics()

    assert radii["Cs-137"][0] >= 5.0
    assert (
        diagnostics["isotopes"]["Cs-137"]["gates"]["surface_path_concentration"]
        is False
    )
    assert diagnostics["ready"] is False


def test_convergence_consumes_native_full_spectrum_innovation_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime model diagnostics must feed stopping without schema drift."""
    import torch

    estimator = _stable_fixed_k_estimator()
    model = estimator._full_spectrum_model()
    line_count = len(tuple(model.line_identity))
    bin_count = int(np.asarray(model.energy_axis_keV).size)
    station = JointStationObservation(
        spectrum_vb=np.zeros((1, bin_count), dtype=np.float64),
        energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(0.5, 0.0, 0.0),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([30.0], dtype=np.float64),
        station_sequence_id=0,
    )
    estimator._joint_station_history = [station]
    total = torch.zeros((2, 1, 1, line_count), dtype=torch.float64)
    features = torch.zeros(
        (2, 1, 1, line_count, 4),
        dtype=torch.float64,
    )
    monkeypatch.setattr(
        estimator,
        "_joint_station_transport_components_torch",
        lambda active_station: (total, total.clone(), features),
    )

    innovation = estimator._latest_joint_station_innovation()
    predictive = estimator.posterior_predictive_check(
        sample_count=4,
        confidence=0.95,
        worst_bin_count=2,
    )
    convergence = estimator.posterior_convergence_diagnostics()

    assert innovation["available"] is True
    assert innovation["view_count"] == 1
    assert innovation["dimension"] == bin_count
    assert innovation["renewal_total_max_abs_z"] is not None
    assert "conditional_mark_tail_probability" in innovation
    assert predictive["available"] is True
    assert predictive["sample_count"] == 4
    assert predictive["stations"][0]["view_count"] == 1
    assert "0" in predictive["shield_pair_summary"]
    assert "Cs-137" in predictive["isotope_response_ablation_summary"]
    assert len(predictive["worst_standardized_bin_residuals"]) == 2
    assert convergence["innovation"] == innovation


def test_convergence_reports_current_ess_without_using_it_as_a_stop_gate() -> None:
    """Current ESS is particle adequacy, not physical posterior convergence."""
    estimator = _stable_fixed_k_estimator()
    particle_filter = estimator.filters["Cs-137"]
    particle_filter.continuous_particles[0].log_weight = float(np.log(0.999))
    particle_filter.continuous_particles[1].log_weight = float(np.log(0.001))
    particle_filter.last_ess_post = 2.0

    diagnostics = estimator.posterior_convergence_diagnostics()
    isotope = diagnostics["isotopes"]["Cs-137"]

    assert isotope["current_ess_ratio"] < 0.8
    assert "current_ess" not in isotope["gates"]
    assert diagnostics["ready"] is False


def test_convergence_fails_closed_when_joint_rejuvenation_is_incomplete() -> None:
    """An under-mixed exact-RJ generation must never authorize stopping."""
    estimator = _stable_fixed_k_estimator()
    estimator.last_joint_rejuvenation_mixing_incomplete = True

    diagnostics = estimator.posterior_convergence_diagnostics()

    assert diagnostics["sampler_health"]["rejuvenation_mixing_complete"] is False
    assert diagnostics["joint_gates"]["rejuvenation_mixing_complete"] is False
    assert diagnostics["ready"] is False


def test_variable_cardinality_cannot_converge_at_the_truncation_boundary() -> None:
    """Material mass at max_sources must remain an unresolved model-order boundary."""
    estimator = _stable_fixed_k_estimator()
    particle_filter = estimator.filters["Cs-137"]
    estimator.pf_config.variable_cardinality = True
    particle_filter.config.variable_cardinality = True

    diagnostics = estimator.posterior_convergence_diagnostics()
    isotope = diagnostics["isotopes"]["Cs-137"]

    assert isotope["maximum_cardinality_boundary_mass"] == pytest.approx(1.0)
    assert isotope["gates"]["cardinality_not_at_upper_boundary"] is False
    assert diagnostics["ready"] is False


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("structural_rj_surface_chart_max_edge_m", 0.0),
        ("structural_rj_move_probability", -0.1),
        ("structural_rj_birth_probability", 1.1),
        ("structural_rj_death_probability", float("nan")),
        ("structural_rj_position_move_probability", -1.0),
        ("structural_rj_position_proposal_prior_weight", 0.0),
        ("structural_rj_local_position_move_probability", 1.1),
        ("structural_rj_strength_move_probability", 2.0),
        ("structural_rj_split_global_position_probability", 0.0),
        ("structural_rj_merge_uniform_pair_probability", 0.0),
        ("structural_rj_merge_distance_sigma_m", 0.0),
        ("adaptive_stop_maximum_surface_path_radius_95_m", -0.1),
        (
            "adaptive_stop_minimum_joint_map_cardinality_probability",
            1.1,
        ),
        ("adaptive_stop_maximum_upper_cardinality_mass", -0.1),
        ("adaptive_stop_innovation_confidence", float("nan")),
        ("joint_rejuvenation_boundary_mass_threshold", -0.1),
    ],
)
def test_exact_rj_numeric_configuration_is_validated(
    field_name: str,
    field_value: float,
) -> None:
    """RJ-MH max_edge_m and attempt probabilities must stay in their domains."""
    with pytest.raises(ValueError, match=field_name):
        _exact_rj_config(**{field_name: field_value})


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("variable_cardinality", "false"),
        ("use_gpu", 1),
        ("num_particles", 12.0),
        ("max_sources", 5.0),
        ("structural_rj_strength_proposal_grid_size", 5.0),
        ("structural_rj_move_probability", "1.0"),
        ("structural_cardinality_prior_probs", ["1.0"] * 6),
    ),
)
def test_pf_configuration_rejects_semantic_type_coercion(
    field_name: str,
    field_value: object,
) -> None:
    """Truthy strings and integral floats must not alter the PF state model."""
    with pytest.raises((TypeError, ValueError)):
        RotatingShieldPFConfig(**{field_name: field_value})


def test_structural_cardinality_prior_is_positive_and_canonical() -> None:
    """Explicit cardinality masses must already be a normalized distribution."""
    with pytest.raises(ValueError, match="sum exactly to 1"):
        _exact_rj_config(
            max_sources=2,
            init_num_sources=(0, 2),
            structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
            structural_cardinality_prior_probs=[1.0, 2.0, 3.0],
        )

    config = _exact_rj_config(
        max_sources=2,
        init_num_sources=(0, 2),
        structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
        structural_cardinality_prior_probs=[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0],
    )
    assert config.structural_cardinality_prior_probs == pytest.approx(
        (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    )

    with pytest.raises(ValueError, match="structural_cardinality_prior_probs"):
        _exact_rj_config(
            max_sources=1,
            init_num_sources=(0, 1),
            structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
            structural_cardinality_prior_probs=[1.0, 0.0],
        )


def test_cardinality_prior_policy_must_match_parameterization() -> None:
    """A named pre-evaluation K policy must bind either its mean or vector."""
    with pytest.raises(ValueError, match="cannot be combined"):
        _exact_rj_config(
            structural_cardinality_prior_policy=(
                TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
            ),
            structural_cardinality_prior_probs=[1.0] * 6,
        )
    with pytest.raises(ValueError, match="requires"):
        _exact_rj_config(
            structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
            structural_cardinality_prior_probs=None,
        )


def test_cardinality_prior_normalization_is_byte_exact_idempotent(
) -> None:
    """Repeated configuration validation must preserve normalized prior bytes."""
    config = RotatingShieldPFConfig(
        num_particles=12,
        max_sources=5,
        init_num_sources=(0, 5),
        variable_cardinality=True,
        structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
        structural_cardinality_prior_probs=tuple(1.0 / 6.0 for _ in range(6)),
        use_gpu=False,
    )
    before = np.asarray(
        config.structural_cardinality_prior_probs,
        dtype="<f8",
    ).tobytes()

    config.__post_init__()

    after = np.asarray(
        config.structural_cardinality_prior_probs,
        dtype="<f8",
    ).tobytes()
    assert after == before


def test_pure_estimator_initializes_the_single_strict_profile() -> None:
    """PurePFEstimator must expose the positive strict-PF capability contract."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(estimator_profile="pf_strict"),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="b" * 64),
    )

    assert estimator.estimator_variant == "pf_strict"
    assert estimator.profile_capabilities.posterior_reporting_only is True
    assert estimator.profile_capabilities.sequential_updates_only is True


@pytest.mark.parametrize("schema_version", [True, 2.0, "2", 1])
def test_pure_estimator_rejects_non_integer_schema_versions(
    schema_version: object,
) -> None:
    """PurePFEstimator must not coerce schema-version compatibility values."""
    with pytest.raises(ValueError, match="schema version 2"):
        PurePFEstimator(
            isotopes=("Cs-137",),
            surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            shield_normals=None,
            mu_by_isotope={"Cs-137": 0.0},
            pf_config=RotatingShieldPFConfig(),
            full_spectrum_generative_model=approved_full_spectrum_model(),
            **_pure_pf_provenance(
                measurement_log_sha256="b" * 64,
                measurement_log_schema_version=schema_version,
            ),
        )


def test_structural_model_manifest_resolves_priors_and_surface_atlases() -> None:
    """Structural provenance must be complete without assuming shared atlases."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137", "Co-60"),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=_exact_rj_config(
            max_sources=2,
            init_num_sources=(0, 2),
            structural_cardinality_prior_policy=(EXPLICIT_CARDINALITY_PRIOR_POLICY),
            structural_cardinality_prior_probs=[
                1.0 / 6.0,
                2.0 / 6.0,
                3.0 / 6.0,
            ],
            strength_prior=BoundedUniformStrengthPriorTestConfig(
                minimum_cps_1m=300_000.0,
                maximum_cps_1m=2_000_000.0,
            ),
            use_gpu=False,
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="b" * 64),
    )

    before_filters = estimator.structural_model_manifest()
    assert before_filters["manifest_completeness"] == "config_only"
    cardinality_prior = before_filters["cardinality_prior"]
    assert cardinality_prior["support"] == [0, 1, 2]
    assert cardinality_prior["probabilities"] == pytest.approx(
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]
    )
    assert cardinality_prior["configuration_source"] == "explicit"
    assert cardinality_prior["policy_name"] == (EXPLICIT_CARDINALITY_PRIOR_POLICY)
    assert cardinality_prior["truncated_poisson_mean_sources_per_isotope"] is None
    assert cardinality_prior["fixed_before_observation"] is True
    assert cardinality_prior["applies_independently_per_isotope"] is True
    assert before_filters["strength_prior"]["units"] == "detector_cps_1m"
    assert before_filters["strength_prior"]["minimum_cps_1m"] == 300_000.0
    assert before_filters["strength_prior"]["kind"] == (
        "bounded_uniform_test_only"
    )
    assert (
        before_filters["strength_prior"]["support_maximum_cps_1m"]
        == 2_000_000.0
    )
    assert "legacy_proposal_grid_maximum_cps_1m" not in (
        before_filters["strength_prior"]
    )
    surface_prior = before_filters["surface_position_prior"]
    assert surface_prior["semantics"] == (
        "iid_uniform_physical_surface_area_canonical_unordered"
    )
    assert surface_prior["same_chart_sources_allowed"] is True
    assert surface_prior["pair_interaction_prior"] == ("none_iid_surface_positions")
    assert surface_prior["proximity_used_only_in_target_preserving_proposals"] is True
    assert surface_prior["continuous_uv_support"] is True
    assert surface_prior["support_quantization"] is False
    assert surface_prior["continuous_coordinates_within_each_chart"] is True
    assert surface_prior["chart_tessellation_role"] == (
        "coordinates_continuous_max_edge_topology_only"
    )
    assert surface_prior["atlas_status"] == "not_initialized"
    assert surface_prior["atlases_identical_across_isotopes"] is None
    assert surface_prior["missing_isotopes"] == ["Co-60", "Cs-137"]
    rj_kernel = before_filters["rj_move_kernel"]
    assert rj_kernel["position_move_attempt_probability"] == 1.0
    assert rj_kernel["position_move_proposal"] == (
        "joint_state_independent_surface_and_chart_conditional_strength_independence_mh"
    )
    assert rj_kernel["position_proposal_prior_component_probability"] == pytest.approx(
        0.5
    )
    assert rj_kernel["position_proposal_full_support"] is True
    assert rj_kernel["position_proposal_fixed_per_structural_sweep"] is True
    assert rj_kernel["position_proposal_chart_conditional"] == (
        "continuous_uniform_unit_square_uv"
    )
    assert rj_kernel["position_proposal_reverse_density"] == (
        "same_state_independent_mixture_for_all_directions"
    )
    assert rj_kernel["position_proposal_alignment_response"] == (
        "target_isotope_positive_transport_lines_only_at_chart_"
        "centers_for_proposal_scoring"
    )
    assert rj_kernel["position_proposal_state_dependence"] == (
        "observations_target_beta_and_immutable_known_model_only_"
        "never_current_particle_population"
    )
    assert rj_kernel["position_proposal_data_component"] == (
        "background_whitened_non_target_line_subspace_matched_filter_v1"
    )
    assert rj_kernel["strength_proposal"] == (
        "bounded_uniform_prior_plus_chart_conditional_truncated_normal_mixture"
    )
    assert (
        rj_kernel["proposal_score_cache"]["stores_spectra_or_particle_state"] is False
    )
    assert rj_kernel["position_proposal_target_response"] == (
        "direct_continuous_xyz_kernel_without_chart_interpolation"
    )
    assert rj_kernel["local_position_move_attempt_probability"] == 1.0
    assert rj_kernel["local_position_move_proposal"] == (
        "gaussian_tangent_geodesic_via_shared_edge_portals"
    )
    assert rj_kernel["local_position_reverse_correction"] == (
        "log_source_chart_area_over_destination_chart_area"
    )
    assert rj_kernel["local_position_physical_area_jacobian"] == 1.0
    assert rj_kernel["local_position_invalid_trace"] == (
        "explicit_self_transition_without_redraw"
    )
    assert rj_kernel["merge_pair_proposal"] == (
        "exact_same_or_one_portal_surface_distance_weighted_ordered_pair_"
        "with_uniform_global_support"
    )
    assert rj_kernel["split_merge_selection_density_in_mh_ratio"] is True
    assert rj_kernel["post_merge_same_sweep_refinement"] is True
    assert rj_kernel["structural_sweep_order"] == (
        "birth_death_then_split_merge_then_block_independence_then_"
        "global_position_then_local_position_then_strength"
    )
    assert rj_kernel["boundary_normalization"]["at_k_zero"] == {
        "birth": 1.0,
        "death": 0.0,
    }
    assert rj_kernel["boundary_normalization"]["at_k_max"] == {
        "cardinality": 2,
        "birth": 0.0,
        "death": 1.0,
    }
    assert (
        rj_kernel["dimension_matching"]["birth_death"]["absolute_jacobian_determinant"]
        == 1.0
    )
    assert (
        rj_kernel["dimension_matching"]["split"]["absolute_jacobian_determinant"]
        == "total_strength"
    )
    assert (
        rj_kernel["dimension_matching"]["merge"]["absolute_jacobian_determinant"]
        == "1_over_merged_total_strength"
    )

    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    shared_charts = build_surface_chart_geometry(environment, None, 1.0)
    estimator.filters = {
        isotope: SimpleNamespace(
            _structural_rj_surface_atlas=SimpleNamespace(geometry=shared_charts),
        )
        for isotope in estimator.isotopes
    }
    shared_manifest = estimator.structural_model_manifest()
    shared_surface = shared_manifest["surface_position_prior"]
    assert shared_manifest["manifest_completeness"] == "complete"
    assert shared_surface["atlases_identical_across_isotopes"] is True
    assert len(shared_surface["atlas_groups"]) == 1
    assert shared_surface["atlas_groups"][0]["isotopes"] == [
        "Co-60",
        "Cs-137",
    ]

    different_charts = build_surface_chart_geometry(
        EnvironmentConfig(size_x=3.0, size_y=2.0, size_z=2.0),
        None,
        1.0,
    )
    estimator.filters["Co-60"] = SimpleNamespace(
        _structural_rj_surface_atlas=SimpleNamespace(geometry=different_charts),
    )
    different_surface = estimator.structural_model_manifest()["surface_position_prior"]
    assert different_surface["atlases_identical_across_isotopes"] is False
    assert len(different_surface["atlas_groups"]) == 2
    different_hashes = {
        group["surface_atlas_contract_sha256"]
        for group in different_surface["atlas_groups"]
    }
    assert (
        shared_surface["atlas_groups"][0]["surface_atlas_contract_sha256"]
        in different_hashes
    )


def test_shifted_gamma_manifest_separates_support_and_proposal_quantile() -> None:
    """Unbounded support must not be reported as a finite proposal extent."""
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=_exact_rj_config(
            max_sources=1,
            init_num_sources=(0, 1),
            strength_prior=ShiftedGammaStrengthPriorConfig(
                minimum_cps_1m=300_000.0,
                shape=2.0,
                scale_cps_1m=425_000.0,
            ),
            use_gpu=False,
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="b" * 64),
    )

    manifest = estimator.structural_model_manifest()["strength_prior"]

    assert manifest["kind"] == "shifted_gamma"
    assert manifest["support_maximum_cps_1m"] is None
    assert manifest["support_upper_unbounded"] is True
    assert manifest["shape"] == 2.0
    assert manifest["scale_cps_1m"] == 425_000.0
    assert manifest["proposal_grid_upper_quantile"]["probability"] == (
        STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY
    )
    assert np.isfinite(manifest["proposal_grid_upper_quantile"]["value_cps_1m"])
    assert "legacy_proposal_grid_maximum_cps_1m" not in manifest


def test_strict_profile_keeps_pf_budget_and_retires_runtime_placeholders() -> None:
    """The PF config keeps its budget without obsolete runtime placeholders."""
    root = Path(__file__).resolve().parents[1]
    resolved = enforce_pure_runtime_settings(
        load_runtime_config(root / "configs/pf/pf_strict_3d.json")
    )
    assert "structural_cardinality_prior_policy" not in resolved
    assert resolved["structural_cardinality_prior_mean"] == pytest.approx(2.0)
    assert int(resolved["max_sources"]) == 5
    assert int(resolved["hard_max_sources"]) == 8
    assert resolved["structural_cardinality_tail_ratio"] == pytest.approx(0.05)
    assert "adaptive_cardinality_dwell_enable" not in resolved
    assert resolved["adaptive_stop"] == {
        "assessment_start_station": 10,
        "required_consecutive_stations": 3,
        "minimum_joint_map_cardinality_probability": 0.95,
        "maximum_upper_cardinality_mass": 0.05,
        "maximum_surface_path_radius_95_m": 0.5,
        "innovation_confidence": 0.99,
    }
    assert "measurement_budget_max_steps" not in resolved
    assert "mission_stop_max_poses" not in resolved
    assert "measurement_live_time_s" not in resolved
    assert "orientation_k" not in resolved
    assert "cui_truth_display_mode" not in resolved
    assert "detector_height_sampling_mode" not in resolved
    assert "measurement_pose_clearance_enabled" not in resolved
    assert "path_planner" not in resolved
    assert "spectrum_count_method" not in resolved
    assert "calibration_count_method" not in resolved
    assert "sim_backend" not in resolved
    assert "measurement_log_output_dir" not in resolved
    dss = resolved["dss_pp"]
    # The nested planner section is an exact, self-contained contract.
    assert "program_length" not in dss
    assert dss["shield_view_count_shadow_enabled"] is True
    assert "shield_view_count_shadow_candidate_counts" not in dss
    assert "shield_view_count_shadow_retention_fraction" not in dss
    assert "shield_view_count_shadow_per_comparison_confidence" not in dss
    assert "planning_method" not in dss
    assert "diagnostic_ranked_node_limit" not in dss
    assert "max_programs" not in dss
    assert "min_station_separation_m" not in dss
    assert "coverage_radius_m" not in dss
    assert STANDARD_EXPERIMENT_PROFILE.acquisition.max_measurements == 128


def test_standard_pf_config_selects_exact_surface_rj_kernel() -> None:
    """The PF config must select only exact surface RJ-MH."""
    root = Path(__file__).resolve().parents[1]
    payload = load_runtime_config(root / "configs/pf/pf_strict_3d.json")
    payload = enforce_pure_runtime_settings(payload)
    pf_values = production_pf_config_values(
        payload,
        position_max=(1.0, 1.0, 1.0),
    )

    assert pf_values["variable_cardinality"] is True
    assert pf_values["init_num_sources"] == (0, payload["max_sources"])
    assert pf_values["structural_cardinality_prior_policy"] == (
        POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
    )
    assert pf_values["structural_cardinality_prior_probs"] is None
    assert pf_values["joint_guided_initialization"] is True
    assert float(payload["structural_rj_surface_chart_max_edge_m"]) > 0.0
    assert float(payload["structural_rj_move_probability"]) > 0.0
    assert float(payload["structural_rj_birth_probability"]) > 0.0
    assert float(payload["structural_rj_death_probability"]) > 0.0
    attempt_probabilities = np.asarray(
        [
            payload["structural_rj_move_probability"],
            payload["structural_rj_position_move_probability"],
            payload["structural_rj_local_position_move_probability"],
            payload["structural_rj_strength_move_probability"],
            payload["structural_rj_split_merge_probability"],
        ],
        dtype=np.float64,
    )
    assert np.all(attempt_probabilities > 0.0)
    assert np.all(attempt_probabilities <= 1.0)
    assert float(payload["structural_cardinality_prior_mean"]) > 0.0


def test_standard_pf_config_selects_batched_cuda_compute() -> None:
    """The standard PF config must select its real batched CUDA backend."""
    root = Path(__file__).resolve().parents[1]
    payload = enforce_pure_runtime_settings(
        load_runtime_config(root / "configs/pf/pf_strict_3d.json")
    )

    assert payload["compute_backend"] == {
        "kind": "cuda_float64",
        "device": "cuda",
    }
    assert production_compute_backend_values(payload) == {
        "use_gpu": True,
        "gpu_device": "cuda",
    }
    assert production_pf_config_values(
        payload,
        position_max=(1.0, 1.0, 1.0),
    )["gpu_dtype"] == "float64"
    assert int(payload["joint_strength_block_batch_size"]) > 1
    assert int(payload["structural_rj_proposal_chart_batch_size"]) > 1
    assert float(payload["joint_cross_isotope_state_block_probability"]) > 0.0
    assert "python_worker_count" not in payload
    assert "pose_selection_workers" not in payload


def test_final_estimates_are_projected_directly_from_pf_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The array view must be a direct PF-posterior projection."""
    estimator = object.__new__(PurePFEstimator)
    point_estimate = posterior_point_estimate_from_states(
        [
            SimpleNamespace(
                num_sources=1,
                strengths=np.asarray([25.0], dtype=float),
            )
        ],
        np.asarray([1.0], dtype=float),
        positions_by_state=[np.asarray([[1.0, 2.0, 3.0]], dtype=float)],
        max_cardinality=1,
    )
    monkeypatch.setattr(
        estimator,
        "posterior_point_estimate",
        lambda: {"Cs-137": point_estimate},
    )

    actual = PurePFEstimator.estimates(estimator)

    np.testing.assert_array_equal(
        actual["Cs-137"][0],
        np.asarray([[1.0, 2.0, 3.0]], dtype=float),
    )
    np.testing.assert_array_equal(
        actual["Cs-137"][1],
        np.asarray([25.0], dtype=float),
    )


def test_pure_posterior_reports_an_actual_surface_particle_medoid() -> None:
    """Separated modes must report an actual posterior surface state."""
    environment = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    estimator = PurePFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 1.0, 1.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(2.0, 2.0, 2.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="b" * 64),
    )
    estimator.add_measurement_pose(np.asarray([1.0, 1.0, 1.0], dtype=float))
    estimator._ensure_kernel_cache()
    row_identities = [
        particle.joint_row_identity
        for particle in estimator.filters["Cs-137"].continuous_particles
    ]
    estimator.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=_surface_state(
                estimator.filters["Cs-137"],
                np.asarray([position], dtype=float),
                np.asarray([10.0], dtype=float),
            ),
            log_weight=float(np.log(0.5)),
            joint_row_identity=row_identities[row],
        )
        for row, position in enumerate(((0.0, 1.0, 1.0), (2.0, 1.0, 1.0)))
    ]

    point_estimate = estimator.posterior_point_estimate()["Cs-137"]
    reported_positions = estimator.estimates()["Cs-137"][0]

    assert len(point_estimate.modes) == 1
    assert (
        source_surface_kind(
            point_estimate.modes[0].position_medoid_xyz,
            environment,
        )
        is not None
    )
    assert source_surface_kind(reported_positions[0], environment) is not None
    chart_ids, surface_uv = estimator.filters[
        "Cs-137"
    ].structural_surface_chart_coordinates(reported_positions)
    assert int(chart_ids[0]) >= 0
    assert np.all(surface_uv >= 0.0)
    assert np.all(surface_uv <= 1.0)
    assert any(
        np.array_equal(reported_positions[0], np.asarray(position, dtype=float))
        for position in ((0.0, 1.0, 1.0), (2.0, 1.0, 1.0))
    )


def test_exact_surface_medoid_matches_scalar_oracle_beyond_old_cap() -> None:
    """Chunked medoids must equal an all-row scalar oracle above 64 rows."""
    rng = np.random.default_rng(724)
    particle_count = 97
    source_count = 3
    chart_ids = np.zeros((particle_count, source_count), dtype=np.int64)
    surface_uv = rng.uniform(
        0.0,
        1.0,
        size=(particle_count, source_count, 2),
    )
    weights = rng.uniform(0.01, 1.0, size=particle_count)
    weights /= np.sum(weights)
    evaluated_chunk_widths: list[int] = []

    def coordinate_distance(
        left_ids: NDArray[np.int64],
        left_uv: NDArray[np.float64],
        right_ids: NDArray[np.int64],
        right_uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return broadcast Euclidean chart distance and record chunk width."""
        left_id_array, right_id_array = np.broadcast_arrays(
            left_ids,
            right_ids,
        )
        left_uv_array, right_uv_array = np.broadcast_arrays(
            left_uv,
            right_uv,
        )
        distances = np.linalg.norm(left_uv_array - right_uv_array, axis=-1)
        evaluated_chunk_widths.append(int(distances.shape[-1]))
        return np.where(left_id_array == right_id_array, distances, np.inf)

    medoid_ids, medoid_uv = _surface_mode_medoid_coordinates_batched(
        chart_ids,
        surface_uv,
        weights,
        coordinate_distance,
        candidate_chunk_size=7,
    )

    oracle_rows: list[int] = []
    for source_index in range(source_count):
        costs = np.asarray(
            [
                np.sum(
                    weights
                    * np.sum(
                        np.square(
                            surface_uv[:, source_index]
                            - surface_uv[candidate_index, source_index]
                        ),
                        axis=1,
                    )
                )
                for candidate_index in range(particle_count)
            ],
            dtype=np.float64,
        )
        minimum = float(np.min(costs))
        tied = np.flatnonzero(np.isclose(costs, minimum, rtol=0.0, atol=1.0e-15))
        maximum_weight = float(np.max(weights[tied]))
        final = tied[
            np.isclose(
                weights[tied],
                maximum_weight,
                rtol=0.0,
                atol=1.0e-15,
            )
        ]
        oracle_rows.append(int(final[0]))

    np.testing.assert_array_equal(medoid_ids, np.zeros(source_count, dtype=np.int64))
    np.testing.assert_allclose(
        medoid_uv,
        surface_uv[
            np.asarray(oracle_rows, dtype=np.int64),
            np.arange(source_count, dtype=np.int64),
        ],
        rtol=0.0,
        atol=0.0,
    )
    assert sum(evaluated_chunk_widths) == particle_count
    assert max(evaluated_chunk_widths) <= 7


def test_exact_surface_atlas_drives_projection_and_surface_kinds() -> None:
    """PF report projection and labels must use the exact transport-box atlas."""
    isotope = "Cs-137"
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
    )
    estimator = PurePFEstimator(
        isotopes=(isotope,),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_surface_chart_max_edge_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="b" * 64),
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    charts = filt._structural_rj_surface_atlas.geometry
    assert charts is not None

    actual_kinds = filt.structural_surface_kinds(
        charts.centers_xyz,
        strict=True,
    )
    np.testing.assert_array_equal(
        actual_kinds,
        np.asarray(charts.kinds, dtype=object),
    )
    assert {
        "floor",
        "ceiling",
        "wall",
        "obstacle_side",
        "obstacle_top",
        "obstacle_bottom",
    }.issubset(set(actual_kinds.tolist()))
    bottom_indices = np.flatnonzero(actual_kinds == "obstacle_bottom")
    np.testing.assert_allclose(
        charts.normals_xyz[bottom_indices],
        np.tile(np.asarray([0.0, 0.0, -1.0]), (bottom_indices.size, 1)),
    )
    assert all(charts.face_ids[index].endswith("_z0") for index in bottom_indices)
    assert np.sum(charts.areas_m2[bottom_indices]) == pytest.approx(0.36)

    representative_indices = np.asarray(
        [
            int(np.flatnonzero(actual_kinds == kind)[0])
            for kind in sorted(set(actual_kinds.tolist()))
        ],
        dtype=np.int64,
    )
    representative_centers = charts.centers_xyz[representative_indices]
    filt.validate_continuous_surface_states()
    assert not hasattr(filt, "_project_positions_to_source_prior")

    query = np.asarray([[1.55, 1.61, 0.83]], dtype=float)
    with pytest.raises(ValueError, match="surface"):
        filt.structural_surface_chart_coordinates(query)
    signed_zero_floor = representative_centers[
        np.flatnonzero(actual_kinds[representative_indices] == "floor")[0]
    ].copy()
    signed_zero_floor[2] = -0.0
    floor_chart_ids, floor_uv = filt.structural_surface_chart_coordinates(
        signed_zero_floor[None, :]
    )
    assert int(floor_chart_ids[0]) >= 0
    assert np.all((floor_uv >= 0.0) & (floor_uv <= 1.0))
    between_old_chart_centers = np.asarray([[0.0, 0.3, 0.3]], dtype=float)
    continuous_chart_ids, continuous_uv = filt.structural_surface_chart_coordinates(
        between_old_chart_centers
    )
    assert int(continuous_chart_ids[0]) >= 0
    assert np.all((continuous_uv >= 0.0) & (continuous_uv <= 1.0))
    assert (
        filt.structural_surface_kinds(
            between_old_chart_centers,
            strict=True,
        )[0]
        == "wall"
    )


def test_posterior_aligns_swapped_labels_and_reports_uncertainty() -> None:
    """Spatial modes must not collapse when particle source labels are swapped."""
    positions_by_state = [
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.asarray([[2.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.asarray([[1.0, 1.0, 0.0]]),
    ]
    states = [
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([10.0, 20.0]),
        ),
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([22.0, 12.0]),
        ),
        SimpleNamespace(
            num_sources=1,
            strengths=np.asarray([7.0]),
        ),
    ]
    estimate = posterior_point_estimate_from_states(
        states,
        np.asarray([0.45, 0.35, 0.20]),
        positions_by_state=positions_by_state,
        max_cardinality=2,
    )

    assert estimate.map_cardinality == 2
    assert estimate.cardinality_distribution == pytest.approx({0: 0.0, 1: 0.2, 2: 0.8})
    assert len(estimate.modes) == 2
    assert estimate.modes[0].position_medoid_xyz[0] < 0.2
    assert estimate.modes[1].position_medoid_xyz[0] >= 2.0
    assert estimate.modes[0].strength_mean_cps_1m < 13.0
    assert estimate.modes[1].strength_mean_cps_1m > 19.0
    for mode in estimate.modes:
        covariance = np.asarray(mode.position_covariance_xyz)
        assert np.allclose(covariance, covariance.T)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-12
        assert mode.credible_radius_95_m >= 0.0
        lower, upper = mode.strength_credible_interval_95_cps_1m
        assert 0.0 <= lower <= upper
        assert mode.posterior_mass == pytest.approx(0.8)
    payload = estimate.to_dict()
    assert "background_rate_mean_cps" not in payload
    assert "background_rate_credible_interval_95_cps" not in payload
    assert "background_mean_counts" not in payload


def test_posterior_alignment_does_not_cross_thin_obstacle_faces() -> None:
    """Surface modes must follow intrinsic faces, not nearest Cartesian points."""
    environment = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 0.0, 1.1, 2.0, 2.0),),
    )
    atlas = ContinuousSurfaceAtlas(
        build_surface_chart_geometry(
            environment,
            obstacle_grid,
            max_edge_m=0.5,
        )
    )
    position_rows = [
        np.asarray(
            [[1.0, 1.2, 1.0], [1.1, 1.8, 1.0]],
            dtype=np.float64,
        ),
        np.asarray(
            [[1.1, 1.2, 1.0], [1.0, 1.8, 1.0]],
            dtype=np.float64,
        ),
    ]
    chart_rows: list[NDArray[np.int64]] = []
    uv_rows: list[NDArray[np.float64]] = []
    for positions in position_rows:
        chart_ids, surface_uv = atlas.locate_positions(positions)
        chart_rows.append(chart_ids)
        uv_rows.append(surface_uv)
    states = [
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([10.0, 20.0], dtype=np.float64),
            surface_chart_ids=chart_rows[0],
            surface_uv=uv_rows[0],
        ),
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([22.0, 12.0], dtype=np.float64),
            surface_chart_ids=chart_rows[1],
            surface_uv=uv_rows[1],
        ),
    ]

    estimate = posterior_point_estimate_from_states(
        states,
        np.asarray([0.5, 0.5], dtype=np.float64),
        positions_by_state=position_rows,
        surface_chart_ids_by_state=chart_rows,
        surface_uv_by_state=uv_rows,
        surface_coordinate_path_distance=(
            atlas.surface_coordinate_path_distance_upper_bound_m
        ),
        max_cardinality=2,
    )

    face_strengths = {
        atlas.geometry.face_ids[int(mode.surface_chart_id)]: (mode.strength_mean_cps_1m)
        for mode in estimate.modes
    }
    left_face = atlas.geometry.face_ids[int(chart_rows[0][0])]
    right_face = atlas.geometry.face_ids[int(chart_rows[0][1])]
    assert left_face != right_face
    assert face_strengths[left_face] == pytest.approx(11.0)
    assert face_strengths[right_face] == pytest.approx(21.0)


def test_surface_alignment_bounds_graph_distance_source_charts() -> None:
    """Batched alignment must run graph searches from medoids, not particles."""
    particle_count = 1000
    source_count = 2
    chart_ids = np.arange(
        particle_count * source_count,
        dtype=np.int64,
    ).reshape(particle_count, source_count)
    surface_uv = np.full(
        (particle_count, source_count, 2),
        0.5,
        dtype=np.float64,
    )
    positions = np.zeros(
        (particle_count, source_count, 3),
        dtype=np.float64,
    )
    positions[:, :, 0] = chart_ids
    strengths = np.ones(
        (particle_count, source_count),
        dtype=np.float64,
    )
    weights = np.full(
        particle_count,
        1.0 / float(particle_count),
        dtype=np.float64,
    )
    unique_source_counts: list[int] = []

    def chart_distance(
        first_ids: NDArray[np.int64],
        first_uv: NDArray[np.float64],
        second_ids: NDArray[np.int64],
        second_uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a deterministic proxy while recording graph source count."""
        del first_uv, second_uv
        unique_source_counts.append(
            int(np.unique(np.asarray(first_ids, dtype=np.int64)).size)
        )
        first, second = np.broadcast_arrays(first_ids, second_ids)
        return np.abs(first.astype(np.float64) - second.astype(np.float64))

    align_surface_modes_batched(
        positions,
        strengths,
        chart_ids,
        surface_uv,
        weights,
        chart_distance,
        max_iterations=2,
    )

    assert unique_source_counts
    assert max(unique_source_counts) <= 64 * source_count


def test_posterior_uncertainty_rejects_opposite_face_support() -> None:
    """A nearby opposite-face particle must not support the reported mode."""
    environment = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.0, 1.0, 0.0, 1.1, 2.0, 2.0),),
    )
    atlas = ContinuousSurfaceAtlas(
        build_surface_chart_geometry(
            environment,
            obstacle_grid,
            max_edge_m=0.5,
        )
    )
    particle_position = np.asarray([[1.0, 1.5, 1.0]], dtype=np.float64)
    reported_position = np.asarray([[1.1, 1.5, 1.0]], dtype=np.float64)
    particle_chart, particle_uv = atlas.locate_positions(particle_position)
    reported_chart, reported_uv = atlas.locate_positions(reported_position)

    diagnostics = posterior_mode_uncertainty_batched(
        particle_position.reshape(1, 1, 3),
        np.asarray([[True]], dtype=bool),
        np.asarray([1.0], dtype=np.float64),
        reported_position,
        packed_surface_kinds=np.asarray(
            [["obstacle_side"]],
            dtype=object,
        ),
        packed_surface_chart_ids=particle_chart.reshape(1, 1),
        packed_surface_uv=particle_uv.reshape(1, 1, 2),
        reported_surface_chart_ids=reported_chart,
        reported_surface_uv=reported_uv,
        surface_coordinate_path_distance=(
            atlas.surface_coordinate_path_distance_upper_bound_m
        ),
        environment=environment,
        obstacle_grid=obstacle_grid,
        match_radius_m=0.5,
    )

    assert diagnostics[0]["posterior_support_available"] is False
    assert diagnostics[0]["existence_mass"] == 0.0
    assert diagnostics[0]["matching_distance"] == (
        "intrinsic_surface_path_upper_bound_m"
    )


def test_posterior_reports_one_joint_particle_configuration() -> None:
    """Reported source positions must coexist in one posterior particle."""
    positions_by_state = [
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        np.asarray([[2.1, 0.0, 0.0], [9.9, 0.0, 0.0]]),
    ]
    states = [
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([1.0, 1.0]),
        ),
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([1.0, 1.0]),
        ),
        SimpleNamespace(
            num_sources=2,
            strengths=np.asarray([1.0, 1.0]),
        ),
    ]
    estimate = posterior_point_estimate_from_states(
        states,
        np.asarray([0.45, 0.45, 0.10]),
        positions_by_state=positions_by_state,
        max_cardinality=2,
    )
    reported = np.asarray(
        [mode.position_medoid_xyz for mode in estimate.modes],
        dtype=float,
    )
    assert any(np.array_equal(reported, positions) for positions in positions_by_state)


def test_pure_posterior_uses_joint_map_cardinality_vector() -> None:
    """Official K values must come from one observed joint particle stratum."""
    isotopes = ("Co-60", "Cs-137")
    estimator = PurePFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=5,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="c" * 64),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5], dtype=float))
    estimator._ensure_kernel_cache()
    weights = np.asarray([0.15, 0.15, 0.15, 0.30, 0.25])
    cardinalities = {
        "Co-60": (1, 1, 1, 0, 1),
        "Cs-137": (0, 0, 0, 1, 1),
    }
    for isotope in isotopes:
        row_identities = [
            particle.joint_row_identity
            for particle in estimator.filters[isotope].continuous_particles
        ]
        particles: list[IsotopeParticle] = []
        for row, (weight, cardinality) in enumerate(
            zip(weights, cardinalities[isotope], strict=True)
        ):
            positions = (
                np.asarray([[float(row % 3), 1.0, 0.0]], dtype=float)
                if cardinality
                else np.zeros((0, 3), dtype=float)
            )
            strengths = (
                np.asarray([100.0 + row], dtype=float)
                if cardinality
                else np.zeros(0, dtype=float)
            )
            particles.append(
                IsotopeParticle(
                    state=_surface_state(
                        estimator.filters[isotope],
                        positions,
                        strengths,
                    ),
                    log_weight=float(np.log(weight)),
                    joint_row_identity=row_identities[row],
                )
            )
        estimator.filters[isotope].continuous_particles = particles

    joint_distribution = estimator.posterior_joint_cardinality_distribution()
    estimates = estimator.posterior_point_estimate()

    assert estimator.joint_isotope_order() == ("Co-60", "Cs-137")
    assert joint_distribution == pytest.approx(
        {
            (0, 1): 0.30,
            (1, 0): 0.45,
            (1, 1): 0.25,
        }
    )
    assert estimates["Co-60"].cardinality_distribution[1] == pytest.approx(0.70)
    assert estimates["Cs-137"].cardinality_distribution[1] == pytest.approx(0.55)
    assert estimates["Co-60"].map_cardinality == 1
    assert estimates["Cs-137"].map_cardinality == 0
    assert not estimates["Cs-137"].modes
    assert estimates["Co-60"].modes[0].posterior_mass == pytest.approx(0.45)
    assert estimates["Co-60"].selected_stratum_mass == pytest.approx(0.45)
    assert estimates["Cs-137"].selected_stratum_mass == pytest.approx(0.45)

    radii = estimator.credible_surface_radii()
    diagnostics = estimator.posterior_convergence_diagnostics()
    uncertainty = estimator.posterior_source_uncertainty()

    assert radii["Cs-137"] == []
    assert diagnostics["joint_cardinality"]["map_cardinalities"] == [1, 0]
    assert diagnostics["joint_cardinality"]["map_probability"] == pytest.approx(0.45)
    assert diagnostics["isotopes"]["Cs-137"]["credible_surface_radii_95_m"] == []
    assert uncertainty["Co-60"][0]["posterior_reference_mass"] == (pytest.approx(0.45))
    assert uncertainty["Co-60"][0]["conditional_support_mass"] == (
        pytest.approx(1.0 / 3.0)
    )
    assert uncertainty["Co-60"][0]["existence_mass"] == pytest.approx(0.15)


def test_pure_posterior_uses_one_joint_configuration_medoid_row() -> None:
    """All isotope positions must be copied from the same aligned PF row."""
    isotopes = ("Co-60", "Cs-137")
    estimator = PurePFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
        **_pure_pf_provenance(measurement_log_sha256="d" * 64),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5], dtype=float))
    estimator._ensure_kernel_cache()
    positions_by_isotope = {
        "Co-60": ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)),
        "Cs-137": ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    }
    for isotope in isotopes:
        row_identities = [
            particle.joint_row_identity
            for particle in estimator.filters[isotope].continuous_particles
        ]
        estimator.filters[isotope].continuous_particles = [
            IsotopeParticle(
                state=_surface_state(
                    estimator.filters[isotope],
                    np.asarray([position], dtype=float),
                    np.asarray([100.0 + row], dtype=float),
                ),
                log_weight=float(-np.log(3.0)),
                joint_row_identity=row_identities[row],
            )
            for row, position in enumerate(positions_by_isotope[isotope])
        ]

    estimates = estimator.posterior_point_estimate()
    reported = {
        isotope: np.asarray(
            estimates[isotope].modes[0].position_medoid_xyz,
            dtype=float,
        )
        for isotope in isotopes
    }
    matching_rows = [
        row
        for row in range(3)
        if all(
            np.array_equal(
                reported[isotope],
                np.asarray(positions_by_isotope[isotope][row], dtype=float),
            )
            for isotope in isotopes
        )
    ]

    assert matching_rows == [0]
    projected = estimator.estimates()
    for isotope in isotopes:
        assert projected[isotope][1] == pytest.approx(np.asarray([100.0], dtype=float))
        assert estimates[isotope].modes[
            0
        ].strength_representative_cps_1m == pytest.approx(100.0)
        assert estimates[isotope].modes[0].strength_mean_cps_1m == pytest.approx(101.0)


def test_packed_joint_transport_keeps_configured_source_slot_axis() -> None:
    """A vanished Kmax stratum must not shift joint isotope slot boundaries."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=3,
            variable_cardinality=True,
            init_num_sources=(0, 3),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                strengths=np.zeros(0, dtype=np.float64),
                surface_chart_ids=np.zeros(0, dtype=np.int64),
                surface_uv=np.zeros((0, 2), dtype=np.float64),
            ),
            log_weight=float(-np.log(4.0)),
        )
        for _ in range(4)
    ]

    positions, strengths, mask = filt._packed_continuous_state_arrays()

    assert positions.shape == (4, 3, 3)
    assert strengths.shape == (4, 3)
    assert mask.shape == (4, 3)
    assert not np.any(mask)


def test_joint_smc_reports_station_and_cumulative_lineage_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genealogical collapse must persist across station boundaries."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            target_ess_ratio=0.5,
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()
    call_count = 0

    def _select_delta_beta(**kwargs: object) -> tuple[float, object, float]:
        """Force one two-ancestor resample followed by a full increment."""
        nonlocal call_count
        import torch

        call_count += 1
        remaining = float(kwargs["remaining"])
        if call_count % 2:
            log_weights = torch.tensor(
                [np.log(0.5), np.log(0.5), -np.inf, -np.inf],
                dtype=torch.float64,
            )
            return 0.5 * remaining, log_weights, 2.0
        log_weights = torch.full(
            (4,),
            -np.log(4.0),
            dtype=torch.float64,
        )
        return remaining, log_weights, 4.0

    monkeypatch.setattr(
        estimator.filters["Cs-137"],
        "_select_delta_beta",
        _select_delta_beta,
    )

    def _zero_prefix_likelihood(
        station: JointStationObservation,
    ) -> object:
        """Return finite inert empty/full prefixes for every aligned row."""
        import torch

        del station
        return torch.zeros((2, 4), dtype=torch.float64)

    monkeypatch.setattr(
        estimator,
        "_joint_station_prefix_log_likelihood_torch",
        _zero_prefix_likelihood,
    )
    monkeypatch.setattr(
        estimator,
        "_joint_rejuvenate",
        lambda stations, target_beta, newest_prefix_count=None: (
            _sufficient_mixing_diagnostics()
        ),
    )
    monkeypatch.setattr(
        estimator,
        "_promote_joint_birth_proposal_station",
        lambda station: None,
    )

    def _station(sequence_id: int) -> JointStationObservation:
        """Build one inert station for SMC lineage bookkeeping."""
        return JointStationObservation(
            spectrum_vb=np.zeros(
                (1, np.asarray(model.energy_axis_keV).size),
                dtype=np.float64,
            ),
            energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
            generative_contract_hash_sha256=model.contract_hash_sha256,
            pose_idx=0,
            detector_position_xyz_m=(1.5, 1.5, 1.5),
            fe_indices=np.asarray([0], dtype=np.int64),
            pb_indices=np.asarray([0], dtype=np.int64),
            live_times_s=np.asarray([1.0], dtype=np.float64),
            station_sequence_id=sequence_id,
        )

    estimator._joint_tempered_station_update(_station(0))
    first_cumulative = estimator.last_joint_cumulative_unique_ancestor_count
    estimator._joint_tempered_station_update(_station(1))

    assert first_cumulative == 2
    assert estimator.last_joint_station_unique_ancestor_count == 2
    assert estimator.last_joint_cumulative_unique_ancestor_count <= 2
    assert not hasattr(estimator, "last_joint_unique_ancestor_count")
    diagnostics = estimator.step_diagnostics(include_estimates=False)["Cs-137"]
    assert "unique_ancestor_count" not in diagnostics
    assert diagnostics["station_unique_ancestor_count"] == 2
    assert diagnostics["cumulative_unique_ancestor_count"] <= 2
    assert not hasattr(
        estimator.filters["Cs-137"],
        "last_unique_ancestor_count",
    )
    assert all(
        {
            "station_unique_ancestors",
            "cumulative_unique_ancestors",
        }.issubset(step)
        for step in estimator.last_joint_temper_steps
    )


def test_joint_temper_step_limit_is_applied_per_view_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each exact view-prefix bridge must receive the configured step limit."""
    torch = pytest.importorskip("torch")
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            target_ess_ratio=0.5,
            max_temper_steps=1,
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()
    filt = estimator.filters["Cs-137"]

    def _finish_prefix(**kwargs: object) -> tuple[float, object, float]:
        """Complete each prefix bridge in one exact tempering step."""
        remaining = float(kwargs["remaining"])
        return (
            remaining,
            torch.full((4,), -np.log(4.0), dtype=torch.float64),
            4.0,
        )

    monkeypatch.setattr(filt, "_select_delta_beta", _finish_prefix)
    monkeypatch.setattr(
        estimator,
        "_joint_station_prefix_log_likelihood_torch",
        lambda station: torch.zeros((3, 4), dtype=torch.float64),
    )
    monkeypatch.setattr(
        estimator,
        "_joint_rejuvenate",
        lambda stations, target_beta, newest_prefix_count=None: (
            _sufficient_mixing_diagnostics()
        ),
    )
    monkeypatch.setattr(
        estimator,
        "_promote_joint_birth_proposal_station",
        lambda station: None,
    )
    station = JointStationObservation(
        spectrum_vb=np.zeros(
            (2, np.asarray(model.energy_axis_keV).size),
            dtype=np.float64,
        ),
        energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(1.5, 1.5, 1.5),
        fe_indices=np.asarray([0, 1], dtype=np.int64),
        pb_indices=np.asarray([0, 1], dtype=np.int64),
        live_times_s=np.asarray([1.0, 1.0], dtype=np.float64),
        station_sequence_id=0,
    )

    estimator._joint_tempered_station_update(station)

    assert len(estimator.last_joint_temper_steps) == 2
    assert [step["prefix_count"] for step in estimator.last_joint_temper_steps] == [
        1.0,
        2.0,
    ]
    assert all(
        step["beta_total"] == pytest.approx(1.0)
        for step in estimator.last_joint_temper_steps
    )


def test_joint_rejuvenation_uses_newest_station_boundary_on_every_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated sweeps must temper only the newest station and clear context."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()

    def _station(
        sequence_id: int,
        view_count: int,
    ) -> JointStationObservation:
        """Build one valid inert station with a requested number of views."""
        return JointStationObservation(
            spectrum_vb=np.zeros(
                (view_count, np.asarray(model.energy_axis_keV).size),
                dtype=np.float64,
            ),
            energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
            generative_contract_hash_sha256=model.contract_hash_sha256,
            pose_idx=sequence_id,
            detector_position_xyz_m=(
                1.0 + sequence_id,
                1.5,
                1.5,
            ),
            fe_indices=np.arange(view_count, dtype=np.int64),
            pb_indices=np.arange(view_count, dtype=np.int64),
            live_times_s=np.ones(view_count, dtype=np.float64),
            station_sequence_id=sequence_id,
        )

    first = _station(0, 2)
    second = _station(1, 3)
    calls: list[tuple[int, int, float]] = []
    row_identities_before = tuple(
        particle.joint_row_identity
        for particle in estimator.filters["Cs-137"].continuous_particles
    )

    def _apply(
        evidence: StructuralGeometryBatch,
        *,
        target_beta: float,
        tempering_start_row: int | None,
        current_target_log_likelihood: np.ndarray,
    ) -> None:
        """Record the estimator-owned evidence and newest-station boundary."""
        assert estimator._active_joint_structural_geometry is evidence
        assert tempering_start_row is not None
        estimator.filters["Cs-137"].last_structural_target_log_likelihood = np.asarray(
            current_target_log_likelihood, dtype=np.float64
        ).copy()
        calls.append(
            (
                int(evidence.row_count),
                int(tempering_start_row),
                float(target_beta),
            )
        )

    def _refresh(stations: object) -> None:
        """Install a sentinel cache for the isolated boundary test."""
        del stations
        total = np.zeros((4, 1, 1, 1), dtype=np.float64)
        estimator._joint_structural_transport_cache = (
            total,
            total.copy(),
            np.zeros(total.shape + (4,), dtype=np.float64),
        )

    monkeypatch.setattr(
        estimator,
        "_refresh_joint_structural_transport_cache",
        _refresh,
    )
    monkeypatch.setattr(
        estimator,
        "_joint_history_log_likelihood_numpy",
        lambda **_: np.zeros(4, dtype=np.float64),
    )
    monkeypatch.setattr(
        estimator.filters["Cs-137"],
        "apply_structural_moves",
        _apply,
    )

    for stations, beta in (
        ((first,), 0.0),
        ((first, second), 0.4),
        ((first,), 1.0),
    ):
        estimator._joint_rejuvenate(stations, target_beta=beta)
        assert estimator._active_joint_structural_geometry is None
        assert estimator._active_joint_station_history is None
        assert (
            tuple(
                particle.joint_row_identity
                for particle in estimator.filters["Cs-137"].continuous_particles
            )
            == row_identities_before
        )

    assert calls == [
        (2, 0, 0.0),
        (5, 2, 0.4),
        (2, 0, 1.0),
    ]


def test_joint_mixing_diagnostics_measure_state_not_ancestry() -> None:
    """Movement diagnostics must respond when one aligned state row changes."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    before = estimator._joint_mixing_snapshot()
    particle = estimator.filters["Cs-137"].continuous_particles[0]
    particle.state.strengths[0] *= 1.1
    after = estimator._joint_mixing_snapshot()

    diagnostics = estimator._joint_mixing_diagnostics(
        before,
        after,
        target_before=np.asarray([-3.0, -2.0, -1.0, 0.0]),
        target_after=np.asarray([-3.1, -2.0, -1.0, 0.0]),
    )

    assert diagnostics["state_change_weight_mass"] == pytest.approx(0.25)
    assert diagnostics["k_transition_weight_mass"] == 0.0
    assert diagnostics["surface_position_esjd_m2"] == 0.0
    assert diagnostics["log_strength_esjd"] > 0.0
    assert diagnostics["distinct_joint_state_count"] >= 2.0
    assert diagnostics["joint_k_vector_lag1_correlation"] == 1.0
    assert diagnostics["k_lag1_correlation.Cs-137"] == 1.0


def test_joint_k_vector_lag1_correlation_tracks_cardinality_reversal() -> None:
    """The joint cardinality diagnostic must expose inverse sweep movement."""
    before = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    after = 1.0 - before
    weights = np.full(4, 0.25, dtype=np.float64)

    correlation = RotatingShieldPFEstimator._weighted_vector_lag1_correlation(
        before,
        after,
        weights,
    )

    assert correlation == pytest.approx(-1.0)


def test_joint_cardinality_stratification_preserves_product_prior() -> None:
    """Guided initialization must stratify K vectors, not only marginals."""
    marginals = (
        np.asarray([0.25, 0.75], dtype=np.float64),
        np.asarray([0.5, 0.5], dtype=np.float64),
    )
    draws = _stratified_joint_cardinality_draws(
        marginals,
        400,
        rng=np.random.default_rng(7),
    )
    vectors, counts = np.unique(draws, axis=0, return_counts=True)
    observed = {
        tuple(vector.tolist()): int(count)
        for vector, count in zip(vectors, counts, strict=True)
    }
    assert observed == {
        (0, 0): 50,
        (0, 1): 50,
        (1, 0): 150,
        (1, 1): 150,
    }


def test_guided_initialization_uses_exact_prior_over_proposal_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guided initialization must weight its defensive joint mixture exactly."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=8,
            max_sources=2,
            variable_cardinality=True,
            init_num_sources=(0, 2),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            joint_guided_initialization=True,
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    alignment = np.zeros(atlas.chart_count, dtype=np.float64)
    alignment[0] = 1.0
    position_proposal = ContinuousSurfacePositionProposal(
        area_prior_probabilities=atlas.chart_probabilities,
        alignment_scores=alignment,
        prior_component_probability=0.5,
    )
    strength_proposal = ContinuousStrengthProposal(
        minimum=filt._strength_prior.minimum,
        maximum=filt._strength_prior.maximum,
        data_locations_by_chart=np.full(
            atlas.chart_count,
            0.5 * (filt._strength_prior.minimum + filt._strength_prior.maximum),
            dtype=np.float64,
        ),
        data_sigma=1.0,
        prior_component_probability=1.0,
        data_informative=False,
    )

    def _build_proposal(
        *_: object,
        **__: object,
    ) -> ContinuousSurfacePositionProposal:
        """Install exact prior position and strength proposal densities."""
        filt._structural_rj_strength_proposal = strength_proposal
        return position_proposal

    monkeypatch.setattr(
        filt,
        "_build_continuous_rj_position_proposal",
        _build_proposal,
    )
    identities_before = tuple(
        particle.joint_row_identity for particle in filt.continuous_particles
    )
    model = estimator._full_spectrum_model()
    station = JointStationObservation(
        spectrum_vb=np.zeros(
            (1, np.asarray(model.energy_axis_keV).size),
            dtype=np.float64,
        ),
        energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(1.5, 1.5, 1.5),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([1.0], dtype=np.float64),
        station_sequence_id=0,
    )

    estimator._apply_joint_guided_initialization(station)

    log_prior = np.zeros(8, dtype=np.float64)
    log_guided = np.zeros(8, dtype=np.float64)
    for row, particle in enumerate(filt.continuous_particles):
        chart_ids = particle.state.surface_chart_ids
        log_prior[row] = float(
            filt._structural_rj_cardinality_prior.log_prob(particle.state.num_sources)
        ) + float(np.sum(atlas.log_chart_probabilities[chart_ids]))
        log_guided[row] = float(
            filt._structural_rj_cardinality_prior.log_prob(particle.state.num_sources)
        ) + float(np.sum(position_proposal.log_density(chart_ids)))
    log_mixture = np.logaddexp(
        np.log(0.5) + log_prior,
        np.log(0.5) + log_guided,
    )
    log_ratio = log_prior - log_mixture
    expected_log_weights = log_ratio - np.logaddexp.reduce(log_ratio)
    expected_weights = np.exp(expected_log_weights)
    assert estimator._joint_guided_initialization_applied
    assert estimator.last_joint_guided_initialization_ess == pytest.approx(
        1.0 / np.sum(np.square(expected_weights))
    )
    assert np.allclose(
        estimator._strict_joint_particle_weights(),
        expected_weights,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert (
        tuple(particle.joint_row_identity for particle in filt.continuous_particles)
        == identities_before
    )


def test_guided_initialization_conditions_later_isotopes_on_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later isotope proposals must subtract earlier guided mean spectra."""
    torch = pytest.importorskip("torch")
    isotopes = ("Co-60", "Cs-137")
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=8,
            max_sources=2,
            variable_cardinality=True,
            init_num_sources=(0, 2),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            joint_guided_initialization=True,
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()
    station = JointStationObservation(
        spectrum_vb=np.zeros(
            (1, np.asarray(model.energy_axis_keV).size),
            dtype=np.float64,
        ),
        energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(1.5, 1.5, 1.5),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([1.0], dtype=np.float64),
        station_sequence_id=0,
    )
    expected_reference = np.full_like(station.spectrum_vb, 7.0)
    observed_references: list[NDArray[np.float64] | None] = []

    def _expected_means(_: object) -> object:
        """Return a deterministic earlier-isotope population mean."""
        return torch.as_tensor(
            np.broadcast_to(
                expected_reference,
                (estimator.pf_config.num_particles,) + expected_reference.shape,
            ).copy(),
            dtype=torch.float64,
        )

    monkeypatch.setattr(
        estimator,
        "_joint_station_expected_means_torch",
        _expected_means,
    )
    for isotope in isotopes:
        filt = estimator.filters[isotope]
        atlas = filt._structural_rj_surface_atlas
        assert atlas is not None

        def _build_proposal(
            *_: object,
            _filt: IsotopeParticleFilter = filt,
            _atlas: object = atlas,
            **__: object,
        ) -> ContinuousSurfacePositionProposal:
            """Record the active sequential residual and install a proposal."""
            reference = estimator._joint_birth_proposal_reference_mean_vb
            observed_references.append(
                None if reference is None else np.asarray(reference).copy()
            )
            proposal = ContinuousSurfacePositionProposal(
                area_prior_probabilities=_atlas.chart_probabilities,
                alignment_scores=np.ones(_atlas.chart_count, dtype=np.float64),
                prior_component_probability=0.5,
            )
            _filt._structural_rj_strength_proposal = ContinuousStrengthProposal(
                minimum=_filt._strength_prior.minimum,
                maximum=_filt._strength_prior.maximum,
                data_locations_by_chart=np.full(
                    _atlas.chart_count,
                    _filt._strength_prior.mean,
                    dtype=np.float64,
                ),
                data_sigma=1.0,
                prior_component_probability=1.0,
                data_informative=False,
                prior_family=_filt._strength_prior.family,
                prior_gamma_shape=_filt._strength_prior.gamma_shape,
                prior_gamma_scale=_filt._strength_prior.gamma_scale,
            )
            return proposal

        monkeypatch.setattr(
            filt,
            "_build_continuous_rj_position_proposal",
            _build_proposal,
        )

    estimator._apply_joint_guided_initialization(station)

    assert observed_references[0] is None
    np.testing.assert_allclose(
        observed_references[1],
        expected_reference,
        rtol=0.0,
        atol=0.0,
    )
    assert estimator._joint_birth_proposal_reference_mean_vb is None


def test_guided_cardinality_draws_cover_prior_strata_deterministically() -> None:
    """Large-probability K strata must not disappear through iid bad luck."""
    probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    draws = _stratified_categorical_draws(
        probabilities,
        100,
        rng=np.random.default_rng(1234),
    )

    np.testing.assert_array_equal(
        np.bincount(draws, minlength=3),
        np.asarray([20, 30, 50], dtype=np.int64),
    )


def test_adaptive_rejuvenation_fails_at_wall_time_without_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wall-time limit must fail rather than bless an unmixed state."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(
        joint_rejuvenation_min_sweeps=1,
        joint_rejuvenation_max_sweeps=2,
        joint_rejuvenation_min_state_change_weight_mass=0.1,
        joint_rejuvenation_min_surface_esjd_m2=1.0e-4,
        joint_rejuvenation_min_log_strength_esjd=1.0e-4,
        joint_rejuvenation_min_k_transition_weight_mass=1.0e-4,
        variable_cardinality=True,
        joint_rejuvenation_boundary_mass_threshold=0.05,
        joint_smc_rejuvenation_wall_time_limit_s=1800.0,
    )
    estimator.last_joint_rejuvenation_diagnostics = []
    estimator.last_joint_smc_wall_time_limit_exceeded = False
    calls = 0

    def _sweep(*_: object, **__: object) -> dict[str, float]:
        """Return inadequate movement once and adequate movement second."""
        nonlocal calls
        calls += 1
        return {
            "state_change_weight_mass": 0.05 if calls == 1 else 0.2,
            "surface_position_esjd_m2": 0.0 if calls == 1 else 0.1,
            "log_strength_esjd": 0.0,
            "k_transition_weight_mass": 0.0,
        }

    monkeypatch.setattr(estimator, "_joint_rejuvenate", _sweep)
    estimator._joint_rejuvenate_adaptive(
        (),
        target_beta=0.5,
        newest_prefix_count=2,
        station_start_s=time.perf_counter(),
    )

    assert calls == 2
    assert len(estimator.last_joint_rejuvenation_diagnostics) == 2
    assert not estimator.last_joint_smc_wall_time_limit_exceeded

    estimator.last_joint_rejuvenation_diagnostics = []
    estimator.pf_config.joint_smc_rejuvenation_wall_time_limit_s = 1.0
    calls = 0
    with pytest.raises(
        RuntimeError,
        match="exceeded its rejuvenation wall-time contract",
    ):
        estimator._joint_rejuvenate_adaptive(
            (),
            target_beta=0.5,
            newest_prefix_count=2,
            station_start_s=time.perf_counter() - 2.0,
        )

    assert calls == 1
    assert estimator.last_joint_smc_wall_time_limit_exceeded

    estimator.last_joint_rejuvenation_diagnostics = []
    estimator.last_joint_smc_wall_time_limit_exceeded = False
    calls = 1
    with pytest.raises(
        RuntimeError,
        match="exceeded its rejuvenation wall-time contract",
    ):
        estimator._joint_rejuvenate_adaptive(
            (),
            target_beta=0.5,
            newest_prefix_count=2,
            station_start_s=time.perf_counter() - 2.0,
        )

    assert calls == 2
    assert estimator.last_joint_smc_wall_time_limit_exceeded
    assert estimator.last_joint_rejuvenation_mixing_incomplete


def test_adaptive_rejuvenation_rejects_continuous_only_boundary_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K-boundary particles require an actual downward structure transition."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(
        joint_rejuvenation_min_sweeps=1,
        joint_rejuvenation_max_sweeps=3,
        joint_rejuvenation_min_state_change_weight_mass=0.1,
        joint_rejuvenation_min_surface_esjd_m2=1.0e-4,
        joint_rejuvenation_min_log_strength_esjd=1.0e-4,
        joint_rejuvenation_min_k_transition_weight_mass=1.0e-4,
        joint_smc_rejuvenation_wall_time_limit_s=1800.0,
        variable_cardinality=True,
        joint_rejuvenation_boundary_mass_threshold=0.05,
    )
    estimator.last_joint_rejuvenation_diagnostics = []
    estimator.last_joint_smc_wall_time_limit_exceeded = False
    estimator.last_joint_structural_mixing_incomplete = False
    calls = 0

    def _sweep(*_: object, **__: object) -> dict[str, float]:
        """Return ample continuous motion but no cardinality transition."""
        nonlocal calls
        calls += 1
        return {
            "state_change_weight_mass": 0.5,
            "surface_position_esjd_m2": 0.1,
            "log_strength_esjd": 0.1,
            "k_transition_weight_mass": 0.0,
            "ordinary_boundary_weight_mass": 1.0,
            "ordinary_boundary_escape_weight_mass": 0.0,
        }

    monkeypatch.setattr(estimator, "_joint_rejuvenate", _sweep)
    with pytest.raises(RuntimeError, match="reached its sweep limit"):
        estimator._joint_rejuvenate_adaptive(
            (),
            target_beta=1.0,
            newest_prefix_count=8,
            station_start_s=time.perf_counter(),
        )

    assert calls == 3
    assert estimator.last_joint_structural_mixing_incomplete
    assert all(
        entry["structural_movement_sufficient"] == 0.0
        for entry in estimator.last_joint_rejuvenation_diagnostics
    )


def test_joint_structural_geometry_rejects_same_length_row_mismatch() -> None:
    """Equal row counts must not hide detector or shield-history mismatch."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator._active_joint_structural_geometry = None
    station = SimpleNamespace(
        detector_position_xyz_m=(1.0, 2.0, 3.0),
        fe_indices=np.asarray([1, 2], dtype=np.int64),
        pb_indices=np.asarray([3, 4], dtype=np.int64),
        live_times_s=np.asarray([5.0, 6.0], dtype=np.float64),
        station_sequence_id=7,
    )
    valid = StructuralGeometryBatch(
        detector_positions=np.asarray(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            dtype=np.float64,
        ),
        fe_indices=np.asarray([1, 2], dtype=np.int64),
        pb_indices=np.asarray([3, 4], dtype=np.int64),
        live_times=np.asarray([5.0, 6.0], dtype=np.float64),
        station_sequence_ids=np.asarray([7, 7], dtype=np.int64),
    )
    estimator._validate_joint_structural_geometry(valid, (station,))
    invalid = StructuralGeometryBatch(
        detector_positions=np.asarray(
            [[1.0, 2.0, 3.0], [1.0, 2.5, 3.0]],
            dtype=np.float64,
        ),
        fe_indices=np.asarray([1, 2], dtype=np.int64),
        pb_indices=np.asarray([3, 4], dtype=np.int64),
        live_times=np.asarray([5.0, 6.0], dtype=np.float64),
        station_sequence_ids=np.asarray([7, 7], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="differs"):
        estimator._validate_joint_structural_geometry(invalid, (station,))


def test_joint_smc_recovers_before_an_inadmissible_minimum_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recoverable minimum-beta failure must rejuvenate the current target."""
    torch = pytest.importorskip("torch")
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            target_ess_ratio=0.5,
            min_delta_beta=0.1,
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()
    filt = estimator.filters["Cs-137"]
    for particle, weight in zip(
        filt.continuous_particles,
        (0.4, 0.3, 0.2, 0.1),
        strict=True,
    ):
        particle.log_weight = float(np.log(weight))
    select_calls = 0

    def _select_delta_beta(**kwargs: object) -> tuple[float, object, float]:
        """Request one current-target recovery, then finish tempering."""
        nonlocal select_calls

        select_calls += 1
        if select_calls == 1:
            raise TemperingIncrementRequiresRejuvenation("recover first")
        remaining = float(kwargs["remaining"])
        return (
            remaining,
            torch.full((4,), -np.log(4.0), dtype=torch.float64),
            4.0,
        )

    monkeypatch.setattr(filt, "_select_delta_beta", _select_delta_beta)
    monkeypatch.setattr(
        estimator,
        "_joint_station_prefix_log_likelihood_torch",
        lambda station: torch.zeros(
            (2, 4),
            dtype=torch.float64,
        ),
    )
    rejuvenation_betas: list[float] = []
    def _record_rejuvenation(
        stations: object,
        target_beta: float,
        newest_prefix_count: int | None = None,
    ) -> dict[str, float]:
        """Record the target and return complete mixing evidence."""
        del stations, newest_prefix_count
        rejuvenation_betas.append(float(target_beta))
        return _sufficient_mixing_diagnostics()

    monkeypatch.setattr(estimator, "_joint_rejuvenate", _record_rejuvenation)
    monkeypatch.setattr(
        estimator,
        "_promote_joint_birth_proposal_station",
        lambda station: None,
    )
    station = JointStationObservation(
        spectrum_vb=np.zeros(
            (1, np.asarray(model.energy_axis_keV).size),
            dtype=np.float64,
        ),
        energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
        generative_contract_hash_sha256=model.contract_hash_sha256,
        pose_idx=0,
        detector_position_xyz_m=(1.5, 1.5, 1.5),
        fe_indices=np.asarray([0], dtype=np.int64),
        pb_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([1.0], dtype=np.float64),
        station_sequence_id=0,
    )

    estimator._joint_tempered_station_update(station)

    assert select_calls == 2
    assert rejuvenation_betas == [0.0, 1.0]
    assert filt.last_temper_resample_count == 1
    assert estimator.last_joint_temper_steps[0] == pytest.approx(
        {
            "prefix_count": 1.0,
            "prefix_view_count": 1.0,
            "station_beta": 0.0,
            "beta_total": 0.0,
            "delta_beta": 0.0,
            "ess": 1.0 / (0.4**2 + 0.3**2 + 0.2**2 + 0.1**2),
            "resampled": 1.0,
            "recovery_rejuvenation": 1.0,
            "station_unique_ancestors": float(
                estimator.last_joint_station_unique_ancestor_count
            ),
            "cumulative_unique_ancestors": float(
                estimator.last_joint_cumulative_unique_ancestor_count
            ),
        }
    )
    assert estimator.last_joint_temper_steps[-1]["beta_total"] == 1.0
    assert estimator.last_joint_temper_steps[-1]["recovery_rejuvenation"] == 0.0


def test_torch_weight_and_likelihood_normalization_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN, positive infinity, and all-impossible mass must never normalize."""
    torch = pytest.importorskip("torch")
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]

    for invalid in (
        torch.tensor([0.0, float("nan")], dtype=torch.float64),
        torch.tensor([0.0, float("inf")], dtype=torch.float64),
        torch.full((2,), float("-inf"), dtype=torch.float64),
    ):
        with pytest.raises(RuntimeError):
            filt._normalized_log_weights_torch(invalid)

    huge_negative = torch.full(
        (4,),
        -1.0e16,
        dtype=torch.float64,
    )
    huge_normalized = filt._normalized_log_weights_torch(huge_negative)
    assert float(torch.logsumexp(huge_normalized, dim=0)) == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert filt._ess_from_logw_torch(huge_normalized) == pytest.approx(4.0)

    normalized = torch.full((4,), -np.log(4.0), dtype=torch.float64)
    for invalid_likelihood in (
        torch.tensor(
            [0.0, float("nan"), 0.0, 0.0],
            dtype=torch.float64,
        ),
        torch.tensor(
            [0.0, float("inf"), 0.0, 0.0],
            dtype=torch.float64,
        ),
        torch.full((4,), float("-inf"), dtype=torch.float64),
    ):
        with pytest.raises(RuntimeError):
            filt._select_delta_beta(
                normalized,
                invalid_likelihood,
                remaining=1.0,
                target_ess=2.0,
            )

    filt.config.min_delta_beta = 0.1
    with pytest.raises(TemperingIncrementRequiresRejuvenation):
        filt._select_delta_beta(
            normalized,
            torch.tensor([0.0, -100.0, -100.0, -100.0]),
            remaining=1.0,
            target_ess=3.9,
        )

    particle_count = len(filt.continuous_particles)
    total = torch.zeros(
        (particle_count, 1, 1, 1),
        dtype=torch.float64,
    )
    features = torch.zeros(
        (particle_count, 1, 1, 1, 4),
        dtype=torch.float64,
    )
    monkeypatch.setattr(
        estimator,
        "_joint_station_transport_components_torch",
        lambda station: (total, total.clone(), features),
    )
    monkeypatch.setattr(
        estimator,
        "_full_spectrum_model",
        lambda: SimpleNamespace(
            log_likelihood_torch=lambda *args: torch.full(
                (particle_count,),
                float("-inf"),
                dtype=torch.float64,
            )
        ),
    )
    station = SimpleNamespace(
        spectrum_vb=np.zeros((1, 1), dtype=np.float64),
        live_times_s=np.ones(1, dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="every particle"):
        estimator._joint_station_log_likelihood_torch(station)


def test_joint_planning_rejects_all_impossible_particle_rows() -> None:
    """Planning must not turn an invalid all-minus-infinity posterior uniform."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    for particle in estimator.filters["Cs-137"].continuous_particles:
        particle.log_weight = float("-inf")

    with pytest.raises(RuntimeError, match="common log weights"):
        estimator.planning_joint_particles()


def test_birth_proposal_prefix_evaluates_only_pending_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed likelihood grids must not be rescanned after prefix promotion."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    model = estimator._full_spectrum_model()

    def _station(sequence_id: int) -> JointStationObservation:
        """Build one single-view station for proposal-prefix bookkeeping."""
        return JointStationObservation(
            spectrum_vb=np.zeros(
                (1, np.asarray(model.energy_axis_keV).size),
                dtype=np.float64,
            ),
            energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
            generative_contract_hash_sha256=model.contract_hash_sha256,
            pose_idx=0,
            detector_position_xyz_m=(1.5, 1.5, 1.5),
            fe_indices=np.asarray([0], dtype=np.int64),
            pb_indices=np.asarray([0], dtype=np.int64),
            live_times_s=np.asarray([1.0], dtype=np.float64),
            station_sequence_id=sequence_id,
        )

    completed = (_station(0), _station(1))
    pending = _station(2)
    estimator._joint_station_history = list(completed)
    estimator._active_joint_station_history = (*completed, pending)
    strength_count = int(estimator.pf_config.structural_rj_strength_proposal_grid_size)
    estimator._joint_birth_proposal_prefix_scores = {
        "Cs-137": np.zeros(
            (atlas.chart_count, strength_count),
            dtype=np.float64,
        )
    }
    estimator._joint_birth_proposal_prefix_station_count = len(completed)
    evaluated_sequence_ids: list[int] = []

    def _score_grid(**kwargs: object) -> NDArray[np.float64]:
        """Record the station passed to the expensive proposal scorer."""
        station = kwargs["station"]
        assert isinstance(station, JointStationObservation)
        evaluated_sequence_ids.append(int(station.station_sequence_id))
        return np.ones(
            (atlas.chart_count, strength_count),
            dtype=np.float64,
        )

    monkeypatch.setattr(
        estimator,
        "_joint_station_birth_proposal_score_grid",
        _score_grid,
    )
    geometry = StructuralGeometryBatch(
        detector_positions=np.full((3, 3), 1.5, dtype=np.float64),
        fe_indices=np.zeros(3, dtype=np.int64),
        pb_indices=np.zeros(3, dtype=np.int64),
        live_times=np.ones(3, dtype=np.float64),
        station_sequence_ids=np.arange(3, dtype=np.int64),
    )

    alignment, _strength_locations, informative = (
        estimator._joint_structural_proposal_evaluator(
            filt=filt,
            data=geometry,
            chart_centers_xyz=np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            ),
            target_beta=0.5,
        )
    )

    assert evaluated_sequence_ids == [2]
    assert informative
    assert np.all(alignment > 0.0)


def test_incremental_transport_cache_refresh_changes_one_isotope_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditional RJ refresh must preserve every unmoved isotope component."""
    torch = pytest.importorskip("torch")
    isotopes = ("Co-60", "Cs-137")
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    total = np.zeros((4, 2, 2, 3), dtype=np.float64)
    features = np.zeros((4, 2, 2, 3, 4), dtype=np.float64)
    estimator._joint_structural_transport_cache = (
        total.copy(),
        total.copy(),
        features.copy(),
    )
    requested_isotopes: list[str] = []

    def _isotope_components(
        station: object,
        isotope: str,
        *,
        particle_indices: NDArray[np.int64] | None = None,
    ) -> tuple[object, object, object]:
        """Return deterministic components for one station and isotope."""
        requested_isotopes.append(str(isotope))
        sequence_id = int(getattr(station, "station_sequence_id"))
        value = float(7 + sequence_id)
        particle_count = 4 if particle_indices is None else int(particle_indices.size)
        line_values = torch.full(
            (particle_count, 1, 1, 3),
            value,
            dtype=torch.float64,
        )
        feature_values = torch.full(
            (particle_count, 1, 1, 3, 4),
            value + 10.0,
            dtype=torch.float64,
        )
        return line_values, line_values + 1.0, feature_values

    monkeypatch.setattr(
        estimator,
        "_joint_isotope_station_transport_components_torch",
        _isotope_components,
    )
    stations = (
        SimpleNamespace(station_sequence_id=0),
        SimpleNamespace(station_sequence_id=1),
    )

    estimator._refresh_joint_structural_transport_cache_isotope(
        stations,
        "Cs-137",
    )

    refreshed = estimator._joint_structural_transport_cache
    assert refreshed is not None
    assert requested_isotopes == ["Cs-137", "Cs-137"]
    assert np.all(refreshed[0][:, :, 0, :] == 0.0)
    assert np.all(refreshed[0][:, 0, 1, :] == 7.0)
    assert np.all(refreshed[0][:, 1, 1, :] == 8.0)
    assert np.all(refreshed[1][:, :, 0, :] == 0.0)
    assert np.all(refreshed[2][:, :, 0, :, :] == 0.0)
    assert np.all(refreshed[2][:, 0, 1, :, :] == 17.0)
    assert np.all(refreshed[2][:, 1, 1, :, :] == 18.0)

    refreshed[0][0, :, 1, :] = -2.0
    requested_isotopes.clear()
    estimator._refresh_joint_structural_transport_cache_isotope(
        stations,
        "Cs-137",
        particle_indices=np.asarray([2], dtype=np.int64),
    )
    assert requested_isotopes == ["Cs-137", "Cs-137"]
    assert np.all(refreshed[0][0, :, 1, :] == -2.0)
    assert np.all(refreshed[0][2, 0, 1, :] == 7.0)
    assert np.all(refreshed[0][2, 1, 1, :] == 8.0)


def test_continuous_pf_state_position_is_independent_of_chart_center() -> None:
    """Chart centers may guide proposals but must never quantize PF support."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    state = IsotopeState(
        num_sources=1,
        strengths=np.asarray([100.0], dtype=np.float64),
        surface_chart_ids=np.asarray([0], dtype=np.int64),
        surface_uv=np.asarray([[0.123, 0.789]], dtype=np.float64),
    )
    original_position = filt.continuous_state_positions(state)
    vertices = np.asarray(
        atlas.geometry.vertices_xyz[0],
        dtype=np.float64,
    )
    expected_position = (
        vertices[0]
        + 0.123 * (vertices[1] - vertices[0])
        + 0.789 * (vertices[3] - vertices[0])
    )
    original_center = np.asarray(
        atlas.geometry.centers_xyz[0],
        dtype=np.float64,
    )

    assert np.allclose(
        original_position[0],
        expected_position,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert not np.array_equal(
        original_position[0],
        original_center,
    )


def test_zero_beta_joint_target_does_not_evaluate_pending_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beta zero must ignore, not multiply, an impossible pending likelihood."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    completed = SimpleNamespace(
        fe_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([1.0], dtype=np.float64),
        spectrum_vb=np.asarray([[0.0]], dtype=np.float64),
        station_sequence_id=0,
    )
    pending = SimpleNamespace(
        fe_indices=np.asarray([0], dtype=np.int64),
        live_times_s=np.asarray([1.0], dtype=np.float64),
        spectrum_vb=np.asarray([[1.0]], dtype=np.float64),
        station_sequence_id=1,
    )
    evaluated: list[int] = []

    def _cross_likelihood(
        observed: NDArray[np.float64],
        *_: object,
        **__: object,
    ) -> NDArray[np.float64]:
        """Return finite mass while recording the batched station action."""
        station_markers = np.asarray(observed, dtype=np.float64)[:, 0, 0, 0]
        evaluated.extend(station_markers.astype(np.int64).tolist())
        if np.any(station_markers == 1.0):
            raise AssertionError("beta-zero pending station was evaluated")
        return np.asarray([[[-2.0, -3.0]]], dtype=np.float64)

    monkeypatch.setattr(
        estimator,
        "_full_spectrum_model",
        lambda: SimpleNamespace(
            transport_feature_order=(
                "tau_fe",
                "tau_pb",
                "tau_obstacle",
                "distance_m",
            ),
            cross_log_likelihood_numpy=_cross_likelihood,
        ),
    )
    total = np.zeros((2, 2, 1, 1), dtype=np.float64)
    features = np.zeros((2, 2, 1, 1, 4), dtype=np.float64)

    result = estimator._joint_history_log_likelihood_numpy(
        filt=filt,
        stations=(completed, pending),
        total_nvsl=total,
        uncollided_nvsl=total.copy(),
        features_nvslf=features,
        target_beta=0.0,
    )

    assert evaluated == [0]
    assert np.array_equal(result, np.asarray([-2.0, -3.0]))


def test_joint_history_station_batch_matches_serial_likelihood_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One action-axis call must equal independent station latent integrals."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    model = estimator._full_spectrum_model()
    view_count = 2
    particle_count = 3
    station_count = 3
    line_count = len(tuple(model.line_identity))
    bin_count = np.asarray(model.energy_axis_keV).size
    live_times = np.asarray([1.0, 2.0], dtype=np.float64)
    stations = tuple(
        SimpleNamespace(
            fe_indices=np.arange(view_count, dtype=np.int64),
            live_times_s=live_times.copy(),
            spectrum_vb=np.zeros(
                (view_count, bin_count),
                dtype=np.float64,
            ),
            station_sequence_id=index,
        )
        for index in range(station_count)
    )
    rng = np.random.default_rng(20260728)
    total = rng.uniform(
        0.0,
        0.05,
        size=(
            particle_count,
            station_count * view_count,
            1,
            line_count,
        ),
    )
    uncollided = 0.8 * total
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    features[..., 3] = 1.0
    beta = 0.35
    expected = np.zeros(particle_count, dtype=np.float64)
    for station_index, station in enumerate(stations):
        row_slice = slice(
            station_index * view_count,
            (station_index + 1) * view_count,
        )
        station_ll = model.log_likelihood_numpy(
            station.spectrum_vb,
            total[:, row_slice, :, :],
            uncollided[:, row_slice, :, :],
            features[:, row_slice, :, :, :],
            station.live_times_s,
        )
        expected += (beta if station_index == station_count - 1 else 1.0) * station_ll
    original_cross = model.cross_log_likelihood_numpy
    call_shapes: list[tuple[int, ...]] = []
    action_chunks: list[int] = []

    def _recording_cross(*args: object, **kwargs: object) -> np.ndarray:
        """Record the station action axis and delegate to the real model."""
        call_shapes.append(tuple(np.asarray(args[0]).shape))
        action_chunks.append(int(kwargs["action_chunk_size"]))
        return np.asarray(original_cross(*args, **kwargs), dtype=np.float64)

    monkeypatch.setattr(
        model,
        "cross_log_likelihood_numpy",
        _recording_cross,
    )
    actual = estimator._joint_history_log_likelihood_numpy(
        filt=filt,
        stations=stations,
        total_nvsl=total,
        uncollided_nvsl=uncollided,
        features_nvslf=features,
        target_beta=beta,
    )

    assert call_shapes == [(station_count, 1, view_count, bin_count)]
    assert action_chunks == [station_count]
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_full_spectrum_likelihood_ignores_exact_zero_source_slots() -> None:
    """Dropping unused slots must preserve NumPy and Torch model likelihoods."""
    model = approved_full_spectrum_model()
    rng = np.random.default_rng(20260805)
    particle_count = 3
    view_count = 2
    source_slots = 6
    line_count = len(tuple(model.line_identity))
    bin_count = np.asarray(model.energy_axis_keV).size
    total = np.zeros(
        (particle_count, view_count, source_slots, line_count),
        dtype=np.float64,
    )
    total[:, :, (1, 4), :] = rng.uniform(
        0.0,
        0.05,
        size=(particle_count, view_count, 2, line_count),
    )
    uncollided = 0.75 * total
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    features[..., 3] = 1.0
    observed = np.zeros((view_count, bin_count), dtype=np.float64)
    live_times = np.asarray([1.0, 2.0], dtype=np.float64)
    active_slots = np.asarray([1, 4], dtype=np.int64)

    expanded = model.log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
    )
    compact = model.log_likelihood_numpy(
        observed,
        total[:, :, active_slots, :],
        uncollided[:, :, active_slots, :],
        features[:, :, active_slots, :, :],
        live_times,
    )
    np.testing.assert_allclose(compact, expanded, rtol=1.0e-13, atol=1.0e-13)

    torch = pytest.importorskip("torch")
    expanded_torch = model.log_likelihood_torch(
        observed,
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        live_times,
    )
    compact_torch = model.log_likelihood_torch(
        observed,
        torch.as_tensor(total[:, :, active_slots, :], dtype=torch.float64),
        torch.as_tensor(
            uncollided[:, :, active_slots, :],
            dtype=torch.float64,
        ),
        torch.as_tensor(
            features[:, :, active_slots, :, :],
            dtype=torch.float64,
        ),
        live_times,
    )
    torch.testing.assert_close(
        compact_torch,
        expanded_torch,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_joint_history_prefix_bridge_matches_exact_shared_latent_oracle() -> None:
    """The newest-station bridge must interpolate exact prefix marginals."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    model = estimator._full_spectrum_model()
    view_count = 3
    particle_count = 4
    line_count = len(tuple(model.line_identity))
    bin_count = np.asarray(model.energy_axis_keV).size
    live_times = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    stations = tuple(
        SimpleNamespace(
            fe_indices=np.arange(view_count, dtype=np.int64),
            live_times_s=live_times.copy(),
            spectrum_vb=np.zeros(
                (view_count, bin_count),
                dtype=np.float64,
            ),
            station_sequence_id=index,
        )
        for index in range(2)
    )
    rng = np.random.default_rng(20260730)
    total = rng.uniform(
        0.0,
        0.05,
        size=(particle_count, 2 * view_count, 1, line_count),
    )
    uncollided = 0.8 * total
    features = np.zeros(total.shape + (4,), dtype=np.float64)
    features[..., 3] = 1.0
    beta = 0.35
    prefix_count = 2
    past_ll = model.log_likelihood_numpy(
        stations[0].spectrum_vb,
        total[:, :view_count],
        uncollided[:, :view_count],
        features[:, :view_count],
        live_times,
    )
    prefixes = model.prefix_log_likelihood_numpy(
        stations[1].spectrum_vb,
        total[:, view_count:],
        uncollided[:, view_count:],
        features[:, view_count:],
        live_times,
    )
    expected = (
        past_ll
        + (1.0 - beta) * prefixes[prefix_count - 1]
        + beta * prefixes[prefix_count]
    )

    actual = estimator._joint_history_log_likelihood_numpy(
        filt=filt,
        stations=stations,
        total_nvsl=total,
        uncollided_nvsl=uncollided,
        features_nvslf=features,
        target_beta=beta,
        newest_prefix_count=prefix_count,
    )

    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_joint_pf_rejects_different_isotope_surface_atlas_digests() -> None:
    """One aligned joint row must have the same geometric meaning per isotope."""
    isotopes = ("Co-60", "Cs-137")
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    estimator.filters["Co-60"]._structural_rj_surface_atlas_sha256 = "0" * 64

    with pytest.raises(RuntimeError, match="different continuous surface"):
        estimator._assert_joint_particle_alignment()
    with pytest.raises(RuntimeError, match="different continuous surface"):
        estimator.surface_atlas_area_quadrature(
            max_points=8,
            maximum_hausdorff_bound_m=1.0,
        )


def test_persistent_transport_cache_reuses_and_appends_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted-state transport must be reused and append only a new station."""
    torch = pytest.importorskip("torch")
    estimator = RotatingShieldPFEstimator(
        isotopes=("Co-60", "Cs-137"),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Co-60": 0.0, "Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    model = estimator._full_spectrum_model()
    line_count = len(model.line_identity)
    feature_count = len(model.transport_feature_order)
    calls: list[int] = []

    def _components(station: object) -> tuple[object, object, object]:
        """Return one deterministic full joint transport cache slab."""
        sequence = int(getattr(station, "station_sequence_id"))
        calls.append(sequence)
        total = torch.full(
            (4, 1, 2, line_count),
            float(sequence + 1),
            dtype=torch.float64,
        )
        features = torch.full(
            (4, 1, 2, line_count, feature_count),
            float(sequence + 11),
            dtype=torch.float64,
        )
        return total, total + 0.5, features

    monkeypatch.setattr(
        estimator,
        "_joint_station_transport_components_torch",
        _components,
    )

    def _station(sequence: int) -> JointStationObservation:
        """Return one minimal full-spectrum station with strict geometry."""
        return JointStationObservation(
            spectrum_vb=np.zeros(
                (1, np.asarray(model.energy_axis_keV).size),
                dtype=np.float64,
            ),
            energy_axis_keV=np.asarray(model.energy_axis_keV),
            generative_contract_hash_sha256=model.contract_hash_sha256,
            pose_idx=sequence,
            detector_position_xyz_m=(1.0 + sequence, 1.0, 1.0),
            fe_indices=np.asarray([sequence % 8], dtype=np.int64),
            pb_indices=np.asarray([(sequence + 1) % 8], dtype=np.int64),
            live_times_s=np.asarray([1.0], dtype=np.float64),
            station_sequence_id=sequence,
        )

    first = _station(0)
    second = _station(1)
    estimator._refresh_joint_structural_transport_cache((first,))
    initial = estimator._joint_persistent_structural_transport_cache
    assert initial is not None
    assert calls == [0]

    estimator._refresh_joint_structural_transport_cache((first,))
    assert calls == [0]
    assert estimator.last_joint_persistent_cache_reuse_count == 1
    assert estimator._joint_structural_transport_cache is initial

    estimator._refresh_joint_structural_transport_cache((first, second))
    appended = estimator._joint_persistent_structural_transport_cache
    assert appended is not None
    assert calls == [0, 1]
    assert estimator.last_joint_persistent_cache_append_count == 1
    np.testing.assert_array_equal(appended[0][:, 0], 1.0)
    np.testing.assert_array_equal(appended[0][:, 1], 2.0)


def test_persistent_transport_cache_follows_joint_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent cache rows must use the exact common ancestor vector."""
    isotopes = ("Co-60", "Cs-137", "Eu-154")
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0 for isotope in isotopes},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            variable_cardinality=True,
            init_num_sources=(0, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(np.asarray([1.5, 1.5, 1.5]))
    estimator._ensure_kernel_cache()
    rows = np.arange(4, dtype=np.float64).reshape(4, 1, 1, 1)
    estimator._joint_persistent_structural_transport_cache = (
        rows.copy(),
        rows.copy() + 10.0,
        rows[..., None].copy() + 20.0,
    )
    estimator._joint_structural_transport_cache = (
        estimator._joint_persistent_structural_transport_cache
    )
    indices = np.asarray([3, 1, 1, 0], dtype=np.int64)
    monkeypatch.setattr(
        "pf.estimator_rejuvenation.systematic_resample",
        lambda *_args, **_kwargs: indices.copy(),
    )

    returned = estimator._resample_joint_particles(
        np.full(4, -np.log(4.0), dtype=np.float64)
    )

    np.testing.assert_array_equal(returned, indices)
    cached = estimator._joint_persistent_structural_transport_cache
    assert cached is not None
    np.testing.assert_array_equal(cached[0][:, 0, 0, 0], indices)
    assert estimator.last_joint_persistent_cache_reindex_count == 1
