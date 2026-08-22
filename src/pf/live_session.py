"""PF construction and record ingestion for one live runtime session."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from hashlib import sha256
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.surface_atlas import ContinuousSurfaceAtlas
from measurement.surface_charts import build_surface_chart_geometry
from numpy.typing import NDArray
from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.forward_context import ResolvedForwardContext
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogValidationError,
    MeasurementLogView,
)
from runtime.prefix import measurement_records_digest
from runtime.provenance import DigestIdentity
from runtime.records import RunContext

from pf.estimator_types import JointPlanningParticles
from pf.gpu_utils import preflight_compute_backend
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig


class PFLiveSessionError(RuntimeError):
    """Report an incompatible live context, setting, or observation."""


@dataclass(frozen=True, slots=True)
class PFLiveParticleSnapshot:
    """Expose one immutable truth-free particle generation for planning."""

    source_run_id: str
    record_count: int
    station_count: int
    covered_records_digest: DigestIdentity
    isotope_order: tuple[str, ...]
    weights_n: NDArray[np.float64]
    positions_nk3_by_isotope: Mapping[str, NDArray[np.float64]]
    surface_chart_ids_nk_by_isotope: Mapping[str, NDArray[np.int64]]
    surface_uv_nk2_by_isotope: Mapping[str, NDArray[np.float64]]
    strengths_nk_by_isotope: Mapping[str, NDArray[np.float64]]
    source_mask_nk_by_isotope: Mapping[str, NDArray[np.bool_]]
    original_particle_indices: NDArray[np.int64]
    posterior_summary_json: bytes
    posterior_summary_sha256: str

    def posterior_summary(self) -> dict[str, object]:
        """Return a detached JSON-compatible copy of the live PF summary."""
        payload = json.loads(self.posterior_summary_json)
        if not isinstance(payload, dict):
            raise PFLiveSessionError("PF live posterior summary must be an object.")
        return payload


@dataclass(frozen=True, slots=True)
class PFCompletedLiveState:
    """Seal a completed live posterior before MeasurementLog publication."""

    source_run_id: str
    runtime_config_sha256: str
    generative_contract_hash_sha256: str
    record_count: int
    station_count: int
    covered_step_ids: tuple[int, ...]
    covered_records_digest: DigestIdentity
    checkpoint_state: bytes
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class PFBoundLiveState:
    """Provide immutable publication inputs after exact final-log binding."""

    completed: PFCompletedLiveState
    measurement_log_sha256: str
    posterior_json: bytes
    posterior_sha256: str

    @property
    def checkpoint_state(self) -> bytes:
        """Return the already-completed PF checkpoint without recomputation."""
        return self.completed.checkpoint_state

    @property
    def checkpoint_sha256(self) -> str:
        """Return the digest of the already-completed PF checkpoint."""
        return self.completed.checkpoint_sha256


_SHA256_PATTERN = frozenset("0123456789abcdef")
_DEFAULT_SURFACE_DIAGNOSTIC_POINT_COUNT = 1024
_PF_CONFIG_ALIASES = {
    "pf_detected_isotopes_only": "detected_isotopes_only",
    "pf_max_sources": "max_sources",
    "pf_hard_max_sources": "hard_max_sources",
    "pf_strength_prior_min_cps_1m": "strength_prior_min_cps_1m",
    "pf_strength_prior_max_cps_1m": "strength_prior_max_cps_1m",
    "pf_strength_prior_family": "strength_prior_family",
    "pf_strength_prior_gamma_shape": "strength_prior_gamma_shape",
    "pf_strength_prior_gamma_scale_cps_1m": "strength_prior_gamma_scale_cps_1m",
}
_PF_PHYSICAL_OVERRIDE_KEYS = frozenset(
    {
        "pf_buildup",
        "pf_detector_aperture_radius_m",
        "pf_detector_aperture_samples",
        "pf_detector_aperture_sampling",
        "pf_detector_count_radius_m",
        "pf_line_resolved_shield_attenuation",
        "pf_obstacle_attenuation",
        "pf_obstacle_material",
        "pf_obstacle_mu_by_isotope",
        "pf_obstacle_source_extent_radius_m",
        "pf_obstacle_source_extent_samples",
        "pf_source_extent_radius_m",
        "pf_source_extent_samples",
    }
)


def _sha256_string(value: object, *, location: str) -> str:
    """Return one exact lowercase SHA-256 string without coercion."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_PATTERN for character in value)
    ):
        raise PFLiveSessionError(
            f"{location} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _json_integer(
    value: object,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    """Return one exact JSON integer without boolean or float coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PFLiveSessionError(f"{location} must be a JSON integer.")
    if minimum is not None and value < minimum:
        raise PFLiveSessionError(f"{location} must be at least {minimum}.")
    return value


def _finite_real(value: object, *, location: str) -> float:
    """Return one exact finite real without boolean or string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PFLiveSessionError(f"{location} must be a finite JSON number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise PFLiveSessionError(f"{location} must be a finite JSON number.")
    return parsed


def _surface_atlas_diagnostic_points(
    environment: EnvironmentConfig,
    *,
    pf_config: RotatingShieldPFConfig,
    obstacle_grid: ObstacleGrid | None,
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Build deterministic PF diagnostics from shared physical geometry."""
    point_count = _DEFAULT_SURFACE_DIAGNOSTIC_POINT_COUNT
    try:
        geometry = build_surface_chart_geometry(
            environment,
            obstacle_grid,
            max_edge_m=pf_config.structural_rj_surface_chart_max_edge_m,
            obstacle_height_m=obstacle_height_m,
        )
        if not geometry.obstacle_surfaces_available:
            raise ValueError("complete obstacle surfaces are unavailable")
        atlas = ContinuousSurfaceAtlas(geometry)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            "Cannot reconstruct the runtime continuous surface atlas."
        ) from exc
    quantiles = (np.arange(point_count, dtype=np.float64) + 0.5) / float(point_count)
    chart_ids = np.searchsorted(
        np.cumsum(atlas.chart_probabilities),
        quantiles,
        side="right",
    ).astype(np.int64)
    if np.any(chart_ids < 0) or np.any(chart_ids >= atlas.chart_count):
        raise PFLiveSessionError("Surface-atlas diagnostic chart IDs are invalid.")
    sequence = np.arange(point_count, dtype=np.float64) + 0.5
    uv = np.column_stack(
        (
            np.mod(sequence * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0),
            np.mod(sequence * (np.sqrt(2.0) - 1.0), 1.0),
        )
    )
    return np.ascontiguousarray(atlas.positions_xyz(chart_ids, uv))


def _pf_config_values(
    config: Mapping[str, Any],
    *,
    profile: str,
    upper: NDArray[np.float64],
) -> dict[str, Any]:
    """Select declared PF dataclass fields from resolved settings."""
    allowed = {field.name for field in fields(RotatingShieldPFConfig)}
    values = {key: value for key, value in config.items() if key in allowed}
    values["estimator_profile"] = str(profile)
    values["position_max"] = tuple(float(value) for value in upper)
    return values


def _external_pf_config(
    external_config: Mapping[str, Any],
    *,
    upper: NDArray[np.float64],
) -> dict[str, Any]:
    """Resolve PF settings without overriding runtime-owned physics."""
    forbidden_physics = sorted(
        key for key in external_config if key in _PF_PHYSICAL_OVERRIDE_KEYS
    )
    if forbidden_physics:
        raise PFLiveSessionError(
            "External PF settings cannot override runtime physics: "
            + ", ".join(forbidden_physics)
        )
    if external_config:
        try:
            enforce_pure_runtime_settings(external_config)
        except ValueError as exc:
            raise PFLiveSessionError(
                "External settings violate the pure-PF schema."
            ) from exc
    normalized = dict(external_config)
    for alias, canonical in _PF_CONFIG_ALIASES.items():
        if alias not in normalized:
            continue
        if canonical in normalized and normalized[canonical] != normalized[alias]:
            raise PFLiveSessionError(
                f"Conflicting PF settings {alias!r} and {canonical!r}."
            )
        normalized[canonical] = normalized[alias]
    declared = {field.name for field in fields(RotatingShieldPFConfig)}
    merged = {
        key: value
        for key, value in normalized.items()
        if key in declared or key in {"pure_pf_schema_version", "estimator_profile"}
    }
    merged.setdefault("pure_pf_schema_version", 1)
    merged["position_max"] = tuple(float(value) for value in upper)
    return merged


def _build_live_estimator_from_forward_context(
    forward: ResolvedForwardContext,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    measurement_log_schema_version: int,
    measurement_runtime_config_sha256: str,
    config_hash: str | None = None,
    inference_isotopes: Sequence[str] | None = None,
) -> PurePFEstimator:
    """Build one live PF from an authenticated shared physical context."""
    if not isinstance(config, Mapping):
        raise PFLiveSessionError("PF configuration must be an object.")
    if any(not isinstance(key, str) for key in config):
        raise PFLiveSessionError("PF configuration keys must be JSON strings.")
    if not isinstance(profile, str):
        raise PFLiveSessionError("PF profile must be a JSON string.")
    session_seed = _json_integer(seed, location="seed", minimum=0)
    schema_version = _json_integer(
        measurement_log_schema_version,
        location="measurement_log_schema_version",
        minimum=1,
    )
    runtime_config_sha256 = _sha256_string(
        measurement_runtime_config_sha256,
        location="measurement_runtime_config_sha256",
    )
    pending_log_digest = "unavailable"
    isotopes = tuple(forward.isotopes)
    active_isotopes = (
        isotopes
        if inference_isotopes is None
        else tuple(str(value) for value in inference_isotopes)
    )
    if (
        not active_isotopes
        or len(set(active_isotopes)) != len(active_isotopes)
        or not set(active_isotopes).issubset(isotopes)
    ):
        raise PFLiveSessionError(
            "Inference isotopes must be a unique nonempty subset of the runtime "
            "candidate isotopes."
        )
    active_isotopes = tuple(
        isotope for isotope in isotopes if isotope in set(active_isotopes)
    )
    _, upper = forward.bounds_xyz
    session_config = _external_pf_config(config, upper=upper)
    try:
        pure_config = enforce_pure_runtime_settings(session_config, profile=profile)
        pf_config = RotatingShieldPFConfig(
            **_pf_config_values(pure_config, profile=profile, upper=upper)
        )
        apply_profile_to_config(pf_config)
        preflight_compute_backend(
            use_gpu=bool(pf_config.use_gpu),
            gpu_device=str(pf_config.gpu_device),
            gpu_dtype=str(pf_config.gpu_dtype),
        )
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError("External PF configuration is incompatible.") from exc
    observation_model = forward.observation_model
    obstacle_grid = forward.obstacle_grid
    obstacle_enabled = forward.obstacle_attenuation_enabled
    if obstacle_grid is not None and not obstacle_enabled:
        raise PFLiveSessionError(
            "A runtime obstacle grid requires physical obstacle attenuation."
        )
    pf_obstacle_grid = obstacle_grid if obstacle_enabled else None
    surface_diagnostic_points = _surface_atlas_diagnostic_points(
        forward.environment,
        pf_config=pf_config,
        obstacle_grid=pf_obstacle_grid,
        obstacle_height_m=observation_model.obstacle_height_m,
    )
    actual_pf = {
        field.name: getattr(pf_config, field.name)
        for field in fields(RotatingShieldPFConfig)
    }
    session_hash_payload: dict[str, object] = {
        "measurement_runtime_config_sha256": runtime_config_sha256,
        "measurement_log_sha256": pending_log_digest,
        "pf_config": actual_pf,
        "pf_random_seed": session_seed,
    }
    if active_isotopes != isotopes:
        session_hash_payload["active_isotopes"] = list(active_isotopes)
    resolved_session_config_sha256 = sha256_json(session_hash_payload)
    input_config_sha256 = (
        sha256_json(dict(config))
        if config_hash is None
        else _sha256_string(config_hash, location="config_hash")
    )
    estimator = PurePFEstimator(
        isotopes=active_isotopes,
        candidate_isotopes=isotopes,
        surface_diagnostic_points=surface_diagnostic_points,
        shield_normals=generate_octant_orientations(),
        mu_by_isotope=observation_model.mu_by_isotope,
        pf_config=pf_config,
        obstacle_grid=pf_obstacle_grid,
        obstacle_height_m=observation_model.obstacle_height_m,
        obstacle_mu_by_isotope=observation_model.obstacle_mu_by_isotope,
        obstacle_buildup_coeff=(
            observation_model.obstacle_buildup_coeff if obstacle_enabled else 0.0
        ),
        detector_radius_m=observation_model.detector_geometry.count_radius_m,
        detector_aperture_radius_m=(
            observation_model.detector_geometry.aperture_radius_m
        ),
        detector_aperture_samples=(
            observation_model.detector_geometry.aperture_samples
        ),
        detector_aperture_sampling=(
            observation_model.detector_geometry.aperture_sampling
        ),
        source_extent_radius_m=observation_model.source_extent_radius_m,
        source_extent_samples=observation_model.source_extent_samples,
        line_mu_by_isotope=observation_model.line_mu_by_isotope,
        full_spectrum_generative_model=forward.spectral_model,
        measurement_log_schema_version=schema_version,
        config_hash=input_config_sha256,
        resolved_config_hash=resolved_session_config_sha256,
        measurement_log_sha256=pending_log_digest,
        random_seed=session_seed,
    )
    environment = forward.environment
    assert environment.detector_position is not None
    initial_pose = np.asarray(environment.detector_position, dtype=np.float64)
    estimator.add_measurement_pose(initial_pose, reset_filters=False)
    return estimator


def build_live_estimator(
    context: RunContext,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    runtime_root: str | Path,
    config_hash: str | None = None,
    inference_isotopes: Sequence[str] | None = None,
) -> PurePFEstimator:
    """Construct a PF from a truth-free live runtime handshake."""
    root = Path(runtime_root).expanduser().resolve()
    try:
        forward = ResolvedForwardContext.from_run_context(context, run_root=root)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            f"Cannot resolve the live runtime-authenticated forward context: {exc}"
        ) from exc
    return _build_live_estimator_from_forward_context(
        forward,
        config,
        profile=profile,
        seed=seed,
        measurement_log_schema_version=context.schema_version,
        measurement_runtime_config_sha256=context.runtime_config_sha256,
        config_hash=config_hash,
        inference_isotopes=inference_isotopes,
    )


def _validate_published_forward_context(log: MeasurementLog) -> None:
    """Validate the runtime-authenticated context of the published live log."""
    try:
        ResolvedForwardContext.from_log(log)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFLiveSessionError(
            f"Cannot validate the published runtime forward context: {exc}"
        ) from exc


def bind_published_measurement_log(
    estimator: PurePFEstimator,
    log: MeasurementLog,
    *,
    live_records: Sequence[MeasurementLogRecord],
) -> None:
    """Bind a live PF to the immutable log produced by its own session."""
    logged_isotopes = tuple(log.run_manifest["isotopes"])
    candidate_isotopes = tuple(
        getattr(estimator, "candidate_isotopes", estimator.joint_isotope_order())
    )
    if candidate_isotopes != logged_isotopes:
        raise PFLiveSessionError(
            "Published MeasurementLog isotopes disagree with live PF candidates."
        )
    active_isotope_set = set(estimator.joint_isotope_order())
    if not active_isotope_set.issubset(logged_isotopes):
        raise PFLiveSessionError("Live PF active isotopes are not runtime candidates.")
    active_isotopes = tuple(
        isotope for isotope in logged_isotopes if isotope in active_isotope_set
    )
    if len(estimator.measurements) != len(log.records):
        raise PFLiveSessionError(
            "Published MeasurementLog record count disagrees with the live PF."
        )
    if len(live_records) != len(log.records):
        raise PFLiveSessionError(
            "Published MeasurementLog record count disagrees with live ingestion."
        )
    try:
        live_records_digest = measurement_records_digest(tuple(live_records))
        published_records_digest = measurement_records_digest(log.records)
    except (TypeError, ValueError) as exc:
        raise PFLiveSessionError(
            "Cannot authenticate the ordered live MeasurementLog records."
        ) from exc
    if live_records_digest != published_records_digest:
        raise PFLiveSessionError(
            "Published MeasurementLog records differ from the ordered live records."
        )
    _validate_published_forward_context(log)
    actual_pf = {
        field.name: getattr(estimator.pf_config, field.name)
        for field in fields(RotatingShieldPFConfig)
    }
    digest = log.log_sha256
    estimator.measurement_log_sha256 = digest
    session_hash_payload: dict[str, object] = {
        "measurement_runtime_config_sha256": log.resolved_config_sha256,
        "measurement_log_sha256": digest,
        "pf_config": actual_pf,
        "pf_random_seed": int(estimator.random_seed),
    }
    if active_isotopes != logged_isotopes:
        session_hash_payload["active_isotopes"] = list(active_isotopes)
    estimator.resolved_config_hash = sha256_json(session_hash_payload)


def measurement_record_to_station_input(
    record: MeasurementLogRecord,
) -> tuple[object, ...]:
    """Translate one live runtime record into the PF station contract."""
    spectrum = np.asarray(record.spectrum_counts)
    if spectrum.ndim != 1 or spectrum.dtype != np.int64 or np.any(spectrum < 0):
        raise PFLiveSessionError(
            "Live observations must contain raw nonnegative int64 spectra."
        )
    fe_index = _json_integer(
        record.fe_orientation_index,
        location="record.fe_orientation_index",
        minimum=0,
    )
    pb_index = _json_integer(
        record.pb_orientation_index,
        location="record.pb_orientation_index",
        minimum=0,
    )
    if fe_index > 7 or pb_index > 7:
        raise PFLiveSessionError("Live Fe/Pb orientation indices must lie in 0..7.")
    live_time_s = _finite_real(record.live_time_s, location="record.live_time_s")
    if live_time_s <= 0.0:
        raise PFLiveSessionError("record.live_time_s must be positive.")
    return (
        np.ascontiguousarray(spectrum),
        fe_index,
        pb_index,
        live_time_s,
    )


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    location: str,
) -> NDArray[Any]:
    """Return one exact-dtype immutable contiguous array copy."""
    array = np.asarray(value)
    if array.dtype != dtype:
        raise PFLiveSessionError(f"{location} must have exact dtype {dtype}.")
    copied = np.array(array, dtype=dtype, copy=True, order="C")
    immutable = np.frombuffer(copied.tobytes(order="C"), dtype=dtype).reshape(
        copied.shape
    )
    return immutable


def _readonly_isotope_arrays(
    values: Mapping[str, object],
    *,
    isotope_order: tuple[str, ...],
    dtype: np.dtype[Any],
    location: str,
) -> Mapping[str, NDArray[Any]]:
    """Copy an exact isotope-keyed array mapping into immutable storage."""
    if not isinstance(values, Mapping) or tuple(values) != isotope_order:
        raise PFLiveSessionError(
            f"{location} keys must exactly match the PF isotope order."
        )
    copied = {
        isotope: _readonly_array(
            values[isotope],
            dtype=dtype,
            location=f"{location}.{isotope}",
        )
        for isotope in isotope_order
    }
    return MappingProxyType(copied)


def _immutable_particle_snapshot(
    particles: JointPlanningParticles,
    *,
    source_run_id: str,
    record_count: int,
    station_count: int,
    covered_records_digest: DigestIdentity,
    posterior_summary_json: bytes,
) -> PFLiveParticleSnapshot:
    """Copy one estimator particle view into a read-only live DTO."""
    if not isinstance(particles, JointPlanningParticles):
        raise PFLiveSessionError(
            "PF planning_joint_particles() returned an incompatible contract."
        )
    isotope_order = tuple(particles.isotope_order)
    if not isotope_order or len(set(isotope_order)) != len(isotope_order):
        raise PFLiveSessionError("PF planning isotope order must be unique and nonempty.")
    weights = _readonly_array(
        particles.weights_n,
        dtype=np.dtype(np.float64),
        location="planning.weights_n",
    )
    indices = _readonly_array(
        particles.original_particle_indices,
        dtype=np.dtype(np.int64),
        location="planning.original_particle_indices",
    )
    if weights.ndim != 1 or indices.shape != weights.shape or weights.size == 0:
        raise PFLiveSessionError(
            "PF planning weights and particle indices must be aligned vectors."
        )
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12)
        or np.any(indices < 0)
    ):
        raise PFLiveSessionError(
            "PF planning weights/indices must describe a normalized generation."
        )
    positions = _readonly_isotope_arrays(
        particles.positions_nk3_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.positions_nk3_by_isotope",
    )
    chart_ids = _readonly_isotope_arrays(
        particles.surface_chart_ids_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.int64),
        location="planning.surface_chart_ids_nk_by_isotope",
    )
    surface_uv = _readonly_isotope_arrays(
        particles.surface_uv_nk2_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.surface_uv_nk2_by_isotope",
    )
    strengths = _readonly_isotope_arrays(
        particles.strengths_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.float64),
        location="planning.strengths_nk_by_isotope",
    )
    masks = _readonly_isotope_arrays(
        particles.source_mask_nk_by_isotope,
        isotope_order=isotope_order,
        dtype=np.dtype(np.bool_),
        location="planning.source_mask_nk_by_isotope",
    )
    for isotope in isotope_order:
        position = positions[isotope]
        chart = chart_ids[isotope]
        uv = surface_uv[isotope]
        strength = strengths[isotope]
        mask = masks[isotope]
        if (
            position.ndim != 3
            or position.shape[0] != weights.size
            or position.shape[2] != 3
            or chart.shape != position.shape[:2]
            or uv.shape != (*position.shape[:2], 2)
            or strength.shape != position.shape[:2]
            or mask.shape != position.shape[:2]
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(uv))
            or np.any(~np.isfinite(strength))
            or np.any(strength < 0.0)
        ):
            raise PFLiveSessionError(
                f"PF planning particle arrays are misaligned for {isotope}."
            )
    return PFLiveParticleSnapshot(
        source_run_id=source_run_id,
        record_count=record_count,
        station_count=station_count,
        covered_records_digest=covered_records_digest,
        isotope_order=isotope_order,
        weights_n=weights,
        positions_nk3_by_isotope=positions,
        surface_chart_ids_nk_by_isotope=chart_ids,
        surface_uv_nk2_by_isotope=surface_uv,
        strengths_nk_by_isotope=strengths,
        source_mask_nk_by_isotope=masks,
        original_particle_indices=indices,
        posterior_summary_json=posterior_summary_json,
        posterior_summary_sha256=sha256(posterior_summary_json).hexdigest(),
    )


def live_posterior_summary(estimator: object) -> dict[str, object]:
    """Return a truth-free, explicitly non-publishable station summary."""
    method = getattr(estimator, "posterior_point_estimate", None)
    if not callable(method):
        raise PFLiveSessionError(
            "PF estimator does not expose posterior_point_estimate()."
        )
    raw = method()
    if not isinstance(raw, Mapping):
        raise PFLiveSessionError("PF live point estimates must be an isotope mapping.")
    isotopes: dict[str, object] = {}
    for isotope, estimate in raw.items():
        to_dict = getattr(estimate, "to_dict", None)
        if not callable(to_dict):
            raise PFLiveSessionError(
                "Every PF live point estimate must be serializable."
            )
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise PFLiveSessionError(
                "Every PF live point estimate must serialize an object."
            )
        isotopes[str(isotope)] = dict(payload)
    return {
        "schema_version": 1,
        "publishable": False,
        "provenance_status": "awaiting_finalized_measurement_log_digest",
        "isotopes": isotopes,
    }


def register_persisted_station_pose(
    estimator: PurePFEstimator,
    records: Sequence[MeasurementLogRecord],
    *,
    station_id: int,
) -> int:
    """Register one canonical single-pose station and return its pose index."""
    rows = tuple(records)
    if not rows:
        raise PFLiveSessionError("A PF station must contain at least one record.")
    if any(not isinstance(record, MeasurementLogRecord) for record in rows):
        raise PFLiveSessionError(
            "PF station ingestion requires MeasurementLogRecord values."
        )
    poses = np.asarray([record.detector_pose_xyz for record in rows], dtype=np.float64)
    quaternions = np.asarray(
        [record.detector_quat_wxyz for record in rows],
        dtype=np.float64,
    )
    if not np.all(poses == poses[0]) or not np.all(quaternions == quaternions[0]):
        raise PFLiveSessionError(
            "Every persisted PF station view must share one detector pose."
        )
    pose = poses[0]
    if station_id == 0 and not estimator.measurements and len(estimator.poses) == 1:
        estimator.poses[0] = pose.copy()
        estimator.kernel_cache = None
        return 0
    estimator.add_measurement_pose(pose, reset_filters=False)
    return len(estimator.poses) - 1


def assimilate_persisted_station(
    estimator: PurePFEstimator,
    records: Sequence[MeasurementLogRecord],
    *,
    station_id: int,
    generative_contract_hash_sha256: str,
) -> None:
    """Assimilate one canonical single-pose station through the PF spectrum path."""
    rows = tuple(records)
    if any(record.station_id != station_id for record in rows):
        raise PFLiveSessionError(
            "Persisted PF station IDs must match the requested station."
        )
    if any(
        record.metadata.get("station_complete") is True for record in rows[:-1]
    ) or (rows and rows[-1].metadata.get("station_complete") is not True):
        raise PFLiveSessionError(
            "PF assimilation requires one final durable station marker."
        )
    pose_index = register_persisted_station_pose(
        estimator,
        rows,
        station_id=station_id,
    )
    estimator.update_spectrum_station(
        tuple(measurement_record_to_station_input(record) for record in rows),
        pose_idx=int(pose_index),
        generative_contract_hash_sha256=generative_contract_hash_sha256,
    )


class PFLiveSession:
    """Own one causal PF estimator from live handshake through final binding."""

    def __init__(
        self,
        context: RunContext,
        config: Mapping[str, Any],
        *,
        profile: str,
        seed: int,
        runtime_root: str | Path,
        config_hash: str | None = None,
        inference_isotopes: Sequence[str] | None = None,
    ) -> None:
        """Authenticate the runtime context and construct one live PF."""
        if not isinstance(context, RunContext):
            raise PFLiveSessionError("context must be a runtime RunContext.")
        contract_hash = _sha256_string(
            context.runtime_config.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY),
            location=(
                "context.runtime_config."
                f"{FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY}"
            ),
        )
        estimator = build_live_estimator(
            context,
            config,
            profile=profile,
            seed=seed,
            runtime_root=runtime_root,
            config_hash=config_hash,
            inference_isotopes=inference_isotopes,
        )
        estimator_contract = getattr(
            getattr(estimator, "full_spectrum_generative_model", None),
            "contract_hash_sha256",
            None,
        )
        if estimator_contract != contract_hash:
            raise PFLiveSessionError(
                "Live PF generative model differs from its runtime handshake."
            )
        self._context = context
        self._estimator = estimator
        self._generative_contract_hash_sha256 = contract_hash
        self._records: list[MeasurementLogRecord] = []
        self._station_count = 0
        self._phase = "receiving"
        self._completed_state: PFCompletedLiveState | None = None
        self._bound_state: PFBoundLiveState | None = None

    @property
    def context(self) -> RunContext:
        """Return the immutable truth-free runtime handshake."""
        return self._context

    @property
    def records(self) -> tuple[MeasurementLogRecord, ...]:
        """Return the exact ordered records received from the runtime."""
        return tuple(self._records)

    @property
    def record_count(self) -> int:
        """Return the count of durably delivered runtime records."""
        return len(self._records)

    @property
    def station_count(self) -> int:
        """Return the count of stations already assimilated exactly once."""
        return self._station_count

    @property
    def phase(self) -> str:
        """Return the current receiving, completed, bound, or failed phase."""
        return self._phase

    def _ensure_receiving(self) -> None:
        """Reject observation delivery after completion or a failed update."""
        if self._phase != "receiving":
            raise PFLiveSessionError(
                f"PF live session cannot receive records while {self._phase}."
            )

    def _validated_view(
        self,
        records: Sequence[MeasurementLogRecord],
    ) -> MeasurementLogView:
        """Validate one exact live prefix through the shared runtime view."""
        try:
            view = MeasurementLogView.from_records(self._context, tuple(records))
            view.station_view()
        except (TypeError, ValueError, MeasurementLogValidationError) as exc:
            raise PFLiveSessionError(
                f"Persisted PF records violate the runtime contract: {exc}"
            ) from exc
        for index, record in enumerate(view.records):
            if (
                record.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY)
                != self._generative_contract_hash_sha256
            ):
                raise PFLiveSessionError(
                    "Persisted PF record generative contract differs from the "
                    f"runtime handshake at row {index}."
                )
        return view

    def receive_persisted(self, record: MeasurementLogRecord) -> bool:
        """Receive one durable record and assimilate only at its station marker."""
        self._ensure_receiving()
        if not isinstance(record, MeasurementLogRecord):
            raise PFLiveSessionError(
                "receive_persisted requires a MeasurementLogRecord."
            )
        if self._records and (
            record.station_id != self._records[-1].station_id
            and self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "A new PF station cannot begin before the prior marker is durable."
            )
        prospective = (*self._records, record)
        view = self._validated_view(prospective)
        self._records.append(record)
        if record.metadata.get("station_complete") is not True:
            return False
        station = view.station_view().stations[-1]
        if station.station_id != self._station_count or not station.marked_complete:
            self._phase = "failed"
            raise PFLiveSessionError(
                "Persisted PF station sequence differs from completed assimilation."
            )
        try:
            assimilate_persisted_station(
                self._estimator,
                station.records,
                station_id=station.station_id,
                generative_contract_hash_sha256=(
                    self._generative_contract_hash_sha256
                ),
            )
        except BaseException:
            self._phase = "failed"
            raise
        self._station_count += 1
        if len(self._estimator.measurements) != len(self._records):
            self._phase = "failed"
            raise PFLiveSessionError(
                "PF estimator measurement history differs from live ingestion."
            )
        return True

    def receive_persisted_station(
        self,
        records: Sequence[MeasurementLogRecord],
    ) -> None:
        """Receive one complete durable station through the record API."""
        self._ensure_receiving()
        rows = tuple(records)
        if not rows:
            raise PFLiveSessionError("A persisted PF station cannot be empty.")
        if any(not isinstance(record, MeasurementLogRecord) for record in rows):
            raise PFLiveSessionError(
                "receive_persisted_station requires MeasurementLogRecord values."
            )
        if self._records and self._records[-1].metadata.get("station_complete") is not True:
            raise PFLiveSessionError(
                "receive_persisted_station cannot continue a partially buffered station."
            )
        if len({record.station_id for record in rows}) != 1:
            raise PFLiveSessionError(
                "receive_persisted_station accepts exactly one station."
            )
        if any(
            record.metadata.get("station_complete") is True for record in rows[:-1]
        ) or rows[-1].metadata.get("station_complete") is not True:
            raise PFLiveSessionError(
                "A persisted station requires one final station_complete marker."
            )
        self._validated_view((*self._records, *rows))
        for record in rows:
            self.receive_persisted(record)

    def planning_particle_snapshot(
        self,
        *,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> PFLiveParticleSnapshot:
        """Return a copied truth-free particle generation at a station boundary."""
        if self._phase != "receiving":
            raise PFLiveSessionError(
                f"PF live session cannot plan while {self._phase}."
            )
        if not self._records or (
            self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "PF planning requires the latest durable station boundary."
            )
        particles = self._estimator.planning_joint_particles(
            max_particles=max_particles,
            method=method,
            rng=rng,
        )
        summary_json = canonical_json_bytes(
            live_posterior_summary(self._estimator)
        )
        return _immutable_particle_snapshot(
            particles,
            source_run_id=self._context.run_id,
            record_count=len(self._records),
            station_count=self._station_count,
            covered_records_digest=measurement_records_digest(self._records),
            posterior_summary_json=summary_json,
        )

    def complete_live_state(self) -> PFCompletedLiveState:
        """Seal the already-assimilated live state before log publication."""
        if self._phase in {"completed", "bound"}:
            assert self._completed_state is not None
            return self._completed_state
        self._ensure_receiving()
        if not self._records:
            raise PFLiveSessionError("A PF live session cannot complete without records.")
        view = self._validated_view(self._records)
        stations = view.station_view()
        if stations.complete_station_count != stations.station_count or (
            self._records[-1].metadata.get("station_complete") is not True
        ):
            raise PFLiveSessionError(
                "PF live completion requires every station marker to be durable."
            )
        if self._station_count != stations.station_count or (
            len(self._estimator.measurements) != len(self._records)
        ):
            raise PFLiveSessionError(
                "PF live completion differs from its assimilated station history."
            )
        raw_checkpoint = self._estimator.serialized_state()
        if not isinstance(raw_checkpoint, (bytes, bytearray, memoryview)):
            raise PFLiveSessionError("PF serialized_state() must return bytes.")
        checkpoint = bytes(raw_checkpoint)
        digest = measurement_records_digest(self._records)
        completed = PFCompletedLiveState(
            source_run_id=self._context.run_id,
            runtime_config_sha256=self._context.runtime_config_sha256,
            generative_contract_hash_sha256=(
                self._generative_contract_hash_sha256
            ),
            record_count=len(self._records),
            station_count=self._station_count,
            covered_step_ids=tuple(record.step_id for record in self._records),
            covered_records_digest=digest,
            checkpoint_state=checkpoint,
            checkpoint_sha256=sha256(checkpoint).hexdigest(),
        )
        self._completed_state = completed
        self._phase = "completed"
        return completed

    def bind_published_log(self, log: MeasurementLog) -> PFBoundLiveState:
        """Bind the sealed PF to its exact published log without assimilation."""
        if not isinstance(log, MeasurementLog) or log.path is None:
            raise PFLiveSessionError(
                "bind_published_log requires a published MeasurementLog."
            )
        if self._phase == "bound":
            assert self._bound_state is not None
            if self._bound_state.measurement_log_sha256 != log.log_sha256:
                raise PFLiveSessionError(
                    "PF live state is already bound to another MeasurementLog."
                )
            return self._bound_state
        if self._phase != "completed" or self._completed_state is None:
            raise PFLiveSessionError(
                "complete_live_state() must seal PF inference before log binding."
            )
        if log.run_id != self._context.run_id:
            raise PFLiveSessionError(
                "Published MeasurementLog belongs to another runtime run."
            )
        if log.context.to_payload() != self._context.to_payload():
            raise PFLiveSessionError(
                "Published MeasurementLog context differs from the live handshake."
            )
        published_digest = measurement_records_digest(log.records)
        if published_digest != self._completed_state.covered_records_digest:
            raise PFLiveSessionError(
                "Published MeasurementLog records differ from the completed PF state."
            )
        before_bind = self._completed_state.checkpoint_state
        bind_published_measurement_log(
            self._estimator,
            log,
            live_records=self._records,
        )
        raw_after_bind = self._estimator.serialized_state()
        if not isinstance(raw_after_bind, (bytes, bytearray, memoryview)):
            self._phase = "failed"
            raise PFLiveSessionError("PF serialized_state() must return bytes.")
        after_bind = bytes(raw_after_bind)
        if after_bind != before_bind:
            self._phase = "failed"
            raise PFLiveSessionError(
                "Published-log binding changed the completed PF posterior state."
            )
        snapshot = self._estimator.posterior_snapshot()
        to_dict = getattr(snapshot, "to_dict", None)
        if not callable(to_dict):
            raise PFLiveSessionError("PF posterior snapshot is not serializable.")
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise PFLiveSessionError("PF posterior snapshot must serialize an object.")
        log_digest = log.log_sha256
        if payload.get("measurement_log_sha256") != log_digest or (
            payload.get("record_count") != len(self._records)
        ):
            raise PFLiveSessionError(
                "Bound PF posterior identity differs from the published log."
            )
        posterior_json = canonical_json_bytes(dict(payload))
        bound = PFBoundLiveState(
            completed=self._completed_state,
            measurement_log_sha256=log_digest,
            posterior_json=posterior_json,
            posterior_sha256=sha256(posterior_json).hexdigest(),
        )
        self._bound_state = bound
        self._phase = "bound"
        return bound

    def publication_input(self) -> PFBoundLiveState:
        """Return bound result/checkpoint bytes without rerunning PF inference."""
        if self._phase != "bound" or self._bound_state is None:
            raise PFLiveSessionError(
                "PF publication input requires an exactly bound published log."
            )
        return self._bound_state


__all__ = [
    "PFBoundLiveState",
    "PFCompletedLiveState",
    "PFLiveSessionError",
    "PFLiveParticleSnapshot",
    "PFLiveSession",
    "assimilate_persisted_station",
    "bind_published_measurement_log",
    "build_live_estimator",
    "live_posterior_summary",
    "measurement_record_to_station_input",
    "register_persisted_station_pose",
]
