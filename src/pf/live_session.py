"""PF construction and record ingestion for one live runtime session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.surface_atlas import ContinuousSurfaceAtlas
from measurement.surface_charts import build_surface_chart_geometry
from numpy.typing import NDArray
from runtime.forward_context import ResolvedForwardContext
from runtime.measurement_log import MeasurementLog, MeasurementLogRecord
from runtime.prefix import measurement_records_digest
from runtime.records import RunContext

from pf.gpu_utils import preflight_compute_backend
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import sha256_json
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig


class PFLiveSessionError(RuntimeError):
    """Report an incompatible live context, setting, or observation."""


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


__all__ = [
    "PFLiveSessionError",
    "bind_published_measurement_log",
    "build_live_estimator",
    "measurement_record_to_station_input",
]
