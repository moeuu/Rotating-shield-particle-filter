"""Sequentially replay a MeasurementLog through the pure particle filter."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.surface_charts import (
    build_surface_chart_geometry,
)
from pf.gpu_utils import preflight_compute_backend
from pf.isotope_gate import FullSpectrumIsotopeGate
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig
from measurement.surface_atlas import ContinuousSurfaceAtlas
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogArrayView,
    MeasurementLogRecord,
    MeasurementLogStationView,
    MeasurementLogValidationError,
    load_measurement_log,
)
from runtime.forward_context import ResolvedForwardContext
from runtime.artifacts import AtomicBundlePublisher
from runtime.records import RunContext
from spectrum.transport_spectral import GeometryConditionedSpectralModel


class PFReplayError(RuntimeError):
    """Report an incompatible log, configuration, or replay observation."""


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
    "pf_strength_prior_gamma_scale_cps_1m": ("strength_prior_gamma_scale_cps_1m"),
}

_PF_REPLAY_PHYSICAL_OVERRIDE_KEYS = frozenset(
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


def _sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_string(value: object, *, location: str) -> str:
    """Return one exact lowercase SHA-256 string without coercion."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_PATTERN for character in value)
    ):
        raise PFReplayError(
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
        raise PFReplayError(f"{location} must be a JSON integer.")
    if minimum is not None and value < minimum:
        raise PFReplayError(f"{location} must be at least {minimum}.")
    return value


def _finite_real(value: object, *, location: str) -> float:
    """Return one exact finite real without boolean or string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PFReplayError(f"{location} must be a finite JSON number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise PFReplayError(f"{location} must be a finite JSON number.")
    return parsed


def _finite_vector(
    value: object,
    *,
    length: int,
    location: str,
) -> NDArray[np.float64]:
    """Return one exact-length finite numeric vector without reshaping."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise PFReplayError(
            f"{location} must contain exactly {length} finite numbers."
        ) from exc
    if raw.shape != (length,):
        raise PFReplayError(f"{location} must have shape ({length},); got {raw.shape}.")
    return np.asarray(
        [
            _finite_real(item, location=f"{location}[{index}]")
            for index, item in enumerate(raw)
        ],
        dtype=np.float64,
    )


def _parse_replay_config_json(text: str, *, location: str) -> dict[str, Any]:
    """Parse one strict replay-config JSON object without duplicate keys."""

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build an object only when every JSON member name is unique."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PFReplayError(f"{location} contains duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def _reject_constant(value: str) -> None:
        """Reject Python's non-standard NaN and infinity JSON extensions."""
        raise PFReplayError(f"{location} contains non-finite JSON constant {value!r}.")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PFReplayError(f"Cannot parse replay config {location}.") from exc
    if not isinstance(payload, dict):
        raise PFReplayError(f"Replay config {location} must contain an object.")
    return payload


def _load_replay_runtime_config(
    path: Path,
    *,
    seen: set[Path],
) -> dict[str, Any]:
    """Load strict replay configuration inheritance without lossy JSON parsing."""
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise PFReplayError(f"Cyclic replay config inheritance at {resolved_path}.")
    seen.add(resolved_path)
    try:
        text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PFReplayError(f"Cannot read replay config {resolved_path}.") from exc
    data = _parse_replay_config_json(text, location=str(resolved_path))
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str) or not parent_ref:
        raise PFReplayError("Replay config extends must be a nonempty string.")
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    parent = _load_replay_runtime_config(parent_path, seen=seen)
    return {**parent, **data}


def load_pf_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one inherited PF configuration and return its source digest."""
    config_path = Path(path).expanduser().resolve()
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise PFReplayError(f"Cannot read PF config {config_path}.") from exc
    return (
        _load_replay_runtime_config(config_path, seen=set()),
        _sha256_bytes(config_bytes),
    )


def _resolved_forward_context(log: MeasurementLog) -> ResolvedForwardContext:
    """Resolve one runtime-authenticated physical context for PF consumption."""
    try:
        return ResolvedForwardContext.from_log(log)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFReplayError(
            "Cannot resolve the runtime-authenticated forward context: "
            f"{exc}"
        ) from exc


def validate_local_forward_model(log: MeasurementLog) -> None:
    """Fail closed unless runtime authenticates the complete physical context."""
    _resolved_forward_context(log)


def validate_local_full_spectrum_contract(
    log: MeasurementLog,
) -> GeometryConditionedSpectralModel:
    """Return the runtime-authenticated full-spectrum observation model."""
    return _resolved_forward_context(log).spectral_model


def _surface_atlas_replay_inputs(
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
        raise PFReplayError(
            "Cannot reconstruct the logged continuous surface atlas."
        ) from exc
    quantiles = (np.arange(point_count, dtype=np.float64) + 0.5) / float(point_count)
    chart_ids = np.searchsorted(
        np.cumsum(atlas.chart_probabilities),
        quantiles,
        side="right",
    ).astype(np.int64)
    if np.any(chart_ids < 0) or np.any(chart_ids >= atlas.chart_count):
        raise PFReplayError("Surface-atlas diagnostic chart IDs are invalid.")
    sequence = np.arange(point_count, dtype=np.float64) + 0.5
    uv = np.column_stack(
        (
            np.mod(sequence * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0),
            np.mod(sequence * (np.sqrt(2.0) - 1.0), 1.0),
        )
    )
    points = np.ascontiguousarray(atlas.positions_xyz(chart_ids, uv))
    return points


def _pf_config_values(
    config: Mapping[str, Any],
    *,
    profile: str,
    upper: NDArray[np.float64],
) -> dict[str, Any]:
    """Select declared PF dataclass fields from a resolved configuration."""
    allowed = {field.name for field in fields(RotatingShieldPFConfig)}
    values = {key: value for key, value in config.items() if key in allowed}
    values["estimator_profile"] = str(profile)
    values["position_max"] = tuple(float(value) for value in upper)
    return values


def _external_replay_config(
    external_config: Mapping[str, Any],
    *,
    upper: NDArray[np.float64],
) -> dict[str, Any]:
    """Resolve PF-owned replay inputs without reading estimator state from the log."""
    forbidden_physics = sorted(
        key for key in external_config if key in _PF_REPLAY_PHYSICAL_OVERRIDE_KEYS
    )
    if forbidden_physics:
        raise PFReplayError(
            "External runtime field cannot override MeasurementLog physics: "
            + ", ".join(forbidden_physics)
        )
    if external_config:
        try:
            enforce_pure_runtime_settings(external_config)
        except ValueError as exc:
            raise PFReplayError(
                "External replay configuration violates the pure-PF schema."
            ) from exc
    normalized = dict(external_config)
    for alias, canonical in _PF_CONFIG_ALIASES.items():
        if alias not in normalized:
            continue
        if canonical in normalized and normalized[canonical] != normalized[alias]:
            raise PFReplayError(f"Conflicting PF settings {alias!r} and {canonical!r}.")
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


def _build_estimator_from_forward_context(
    forward: ResolvedForwardContext,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    measurement_log_schema_version: int,
    measurement_runtime_config_sha256: str,
    measurement_log_digest: str,
    config_hash: str | None = None,
    resolved_config_hash: str | None = None,
    inference_isotopes: Sequence[str] | None = None,
) -> PurePFEstimator:
    """Build one PF from an authenticated shared physical context."""
    if not isinstance(config, Mapping):
        raise PFReplayError("Replay configuration must be an object.")
    if any(not isinstance(key, str) for key in config):
        raise PFReplayError("Replay configuration keys must be JSON strings.")
    if not isinstance(profile, str):
        raise PFReplayError("Replay profile must be a JSON string.")
    replay_seed = _json_integer(seed, location="seed", minimum=0)
    schema_version = _json_integer(
        measurement_log_schema_version,
        location="measurement_log_schema_version",
        minimum=1,
    )
    logged_config_sha256 = _sha256_string(
        measurement_runtime_config_sha256,
        location="measurement_runtime_config_sha256",
    )
    if not isinstance(measurement_log_digest, str) or not measurement_log_digest:
        raise PFReplayError("measurement_log_digest must be a nonempty string.")
    full_spectrum_model = forward.spectral_model
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
        raise PFReplayError(
            "Inference isotopes must be a unique nonempty subset of the logged "
            "candidate isotopes."
        )
    active_isotopes = tuple(
        isotope for isotope in isotopes if isotope in set(active_isotopes)
    )
    _, upper = forward.bounds_xyz
    replay_config = _external_replay_config(
        config,
        upper=upper,
    )
    try:
        pure_config = enforce_pure_runtime_settings(
            replay_config,
            profile=profile,
        )
        pf_config = RotatingShieldPFConfig(
            **_pf_config_values(
                pure_config,
                profile=profile,
                upper=upper,
            )
        )
        apply_profile_to_config(pf_config)
        preflight_compute_backend(
            use_gpu=bool(pf_config.use_gpu),
            gpu_device=str(pf_config.gpu_device),
            gpu_dtype=str(pf_config.gpu_dtype),
        )
    except (TypeError, ValueError) as exc:
        raise PFReplayError(
            "External PF replay configuration is incompatible."
        ) from exc
    observation_model = forward.observation_model
    obstacle_grid = forward.obstacle_grid
    obstacle_enabled = forward.obstacle_attenuation_enabled
    if obstacle_grid is not None and not obstacle_enabled:
        raise PFReplayError(
            "A logged obstacle grid requires physical obstacle attenuation."
        )
    pf_obstacle_grid = obstacle_grid if obstacle_enabled else None
    surface_diagnostic_points = _surface_atlas_replay_inputs(
        forward.environment,
        pf_config=pf_config,
        obstacle_grid=pf_obstacle_grid,
        obstacle_height_m=observation_model.obstacle_height_m,
    )
    actual_pf = {
        field.name: getattr(pf_config, field.name)
        for field in fields(RotatingShieldPFConfig)
    }
    replay_hash_payload: dict[str, object] = {
        "measurement_runtime_config_sha256": logged_config_sha256,
        "measurement_log_sha256": measurement_log_digest,
        "pf_config": actual_pf,
        "pf_random_seed": replay_seed,
    }
    if active_isotopes != isotopes:
        replay_hash_payload["active_isotopes"] = list(active_isotopes)
    computed_replay_config_sha256 = sha256_json(replay_hash_payload)
    if resolved_config_hash is not None:
        supplied_resolved_hash = _sha256_string(
            resolved_config_hash,
            location="resolved_config_hash",
        )
        if supplied_resolved_hash != computed_replay_config_sha256:
            raise PFReplayError(
                "Supplied resolved_config_hash does not bind the effective "
                "replay configuration."
            )
    replay_config_sha256 = computed_replay_config_sha256
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
        full_spectrum_generative_model=full_spectrum_model,
        measurement_log_schema_version=schema_version,
        config_hash=input_config_sha256,
        resolved_config_hash=replay_config_sha256,
        measurement_log_sha256=measurement_log_digest,
        random_seed=replay_seed,
    )
    environment = forward.environment
    assert environment.detector_position is not None
    initial_pose = np.asarray(environment.detector_position, dtype=np.float64)
    estimator.add_measurement_pose(initial_pose, reset_filters=False)
    return estimator


def build_replay_estimator(
    log: MeasurementLog,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    config_hash: str | None = None,
    resolved_config_hash: str | None = None,
    measurement_log_digest: str | None = None,
    inference_isotopes: Sequence[str] | None = None,
) -> PurePFEstimator:
    """Construct a locally authenticated pure estimator for replay."""
    forward = _resolved_forward_context(log)
    return _build_estimator_from_forward_context(
        forward,
        config,
        profile=profile,
        seed=seed,
        measurement_log_schema_version=log.schema_version,
        measurement_runtime_config_sha256=log.resolved_config_sha256,
        measurement_log_digest=(
            log.log_sha256
            if measurement_log_digest is None
            else measurement_log_digest
        ),
        config_hash=config_hash,
        resolved_config_hash=resolved_config_hash,
        inference_isotopes=inference_isotopes,
    )


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
    """Construct a PF from a truth-free live runtime handshake.

    The final MeasurementLog digest is unavailable during acquisition. The
    caller must bind the published digest before serializing a posterior.
    """
    root = Path(runtime_root).expanduser().resolve()
    try:
        forward = ResolvedForwardContext.from_run_context(
            context,
            run_root=root,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFReplayError(
            "Cannot resolve the live runtime-authenticated forward context: "
            f"{exc}"
        ) from exc
    return _build_estimator_from_forward_context(
        forward,
        config,
        profile=profile,
        seed=seed,
        measurement_log_schema_version=context.schema_version,
        measurement_runtime_config_sha256=context.runtime_config_sha256,
        measurement_log_digest="unavailable",
        config_hash=config_hash,
        inference_isotopes=inference_isotopes,
    )


def bind_finalized_measurement_log(
    estimator: PurePFEstimator,
    log: MeasurementLog,
) -> None:
    """Bind a live PF to the immutable log digest it actually assimilated."""
    logged_isotopes = tuple(log.run_manifest["isotopes"])
    candidate_isotopes = tuple(
        getattr(estimator, "candidate_isotopes", estimator.joint_isotope_order())
    )
    if candidate_isotopes != logged_isotopes:
        raise PFReplayError(
            "Final MeasurementLog isotopes disagree with the live PF candidates."
        )
    active_isotope_set = set(estimator.joint_isotope_order())
    if not active_isotope_set.issubset(logged_isotopes):
        raise PFReplayError("Live PF active isotopes are not logged candidates.")
    active_isotopes = tuple(
        isotope for isotope in logged_isotopes if isotope in active_isotope_set
    )
    if len(estimator.measurements) != len(log.records):
        raise PFReplayError(
            "Final MeasurementLog record count disagrees with the live PF."
        )
    _resolved_forward_context(log)
    actual_pf = {
        field.name: getattr(estimator.pf_config, field.name)
        for field in fields(RotatingShieldPFConfig)
    }
    digest = log.log_sha256
    estimator.measurement_log_sha256 = digest
    replay_hash_payload: dict[str, object] = {
        "measurement_runtime_config_sha256": log.resolved_config_sha256,
        "measurement_log_sha256": digest,
        "pf_config": actual_pf,
        "pf_random_seed": int(estimator.random_seed),
    }
    if active_isotopes != logged_isotopes:
        replay_hash_payload["active_isotopes"] = list(active_isotopes)
    estimator.resolved_config_hash = sha256_json(replay_hash_payload)


def _station_complete(record: MeasurementLogRecord) -> bool:
    """Return the writer-owned causal station-boundary marker."""
    raw = record.metadata.get("station_complete", False)
    if not isinstance(raw, bool):
        raise PFReplayError("record.metadata.station_complete must be a boolean.")
    return raw


def measurement_record_to_spectrum_input(
    record: MeasurementLogRecord,
) -> tuple[object, ...]:
    """Translate one raw log row into the spectrum-station contract."""
    spectrum = np.asarray(record.spectrum_counts)
    if spectrum.ndim != 1 or spectrum.dtype != np.int64 or np.any(spectrum < 0):
        raise PFReplayError(
            "Replay observations must contain raw nonnegative int64 spectra."
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
        raise PFReplayError("Replay Fe/Pb orientation indices must lie in 0..7.")
    live_time_s = _finite_real(
        record.live_time_s,
        location="record.live_time_s",
    )
    if live_time_s <= 0.0:
        raise PFReplayError("record.live_time_s must be positive.")
    return (
        np.ascontiguousarray(spectrum),
        fe_index,
        pb_index,
        live_time_s,
    )


def detect_replay_isotopes(
    log: MeasurementLog,
    config: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    config_hash: str,
    stop_after: int | None = None,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Detect active isotopes from a truth-free MeasurementLog prefix."""
    limit = (
        len(log.records)
        if stop_after is None
        else _json_integer(
            stop_after,
            location="stop_after",
            minimum=0,
        )
    )
    if limit > len(log.records):
        raise PFReplayError("stop_after exceeds the MeasurementLog record count.")
    detector_config = dict(config)
    detector_config["num_particles"] = 1
    detector = build_replay_estimator(
        log,
        detector_config,
        profile=profile,
        seed=seed,
        config_hash=config_hash,
    )
    gate = FullSpectrumIsotopeGate(
        candidate_isotopes=tuple(log.run_manifest["isotopes"]),
        false_activation_probability=float(
            config.get("detected_isotope_false_activation_probability", 1.0e-3)
        ),
    )
    contract_hash = _sha256_string(
        log.run_manifest.get("full_spectrum_contract_hash_sha256"),
        location="run_manifest.full_spectrum_contract_hash_sha256",
    )
    pending: list[tuple[object, ...]] = []
    pending_pose: NDArray[np.float64] | None = None
    diagnostics: dict[str, object] | None = None
    for record in log.records[:limit]:
        pose = np.asarray(record.detector_pose_xyz, dtype=np.float64)
        if pending_pose is None:
            pending_pose = pose.copy()
        elif not np.array_equal(pending_pose, pose):
            raise PFReplayError(
                "An isotope-detection station contains multiple detector poses."
            )
        pending.append(measurement_record_to_spectrum_input(record))
        if not _station_complete(record):
            continue
        assert pending_pose is not None
        if gate.station_count == 0 and len(detector.poses) == 1:
            detector.poses[0] = pending_pose.copy()
            detector.kernel_cache = None
            pose_index = 0
        else:
            detector.add_measurement_pose(pending_pose, reset_filters=False)
            pose_index = len(detector.poses) - 1
        diagnostics = gate.update(
            detector.full_spectrum_isotope_detection_score_grids(
                tuple(pending),
                pose_idx=pose_index,
                generative_contract_hash_sha256=contract_hash,
            )
        )
        pending = []
        pending_pose = None
    if pending:
        raise PFReplayError(
            "The selected replay prefix ends inside an incomplete station."
        )
    active = tuple(
        isotope
        for isotope in log.run_manifest["isotopes"]
        if isotope in gate.active_isotopes
    )
    if not active or diagnostics is None:
        raise PFReplayError(
            "No candidate isotope crossed the truth-free full-spectrum "
            "activation threshold in the selected replay prefix."
        )
    return active, diagnostics


def _trace_row(
    estimator: PurePFEstimator,
    record: MeasurementLogRecord,
    *,
    record_index: int,
    state_payload: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return one causal posterior row at a record boundary."""
    if state_payload is None:
        state_payload = _trace_state_payload(estimator)
    state_sha256, posterior = state_payload
    return {
        "schema_version": 2,
        "estimator_family": "pure_particle_filter",
        "estimator_variant": estimator.estimator_variant,
        "record_index": int(record_index),
        "step_id": int(record.step_id),
        "action_id": int(record.action_id),
        "station_id": int(record.station_id),
        "station_complete": _station_complete(record),
        "state_sha256": state_sha256,
        "posterior": copy.deepcopy(posterior),
    }


def _trace_state_payload(
    estimator: PurePFEstimator,
) -> tuple[str, dict[str, Any]]:
    """Build the reusable trace payload for one unchanged estimator state."""
    serialized = estimator.serialized_state()
    return _sha256_bytes(serialized), estimator.posterior_snapshot().to_dict()


def _validated_replay_views(
    log: MeasurementLog,
    record_count: int,
) -> tuple[MeasurementLogStationView, MeasurementLogArrayView]:
    """Return shared aligned views for one complete replay prefix."""
    try:
        station_view = log.prefix(record_count).station_view()
        array_view = station_view.array_view()
    except MeasurementLogValidationError as exc:
        raise PFReplayError(f"Invalid shared MeasurementLog replay view: {exc}") from exc
    incomplete = [
        station.station_id
        for station in station_view.stations
        if not station.marked_complete
    ]
    if incomplete:
        raise PFReplayError(
            f"station_id {incomplete[0]} lacks station_complete at the "
            "selected replay boundary."
        )
    return station_view, array_view


def replay_records(
    log: MeasurementLog,
    estimator: PurePFEstimator,
    *,
    stop_after: int | None = None,
    pre_record_callback: Callable[
        [PurePFEstimator, MeasurementLogRecord, int, int], None
    ]
    | None = None,
    station_complete_callback: Callable[
        [PurePFEstimator, MeasurementLogRecord, int], None
    ]
    | None = None,
) -> tuple[dict[str, Any], ...]:
    """Replay a causal prefix using durable station boundaries."""
    if stop_after is None:
        limit = len(log.records)
    else:
        limit = _json_integer(
            stop_after,
            location="stop_after",
            minimum=0,
        )
        if limit > len(log.records):
            raise PFReplayError("stop_after exceeds the MeasurementLog record count.")
    contract_hash = _sha256_string(
        log.run_manifest.get("full_spectrum_contract_hash_sha256"),
        location="run_manifest.full_spectrum_contract_hash_sha256",
    )
    environment_lower, environment_upper = _resolved_forward_context(log).bounds_xyz
    station_view, array_view = _validated_replay_views(log, limit)
    outside = np.flatnonzero(
        np.any(
            (array_view.detector_pose_xyz < environment_lower[None, :])
            | (array_view.detector_pose_xyz > environment_upper[None, :]),
            axis=1,
        )
    )
    if outside.size:
        raise PFReplayError(
            f"records[{int(outside[0])}] detector pose lies outside the "
            "logged environment."
        )
    trace: list[dict[str, Any]] = []
    trace_state_payload: tuple[str, dict[str, Any]] | None = None
    for station_index, station in enumerate(station_view.stations):
        pose = array_view.detector_pose_xyz[station.start_index]
        if station_index == 0 and estimator.poses:
            estimator.poses[-1] = pose.copy()
            estimator.kernel_cache = None
            pose_index = len(estimator.poses) - 1
        else:
            estimator.add_measurement_pose(pose, reset_filters=False)
            pose_index = len(estimator.poses) - 1
        trace_state_payload = None
        pending_records: list[tuple[object, ...]] = []
        for record_index in range(station.start_index, station.stop_index):
            record = station_view.records[record_index]
            if pre_record_callback is not None:
                pre_record_callback(estimator, record, record_index, pose_index)
                # Callbacks may mutate estimator internals outside the replay API.
                trace_state_payload = None
            pending_records.append(measurement_record_to_spectrum_input(record))
            if record_index + 1 == station.stop_index:
                estimator.update_spectrum_station(
                    tuple(pending_records),
                    pose_idx=pose_index,
                    generative_contract_hash_sha256=contract_hash,
                )
                if station_complete_callback is not None:
                    station_complete_callback(estimator, record, record_index)
                trace_state_payload = None
            if trace_state_payload is None:
                trace_state_payload = _trace_state_payload(estimator)
            trace.append(
                _trace_row(
                    estimator,
                    record,
                    record_index=record_index,
                    state_payload=trace_state_payload,
                )
            )
    return tuple(trace)


def _write_replay_outputs(
    output_dir: str | Path,
    *,
    estimator: PurePFEstimator,
    trace: Sequence[Mapping[str, Any]],
    log: MeasurementLog,
) -> Path:
    """Atomically publish the pure-PF replay result contract."""
    target = Path(output_dir)
    with AtomicBundlePublisher(target, policy="create") as publisher:
        posterior = estimator.posterior_snapshot().to_dict()
        publisher.write_bytes(
            "pf_posterior.json",
            canonical_json_bytes(posterior),
        )
        trace_bytes = b"".join(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for row in trace
        )
        publisher.write_bytes("pf_trace.jsonl", trace_bytes)
        final_state = estimator.serialized_state()
        structural = estimator.structural_transition_diagnostics()
        structural_model = dict(posterior["structural_model_manifest"])
        diagnostics = {
            "schema_version": 2,
            "estimator_family": "pure_particle_filter",
            "estimator_variant": estimator.estimator_variant,
            "measurement_log_schema_version": log.schema_version,
            "measurement_log_sha256": log.log_sha256,
            "measurement_log_resolved_config_sha256": (log.resolved_config_sha256),
            "full_spectrum_contract_hash_sha256": log.run_manifest[
                "full_spectrum_contract_hash_sha256"
            ],
            "config_sha256": estimator.config_hash,
            "resolved_config_sha256": estimator.resolved_config_hash,
            "record_count": len(trace),
            "records_processed": len(trace),
            "final_state_sha256": _sha256_bytes(final_state),
            "forward_model_compatibility": "local_manifest_exact_match",
            "posterior_semantics": str(structural["posterior_semantics"]),
            "structural_kernel_family": str(structural["structural_kernel_family"]),
            "structural_kernel_target_preserving": bool(
                structural["structural_kernel_target_preserving"]
            ),
            "structural_kernel_exact_rj": bool(
                structural["structural_kernel_exact_rj"]
            ),
            "reversible_jump_mcmc_used": bool(structural["reversible_jump_mcmc_used"]),
            "structural_transition_provenance": dict(structural),
            "structural_model_manifest": structural_model,
            "posterior_predictive_check": (
                estimator.posterior_predictive_check()
            ),
            "detected_isotope_gate": getattr(
                estimator,
                "detected_isotope_gate_diagnostics",
                None,
            ),
            "candidate_isotopes": list(
                getattr(estimator, "candidate_isotopes", estimator.isotopes)
            ),
            "active_isotopes": list(estimator.joint_isotope_order()),
        }
        publisher.write_bytes(
            "pf_diagnostics.json",
            canonical_json_bytes(diagnostics),
        )
        publisher.publish()
    return target


def replay_measurement_log(
    measurement_log: str | Path | MeasurementLog,
    config: str | Path | Mapping[str, Any],
    *,
    profile: str = "pf_strict",
    seed: int = 0,
    stop_after: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PurePFEstimator, tuple[dict[str, Any], ...]]:
    """Validate, replay, and optionally persist one pure-PF result."""
    if not isinstance(profile, str):
        raise PFReplayError("Replay profile must be a JSON string.")
    replay_seed = _json_integer(seed, location="seed", minimum=0)
    log = (
        measurement_log
        if isinstance(measurement_log, MeasurementLog)
        else load_measurement_log(measurement_log)
    )
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        try:
            config_bytes = config_path.read_bytes()
        except OSError as exc:
            raise PFReplayError(f"Cannot read replay config {config_path}.") from exc
        config_hash = _sha256_bytes(config_bytes)
        resolved = _load_replay_runtime_config(config_path, seen=set())
    else:
        if not isinstance(config, Mapping):
            raise PFReplayError("Replay configuration must be an object.")
        if any(not isinstance(key, str) for key in config):
            raise PFReplayError("Replay configuration keys must be JSON strings.")
        raw_config = dict(config)
        config_hash = sha256_json(raw_config)
        resolved = raw_config
    try:
        resolved = enforce_pure_runtime_settings(resolved, profile=profile)
    except (TypeError, ValueError) as exc:
        raise PFReplayError("Replay configuration is incompatible.") from exc
    detected_only = resolved.get(
        "pf_detected_isotopes_only",
        resolved.get("detected_isotopes_only", False),
    )
    if not isinstance(detected_only, bool):
        raise PFReplayError("pf_detected_isotopes_only must be a boolean.")
    inference_isotopes: tuple[str, ...] | None = None
    gate_diagnostics: dict[str, object] | None = None
    if detected_only:
        inference_isotopes, gate_diagnostics = detect_replay_isotopes(
            log,
            resolved,
            profile=profile,
            seed=replay_seed,
            config_hash=config_hash,
            stop_after=stop_after,
        )
    estimator = build_replay_estimator(
        log,
        resolved,
        profile=profile,
        seed=replay_seed,
        config_hash=config_hash,
        resolved_config_hash=None,
        inference_isotopes=inference_isotopes,
    )
    estimator.detected_isotope_gate_diagnostics = gate_diagnostics
    trace = replay_records(log, estimator, stop_after=stop_after)
    if output_dir is not None:
        _write_replay_outputs(
            output_dir,
            estimator=estimator,
            trace=trace,
            log=log,
        )
    return estimator, trace


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the public sequential replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("pf_strict",),
        default="pf_strict",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    replay_measurement_log(
        args.measurement_log,
        args.config,
        profile=args.profile,
        seed=args.seed,
        stop_after=args.stop_after,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
