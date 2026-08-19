"""Sequentially replay a MeasurementLog v2 through the pure particle filter."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.observation_model import build_runtime_observation_model
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.surface_charts import (
    build_surface_chart_geometry,
)
from pf.full_spectrum import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from pf.gpu_utils import preflight_compute_backend
from pf.isotope_gate import FullSpectrumIsotopeGate
from pf.profiles import apply_profile_to_config, enforce_pure_runtime_settings
from pf.provenance import canonical_json_bytes, sha256_json
from pf.pure_estimator import PurePFEstimator, RotatingShieldPFConfig
from measurement.surface_atlas import ContinuousSurfaceAtlas
from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    MeasurementLogRecord,
    build_forward_model_manifest,
    load_measurement_log,
)
from runtime.records import RunContext
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    geometry_conditioned_model_from_runtime_config,
)


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


def validate_local_forward_model(log: MeasurementLog) -> None:
    """Fail closed unless the log identifies the current physical model."""
    obstacle_layout_path = log.run_manifest.get("obstacle_layout_path")
    if obstacle_layout_path is not None and not isinstance(
        obstacle_layout_path,
        str,
    ):
        raise PFReplayError(
            "run_manifest.obstacle_layout_path must be a string or null."
        )
    try:
        expected = build_forward_model_manifest(
            runtime_config=log.runtime_config,
            environment=log.environment,
            obstacle_layout_path=obstacle_layout_path,
            isotopes=tuple(log.run_manifest.get("isotopes", ())),
            repository_commit=log.run_manifest["repository_commit"],
            resolved_config_sha256=log.run_manifest["resolved_config_sha256"],
            run_root=log.path,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFReplayError(
            "Cannot reconstruct the local forward-model identity."
        ) from exc
    if dict(log.forward_model_manifest) != expected:
        raise PFReplayError(
            "Forward-model compatibility check failed; replay refuses a "
            "missing, unknown, or mismatched model field."
        )


def validate_local_full_spectrum_contract(
    log: MeasurementLog,
) -> GeometryConditionedSpectralModel:
    """Reconstruct and authenticate the sole approved observation law."""
    raw_schema_version = log.run_manifest.get("schema_version")
    if (
        isinstance(raw_schema_version, bool)
        or not isinstance(raw_schema_version, int)
        or raw_schema_version != MEASUREMENT_LOG_SCHEMA_VERSION
    ):
        raise PFReplayError(
            "Production replay supports only MeasurementLog schema version "
            f"{MEASUREMENT_LOG_SCHEMA_VERSION}."
        )
    raw_isotopes = log.run_manifest.get("isotopes")
    if (
        not isinstance(raw_isotopes, list)
        or not raw_isotopes
        or any(not isinstance(value, str) or not value for value in raw_isotopes)
    ):
        raise PFReplayError("Replay isotopes must be nonempty exact JSON strings.")
    isotopes = tuple(raw_isotopes)
    if len(set(isotopes)) != len(isotopes) or isotopes != tuple(sorted(isotopes)):
        raise PFReplayError(
            "Replay isotopes must be nonempty, unique, and canonically sorted."
        )
    try:
        model = geometry_conditioned_model_from_runtime_config(
            log.runtime_config,
            run_root=log.path,
        )
        model.require_environment_applicable(log.environment)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFReplayError(
            "Cannot authenticate the logged full-spectrum generative model."
        ) from exc
    raw_model_isotopes = [row.get("isotope") for row in model.line_identity]
    if any(not isinstance(value, str) or not value for value in raw_model_isotopes):
        raise PFReplayError(
            "Full-spectrum line isotopes must be nonempty exact strings."
        )
    model_isotopes = tuple(sorted(set(raw_model_isotopes)))
    if not set(isotopes).issubset(model_isotopes):
        raise PFReplayError(
            "Full-spectrum line isotopes do not cover the run manifest."
        )
    manifest_hash = _sha256_string(
        log.run_manifest.get("full_spectrum_contract_hash_sha256"),
        location="run_manifest.full_spectrum_contract_hash_sha256",
    )
    if manifest_hash != model.contract_hash_sha256:
        raise PFReplayError("Run and full-spectrum model contract hashes differ.")
    axis = np.asarray(model.energy_axis_keV, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 2
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise PFReplayError(
            "Full-spectrum model energy axis must be a finite increasing vector."
        )
    bin_width = float(axis[1] - axis[0])
    canonical_edges = np.concatenate(
        (axis, np.asarray([axis[-1] + bin_width], dtype=np.float64))
    )
    for record_index, record in enumerate(log.records):
        if (
            record.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY)
            != model.contract_hash_sha256
        ):
            raise PFReplayError(
                f"Full-spectrum model hash mismatch at record {record_index}."
            )
        if not np.array_equal(
            np.asarray(record.energy_bin_edges_keV, dtype=np.float64),
            canonical_edges,
        ):
            raise PFReplayError(
                f"Full-spectrum energy-bin edges mismatch at record {record_index}."
            )
        spectrum = np.asarray(record.spectrum_counts)
        if (
            spectrum.dtype != np.int64
            or spectrum.shape != axis.shape
            or np.any(spectrum < 0)
        ):
            raise PFReplayError(
                "Replay spectra must be nonnegative int64 arrays on the "
                f"approved axis (record {record_index})."
            )
    return model


def _resolved_physical_config(
    log: MeasurementLog,
    full_spectrum_model: GeometryConditionedSpectralModel,
) -> dict[str, Any]:
    """Return replay physics with the authenticated spectrum asset inlined."""
    physical_config = dict(log.runtime_config)
    physical_config.pop("full_spectrum_generative_model_path", None)
    physical_config.pop("full_spectrum_generative_model_file_sha256", None)
    physical_config["full_spectrum_generative_model"] = (
        full_spectrum_model.manifest_payload()
    )
    return physical_config


def _obstacle_grid_from_log(log: MeasurementLog) -> ObstacleGrid | None:
    """Build the logged obstacle grid without reading evaluation truth."""
    if "obstacle_grid" not in log.environment:
        raise PFReplayError("environment requires explicit obstacle_grid.")
    raw = log.environment["obstacle_grid"]
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PFReplayError("environment.obstacle_grid must be an object or null.")
    try:
        return ObstacleGrid.from_dict(dict(raw))
    except (TypeError, ValueError) as exc:
        raise PFReplayError("environment.obstacle_grid is incompatible.") from exc


def _required_boolean(
    config: Mapping[str, Any],
    key: str,
) -> bool:
    """Return one exact boolean without accepting truthy strings or integers."""
    if key not in config:
        raise PFReplayError(f"{key} must be explicitly logged.")
    raw = config[key]
    if not isinstance(raw, bool):
        raise PFReplayError(f"{key} must be a JSON boolean.")
    return raw


def _environment_config(log: MeasurementLog) -> EnvironmentConfig:
    """Return strict logged room geometry without numeric coercion."""
    required = ("size_x", "size_y", "size_z", "detector_position")
    missing = [key for key in required if key not in log.environment]
    if missing:
        raise PFReplayError(
            "MeasurementLog environment is incomplete: " + ", ".join(missing)
        )
    try:
        return EnvironmentConfig(
            size_x=log.environment["size_x"],
            size_y=log.environment["size_y"],
            size_z=log.environment["size_z"],
            detector_position=log.environment["detector_position"],
        )
    except (TypeError, ValueError) as exc:
        raise PFReplayError("MeasurementLog environment geometry is invalid.") from exc


def _environment_bounds(
    log: MeasurementLog,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return source-position bounds from the logged environment."""
    environment = _environment_config(log)
    upper = np.asarray(
        [environment.size_x, environment.size_y, environment.size_z],
        dtype=np.float64,
    )
    return np.zeros(3, dtype=np.float64), upper


def _surface_atlas_replay_inputs(
    log: MeasurementLog,
    *,
    pf_config: RotatingShieldPFConfig,
    obstacle_grid: ObstacleGrid | None,
    obstacle_height_m: float,
) -> NDArray[np.float64]:
    """Build deterministic PF diagnostics from shared physical geometry."""
    point_count = _DEFAULT_SURFACE_DIAGNOSTIC_POINT_COUNT
    environment = _environment_config(log)
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
    if not isinstance(config, Mapping):
        raise PFReplayError("Replay configuration must be an object.")
    if any(not isinstance(key, str) for key in config):
        raise PFReplayError("Replay configuration keys must be JSON strings.")
    if not isinstance(profile, str):
        raise PFReplayError("Replay profile must be a JSON string.")
    replay_seed = _json_integer(seed, location="seed", minimum=0)
    full_spectrum_model = validate_local_full_spectrum_contract(log)
    validate_local_forward_model(log)
    isotopes = tuple(log.run_manifest["isotopes"])
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
    _, upper = _environment_bounds(log)
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
    physical_config = _resolved_physical_config(log, full_spectrum_model)
    try:
        observation_model = build_runtime_observation_model(
            physical_config,
            isotopes=isotopes,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PFReplayError(
            "Cannot reconstruct the logged PF observation model."
        ) from exc
    obstacle_grid = _obstacle_grid_from_log(log)
    obstacle_enabled = _required_boolean(
        physical_config,
        "obstacle_attenuation_enabled",
    )
    if obstacle_grid is not None and not obstacle_enabled:
        raise PFReplayError(
            "A logged obstacle grid requires physical obstacle attenuation."
        )
    pf_obstacle_grid = obstacle_grid if obstacle_enabled else None
    surface_diagnostic_points = _surface_atlas_replay_inputs(
        log,
        pf_config=pf_config,
        obstacle_grid=pf_obstacle_grid,
        obstacle_height_m=observation_model.obstacle_height_m,
    )
    actual_pf = {
        field.name: getattr(pf_config, field.name)
        for field in fields(RotatingShieldPFConfig)
    }
    logged_config_sha256 = _sha256_string(
        log.run_manifest.get("resolved_config_sha256"),
        location="run_manifest.resolved_config_sha256",
    )
    effective_measurement_digest = (
        log.log_sha256
        if measurement_log_digest is None
        else str(measurement_log_digest)
    )
    replay_hash_payload: dict[str, object] = {
        "measurement_runtime_config_sha256": logged_config_sha256,
        "measurement_log_sha256": effective_measurement_digest,
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
        measurement_log_schema_version=log.schema_version,
        config_hash=input_config_sha256,
        resolved_config_hash=replay_config_sha256,
        measurement_log_sha256=effective_measurement_digest,
        random_seed=replay_seed,
    )
    environment = _environment_config(log)
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
    """Construct a PF from a truth-free live runtime handshake.

    The final MeasurementLog digest is unavailable during acquisition. The
    caller must bind the published digest before serializing a posterior.
    """
    root = Path(runtime_root).expanduser().resolve()
    runtime_config = dict(context.runtime_config)
    contract_hash = runtime_config.get("full_spectrum_contract_hash_sha256")
    if not isinstance(contract_hash, str):
        raise PFReplayError(
            "Adaptive runtime context lacks the full-spectrum contract hash."
        )
    manifest = {
        "schema_version": int(context.schema_version),
        "run_id": str(context.run_id),
        "record_count": 0,
        "repository_commit": str(context.repository_commit),
        "resolved_config_sha256": str(context.runtime_config_sha256),
        "source_rate_model": str(context.source_rate_model),
        "source_rate_semantics": dict(context.source_rate_semantics),
        "isotopes": list(context.isotopes),
        "environment": dict(context.environment),
        "obstacle_layout_path": context.obstacle_layout_path,
        "source_layout_path": None,
        "sim_backend": str(context.sim_backend),
        "full_spectrum_contract_hash_sha256": contract_hash,
        "metadata": dict(context.metadata),
    }
    live_log = MeasurementLog(
        run_manifest=manifest,
        runtime_config=runtime_config,
        environment=dict(context.environment),
        forward_model_manifest=dict(context.forward_model_manifest),
        records=(),
        path=root,
    )
    return build_replay_estimator(
        live_log,
        config,
        profile=profile,
        seed=seed,
        config_hash=config_hash,
        measurement_log_digest="unavailable",
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
    validate_local_forward_model(log)
    validate_local_full_spectrum_contract(log)
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


def _spectrum_record(record: MeasurementLogRecord) -> tuple[object, ...]:
    """Retain the private compatibility name for existing replay tests."""
    return measurement_record_to_spectrum_input(record)


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
) -> dict[str, Any]:
    """Serialize one causal posterior state at a record boundary."""
    serialized = estimator.serialized_state()
    return {
        "schema_version": 2,
        "estimator_family": "pure_particle_filter",
        "estimator_variant": estimator.estimator_variant,
        "record_index": int(record_index),
        "step_id": int(record.step_id),
        "action_id": int(record.action_id),
        "station_id": int(record.station_id),
        "station_complete": _station_complete(record),
        "state_sha256": _sha256_bytes(serialized),
        "posterior": estimator.posterior_snapshot().to_dict(),
    }


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
    _, environment_upper = _environment_bounds(log)
    station_pose: dict[int, NDArray[np.float64]] = {}
    station_quaternion: dict[int, NDArray[np.float64]] = {}
    station_pose_index: dict[int, int] = {}
    trace: list[dict[str, Any]] = []
    pending_station_id: int | None = None
    pending_pose_idx: int | None = None
    pending_records: list[tuple[object, ...]] = []
    completed_station_ids: set[int] = set()
    previous_station_id: int | None = None

    for record_index, record in enumerate(log.records[:limit]):
        step_id = _json_integer(
            record.step_id,
            location=f"records[{record_index}].step_id",
            minimum=0,
        )
        action_id = _json_integer(
            record.action_id,
            location=f"records[{record_index}].action_id",
            minimum=0,
        )
        station_id = _json_integer(
            record.station_id,
            location=f"records[{record_index}].station_id",
            minimum=0,
        )
        if step_id != record_index or action_id != record_index:
            raise PFReplayError(
                "Replay step_id and action_id must equal zero-based record order."
            )
        if previous_station_id is None:
            if station_id != 0:
                raise PFReplayError("Replay station_id must start at zero.")
        elif station_id not in {
            previous_station_id,
            previous_station_id + 1,
        }:
            raise PFReplayError(
                "Replay station_id must form contiguous nondecreasing groups."
            )
        pose = _finite_vector(
            record.detector_pose_xyz,
            length=3,
            location=f"records[{record_index}].detector_pose_xyz",
        )
        quaternion = _finite_vector(
            record.detector_quat_wxyz,
            length=4,
            location=f"records[{record_index}].detector_quat_wxyz",
        )
        if np.any(pose < 0.0) or np.any(pose > environment_upper):
            raise PFReplayError(
                f"records[{record_index}] detector pose lies outside the "
                "logged environment."
            )
        station_complete = _station_complete(record)
        spectrum_record = _spectrum_record(record)
        if station_id in completed_station_ids:
            raise PFReplayError(
                f"station_id {station_id} has records after station_complete."
            )
        if pending_station_id is not None and station_id != pending_station_id:
            raise PFReplayError(
                f"station_id {pending_station_id} lacks station_complete."
            )
        if station_id not in station_pose:
            if not station_pose and estimator.poses:
                estimator.poses[-1] = pose.copy()
                estimator.kernel_cache = None
                pose_index = len(estimator.poses) - 1
            else:
                estimator.add_measurement_pose(pose, reset_filters=False)
                pose_index = len(estimator.poses) - 1
            station_pose[station_id] = pose.copy()
            station_quaternion[station_id] = quaternion.copy()
            station_pose_index[station_id] = pose_index
        elif not np.array_equal(station_pose[station_id], pose) or not np.array_equal(
            station_quaternion[station_id],
            quaternion,
        ):
            raise PFReplayError(
                f"station_id {station_id} contains multiple detector poses "
                "or quaternions."
            )
        pose_idx = station_pose_index[station_id]
        if pending_station_id is None:
            pending_station_id = station_id
            pending_pose_idx = pose_idx
        if pre_record_callback is not None:
            pre_record_callback(estimator, record, record_index, pose_idx)
        pending_records.append(spectrum_record)

        if station_complete:
            assert pending_pose_idx is not None
            estimator.update_spectrum_station(
                tuple(pending_records),
                pose_idx=pending_pose_idx,
                generative_contract_hash_sha256=contract_hash,
            )
            if station_complete_callback is not None:
                station_complete_callback(estimator, record, record_index)
            completed_station_ids.add(station_id)
            pending_station_id = None
            pending_pose_idx = None
            pending_records = []
        trace.append(_trace_row(estimator, record, record_index=record_index))
        previous_station_id = station_id
    if pending_station_id is not None:
        raise PFReplayError(
            f"station_id {pending_station_id} lacks station_complete at the "
            "selected replay boundary."
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
    if target.exists():
        raise FileExistsError(f"Refusing to replace replay output {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary replay output exists: {temporary}.")
    temporary.mkdir()
    try:
        posterior = estimator.posterior_snapshot().to_dict()
        (temporary / "pf_posterior.json").write_bytes(canonical_json_bytes(posterior))
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
        (temporary / "pf_trace.jsonl").write_bytes(trace_bytes)
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
        (temporary / "pf_diagnostics.json").write_bytes(
            canonical_json_bytes(diagnostics)
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def replay_measurement_log(
    measurement_log: str | Path,
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
    log = load_measurement_log(measurement_log)
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
