"""Read and write raw full-spectrum MeasurementLog schema version 2."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import io
import json
from numbers import Real
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from pf.provenance import canonical_json_bytes, json_safe
from pf.full_spectrum import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.forward_model_manifest import (
    CANONICAL_UNITS,
    FORWARD_MODEL_MANIFEST_SCHEMA_VERSION,
    REQUIRED_MODEL_NAMES,
    RESPONSE_SEMANTICS,
    SOURCE_RATE_MODEL,
    SOURCE_RATE_SEMANTICS,
    build_forward_model_manifest as _build_forward_model_manifest,
    validate_forward_model_manifest as _validate_forward_model_manifest,
)

MEASUREMENT_LOG_SCHEMA_VERSION = 2
FULL_SPECTRUM_MODEL_SCHEMA_VERSION = 3
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_CANONICAL_REQUIRED_FILES = (
    "run_manifest.json",
    "runtime_config.resolved.json",
    "environment.json",
    "forward_model_manifest.json",
    "observations.npz",
    "observation_metadata.jsonl",
    "repository_commit.txt",
)
_STREAM_STATIC_FILES = frozenset(
    {
        "runtime_config.resolved.json",
        "environment.json",
        "forward_model_manifest.input.json",
        "observation_metadata.jsonl",
        "repository_commit.txt",
    }
)
_STREAM_RECORD_PATTERN = re.compile(r"^record_([0-9]{8})\.npz$")
_STREAM_METADATA_TEMP_PATTERN = re.compile(
    r"^\.observation_metadata\.jsonl\.tmp-[0-9]+$"
)
_MODEL_KEYS = REQUIRED_MODEL_NAMES
_SOURCE_RATE_SEMANTICS = SOURCE_RATE_SEMANTICS
_INDEX_CONVENTIONS = {
    "record_order": "causal_step_order",
    "step_id": "zero_based_strictly_increasing",
    "action_id": "zero_based_unique_measurement_action",
    "station_id": "zero_based_nondecreasing_station_group",
}
_FORWARD_UNITS = CANONICAL_UNITS
_RESPONSE_SEMANTICS = RESPONSE_SEMANTICS
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REALIZED_SOURCE_KEYS = {
    "sourcelayout",
    "sourcelayoutpath",
    "sourcepositions",
    "pointsources",
    "sources",
    "sourcelist",
}
_FORBIDDEN_TRUTH_VALUE_TERMS = (
    "truth",
    "sourcelayout",
    "sourcepositions",
    "pointsources",
)
_REMOVED_RECORD_METADATA_KEYS = frozenset(
    {
        "censored_observation_by_isotope",
        "fixed_estimator_contract_hash_sha256",
        "runtime_likelihood_route_by_isotope",
    }
)
_RUN_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "record_count",
        "repository_commit",
        "resolved_config_sha256",
        "forward_model_manifest_sha256",
        "source_rate_model",
        "source_rate_semantics",
        "isotopes",
        "environment",
        "obstacle_layout_path",
        "source_layout_path",
        "sim_backend",
        "observation_model",
        "energy_bin_count",
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
        "full_spectrum_contract_hash_sha256",
        "full_spectrum_contract_schema_version",
        "model_identifiers",
        "index_conventions",
        "artifact_hashes",
        "metadata",
    }
)


def _normalized_contract_name(value: object) -> str:
    """Collapse case and separators for truth-contract comparisons."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _indicates_realized_truth(name: str, *, key: bool) -> bool:
    """Return whether a normalized name exposes realized source truth."""
    if key and any(term in name for term in _FORBIDDEN_TRUTH_VALUE_TERMS):
        return True
    if name.startswith(("sourcerate", "sourceextent")):
        return False
    if key:
        return name in _REALIZED_SOURCE_KEYS
    return any(
        term in name
        for term in ("sourcelayout", "sourcepositions", "pointsources")
    )


def _validate_truth_free_payload(value: object, *, location: str) -> None:
    """Reject recursively embedded realized truth while retaining physics fields."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MeasurementLogValidationError(
                    f"{location} keys must be exact JSON strings."
                )
            normalized = _normalized_contract_name(key)
            aggregate_validation_metric = location.endswith(
                ".full_spectrum_generative_model.validation.metrics"
            )
            if (
                _indicates_realized_truth(normalized, key=True)
                and not aggregate_validation_metric
            ):
                raise MeasurementLogValidationError(
                    f"{location}.{key} contains estimator-visible realized truth."
                )
            _validate_truth_free_payload(
                nested,
                location=f"{location}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_truth_free_payload(
                nested,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(value, str):
        normalized = _normalized_contract_name(value)
        lower_value = value.casefold()
        truth_path = (
            "truth" in normalized
            and (
                "/" in value
                or "\\" in value
                or lower_value.endswith(
                    (
                        ".json",
                        ".jsonl",
                        ".npy",
                        ".npz",
                        ".csv",
                        ".yaml",
                        ".yml",
                    )
                )
            )
        )
        if _indicates_realized_truth(normalized, key=False) or truth_path:
            raise MeasurementLogValidationError(
                f"{location} points to estimator-visible realized truth."
            )


def _validate_source_layout_sentinel(value: object, *, location: str) -> None:
    """Require the truth-free source-layout pointer to remain null."""
    if value is not None:
        raise MeasurementLogValidationError(
            f"{location} must be null; source truth belongs outside MeasurementLog."
        )


def build_forward_model_manifest(
    *,
    runtime_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    obstacle_layout_path: str | None,
    isotopes: Sequence[str],
    repository_commit: str,
    resolved_config_sha256: str,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build the shared strict identity contract for the production PF model."""
    return dict(
        _build_forward_model_manifest(
            runtime_config=runtime_config,
            environment=environment,
            obstacle_layout_path=obstacle_layout_path,
            isotopes=isotopes,
            repository_commit=repository_commit,
            resolved_config_sha256=resolved_config_sha256,
            source_rate_model=SOURCE_RATE_MODEL,
            run_root=run_root,
            repository_root=repository_root,
        )
    )


class MeasurementLogValidationError(ValueError):
    """Report a schema, hash, shape, or forward-model incompatibility."""


def _canonical_isotope_names(
    isotopes: Sequence[str],
    *,
    location: str,
) -> tuple[str, ...]:
    """Return exact nonempty unique isotope strings in canonical order."""
    if not isinstance(isotopes, (list, tuple)) or any(
        not isinstance(value, str) or not value for value in isotopes
    ):
        raise MeasurementLogValidationError(
            f"{location} must be an array of nonempty JSON strings."
        )
    names = tuple(isotopes)
    if (
        not names
        or len(set(names)) != len(names)
        or names != tuple(sorted(names))
    ):
        raise MeasurementLogValidationError(
            f"{location} must be nonempty, unique, and canonically sorted."
        )
    return names


def _validate_run_identity(run_id: object, repository_commit: object) -> None:
    """Require exact run and repository identifiers before writing artifacts."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise MeasurementLogValidationError("run_id must be a nonempty string.")
    if (
        not isinstance(repository_commit, str)
        or _GIT_COMMIT_PATTERN.fullmatch(repository_commit) is None
    ):
        raise MeasurementLogValidationError(
            "repository_commit must be a full lowercase Git hash."
        )


def _finite_real(value: object, *, location: str) -> float:
    """Return one exact finite real without accepting booleans or strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MeasurementLogValidationError(
            f"{location} must be a finite JSON number."
        )
    parsed = float(value)
    if not np.isfinite(parsed):
        raise MeasurementLogValidationError(
            f"{location} must be a finite JSON number."
        )
    return parsed


def _finite_real_array(
    value: object,
    *,
    shape: tuple[int, ...],
    location: str,
) -> NDArray[np.float64]:
    """Return an exact-rank finite real array without dtype coercion."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise MeasurementLogValidationError(
            f"{location} must have shape {shape}."
        ) from exc
    if raw.shape != shape:
        raise MeasurementLogValidationError(
            f"{location} must have shape {shape}; got {raw.shape}."
        )
    parsed = np.asarray(
        [
            _finite_real(item, location=f"{location}[{index}]")
            for index, item in enumerate(raw.flat)
        ],
        dtype=np.float64,
    )
    return parsed.reshape(shape)


def _validate_environment_payload(payload: Mapping[str, Any]) -> None:
    """Validate the exact room inputs consumed by replay and surface support."""
    if not isinstance(payload, Mapping):
        raise MeasurementLogValidationError("environment must be an object.")
    required = {
        "environment_model_id",
        "size_x",
        "size_y",
        "size_z",
        "detector_position",
        "obstacle_grid",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise MeasurementLogValidationError(
            "environment is missing replay fields: " + ", ".join(missing)
        )
    model_id = payload["environment_model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise MeasurementLogValidationError(
            "environment.environment_model_id must be a nonempty JSON string."
        )
    try:
        EnvironmentConfig(
            size_x=payload["size_x"],
            size_y=payload["size_y"],
            size_z=payload["size_z"],
            detector_position=payload["detector_position"],
        )
        obstacle_grid = payload["obstacle_grid"]
        if obstacle_grid is not None:
            if not isinstance(obstacle_grid, dict):
                raise ValueError("obstacle_grid must be an object or null.")
            ObstacleGrid.from_dict(dict(obstacle_grid))
    except (TypeError, ValueError) as exc:
        raise MeasurementLogValidationError(
            f"environment is incompatible with replay geometry: {exc}"
        ) from exc


def _validate_runtime_replay_contract(
    runtime_config: Mapping[str, Any],
    *,
    isotopes: Sequence[str],
) -> None:
    """Validate explicit runtime fields that replay must never default."""
    if not isinstance(runtime_config, Mapping):
        raise MeasurementLogValidationError("runtime_config must be an object.")
    if runtime_config.get("pure_pf_schema_version") != 1 or isinstance(
        runtime_config.get("pure_pf_schema_version"),
        bool,
    ):
        raise MeasurementLogValidationError(
            "runtime_config.pure_pf_schema_version must be JSON integer 1."
        )
    if runtime_config.get("estimator_profile") != "pf_strict":
        raise MeasurementLogValidationError(
            "runtime_config.estimator_profile must be 'pf_strict'."
        )
    obstacle_attenuation = runtime_config.get("pf_obstacle_attenuation")
    if not isinstance(obstacle_attenuation, bool):
        raise MeasurementLogValidationError(
            "runtime_config.pf_obstacle_attenuation must be an explicit boolean."
        )
    sim_backend = runtime_config.get("sim_backend")
    if not isinstance(sim_backend, str) or not sim_backend.strip():
        raise MeasurementLogValidationError(
            "runtime_config.sim_backend must be a nonempty JSON string."
        )
    candidate_isotopes = runtime_config.get("candidate_isotopes")
    if candidate_isotopes is not None:
        configured = _canonical_isotope_names(
            candidate_isotopes,
            location="runtime_config.candidate_isotopes",
        )
        if configured != tuple(isotopes):
            raise MeasurementLogValidationError(
                "runtime_config.candidate_isotopes differs from the run isotopes."
            )


def _station_complete_marker(
    metadata: Mapping[str, Any],
    *,
    location: str = "record.metadata",
) -> bool:
    """Return the exact writer-owned station completion marker."""
    if "station_complete" not in metadata:
        return False
    marker = metadata["station_complete"]
    if not isinstance(marker, bool):
        raise MeasurementLogValidationError(
            f"{location}.station_complete must be a boolean."
        )
    return marker


def _required_json_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    """Return one exact JSON integer from an external manifest."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeasurementLogValidationError(
            f"{location}.{key} must be a JSON integer."
        )
    if minimum is not None and value < minimum:
        raise MeasurementLogValidationError(
            f"{location}.{key} must be at least {minimum}."
        )
    return value


def _required_json_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    location: str,
) -> float:
    """Return one finite exact JSON number from an external manifest."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementLogValidationError(
            f"{location}.{key} must be a JSON number."
        )
    parsed = float(value)
    if not np.isfinite(parsed):
        raise MeasurementLogValidationError(
            f"{location}.{key} must be finite."
        )
    return parsed


@dataclass(frozen=True)
class MeasurementLogRecord:
    """Store one ordered, estimator-independent measurement action."""

    step_id: int
    action_id: int
    station_id: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_orientation_index: int
    pb_orientation_index: int
    live_time_s: float
    travel_time_s: float
    shield_actuation_time_s: float
    energy_bin_edges_keV: NDArray[np.float64]
    spectrum_counts: NDArray[np.int64]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate record identifiers, pose, time, and observation shapes."""
        for name in ("step_id", "action_id", "station_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise MeasurementLogValidationError(f"{name} must be an integer.")
            parsed = int(value)
            if parsed < 0 or parsed > np.iinfo(np.int64).max:
                raise MeasurementLogValidationError(f"{name} must be non-negative.")
            object.__setattr__(self, name, parsed)
        xyz = _finite_real_array(
            self.detector_pose_xyz,
            shape=(3,),
            location="detector_pose_xyz",
        )
        quaternion = _finite_real_array(
            self.detector_quat_wxyz,
            shape=(4,),
            location="detector_quat_wxyz",
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 0.0 or not np.isclose(
            quaternion_norm,
            1.0,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise MeasurementLogValidationError(
                "detector_quat_wxyz must be a normalized quaternion."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in (self.fe_orientation_index, self.pb_orientation_index)
        ):
            raise MeasurementLogValidationError(
                "Fe/Pb orientation indices must be integers."
            )
        if (
            not 0 <= int(self.fe_orientation_index) <= 7
            or not 0 <= int(self.pb_orientation_index) <= 7
        ):
            raise MeasurementLogValidationError(
                "Fe/Pb orientation indices must be in the shared octant range 0..7."
            )
        object.__setattr__(
            self,
            "fe_orientation_index",
            int(self.fe_orientation_index),
        )
        object.__setattr__(
            self,
            "pb_orientation_index",
            int(self.pb_orientation_index),
        )
        for name in ("live_time_s", "travel_time_s", "shield_actuation_time_s"):
            value = _finite_real(getattr(self, name), location=name)
            if value < 0.0:
                raise MeasurementLogValidationError(
                    f"{name} must be finite and non-negative."
                )
            object.__setattr__(self, name, value)
        if self.live_time_s <= 0.0:
            raise MeasurementLogValidationError("live_time_s must be positive.")
        try:
            raw_edges = np.asarray(self.energy_bin_edges_keV)
        except (TypeError, ValueError) as exc:
            raise MeasurementLogValidationError(
                "energy_bin_edges_keV must be a one-dimensional numeric array."
            ) from exc
        if raw_edges.ndim != 1:
            raise MeasurementLogValidationError(
                "energy_bin_edges_keV must be a one-dimensional numeric array."
            )
        edges = _finite_real_array(
            raw_edges,
            shape=raw_edges.shape,
            location="energy_bin_edges_keV",
        )
        raw_spectrum = np.asarray(self.spectrum_counts)
        if edges.size != raw_spectrum.size + 1:
            raise MeasurementLogValidationError(
                "energy_bin_edges_keV must have one more value than spectrum_counts."
            )
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise MeasurementLogValidationError(
                "energy_bin_edges_keV must be finite and strictly increasing."
            )
        if (
            raw_spectrum.ndim != 1
            or raw_spectrum.dtype != np.dtype(np.int64)
            or np.any(raw_spectrum < 0)
        ):
            raise MeasurementLogValidationError(
                "spectrum_counts must be a one-dimensional array of exact "
                "nonnegative unit-weight integer event counts."
            )
        if not isinstance(self.metadata, Mapping):
            raise MeasurementLogValidationError("metadata must be an object.")
        _station_complete_marker(self.metadata)
        removed_metadata = sorted(
            set(self.metadata).intersection(_REMOVED_RECORD_METADATA_KEYS)
        )
        if removed_metadata:
            raise MeasurementLogValidationError(
                "Production MeasurementLog records contain removed metadata "
                f"keys: {removed_metadata}."
            )
        _validate_sha256(
            self.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY),
            (
                "record.metadata."
                f"{FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY}"
            ),
        )
        _validate_truth_free_payload(self.metadata, location="record.metadata")
        try:
            metadata_copy = json.loads(
                canonical_json_bytes(dict(self.metadata)).decode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise MeasurementLogValidationError(
                "metadata must contain only finite JSON values."
            ) from exc
        immutable_edges = np.array(edges, dtype=np.float64, copy=True)
        immutable_spectrum = np.array(
            raw_spectrum,
            dtype=np.int64,
            copy=True,
        )
        immutable_edges.setflags(write=False)
        immutable_spectrum.setflags(write=False)
        object.__setattr__(
            self,
            "detector_pose_xyz",
            tuple(float(value) for value in xyz),
        )
        object.__setattr__(
            self,
            "detector_quat_wxyz",
            tuple(float(value) for value in quaternion),
        )
        object.__setattr__(self, "energy_bin_edges_keV", immutable_edges)
        object.__setattr__(self, "spectrum_counts", immutable_spectrum)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_copy),
        )


@dataclass(frozen=True)
class MeasurementLog:
    """Store a validated MeasurementLog v2 bundle without evaluation truth."""

    run_manifest: Mapping[str, Any]
    runtime_config: Mapping[str, Any]
    environment: Mapping[str, Any]
    forward_model_manifest: Mapping[str, Any]
    records: tuple[MeasurementLogRecord, ...]
    path: Path | None = None

    @property
    def run_id(self) -> str:
        """Return the manifest run identifier."""
        value = self.run_manifest["run_id"]
        if not isinstance(value, str):
            raise MeasurementLogValidationError(
                "run_manifest.run_id must be a string."
            )
        return value

    @property
    def schema_version(self) -> int:
        """Return the MeasurementLog schema version."""
        value = self.run_manifest["schema_version"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise MeasurementLogValidationError(
                "run_manifest.schema_version must be a JSON integer."
            )
        return value

    @property
    def resolved_config_sha256(self) -> str:
        """Return the resolved runtime configuration digest."""
        return _validate_sha256(
            self.run_manifest["resolved_config_sha256"],
            "run_manifest.resolved_config_sha256",
        )

    @property
    def log_sha256(self) -> str:
        """Return the shared raw-file inventory digest for the log directory."""
        if self.path is None:
            raise MeasurementLogValidationError(
                "An in-memory MeasurementLog prefix has no independent directory digest."
            )
        return measurement_log_sha256(self.path)

    def prefix(self, record_count: int) -> "MeasurementLog":
        """Return an in-memory causal prefix without inspecting a future record."""
        if isinstance(record_count, bool) or not isinstance(record_count, int):
            raise MeasurementLogValidationError(
                "record_count must be a JSON integer."
            )
        if record_count < 0 or record_count > len(self.records):
            raise MeasurementLogValidationError(
                "record_count must lie within the available record range."
            )
        count = record_count
        manifest = dict(self.run_manifest)
        manifest["record_count"] = count
        return MeasurementLog(
            run_manifest=manifest,
            runtime_config=self.runtime_config,
            environment=self.environment,
            forward_model_manifest=self.forward_model_manifest,
            records=self.records[:count],
            path=self.path,
        )


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one canonical JSON artifact."""
    path.write_bytes(canonical_json_bytes(dict(payload)))


def _json_line_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one compact deterministic JSONL record."""
    text = json.dumps(
        json_safe(dict(payload)),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"{text}\n".encode("utf-8")


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    """Write an NPZ archive with fixed member order, metadata, and timestamps."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        # Insertion order is part of the shared byte-stable representation.
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(array),
                allow_pickle=False,
            )
            member = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            member.create_system = 3
            archive.writestr(member, buffer.getvalue())


def measurement_log_sha256(path: str | Path) -> str:
    """Return the shared digest of every non-truth regular artifact in a log."""
    root = Path(path).resolve()
    inventory: dict[str, str] = {}
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise MeasurementLogValidationError(
                f"MeasurementLog must not contain symlink {relative}."
            )
        if candidate.is_dir():
            continue
        for component in Path(relative).parts:
            for name in (component, Path(component).stem):
                normalized = _normalized_contract_name(name)
                if _indicates_realized_truth(normalized, key=True):
                    raise MeasurementLogValidationError(
                        "Truth/source-layout artifacts must be stored outside "
                        f"MeasurementLog ({relative})."
                    )
        if not candidate.is_file():
            raise MeasurementLogValidationError(
                f"MeasurementLog artifact {relative} is not a regular file."
            )
        inventory[relative] = _sha256_file(candidate)
    return _sha256_bytes(canonical_json_bytes(inventory))


def _validate_sha256(value: object, field_name: str) -> str:
    """Validate and return a lowercase SHA-256 string."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise MeasurementLogValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _validate_model_identifiers(payload: object) -> dict[str, dict[str, str]]:
    """Validate required physical-model identifiers and hashes."""
    if not isinstance(payload, Mapping):
        raise MeasurementLogValidationError("model_identifiers must be an object.")
    if set(payload) != set(_MODEL_KEYS):
        raise MeasurementLogValidationError(
            "model_identifiers must contain exactly the six physical components."
        )
    result: dict[str, dict[str, str]] = {}
    for key in _MODEL_KEYS:
        entry = payload.get(key)
        if not isinstance(entry, Mapping) or set(entry) != {"id", "sha256"}:
            raise MeasurementLogValidationError(
                f"model_identifiers.{key} must contain exactly id and sha256."
            )
        raw_identifier = entry.get("id")
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise MeasurementLogValidationError(
                f"model_identifiers.{key}.id must be non-empty."
            )
        identifier = raw_identifier
        result[key] = {
            "id": identifier,
            "sha256": _validate_sha256(
                entry.get("sha256"),
                f"model_identifiers.{key}.sha256",
            ),
        }
    return result


def validate_forward_model_manifest(
    payload: Mapping[str, Any],
    *,
    runtime_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    obstacle_layout_path: str | None,
    isotopes: Sequence[str],
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str = SOURCE_RATE_MODEL,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Fail closed unless the manifest exactly identifies production physics."""
    try:
        return dict(
            _validate_forward_model_manifest(
                payload,
                runtime_config=runtime_config,
                environment=environment,
                obstacle_layout_path=obstacle_layout_path,
                isotopes=isotopes,
                repository_commit=repository_commit,
                resolved_config_sha256=resolved_config_sha256,
                source_rate_model=source_rate_model,
                run_root=run_root,
                repository_root=repository_root,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementLogValidationError(
            f"Forward-model manifest is incompatible: {exc}"
        ) from exc


def _validate_record_sequence(
    records: Sequence[MeasurementLogRecord],
    isotopes: Sequence[str],
) -> tuple[str, ...]:
    """Validate one homogeneous, causally ordered MeasurementLog sequence."""
    if not records:
        raise MeasurementLogValidationError(
            "A MeasurementLog needs at least one record."
        )
    isotope_names = _canonical_isotope_names(
        isotopes,
        location="Manifest isotope names",
    )
    step_ids = np.asarray([record.step_id for record in records], dtype=np.int64)
    action_ids = np.asarray([record.action_id for record in records], dtype=np.int64)
    station_ids = np.asarray([record.station_id for record in records], dtype=np.int64)
    row_order = np.arange(len(records), dtype=np.int64)
    if not np.array_equal(step_ids, row_order):
        raise MeasurementLogValidationError(
            "step_id must equal zero-based causal record order."
        )
    if not np.array_equal(action_ids, row_order):
        raise MeasurementLogValidationError(
            "action_id must equal zero-based measurement-action order."
        )
    station_deltas = np.diff(station_ids)
    if station_ids[0] != 0 or np.any(
        (station_deltas < 0) | (station_deltas > 1)
    ):
        raise MeasurementLogValidationError(
            "station_id must form contiguous zero-based nondecreasing groups."
        )
    contract_hashes: set[str] = set()
    for index, record in enumerate(records):
        if index and station_ids[index] == station_ids[index - 1]:
            previous = records[index - 1]
            if (
                record.detector_pose_xyz != previous.detector_pose_xyz
                or record.detector_quat_wxyz != previous.detector_quat_wxyz
            ):
                raise MeasurementLogValidationError(
                    "All records in one station must share one detector pose "
                    "and quaternion."
                )
        contract_hashes.add(
            _validate_sha256(
                record.metadata.get(
                    FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
                ),
                (
                    f"records[{index}].metadata."
                    f"{FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY}"
                ),
            )
        )
    if len(contract_hashes) > 1:
        raise MeasurementLogValidationError(
            "The full-spectrum generative contract changed within one "
            "MeasurementLog."
        )
    return isotope_names


def _validate_full_spectrum_contract_alignment(
    *,
    run_manifest: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    records: Sequence[MeasurementLogRecord],
) -> None:
    """Require full-spectrum axis, isotope, and hash identity end to end."""
    raw_isotopes = run_manifest.get("isotopes")
    if (
        not isinstance(raw_isotopes, list)
        or not raw_isotopes
        or any(
            type(value) is not str or not value
            for value in raw_isotopes
        )
        or len(set(raw_isotopes)) != len(raw_isotopes)
    ):
        raise MeasurementLogValidationError(
            "run_manifest.isotopes must contain unique nonempty JSON strings."
        )
    isotope_order = tuple(raw_isotopes)
    contract = runtime_config.get("full_spectrum_generative_model")
    if not isinstance(contract, Mapping):
        raise MeasurementLogValidationError(
            "runtime_config requires full_spectrum_generative_model."
        )
    contract_schema_version = _required_json_integer(
        contract,
        "schema_version",
        location="runtime_config.full_spectrum_generative_model",
    )
    if contract_schema_version != FULL_SPECTRUM_MODEL_SCHEMA_VERSION:
        raise MeasurementLogValidationError(
            "runtime_config requires full-spectrum model schema "
            f"{FULL_SPECTRUM_MODEL_SCHEMA_VERSION}."
        )
    line_rows = contract.get("line_identity")
    if (
        not isinstance(line_rows, list)
        or not line_rows
        or any(not isinstance(row, Mapping) for row in line_rows)
    ):
        raise MeasurementLogValidationError(
            "Full-spectrum model requires a nonempty line_identity."
        )
    raw_line_isotopes = [row.get("isotope") for row in line_rows]
    if any(
        type(value) is not str or not value
        for value in raw_line_isotopes
    ):
        raise MeasurementLogValidationError(
            "Full-spectrum line isotopes must be nonempty JSON strings."
        )
    line_isotopes = tuple(
        sorted(set(raw_line_isotopes))
    )
    if line_isotopes != isotope_order:
        raise MeasurementLogValidationError(
            "Full-spectrum line isotopes must equal the canonical "
            "run-manifest isotope order."
        )
    contract_hash = _validate_sha256(
        contract.get("contract_hash_sha256"),
        "runtime_config.full_spectrum_generative_model.contract_hash_sha256",
    )
    runtime_hash = _validate_sha256(
        runtime_config.get("full_spectrum_contract_hash_sha256"),
        "runtime_config.full_spectrum_contract_hash_sha256",
    )
    manifest_hash = _validate_sha256(
        run_manifest.get("full_spectrum_contract_hash_sha256"),
        "run_manifest.full_spectrum_contract_hash_sha256",
    )
    if len({contract_hash, runtime_hash, manifest_hash}) != 1:
        raise MeasurementLogValidationError(
            "Full-spectrum contract hashes differ across runtime and run "
            "manifests."
        )
    manifest_contract_schema_version = _required_json_integer(
        run_manifest,
        "full_spectrum_contract_schema_version",
        location="run_manifest",
    )
    if manifest_contract_schema_version != contract_schema_version:
        raise MeasurementLogValidationError(
            "Full-spectrum contract schema differs across manifests."
        )
    numeric_fields = (
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
    )
    for field_name in numeric_fields:
        values = (
            _required_json_number(
                contract,
                field_name,
                location="runtime_config.full_spectrum_generative_model",
            ),
            _required_json_number(
                runtime_config,
                field_name,
                location="runtime_config",
            ),
            _required_json_number(
                run_manifest,
                field_name,
                location="run_manifest",
            ),
        )
        if not np.allclose(
            np.asarray(values, dtype=np.float64),
            values[0],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise MeasurementLogValidationError(
                f"Full-spectrum {field_name} differs across manifests."
            )
    bin_counts = (
        _required_json_integer(
            contract,
            "energy_bin_count",
            location="runtime_config.full_spectrum_generative_model",
            minimum=1,
        ),
        _required_json_integer(
            runtime_config,
            "energy_bin_count",
            location="runtime_config",
            minimum=1,
        ),
        _required_json_integer(
            run_manifest,
            "energy_bin_count",
            location="run_manifest",
            minimum=1,
        ),
    )
    if len(set(bin_counts)) != 1:
        raise MeasurementLogValidationError(
            "Full-spectrum energy_bin_count differs across manifests."
        )
    energy_min = _required_json_number(
        contract,
        "energy_min_keV",
        location="runtime_config.full_spectrum_generative_model",
    )
    energy_max = _required_json_number(
        contract,
        "energy_max_keV",
        location="runtime_config.full_spectrum_generative_model",
    )
    bin_width = _required_json_number(
        contract,
        "bin_width_keV",
        location="runtime_config.full_spectrum_generative_model",
    )
    expected_axis_max = energy_min + bin_width * (bin_counts[0] - 1)
    if not np.isclose(
        energy_max,
        expected_axis_max,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise MeasurementLogValidationError(
            "Full-spectrum axis bounds, width, and bin count are inconsistent."
        )
    expected_edges = (
        energy_min
        + bin_width * np.arange(bin_counts[0] + 1, dtype=np.float64)
    )
    for index, record in enumerate(records):
        record_hash = _validate_sha256(
            record.metadata.get(FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY),
            (
                f"records[{index}].metadata."
                f"{FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY}"
            ),
        )
        if record_hash != contract_hash:
            raise MeasurementLogValidationError(
                f"records[{index}] full-spectrum contract hash differs from "
                "the run contract."
            )
        if int(np.asarray(record.spectrum_counts).size) != bin_counts[0]:
            raise MeasurementLogValidationError(
                f"records[{index}] spectrum length differs from the "
                "full-spectrum contract."
            )
        if not np.array_equal(
            np.asarray(record.energy_bin_edges_keV, dtype=np.float64),
            expected_edges,
        ):
            raise MeasurementLogValidationError(
                f"records[{index}] energy-bin edges differ from the "
                "full-spectrum contract."
            )


def _records_to_arrays(
    records: Sequence[MeasurementLogRecord],
    isotopes: Sequence[str],
    *,
    validate_causal_sequence: bool = True,
) -> dict[str, NDArray[Any]]:
    """Pack records into dense, estimator-independent NumPy arrays."""
    if validate_causal_sequence:
        _validate_record_sequence(records, isotopes)
    elif not records:
        raise MeasurementLogValidationError(
            "A MeasurementLog shard needs one record."
        )
    energy_edges = np.asarray(records[0].energy_bin_edges_keV, dtype=np.float64)
    spectrum_size = energy_edges.size - 1
    for record in records:
        if not np.array_equal(
            np.asarray(record.energy_bin_edges_keV, dtype=np.float64),
            energy_edges,
        ):
            raise MeasurementLogValidationError(
                "Every schema-v2 record must use identical energy-bin edges."
            )
    count = len(records)
    spectra = np.stack(
        [np.asarray(record.spectrum_counts, dtype=np.int64) for record in records]
    )
    if spectra.shape != (count, spectrum_size):
        raise MeasurementLogValidationError("spectrum arrays have inconsistent shapes.")
    return {
        "step_id": np.asarray([record.step_id for record in records], dtype=np.int64),
        "action_id": np.asarray(
            [record.action_id for record in records], dtype=np.int64
        ),
        "station_id": np.asarray(
            [record.station_id for record in records], dtype=np.int64
        ),
        "detector_pose_xyz": np.asarray(
            [record.detector_pose_xyz for record in records], dtype=np.float64
        ),
        "detector_quat_wxyz": np.asarray(
            [record.detector_quat_wxyz for record in records], dtype=np.float64
        ),
        "fe_orientation_index": np.asarray(
            [record.fe_orientation_index for record in records], dtype=np.int64
        ),
        "pb_orientation_index": np.asarray(
            [record.pb_orientation_index for record in records], dtype=np.int64
        ),
        "live_time_s": np.asarray(
            [record.live_time_s for record in records], dtype=np.float64
        ),
        "travel_time_s": np.asarray(
            [record.travel_time_s for record in records], dtype=np.float64
        ),
        "shield_actuation_time_s": np.asarray(
            [record.shield_actuation_time_s for record in records], dtype=np.float64
        ),
        "energy_bin_edges_keV": energy_edges.astype(np.float64, copy=True),
        "spectrum_counts": spectra.astype(np.int64, copy=False),
    }


def _metadata_line(
    record: MeasurementLogRecord,
    *,
    run_id: str,
    record_index: int,
) -> dict[str, Any]:
    """Return one JSONL metadata row aligned to the observation arrays."""
    return {
        "run_id": str(run_id),
        "array_index": int(record_index),
        "step_id": int(record.step_id),
        "action_id": int(record.action_id),
        "station_id": int(record.station_id),
        "metadata": json_safe(dict(record.metadata)),
    }


def _write_measurement_log_unpublished(
    output_dir: str | Path,
    *,
    run_id: str,
    repository_commit: str,
    runtime_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    forward_model_manifest: Mapping[str, Any],
    isotopes: Sequence[str],
    records: Sequence[MeasurementLogRecord],
    metadata: Mapping[str, Any] | None = None,
    obstacle_layout_path: str | None = None,
    source_layout_path: str | None = None,
) -> MeasurementLog:
    """Write a complete canonical MeasurementLog v2 bundle and reload it."""
    _validate_source_layout_sentinel(
        source_layout_path,
        location="run_manifest.source_layout_path",
    )
    _validate_run_identity(run_id, repository_commit)
    if obstacle_layout_path is not None and not isinstance(
        obstacle_layout_path,
        str,
    ):
        raise MeasurementLogValidationError(
            "obstacle_layout_path must be a JSON string or null."
        )
    if not isinstance(metadata, (Mapping, type(None))):
        raise MeasurementLogValidationError("metadata must be an object or null.")
    _validate_truth_free_payload(runtime_config, location="runtime_config")
    _validate_truth_free_payload(environment, location="environment")
    _validate_truth_free_payload(metadata or {}, location="run_manifest.metadata")
    _validate_environment_payload(environment)
    isotope_names = _canonical_isotope_names(
        isotopes,
        location="isotopes",
    )
    _validate_runtime_replay_contract(
        runtime_config,
        isotopes=isotope_names,
    )
    full_spectrum_contract = runtime_config.get(
        "full_spectrum_generative_model"
    )
    if not isinstance(full_spectrum_contract, Mapping):
        raise MeasurementLogValidationError(
            "runtime_config.full_spectrum_generative_model must be an object."
        )
    _validate_record_sequence(records, isotope_names)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtime_path = root / "runtime_config.resolved.json"
    environment_path = root / "environment.json"
    forward_path = root / "forward_model_manifest.json"
    observations_path = root / "observations.npz"
    metadata_path = root / "observation_metadata.jsonl"
    commit_path = root / "repository_commit.txt"

    _write_json(runtime_path, runtime_config)
    _write_json(environment_path, environment)
    resolved_hash = _sha256_file(runtime_path)
    forward = dict(forward_model_manifest)
    forward["schema_version"] = FORWARD_MODEL_MANIFEST_SCHEMA_VERSION
    forward["repository_commit"] = repository_commit
    forward["resolved_config_sha256"] = resolved_hash
    forward["units"] = dict(_FORWARD_UNITS)
    forward["response_semantics"] = dict(_RESPONSE_SEMANTICS)
    validated_forward = validate_forward_model_manifest(
        forward,
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotope_names,
        repository_commit=repository_commit,
        resolved_config_sha256=resolved_hash,
        source_rate_model=SOURCE_RATE_MODEL,
        run_root=root,
        repository_root=_REPOSITORY_ROOT,
    )
    _write_json(forward_path, validated_forward)
    arrays = _records_to_arrays(records, isotope_names)
    _write_deterministic_npz(observations_path, arrays)
    metadata_bytes = b"".join(
        _json_line_bytes(_metadata_line(record, run_id=run_id, record_index=index))
        for index, record in enumerate(records)
    )
    metadata_path.write_bytes(metadata_bytes)
    commit_path.write_text(f"{repository_commit}\n", encoding="utf-8")

    artifact_paths = {
        path.name: path
        for path in (
            runtime_path,
            environment_path,
            forward_path,
            observations_path,
            metadata_path,
            commit_path,
        )
    }
    artifact_hashes = {
        name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
    }
    model_identifiers = validated_forward["model_identifiers"]
    run_manifest = {
        "schema_version": MEASUREMENT_LOG_SCHEMA_VERSION,
        "run_id": run_id,
        "record_count": int(len(records)),
        "repository_commit": repository_commit,
        "resolved_config_sha256": resolved_hash,
        "forward_model_manifest_sha256": artifact_hashes["forward_model_manifest.json"],
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": dict(_SOURCE_RATE_SEMANTICS),
        "isotopes": list(isotope_names),
        "environment": json_safe(dict(environment)),
        "obstacle_layout_path": obstacle_layout_path,
        "source_layout_path": source_layout_path,
        "sim_backend": runtime_config["sim_backend"],
        "observation_model": "joint_full_spectrum_generative",
        "energy_bin_count": int(np.asarray(records[0].spectrum_counts).size),
        "energy_min_keV": _required_json_number(
            runtime_config,
            "energy_min_keV",
            location="runtime_config",
        ),
        "energy_max_keV": _required_json_number(
            runtime_config,
            "energy_max_keV",
            location="runtime_config",
        ),
        "bin_width_keV": _required_json_number(
            runtime_config,
            "bin_width_keV",
            location="runtime_config",
        ),
        "full_spectrum_contract_hash_sha256": _validate_sha256(
            runtime_config.get("full_spectrum_contract_hash_sha256"),
            "runtime_config.full_spectrum_contract_hash_sha256",
        ),
        "full_spectrum_contract_schema_version": _required_json_integer(
            full_spectrum_contract,
            "schema_version",
            location="runtime_config.full_spectrum_generative_model",
        ),
        "model_identifiers": model_identifiers,
        "index_conventions": dict(_INDEX_CONVENTIONS),
        "artifact_hashes": artifact_hashes,
        "metadata": json_safe(dict(metadata or {})),
    }
    _validate_full_spectrum_contract_alignment(
        run_manifest=run_manifest,
        runtime_config=runtime_config,
        records=records,
    )
    _write_json(root / "run_manifest.json", run_manifest)
    return load_measurement_log(root)


def write_measurement_log(
    output_dir: str | Path,
    *,
    run_id: str,
    repository_commit: str,
    runtime_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    forward_model_manifest: Mapping[str, Any],
    isotopes: Sequence[str],
    records: Sequence[MeasurementLogRecord],
    metadata: Mapping[str, Any] | None = None,
    obstacle_layout_path: str | None = None,
    source_layout_path: str | None = None,
) -> MeasurementLog:
    """Atomically publish a canonical log and refuse to replace any prior run."""
    _validate_source_layout_sentinel(
        source_layout_path,
        location="run_manifest.source_layout_path",
    )
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Refusing to replace MeasurementLog directory {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary MeasurementLog path already exists: {temporary}."
        )
    try:
        _write_measurement_log_unpublished(
            temporary,
            run_id=run_id,
            repository_commit=repository_commit,
            runtime_config=runtime_config,
            environment=environment,
            forward_model_manifest=forward_model_manifest,
            isotopes=isotopes,
            records=records,
            metadata=metadata,
            obstacle_layout_path=obstacle_layout_path,
            source_layout_path=source_layout_path,
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_measurement_log(target)


def _parse_json_value(text: str, *, location: str) -> Any:
    """Parse strict JSON while rejecting duplicate keys and non-finite constants."""
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build one JSON object only when every member name is unique."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MeasurementLogValidationError(
                    f"{location} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    def _reject_constant(value: str) -> None:
        """Reject Python's non-standard NaN and infinity JSON extensions."""
        raise MeasurementLogValidationError(
            f"{location} contains non-finite JSON constant {value!r}."
        )

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one strict JSON object with a schema-focused error."""
    try:
        payload = _parse_json_value(
            path.read_text(encoding="utf-8"),
            location=path.name,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementLogValidationError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MeasurementLogValidationError(f"{path.name} must contain an object.")
    return payload


def _validate_run_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical schema-v2 manifest without legacy aliases."""
    if set(payload) != _RUN_MANIFEST_FIELDS:
        missing = sorted(_RUN_MANIFEST_FIELDS - set(payload))
        unknown = sorted(set(payload) - _RUN_MANIFEST_FIELDS)
        raise MeasurementLogValidationError(
            "run_manifest schema mismatch; "
            f"missing={missing}, unknown={unknown}."
        )
    if "source_layout_path" not in payload:
        raise MeasurementLogValidationError(
            "run_manifest requires null source_layout_path."
        )
    _validate_source_layout_sentinel(
        payload.get("source_layout_path"),
        location="run_manifest.source_layout_path",
    )
    _validate_truth_free_payload(
        {
            key: value
            for key, value in payload.items()
            if _normalized_contract_name(key) != "sourcelayoutpath"
        },
        location="run_manifest",
    )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MEASUREMENT_LOG_SCHEMA_VERSION
    ):
        raise MeasurementLogValidationError("run_manifest schema_version must be 2.")
    raw_run_id = payload.get("run_id")
    raw_repository = payload.get("repository_commit")
    if (
        not isinstance(raw_run_id, str)
        or not raw_run_id.strip()
        or not isinstance(raw_repository, str)
        or _GIT_COMMIT_PATTERN.fullmatch(raw_repository) is None
    ):
        raise MeasurementLogValidationError(
            "run_manifest requires a nonempty string run_id and a full "
            "lowercase Git repository_commit."
        )
    run_id = raw_run_id
    repository_value = raw_repository
    resolved_hash = _validate_sha256(
        payload.get("resolved_config_sha256"),
        "run_manifest.resolved_config_sha256",
    )
    forward_hash = _validate_sha256(
        payload.get("forward_model_manifest_sha256"),
        "run_manifest.forward_model_manifest_sha256",
    )
    if payload.get("source_rate_semantics") != _SOURCE_RATE_SEMANTICS:
        raise MeasurementLogValidationError(
            "run_manifest source_rate_semantics are incompatible."
        )
    if payload.get("source_rate_model") != SOURCE_RATE_MODEL:
        raise MeasurementLogValidationError(
            "run_manifest source_rate_model is incompatible."
        )
    for field_name in (
        "environment",
        "sim_backend",
        "observation_model",
        "isotopes",
        "record_count",
        "energy_bin_count",
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
        "full_spectrum_contract_hash_sha256",
        "full_spectrum_contract_schema_version",
    ):
        if field_name not in payload:
            raise MeasurementLogValidationError(f"run_manifest requires {field_name}.")
    if not isinstance(payload.get("environment"), Mapping):
        raise MeasurementLogValidationError(
            "run_manifest.environment must be an object."
        )
    obstacle_layout_path = payload.get("obstacle_layout_path")
    if obstacle_layout_path is not None and not isinstance(
        obstacle_layout_path,
        str,
    ):
        raise MeasurementLogValidationError(
            "run_manifest.obstacle_layout_path must be a JSON string or null."
        )
    if not isinstance(payload.get("metadata"), Mapping):
        raise MeasurementLogValidationError(
            "run_manifest.metadata must be an object."
        )
    sim_backend = payload.get("sim_backend")
    if not isinstance(sim_backend, str) or not sim_backend.strip():
        raise MeasurementLogValidationError(
            "run_manifest.sim_backend must be non-empty."
        )
    if payload.get("observation_model") != "joint_full_spectrum_generative":
        raise MeasurementLogValidationError(
            "run_manifest.observation_model must be "
            "'joint_full_spectrum_generative'."
        )
    _required_json_integer(
        payload,
        "energy_bin_count",
        location="run_manifest",
        minimum=1,
    )
    for field_name in ("energy_min_keV", "energy_max_keV", "bin_width_keV"):
        _required_json_number(
            payload,
            field_name,
            location="run_manifest",
        )
    if (
        float(payload["energy_max_keV"])
        <= float(payload["energy_min_keV"])
        or float(payload["bin_width_keV"]) <= 0.0
    ):
        raise MeasurementLogValidationError(
            "run_manifest spectrum-axis bounds are invalid."
        )
    contract_hash = payload["full_spectrum_contract_hash_sha256"]
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash)
    ):
        raise MeasurementLogValidationError(
            "run_manifest full-spectrum contract hash is invalid."
        )
    if _required_json_integer(
        payload,
        "full_spectrum_contract_schema_version",
        location="run_manifest",
    ) != FULL_SPECTRUM_MODEL_SCHEMA_VERSION:
        raise MeasurementLogValidationError(
            "run_manifest requires full-spectrum contract schema "
            f"{FULL_SPECTRUM_MODEL_SCHEMA_VERSION}."
        )
    _required_json_integer(
        payload,
        "record_count",
        location="run_manifest",
        minimum=1,
    )
    raw_isotopes = payload.get("isotopes")
    if (
        not isinstance(raw_isotopes, list)
        or not raw_isotopes
        or not all(isinstance(value, str) and value for value in raw_isotopes)
        or len(set(raw_isotopes)) != len(raw_isotopes)
        or raw_isotopes != sorted(raw_isotopes)
    ):
        raise MeasurementLogValidationError(
            "run_manifest.isotopes must be a non-empty unique string array."
        )
    models = _validate_model_identifiers(payload.get("model_identifiers"))
    conventions = payload.get("index_conventions")
    if conventions != _INDEX_CONVENTIONS:
        raise MeasurementLogValidationError(
            "index_conventions do not match schema-v2 causal index semantics."
        )
    artifact_hashes = payload.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise MeasurementLogValidationError("artifact_hashes must be an object.")
    if any(not isinstance(name, str) or not name for name in artifact_hashes):
        raise MeasurementLogValidationError(
            "artifact_hashes keys must be nonempty JSON strings."
        )
    normalized_artifacts = {
        name: _validate_sha256(value, f"artifact_hashes.{name}")
        for name, value in artifact_hashes.items()
    }
    result = dict(payload)
    result.update(
        {
            "run_id": run_id,
            "repository_commit": repository_value,
            "resolved_config_sha256": resolved_hash,
            "forward_model_manifest_sha256": forward_hash,
            "model_identifiers": models,
            "artifact_hashes": normalized_artifacts,
        }
    )
    return result


def _required_array(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type,
) -> NDArray[Any]:
    """Return a required observation array after exact dtype/shape validation."""
    value = np.asarray(arrays[name])
    expected_dtype = np.dtype(dtype)
    if value.shape != shape:
        raise MeasurementLogValidationError(
            f"observations.npz {name} has shape {value.shape}; expected {shape}."
        )
    if value.dtype != expected_dtype:
        raise MeasurementLogValidationError(
            f"observations.npz {name} has dtype {value.dtype}; "
            f"expected {expected_dtype}."
        )
    return np.array(value, copy=True)


def _records_from_arrays(
    arrays: Mapping[str, NDArray[Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    isotopes: Sequence[str],
    *,
    run_id: str,
    record_count: int,
    energy_bin_count: int,
) -> tuple[MeasurementLogRecord, ...]:
    """Reconstruct ordered immutable records from validated dense arrays."""
    del isotopes
    required = {
        "step_id",
        "action_id",
        "station_id",
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "fe_orientation_index",
        "pb_orientation_index",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "energy_bin_edges_keV",
        "spectrum_counts",
    }
    missing = sorted(required - set(arrays))
    extra = sorted(set(arrays) - required)
    if missing or extra:
        raise MeasurementLogValidationError(
            f"observations.npz schema mismatch; missing={missing}, extra={extra}."
        )
    if isinstance(record_count, bool) or not isinstance(
        record_count,
        (int, np.integer),
    ):
        raise MeasurementLogValidationError(
            "record_count must be an integer."
        )
    if isinstance(energy_bin_count, bool) or not isinstance(
        energy_bin_count,
        (int, np.integer),
    ):
        raise MeasurementLogValidationError(
            "energy_bin_count must be an integer."
        )
    row_count = int(record_count)
    bin_count = int(energy_bin_count)
    if row_count <= 0 or bin_count <= 0:
        raise MeasurementLogValidationError(
            "record_count and energy_bin_count must be positive."
        )
    if len(metadata_rows) != row_count:
        raise MeasurementLogValidationError(
            "observation_metadata.jsonl row count does not match observations.npz."
        )
    typed_arrays: dict[str, NDArray[Any]] = {
        "step_id": _required_array(
            arrays, "step_id", shape=(row_count,), dtype=np.int64
        ),
        "action_id": _required_array(
            arrays, "action_id", shape=(row_count,), dtype=np.int64
        ),
        "station_id": _required_array(
            arrays, "station_id", shape=(row_count,), dtype=np.int64
        ),
        "detector_pose_xyz": _required_array(
            arrays,
            "detector_pose_xyz",
            shape=(row_count, 3),
            dtype=np.float64,
        ),
        "detector_quat_wxyz": _required_array(
            arrays,
            "detector_quat_wxyz",
            shape=(row_count, 4),
            dtype=np.float64,
        ),
        "fe_orientation_index": _required_array(
            arrays,
            "fe_orientation_index",
            shape=(row_count,),
            dtype=np.int64,
        ),
        "pb_orientation_index": _required_array(
            arrays,
            "pb_orientation_index",
            shape=(row_count,),
            dtype=np.int64,
        ),
        "live_time_s": _required_array(
            arrays, "live_time_s", shape=(row_count,), dtype=np.float64
        ),
        "travel_time_s": _required_array(
            arrays, "travel_time_s", shape=(row_count,), dtype=np.float64
        ),
        "shield_actuation_time_s": _required_array(
            arrays,
            "shield_actuation_time_s",
            shape=(row_count,),
            dtype=np.float64,
        ),
        "energy_bin_edges_keV": _required_array(
            arrays,
            "energy_bin_edges_keV",
            shape=(bin_count + 1,),
            dtype=np.float64,
        ),
        "spectrum_counts": _required_array(
            arrays,
            "spectrum_counts",
            shape=(row_count, bin_count),
            dtype=np.int64,
        ),
    }
    arrays = typed_arrays
    step_ids = arrays["step_id"]
    edges = arrays["energy_bin_edges_keV"]
    spectra = arrays["spectrum_counts"]
    if np.any(spectra < 0):
        raise MeasurementLogValidationError(
            "observations.npz spectrum_counts must contain nonnegative "
            "unit-weight integer event counts."
        )
    records: list[MeasurementLogRecord] = []
    for row in range(row_count):
        metadata_row = metadata_rows[row]
        if set(metadata_row) != {
            "run_id",
            "array_index",
            "step_id",
            "action_id",
            "station_id",
            "metadata",
        }:
            raise MeasurementLogValidationError(
                "Metadata rows must contain exactly the canonical schema fields."
            )
        if metadata_row.get("run_id") != run_id:
            raise MeasurementLogValidationError(
                "Metadata run_id does not match run_manifest."
            )
        if not isinstance(metadata_row.get("metadata"), Mapping):
            raise MeasurementLogValidationError("Metadata payload must be an object.")
        if _required_json_integer(
            metadata_row,
            "array_index",
            location=f"metadata_rows[{row}]",
            minimum=0,
        ) != row:
            raise MeasurementLogValidationError(
                "metadata array_index must equal zero-based row order."
            )
        for identifier in ("step_id", "action_id", "station_id"):
            if _required_json_integer(
                metadata_row,
                identifier,
                location=f"metadata_rows[{row}]",
                minimum=0,
            ) != int(
                np.asarray(arrays[identifier])[row]
            ):
                raise MeasurementLogValidationError(
                    f"metadata {identifier} disagrees with observations.npz."
                )
        records.append(
            MeasurementLogRecord(
                step_id=int(step_ids[row]),
                action_id=int(np.asarray(arrays["action_id"])[row]),
                station_id=int(np.asarray(arrays["station_id"])[row]),
                detector_pose_xyz=tuple(
                    float(value)
                    for value in np.asarray(arrays["detector_pose_xyz"])[row]
                ),
                detector_quat_wxyz=tuple(
                    float(value)
                    for value in np.asarray(arrays["detector_quat_wxyz"])[row]
                ),
                fe_orientation_index=int(
                    np.asarray(arrays["fe_orientation_index"])[row]
                ),
                pb_orientation_index=int(
                    np.asarray(arrays["pb_orientation_index"])[row]
                ),
                live_time_s=float(np.asarray(arrays["live_time_s"])[row]),
                travel_time_s=float(np.asarray(arrays["travel_time_s"])[row]),
                shield_actuation_time_s=float(
                    np.asarray(arrays["shield_actuation_time_s"])[row]
                ),
                energy_bin_edges_keV=edges.copy(),
                spectrum_counts=np.asarray(
                    arrays["spectrum_counts"],
                    dtype=np.int64,
                )[row].copy(),
                metadata=dict(metadata_row.get("metadata", {})),
            )
        )
    return tuple(records)


def load_measurement_log(path: str | Path) -> MeasurementLog:
    """Load and fully validate a MeasurementLog without reading truth artifacts."""
    supplied = Path(path)
    root = supplied.parent if supplied.name == "run_manifest.json" else supplied
    root = root.resolve()
    if not root.is_dir():
        raise MeasurementLogValidationError(
            f"MeasurementLog directory does not exist: {root}."
        )
    # Reject truth/symlinks before parsing any estimator input artifact.
    measurement_log_sha256(root)
    for filename in _CANONICAL_REQUIRED_FILES:
        if not (root / filename).is_file():
            raise MeasurementLogValidationError(
                f"Missing MeasurementLog file {filename}."
            )
    commit_path = root / "repository_commit.txt"

    manifest = _validate_run_manifest(_load_json_object(root / "run_manifest.json"))
    runtime_config = _load_json_object(root / "runtime_config.resolved.json")
    environment = _load_json_object(root / "environment.json")
    _validate_truth_free_payload(runtime_config, location="runtime_config")
    _validate_truth_free_payload(environment, location="environment")
    _validate_environment_payload(environment)
    manifest_isotopes = _canonical_isotope_names(
        manifest["isotopes"],
        location="run_manifest.isotopes",
    )
    _validate_runtime_replay_contract(
        runtime_config,
        isotopes=manifest_isotopes,
    )
    if manifest["environment"] != environment:
        raise MeasurementLogValidationError(
            "environment.json does not match run_manifest."
        )
    forward = validate_forward_model_manifest(
        _load_json_object(root / "forward_model_manifest.json"),
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=(
            None
            if manifest.get("obstacle_layout_path") is None
            else manifest["obstacle_layout_path"]
        ),
        isotopes=manifest_isotopes,
        repository_commit=manifest["repository_commit"],
        resolved_config_sha256=manifest["resolved_config_sha256"],
        source_rate_model=manifest["source_rate_model"],
        run_root=root,
        repository_root=_REPOSITORY_ROOT,
    )
    if (
        _sha256_file(root / "runtime_config.resolved.json")
        != manifest["resolved_config_sha256"]
    ):
        raise MeasurementLogValidationError("Resolved runtime config hash mismatch.")
    if (
        _sha256_file(root / "forward_model_manifest.json")
        != manifest["forward_model_manifest_sha256"]
    ):
        raise MeasurementLogValidationError("Forward-model manifest hash mismatch.")
    if forward["model_identifiers"] != manifest["model_identifiers"]:
        raise MeasurementLogValidationError(
            "Run and forward-model manifests identify different physical models."
        )
    declared_artifacts = dict(manifest["artifact_hashes"])
    actual_artifacts = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "run_manifest.json"
    }
    required_artifacts = set(_CANONICAL_REQUIRED_FILES) - {"run_manifest.json"}
    if not required_artifacts.issubset(actual_artifacts):
        raise MeasurementLogValidationError(
            "MeasurementLog is missing a required hashed artifact."
        )
    if set(declared_artifacts) != actual_artifacts:
        raise MeasurementLogValidationError(
            "artifact_hashes must name every estimator-input artifact."
        )
    for filename, expected_hash in declared_artifacts.items():
        artifact_path = root / filename
        if not artifact_path.is_file():
            raise MeasurementLogValidationError(
                f"Declared artifact {filename} is missing."
            )
        if _sha256_file(artifact_path) != expected_hash:
            raise MeasurementLogValidationError(
                f"Artifact hash mismatch for {filename}."
            )
    commit_value = commit_path.read_text(encoding="utf-8").strip()
    if commit_value != manifest["repository_commit"]:
        raise MeasurementLogValidationError(
            "repository_commit.txt does not match run_manifest."
        )

    metadata_rows: list[dict[str, Any]] = []
    with (root / "observation_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = _parse_json_value(
                    line,
                    location=f"observation_metadata.jsonl line {line_number}",
                )
            except json.JSONDecodeError as exc:
                raise MeasurementLogValidationError(
                    f"Invalid metadata JSON on line {line_number}."
                ) from exc
            if not isinstance(row, dict):
                raise MeasurementLogValidationError(
                    f"Metadata line {line_number} must be an object."
                )
            metadata_rows.append(row)
    try:
        with np.load(root / "observations.npz", allow_pickle=False) as loaded:
            if len(loaded.files) != len(set(loaded.files)):
                raise MeasurementLogValidationError(
                    "observations.npz contains duplicate array names."
                )
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except MeasurementLogValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise MeasurementLogValidationError(
            "Cannot read a valid observations.npz archive."
        ) from exc
    records = _records_from_arrays(
        arrays,
        metadata_rows,
        manifest_isotopes,
        run_id=manifest["run_id"],
        record_count=manifest["record_count"],
        energy_bin_count=manifest["energy_bin_count"],
    )
    _validate_record_sequence(records, manifest_isotopes)
    _validate_full_spectrum_contract_alignment(
        run_manifest=manifest,
        runtime_config=runtime_config,
        records=records,
    )
    return MeasurementLog(
        run_manifest=manifest,
        runtime_config=runtime_config,
        environment=environment,
        forward_model_manifest=forward,
        records=records,
        path=root,
    )


class MeasurementLogStreamWriter:
    """Persist every observation before PF ingestion, then finalize one bundle."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        repository_commit: str,
        runtime_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        forward_model_manifest: Mapping[str, Any],
        isotopes: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
        obstacle_layout_path: str | None = None,
        source_layout_path: str | None = None,
    ) -> None:
        """Initialize durable per-record staging for a live acquisition."""
        _validate_source_layout_sentinel(
            source_layout_path,
            location="run_manifest.source_layout_path",
        )
        _validate_run_identity(run_id, repository_commit)
        isotope_names = _canonical_isotope_names(
            isotopes,
            location="isotopes",
        )
        _validate_runtime_replay_contract(
            runtime_config,
            isotopes=isotope_names,
        )
        if obstacle_layout_path is not None and not isinstance(
            obstacle_layout_path,
            str,
        ):
            raise MeasurementLogValidationError(
                "obstacle_layout_path must be a JSON string or null."
            )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise MeasurementLogValidationError(
                "metadata must be an object or null."
            )
        _validate_truth_free_payload(runtime_config, location="runtime_config")
        _validate_truth_free_payload(environment, location="environment")
        _validate_truth_free_payload(metadata or {}, location="run_manifest.metadata")
        _validate_environment_payload(environment)
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(
                f"Refusing to replace MeasurementLog directory {self.output_dir}."
            )
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.stage_dir = self.output_dir.with_name(
            f".{self.output_dir.name}.stream-{os.getpid()}"
        )
        if self.stage_dir.exists():
            raise FileExistsError(
                f"MeasurementLog staging path exists: {self.stage_dir}."
            )
        self.stage_dir.mkdir(parents=True)
        self.metadata_stage_path = self.stage_dir / "observation_metadata.jsonl"
        self.run_id = run_id
        self.repository_commit = repository_commit
        self.runtime_config = dict(runtime_config)
        self.environment = dict(environment)
        self.forward_model_manifest = dict(forward_model_manifest)
        self.isotopes = isotope_names
        self.metadata = dict(metadata or {})
        self.obstacle_layout_path = obstacle_layout_path
        self.source_layout_path = source_layout_path
        self.resume_record_metadata: dict[str, Any] = {}
        self.records: list[MeasurementLogRecord] = []
        self.metadata_stage_path.write_text("", encoding="utf-8")
        _write_json(
            self.stage_dir / "runtime_config.resolved.json", self.runtime_config
        )
        _write_json(self.stage_dir / "environment.json", self.environment)
        _write_json(
            self.stage_dir / "forward_model_manifest.input.json",
            self.forward_model_manifest,
        )
        (self.stage_dir / "repository_commit.txt").write_text(
            f"{self.repository_commit}\n",
            encoding="utf-8",
        )

    @classmethod
    def resume_from_stage(
        cls,
        output_dir: str | Path,
        *,
        stage_dir: str | Path,
        run_id: str,
        repository_commit: str,
        runtime_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        forward_model_manifest: Mapping[str, Any],
        isotopes: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
        obstacle_layout_path: str | None = None,
        source_layout_path: str | None = None,
        resume_execution_commit: str,
        resume_compatibility: Mapping[str, Any],
    ) -> "MeasurementLogStreamWriter":
        """Copy a verified station prefix, dropping only one incomplete WAL tail."""
        _validate_source_layout_sentinel(
            source_layout_path,
            location="run_manifest.source_layout_path",
        )
        _validate_run_identity(run_id, repository_commit)
        if (
            not isinstance(resume_execution_commit, str)
            or _GIT_COMMIT_PATTERN.fullmatch(resume_execution_commit) is None
        ):
            raise MeasurementLogValidationError(
                "resume_execution_commit must be a full lowercase Git hash."
            )
        isotope_names = _canonical_isotope_names(
            isotopes,
            location="isotopes",
        )
        _validate_runtime_replay_contract(
            runtime_config,
            isotopes=isotope_names,
        )
        if obstacle_layout_path is not None and not isinstance(
            obstacle_layout_path,
            str,
        ):
            raise MeasurementLogValidationError(
                "obstacle_layout_path must be a JSON string or null."
            )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise MeasurementLogValidationError(
                "metadata must be an object or null."
            )
        if not isinstance(resume_compatibility, Mapping):
            raise MeasurementLogValidationError(
                "resume_compatibility must be an object."
            )
        _validate_truth_free_payload(runtime_config, location="runtime_config")
        _validate_truth_free_payload(environment, location="environment")
        _validate_truth_free_payload(metadata or {}, location="run_manifest.metadata")
        _validate_environment_payload(environment)

        target = Path(output_dir).resolve()
        stage_input = Path(stage_dir)
        if stage_input.is_symlink():
            raise MeasurementLogValidationError(
                "The MeasurementLog resume stage must not be a symlink."
            )
        stage = stage_input.resolve()
        if target.exists():
            raise FileExistsError(
                f"Refusing to replace MeasurementLog directory {target}."
            )
        expected_stage_pattern = re.compile(
            rf"^\.{re.escape(target.name)}\.stream-[0-9]+$"
        )
        if stage.parent != target.parent or expected_stage_pattern.fullmatch(
            stage.name
        ) is None:
            raise MeasurementLogValidationError(
                "The resume stage must be the hidden stream directory belonging "
                "to the requested MeasurementLog output."
            )
        if not stage.is_dir() or stage.is_symlink():
            raise MeasurementLogValidationError(
                f"MeasurementLog stream stage is not a real directory: {stage}."
            )

        entries = list(stage.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise MeasurementLogValidationError(
                "A resumable MeasurementLog stage may contain regular files only."
            )
        entry_names = {entry.name for entry in entries}
        shard_entries: list[tuple[int, Path]] = []
        for entry in entries:
            match = _STREAM_RECORD_PATTERN.fullmatch(entry.name)
            if match is not None:
                shard_entries.append((int(match.group(1)), entry))
        shard_entries.sort(key=lambda item: item[0])
        expected_indices = list(range(len(shard_entries)))
        if [index for index, _ in shard_entries] != expected_indices:
            raise MeasurementLogValidationError(
                "MeasurementLog stream shards must be contiguous from record zero."
            )
        metadata_temp_entries = tuple(
            entry
            for entry in entries
            if _STREAM_METADATA_TEMP_PATTERN.fullmatch(entry.name) is not None
        )
        if len(metadata_temp_entries) > 1:
            raise MeasurementLogValidationError(
                "A resumable MeasurementLog stage has multiple metadata "
                "rewrite orphans."
            )
        expected_inventory = (
            _STREAM_STATIC_FILES
            | {path.name for _, path in shard_entries}
            | {path.name for path in metadata_temp_entries}
        )
        if entry_names != expected_inventory:
            missing = sorted(expected_inventory - entry_names)
            extra = sorted(entry_names - expected_inventory)
            raise MeasurementLogValidationError(
                "MeasurementLog stream inventory mismatch; "
                f"missing={missing}, extra={extra}."
            )
        if not shard_entries:
            raise MeasurementLogValidationError(
                "A resumable MeasurementLog stage needs at least one record."
            )

        staged_runtime = _load_json_object(
            stage / "runtime_config.resolved.json"
        )
        staged_environment = _load_json_object(stage / "environment.json")
        staged_forward = _load_json_object(
            stage / "forward_model_manifest.input.json"
        )
        identity_pairs = (
            ("runtime configuration", staged_runtime, dict(runtime_config)),
            ("environment", staged_environment, dict(environment)),
            (
                "forward-model manifest",
                staged_forward,
                dict(forward_model_manifest),
            ),
        )
        for name, staged_value, expected_value in identity_pairs:
            if canonical_json_bytes(staged_value) != canonical_json_bytes(
                expected_value
            ):
                raise MeasurementLogValidationError(
                    f"Resume {name} does not match the staged acquisition."
                )
        commit_bytes = (stage / "repository_commit.txt").read_bytes()
        expected_commit_bytes = f"{repository_commit}\n".encode("utf-8")
        if commit_bytes != expected_commit_bytes:
            raise MeasurementLogValidationError(
                "Resume repository commit does not match the staged acquisition."
            )
        compatibility_payload = dict(resume_compatibility)
        _validate_truth_free_payload(
            compatibility_payload,
            location="run_manifest.metadata.resume_compatibility",
        )
        if (
            compatibility_payload.get("prefix_repository_commit")
            != repository_commit
            or compatibility_payload.get("resume_execution_commit")
            != resume_execution_commit
        ):
            raise MeasurementLogValidationError(
                "Resume compatibility provenance identifies different commits."
            )
        metadata_path = stage / "observation_metadata.jsonl"
        compatibility_payload["prefix_identity_sha256"] = {
            filename: _sha256_file(stage / filename)
            for filename in (
                "runtime_config.resolved.json",
                "environment.json",
                "forward_model_manifest.input.json",
                "repository_commit.txt",
            )
        }
        source_shard_hashes = {
            path.name: _sha256_file(path) for _, path in shard_entries
        }
        source_inventory_hashes = {
            entry.name: _sha256_file(entry)
            for entry in sorted(entries, key=lambda value: value.name)
        }
        compatibility_payload["source_stage_inventory_sha256"] = _sha256_bytes(
            canonical_json_bytes(source_inventory_hashes)
        )
        compatibility_payload["source_stage_record_shards_sha256"] = _sha256_bytes(
            canonical_json_bytes(source_shard_hashes)
        )
        compatibility_payload["source_stage_metadata_sha256"] = _sha256_file(
            metadata_path
        )

        configured_isotopes = staged_runtime.get("candidate_isotopes")
        if configured_isotopes is not None and (
            not isinstance(configured_isotopes, list)
            or any(not isinstance(value, str) for value in configured_isotopes)
            or tuple(configured_isotopes) != isotope_names
        ):
            raise MeasurementLogValidationError(
                "Resume isotope order does not match candidate_isotopes in the "
                "staged runtime configuration."
            )
        line_table = staged_forward.get("line_mu_by_isotope")
        if not isinstance(line_table, Mapping) or set(line_table) != set(
            isotope_names
        ):
            raise MeasurementLogValidationError(
                "Resume isotopes do not match the staged forward model."
            )

        metadata_rows: list[dict[str, Any]] = []
        truncated_metadata_tail = False
        try:
            metadata_bytes = metadata_path.read_bytes()
            metadata_lines = metadata_bytes.splitlines(keepends=True)
            for line_number, line_bytes in enumerate(metadata_lines, start=1):
                if not line_bytes.endswith(b"\n"):
                    if line_number != len(metadata_lines):
                        raise MeasurementLogValidationError(
                            "Only the final staged metadata line may be truncated."
                        )
                    truncated_metadata_tail = True
                    break
                try:
                    line = line_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise MeasurementLogValidationError(
                        "The staged metadata JSONL is not valid UTF-8."
                    ) from exc
                try:
                    row = _parse_json_value(
                        line,
                        location=(
                            "staged observation_metadata.jsonl line "
                            f"{line_number}"
                        ),
                    )
                except json.JSONDecodeError as exc:
                    raise MeasurementLogValidationError(
                        f"Invalid staged metadata JSON on line {line_number}."
                    ) from exc
                if not isinstance(row, dict):
                    raise MeasurementLogValidationError(
                        f"Metadata line {line_number} must be an object."
                    )
                metadata_rows.append(row)
        except OSError as exc:
            raise MeasurementLogValidationError(
                "Cannot read the staged metadata JSONL."
            ) from exc
        if len(metadata_rows) > len(shard_entries):
            raise MeasurementLogValidationError(
                "Staged metadata cannot outnumber record shards."
            )
        orphan_shard_count = len(shard_entries) - len(metadata_rows)
        if orphan_shard_count > 1:
            raise MeasurementLogValidationError(
                "At most one shard may follow the durable metadata tail."
            )
        if truncated_metadata_tail and orphan_shard_count != 1:
            raise MeasurementLogValidationError(
                "A truncated final metadata line is recoverable only when its "
                "corresponding record shard is durably present."
            )

        recovered_records: list[MeasurementLogRecord] = []
        for record_index, shard_path in shard_entries[: len(metadata_rows)]:
            metadata_row = metadata_rows[record_index]
            if set(metadata_row) != {
                "run_id",
                "array_index",
                "step_id",
                "action_id",
                "station_id",
                "metadata",
            }:
                raise MeasurementLogValidationError(
                    "Staged metadata rows must contain exactly the canonical "
                    "schema fields."
                )
            if metadata_row.get("run_id") != run_id:
                raise MeasurementLogValidationError(
                    "Resume run_id does not match the staged acquisition."
                )
            if _required_json_integer(
                metadata_row,
                "array_index",
                location=f"metadata_rows[{record_index}]",
                minimum=0,
            ) != record_index:
                raise MeasurementLogValidationError(
                    "Staged metadata array_index must match its record shard."
                )
            try:
                with np.load(shard_path, allow_pickle=False) as loaded:
                    if len(loaded.files) != len(set(loaded.files)):
                        raise MeasurementLogValidationError(
                            f"{shard_path.name} contains duplicate array names."
                        )
                    arrays = {
                        name: np.array(loaded[name], copy=True)
                        for name in loaded.files
                    }
            except MeasurementLogValidationError:
                raise
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise MeasurementLogValidationError(
                    f"Cannot read valid staged shard {shard_path.name}."
                ) from exc
            edges = np.asarray(arrays.get("energy_bin_edges_keV", ()))
            local_metadata = dict(metadata_row)
            local_metadata["array_index"] = 0
            record = _records_from_arrays(
                arrays,
                (local_metadata,),
                isotope_names,
                run_id=run_id,
                record_count=1,
                energy_bin_count=int(edges.size - 1),
            )[0]
            recovered_records.append(record)

        if not recovered_records:
            raise MeasurementLogValidationError(
                "A resumable MeasurementLog stage has no durable metadata records."
            )
        _validate_record_sequence(recovered_records, isotope_names)
        for record_index, record in enumerate(recovered_records):
            if int(record.step_id) != record_index or int(record.action_id) != (
                record_index
            ):
                raise MeasurementLogValidationError(
                    "Live resume requires zero-based step_id and action_id to "
                    "equal record order."
                )
        station_ids = [int(record.station_id) for record in recovered_records]
        if sorted(set(station_ids)) != list(range(station_ids[-1] + 1)):
            raise MeasurementLogValidationError(
                "Live resume requires contiguous zero-based station identifiers."
            )
        for record_index, record in enumerate(recovered_records):
            _station_complete_marker(
                record.metadata,
                location=f"records[{record_index}].metadata",
            )
            station_start = record_index == 0 or (
                station_ids[record_index - 1] != station_ids[record_index]
            )
            if not station_start:
                previous = recovered_records[record_index - 1]
                if (
                    record.detector_pose_xyz != previous.detector_pose_xyz
                    or record.detector_quat_wxyz != previous.detector_quat_wxyz
                ):
                    raise MeasurementLogValidationError(
                        "All records in a resumable station must share one "
                        "detector pose."
                    )

        completed_indices = [
            index
            for index, record in enumerate(recovered_records)
            if _station_complete_marker(
                record.metadata,
                location=f"records[{index}].metadata",
            )
        ]
        if not completed_indices:
            raise MeasurementLogValidationError(
                "Live resume requires at least one durable station_complete boundary."
            )
        prefix_record_count = int(completed_indices[-1] + 1)
        complete_records = recovered_records[:prefix_record_count]
        complete_station_ids = station_ids[:prefix_record_count]
        for record_index, record in enumerate(complete_records):
            station_end = (
                record_index + 1 == len(complete_records)
                or complete_station_ids[record_index + 1]
                != complete_station_ids[record_index]
            )
            marker = _station_complete_marker(
                record.metadata,
                location=f"records[{record_index}].metadata",
            )
            if marker is not station_end:
                raise MeasurementLogValidationError(
                    "The committed prefix must have exactly one station_complete "
                    "marker on each station's final record."
                )

        tail_records = recovered_records[prefix_record_count:]
        if any(
            _station_complete_marker(
                record.metadata,
                location=(
                    f"records[{prefix_record_count + index}].metadata"
                ),
            )
            for index, record in enumerate(tail_records)
        ):
            raise MeasurementLogValidationError(
                "A station_complete marker cannot appear after the committed "
                "resume boundary."
            )
        if tail_records:
            expected_tail_station = int(complete_records[-1].station_id) + 1
            if any(
                int(record.station_id) != expected_tail_station
                for record in tail_records
            ):
                raise MeasurementLogValidationError(
                    "Discardable resume tail records must belong to exactly the "
                    "next incomplete station."
                )

        discarded_record_count = len(shard_entries) - prefix_record_count

        prefix_metadata_bytes = b"".join(
            _json_line_bytes(
                _metadata_line(
                    record,
                    run_id=run_id,
                    record_index=index,
                )
            )
            for index, record in enumerate(complete_records)
        )
        prefix_shard_entries = shard_entries[:prefix_record_count]
        prefix_shard_hashes = {
            path.name: source_shard_hashes[path.name]
            for _, path in prefix_shard_entries
        }
        compatibility_payload["prefix_record_shards_sha256"] = _sha256_bytes(
            canonical_json_bytes(prefix_shard_hashes)
        )
        compatibility_payload["prefix_metadata_sha256"] = _sha256_bytes(
            prefix_metadata_bytes
        )
        compatibility_payload["source_stage_recovery"] = {
            "source_stage_name": stage.name,
            "source_record_shard_count": int(len(shard_entries)),
            "source_metadata_record_count": int(len(metadata_rows)),
            "committed_prefix_record_count": int(prefix_record_count),
            "discarded_tail_record_count": int(discarded_record_count),
            "orphan_shard_count": int(orphan_shard_count),
            "truncated_metadata_tail": bool(truncated_metadata_tail),
            "metadata_temp_orphan_count": int(len(metadata_temp_entries)),
            "copy_on_adopt": True,
        }

        fork_stage: Path | None = None
        for attempt in range(1000):
            suffix = (
                str(os.getpid())
                if attempt == 0
                else f"{os.getpid()}{attempt:03d}"
            )
            candidate = target.with_name(f".{target.name}.stream-{suffix}")
            if candidate == stage:
                continue
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            fork_stage = candidate
            break
        if fork_stage is None:
            raise FileExistsError(
                "Cannot allocate a new copy-on-resume MeasurementLog stage."
            )
        fork_metadata_path = fork_stage / "observation_metadata.jsonl"
        try:
            for filename in _STREAM_STATIC_FILES - {
                "observation_metadata.jsonl"
            }:
                shutil.copyfile(stage / filename, fork_stage / filename)
            for _, shard_path in prefix_shard_entries:
                shutil.copyfile(shard_path, fork_stage / shard_path.name)
            with fork_metadata_path.open("wb") as handle:
                handle.write(prefix_metadata_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            for copied_path in fork_stage.iterdir():
                if copied_path == fork_metadata_path:
                    continue
                with copied_path.open("rb") as handle:
                    os.fsync(handle.fileno())
            directory_fd = os.open(fork_stage, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            current_source_entries = list(stage.iterdir())
            if any(
                entry.is_symlink() or not entry.is_file()
                for entry in current_source_entries
            ):
                raise MeasurementLogValidationError(
                    "The source MeasurementLog stage changed during adoption."
                )
            current_source_hashes = {
                entry.name: _sha256_file(entry)
                for entry in sorted(
                    current_source_entries,
                    key=lambda value: value.name,
                )
            }
            if current_source_hashes != source_inventory_hashes:
                raise MeasurementLogValidationError(
                    "The source MeasurementLog stage changed during adoption."
                )

            expected_fork_hashes = {
                filename: source_inventory_hashes[filename]
                for filename in _STREAM_STATIC_FILES
                if filename != "observation_metadata.jsonl"
            }
            expected_fork_hashes.update(prefix_shard_hashes)
            expected_fork_hashes["observation_metadata.jsonl"] = _sha256_bytes(
                prefix_metadata_bytes
            )
            actual_fork_entries = list(fork_stage.iterdir())
            if any(
                entry.is_symlink() or not entry.is_file()
                for entry in actual_fork_entries
            ):
                raise MeasurementLogValidationError(
                    "The copied MeasurementLog stage has an invalid inventory."
                )
            actual_fork_hashes = {
                entry.name: _sha256_file(entry)
                for entry in sorted(actual_fork_entries, key=lambda value: value.name)
            }
            if actual_fork_hashes != expected_fork_hashes:
                raise MeasurementLogValidationError(
                    "The copied MeasurementLog stage failed hash verification."
                )
            parent_directory_fd = os.open(fork_stage.parent, os.O_RDONLY)
            try:
                os.fsync(parent_directory_fd)
            finally:
                os.close(parent_directory_fd)
        except Exception:
            shutil.rmtree(fork_stage, ignore_errors=True)
            raise

        instance = cls.__new__(cls)
        instance.output_dir = target
        instance.stage_dir = fork_stage
        instance.metadata_stage_path = fork_metadata_path
        instance.run_id = run_id
        instance.repository_commit = repository_commit
        instance.runtime_config = dict(runtime_config)
        instance.environment = dict(environment)
        instance.forward_model_manifest = dict(forward_model_manifest)
        instance.isotopes = isotope_names
        instance.metadata = {
            **dict(metadata or {}),
            "resume_prefix_repository_commit": repository_commit,
            "resume_execution_commit": resume_execution_commit,
            "resume_prefix_record_count": int(prefix_record_count),
            "resume_compatibility": compatibility_payload,
        }
        instance.obstacle_layout_path = obstacle_layout_path
        instance.source_layout_path = source_layout_path
        instance.resume_record_metadata = {
            "resume_execution_commit": resume_execution_commit,
            "resume_prefix_record_count": int(prefix_record_count),
        }
        instance.records = list(complete_records)
        return instance

    def write_canonical_prefix(self, output_dir: str | Path) -> MeasurementLog:
        """Write the currently staged prefix as an independently hashed bundle."""
        return write_measurement_log(
            output_dir,
            run_id=self.run_id,
            repository_commit=self.repository_commit,
            runtime_config=self.runtime_config,
            environment=self.environment,
            forward_model_manifest=self.forward_model_manifest,
            isotopes=self.isotopes,
            records=self.records,
            metadata=self.metadata,
            obstacle_layout_path=self.obstacle_layout_path,
            source_layout_path=self.source_layout_path,
        )

    def append_before_update(self, record: MeasurementLogRecord) -> int:
        """Durably stage one record and return its index before any PF update."""
        if self.resume_record_metadata:
            record = replace(
                record,
                metadata={
                    **dict(record.metadata),
                    **self.resume_record_metadata,
                },
            )
        if "station_complete" in record.metadata:
            raise MeasurementLogValidationError(
                "station_complete is writer-owned and must be marked only after "
                "the station acquisition finishes."
            )
        if self.records:
            previous = self.records[-1]
            if (
                int(previous.station_id) != int(record.station_id)
                and not _station_complete_marker(previous.metadata)
            ):
                raise MeasurementLogValidationError(
                    "A station must be durably marked complete before staging "
                    "the next station."
                )
            if _station_complete_marker(previous.metadata) and int(
                previous.station_id
            ) == int(record.station_id):
                raise MeasurementLogValidationError(
                    "A completed station cannot accept additional observations."
                )
        _validate_record_sequence((*self.records, record), self.isotopes)
        record_index = len(self.records)
        stage_path = self.stage_dir / f"record_{record_index:08d}.npz"
        _write_deterministic_npz(
            stage_path,
            _records_to_arrays(
                (record,),
                self.isotopes,
                validate_causal_sequence=False,
            ),
        )
        with stage_path.open("rb") as handle:
            os.fsync(handle.fileno())
        directory_fd = os.open(self.stage_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        line = _json_line_bytes(
            _metadata_line(
                record,
                run_id=self.run_id,
                record_index=record_index,
            )
        )
        with self.metadata_stage_path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record_index

    def mark_station_complete_before_update(
        self,
        station_id: int,
        *,
        completion_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Durably mark the last staged record as a causal station boundary."""
        if isinstance(station_id, bool) or not isinstance(
            station_id,
            (int, np.integer),
        ):
            raise MeasurementLogValidationError(
                "station_id must be an integer."
            )
        parsed_station_id = int(station_id)
        if not self.records:
            raise MeasurementLogValidationError(
                "Cannot complete a station before staging an observation."
            )
        record_index = len(self.records) - 1
        record = self.records[record_index]
        if record.station_id != parsed_station_id:
            raise MeasurementLogValidationError(
                "The completed station must match the last staged observation."
            )
        if _station_complete_marker(record.metadata):
            raise MeasurementLogValidationError(
                f"Station {parsed_station_id} is already marked complete."
            )
        completion_payload = dict(completion_metadata or {})
        _validate_truth_free_payload(
            completion_payload,
            location="station completion metadata",
        )
        if "station_complete" in completion_payload:
            raise MeasurementLogValidationError(
                "station_complete is writer-owned and cannot appear in completion "
                "metadata."
            )
        conflicting_keys = sorted(set(record.metadata) & set(completion_payload))
        if conflicting_keys:
            raise MeasurementLogValidationError(
                "Station completion metadata cannot replace observation metadata; "
                f"conflicts={conflicting_keys}."
            )
        completed = replace(
            record,
            metadata={
                **dict(record.metadata),
                **completion_payload,
                "station_complete": True,
            },
        )
        staged_records = [*self.records]
        staged_records[record_index] = completed
        metadata_bytes = b"".join(
            _json_line_bytes(
                _metadata_line(
                    staged_record,
                    run_id=self.run_id,
                    record_index=index,
                )
            )
            for index, staged_record in enumerate(staged_records)
        )
        temporary = self.metadata_stage_path.with_name(
            f".{self.metadata_stage_path.name}.tmp-{os.getpid()}"
        )
        if temporary.exists():
            raise FileExistsError(f"Metadata staging rewrite exists: {temporary}.")
        try:
            with temporary.open("wb") as handle:
                handle.write(metadata_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.metadata_stage_path)
            directory_fd = os.open(self.stage_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.records[record_index] = completed
        return record_index

    def finalize(self) -> MeasurementLog:
        """Consolidate staged records into the canonical hashed bundle."""
        if self.records:
            for index, record in enumerate(self.records):
                is_station_end = index + 1 == len(self.records) or int(
                    self.records[index + 1].station_id
                ) != int(record.station_id)
                marker = _station_complete_marker(
                    record.metadata,
                    location=f"records[{index}].metadata",
                )
                if marker is not is_station_end:
                    raise MeasurementLogValidationError(
                        "Every station must have exactly one causal station_complete "
                        "marker on its final record."
                    )
        result = write_measurement_log(
            self.output_dir,
            run_id=self.run_id,
            repository_commit=self.repository_commit,
            runtime_config=self.runtime_config,
            environment=self.environment,
            forward_model_manifest=self.forward_model_manifest,
            isotopes=self.isotopes,
            records=self.records,
            metadata=self.metadata,
            obstacle_layout_path=self.obstacle_layout_path,
            source_layout_path=self.source_layout_path,
        )
        shutil.rmtree(self.stage_dir)
        return result
