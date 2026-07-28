"""Strict physical identity contract for full-spectrum MeasurementLog v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from measurement.shielding import line_resolved_shield_mu_by_isotope
from pf.provenance import sha256_json


FORWARD_MODEL_MANIFEST_SCHEMA_VERSION = 2
SOURCE_RATE_MODEL = "detector_cps_1m"
SOURCE_RATE_SEMANTICS = {
    "quantity": "expected_pre_dead_time_detector_pulse_rate",
    "unit": "cps",
    "normalization_distance_m": 1.0,
}
CANONICAL_UNITS = {
    "distance": "m",
    "time": "s",
    "energy": "keV",
    "source_strength": "detector_cps_1m",
    "linear_attenuation": "cm^-1",
}
RESPONSE_SEMANTICS = {
    "distance_attenuation": "inverse_square_with_modelled_near_field",
    "detector_geometry": "model_identifier_bound",
    "shield_attenuation": "fe_pb_orientation_pair_8x8",
    "obstacle_attenuation": "line_segment_material_attenuation",
    "live_time_scaling": (
        "incident_rate_linear_then_nonparalyzable_renewal_detection"
    ),
    "line_resolved_response": (
        "source_resolved_geometry_conditioned_full_spectrum"
    ),
    "observation_distribution": (
        "joint_renewal_total_and_conditional_energy_marks"
    ),
}
REQUIRED_MODEL_NAMES = (
    "detector",
    "shield",
    "environment",
    "obstacle",
    "transport",
    "spectrum",
)

_NATIVE_FIELDS = {
    "schema_version",
    "repository_commit",
    "resolved_config_sha256",
    "source_rate_model",
    "source_rate_semantics",
    "model_identifiers",
    "units",
    "response_semantics",
    "line_mu_by_isotope",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def production_line_mu_by_isotope(
    isotopes: Sequence[str],
) -> dict[str, list[dict[str, float]]]:
    """Project the exact production ``ContinuousKernel`` spectral line table."""
    isotope_order = tuple(str(value) for value in isotopes)
    if not isotope_order or len(set(isotope_order)) != len(isotope_order):
        raise ValueError("Forward-model isotopes must be non-empty and unique.")
    raw = line_resolved_shield_mu_by_isotope(
        isotopes=isotope_order,
        normalize_line_intensities=True,
    )
    missing = [isotope for isotope in isotope_order if not raw.get(isotope)]
    if missing:
        raise ValueError(
            "No production line-resolved shield response exists for isotopes "
            f"{missing}."
        )
    return {
        isotope: [
            {name: float(entry[name]) for name in ("energy_keV", "weight", "fe", "pb")}
            for entry in raw[isotope]
        ]
        for isotope in isotope_order
    }


def line_energy_weight_by_isotope(
    line_table: Mapping[str, object],
) -> dict[str, list[dict[str, float]]]:
    """Return the spectral-identity subset of a full attenuation line table."""
    return {
        str(isotope): [
            {
                "energy_keV": float(entry["energy_keV"]),
                "weight": float(entry["weight"]),
            }
            for entry in entries
        ]
        for isotope, entries in line_table.items()
    }


def _selected(payload: Mapping[str, object], *tokens: str) -> dict[str, object]:
    """Return a deterministic copy of keys related to any supplied token."""
    lowered = tuple(token.lower() for token in tokens)
    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
        if any(token in str(key).lower() for token in lowered)
    }


def _identifier(
    payloads: tuple[Mapping[str, object], ...],
    keys: tuple[str, ...],
    default: str,
) -> str:
    """Return the first explicit identifier or a stable production default."""
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


def _safe_relative_asset_path(path_value: object, *, field_name: str) -> Path:
    """Return one canonical relative asset path without traversal ambiguity."""
    raw = str(path_value)
    if not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path.")
    if "\\" in raw:
        raise ValueError(f"{field_name} must use portable forward-slash separators.")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(
            f"{field_name} must be relative; absolute paths are forbidden."
        )
    if not relative.parts or any(part in {"", ".."} for part in relative.parts):
        raise ValueError(f"{field_name} must not contain parent-directory traversal.")
    return relative


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_file_backed_model_asset(
    path_value: object,
    *,
    field_name: str,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> Path:
    """Resolve an asset from a contained run root, then this repository."""
    relative = _safe_relative_asset_path(path_value, field_name=field_name)
    roots: list[Path] = []
    if run_root is not None:
        roots.append(Path(run_root))
    roots.append(Path(repository_root))
    for root in roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        if not _is_within(candidate, resolved_root):
            raise ValueError(f"{field_name} escapes an allowed local asset root.")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{field_name} was not found in the measurement run or local repository: "
        f"{relative.as_posix()!r}."
    )


def _file_asset_identity(
    path_value: object,
    *,
    field_name: str,
    run_root: str | Path | None,
    repository_root: str | Path,
) -> dict[str, str]:
    """Return the portable path plus its raw-byte digest."""
    relative = _safe_relative_asset_path(path_value, field_name=field_name)
    resolved = resolve_file_backed_model_asset(
        relative,
        field_name=field_name,
        run_root=run_root,
        repository_root=repository_root,
    )
    return {
        "path": relative.as_posix(),
        "sha256": sha256(resolved.read_bytes()).hexdigest(),
    }


def _runtime_file_asset_references(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[str, str, object]]:
    """Discover detector/transport/spectrum path fields recursively."""
    references: list[tuple[str, str, object]] = []
    if isinstance(value, Mapping):
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            child_path = (*path, key)
            normalized_key = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            normalized_path = "".join(
                character
                for part in child_path
                for character in part.casefold()
                if character.isalnum()
            )
            is_path_field = normalized_key.endswith(("path", "file"))
            component = next(
                (
                    name
                    for name in ("transport", "detector", "spectrum")
                    if name in normalized_path
                ),
                None,
            )
            if child is not None and is_path_field and component is not None:
                if not isinstance(child, (str, Path)):
                    raise TypeError(
                        f"runtime_config.{'.'.join(child_path)} must be a path string."
                    )
                references.append((".".join(child_path), component, child))
            else:
                references.extend(
                    _runtime_file_asset_references(child, path=child_path)
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            references.extend(
                _runtime_file_asset_references(child, path=(*path, f"[{index}]"))
            )
    return references


def _runtime_file_asset_identities(
    runtime_config: Mapping[str, object],
    *,
    run_root: str | Path | None,
    repository_root: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Resolve and hash file assets grouped by physical component."""
    grouped: dict[str, dict[str, dict[str, str]]] = {
        "transport": {},
        "detector": {},
        "spectrum": {},
    }
    for field_path, component, path_value in _runtime_file_asset_references(
        runtime_config
    ):
        grouped[component][field_path] = _file_asset_identity(
            path_value,
            field_name=f"runtime_config.{field_path}",
            run_root=run_root,
            repository_root=repository_root,
        )
    return grouped


def forward_model_component_payloads(
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: Sequence[str],
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, dict[str, object]]:
    """Return the exact payload whose digest identifies every native component."""
    runtime = deepcopy(dict(runtime_config))
    environment_payload = deepcopy(dict(environment))
    isotope_order = tuple(str(value) for value in isotopes)
    line_table = production_line_mu_by_isotope(isotope_order)
    payloads: dict[str, dict[str, object]] = {
        "detector": _selected(runtime, "detector", "aperture", "crystal", "housing"),
        # The shield hash is the full production line table and the spectrum
        # hash is exactly its energy/weight projection. Runtime-specific
        # settings remain bound by the resolved-config artifact hash.
        "shield": line_table,
        "environment": environment_payload,
        "obstacle": {
            "environment": _selected(
                environment_payload,
                "obstacle",
                "blocked_cells",
                "grid_shape",
                "cell_size",
                "origin",
            ),
            "runtime_config": _selected(
                runtime,
                "obstacle",
                "material",
                "buildup",
                "source_extent",
            ),
            "layout_path": obstacle_layout_path,
        },
        "transport": {
            "runtime_config": _selected(
                runtime,
                "transport",
                "attenuation",
                "inverse_square",
                "buildup",
            )
        },
        "spectrum": {
            "line_energy_weight_by_isotope": (
                line_energy_weight_by_isotope(line_table)
            ),
            "full_spectrum_generative_model": deepcopy(
                runtime.get("full_spectrum_generative_model")
            ),
            "full_spectrum_contract_hash_sha256": runtime.get(
                "full_spectrum_contract_hash_sha256"
            ),
            "energy_min_keV": runtime.get("energy_min_keV"),
            "energy_max_keV": runtime.get("energy_max_keV"),
            "bin_width_keV": runtime.get("bin_width_keV"),
            "energy_bin_count": runtime.get("energy_bin_count"),
        },
    }
    if obstacle_layout_path is not None:
        payloads["obstacle"]["layout_asset"] = _file_asset_identity(
            obstacle_layout_path,
            field_name="obstacle_layout_path",
            run_root=run_root,
            repository_root=repository_root,
        )
    file_assets = _runtime_file_asset_identities(
        runtime,
        run_root=run_root,
        repository_root=repository_root,
    )
    for component, identities in file_assets.items():
        if identities:
            payloads[component]["file_assets"] = identities
    return payloads


def build_forward_model_manifest(
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: Sequence[str],
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str = SOURCE_RATE_MODEL,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, object]:
    """Build a complete native manifest bound to resolved production physics."""
    if str(source_rate_model).strip().lower() != SOURCE_RATE_MODEL:
        raise ValueError(f"source_rate_model must be {SOURCE_RATE_MODEL!r}.")
    component_payloads = forward_model_component_payloads(
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotopes,
        run_root=run_root,
        repository_root=repository_root,
    )
    line_table = production_line_mu_by_isotope(isotopes)
    runtime = dict(runtime_config)
    environment_payload = dict(environment)
    identifiers = {
        "detector": _identifier(
            (runtime,),
            ("detector_model_id", "detector_model_identifier"),
            "local_detector_observation_geometry.v1",
        ),
        "shield": _identifier(
            (runtime,),
            ("shield_model_id", "shield_model_identifier"),
            "rotating_nested_octant_shield.v1",
        ),
        "environment": _identifier(
            (environment_payload, runtime),
            ("environment_model_id", "environment_id", "environment_mode"),
            "rectangular_room_surface_environment.v1",
        ),
        "obstacle": _identifier(
            (environment_payload, runtime),
            ("obstacle_model_id", "obstacle_layout_id"),
            obstacle_layout_path or "embedded_or_empty_obstacle_grid.v1",
        ),
        "transport": _identifier(
            (runtime,),
            ("transport_model_id",),
            "continuous_geometry_additive_noncollided_transport.v1",
        ),
        "spectrum": _identifier(
            (runtime,),
            ("spectrum_model_id", "spectrum_response_model_id"),
            "geometry_conditioned_joint_full_spectrum.v1",
        ),
    }
    return {
        "schema_version": FORWARD_MODEL_MANIFEST_SCHEMA_VERSION,
        "repository_commit": str(repository_commit).strip(),
        "resolved_config_sha256": _sha256(
            resolved_config_sha256,
            name="resolved_config_sha256",
        ),
        "source_rate_model": SOURCE_RATE_MODEL,
        "source_rate_semantics": deepcopy(SOURCE_RATE_SEMANTICS),
        "units": deepcopy(CANONICAL_UNITS),
        "response_semantics": deepcopy(RESPONSE_SEMANTICS),
        "line_mu_by_isotope": line_table,
        "model_identifiers": {
            name: {
                "id": identifiers[name],
                "sha256": sha256_json(component_payloads[name]),
            }
            for name in REQUIRED_MODEL_NAMES
        },
    }


def _sha256(value: object, *, name: str) -> str:
    """Return a validated lowercase SHA-256 string."""
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return normalized


def _validate_model_identifiers(
    raw_identifiers: object,
    *,
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Validate the exact six physical component identifiers and hashes."""
    if not isinstance(raw_identifiers, Mapping):
        raise ValueError("forward_model_manifest.model_identifiers must be an object.")
    if set(raw_identifiers) != set(REQUIRED_MODEL_NAMES):
        raise ValueError(
            "forward_model_manifest.model_identifiers must contain exactly "
            f"{list(REQUIRED_MODEL_NAMES)}."
        )
    normalized: dict[str, dict[str, str]] = {}
    for name in REQUIRED_MODEL_NAMES:
        entry = raw_identifiers[name]
        if not isinstance(entry, Mapping) or set(entry) != {"id", "sha256"}:
            raise ValueError(
                f"model_identifiers.{name} must contain exactly id and sha256."
            )
        identifier = str(entry.get("id", "")).strip()
        digest = _sha256(
            entry.get("sha256"),
            name=f"model_identifiers.{name}.sha256",
        )
        expected_entry = expected[name]
        if identifier != str(expected_entry["id"]) or digest != str(
            expected_entry["sha256"]
        ):
            raise ValueError(
                f"Forward-model compatibility error for {name}: identifier or "
                "SHA-256 differs from the resolved production model."
            )
        normalized[name] = {"id": identifier, "sha256": digest}
    return normalized


def _validate_common(
    payload: Mapping[str, object],
    *,
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str,
) -> None:
    """Validate semantics shared by production-native manifests."""
    if payload.get("schema_version") != FORWARD_MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported forward-model manifest schema_version.")
    if str(source_rate_model).strip().lower() != SOURCE_RATE_MODEL:
        raise ValueError("run-manifest source_rate_model is incompatible.")
    if payload.get("source_rate_model") != SOURCE_RATE_MODEL:
        raise ValueError("forward-model source_rate_model is incompatible.")
    if payload.get("source_rate_semantics") != SOURCE_RATE_SEMANTICS:
        raise ValueError("forward-model source_rate_semantics is incompatible.")
    if payload.get("repository_commit") != str(repository_commit).strip():
        raise ValueError("forward-model repository_commit does not match the log.")
    if payload.get("resolved_config_sha256") != _sha256(
        resolved_config_sha256,
        name="resolved_config_sha256",
    ):
        raise ValueError("forward-model resolved_config_sha256 does not match the log.")
    if payload.get("units") != CANONICAL_UNITS:
        raise ValueError("forward-model units are incompatible.")
    if payload.get("response_semantics") != RESPONSE_SEMANTICS:
        raise ValueError("forward-model response_semantics are incompatible.")


def validate_forward_model_manifest(
    manifest: Mapping[str, object],
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: Sequence[str],
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str = SOURCE_RATE_MODEL,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, object]:
    """Prove that a manifest exactly matches the local production model."""
    if not isinstance(manifest, Mapping):
        raise TypeError("forward_model_manifest must be a mapping.")
    payload = deepcopy(dict(manifest))
    isotope_order = tuple(str(value) for value in isotopes)
    if set(payload) != _NATIVE_FIELDS:
        raise ValueError(
            f"Native forward-model fields must be exactly {sorted(_NATIVE_FIELDS)}."
        )
    _validate_common(
        payload,
        repository_commit=repository_commit,
        resolved_config_sha256=resolved_config_sha256,
        source_rate_model=source_rate_model,
    )
    expected = build_forward_model_manifest(
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotope_order,
        repository_commit=repository_commit,
        resolved_config_sha256=resolved_config_sha256,
        source_rate_model=source_rate_model,
        run_root=run_root,
        repository_root=repository_root,
    )
    if payload.get("line_mu_by_isotope") != expected["line_mu_by_isotope"]:
        raise ValueError(
            "forward_model_manifest line_mu_by_isotope differs from production."
        )
    payload["model_identifiers"] = _validate_model_identifiers(
        payload.get("model_identifiers"),
        expected=expected["model_identifiers"],
    )
    return payload


__all__ = [
    "CANONICAL_UNITS",
    "FORWARD_MODEL_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_MODEL_NAMES",
    "RESPONSE_SEMANTICS",
    "SOURCE_RATE_MODEL",
    "SOURCE_RATE_SEMANTICS",
    "build_forward_model_manifest",
    "forward_model_component_payloads",
    "line_energy_weight_by_isotope",
    "production_line_mu_by_isotope",
    "resolve_file_backed_model_asset",
    "validate_forward_model_manifest",
]
