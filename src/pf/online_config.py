"""PF-owned online configuration helpers independent of legacy live demos."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pf.profiles import enforce_pure_runtime_settings
from runtime.session import estimator_neutral_physical_runtime_config
from sim.runtime import load_runtime_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PF_CONFIG = ROOT / "configs" / "pf" / "pf_strict_3d.json"


def _deep_merge_runtime_config(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge physical and estimator configuration objects."""
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_runtime_config(existing, value)
        else:
            merged[key] = value
    return merged


def load_online_runtime_configs(
    sim_config_path: str | Path | None,
    pf_config_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load physical runtime config and PF-owned online estimator settings."""
    combined_config = load_runtime_config(sim_config_path)
    physical_config = estimator_neutral_physical_runtime_config(combined_config)
    estimator_defaults = (
        {}
        if pf_config_path is None
        else load_runtime_config(pf_config_path)
    )
    merged = _deep_merge_runtime_config(estimator_defaults, combined_config)
    return physical_config, enforce_pure_runtime_settings(merged)


def _sanitize_json_payload(
    payload: object,
    *,
    _path: str = "$",
    _unsafe_integers_as_decimal_strings: bool = False,
) -> object:
    """Return recursively strict-JSON-compatible data."""
    if payload is None or isinstance(payload, (str, bool)):
        return payload
    if isinstance(payload, np.bool_):
        return bool(payload)
    if isinstance(payload, (int, np.integer)):
        value = int(payload)
        if _unsafe_integers_as_decimal_strings and abs(value) > 2**53:
            return str(value)
        return value
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        if not np.isfinite(value):
            raise ValueError(
                f"Strict JSON payload contains a non-finite number at {_path}."
            )
        return value
    if isinstance(payload, Path):
        return payload.as_posix()
    if isinstance(payload, np.ndarray):
        return _sanitize_json_payload(
            payload.tolist(),
            _path=_path,
            _unsafe_integers_as_decimal_strings=(
                _unsafe_integers_as_decimal_strings
            ),
        )
    if isinstance(payload, np.generic):
        return _sanitize_json_payload(
            payload.item(),
            _path=_path,
            _unsafe_integers_as_decimal_strings=(
                _unsafe_integers_as_decimal_strings
            ),
        )
    if isinstance(payload, Mapping):
        result: dict[str, object] = {}
        for key, value in payload.items():
            resolved_key = str(key)
            if resolved_key in result:
                raise ValueError(
                    "Strict JSON payload contains colliding stringified keys at "
                    f"{_path}: {resolved_key!r}."
                )
            result[resolved_key] = _sanitize_json_payload(
                value,
                _path=f"{_path}[{resolved_key!r}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
        return result
    if isinstance(payload, (list, tuple)):
        return [
            _sanitize_json_payload(
                value,
                _path=f"{_path}[{index}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
            for index, value in enumerate(payload)
        ]
    if isinstance(payload, (set, frozenset)):
        return [
            _sanitize_json_payload(
                value,
                _path=f"{_path}[{index}]",
                _unsafe_integers_as_decimal_strings=(
                    _unsafe_integers_as_decimal_strings
                ),
            )
            for index, value in enumerate(
                sorted(payload, key=lambda item: repr(item))
            )
        ]
    raise TypeError(
        f"Unsupported value in JSON payload at {_path}: "
        f"{type(payload).__module__}.{type(payload).__qualname__}"
    )


def _validated_provided_source_provenance(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Validate explicit source-file provenance before run metadata publication."""
    required_keys = {
        "provided_file_path",
        "provided_file_path_kind",
        "provided_file_bytes_sha256",
        "provided_file_declared_metadata",
    }
    if set(provenance) != required_keys:
        missing = sorted(required_keys - set(provenance))
        unexpected = sorted(set(provenance) - required_keys)
        raise ValueError(
            "provided-file source provenance has incompatible fields: "
            f"missing={missing}, unexpected={unexpected}."
        )
    path_value = provenance["provided_file_path"]
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("provided_file_path must be a non-empty string.")
    if "\\" in path_value:
        raise ValueError("provided_file_path must use normalized POSIX separators.")
    path_kind = provenance["provided_file_path_kind"]
    if path_kind not in {"repository_relative", "resolved_absolute"}:
        raise ValueError(
            "provided_file_path_kind must be 'repository_relative' or "
            "'resolved_absolute'."
        )
    normalized_path = Path(path_value)
    if path_kind == "repository_relative":
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            raise ValueError(
                "repository-relative provided_file_path must remain within the "
                "repository."
            )
    elif not normalized_path.is_absolute():
        raise ValueError("resolved-absolute provided_file_path must be absolute.")
    digest = provenance["provided_file_bytes_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "provided_file_bytes_sha256 must be a lowercase SHA-256 digest."
        )
    declared_metadata = provenance["provided_file_declared_metadata"]
    if not isinstance(declared_metadata, Mapping):
        raise ValueError("provided_file_declared_metadata must be an object.")
    sanitized_metadata = _sanitize_json_payload(
        dict(declared_metadata),
        _unsafe_integers_as_decimal_strings=True,
    )
    if not isinstance(sanitized_metadata, dict):
        raise TypeError("provided_file_declared_metadata must sanitize to an object.")
    return {
        "provided_file_path": path_value,
        "provided_file_path_kind": path_kind,
        "provided_file_bytes_sha256": digest,
        "provided_file_declared_metadata": sanitized_metadata,
    }


__all__ = [
    "DEFAULT_PF_CONFIG",
    "load_online_runtime_configs",
    "_validated_provided_source_provenance",
]
