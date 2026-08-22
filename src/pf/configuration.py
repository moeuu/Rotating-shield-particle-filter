"""Strict loading for PF-owned JSON configuration files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PFConfigError(RuntimeError):
    """Report an invalid or unreadable PF configuration."""


def _sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _parse_config_json(text: str, *, location: str) -> dict[str, Any]:
    """Parse one strict PF JSON object without duplicate keys."""

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build an object only when every JSON member name is unique."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PFConfigError(f"{location} contains duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def _reject_constant(value: str) -> None:
        """Reject Python's non-standard NaN and infinity JSON extensions."""
        raise PFConfigError(f"{location} contains non-finite JSON constant {value!r}.")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PFConfigError(f"Cannot parse PF config {location}.") from exc
    if not isinstance(payload, dict):
        raise PFConfigError(f"PF config {location} must contain an object.")
    return payload


def _load_inherited_config(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    """Load strict PF configuration inheritance without lossy parsing."""
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise PFConfigError(f"Cyclic PF config inheritance at {resolved_path}.")
    seen.add(resolved_path)
    try:
        text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PFConfigError(f"Cannot read PF config {resolved_path}.") from exc
    data = _parse_config_json(text, location=str(resolved_path))
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str) or not parent_ref:
        raise PFConfigError("PF config extends must be a nonempty string.")
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    parent = _load_inherited_config(parent_path, seen=seen)
    return {**parent, **data}


def load_pf_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one inherited PF configuration and return its source digest."""
    config_path = Path(path).expanduser().resolve()
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise PFConfigError(f"Cannot read PF config {config_path}.") from exc
    return (
        _load_inherited_config(config_path, seen=set()),
        _sha256_bytes(config_bytes),
    )


__all__ = ["PFConfigError", "load_pf_config"]
