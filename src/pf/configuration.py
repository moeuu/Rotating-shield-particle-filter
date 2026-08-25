"""Strict loading for PF-owned JSON configuration files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PFConfigError(RuntimeError):
    """Report an invalid or unreadable PF configuration."""


_PF_CONFIG_LOAD_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PFConfigDocument:
    """Bind one parsed PF configuration to its exact immutable source bytes."""

    source_path: Path
    source_bytes: bytes
    source_sha256: str
    canonical_config_json: bytes
    _loader_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Verify source and canonical payload identities."""
        if self._loader_token is not _PF_CONFIG_LOAD_TOKEN:
            raise PFConfigError(
                "PFConfigDocument values may only be created by load_pf_config()."
            )
        if hashlib.sha256(self.source_bytes).hexdigest() != self.source_sha256:
            raise PFConfigError("PF config source digest does not match its bytes.")
        payload = _parse_config_json(
            self.canonical_config_json.decode("utf-8"),
            location="canonical PF config",
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if canonical != self.canonical_config_json:
            raise PFConfigError("PF config canonical payload is not canonical JSON.")

    def config(self) -> dict[str, Any]:
        """Return a detached copy of the parsed configuration object."""
        return _parse_config_json(
            self.canonical_config_json.decode("utf-8"),
            location=str(self.source_path),
        )


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


def load_pf_config(path: str | Path) -> PFConfigDocument:
    """Load one self-contained PF configuration with exact byte provenance."""
    config_path = Path(path).expanduser().resolve()
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise PFConfigError(f"Cannot read PF config {config_path}.") from exc
    try:
        text = config_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PFConfigError(f"PF config {config_path} must be UTF-8 JSON.") from exc
    payload = _parse_config_json(text, location=str(config_path))
    if "extends" in payload:
        raise PFConfigError(
            f"PF config {config_path} uses retired 'extends' inheritance; "
            "production configurations must be self-contained."
        )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return PFConfigDocument(
        source_path=config_path,
        source_bytes=config_bytes,
        source_sha256=_sha256_bytes(config_bytes),
        canonical_config_json=canonical,
        _loader_token=_PF_CONFIG_LOAD_TOKEN,
    )


__all__ = ["PFConfigDocument", "PFConfigError", "load_pf_config"]
