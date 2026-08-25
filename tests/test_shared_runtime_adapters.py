"""Conformance tests for PF-specific shared-runtime integration."""

from __future__ import annotations

from pathlib import Path

from pf import provenance
from runtime import provenance as runtime_provenance


def test_pf_repository_provenance_keeps_pf_as_default_root() -> None:
    """Shared helpers should hash the PF checkout when called through its shim."""
    root = Path(__file__).resolve().parents[1]

    assert provenance.repository_commit() == runtime_provenance.repository_commit(
        root
    )


def test_pf_provenance_exports_only_strict_deterministic_json_helpers() -> None:
    """The PF facade must not retain lossy or ambiguously named serializers."""
    left = {"schema_version": 2, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "schema_version": 2}

    assert provenance.strict_canonical_json_bytes(left) == (
        provenance.strict_canonical_json_bytes(right)
    )
    assert provenance.strict_sha256_json(left) == provenance.strict_sha256_json(
        right
    )
    assert not hasattr(provenance, "canonical_json_bytes")
    assert not hasattr(provenance, "sha256_json")
    assert not hasattr(provenance, "json_safe")
