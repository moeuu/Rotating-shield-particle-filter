"""Provide PF-rooted wrappers around shared provenance helpers."""

from __future__ import annotations

from pathlib import Path

from runtime.provenance import (
    canonical_json_bytes,
    json_safe,
    repository_commit as _shared_repository_commit,
    repository_source_snapshot_sha256 as _shared_source_snapshot_sha256,
    sha256_json,
)


_PF_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_commit(repository_root: Path | None = None) -> str:
    """Return the checked-out Git commit or an explicit unavailable marker."""
    return _shared_repository_commit(
        _PF_REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).resolve()
    )


def repository_source_snapshot_sha256(
    repository_root: Path | None = None,
) -> str:
    """Hash the actual estimator source/config snapshot, including dirty files."""
    return _shared_source_snapshot_sha256(
        _PF_REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).resolve()
    )


__all__ = [
    "canonical_json_bytes",
    "json_safe",
    "repository_commit",
    "repository_source_snapshot_sha256",
    "sha256_json",
]
