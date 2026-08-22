"""Provide deterministic PF configuration and repository provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert supported configuration values into canonical JSON objects."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                "Canonical provenance JSON cannot contain NaN or infinity."
            )
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical provenance JSON with stable indentation and newline."""
    text = json.dumps(
        json_safe(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def repository_commit(repository_root: Path | None = None) -> str:
    """Return the checked-out Git commit or an explicit unavailable marker."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = completed.stdout.strip()
    return commit if commit else "unavailable"


def repository_source_snapshot_sha256(
    repository_root: Path | None = None,
) -> str:
    """Hash the actual estimator source/config snapshot, including dirty files."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Cannot enumerate the repository source snapshot."
        ) from exc
    prefixes = ("src/", "scripts/", "configs/", "native/", "tests/")
    root_files = frozenset(
        {"AGENTS.md", "main.py", "pyproject.toml", "uv.lock"}
    )
    paths = sorted(
        {
            Path(raw.decode("utf-8"))
            for raw in completed.stdout.split(b"\0")
            if raw
            and (
                raw.decode("utf-8") in root_files
                or raw.decode("utf-8").startswith(prefixes)
            )
        },
        key=lambda value: value.as_posix(),
    )
    digest = hashlib.sha256(b"repository_source_snapshot_v1\0")
    for relative in paths:
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(path.readlink().as_posix().encode("utf-8"))
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing\0")
        digest.update(b"\0")
    return digest.hexdigest()
