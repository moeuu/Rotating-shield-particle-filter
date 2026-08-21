"""Small atomic file-publication helpers shared by PF artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace one file with already serialized bytes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace one text file using the requested encoding."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Serialize strict deterministic JSON and publish it atomically."""
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return atomic_write_text(path, serialized + "\n")


__all__ = ["atomic_write_bytes", "atomic_write_json", "atomic_write_text"]
