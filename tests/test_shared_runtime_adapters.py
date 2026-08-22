"""Conformance tests for PF compatibility adapters to shared runtime APIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pf import atomic_io, provenance, runtime_defaults
from runtime import artifacts
from runtime import provenance as runtime_provenance
from runtime.defaults import DEFAULT_CUI_SPLIT_VIEW_DIR


def test_pf_canonical_json_and_atomic_io_delegate_to_runtime(
    tmp_path: Path,
) -> None:
    """PF compatibility modules should retain byte-identical shared behavior."""
    payload = {
        "array": np.asarray([2, 1], dtype=np.int64),
        "path": tmp_path / "value",
    }

    assert provenance.canonical_json_bytes(
        payload
    ) == runtime_provenance.canonical_json_bytes(payload)
    assert atomic_io.atomic_write_bytes is artifacts.atomic_write_bytes
    assert atomic_io.atomic_write_json is artifacts.atomic_write_json
    assert atomic_io.atomic_write_text is artifacts.atomic_write_text


def test_pf_cui_defaults_delegate_to_installable_runtime_package() -> None:
    """PF CUI entry points should use the shared presentation defaults."""
    assert runtime_defaults.DEFAULT_CUI_SPLIT_VIEW_DIR == DEFAULT_CUI_SPLIT_VIEW_DIR


def test_pf_repository_provenance_keeps_pf_as_default_root() -> None:
    """Shared helpers should hash the PF checkout when called through its shim."""
    root = Path(__file__).resolve().parents[1]

    assert provenance.repository_commit() == runtime_provenance.repository_commit(
        root
    )
