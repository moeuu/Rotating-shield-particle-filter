"""Tests for strict PF configuration loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pf.configuration import PFConfigDocument, PFConfigError, load_pf_config


def test_load_pf_config_loads_and_hashes_one_self_contained_file(
    tmp_path: Path,
) -> None:
    """Configuration provenance must bind one self-contained source file."""
    config_path = tmp_path / "pf.json"
    config_bytes = b'{"num_particles":24,"use_gpu":false}'
    config_path.write_bytes(config_bytes)

    document = load_pf_config(config_path)

    assert document.config() == {"num_particles": 24, "use_gpu": False}
    assert document.source_bytes == config_bytes
    assert document.source_sha256 == hashlib.sha256(config_bytes).hexdigest()


def test_pf_config_document_rejects_caller_forged_provenance(
    tmp_path: Path,
) -> None:
    """Only the byte-reading loader may mint a provenance document."""
    payload = b'{"num_particles":24}'
    with pytest.raises(PFConfigError, match="load_pf_config"):
        PFConfigDocument(
            source_path=tmp_path / "pf.json",
            source_bytes=payload,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            canonical_config_json=payload,
            _loader_token=object(),
        )


@pytest.mark.parametrize(
    "config_text",
    (
        '{"num_particles":12,"num_particles":24}',
        '{"num_particles":NaN}',
        "[]",
    ),
)
def test_load_pf_config_rejects_noncanonical_json(
    tmp_path: Path,
    config_text: str,
) -> None:
    """Duplicate keys, non-finite values, and non-objects fail closed."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(PFConfigError):
        load_pf_config(config_path)


def test_load_pf_config_rejects_retired_inheritance(tmp_path: Path) -> None:
    """The retired extends mechanism must fail at the file boundary."""
    config_path = tmp_path / "pf.json"
    config_path.write_text(
        '{"extends":"parent.json","num_particles":24}',
        encoding="utf-8",
    )

    with pytest.raises(PFConfigError, match="retired 'extends'"):
        load_pf_config(config_path)
