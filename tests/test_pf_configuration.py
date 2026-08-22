"""Tests for strict PF configuration loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pf.configuration import PFConfigError, load_pf_config


def test_load_pf_config_resolves_inheritance_and_hashes_leaf(tmp_path: Path) -> None:
    """Inherited settings merge while provenance binds the selected leaf file."""
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    parent.write_text('{"num_particles":12,"use_gpu":false}', encoding="utf-8")
    child_bytes = b'{"extends":"parent.json","num_particles":24}'
    child.write_bytes(child_bytes)

    config, digest = load_pf_config(child)

    assert config == {"num_particles": 24, "use_gpu": False}
    assert digest == hashlib.sha256(child_bytes).hexdigest()


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


def test_load_pf_config_rejects_inheritance_cycles(tmp_path: Path) -> None:
    """A cyclic extends chain must fail before settings reach the live PF."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"extends":"second.json"}', encoding="utf-8")
    second.write_text('{"extends":"first.json"}', encoding="utf-8")

    with pytest.raises(PFConfigError, match="Cyclic PF config inheritance"):
        load_pf_config(first)
