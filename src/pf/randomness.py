"""Define reproducible, order-independent random streams for the pure PF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pf.provenance import json_safe


PF_RNG_PROVENANCE_SCHEMA_VERSION = 1
PF_RNG_DERIVATION_METHOD = "numpy_seedsequence_isotope_utf8_spawn_key_v1"
PF_RNG_BIT_GENERATOR = "PCG64"
_PF_RNG_ISOTOPE_DOMAIN = 0x5046524E
_PF_RNG_NAMED_STREAM_DOMAIN = 0x50464E53
_PF_RNG_MAX_ROOT_SEED = (1 << 128) - 1


def normalize_pf_random_seed(value: object) -> int:
    """Return one canonical non-negative 128-bit PF root seed."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError("PF random seed must be an integer.")
    seed = int(value)
    if seed < 0 or seed > _PF_RNG_MAX_ROOT_SEED:
        raise ValueError("PF random seed must be in [0, 2**128 - 1].")
    return seed


def isotope_spawn_key(isotope: str) -> tuple[int, ...]:
    """Encode an isotope name as a stable NumPy SeedSequence spawn key."""
    name = str(isotope)
    if not name:
        raise ValueError("PF isotope names must be non-empty.")
    encoded = name.encode("utf-8")
    return (
        _PF_RNG_ISOTOPE_DOMAIN,
        PF_RNG_PROVENANCE_SCHEMA_VERSION,
        len(encoded),
        *(int(value) for value in encoded),
    )


def isotope_seed_sequence(
    root_seed: int,
    isotope: str,
) -> np.random.SeedSequence:
    """Derive one isotope stream independently of construction order."""
    return np.random.SeedSequence(
        normalize_pf_random_seed(root_seed),
        spawn_key=isotope_spawn_key(isotope),
    )


def isotope_random_generator(
    root_seed: int,
    isotope: str,
) -> np.random.Generator:
    """Build the explicitly versioned PCG64 generator for one isotope PF."""
    return np.random.Generator(
        np.random.PCG64(isotope_seed_sequence(root_seed, isotope))
    )


def named_stream_spawn_key(*components: object) -> tuple[int, ...]:
    """Encode a stable named substream without using Python's process hash."""
    if not components:
        raise ValueError("A named random stream requires at least one component.")
    encoded_components = [str(component).encode("utf-8") for component in components]
    if any(not component for component in encoded_components):
        raise ValueError("Named random-stream components must be non-empty.")
    key: list[int] = [
        _PF_RNG_NAMED_STREAM_DOMAIN,
        PF_RNG_PROVENANCE_SCHEMA_VERSION,
        len(encoded_components),
    ]
    for component in encoded_components:
        key.extend((len(component), *(int(value) for value in component)))
    return tuple(key)


def named_seed_sequence(
    root_seed: int,
    *components: object,
) -> np.random.SeedSequence:
    """Derive one order-independent named stream from a logged root seed."""
    return np.random.SeedSequence(
        normalize_pf_random_seed(root_seed),
        spawn_key=named_stream_spawn_key(*components),
    )


def named_random_generator(
    root_seed: int,
    *components: object,
) -> np.random.Generator:
    """Build a deterministic PCG64 generator for a named runtime operation."""
    return np.random.Generator(
        np.random.PCG64(named_seed_sequence(root_seed, *components))
    )


def named_stream_seed(
    root_seed: int,
    *components: object,
) -> int:
    """Return a deterministic unsigned 64-bit seed for a named runtime stream."""
    words = named_seed_sequence(root_seed, *components).generate_state(
        2,
        dtype=np.uint32,
    )
    return int(words[0]) | (int(words[1]) << 32)


def named_rng_provenance(
    root_seed: int,
    stream_names: Sequence[str],
) -> dict[str, Any]:
    """Describe named planning streams derived from one logged root seed."""
    seed = normalize_pf_random_seed(root_seed)
    names = tuple(sorted({str(name) for name in stream_names}))
    if not names or any(not name for name in names):
        raise ValueError("Named RNG provenance requires non-empty stream names.")
    return {
        "schema_version": PF_RNG_PROVENANCE_SCHEMA_VERSION,
        "root_seed": seed,
        "derivation_method": (
            "numpy_seedsequence_named_utf8_spawn_key_v1"
        ),
        "bit_generator": PF_RNG_BIT_GENERATOR,
        "streams": {
            name: {
                "domain": name,
                "spawn_key": list(named_stream_spawn_key(name)),
                "derived_seed_u64": named_stream_seed(seed, name),
            }
            for name in names
        },
    }


def pf_rng_provenance(
    root_seed: int,
    isotopes: Sequence[str],
) -> dict[str, Any]:
    """Return canonical provenance for all independent isotope PF streams."""
    seed = normalize_pf_random_seed(root_seed)
    names = tuple(sorted({str(isotope) for isotope in isotopes}))
    if not names or any(not name for name in names):
        raise ValueError("PF RNG provenance requires non-empty isotope names.")
    return {
        "schema_version": PF_RNG_PROVENANCE_SCHEMA_VERSION,
        "root_seed": seed,
        "derivation_method": PF_RNG_DERIVATION_METHOD,
        "bit_generator": PF_RNG_BIT_GENERATOR,
        "isotope_key_encoding": (
            "domain_u32,schema_version_u32,utf8_length_u32,utf8_bytes_u32"
        ),
        "isotope_streams": {
            isotope: {
                "spawn_key": list(isotope_spawn_key(isotope)),
            }
            for isotope in names
        },
    }


def validate_pf_rng_provenance(
    value: object,
    *,
    root_seed: int,
    isotopes: Sequence[str],
) -> dict[str, Any]:
    """Validate logged PF RNG provenance and return its canonical mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("PF RNG provenance must be an object.")
    actual = json_safe(dict(value))
    expected = pf_rng_provenance(root_seed, isotopes)
    if actual != expected:
        raise ValueError(
            "PF RNG provenance does not match the declared root seed and isotopes."
        )
    return expected


__all__ = [
    "PF_RNG_BIT_GENERATOR",
    "PF_RNG_DERIVATION_METHOD",
    "PF_RNG_PROVENANCE_SCHEMA_VERSION",
    "isotope_random_generator",
    "isotope_seed_sequence",
    "isotope_spawn_key",
    "named_random_generator",
    "named_rng_provenance",
    "named_seed_sequence",
    "named_stream_seed",
    "named_stream_spawn_key",
    "normalize_pf_random_seed",
    "pf_rng_provenance",
    "validate_pf_rng_provenance",
]
