"""Exact token helpers shared with the native Geant4 metadata protocol."""

from __future__ import annotations

import math
from numbers import Integral, Real


_CXX_ISSPACE_CHARACTERS = frozenset(" \t\n\v\f\r")


def sanitize_native_metadata_token(value: str) -> str:
    """Mirror ``SanitizeMetadataToken`` in the native Geant4 sidecar."""
    if not isinstance(value, str):
        raise TypeError("Native metadata token values must be strings.")
    sanitized = "".join(
        "_"
        if character in _CXX_ISSPACE_CHARACTERS or character in {",", "="}
        else character
        for character in value
    )
    return sanitized or "unknown"


def native_source_line_token(
    *,
    source_index: int,
    isotope: str,
    energy_keV: float,
) -> str:
    """Return the exact native source/initial-line metadata token."""
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, Integral)
        or source_index < 0
    ):
        raise ValueError("source_index must be a nonnegative integer.")
    if (
        isinstance(energy_keV, bool)
        or not isinstance(energy_keV, Real)
        or not math.isfinite(float(energy_keV))
    ):
        raise ValueError("energy_keV must be a finite real number.")
    energy_token = f"e{float(energy_keV):.1f}".replace(".", "p")
    return (
        f"src{int(source_index)}_"
        f"{sanitize_native_metadata_token(isotope)}_{energy_token}"
    )
