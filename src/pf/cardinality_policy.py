"""Shared live and post-run cardinality health policy."""

from __future__ import annotations


HARD_CAP_POSTERIOR_MASS_LIMIT = 0.05
"""Maximum allowed posterior mass at the per-isotope hard capacity."""


def hard_cap_mass_is_acceptable(mass: float) -> bool:
    """Return whether finite posterior hard-cap mass satisfies the policy."""
    value = float(mass)
    return value == value and 0.0 <= value <= HARD_CAP_POSTERIOR_MASS_LIMIT


__all__ = [
    "HARD_CAP_POSTERIOR_MASS_LIMIT",
    "hard_cap_mass_is_acceptable",
]
