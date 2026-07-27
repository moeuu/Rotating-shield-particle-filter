"""Canonical likelihood-route metadata shared by live and replayed pure PF runs."""

from __future__ import annotations

from typing import Mapping, Sequence


COUNT_LIKELIHOOD_ROUTE = "count"
COUNT_COVARIANCE_LIKELIHOOD_ROUTE = "count_covariance"
RUNTIME_LIKELIHOOD_ROUTES = frozenset(
    {
        COUNT_LIKELIHOOD_ROUTE,
        COUNT_COVARIANCE_LIKELIHOOD_ROUTE,
    }
)
RUNTIME_LIKELIHOOD_ROUTE_METADATA_KEY = (
    "runtime_likelihood_route_by_isotope"
)


def normalize_runtime_likelihood_route(value: object) -> str:
    """Return one exact canonical pure-PF likelihood route."""
    if not isinstance(value, str) or value not in RUNTIME_LIKELIHOOD_ROUTES:
        raise ValueError(
            "Runtime likelihood route must be exactly 'count' or "
            "'count_covariance'."
        )
    return value


def canonical_runtime_likelihood_route_mapping(
    value: object,
    isotopes: Sequence[str],
) -> dict[str, str]:
    """Validate and return one exact route for every configured isotope."""
    isotope_names = tuple(str(isotope) for isotope in isotopes)
    if (
        not isotope_names
        or len(set(isotope_names)) != len(isotope_names)
        or any(not isotope for isotope in isotope_names)
    ):
        raise ValueError("Configured isotope names must be non-empty and unique.")
    if not isinstance(value, Mapping):
        raise ValueError(
            "runtime_likelihood_route_by_isotope must be an object."
        )
    supplied_keys = set(value)
    expected_keys = set(isotope_names)
    if supplied_keys != expected_keys:
        raise ValueError(
            "runtime_likelihood_route_by_isotope must contain exactly every "
            f"configured isotope; expected={sorted(expected_keys)}, "
            f"actual={sorted(str(key) for key in supplied_keys)}."
        )
    return {
        isotope: normalize_runtime_likelihood_route(value[isotope])
        for isotope in isotope_names
    }
