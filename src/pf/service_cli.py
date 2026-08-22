"""Optional-dependency bootstrap for the independent PF service."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import sys


_SERVICE_CONTRACT_PACKAGE = "radiation_estimator_service_contracts"
_SERVICE_EXTRA_HINT = (
    "rotating-shield-pf-service requires the optional service dependencies; "
    "install 'rotating-shield-particle-filter[service]'."
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the independent service or report how to install its wire contract."""
    try:
        service_module = import_module("pf.service")
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name != _SERVICE_CONTRACT_PACKAGE and not missing_name.startswith(
            f"{_SERVICE_CONTRACT_PACKAGE}."
        ):
            raise
        print(f"rotating-shield-pf-service: {_SERVICE_EXTRA_HINT}", file=sys.stderr)
        return 69
    return int(service_module.main(argv))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
