"""Runtime compatibility lookups for the public estimator facade."""

from __future__ import annotations

import sys
from typing import Any


def runtime_estimator_export(name: str) -> Any:
    """Return one export from the initialized ``pf.estimator`` facade."""
    facade = sys.modules.get("pf.estimator")
    if facade is None:
        raise RuntimeError("The PF estimator facade is not initialized.")
    return getattr(facade, name)
