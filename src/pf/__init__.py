"""Public live facade and particle-filter implementation package."""

from __future__ import annotations

from typing import Any


__all__ = [
    "PFExternalSurfaceGuidance",
    "PFExternalSurfaceGuidanceReceipt",
    "PFBoundLiveState",
    "PFCompletedLiveState",
    "PFLiveParticleSnapshot",
    "PFLiveSession",
    "PFLiveSessionError",
    "PFNextAction",
    "PFPlanningConfig",
    "PFPublishedLiveResult",
    "load_live_pf_config",
    "validate_live_pf_config",
]


def __getattr__(name: str) -> Any:
    """Load the live facade lazily without coupling PF submodule imports to DSS."""
    if name not in __all__:
        raise AttributeError(f"module 'pf' has no attribute {name!r}")
    from pf import live_session

    value = getattr(live_session, name)
    globals()[name] = value
    return value
