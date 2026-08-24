"""Generic shield-program values shared by production and baseline planners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShieldProgram:
    """Represent an ordered sequence of Fe/Pb shield orientation pairs."""

    name: str
    pair_ids: tuple[int, ...]
    kind: str


__all__ = ["ShieldProgram"]
