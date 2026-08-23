"""Explicit post-run join for private runtime truth and completed PF output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


_TRUTH_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "source_profile",
        "scene_seed",
        "scene_rng_provenance",
        "sources",
    }
)


@dataclass(frozen=True, slots=True)
class PrivateEvaluationTruth:
    """Hold private truth after an exact completed-run identifier join."""

    run_id: str
    source_profile: str
    scene_seed: int
    scene_rng_provenance: Mapping[str, object]
    sources: tuple[Mapping[str, object], ...]


def _load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    """Load one strict JSON object for explicit post-run evaluation."""
    target = Path(path).expanduser().resolve()

    def reject_constant(value: str) -> object:
        """Reject non-finite constants accepted by Python's JSON extension."""
        raise ValueError(f"{name} contains forbidden JSON constant {value}.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate member names before evaluation sees the object."""
        payload_object: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload_object:
                raise ValueError(f"{name} contains duplicate field {key!r}.")
            payload_object[key] = value
        return payload_object

    payload = json.loads(
        target.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return payload


def load_private_truth_for_completed_result(
    result_path: str | Path,
    truth_manifest_path: str | Path,
) -> PrivateEvaluationTruth:
    """Join truth only after PF completion and require an exact run_id match."""
    result = _load_json_object(result_path, name="PF result")
    if result.get("status") != "complete":
        raise ValueError("Private truth may be joined only to a completed PF result.")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Completed PF result must declare a nonempty run_id.")
    truth = _load_json_object(truth_manifest_path, name="private truth manifest")
    if set(truth) != _TRUTH_MANIFEST_FIELDS or truth.get("schema_version") != 1:
        raise ValueError("Private truth manifest must match schema version 1 exactly.")
    if truth.get("run_id") != run_id:
        raise ValueError("Private truth manifest run_id differs from PF result run_id.")
    source_profile = truth.get("source_profile")
    scene_seed = truth.get("scene_seed")
    provenance = truth.get("scene_rng_provenance")
    sources = truth.get("sources")
    if not isinstance(source_profile, str) or not source_profile:
        raise ValueError("Private truth source_profile must be nonempty.")
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise ValueError("Private truth scene_seed must be an integer.")
    if not isinstance(provenance, Mapping):
        raise ValueError("Private truth scene_rng_provenance must be an object.")
    if not isinstance(sources, list) or any(
        not isinstance(source, Mapping) for source in sources
    ):
        raise ValueError("Private truth sources must be an array of objects.")
    frozen_provenance = MappingProxyType(
        json.loads(json.dumps(dict(provenance), allow_nan=False))
    )
    frozen_sources = tuple(
        MappingProxyType(json.loads(json.dumps(dict(source), allow_nan=False)))
        for source in sources
    )
    return PrivateEvaluationTruth(
        run_id=run_id,
        source_profile=source_profile,
        scene_seed=scene_seed,
        scene_rng_provenance=frozen_provenance,
        sources=frozen_sources,
    )


__all__ = ["PrivateEvaluationTruth", "load_private_truth_for_completed_result"]
