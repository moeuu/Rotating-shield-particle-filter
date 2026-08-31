"""Seal one authored RA-L batch before any adaptive session starts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from runtime.artifacts import atomic_write_json
from runtime.experiment_profiles import (
    MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE,
    acquisition_contract_from_environment,
)

from baselines.ral_ablation.config_factory import (
    DEFAULT_ABLATION_VARIANTS,
    MANIFEST_FIELDS,
    RAL_CASE_NAME,
    RAL_EXPERIMENT_PROFILE_ID,
    RAL_SCENE_VARIANT_ID,
)


_SCENARIO_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "backend",
        "runtime_config_path",
        "output_dir",
        "isotopes",
        "environment",
        "obstacle_layout_path",
        "scene",
        "metadata",
    }
)
_TRUTH_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "experiment_profile_id",
        "scene_variant_id",
        "scene_seed",
        "scene_rng_provenance",
        "sources",
    }
)


def _strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    """Load one regular JSON object without duplicate or non-finite values."""
    target = Path(path).expanduser().resolve()
    if target.is_symlink() or not target.is_file():
        raise FileNotFoundError(f"{name} must be a regular file: {target}")

    def reject_constant(value: str) -> object:
        """Reject Python JSON extensions such as NaN and Infinity."""
        raise ValueError(f"{name} contains forbidden JSON constant {value}.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate JSON member names."""
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"{name} contains duplicate field {key!r}.")
            payload[key] = value
        return payload

    payload = json.loads(
        target.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return payload


def _canonical_sha256(payload: object) -> str:
    """Return the canonical strict-JSON digest for one comparison payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Load the exact current private RA-L manifest schema."""
    target = Path(path).expanduser().resolve()
    if target.is_symlink() or not target.is_file():
        raise FileNotFoundError(f"RA-L manifest must be a regular file: {target}")
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("RA-L manifest header differs from the current schema.")
        rows = list(reader)
    if not rows or any(
        None in row or any(row.get(field) is None for field in MANIFEST_FIELDS)
        for row in rows
    ):
        raise ValueError("RA-L manifest must contain complete rows.")
    return [
        {field: str(row[field]) for field in MANIFEST_FIELDS}
        for row in rows
    ]


def _same_path(left: object, right: object) -> bool:
    """Return whether two declared path values resolve to one location."""
    return Path(str(left)).expanduser().resolve() == Path(
        str(right)
    ).expanduser().resolve()


def _require_authored_identity(
    row: Mapping[str, str],
    scenario: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> None:
    """Bind one authored scenario and truth manifest to its manifest row."""
    if set(scenario) != _SCENARIO_FIELDS or scenario.get("schema_version") != 1:
        raise ValueError("Authored RA-L scenario must match schema version 1 exactly.")
    if set(truth) != _TRUTH_FIELDS or truth.get("schema_version") != 1:
        raise ValueError(
            "Authored RA-L truth manifest must match schema version 1 exactly."
        )
    if scenario.get("run_id") != row["run_id"] or truth.get("run_id") != row[
        "run_id"
    ]:
        raise ValueError("Authored scenario/truth run_id differs from the manifest.")
    if not _same_path(scenario.get("runtime_config_path"), row["runtime_config_path"]):
        raise ValueError("Authored scenario runtime config differs from the manifest.")
    if not _same_path(scenario.get("output_dir"), row["measurement_log_path"]):
        raise ValueError("Authored scenario output directory differs from the manifest.")
    if scenario.get("backend") != "geant4":
        raise ValueError("RA-L authored scenarios must use Geant4.")
    if truth.get("experiment_profile_id") != row["experiment_profile_id"] or (
        truth.get("scene_variant_id") != row["scene_variant_id"]
    ):
        raise ValueError("Authored truth profile differs from the manifest.")
    if str(truth.get("scene_seed")) != row["scene_seed"]:
        raise ValueError("Authored truth seed differs from the manifest.")
    metadata = scenario.get("metadata")
    environment = scenario.get("environment")
    scene = scenario.get("scene")
    if not isinstance(metadata, Mapping):
        raise TypeError("Authored scenario metadata must be an object.")
    if not isinstance(environment, Mapping):
        raise TypeError("Authored scenario environment must be an object.")
    if not isinstance(scene, Mapping):
        raise TypeError("Authored scenario scene must be an object.")
    expected_metadata = {
        "experiment_profile_id": row["experiment_profile_id"],
        "private_scene_variant_id": row["scene_variant_id"],
        "scene_seed": int(row["scene_seed"]),
        "scene_rng_provenance": truth["scene_rng_provenance"],
    }
    mismatched = sorted(
        field
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    )
    if mismatched:
        raise ValueError(
            "Authored scenario metadata differs from private truth: "
            f"{mismatched}."
        )
    if environment.get("experiment_profile_id") != row["experiment_profile_id"]:
        raise ValueError("Authored environment profile differs from the manifest.")
    if scene.get("sources") != truth.get("sources"):
        raise ValueError("Authored scenario sources differ from private truth.")
    acquisition = acquisition_contract_from_environment(environment)
    if acquisition != MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition:
        raise ValueError("Authored acquisition contract differs from the RA-L profile.")


def seal_authored_batch(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    batch_id: str | None = None,
) -> dict[str, object]:
    """Verify exact shared scene/source conditions and publish their contract."""
    rows = _read_manifest(Path(manifest_path))
    available = sorted({row["batch_id"] for row in rows})
    selected_batch = batch_id
    if selected_batch is None:
        if len(available) != 1:
            raise ValueError(
                "Authored-batch verification requires one batch or --batch-id; "
                f"found {available}."
            )
        selected_batch = available[0]
    if selected_batch not in available:
        raise ValueError(f"Unknown RA-L batch_id {selected_batch!r}.")
    selected = [row for row in rows if row["batch_id"] == selected_batch]
    expected_variants = tuple(
        variant.name for variant in DEFAULT_ABLATION_VARIANTS
    )
    if tuple(row["variant"] for row in selected) != expected_variants:
        raise ValueError(
            "Authored RA-L batch must contain every variant once in canonical order."
        )
    shared_manifest_fields = (
        "case",
        "experiment_profile_id",
        "scene_variant_id",
        "scene_seed",
        "pf_seed",
        "transport_seed",
        "seed_policy",
    )
    for field in shared_manifest_fields:
        values = {row[field] for row in selected}
        if len(values) != 1:
            raise ValueError(
                f"Authored RA-L variants do not share one exact {field}."
            )
    if selected[0]["case"] != RAL_CASE_NAME or (
        selected[0]["experiment_profile_id"] != RAL_EXPERIMENT_PROFILE_ID
        or selected[0]["scene_variant_id"] != RAL_SCENE_VARIANT_ID
    ):
        raise ValueError("Authored batch differs from the declared RA-L task.")

    shared_comparison_digest: str | None = None
    shared_truth_digest: str | None = None
    per_variant: dict[str, object] = {}
    source_counts: Counter[str] = Counter()
    for row in selected:
        scenario_path = Path(row["scenario_path"]).expanduser().resolve()
        truth_path = Path(row["truth_manifest_path"]).expanduser().resolve()
        scenario = _strict_json_object(
            scenario_path,
            name=f"{row['variant']} scenario",
        )
        truth = _strict_json_object(
            truth_path,
            name=f"{row['variant']} truth manifest",
        )
        _require_authored_identity(row, scenario, truth)
        comparison_payload = {
            "backend": scenario["backend"],
            "environment": scenario["environment"],
            "isotopes": scenario["isotopes"],
            "metadata": scenario["metadata"],
            "scene": scenario["scene"],
        }
        truth_payload = {
            key: value for key, value in truth.items() if key != "run_id"
        }
        comparison_digest = _canonical_sha256(comparison_payload)
        truth_digest = _canonical_sha256(truth_payload)
        if shared_comparison_digest is None:
            shared_comparison_digest = comparison_digest
            shared_truth_digest = truth_digest
            sources = truth["sources"]
            if not isinstance(sources, list):
                raise TypeError("Private RA-L truth sources must be an array.")
            for source in sources:
                if not isinstance(source, Mapping):
                    raise TypeError("Every private RA-L source must be an object.")
                isotope = source.get("isotope")
                if not isinstance(isotope, str) or not isotope:
                    raise ValueError("Every private RA-L source needs an isotope.")
                source_counts[isotope] += 1
        elif comparison_digest != shared_comparison_digest or (
            truth_digest != shared_truth_digest
        ):
            raise ValueError(
                "RA-L variants do not share the exact authored environment and "
                "source parameters."
            )
        per_variant[row["variant"]] = {
            "run_id": row["run_id"],
            "scenario_path": scenario_path.as_posix(),
            "scenario_file_sha256": sha256(scenario_path.read_bytes()).hexdigest(),
            "truth_manifest_path": truth_path.as_posix(),
            "truth_manifest_file_sha256": sha256(truth_path.read_bytes()).hexdigest(),
        }
    assert shared_comparison_digest is not None
    assert shared_truth_digest is not None
    contract: dict[str, object] = {
        "schema_version": 1,
        "artifact_family": "ral_authored_batch_comparison_contract",
        "batch_id": selected_batch,
        "case": RAL_CASE_NAME,
        "experiment_profile_id": RAL_EXPERIMENT_PROFILE_ID,
        "scene_variant_id": RAL_SCENE_VARIANT_ID,
        "variants": list(expected_variants),
        "comparison_contract_sha256": shared_comparison_digest,
        "private_truth_contract_sha256": shared_truth_digest,
        "source_count_by_isotope": dict(sorted(source_counts.items())),
        "acquisition_contract": {
            "max_stations": MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.max_stations,
            "views_per_station": (
                MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.views_per_station
            ),
            "live_time_s": MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.live_time_s,
            "max_measurements": (
                MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.max_measurements
            ),
            "min_station_separation_m": (
                MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition
                .min_station_separation_m
            ),
            "coverage_radius_m": (
                MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.coverage_radius_m
            ),
        },
        "per_variant": per_variant,
    }
    target = Path(output_path).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"Refusing to replace an authored-batch contract: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    atomic_write_json(target, contract)
    target.chmod(0o600)
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    """Parse authored-batch verification arguments and seal one contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args(None if argv is None else list(argv))
    contract = seal_authored_batch(
        args.manifest,
        args.output,
        batch_id=args.batch_id,
    )
    print(
        "Sealed RA-L authored batch "
        f"{contract['batch_id']} with comparison digest "
        f"{contract['comparison_contract_sha256']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["seal_authored_batch"]
