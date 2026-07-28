"""Tests for strict all-64 training/holdout acceptance aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from spectrum.full_spectrum_acceptance import (
    SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS,
    build_independent_validation_manifest,
    load_training_scene_artifacts,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
)


def _metric_values(*, passing: bool = True) -> dict[str, float]:
    """Return deterministic per-scene metrics at their fixed thresholds."""
    values = {
        metric_id: float(threshold)
        for metric_id, (_, threshold) in ACCEPTANCE_METRIC_CONTRACT.items()
    }
    if not passing:
        values["background_pairwise_95_coverage_fraction"] = 0.0
    return values


def _write_payload(path: Path, payload: object) -> None:
    """Write one immutable canonical acceptance fixture."""
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifact(
    root: Path,
    *,
    scene_seed: int,
    metrics: dict[str, float] | None = None,
    pair_ids: list[int] | None = None,
) -> Path:
    """Write one complete synthetic scene artifact for contract tests."""
    split = (
        "training"
        if scene_seed in DESIGNATED_TRAINING_SCENE_SEEDS
        else "holdout"
    )
    payload = {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "scene_seed": scene_seed,
        "split": split,
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "shield_pair_ids": (
            list(range(64)) if pair_ids is None else pair_ids
        ),
        "pair_ids_by_scenario": {
            scenario: list(range(64))
            for scenario in VALIDATION_SCENARIO_IDS
        },
        "observation_count_by_scenario": {
            scenario: 64 for scenario in VALIDATION_SCENARIO_IDS
        },
        "approved_model_contract_sha256": "a" * 64,
        "native_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "additive_scatter_contract_sha256": "b" * 64,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "scene_hash_by_scenario": {
            scenario: (
                f"{scene_seed + scenario_index:064x}"
            )
            for scenario_index, scenario in enumerate(
                VALIDATION_SCENARIO_IDS
            )
        },
        "surface_source_contract_sha256_by_scenario": {
            scenario: (
                f"{scene_seed + len(VALIDATION_SCENARIO_IDS) + scenario_index:064x}"
            )
            for scenario_index, scenario in enumerate(
                VALIDATION_SCENARIO_IDS
            )
        },
        "surface_boundary_gate": {
            "schema_version": 1,
            "surface_emission_policy_sha256": (
                surface_emission_policy_sha256()
            ),
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "native_position_variants": list(
                SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS
            ),
            "exact_anchor_vs_air_gate_passed": True,
            "solid_minus_air_gate_passed": True,
            "passed": True,
        },
        "metrics": _metric_values() if metrics is None else metrics,
    }
    path = root / f"scene_{scene_seed}.json"
    _write_payload(path, payload)
    return path


def _all_artifacts(
    root: Path,
    *,
    failing_holdout: bool = False,
    hostile_training: bool = False,
) -> list[Path]:
    """Write the complete designated scene set."""
    paths: list[Path] = []
    for seed in (
        DESIGNATED_TRAINING_SCENE_SEEDS
        + DESIGNATED_HOLDOUT_SCENE_SEEDS
    ):
        metrics = _metric_values()
        if hostile_training and seed in DESIGNATED_TRAINING_SCENE_SEEDS:
            for metric_id, (comparison, _) in (
                ACCEPTANCE_METRIC_CONTRACT.items()
            ):
                metrics[metric_id] = (
                    1.0e12 if comparison == "le" else -1.0e12
                )
        if failing_holdout and seed == DESIGNATED_HOLDOUT_SCENE_SEEDS[0]:
            metrics = _metric_values(passing=False)
        paths.append(
            _write_artifact(
                root,
                scene_seed=seed,
                metrics=metrics,
            )
        )
    return paths


def test_validation_metrics_use_holdout_scenes_only(tmp_path: Path) -> None:
    """Arbitrarily bad training metrics cannot change holdout acceptance."""
    manifest = build_independent_validation_manifest(
        _all_artifacts(tmp_path, hostile_training=True)
    )

    assert manifest["metric_split"] == "holdout_only"
    assert tuple(manifest["metric_scene_seeds"]) == (
        DESIGNATED_HOLDOUT_SCENE_SEEDS
    )
    assert manifest["metric_aggregation"] == (
        "holdout_scene_conservative_worst_case"
    )
    assert manifest["all_passed"] is True
    for metric_id, (_, threshold) in ACCEPTANCE_METRIC_CONTRACT.items():
        assert manifest["metrics"][metric_id]["value"] == float(threshold)


def test_one_failing_holdout_cannot_be_masked_by_training(
    tmp_path: Path,
) -> None:
    """Conservative holdout aggregation must preserve an absent-tail failure."""
    manifest = build_independent_validation_manifest(
        _all_artifacts(
            tmp_path,
            failing_holdout=True,
            hostile_training=False,
        )
    )

    result = manifest["metrics"][
        "background_pairwise_95_coverage_fraction"
    ]
    assert result["value"] == 0.0
    assert result["passed"] is False
    assert manifest["all_passed"] is False


def test_scene_artifact_requires_all_64_pairs(tmp_path: Path) -> None:
    """A nonempty but incomplete shield program must fail closed."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["shield_pair_ids"] = list(range(63))
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="all-64"):
        build_independent_validation_manifest(paths)


def test_every_scenario_independently_requires_all_64_pairs(
    tmp_path: Path,
) -> None:
    """A global pair list cannot hide an incomplete absent-isotope scenario."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["pair_ids_by_scenario"][
        "dominant_plus_absent_isotope"
    ] = list(range(63))
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="independently cover all 64"):
        build_independent_validation_manifest(paths)


def test_k0_and_absent_isotope_false_positive_gates_are_required(
    tmp_path: Path,
) -> None:
    """Production acceptance must expose both cardinality false-positive gates."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    del payload["metrics"][
        "absent_isotope_k_positive_decision_rate_at_p0p95"
    ]
    _write_payload(paths[-1], payload)

    assert (
        "background_k_positive_decision_rate_at_p0p95"
        in ACCEPTANCE_METRIC_CONTRACT
    )
    with pytest.raises(ValueError, match="fixed contract"):
        build_independent_validation_manifest(paths)


def test_holdout_artifact_cannot_enter_training_selection(
    tmp_path: Path,
) -> None:
    """The training loader accepts exactly the three predeclared seeds."""
    all_paths = _all_artifacts(tmp_path)
    contaminated = [
        all_paths[0],
        all_paths[1],
        all_paths[-1],
    ]

    with pytest.raises(ValueError, match="designated scene set"):
        load_training_scene_artifacts(contaminated)


def test_native_signed_epsilon_gate_is_mandatory(tmp_path: Path) -> None:
    """A scene without the solid-side counterfactual cannot be approved."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["surface_boundary_gate"][
        "solid_minus_air_gate_passed"
    ] = False
    _write_payload(paths[-1], payload)

    with pytest.raises(ValueError, match="signed-epsilon"):
        build_independent_validation_manifest(paths)
