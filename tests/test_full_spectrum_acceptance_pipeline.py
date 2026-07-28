"""End-to-end tests for the resumable full-spectrum acceptance pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
import scripts.run_full_spectrum_all64_acceptance as acceptance_cli
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_TARGET_SEMANTICS,
)
import spectrum.full_spectrum_acceptance_evaluator as evaluator
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_SCENARIO_SOURCE_SPEC,
    NATIVE_ACCEPTANCE_FIDELITY,
    AcceptanceRunLayout,
    acceptance_transport_seed,
    canonical_json_sha256,
    line_identity_contract_sha256,
    validate_scene_corpus,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    GeometryConditionedSpectralModel,
)


def _line_token(
    source_index: int,
    isotope: str,
    energy_keV: float,
) -> str:
    """Return one exact native validation-line token."""
    return (
        f"src{source_index}_{isotope.replace('-', '_')}_"
        f"e{energy_keV:.1f}"
    ).replace(".", "p")


def _source(
    isotope: str,
    intensity: float,
    *,
    source_index: int,
) -> dict[str, object]:
    """Return one canonical floor-surface source for the fake transport."""
    anchor = [1.0 + source_index, 2.0 + source_index, 0.0]
    normal = [0.0, 0.0, 1.0]
    transport = [
        anchor[index] + SURFACE_EMISSION_EPSILON_M * normal[index]
        for index in range(3)
    ]
    return {
        "isotope": isotope,
        "position": anchor,
        "transport_position": transport,
        "intensity_cps_1m": float(intensity),
        "surface_chart_id": int(source_index),
        "surface_uv": [0.25, 0.75],
        "surface_normal": normal,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
    }


def _boundary_gate(scene_seed: int) -> dict[str, object]:
    """Return distinct deterministic fake evidence for one native gate."""
    variants = (
        "exact_surface_anchor",
        "air_plus_epsilon",
        "solid_minus_epsilon",
    )
    return {
        "schema_version": 1,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "native_position_variants": list(variants),
        "evidence_sha256_by_variant": {
            variant: canonical_json_sha256([scene_seed, variant])
            for variant in variants
        },
        "exact_anchor_vs_air_gate_passed": True,
        "solid_minus_air_gate_passed": True,
        "passed": True,
    }


class _FakeSession:
    """Return exact artifacts without invoking native transport."""

    def __init__(
        self,
        *,
        scene_seed: int,
        split: str,
        scenario_id: str,
        line_hash: str,
    ) -> None:
        """Store one deterministic fake scene identity."""
        self.scene_seed = scene_seed
        self.split = split
        self.scenario_id = scenario_id
        self.line_hash = line_hash
        self.model = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=0.0,
            background_rate_cps=0.0,
        )
        self.sources = [
            _source(
                isotope,
                intensity,
                source_index=index,
            )
            for index, (isotope, intensity) in enumerate(
                ACCEPTANCE_SCENARIO_SOURCE_SPEC[scenario_id]
            )
        ]

    def acquire_pair(self, shield_pair_id: int) -> Mapping[str, object]:
        """Return one strict pair checkpoint with positive scatter labels."""
        source_count = len(self.sources)
        line_count = len(self.model.line_identity)
        shape = (1, source_count, line_count)
        if source_count:
            unattenuated = np.zeros(shape, dtype=np.float64)
            uncollided = np.zeros_like(unattenuated)
            features = np.zeros(shape + (4,), dtype=np.float64)
            scatter = np.zeros(
                shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),),
                dtype=np.float64,
            )
            for source_index, source in enumerate(self.sources):
                for line_index, line in enumerate(self.model.line_identity):
                    if line["isotope"] == source["isotope"]:
                        unattenuated[0, source_index, line_index] = 10.0
                        uncollided[0, source_index, line_index] = 9.0
                        features[0, source_index, line_index, 3] = 1.0
                        scatter[0, source_index, line_index, 0] = 0.5
            geometry: dict[str, object] = {
                "unattenuated_source_line_rate_vsl": unattenuated.tolist(),
                "uncollided_source_line_rate_vsl": uncollided.tolist(),
                "transport_features_vslf": features.tolist(),
                "additive_scatter_basis_vslf": scatter.tolist(),
            }
        else:
            unattenuated = np.empty(shape, dtype=np.float64)
            geometry = {
                "unattenuated_source_line_rate_vsl": None,
                "uncollided_source_line_rate_vsl": None,
                "transport_features_vslf": None,
                "additive_scatter_basis_vslf": None,
            }
        if self.scenario_id == "continuous_surface_perturbation_ranking":
            geometry.update(
                {
                    "perturbed_unattenuated_source_line_rate_vsl": (
                        (0.8 * unattenuated).tolist()
                    ),
                    "perturbed_uncollided_source_line_rate_vsl": (
                        (0.72 * unattenuated).tolist()
                    ),
                    "perturbed_transport_features_vslf": features.tolist(),
                    "perturbed_additive_scatter_basis_vslf": scatter.tolist(),
                }
            )
        else:
            geometry.update(
                {
                    "perturbed_unattenuated_source_line_rate_vsl": None,
                    "perturbed_uncollided_source_line_rate_vsl": None,
                    "perturbed_transport_features_vslf": None,
                    "perturbed_additive_scatter_basis_vslf": None,
                }
            )
        totals: dict[str, object] = {}
        hashes: dict[str, object] = {}
        for source_index, source in enumerate(self.sources):
            for line in self.model.line_identity:
                if line["isotope"] != source["isotope"]:
                    continue
                token = _line_token(
                    source_index,
                    str(source["isotope"]),
                    float(line["energy_keV"]),
                )
                totals[token] = {
                    "uncollided_primary": 270,
                    "interacted_primary": 20,
                    "secondary": 10,
                }
                hashes[token] = {
                    entry_class: canonical_json_sha256(
                        [
                            self.scene_seed,
                            self.scenario_id,
                            shield_pair_id,
                            token,
                            entry_class,
                        ]
                    )
                    for entry_class in (
                        "uncollided_primary",
                        "interacted_primary",
                        "secondary",
                    )
                }
        return {
            "schema_version": 1,
            "acceptance_contract_sha256": (
                self.model.manifest_payload()[
                    "acceptance_contract_sha256"
                ]
            ),
            "scene_seed": self.scene_seed,
            "split": self.split,
            "scenario_id": self.scenario_id,
            "shield_pair_id": int(shield_pair_id),
            "transport_seed": acceptance_transport_seed(
                scene_seed=self.scene_seed,
                scenario_id=self.scenario_id,
                shield_pair_id=shield_pair_id,
            ),
            "dwell_time_s": 30.0,
            "scene_hash": canonical_json_sha256(
                [self.scene_seed, self.scenario_id, "scene"]
            ),
            "surface_source_contract_sha256": (
                surface_source_runtime_contract_sha256(self.sources)
            ),
            "surface_boundary_gate": _boundary_gate(self.scene_seed),
            "detector_pose_xyz": [1.0, 1.0, 0.5],
            "sources": self.sources,
            "line_identity_contract_sha256": self.line_hash,
            "observed_spectrum_counts": [0] * NATIVE_GEANT4_BIN_COUNT,
            "geometry": geometry,
            "validation_labels": {
                "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
                "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
                "entry_class_totals_by_source_line": totals,
                "entry_spectrum_sha256_by_source_line_class": hashes,
                "background_entry_total": 0,
                "background_entry_spectrum_sha256": canonical_json_sha256(
                    [self.scene_seed, self.scenario_id, shield_pair_id, "bg"]
                ),
            },
            "native_fidelity": dict(NATIVE_ACCEPTANCE_FIDELITY),
        }


class _FakeBackend:
    """Fake transport backend used only to exercise orchestration contracts."""

    backend_id = "test_fake_external_geant4"
    opened: list[tuple[str, int, str]] = []

    def __init__(self, **_: object) -> None:
        """Expose the hashes and rates required by the production CLI."""
        self.runtime_config_sha256 = "a" * 64
        self.native_executable_sha256 = "b" * 64
        self.app_config = SimpleNamespace(
            dead_time_tau_s=0.0,
            background_cps=0.0,
        )

    def open_scenario(
        self,
        *,
        scene_seed: int,
        split: str,
        scenario_id: str,
        line_identity_sha256: str,
    ) -> nullcontext[_FakeSession]:
        """Open one deterministic fake session and record phase ordering."""
        self.opened.append((split, scene_seed, scenario_id))
        return nullcontext(
            _FakeSession(
                scene_seed=scene_seed,
                split=split,
                scenario_id=scenario_id,
                line_hash=line_identity_sha256,
            )
        )


def _passing_metrics(
    *_: object,
    **__: object,
) -> dict[str, float]:
    """Return values passing every immutable threshold for orchestration tests."""
    return {
        metric: 0.0 if comparison == "le" else 1.0
        for metric, (comparison, _) in ACCEPTANCE_METRIC_CONTRACT.items()
    }


def test_fake_backend_full_pipeline_is_resumable_and_holdout_cannot_refit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Training, freeze, holdout, evaluation, and approval must be one-way."""
    root = tmp_path / "acceptance"
    arguments = ["--output-root", root.as_posix()]
    _FakeBackend.opened.clear()
    monkeypatch.setattr(
        acceptance_cli,
        "ExternalGeant4AcceptanceBackend",
        _FakeBackend,
    )
    monkeypatch.setattr(acceptance_cli, "_progress", lambda _: None)
    monkeypatch.setattr(evaluator, "_scene_metrics", _passing_metrics)

    assert acceptance_cli.main(["training", *arguments]) == 0
    assert all(split == "training" for split, _, _ in _FakeBackend.opened)
    assert acceptance_cli.main(["fit-freeze", *arguments]) == 0
    layout = AcceptanceRunLayout(root)
    frozen_before_holdout = layout.candidate_model_path.read_bytes()

    assert acceptance_cli.main(["holdout", *arguments]) == 0
    assert any(split == "holdout" for split, _, _ in _FakeBackend.opened)
    assert layout.candidate_model_path.read_bytes() == frozen_before_holdout
    assert acceptance_cli.main(["evaluate", *arguments]) == 0
    assert layout.candidate_model_path.read_bytes() == frozen_before_holdout
    assert acceptance_cli.main(["approve", *arguments]) == 0
    assert layout.validation_manifest_path.is_file()
    assert layout.production_model_path.is_file()

    opened_before_resume = tuple(_FakeBackend.opened)
    assert acceptance_cli.main(["all", *arguments]) == 0
    assert tuple(_FakeBackend.opened) == opened_before_resume
    assert layout.candidate_model_path.read_bytes() == frozen_before_holdout


def test_validation_labels_are_absent_from_production_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Changing labels alone must change no production likelihood tensor."""
    root = tmp_path / "acceptance"
    arguments = ["--output-root", root.as_posix()]
    _FakeBackend.opened.clear()
    monkeypatch.setattr(
        acceptance_cli,
        "ExternalGeant4AcceptanceBackend",
        _FakeBackend,
    )
    monkeypatch.setattr(acceptance_cli, "_progress", lambda _: None)
    assert acceptance_cli.main(["training", *arguments]) == 0
    assert acceptance_cli.main(["fit-freeze", *arguments]) == 0
    layout = AcceptanceRunLayout(root)
    model = evaluator.load_frozen_candidate_model(layout)
    line_hash = line_identity_contract_sha256(model)
    records = validate_scene_corpus(
        layout.scene_corpus_path(
            split="training",
            scene_seed=2026072701,
        ),
        layout=layout,
        expected_line_identity_sha256=line_hash,
    )
    selected = tuple(
        record
        for record in records
        if record.scenario_id == "single_line_source_resolved"
    )
    original = evaluator._scenario_data(selected, model=model)
    mutated = tuple(
        replace(record, labels={"holdout_canary": index})
        for index, record in enumerate(selected)
    )
    canary = evaluator._scenario_data(mutated, model=model)

    assert np.array_equal(original.observed_vb, canary.observed_vb)
    assert np.array_equal(original.total_vsl, canary.total_vsl)
    assert np.array_equal(original.uncollided_vsl, canary.uncollided_vsl)
    assert np.array_equal(original.features_vslf, canary.features_vslf)
