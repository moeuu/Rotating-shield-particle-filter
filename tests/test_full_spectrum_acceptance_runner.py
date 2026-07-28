"""Tests for fail-closed resumable full-spectrum acceptance artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_TARGET_SEMANTICS,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_DWELL_TIME_S,
    NATIVE_ACCEPTANCE_FIDELITY,
    acceptance_transport_seed,
    build_acceptance_run_contract,
    canonical_json_bytes,
    line_identity_contract_sha256,
    load_acceptance_pair,
)
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import (
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    TRANSPORT_FEATURE_ORDER,
    GeometryConditionedSpectralModel,
)


def _base_model() -> GeometryConditionedSpectralModel:
    """Return the fixed native line contract used by acceptance artifacts."""
    return GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
    )


def test_acceptance_run_contract_authenticates_python_implementation() -> None:
    """Every resumable phase must bind the exact Python implementation."""
    contract = build_acceptance_run_contract(
        runtime_config_sha256="a" * 64,
        native_executable_sha256="b" * 64,
        implementation_bundle_sha256="c" * 64,
    )

    assert contract["schema_version"] == 2
    assert contract["implementation_bundle_sha256"] == "c" * 64


def test_acceptance_run_contract_rejects_invalid_implementation_digest() -> None:
    """A missing implementation digest must not permit mixed phase code."""
    with pytest.raises(ValueError, match="implementation_bundle_sha256"):
        build_acceptance_run_contract(
            runtime_config_sha256="a" * 64,
            native_executable_sha256="b" * 64,
            implementation_bundle_sha256="not-a-digest",
        )


def _boundary_gate() -> dict[str, object]:
    """Return deterministic distinct signed-epsilon evidence."""
    return {
        "schema_version": 1,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "native_position_variants": [
            "exact_surface_anchor",
            "air_plus_epsilon",
            "solid_minus_epsilon",
        ],
        "evidence_sha256_by_variant": {
            "exact_surface_anchor": "1" * 64,
            "air_plus_epsilon": "2" * 64,
            "solid_minus_epsilon": "3" * 64,
        },
        "exact_anchor_vs_air_gate_passed": True,
        "solid_minus_air_gate_passed": True,
        "passed": True,
    }


def _source() -> dict[str, object]:
    """Return one valid continuous-surface Cs-137 source contract."""
    normal = [0.0, 0.0, 1.0]
    anchor = [3.0, 4.0, 0.0]
    transport = [
        anchor[index] + SURFACE_EMISSION_EPSILON_M * normal[index]
        for index in range(3)
    ]
    return {
        "isotope": "Cs-137",
        "position": anchor,
        "transport_position": transport,
        "intensity_cps_1m": 800_000.0,
        "surface_chart_id": 7,
        "surface_uv": [0.25, 0.75],
        "surface_normal": normal,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
    }


def _pair_payload(*, background_only: bool) -> dict[str, object]:
    """Return one exact pair payload for a source or background scenario."""
    model = _base_model()
    sources = [] if background_only else [_source()]
    source_count = len(sources)
    line_count = len(model.line_identity)
    if source_count == 0:
        geometry = {
            "unattenuated_source_line_rate_vsl": None,
            "uncollided_source_line_rate_vsl": None,
            "transport_features_vslf": None,
            "additive_scatter_basis_vslf": None,
            "perturbed_unattenuated_source_line_rate_vsl": None,
            "perturbed_uncollided_source_line_rate_vsl": None,
            "perturbed_transport_features_vslf": None,
            "perturbed_additive_scatter_basis_vslf": None,
        }
    else:
        unattenuated = np.zeros((1, 1, line_count), dtype=np.float64)
        uncollided = np.zeros_like(unattenuated)
        cs_line_index = next(
            index
            for index, line in enumerate(model.line_identity)
            if line["isotope"] == "Cs-137"
        )
        unattenuated[0, 0, cs_line_index] = 25_000.0
        uncollided[0, 0, cs_line_index] = 20_000.0
        geometry = {
            "unattenuated_source_line_rate_vsl": unattenuated.tolist(),
            "uncollided_source_line_rate_vsl": uncollided.tolist(),
            "transport_features_vslf": np.zeros(
                (1, 1, line_count, len(TRANSPORT_FEATURE_ORDER)),
                dtype=np.float64,
            ).tolist(),
            "additive_scatter_basis_vslf": np.zeros(
                (
                    1,
                    1,
                    line_count,
                    len(ADDITIVE_SCATTER_FEATURE_ORDER),
                ),
                dtype=np.float64,
            ).tolist(),
            "perturbed_unattenuated_source_line_rate_vsl": None,
            "perturbed_uncollided_source_line_rate_vsl": None,
            "perturbed_transport_features_vslf": None,
            "perturbed_additive_scatter_basis_vslf": None,
        }
    totals: dict[str, object] = {}
    hashes: dict[str, object] = {}
    for source_index, source in enumerate(sources):
        for line in model.line_identity:
            if line["isotope"] != source["isotope"]:
                continue
            token = native_source_line_token(
                source_index=source_index,
                isotope=str(source["isotope"]),
                energy_keV=float(line["energy_keV"]),
            )
            totals[token] = {
                "uncollided_primary": 10,
                "interacted_primary": 2,
                "secondary": 1,
            }
            hashes[token] = {
                "uncollided_primary": "4" * 64,
                "interacted_primary": "5" * 64,
                "secondary": "6" * 64,
            }
    scenario = (
        "background_only"
        if background_only
        else "single_line_source_resolved"
    )
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "scene_seed": 2026072701,
        "split": "training",
        "scenario_id": scenario,
        "shield_pair_id": 0,
        "transport_seed": acceptance_transport_seed(
            scene_seed=2026072701,
            scenario_id=scenario,
            shield_pair_id=0,
        ),
        "dwell_time_s": ACCEPTANCE_DWELL_TIME_S,
        "scene_hash": "7" * 64,
        "surface_source_contract_sha256": (
            surface_source_runtime_contract_sha256(sources)
        ),
        "surface_boundary_gate": _boundary_gate(),
        "detector_pose_xyz": [1.0, 2.0, 1.0],
        "sources": sources,
        "line_identity_contract_sha256": (
            line_identity_contract_sha256(model)
        ),
        "observed_spectrum_counts": [0] * NATIVE_GEANT4_BIN_COUNT,
        "geometry": geometry,
        "validation_labels": {
            "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
            "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
            "entry_class_totals_by_source_line": totals,
            "entry_spectrum_sha256_by_source_line_class": hashes,
            "background_entry_total": 0,
            "background_entry_spectrum_sha256": "8" * 64,
        },
        "native_fidelity": dict(NATIVE_ACCEPTANCE_FIDELITY),
    }


def _write_pair(path: Path, payload: dict[str, object]) -> Path:
    """Write one canonical test artifact."""
    path.write_bytes(canonical_json_bytes(payload))
    return path


@pytest.mark.parametrize("background_only", (True, False))
def test_pair_loader_accepts_exact_background_and_source_contracts(
    tmp_path: Path,
    background_only: bool,
) -> None:
    """Zero-source and nonzero-source tensors must reconstruct unambiguously."""
    model = _base_model()
    record = load_acceptance_pair(
        _write_pair(
            tmp_path / "pair.json",
            _pair_payload(background_only=background_only),
        ),
        expected_line_identity_sha256=line_identity_contract_sha256(model),
    )

    assert record.source_count == (0 if background_only else 1)
    assert record.unattenuated_vsl.shape == (
        1,
        record.source_count,
        len(model.line_identity),
    )


def test_pair_loader_rejects_numeric_string_geometry(tmp_path: Path) -> None:
    """Geometry arrays must not gain physical meaning through float coercion."""
    payload = _pair_payload(background_only=False)
    payload["geometry"]["unattenuated_source_line_rate_vsl"][0][0][0] = "0"

    with pytest.raises(TypeError, match="JSON numbers"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_authenticates_embedded_source_payload(tmp_path: Path) -> None:
    """A source coordinate change must invalidate the source-contract digest."""
    payload = _pair_payload(background_only=False)
    payload["sources"][0]["surface_uv"][0] = 0.5

    with pytest.raises(ValueError, match="source hash"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_every_native_source_line_label(
    tmp_path: Path,
) -> None:
    """Missing entry-class labels must not silently become zero training data."""
    payload = _pair_payload(background_only=False)
    totals = payload["validation_labels"][
        "entry_class_totals_by_source_line"
    ]
    token = next(iter(totals))
    del totals[token]

    with pytest.raises(ValueError, match="label payload"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_null_for_absent_tensor_axes(
    tmp_path: Path,
) -> None:
    """JSON empty lists must not masquerade as a higher-rank zero-source tensor."""
    payload = _pair_payload(background_only=True)
    payload["geometry"]["unattenuated_source_line_rate_vsl"] = []

    with pytest.raises(ValueError, match="Background-only geometry"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_distinct_boundary_probe_evidence(
    tmp_path: Path,
) -> None:
    """One reused digest cannot authenticate three signed native probes."""
    payload = copy.deepcopy(_pair_payload(background_only=True))
    payload["surface_boundary_gate"]["evidence_sha256_by_variant"] = {
        name: "1" * 64
        for name in (
            "exact_surface_anchor",
            "air_plus_epsilon",
            "solid_minus_epsilon",
        )
    }

    with pytest.raises(ValueError, match="boundary gate"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_artifacts_remain_strict_json(tmp_path: Path) -> None:
    """The test fixture itself must use only round-trippable JSON values."""
    payload = _pair_payload(background_only=False)
    assert json.loads(canonical_json_bytes(payload)) == payload


def test_pair_loader_rejects_noncanonical_duplicate_keys(
    tmp_path: Path,
) -> None:
    """A resumable checkpoint must never use last-key-wins semantics."""
    payload = _pair_payload(background_only=False)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    raw = (
        '{"schema_version":1,' + canonical.removeprefix("{")
    ).encode("utf-8")
    path = tmp_path / "duplicate.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="canonical JSON"):
        load_acceptance_pair(
            path,
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )
