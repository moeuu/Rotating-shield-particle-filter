"""Conformance tests for the truth-free MeasurementLog v1 contract."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from pf.provenance import canonical_json_bytes
from runtime.forward_model_manifest import (
    CONFORMANCE_MODEL_IDENTIFIERS,
    forward_model_component_payloads,
    line_energy_weight_by_isotope,
    registered_conformance_line_mu_by_isotope,
)
from runtime.measurement_log import (
    MeasurementLogStreamWriter,
    MeasurementLogValidationError,
    _records_from_arrays,
    build_forward_model_manifest,
    load_measurement_log,
)
from tests.pure_pf_test_support import (
    TEST_COMMIT,
    TEST_ISOTOPES,
    environment,
    make_measurement_log,
    records,
    runtime_config,
)


def test_measurement_log_is_byte_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    """Equivalent logs must have identical artifacts and complete records."""
    first = make_measurement_log(tmp_path / "first")
    second = make_measurement_log(tmp_path / "second")
    first_log = load_measurement_log(first)
    second_log = load_measurement_log(second)

    assert first_log.log_sha256 == second_log.log_sha256
    assert len(first_log.records) == 4
    assert first_log.records[0].isotope_counts == {
        "Cs-137": 20.0,
        "Co-60": 12.0,
        "Eu-154": 8.0,
    }
    for artifact in sorted(path.name for path in first.iterdir()):
        assert (first / artifact).read_bytes() == (second / artifact).read_bytes()


def test_forward_line_registry_is_bound_to_production_hashes() -> None:
    """The shared registry must bind the actual Cs/Co/Eu kernel line table."""
    table = registered_conformance_line_mu_by_isotope()
    assert [len(table[name]) for name in TEST_ISOTOPES] == [1, 2, 6]
    assert [entry["energy_keV"] for entry in table["Eu-154"]] == pytest.approx(
        [723.3, 873.2, 996.3, 1274.5, 1494.0, 1596.5]
    )
    assert (
        sha256(canonical_json_bytes(table)).hexdigest()
        == (CONFORMANCE_MODEL_IDENTIFIERS["shield"]["sha256"])
    )
    assert (
        sha256(canonical_json_bytes(line_energy_weight_by_isotope(table))).hexdigest()
        == CONFORMANCE_MODEL_IDENTIFIERS["spectrum"]["sha256"]
    )


def test_forward_identity_hashes_file_asset_bytes_and_rejects_unsafe_paths(
    tmp_path: Path,
) -> None:
    """A stable path cannot hide changed, missing, or non-contained model bytes."""
    repository_root = tmp_path / "repository"
    asset = repository_root / "assets" / "transport.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'{"response":1}\n')
    config = {
        **runtime_config(),
        "pf_transport_response_model_path": "assets/transport.json",
    }
    first = forward_model_component_payloads(
        runtime_config=config,
        environment=environment(),
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_root=repository_root,
    )
    identity = first["transport"]["file_assets"]["pf_transport_response_model_path"]
    assert identity == {
        "path": "assets/transport.json",
        "sha256": sha256(asset.read_bytes()).hexdigest(),
    }

    asset.write_bytes(b'{"response":2}\n')
    second = forward_model_component_payloads(
        runtime_config=config,
        environment=environment(),
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_root=repository_root,
    )
    assert first["transport"] != second["transport"]

    missing = {**config, "pf_transport_response_model_path": "assets/missing.json"}
    with pytest.raises(FileNotFoundError):
        forward_model_component_payloads(
            runtime_config=missing,
            environment=environment(),
            obstacle_layout_path=None,
            isotopes=TEST_ISOTOPES,
            repository_root=repository_root,
        )
    for unsafe in (asset.as_posix(), "../transport.json", "assets\\transport.json"):
        with pytest.raises(ValueError):
            forward_model_component_payloads(
                runtime_config={
                    **config,
                    "pf_transport_response_model_path": unsafe,
                },
                environment=environment(),
                obstacle_layout_path=None,
                isotopes=TEST_ISOTOPES,
                repository_root=repository_root,
            )


def test_reader_rejects_truth_tampering_and_noncanonical_arrays(
    tmp_path: Path,
) -> None:
    """Truth, modified artifacts, and dtype-coerced NPZ arrays fail closed."""
    truth_log = make_measurement_log(tmp_path / "truth-log")
    (truth_log / "truth.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(MeasurementLogValidationError, match="Truth"):
        load_measurement_log(truth_log)

    tampered = make_measurement_log(tmp_path / "tampered")
    (tampered / "environment.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MeasurementLogValidationError):
        load_measurement_log(tampered)

    alternate_truth = make_measurement_log(tmp_path / "alternate-truth")
    (alternate_truth / "source-layout-copy.bin").write_bytes(b"forbidden")
    with pytest.raises(MeasurementLogValidationError, match="Truth/source-layout"):
        load_measurement_log(alternate_truth)

    camel_case_truth = make_measurement_log(tmp_path / "camel-case-truth")
    (camel_case_truth / "sourceLayout-copy.bin").write_bytes(b"forbidden")
    with pytest.raises(MeasurementLogValidationError, match="Truth/source-layout"):
        load_measurement_log(camel_case_truth)

    valid = make_measurement_log(tmp_path / "dtype")
    log = load_measurement_log(valid)
    with np.load(valid / "observations.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["step_id"] = arrays["step_id"].astype(np.int32)
    metadata = [
        json.loads(line)
        for line in (valid / "observation_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with pytest.raises(MeasurementLogValidationError, match="dtype"):
        _records_from_arrays(
            arrays,
            metadata,
            TEST_ISOTOPES,
            run_id=log.run_id,
            record_count=len(log.records),
            energy_bin_count=4,
        )


def test_truth_hygiene_is_recursive_and_source_model_semantics_remain_allowed(
    tmp_path: Path,
) -> None:
    """Nested realized sources/pointers fail closed without banning physics fields."""
    base_record = records(1)[0]
    allowed = replace(
        base_record,
        metadata={
            "source_rate_model": "detector_cps_1m",
            "source_extent_radius_m": 0.05,
            "sourceRateModel": "detector_cps_1m",
            "sourceExtentRadiusM": 0.05,
        },
    )
    assert allowed.metadata["source_extent_radius_m"] == pytest.approx(0.05)
    assert allowed.metadata["sourceExtentRadiusM"] == pytest.approx(0.05)

    for metadata in (
        {"nested": [{"source_positions": [[1.0, 2.0, 3.0]]}]},
        {"nested": [{"sourcePositions": [[1.0, 2.0, 3.0]]}]},
        {"nested": {"sources": []}},
        {"note": "external/ground-truth.json"},
        {"groundTruth": {"sourceCount": 1}},
        {"point_sources": [{"position": [1.0, 2.0, 3.0]}]},
        {"pointSources": [{"position": [1.0, 2.0, 3.0]}]},
        {"sourceLayoutPath": "external/sourceLayout.json"},
        {"source_rate_ground_truth": "hidden"},
        {"source_extent_source_layout": "hidden"},
        {"source_rate_semantics": {"ground_truth": "hidden"}},
    ):
        with pytest.raises(MeasurementLogValidationError, match="realized truth"):
            replace(base_record, metadata=metadata)

    config = runtime_config()
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    with pytest.raises(MeasurementLogValidationError, match="must be null"):
        MeasurementLogStreamWriter(
            tmp_path / "source-pointer",
            run_id="source-pointer",
            repository_commit=TEST_COMMIT,
            runtime_config=config,
            environment=env,
            forward_model_manifest=forward,
            isotopes=TEST_ISOTOPES,
            source_layout_path="external/sources.json",
        )
    with pytest.raises(MeasurementLogValidationError, match="realized truth"):
        MeasurementLogStreamWriter(
            tmp_path / "nested-run-metadata",
            run_id="nested-run-metadata",
            repository_commit=TEST_COMMIT,
            runtime_config=config,
            environment=env,
            forward_model_manifest=forward,
            isotopes=TEST_ISOTOPES,
            metadata={"evaluation": {"truth_location": "elsewhere"}},
        )


def test_stream_writer_stages_record_before_estimator_update(
    tmp_path: Path,
) -> None:
    """A durable record shard must exist before the PF update is allowed."""
    config = runtime_config()
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    writer = MeasurementLogStreamWriter(
        tmp_path / "stream-log",
        run_id="stream-test",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
    )
    record = records(1)[0]

    writer.append_before_update(record)

    shard = writer.stage_dir / "record_00000000.npz"
    assert shard.is_file()
    assert shard.stat().st_size > 0

    class Estimator:
        def update(self) -> None:
            """Assert the durable shard exists before an estimator update."""
            assert shard.is_file()
            assert len(writer.records) == 1

    Estimator().update()
    finalized = writer.finalize()
    assert len(finalized.records) == 1
    assert not writer.stage_dir.exists()


def test_stream_writer_owns_and_durably_persists_station_boundaries(
    tmp_path: Path,
) -> None:
    """Only the writer may mark a completed station immediately before update."""
    config = {
        **runtime_config(),
        "joint_observation_update": True,
        "delayed_resample_update": False,
    }
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    writer = MeasurementLogStreamWriter(
        tmp_path / "station-log",
        run_id="station-test",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
    )
    first, second, third = records(3)
    with pytest.raises(MeasurementLogValidationError, match="writer-owned"):
        writer.append_before_update(replace(first, metadata={"station_complete": True}))

    writer.append_before_update(first)
    writer.append_before_update(second)
    with pytest.raises(MeasurementLogValidationError, match="marked complete"):
        writer.append_before_update(third)
    marked_index = writer.mark_station_complete_before_update(0)
    assert marked_index == 1
    staged_rows = [
        json.loads(line)
        for line in writer.metadata_stage_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "station_complete" not in staged_rows[0]["metadata"]
    assert staged_rows[1]["metadata"]["station_complete"] is True

    writer.append_before_update(third)
    with pytest.raises(MeasurementLogValidationError, match="exactly one"):
        writer.finalize()
    writer.mark_station_complete_before_update(1)
    finalized = writer.finalize()
    assert finalized.records[1].metadata["station_complete"] is True
    assert finalized.records[2].metadata["station_complete"] is True


def test_live_runtime_appends_each_record_before_any_pf_update() -> None:
    """The active live loop must stage observations before direct or joint updates."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "realtime_demo.py"
    ).read_text(encoding="utf-8")
    append_index = source.index("measurement_log_writer.append_before_update(")
    marker_index = source.index(
        "measurement_log_writer.mark_station_complete_before_update(",
        append_index,
    )
    direct_update_index = source.index("estimator.update_pair(", append_index)
    joint_update_index = source.index("estimator.update_pair_sequence(", append_index)
    assert append_index < direct_update_index
    assert append_index < marker_index < joint_update_index


def test_forward_manifest_mutation_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    """Unknown spectral/physical identities must not gain a replay fallback."""
    config = runtime_config()
    env = environment()
    config_hash = sha256(canonical_json_bytes(config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=None,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    mutated = json.loads(json.dumps(forward))
    mutated["line_mu_by_isotope"]["Eu-154"][0]["energy_keV"] += 1.0
    writer = MeasurementLogStreamWriter(
        tmp_path / "bad-forward",
        run_id="bad-forward",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=mutated,
        isotopes=TEST_ISOTOPES,
    )
    writer.append_before_update(records(1)[0])
    with pytest.raises(MeasurementLogValidationError, match="Forward-model"):
        writer.finalize()
    assert not (tmp_path / "bad-forward").exists()
