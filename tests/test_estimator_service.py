"""Subprocess conformance tests for the fixed particle-filter service."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

from radiation_estimator_service_contracts import (
    FILE_SHA256_DIGEST_ALGORITHM,
    ArtifactRef,
    ArtifactTarget,
    Capabilities,
    DigestRef,
    ExecuteRequest,
    ExecuteResponse,
    MeasurementLogRef,
    NamedArtifactRef,
    canonical_json_bytes,
    digest_artifact_directory,
    file_uri_from_path,
    sha256_bytes,
    validate_artifact_ref,
)
from runtime.measurement_log import load_measurement_log
from runtime.prefix import measurement_records_digest

from pf.service import (
    ESTIMATE_OPERATION,
    ESTIMATOR_FAMILY,
    MEASUREMENT_LOG_CONTRACT,
    PF_CONFIG_CONTRACT,
    PF_RESULT_CONTRACT,
)
from pf.provenance import repository_commit
from tests.pure_pf_test_support import make_measurement_log, replay_config


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = ROOT.parent / "Rotating-shield-simulation-runtime" / "src"
SERVICE = Path(sys.executable).parent / "rotating-shield-pf-service"


def _service_environment() -> dict[str, str]:
    """Return an environment that uses current sibling worktrees."""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = ":".join(
        (RUNTIME_SRC.as_posix(), *((existing,) if existing else ()))
    )
    return environment


def _run_service(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed service entry point without a shell."""
    return subprocess.run(
        (SERVICE.as_posix(), *arguments),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env=_service_environment(),
    )


def _capabilities(tmp_path: Path) -> Capabilities:
    """Probe and parse one capability response through the executable."""
    response = tmp_path / "capabilities.json"
    completed = _run_service("capabilities", "--response", response.as_posix())
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    return Capabilities.from_json_bytes(response.read_bytes())


def _file_reference(path: Path) -> ArtifactRef:
    """Return one authenticated PF config reference."""
    payload = path.read_bytes()
    return ArtifactRef(
        uri=file_uri_from_path(path),
        kind="file",
        digest=DigestRef(
            algorithm=FILE_SHA256_DIGEST_ALGORITHM,
            value=sha256_bytes(payload),
        ),
        media_type="application/json",
        contract=PF_CONFIG_CONTRACT,
        size_bytes=len(payload),
    )


def _measurement_reference(path: Path) -> MeasurementLogRef:
    """Return a fully authenticated reference to one runtime log."""
    log = load_measurement_log(path)
    artifact_digest, artifact_size = digest_artifact_directory(path)
    records_digest = measurement_records_digest(log.records)
    return MeasurementLogRef(
        artifact=ArtifactRef(
            uri=file_uri_from_path(path),
            kind="directory",
            digest=artifact_digest,
            media_type="application/vnd.radiation.measurement-log",
            contract=MEASUREMENT_LOG_CONTRACT,
            size_bytes=artifact_size,
        ),
        schema_version=log.schema_version,
        run_id=log.run_id,
        record_count=len(log.records),
        terminal_step_id=log.records[-1].step_id,
        records_digest=DigestRef(
            algorithm=records_digest.algorithm,
            value=records_digest.sha256,
        ),
    )


def _request(tmp_path: Path, capabilities: Capabilities) -> ExecuteRequest:
    """Create one small deterministic service request."""
    log_path = make_measurement_log(
        tmp_path / "measurement-log",
        record_count=1,
        station_complete_markers=True,
    )
    config_path = tmp_path / "pf-config.json"
    config_path.write_bytes(canonical_json_bytes(replay_config()))
    return ExecuteRequest(
        request_id="particle-filter-service:test-1",
        estimator_family=ESTIMATOR_FAMILY,
        operation=ESTIMATE_OPERATION,
        measurement_log=_measurement_reference(log_path),
        config=_file_reference(config_path),
        random_seed=17,
        input_artifacts=(),
        output=ArtifactTarget(file_uri_from_path(tmp_path / "pf-result")),
        expected_capabilities_digest=capabilities.digest,
        requested_result_contract=PF_RESULT_CONTRACT,
    )


def _execute_request(
    tmp_path: Path,
    request: ExecuteRequest,
    *,
    name: str = "execute",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Persist and execute one request through the real service process."""
    request_path = tmp_path / f"{name}-request.json"
    response_path = tmp_path / f"{name}-response.json"
    request_path.write_bytes(request.to_json_bytes())
    completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        response_path.as_posix(),
    )
    return completed, response_path


def test_capabilities_advertise_one_truth_free_pf_contract(tmp_path: Path) -> None:
    """The executable must expose exactly one explicit PF operation."""
    capabilities = _capabilities(tmp_path)

    assert capabilities.estimator_family == ESTIMATOR_FAMILY
    assert capabilities.accepts_truth is False
    assert capabilities.implementation.revision == repository_commit()
    assert capabilities.measurement_log_schema_versions == (2,)
    assert len(capabilities.operations) == 1
    operation = capabilities.operations[0]
    assert operation.operation == ESTIMATE_OPERATION
    assert operation.config_contracts == (PF_CONFIG_CONTRACT,)
    assert operation.result_contracts == (PF_RESULT_CONTRACT,)
    assert operation.required_input_roles == ()
    assert operation.optional_input_roles == ()


def test_execute_publishes_authenticated_existing_pf_replay(tmp_path: Path) -> None:
    """A valid request must return the existing PF replay artifact contract."""
    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)

    completed, response_path = _execute_request(tmp_path, request)

    assert completed.returncode == 0, completed.stderr
    assert "[joint-smc]" in completed.stdout
    response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
    assert response.status == "succeeded"
    assert response.error is None
    assert response.result_artifact is not None
    assert response.result_artifact.contract == PF_RESULT_CONTRACT
    result_path = validate_artifact_ref(response.result_artifact)
    assert result_path == tmp_path / "pf-result"
    assert {artifact.role for artifact in response.artifacts} == {
        "diagnostics",
        "posterior",
        "trace",
    }
    assert all(validate_artifact_ref(item.artifact).is_file() for item in response.artifacts)
    assert {path.name for path in result_path.iterdir()} == {
        "pf_diagnostics.json",
        "pf_posterior.json",
        "pf_trace.jsonl",
    }
    assert not any("truth" in path.name.lower() for path in result_path.rglob("*"))


def test_execute_returns_typed_failures_for_mismatch_and_truth(tmp_path: Path) -> None:
    """Contract mismatches and undeclared truth inputs must produce no result."""
    capabilities = _capabilities(tmp_path)
    base = _request(tmp_path, capabilities)
    cases = {
        "capabilities": replace(
            base,
            expected_capabilities_digest=DigestRef(
                algorithm=base.expected_capabilities_digest.algorithm,
                value="0" * 64,
            ),
        ),
        "log-identity": replace(
            base,
            measurement_log=replace(
                base.measurement_log,
                record_count=base.measurement_log.record_count + 1,
            ),
        ),
        "truth": replace(
            base,
            input_artifacts=(
                NamedArtifactRef(role="source-truth", artifact=base.config),
            ),
        ),
    }
    for name, request in cases.items():
        request = replace(
            request,
            output=ArtifactTarget(file_uri_from_path(tmp_path / f"{name}-result")),
        )
        completed, response_path = _execute_request(tmp_path, request, name=name)
        assert completed.returncode == 1
        response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
        assert response.status == "failed"
        assert response.result_artifact is None
        assert response.artifacts == ()
        assert response.error is not None
        assert response.capabilities_digest == capabilities.digest
        if name == "capabilities":
            assert response.capabilities_digest != (
                request.expected_capabilities_digest
            )
        assert not (tmp_path / f"{name}-result").exists()


def test_execute_rejects_config_inheritance_and_embedded_truth(tmp_path: Path) -> None:
    """Service configs must not open undeclared files or carry realized truth."""
    capabilities = _capabilities(tmp_path)
    base = _request(tmp_path, capabilities)
    for name, payload in (
        ("extends", {**replay_config(), "extends": "/tmp/parent.json"}),
        ("embedded-truth", {**replay_config(), "source_truth": {"x": 1.0}}),
        (
            "truth-under-display-mode",
            {
                **replay_config(),
                "cui_truth_display_mode": {"source_truth": {"x": 1.0}},
            },
        ),
        (
            "live-truth-display",
            {**replay_config(), "cui_truth_display_mode": "evaluation_live"},
        ),
        (
            "undeclared-data",
            {
                **replay_config(),
                "auxiliary_blob": {
                    "coordinates": [[1.0, 2.0, 3.0]],
                    "rates": [42.0],
                },
            },
        ),
        (
            "ignored-physical-bound",
            {
                **replay_config(),
                "position_max": {
                    "coordinates": [[1.0, 2.0, 3.0]],
                    "rates": [42.0],
                },
            },
        ),
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(canonical_json_bytes(payload))
        request = replace(
            base,
            config=_file_reference(path),
            output=ArtifactTarget(file_uri_from_path(tmp_path / f"{name}-result")),
        )
        completed, response_path = _execute_request(tmp_path, request, name=name)
        assert completed.returncode == 1
        response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
        assert response.status == "failed"
        assert response.result_artifact is None
        assert response.error is not None


def test_service_rejects_non_contract_cli_shapes_and_request_links(
    tmp_path: Path,
) -> None:
    """Relative paths, shell arguments, and symlinked requests must fail closed."""
    relative = _run_service("capabilities", "--response", "capabilities.json")
    assert relative.returncode == 64

    response_path = tmp_path / "capabilities.json"
    extra = _run_service(
        "capabilities",
        "--response",
        response_path.as_posix(),
        "--command-template",
        "touch SHOULD_NOT_EXIST",
    )
    assert extra.returncode == 64
    assert not response_path.exists()
    assert not (ROOT / "SHOULD_NOT_EXIST").exists()

    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request.to_json_bytes())
    request_link = tmp_path / "request-link.json"
    request_link.symlink_to(request_path)
    linked_response = tmp_path / "linked-response.json"
    linked = _run_service(
        "execute",
        "--request",
        request_link.as_posix(),
        "--response",
        linked_response.as_posix(),
    )
    assert linked.returncode == 65
    assert not linked_response.exists()


def test_service_preserves_existing_targets_and_immutable_log(
    tmp_path: Path,
) -> None:
    """Output and response paths must not replace or mutate input artifacts."""
    capabilities = _capabilities(tmp_path)
    base = _request(tmp_path, capabilities)
    existing_output = tmp_path / "pf-result"
    existing_output.mkdir()
    sentinel = existing_output / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    completed, response_path = _execute_request(tmp_path, base, name="existing")

    assert completed.returncode == 1
    response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
    assert response.status == "failed"
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"

    log_path = validate_artifact_ref(base.measurement_log.artifact)
    before_digest, before_size = digest_artifact_directory(log_path)
    nested_request = replace(
        base,
        output=ArtifactTarget(file_uri_from_path(log_path / "pf-result")),
    )
    nested_completed, nested_response = _execute_request(
        tmp_path,
        nested_request,
        name="nested-output",
    )
    assert nested_completed.returncode == 1
    assert ExecuteResponse.from_json_bytes(nested_response.read_bytes()).status == (
        "failed"
    )
    assert digest_artifact_directory(log_path) == (before_digest, before_size)


def test_service_rejects_response_collisions_without_writing_a_dto(
    tmp_path: Path,
) -> None:
    """Transport paths must not collide with output or MeasurementLog storage."""
    capabilities = _capabilities(tmp_path)
    base = _request(tmp_path, capabilities)
    response_path = tmp_path / "colliding-response.json"
    request = replace(
        base,
        output=ArtifactTarget(file_uri_from_path(response_path)),
    )
    request_path = tmp_path / "colliding-request.json"
    request_path.write_bytes(request.to_json_bytes())

    completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        response_path.as_posix(),
    )

    assert completed.returncode == 65
    assert not response_path.exists()

    log_path = validate_artifact_ref(base.measurement_log.artifact)
    before = digest_artifact_directory(log_path)
    same_log = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        log_path.as_posix(),
    )

    assert same_log.returncode == 65
    assert digest_artifact_directory(log_path) == before

    request_path = tmp_path / "inside-log-request.json"
    request_path.write_bytes(base.to_json_bytes())
    forbidden_response = log_path / "service-response.json"
    inside_completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        forbidden_response.as_posix(),
    )
    assert inside_completed.returncode == 65
    assert not forbidden_response.exists()
    assert digest_artifact_directory(log_path) == before


def test_malformed_request_does_not_claim_a_typed_execution_result(
    tmp_path: Path,
) -> None:
    """A request without a trustworthy identity must not produce a response DTO."""
    request_path = tmp_path / "malformed.json"
    request_path.write_bytes(b'{"schema_version": 1, "schema_version": 1}\n')
    response_path = tmp_path / "response.json"

    completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        response_path.as_posix(),
    )

    assert completed.returncode == 65
    assert not response_path.exists()
