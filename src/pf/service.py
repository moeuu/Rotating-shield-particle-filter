"""Fixed two-verb service adapter for the local particle-filter replay.

The adapter authenticates transport-neutral inputs, delegates MeasurementLog
validation to the shared runtime, and invokes the existing PF replay entry path.
It neither accepts realized source truth nor owns process-observed receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from radiation_estimator_service_contracts import (
    Capabilities,
    ContractRef,
    DigestRef,
    EstimatorServiceContractError,
    ExecuteRequest,
    ExecuteResponse,
    ImplementationRef,
    MeasurementLogRef,
    NamedArtifactRef,
    OperationCapability,
    OperationRef,
    ServiceError,
    artifact_ref_from_path,
    parse_service_argv,
    path_from_file_uri,
    read_bounded_regular_file,
    strict_json_object_from_bytes,
    validate_artifact_ref,
    validate_artifact_target,
    validate_new_file_path,
    validate_request_against_capabilities,
    write_new_file,
)
from runtime import AtomicBundlePublisher
from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    load_measurement_log,
)
from runtime.prefix import measurement_records_digest
from runtime.records import validate_truth_free_estimator_input

from pf.provenance import repository_commit
from pf.replay import PF_SERVICE_CONFIG_KEYS, replay_measurement_log


ESTIMATOR_FAMILY = "particle-filter"
ESTIMATE_OPERATION = OperationRef("estimate", 1)
PF_CONFIG_CONTRACT = ContractRef("radiation.pf-config", 1)
PF_RESULT_CONTRACT = ContractRef("radiation.pf-result", 1)
MEASUREMENT_LOG_CONTRACT = ContractRef(
    "runtime.measurement-log",
    MEASUREMENT_LOG_SCHEMA_VERSION,
)
PF_RESULT_MEDIA_TYPE = "application/vnd.radiation.pf-result"
PF_RESOLVED_CONFIG_DIGEST_ALGORITHM = (
    "rotating-shield-pf.resolved-config-v1+canonical-json-sha256"
)
_DISTRIBUTION = "rotating-shield-particle-filter"
_MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
_RESULT_MEMBERS = (
    ("diagnostics", "pf_diagnostics.json", "application/json"),
    ("posterior", "pf_posterior.json", "application/json"),
    ("trace", "pf_trace.jsonl", "application/x-ndjson"),
)


class PFServiceError(ValueError):
    """Report one controlled adapter failure without exposing solver internals."""


def _implementation_version() -> str:
    """Return the installed PF distribution version."""
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "0.1.0"


def _implementation_revision() -> str | None:
    """Return the full PF source revision when Git provenance is available."""
    revision = repository_commit()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        return None
    return revision


def service_capabilities() -> Capabilities:
    """Return the immutable truth-free operation supported by this service."""
    return Capabilities(
        estimator_family=ESTIMATOR_FAMILY,
        implementation=ImplementationRef(
            distribution=_DISTRIBUTION,
            version=_implementation_version(),
            revision=_implementation_revision(),
        ),
        operations=(
            OperationCapability(
                operation=ESTIMATE_OPERATION,
                config_contracts=(PF_CONFIG_CONTRACT,),
                result_contracts=(PF_RESULT_CONTRACT,),
                required_input_roles=(),
                optional_input_roles=(),
            ),
        ),
        measurement_log_schema_versions=(MEASUREMENT_LOG_SCHEMA_VERSION,),
        accepts_truth=False,
    )


def _validate_measurement_identity(
    reference: MeasurementLogRef,
    log: MeasurementLog,
) -> None:
    """Bind the wire MeasurementLog identity to runtime-validated records."""
    if reference.artifact.contract != MEASUREMENT_LOG_CONTRACT:
        raise PFServiceError("MeasurementLog contract is unsupported.")
    if reference.schema_version != log.schema_version:
        raise PFServiceError("MeasurementLog schema version differs from its content.")
    if reference.run_id != log.run_id:
        raise PFServiceError("MeasurementLog run_id differs from its content.")
    if reference.record_count != len(log.records):
        raise PFServiceError("MeasurementLog record_count differs from its content.")
    if reference.terminal_step_id != log.records[-1].step_id:
        raise PFServiceError("MeasurementLog terminal_step_id differs from its content.")
    records_digest = measurement_records_digest(log.records)
    if (
        reference.records_digest.algorithm != records_digest.algorithm
        or reference.records_digest.value != records_digest.sha256
    ):
        raise PFServiceError("MeasurementLog records digest differs from its content.")


def _authenticate_measurement_log(
    reference: MeasurementLogRef,
) -> tuple[Path, MeasurementLog]:
    """Authenticate and load one truth-free MeasurementLog through runtime APIs."""
    path = validate_artifact_ref(reference.artifact)
    log = load_measurement_log(path)
    _validate_measurement_identity(reference, log)
    validate_artifact_ref(reference.artifact)
    return path, log


def _authenticated_config(request: ExecuteRequest) -> Mapping[str, object]:
    """Authenticate one self-contained truth-free PF replay configuration."""
    path = validate_artifact_ref(request.config)
    payload_bytes = read_bounded_regular_file(
        path,
        maximum_bytes=_MAX_CONTROL_FILE_BYTES,
    )
    if sha256(payload_bytes).hexdigest() != request.config.digest.value:
        raise PFServiceError("PF config changed after artifact authentication.")
    if request.config.size_bytes is not None and len(payload_bytes) != (
        request.config.size_bytes
    ):
        raise PFServiceError("PF config size differs from its reference.")
    payload = strict_json_object_from_bytes(payload_bytes)
    if "extends" in payload:
        raise PFServiceError(
            "Service PF config must be self-contained and cannot use extends."
        )
    unknown_keys = sorted(set(payload) - PF_SERVICE_CONFIG_KEYS)
    if unknown_keys:
        raise PFServiceError(
            "Service PF config contains unsupported fields: "
            + ", ".join(unknown_keys)
        )
    truth_display_mode = payload.get("cui_truth_display_mode", "hidden")
    if truth_display_mode != "hidden":
        raise PFServiceError(
            "Service PF config requires cui_truth_display_mode='hidden'."
        )
    truth_scan = dict(payload)
    truth_scan.pop("cui_truth_display_mode", None)
    validate_truth_free_estimator_input(truth_scan, path="service.pf_config")
    return payload


def _promote_replay_result(
    publisher: AtomicBundlePublisher,
    replay_output: Path,
) -> None:
    """Copy replay files into a durable shared-runtime publication stage."""
    entries = tuple(sorted(replay_output.iterdir(), key=lambda item: item.name))
    if not entries or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise PFServiceError("PF replay produced an unsupported result bundle.")
    for entry in entries:
        publisher.copy_file(entry, entry.name)
    for entry in entries:
        entry.unlink()
    replay_output.rmdir()


def _result_artifacts(output_path: Path) -> tuple[NamedArtifactRef, ...]:
    """Return stable named references to each public PF replay result member."""
    artifacts: list[NamedArtifactRef] = []
    for role, filename, media_type in _RESULT_MEMBERS:
        path = output_path / filename
        artifacts.append(
            NamedArtifactRef(
                role=role,
                artifact=artifact_ref_from_path(path, media_type=media_type),
            )
        )
    return tuple(artifacts)


def _successful_response(
    request: ExecuteRequest,
    capabilities: Capabilities,
    *,
    response_path: Path,
) -> ExecuteResponse:
    """Invoke existing PF replay and attest its opaque result directory."""
    validate_request_against_capabilities(request, capabilities)
    output_path = validate_artifact_target(request.output)
    log_path, log = _authenticate_measurement_log(request.measurement_log)
    if output_path.is_relative_to(log_path):
        raise PFServiceError("PF result cannot be written inside MeasurementLog.")
    if output_path == response_path:
        raise PFServiceError("PF result and service response paths must differ.")
    config = _authenticated_config(request)
    with AtomicBundlePublisher(output_path, policy="create") as publisher:
        replay_output = publisher.staging_path / "replay-result"
        estimator, _ = replay_measurement_log(
            log,
            config,
            profile="pf_strict",
            seed=request.random_seed,
            output_dir=replay_output,
        )
        validate_artifact_ref(request.measurement_log.artifact)
        if estimator.measurement_log_sha256 != log.log_sha256:
            raise PFServiceError(
                "PF replay result is bound to a different MeasurementLog."
            )
        _promote_replay_result(publisher, replay_output)
        publisher.publish()
    result_artifact = artifact_ref_from_path(
        output_path,
        media_type=PF_RESULT_MEDIA_TYPE,
        contract=request.requested_result_contract,
    )
    return ExecuteResponse(
        request_id=request.request_id,
        request_digest=request.digest,
        estimator_family=request.estimator_family,
        operation=request.operation,
        capabilities_digest=capabilities.digest,
        status="succeeded",
        resolved_config_digest=DigestRef(
            algorithm=PF_RESOLVED_CONFIG_DIGEST_ALGORITHM,
            value=estimator.resolved_config_hash,
        ),
        result_artifact=result_artifact,
        artifacts=_result_artifacts(output_path),
        error=None,
    )


def _failed_response(
    request: ExecuteRequest,
    error: Exception,
    capabilities: Capabilities,
) -> ExecuteResponse:
    """Return one bounded failure response without artifact attestation."""
    message = " ".join(str(error).splitlines())[:2048] or type(error).__name__
    return ExecuteResponse(
        request_id=request.request_id,
        request_digest=request.digest,
        estimator_family=request.estimator_family,
        operation=request.operation,
        capabilities_digest=capabilities.digest,
        status="failed",
        resolved_config_digest=None,
        result_artifact=None,
        artifacts=(),
        error=ServiceError(
            code="request-rejected",
            message=message,
            retryable=False,
        ),
    )


def _execute(request_path: Path, response_path: Path) -> int:
    """Decode, execute, and persist one authenticated estimator request."""
    try:
        request = ExecuteRequest.from_json_bytes(
            read_bounded_regular_file(
                request_path,
                maximum_bytes=_MAX_CONTROL_FILE_BYTES,
            )
        )
    except Exception as exc:
        print(f"rotating-shield-pf-service: invalid request: {exc}", file=sys.stderr)
        return 65
    measurement_path = path_from_file_uri(request.measurement_log.artifact.uri)
    output_path = path_from_file_uri(request.output.uri)
    if response_path.is_relative_to(measurement_path):
        print(
            "rotating-shield-pf-service: invalid request: service response "
            "cannot be written inside MeasurementLog.",
            file=sys.stderr,
        )
        return 65
    if output_path == response_path:
        print(
            "rotating-shield-pf-service: invalid request: service response "
            "and estimator output paths must differ.",
            file=sys.stderr,
        )
        return 65
    validate_new_file_path(response_path)
    capabilities = service_capabilities()
    try:
        response = _successful_response(
            request,
            capabilities,
            response_path=response_path,
        )
    except Exception as exc:
        response = _failed_response(request, exc, capabilities)
        write_new_file(response_path, response.to_json_bytes())
        assert response.error is not None
        print(f"rotating-shield-pf-service: {response.error.message}", file=sys.stderr)
        return 1
    write_new_file(response_path, response.to_json_bytes())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run exactly the shared capabilities or execute service invocation."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        invocation = parse_service_argv(arguments)
        if invocation.verb == "capabilities":
            validate_new_file_path(invocation.response_path)
            write_new_file(
                invocation.response_path,
                service_capabilities().to_json_bytes(),
            )
            return 0
        assert invocation.request_path is not None
        return _execute(invocation.request_path, invocation.response_path)
    except (EstimatorServiceContractError, PFServiceError, OSError) as exc:
        print(f"rotating-shield-pf-service: {exc}", file=sys.stderr)
        return 64


__all__ = [
    "ESTIMATE_OPERATION",
    "ESTIMATOR_FAMILY",
    "MEASUREMENT_LOG_CONTRACT",
    "PF_CONFIG_CONTRACT",
    "PF_RESULT_CONTRACT",
    "main",
    "service_capabilities",
]


if __name__ == "__main__":
    raise SystemExit(main())
