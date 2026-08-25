"""Production provenance and sealed execution boundary for control policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from runtime.provenance import strict_canonical_json_bytes, strict_json_loads


@dataclass(frozen=True, slots=True)
class PFExternalPathSelection:
    """Describe one externally selected candidate pose without experiment labels."""

    next_pose: NDArray[np.float64]
    candidate_index: int
    score: float
    policy_name: str


@dataclass(frozen=True, slots=True)
class PFControlPolicyProvenance:
    """Seal exact control-policy content before a live runtime connection."""

    policy_family: str
    source_sha256: str | None
    canonical_sha256: str
    canonical_policy_json: bytes
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate the exact policy family, bytes, and both digest identities."""
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Control-policy provenance schema_version must be 1.")
        if self.policy_family not in {"native_dss_pp", "ral_ablation"}:
            raise ValueError("Unknown control-policy provenance family.")
        if self.source_sha256 is not None:
            _validate_sha256(self.source_sha256, name="source_sha256")
        _validate_sha256(self.canonical_sha256, name="canonical_sha256")
        if not isinstance(self.canonical_policy_json, bytes):
            raise TypeError("canonical_policy_json must be immutable bytes.")
        if sha256(self.canonical_policy_json).hexdigest() != self.canonical_sha256:
            raise ValueError("Canonical control-policy digest differs from its bytes.")
        try:
            policy = strict_json_loads(self.canonical_policy_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("Canonical control-policy JSON is invalid.") from exc
        if strict_canonical_json_bytes(policy) != self.canonical_policy_json:
            raise ValueError("Control-policy content is not strict canonical JSON.")
        if self.policy_family == "native_dss_pp":
            if self.source_sha256 is not None or policy is not None:
                raise ValueError(
                    "Native DSS-PP policy provenance requires null source/content."
                )
        elif self.source_sha256 is None or not isinstance(policy, Mapping):
            raise ValueError(
                "RA-L policy provenance requires a source digest and policy object."
            )

    @classmethod
    def native_dss_pp(cls) -> "PFControlPolicyProvenance":
        """Return the explicit provenance descriptor for native DSS-PP control."""
        canonical = strict_canonical_json_bytes(None)
        return cls(
            policy_family="native_dss_pp",
            source_sha256=None,
            canonical_sha256=sha256(canonical).hexdigest(),
            canonical_policy_json=canonical,
        )

    def policy(self) -> object:
        """Return a detached copy of the exact canonical policy content."""
        return strict_json_loads(self.canonical_policy_json)

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON provenance including exact policy content."""
        return {
            "schema_version": 1,
            "policy_family": self.policy_family,
            "source_sha256": self.source_sha256,
            "canonical_sha256": self.canonical_sha256,
            "policy": self.policy(),
        }


def _validate_sha256(value: object, *, name: str) -> str:
    """Return a strict lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def validate_control_policy(
    policy: object | None,
) -> PFControlPolicyProvenance:
    """Accept only the official loader-sealed RA-L policy concrete type."""
    if policy is None:
        return PFControlPolicyProvenance.native_dss_pp()
    from baselines.ral_ablation.control_policy import RALControlPolicy

    if type(policy) is not RALControlPolicy:
        raise TypeError(
            "Production control_policy must be the exact loader-sealed "
            "RALControlPolicy type."
        )
    provenance = policy.provenance
    if provenance.policy_family == "native_dss_pp":
        raise ValueError(
            "Injected control_policy cannot claim native DSS-PP provenance."
        )
    if provenance.policy() != policy.to_payload():
        raise ValueError(
            "Executable control policy differs from its canonical provenance."
        )
    return provenance


__all__ = [
    "PFControlPolicyProvenance",
    "PFExternalPathSelection",
    "validate_control_policy",
]
