"""RA-L-only control-policy adapter kept outside the generic PF package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from runtime.provenance import strict_canonical_json_bytes, strict_json_loads

from baselines.ral_ablation.shield_policies import (
    select_baseline_shield_program,
    validate_baseline_shield_policy,
)
from pf.control_policy import PFControlPolicyProvenance
from planning.dss_pp import ShieldProgram


class RALControlPolicyError(ValueError):
    """Report a malformed or incorrectly bound RA-L control policy."""


_RAL_CONTROL_POLICY_LOAD_TOKEN = object()
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
RAL_CONTROL_VARIANTS = (
    "proposed",
    "no_shield_native_path",
    "round_robin_shield",
    "eig_only_path",
)


def _ral_variant(value: object) -> str:
    """Return one exact current RA-L comparison variant name."""
    if not isinstance(value, str) or value not in RAL_CONTROL_VARIANTS:
        raise RALControlPolicyError(
            "RA-L control-policy variant must be one of "
            f"{list(RAL_CONTROL_VARIANTS)}."
        )
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    """Return one exact lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise RALControlPolicyError(
            f"{name} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def validate_ral_control_policy_payload(
    payload: Mapping[str, Any],
) -> dict[str, object]:
    """Validate and detach one exact version-2 RA-L policy document."""
    if not isinstance(payload, Mapping):
        raise TypeError("RA-L control policy must be a JSON object.")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError("RA-L control-policy keys must be JSON strings.")
    expected = {"schema_version", "variant", "shield_policy"}
    actual = set(payload)
    if actual != expected:
        raise RALControlPolicyError(
            "RA-L control policy must contain exactly schema_version, variant, "
            "and shield_policy; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 2:
        raise RALControlPolicyError(
            "RA-L control policy schema_version must be the JSON integer 2."
        )
    variant = _ral_variant(payload["variant"])
    shield_policy = validate_baseline_shield_policy(payload["shield_policy"])
    if variant == "round_robin_shield":
        if shield_policy is None or shield_policy.get("name") != "round_robin":
            raise RALControlPolicyError(
                "round_robin_shield requires the independent round-robin shield "
                "policy."
            )
    elif shield_policy is not None:
        raise RALControlPolicyError(
            f"RA-L variant {variant!r} must not inject a shield policy."
        )
    return {
        "schema_version": 2,
        "variant": variant,
        "shield_policy": shield_policy,
    }


def validate_ral_control_policy_pf_settings(
    policy: "RALControlPolicy",
    settings: Mapping[str, Any],
) -> None:
    """Reject active planner knobs that an external RA-L policy would ignore."""
    if not isinstance(policy, RALControlPolicy):
        raise TypeError("policy must be a validated RALControlPolicy.")
    if not isinstance(settings, Mapping):
        raise TypeError("PF settings must be a mapping.")
    payload = policy.to_payload()
    variant = str(payload["variant"])
    forced_shield = variant == "round_robin_shield"
    dss_pp = settings.get("dss_pp")
    planning_samples = settings.get("planning_eig_samples")
    if not isinstance(dss_pp, Mapping):
        raise RALControlPolicyError(
            "Every current RA-L variant requires the native DSS-PP path planner."
        )
    if (
        isinstance(planning_samples, bool)
        or not isinstance(planning_samples, int)
        or planning_samples < 2
    ):
        raise RALControlPolicyError(
            "Native-path RA-L control requires planning_eig_samples>=2."
        )
    if forced_shield:
        required_false = (
            "shield_view_count_shadow_enabled",
            "conditional_greedy_one_swap",
        )
        active = [field for field in required_false if dss_pp.get(field) is not False]
        if active:
            raise RALControlPolicyError(
                "Externally forced shield policies require explicit false sentinels "
                f"for inactive planner fields: {active}."
            )
    if variant == "eig_only_path":
        required_values: dict[str, object] = {
            "coverage_weight": 0.0,
            "coverage_floor_quantile": 0.0,
            "coverage_floor_weight": 0.0,
            "coverage_surface_max_hausdorff_m": None,
            "coverage_surface_quadrature_max_points": None,
            "exact_eig_coverage_reserve": 0,
            "bearing_diversity_weight": 0.0,
            "frontier_weight": 0.0,
            "local_orbit_weight": 0.0,
            "local_orbit_ring_radii_m": [],
            "local_orbit_sigma_m": None,
            "elevation_condition_weight": 0.0,
            "elevation_pair_xy_scale_m": None,
            "elevation_pair_z_scale_m": None,
            "elevation_angle_threshold_deg": None,
            "revisit_penalty_weight": 0.0,
            "turn_smoothness_weight": 0.0,
        }
        invalid = [
            field
            for field, expected_value in required_values.items()
            if dss_pp.get(field) != expected_value
        ]
        if invalid:
            raise RALControlPolicyError(
                "eig_only_path requires exact inactive sentinels for every "
                f"non-EIG spatial term: {invalid}."
            )


@dataclass(frozen=True, slots=True, init=False)
class RALControlPolicy:
    """Execute loader-sealed RA-L baseline choices in the live controller."""

    _variant: str = field(repr=False)
    _shield_policy: Mapping[str, Any] | None = field(repr=False)
    _provenance: PFControlPolicyProvenance | None = field(repr=False)

    def __init__(
        self,
        *,
        variant: str,
        shield_policy: Mapping[str, Any] | None,
        _provenance: PFControlPolicyProvenance | None = None,
        _loader_token: object | None = None,
    ) -> None:
        """Validate and retain immutable copies of the experiment-only policies."""
        if (_provenance is None) != (_loader_token is None):
            raise RALControlPolicyError(
                "Sealed RA-L policy construction requires provenance and its "
                "loader token together."
            )
        if (
            _loader_token is not None
            and _loader_token is not _RAL_CONTROL_POLICY_LOAD_TOKEN
        ):
            raise RALControlPolicyError(
                "Only load_ral_control_policy_document() may seal an executable "
                "RA-L control policy."
            )
        validated = validate_ral_control_policy_payload(
            {
                "schema_version": 2,
                "variant": variant,
                "shield_policy": shield_policy,
            }
        )
        validated_variant = str(validated["variant"])
        validated_shield = validated["shield_policy"]
        object.__setattr__(self, "_variant", validated_variant)
        object.__setattr__(
            self,
            "_shield_policy",
            (
                None
                if validated_shield is None
                else MappingProxyType(dict(validated_shield))
            ),
        )
        if _provenance is not None:
            if _provenance.policy_family != "ral_ablation" or (
                _provenance.policy() != validated
            ):
                raise RALControlPolicyError(
                    "Executable RA-L policy differs from its sealed provenance."
                )
        object.__setattr__(self, "_provenance", _provenance)

    @property
    def variant(self) -> str:
        """Return the exact sealed comparison variant."""
        return self._variant

    @property
    def provenance(self) -> PFControlPolicyProvenance:
        """Return loader-sealed provenance required by production live control."""
        if self._provenance is None:
            raise RALControlPolicyError(
                "Production RA-L control requires a loader-sealed policy document."
            )
        return self._provenance

    def to_payload(self) -> dict[str, object]:
        """Return a detached exact version-2 policy payload."""
        return {
            "schema_version": 2,
            "variant": self._variant,
            "shield_policy": (
                None if self._shield_policy is None else dict(self._shield_policy)
            ),
        }

    def validate_pf_settings(self, settings: Mapping[str, object]) -> None:
        """Fail when this policy would make configured planner knobs inactive."""
        validate_ral_control_policy_pf_settings(self, settings)

    def select_shield_program(
        self,
        *,
        total_pairs: int,
        program_length: int,
        pose_index: int,
        current_pair_id: int | None,
    ) -> ShieldProgram | None:
        """Return the declared RA-L shield baseline as a generic program."""
        selection = select_baseline_shield_program(
            self._shield_policy,
            total_pairs=total_pairs,
            program_length=program_length,
            pose_index=pose_index,
            current_pair_id=current_pair_id,
        )
        if selection is None:
            return None
        return ShieldProgram(
            name=selection.name,
            pair_ids=selection.pair_ids,
            kind="external_control",
        )

@dataclass(frozen=True, slots=True)
class RALControlPolicyDocument:
    """Bind a validated policy to immutable source and canonical identities."""

    source_path: Path
    source_bytes: bytes
    source_sha256: str
    canonical_policy_json: bytes
    canonical_sha256: str
    _loader_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Verify both byte identities and the canonical policy representation."""
        if self._loader_token is not _RAL_CONTROL_POLICY_LOAD_TOKEN:
            raise RALControlPolicyError(
                "RALControlPolicyDocument values may only be created by "
                "load_ral_control_policy_document()."
            )
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise RALControlPolicyError("Policy source_path must be absolute.")
        if not isinstance(self.source_bytes, bytes):
            raise TypeError("Policy source_bytes must be immutable bytes.")
        source_digest = _sha256_digest(self.source_sha256, name="source_sha256")
        if sha256(self.source_bytes).hexdigest() != source_digest:
            raise RALControlPolicyError(
                "RA-L control-policy source digest does not match its bytes."
            )
        if not isinstance(self.canonical_policy_json, bytes):
            raise TypeError("canonical_policy_json must be immutable bytes.")
        canonical_digest = _sha256_digest(
            self.canonical_sha256,
            name="canonical_sha256",
        )
        if sha256(self.canonical_policy_json).hexdigest() != canonical_digest:
            raise RALControlPolicyError(
                "RA-L control-policy canonical digest does not match its bytes."
            )
        try:
            canonical_payload = strict_json_loads(self.canonical_policy_json)
        except (TypeError, ValueError) as exc:
            raise RALControlPolicyError(
                "RA-L canonical control-policy JSON is invalid."
            ) from exc
        validated = validate_ral_control_policy_payload(canonical_payload)
        if strict_canonical_json_bytes(validated) != self.canonical_policy_json:
            raise RALControlPolicyError(
                "RA-L control-policy canonical payload is not canonical JSON."
            )

    def payload(self) -> dict[str, object]:
        """Return a detached validated copy of the canonical policy object."""
        payload = strict_json_loads(self.canonical_policy_json)
        if not isinstance(
            payload, Mapping
        ):  # pragma: no cover - constructor invariant.
            raise RALControlPolicyError("Canonical RA-L policy is not an object.")
        return validate_ral_control_policy_payload(payload)

    def policy(self) -> RALControlPolicy:
        """Return the immutable executable adapter for this exact document."""
        payload = self.payload()
        variant = payload["variant"]
        shield_policy = payload["shield_policy"]
        assert isinstance(variant, str)
        assert shield_policy is None or isinstance(shield_policy, Mapping)
        provenance = PFControlPolicyProvenance(
            policy_family="ral_ablation",
            source_sha256=self.source_sha256,
            canonical_sha256=self.canonical_sha256,
            canonical_policy_json=self.canonical_policy_json,
        )
        return RALControlPolicy(
            variant=variant,
            shield_policy=shield_policy,
            _provenance=provenance,
            _loader_token=_RAL_CONTROL_POLICY_LOAD_TOKEN,
        )


def load_ral_control_policy_document(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> RALControlPolicyDocument:
    """Load one strict policy and optionally bind it to an expected source digest."""
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise RALControlPolicyError(
            "RA-L control-policy source must not be a symbolic link."
        )
    target = unresolved.resolve()
    try:
        source_bytes = target.read_bytes()
    except OSError as exc:
        raise RALControlPolicyError(
            f"Cannot read RA-L control policy {target}."
        ) from exc
    source_digest = sha256(source_bytes).hexdigest()
    if expected_source_sha256 is not None:
        expected_digest = _sha256_digest(
            expected_source_sha256,
            name="expected_source_sha256",
        )
        if source_digest != expected_digest:
            raise RALControlPolicyError(
                "RA-L control-policy source digest differs from the expected "
                "variant-policy digest."
            )
    try:
        payload = strict_json_loads(source_bytes)
    except (TypeError, ValueError) as exc:
        raise RALControlPolicyError(
            f"Cannot parse strict RA-L control policy {target}."
        ) from exc
    validated = validate_ral_control_policy_payload(payload)
    canonical = strict_canonical_json_bytes(validated)
    return RALControlPolicyDocument(
        source_path=target,
        source_bytes=source_bytes,
        source_sha256=source_digest,
        canonical_policy_json=canonical,
        canonical_sha256=sha256(canonical).hexdigest(),
        _loader_token=_RAL_CONTROL_POLICY_LOAD_TOKEN,
    )


def load_ral_control_policy(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> RALControlPolicy:
    """Load one strict RA-L-only control policy from a separate artifact."""
    return load_ral_control_policy_document(
        path,
        expected_source_sha256=expected_source_sha256,
    ).policy()


__all__ = [
    "RAL_CONTROL_VARIANTS",
    "RALControlPolicy",
    "RALControlPolicyDocument",
    "RALControlPolicyError",
    "load_ral_control_policy",
    "load_ral_control_policy_document",
    "validate_ral_control_policy_pf_settings",
    "validate_ral_control_policy_payload",
]
