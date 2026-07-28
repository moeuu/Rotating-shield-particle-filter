"""Shared protocol objects for simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real
from typing import Any


def _strict_nonnegative_integer(value: object, *, name: str) -> int:
    """Return one exact nonnegative integer without string coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return resolved


def _strict_finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    """Return one finite real number satisfying an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and (
        resolved <= minimum if minimum_exclusive else resolved < minimum
    ):
        qualifier = "greater than" if minimum_exclusive else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}.")
    return resolved


def _strict_vector(
    value: object,
    *,
    name: str,
    length: int,
) -> tuple[float, ...]:
    """Return one fixed-length finite numeric vector."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} coordinates.")
    return tuple(
        _strict_finite_number(entry, name=f"{name}[{index}]")
        for index, entry in enumerate(value)
    )


def _strict_orientation_index(value: object, *, name: str) -> int:
    """Return one exact zero-based spherical-octant index."""
    resolved = _strict_nonnegative_integer(value, name=name)
    if resolved >= 8:
        raise ValueError(f"{name} must lie in [0, 8).")
    return resolved


def _validate_json_value(value: object, *, path: str) -> None:
    """Reject values that JSON would coerce or represent non-portably."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, Integral) and not isinstance(value, bool):
        if abs(int(value)) > 2**53:
            raise ValueError(f"{path} exceeds the exact JSON integer range.")
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite.")
        return
    if isinstance(value, list):
        for index, entry in enumerate(value):
            _validate_json_value(entry, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, entry in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            _validate_json_value(entry, path=f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains non-JSON-native value {type(value).__name__}."
    )


def _decode_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard NaN and infinity JSON constants."""
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _require_exact_object_keys(
    payload: dict[str, Any],
    *,
    name: str,
    expected: frozenset[str],
) -> None:
    """Reject missing or unknown fields in one versioned wire object."""
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        raise ValueError(
            f"{name} fields disagree with the wire schema; "
            f"missing={missing}, unknown={unknown}."
        )


@dataclass(frozen=True)
class SimulationCommand:
    """Describe a single simulator step request."""

    step_id: int
    target_pose_xyz: tuple[float, float, float]
    target_base_yaw_rad: float
    fe_orientation_index: int
    pb_orientation_index: int
    dwell_time_s: float
    travel_time_s: float = 0.0
    shield_actuation_time_s: float = 0.0
    travel_waypoints_xyz: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Validate and canonicalize the command before any backend sees it."""
        object.__setattr__(
            self,
            "step_id",
            _strict_nonnegative_integer(self.step_id, name="step_id"),
        )
        object.__setattr__(
            self,
            "target_pose_xyz",
            _strict_vector(
                self.target_pose_xyz,
                name="target_pose_xyz",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "target_base_yaw_rad",
            _strict_finite_number(
                self.target_base_yaw_rad,
                name="target_base_yaw_rad",
            ),
        )
        object.__setattr__(
            self,
            "fe_orientation_index",
            _strict_orientation_index(
                self.fe_orientation_index,
                name="fe_orientation_index",
            ),
        )
        object.__setattr__(
            self,
            "pb_orientation_index",
            _strict_orientation_index(
                self.pb_orientation_index,
                name="pb_orientation_index",
            ),
        )
        object.__setattr__(
            self,
            "dwell_time_s",
            _strict_finite_number(
                self.dwell_time_s,
                name="dwell_time_s",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "travel_time_s",
            _strict_finite_number(
                self.travel_time_s,
                name="travel_time_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "shield_actuation_time_s",
            _strict_finite_number(
                self.shield_actuation_time_s,
                name="shield_actuation_time_s",
                minimum=0.0,
            ),
        )
        if self.travel_waypoints_xyz is not None:
            if not isinstance(self.travel_waypoints_xyz, (list, tuple)):
                raise TypeError("travel_waypoints_xyz must be a sequence or null.")
            object.__setattr__(
                self,
                "travel_waypoints_xyz",
                tuple(
                    _strict_vector(
                        waypoint,
                        name=f"travel_waypoints_xyz[{index}]",
                        length=3,
                    )
                    for index, waypoint in enumerate(
                        self.travel_waypoints_xyz
                    )
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the command."""
        return {
            "step_id": int(self.step_id),
            "target_pose_xyz": [float(v) for v in self.target_pose_xyz],
            "target_base_yaw_rad": float(self.target_base_yaw_rad),
            "fe_orientation_index": int(self.fe_orientation_index),
            "pb_orientation_index": int(self.pb_orientation_index),
            "dwell_time_s": float(self.dwell_time_s),
            "travel_time_s": float(self.travel_time_s),
            "shield_actuation_time_s": float(self.shield_actuation_time_s),
            "travel_waypoints_xyz": (
                None
                if self.travel_waypoints_xyz is None
                else [
                    [float(value) for value in waypoint]
                    for waypoint in self.travel_waypoints_xyz
                ]
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationCommand":
        """Build a command without silently coercing corrupt wire values."""
        if not isinstance(data, dict):
            raise TypeError("SimulationCommand payload must be an object.")
        _require_exact_object_keys(
            data,
            name="SimulationCommand",
            expected=frozenset(
                {
                    "step_id",
                    "target_pose_xyz",
                    "target_base_yaw_rad",
                    "fe_orientation_index",
                    "pb_orientation_index",
                    "dwell_time_s",
                    "travel_time_s",
                    "shield_actuation_time_s",
                    "travel_waypoints_xyz",
                }
            ),
        )
        return cls(
            step_id=data["step_id"],
            target_pose_xyz=data["target_pose_xyz"],
            target_base_yaw_rad=data["target_base_yaw_rad"],
            fe_orientation_index=data["fe_orientation_index"],
            pb_orientation_index=data["pb_orientation_index"],
            dwell_time_s=data["dwell_time_s"],
            travel_time_s=data["travel_time_s"],
            shield_actuation_time_s=data["shield_actuation_time_s"],
            travel_waypoints_xyz=data["travel_waypoints_xyz"],
        )


@dataclass(frozen=True)
class SimulationObservation:
    """Carry simulator output back to the estimator process."""

    step_id: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_orientation_index: int
    pb_orientation_index: int
    spectrum_counts: list[int | float]
    energy_bin_edges_keV: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and canonicalize simulator output at the wire boundary."""
        object.__setattr__(
            self,
            "step_id",
            _strict_nonnegative_integer(self.step_id, name="step_id"),
        )
        object.__setattr__(
            self,
            "detector_pose_xyz",
            _strict_vector(
                self.detector_pose_xyz,
                name="detector_pose_xyz",
                length=3,
            ),
        )
        quaternion = _strict_vector(
            self.detector_quat_wxyz,
            name="detector_quat_wxyz",
            length=4,
        )
        if math.sqrt(sum(value * value for value in quaternion)) <= 0.0:
            raise ValueError("detector_quat_wxyz must be nonzero.")
        object.__setattr__(self, "detector_quat_wxyz", quaternion)
        object.__setattr__(
            self,
            "fe_orientation_index",
            _strict_orientation_index(
                self.fe_orientation_index,
                name="fe_orientation_index",
            ),
        )
        object.__setattr__(
            self,
            "pb_orientation_index",
            _strict_orientation_index(
                self.pb_orientation_index,
                name="pb_orientation_index",
            ),
        )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be an object.")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings.")
        metadata = dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.spectrum_counts, (list, tuple)):
            raise TypeError("spectrum_counts must be a sequence.")
        sampled_event_counts = metadata.get(
            "detector_response_sampling_mode"
        ) == "multinomial_marking_with_nonparalyzable_event_time"
        if sampled_event_counts:
            spectrum: list[int | float] = []
            for index, value in enumerate(self.spectrum_counts):
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError(
                        "Sampled detector-response spectrum_counts must contain "
                        "integer event counts; "
                        f"spectrum_counts[{index}]={value!r} is not an integer."
                    )
                resolved = int(value)
                if resolved < 0 or resolved > 2**53:
                    raise ValueError(
                        "Sampled detector-response event counts must lie in "
                        "[0, 2**53]."
                    )
                spectrum.append(resolved)
        else:
            spectrum = [
                _strict_finite_number(
                    value,
                    name=f"spectrum_counts[{index}]",
                    minimum=0.0,
                )
                for index, value in enumerate(self.spectrum_counts)
            ]
        if not spectrum:
            raise ValueError("spectrum_counts must be nonempty.")
        object.__setattr__(self, "spectrum_counts", spectrum)
        if not isinstance(self.energy_bin_edges_keV, (list, tuple)):
            raise TypeError("energy_bin_edges_keV must be a sequence.")
        edges = [
            _strict_finite_number(
                value,
                name=f"energy_bin_edges_keV[{index}]",
            )
            for index, value in enumerate(self.energy_bin_edges_keV)
        ]
        if len(edges) != len(spectrum) + 1 or any(
            right <= left for left, right in zip(edges, edges[1:])
        ):
            raise ValueError(
                "energy_bin_edges_keV must be strictly increasing and contain "
                "one more value than spectrum_counts."
            )
        object.__setattr__(self, "energy_bin_edges_keV", edges)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the observation."""
        return {
            "step_id": int(self.step_id),
            "detector_pose_xyz": [float(v) for v in self.detector_pose_xyz],
            "detector_quat_wxyz": [float(v) for v in self.detector_quat_wxyz],
            "fe_orientation_index": int(self.fe_orientation_index),
            "pb_orientation_index": int(self.pb_orientation_index),
            "spectrum_counts": [
                int(value)
                if isinstance(value, Integral) and not isinstance(value, bool)
                else float(value)
                for value in self.spectrum_counts
            ],
            "energy_bin_edges_keV": [float(v) for v in self.energy_bin_edges_keV],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationObservation":
        """Build an observation without permissive numeric conversion."""
        if not isinstance(data, dict):
            raise TypeError("SimulationObservation payload must be an object.")
        _require_exact_object_keys(
            data,
            name="SimulationObservation",
            expected=frozenset(
                {
                    "step_id",
                    "detector_pose_xyz",
                    "detector_quat_wxyz",
                    "fe_orientation_index",
                    "pb_orientation_index",
                    "spectrum_counts",
                    "energy_bin_edges_keV",
                    "metadata",
                }
            ),
        )
        if not isinstance(data["spectrum_counts"], list):
            raise TypeError("spectrum_counts wire value must be a JSON array.")
        if not isinstance(data["energy_bin_edges_keV"], list):
            raise TypeError(
                "energy_bin_edges_keV wire value must be a JSON array."
            )
        if not isinstance(data["metadata"], dict):
            raise TypeError("metadata wire value must be a JSON object.")
        return cls(
            step_id=data["step_id"],
            detector_pose_xyz=data["detector_pose_xyz"],
            detector_quat_wxyz=data["detector_quat_wxyz"],
            fe_orientation_index=data["fe_orientation_index"],
            pb_orientation_index=data["pb_orientation_index"],
            spectrum_counts=data["spectrum_counts"],
            energy_bin_edges_keV=data["energy_bin_edges_keV"],
            metadata=data["metadata"],
        )


def encode_message(message_type: str, payload: dict[str, Any]) -> bytes:
    """Encode a message envelope as newline-delimited JSON bytes."""
    if not isinstance(message_type, str) or not message_type:
        raise TypeError("message_type must be a nonempty string.")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object.")
    _validate_json_value(payload, path="payload")
    envelope = {"type": message_type, "payload": payload}
    return (
        json.dumps(envelope, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def decode_message(data: bytes) -> tuple[str, dict[str, Any]]:
    """Decode a message envelope from newline-delimited JSON bytes."""
    if not isinstance(data, bytes):
        raise TypeError("Encoded simulator message must be bytes.")
    envelope = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_decode_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(envelope, dict):
        raise TypeError("Simulator message envelope must be a JSON object.")
    if set(envelope) != {"type", "payload"}:
        raise ValueError(
            "Simulator message envelope must contain exactly type and payload."
        )
    message_type = envelope["type"]
    payload = envelope["payload"]
    if not isinstance(message_type, str) or not message_type:
        raise TypeError("Simulator message type must be a nonempty string.")
    if not isinstance(payload, dict):
        raise TypeError("Simulator message payload must be a JSON object.")
    _validate_json_value(payload, path="payload")
    return message_type, payload
