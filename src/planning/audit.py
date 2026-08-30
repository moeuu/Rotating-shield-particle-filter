"""Durable, truth-free audit records for PF action selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np
from runtime.artifacts import DurableJSONLWriter

from planning.dss_pp import DSSPPResult
from planning.program_types import ShieldProgram


SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES = (
    "particle_diversity_evidence_available_and_warning_absent",
    "smc_rejuvenation_wall_time_respected",
    "rejuvenation_mixing_complete",
    "structural_mixing_complete",
    "posterior_predictive_innovation_available_and_passed",
    "cardinality_not_at_upper_boundary_for_every_isotope",
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Return one mapping or fail on a malformed planner diagnostic."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _leader(
    leaders: Mapping[str, object],
    name: str,
) -> dict[str, object] | None:
    """Return one compact JSON-safe planner component leader."""
    value = leaders.get(name)
    if value is None:
        return None
    return _compact_ranked_action(
        _mapping(value, name=f"component_leaders.{name}"),
        name=f"component_leaders.{name}",
    )


def _compact_ranked_action(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, object]:
    """Keep only fields needed to audit one ranked pose/program action."""
    pose_xyz = _sequence(value.get("pose_xyz"), name=f"{name}.pose_xyz")
    pair_ids = _sequence(value.get("pair_ids"), name=f"{name}.pair_ids")
    pose = [float(item) for item in pose_xyz]
    pairs = [int(item) for item in pair_ids]
    if len(pose) != 3 or any(not math.isfinite(item) for item in pose):
        raise ValueError(f"{name} must contain one finite 3-D pose.")
    if any(item < 0 or item >= 64 for item in pairs):
        raise ValueError(f"{name} pair IDs must lie in the 64-pair domain.")
    return {
        "rank": int(value["rank"]),
        "pose_index": int(value["pose_index"]),
        "pose_xyz": pose,
        "program_name": str(value["program_name"]),
        "program_kind": str(value["program_kind"]),
        "pair_ids": pairs,
        "score": float(value["score"]),
        "information_gain": float(value["information_gain"]),
    }


def _shadow_action(
    raw: Mapping[str, object],
    *,
    score_field: str,
    require_unique_pairs: bool = True,
) -> dict[str, object]:
    """Normalize one compact hypothetical shadow action."""
    pose_xyz = raw.get("pose_xyz")
    pair_ids = raw.get("pair_ids")
    if not isinstance(pose_xyz, Sequence) or isinstance(pose_xyz, (str, bytes)):
        raise TypeError("Shadow action pose_xyz must be a sequence.")
    if not isinstance(pair_ids, Sequence) or isinstance(pair_ids, (str, bytes)):
        raise TypeError("Shadow action pair_ids must be a sequence.")
    selected_view_count = int(raw["selected_view_count"])
    pose_index = int(raw["pose_index"])
    resolved_pose = [float(value) for value in pose_xyz]
    resolved_pairs = [int(value) for value in pair_ids]
    if (
        pose_index < 0
        or len(resolved_pose) != 3
        or any(not math.isfinite(value) for value in resolved_pose)
    ):
        raise ValueError(
            "Shadow action pose must be a finite nonnegative-index 3-D pose."
        )
    if (
        selected_view_count < 1
        or len(resolved_pairs) != selected_view_count
        or any(value < 0 or value >= 64 for value in resolved_pairs)
        or (require_unique_pairs and len(set(resolved_pairs)) != len(resolved_pairs))
    ):
        raise ValueError("Shadow action view count and pair IDs disagree.")
    score = raw.get(score_field)
    information_gain = raw.get("information_gain_mean_nat")
    if score is not None and not math.isfinite(float(score)):
        raise ValueError("Shadow action score must be finite or None.")
    if information_gain is not None and (
        not math.isfinite(float(information_gain)) or float(information_gain) < 0.0
    ):
        raise ValueError("Shadow action information gain must be nonnegative finite.")
    return {
        "pose_index": pose_index,
        "pose_xyz": resolved_pose,
        "selected_view_count": selected_view_count,
        "pair_ids": resolved_pairs,
        "information_gain_mean_nat": (
            None if information_gain is None else float(information_gain)
        ),
        "selection_score": None if score is None else float(score),
    }


def _executed_shadow_action(
    *,
    pose_index: int,
    pose_xyz: Sequence[float],
    program: ShieldProgram,
    information_gain: float | None,
    score: float | None,
) -> dict[str, object]:
    """Return one normalized actually executed fixed-view action."""
    return _shadow_action(
        {
            "pose_index": int(pose_index),
            "pose_xyz": list(pose_xyz),
            "selected_view_count": int(len(program.pair_ids)),
            "pair_ids": list(program.pair_ids),
            "information_gain_mean_nat": information_gain,
            "pose_score": score,
        },
        score_field="pose_score",
        require_unique_pairs=False,
    )


def _unavailable_shadow_health(
    belief_after_station_id: int | None,
    *,
    reason: str,
) -> dict[str, object]:
    """Return one truth-free fail-closed health payload."""
    return {
        "available": False,
        "passed": False,
        "source_station_id": belief_after_station_id,
        "hard_failure_reasons": [str(reason)],
        "truth_used": False,
    }


def _validated_shadow_health(
    value: Mapping[str, object],
    *,
    belief_after_station_id: int | None,
) -> dict[str, object]:
    """Validate complete truth-free health before it can permit shortening."""
    health = dict(value)
    if health.get("source_station_id") != belief_after_station_id:
        raise ValueError("Posterior health and planner belief stations differ.")
    if health.get("truth_used") is not False:
        raise ValueError("Posterior health must explicitly be truth-free.")
    if health.get("policy_schema_version") != 1:
        raise ValueError("Posterior health policy schema must be version 1.")
    gates = health.get("hard_gate_contract")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        raise TypeError("Posterior health hard-gate contract must be a sequence.")
    if tuple(str(value) for value in gates) != SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES:
        raise ValueError("Posterior health hard-gate contract is incomplete.")
    available = health.get("available")
    passed = health.get("passed")
    if not isinstance(available, bool) or not isinstance(passed, bool):
        raise TypeError("Posterior health available/passed flags must be booleans.")
    reasons = health.get("hard_failure_reasons")
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        raise TypeError("Posterior health failure reasons must be a sequence.")
    resolved_reasons = [str(reason) for reason in reasons]
    if passed and (not available or resolved_reasons):
        raise ValueError("Passing posterior health must be available and reason-free.")
    if not passed and not resolved_reasons:
        raise ValueError("Failing posterior health must state at least one reason.")
    return {
        "available": available,
        "passed": passed,
        "source_station_id": belief_after_station_id,
        "hard_failure_reasons": resolved_reasons,
        "truth_used": False,
    }


def _validated_pose_rows(
    pose_indices: object,
    pose_xyz: object,
    *,
    name: str,
) -> list[int]:
    """Validate aligned finite 3-D pose rows at the durable audit boundary."""
    if not isinstance(pose_indices, Sequence) or isinstance(
        pose_indices,
        (str, bytes),
    ):
        raise TypeError(f"{name} pose indices must be a sequence.")
    if not isinstance(pose_xyz, Sequence) or isinstance(pose_xyz, (str, bytes)):
        raise TypeError(f"{name} pose coordinates must be a sequence.")
    raw_indices = np.asarray(pose_indices)
    if (
        raw_indices.ndim != 1
        or np.issubdtype(raw_indices.dtype, np.bool_)
        or not np.issubdtype(raw_indices.dtype, np.integer)
    ):
        raise ValueError(f"{name} pose indices must be an integer vector.")
    indices = np.asarray(raw_indices, dtype=np.int64)
    poses = np.asarray(pose_xyz, dtype=np.float64)
    if (
        poses.shape != (indices.size, 3)
        or np.any(~np.isfinite(poses))
        or np.any(indices < 0)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(f"{name} pose rows must be aligned finite 3-D values.")
    return [int(value) for value in indices]


def _validate_view_count_pair_rows(
    by_view_count: Mapping[str, object],
    *,
    candidate_view_counts: tuple[int, ...],
    pose_count: int,
    name: str,
) -> None:
    """Validate nested shield-pair rows before retaining only K=8 rows."""
    pairs_by_view_count: dict[int, np.ndarray] = {}
    for view_count in candidate_view_counts:
        count_payload = _mapping(
            by_view_count.get(str(view_count)),
            name=f"{name}.by_view_count[{view_count}]",
        )
        raw_pairs = np.asarray(count_payload.get("pair_ids"))
        if (
            raw_pairs.shape != (pose_count, view_count)
            or np.issubdtype(raw_pairs.dtype, np.bool_)
            or not np.issubdtype(raw_pairs.dtype, np.integer)
        ):
            raise ValueError(f"{name} shield pair rows must align with poses.")
        pairs = np.asarray(raw_pairs, dtype=np.int64)
        duplicated = np.any(
            np.diff(np.sort(pairs, axis=1), axis=1) == 0,
            axis=1,
        )
        if np.any(pairs < 0) or np.any(pairs >= 64) or np.any(duplicated):
            raise ValueError(f"{name} shield pair row violates the 64-pair domain.")
        pairs_by_view_count[int(view_count)] = pairs
    reference = pairs_by_view_count[int(candidate_view_counts[-1])]
    for view_count in candidate_view_counts[:-1]:
        if not np.array_equal(
            pairs_by_view_count[int(view_count)],
            reference[:, : int(view_count)],
        ):
            raise ValueError(f"{name} shield programs must be nested prefixes.")


def _float_vector(
    value: object,
    *,
    size: int,
    name: str,
    nonnegative: bool = False,
) -> list[float]:
    """Return one aligned finite vector as JSON-native floats."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be one aligned finite vector.")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return [float(item) for item in vector]


def _compact_seed_blocks(value: object, *, name: str) -> list[dict[str, object]]:
    """Remove repeated RNG stream labels while preserving resolved seeds."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    compact: list[dict[str, object]] = []
    for offset, raw in enumerate(value):
        block = _mapping(raw, name=f"{name}[{offset}]")
        pose_indices = block.get("pose_indices")
        if not isinstance(pose_indices, Sequence) or isinstance(
            pose_indices,
            (str, bytes),
        ):
            raise TypeError(f"{name} pose indices must be a sequence.")
        compact.append(
            {
                "seed": int(block["seed"]),
                "pose_indices": [int(item) for item in pose_indices],
                "samples_per_pose": int(block["samples_per_pose"]),
            }
        )
    return compact


def _compact_proxy_shadow(
    proxy: Mapping[str, object],
    *,
    candidate_view_counts: tuple[int, ...],
) -> dict[str, object]:
    """Keep all-pose proxy EIG while dropping derivable scores and prefixes."""
    status = str(proxy.get("status", "not_evaluated"))
    if status != "evaluated":
        compact: dict[str, object] = {"status": status}
        for key in (
            "executed_fixed_8_shortlist_pose_indices",
            "view_count_union_exact_pose_indices",
        ):
            raw_indices = proxy.get(key)
            if isinstance(raw_indices, Sequence) and not isinstance(
                raw_indices,
                (str, bytes),
            ):
                compact[key] = [int(item) for item in raw_indices]
        return compact

    pose_indices = _validated_pose_rows(
        proxy.get("pose_indices"),
        proxy.get("pose_xyz"),
        name="proxy shadow",
    )
    pose_count = len(pose_indices)
    by_view_count = _mapping(
        proxy.get("by_view_count"),
        name="proxy.by_view_count",
    )
    reference_count = int(candidate_view_counts[-1])
    compact_by_count: dict[str, object] = {}
    for view_count in candidate_view_counts:
        raw_count = _mapping(
            by_view_count.get(str(view_count)),
            name=f"proxy.by_view_count[{view_count}]",
        )
        count_payload: dict[str, object] = {
            "information_gain_mean_nat": _float_vector(
                raw_count.get("information_gain_mean_nat"),
                size=pose_count,
                name=f"proxy I_{view_count}",
                nonnegative=True,
            )
        }
        if view_count == reference_count:
            count_payload["pair_ids"] = [
                [int(item) for item in row]
                for row in _mapping_sequence(
                    raw_count.get("pair_ids"),
                    name="proxy reference pair rows",
                )
            ]
        compact_by_count[str(view_count)] = count_payload
    seed_blocks = _compact_seed_blocks(
        proxy.get("selection_seed_blocks"),
        name="proxy selection seed blocks",
    )
    if len(seed_blocks) != 1:
        raise ValueError("Proxy shadow must have one resolved selection seed.")
    return {
        "status": "evaluated",
        "particle_count": int(proxy["particle_count"]),
        "samples_per_pose": int(proxy["samples_per_seed"]),
        "pose_indices": pose_indices,
        "pose_xyz": [
            [float(item) for item in row]
            for row in _mapping_sequence(
                proxy.get("pose_xyz"),
                name="proxy pose rows",
            )
        ],
        "selection_seed": int(seed_blocks[0]["seed"]),
        "by_view_count": compact_by_count,
        "executed_fixed_8_shortlist_pose_indices": [
            int(item)
            for item in _sequence(
                proxy.get("executed_fixed_8_shortlist_pose_indices"),
                name="executed fixed-eight shortlist pose indices",
            )
        ],
        "view_count_union_exact_pose_indices": [
            int(item)
            for item in _sequence(
                proxy.get("view_count_union_exact_pose_indices"),
                name="view-count union exact pose indices",
            )
        ],
    }


def _sequence(value: object, *, name: str) -> Sequence[object]:
    """Return one non-string sequence or reject malformed diagnostics."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _mapping_sequence(value: object, *, name: str) -> Sequence[Sequence[object]]:
    """Return one sequence of non-string rows."""
    rows = _sequence(value, name=name)
    for row in rows:
        _sequence(row, name=f"{name} row")
    return rows  # type: ignore[return-value]


def _compact_exact_shadow(
    exact: Mapping[str, object],
    *,
    candidate_view_counts: tuple[int, ...],
    include_pose_xyz: bool,
) -> dict[str, object]:
    """Keep paired exact evidence and omit poses already stored by proxy."""
    pose_indices = _validated_pose_rows(
        exact.get("pose_indices"),
        exact.get("pose_xyz"),
        name="exact shadow",
    )
    pose_count = len(pose_indices)
    by_view_count = _mapping(exact.get("by_view_count"), name="exact.by_view_count")
    reference_count = int(candidate_view_counts[-1])
    compact_by_count: dict[str, object] = {}
    for view_count in candidate_view_counts:
        raw_count = _mapping(
            by_view_count.get(str(view_count)),
            name=f"exact.by_view_count[{view_count}]",
        )
        increment = _mapping(
            raw_count.get("nested_prefix_increment"),
            name=f"exact I_{view_count} marginal evidence",
        )
        count_payload: dict[str, object] = {
            "information_gain_mean_nat": _float_vector(
                raw_count.get("information_gain_mean_nat"),
                size=pose_count,
                name=f"exact I_{view_count}",
                nonnegative=True,
            ),
            "information_gain_standard_error_nat": _float_vector(
                raw_count.get("information_gain_standard_error_nat"),
                size=pose_count,
                name=f"exact I_{view_count} standard error",
                nonnegative=True,
            ),
            "marginal_information_gain": {
                "mean_nat": _float_vector(
                    increment.get("mean_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} increment",
                ),
                "paired_standard_error_nat": _float_vector(
                    increment.get("paired_standard_error_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} increment standard error",
                    nonnegative=True,
                ),
                "one_sided_mc_lcb_nat": _float_vector(
                    increment.get("one_sided_mc_lcb_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} increment LCB",
                ),
                "mean_nat_per_added_live_second": _float_vector(
                    increment.get("mean_nat_per_added_live_second"),
                    size=pose_count,
                    name=f"exact I_{view_count} increment rate",
                ),
            },
        }
        if view_count == reference_count:
            count_payload["pair_ids"] = [
                [int(item) for item in row]
                for row in _mapping_sequence(
                    raw_count.get("pair_ids"),
                    name="exact reference pair rows",
                )
            ]
        else:
            retention = _mapping(
                raw_count.get("retention_vs_reference"),
                name=f"exact I_{view_count} retention evidence",
            )
            count_payload["retention_vs_reference"] = {
                "paired_margin_mean_nat": _float_vector(
                    retention.get("paired_margin_mean_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} retention margin",
                ),
                "paired_margin_standard_error_nat": _float_vector(
                    retention.get("paired_margin_standard_error_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} retention standard error",
                    nonnegative=True,
                ),
                "paired_margin_one_sided_mc_lcb_nat": _float_vector(
                    retention.get("paired_margin_one_sided_mc_lcb_nat"),
                    size=pose_count,
                    name=f"exact I_{view_count} retention LCB",
                ),
            }
        compact_by_count[str(view_count)] = count_payload
    compact: dict[str, object] = {
        "status": "evaluated",
        "particle_count": int(exact["particle_count"]),
        "sample_count": int(exact["sample_count"]),
        "pose_indices": pose_indices,
        "prefix_selection_seed_blocks": _compact_seed_blocks(
            exact.get("prefix_selection_seed_blocks"),
            name="exact prefix selection seed blocks",
        ),
        "paired_holdout_seed_blocks": _compact_seed_blocks(
            exact.get("paired_evaluation_holdout_seed_blocks"),
            name="exact paired holdout seed blocks",
        ),
        "by_view_count": compact_by_count,
    }
    if include_pose_xyz:
        compact["pose_xyz"] = [
            [float(item) for item in row]
            for row in _mapping_sequence(
                exact.get("pose_xyz"),
                name="exact pose rows",
            )
        ]
    return compact


def _required_nonnegative_integer(
    values: Mapping[str, object],
    key: str,
    *,
    name: str,
) -> int:
    """Return one required exact nonnegative integer audit field."""
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}.{key} must be an integer.")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name}.{key} must be nonnegative.")
    return resolved


def build_planner_audit(
    *,
    station_id: int,
    result: DSSPPResult,
    top_k: int = 10,
    belief_after_station_id: int | None = None,
    posterior_health: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one schema-v3 compact audit of a PF planner decision."""
    if isinstance(station_id, bool) or not isinstance(station_id, int):
        raise TypeError("station_id must be an integer.")
    if station_id < 0:
        raise ValueError("station_id must be nonnegative.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer.")
    if belief_after_station_id is not None and (
        isinstance(belief_after_station_id, bool)
        or not isinstance(belief_after_station_id, int)
        or belief_after_station_id < 0
    ):
        raise ValueError("belief_after_station_id must be nonnegative or None.")
    expected_belief_station = None if station_id == 0 else int(station_id - 1)
    if belief_after_station_id != expected_belief_station:
        raise ValueError(
            "A planned station must be audited against the immediately "
            "preceding PF belief station."
        )
    diagnostics = _mapping(result.diagnostics, name="planner diagnostics")
    selection_mode = diagnostics.get("selection_mode")
    if selection_mode is not None:
        raise ValueError(f"Unsupported planner selection_mode {selection_mode!r}.")
    validated_health = (
        None
        if posterior_health is None
        else _validated_shadow_health(
            posterior_health,
            belief_after_station_id=belief_after_station_id,
        )
    )
    shortlist = _mapping(
        diagnostics.get("planning_eig_shortlist", {}),
        name="planning_eig_shortlist",
    )
    leaders = _mapping(
        diagnostics.get("component_leaders", {}),
        name="component_leaders",
    )
    ranked = diagnostics.get("ranked_nodes", [])
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
        raise TypeError("ranked_nodes must be a sequence.")
    selected_eig = (
        None if not result.sequence else float(result.sequence[0].information_gain)
    )
    information_leader = _leader(leaders, "information_gain")
    shadow = _resolved_shield_view_count_shadow(
        raw=shortlist.get("shield_view_count_shadow"),
        result=result,
        selected_information_gain=selected_eig,
        belief_after_station_id=belief_after_station_id,
        posterior_health=validated_health,
    )
    compact_ranked = [
        _compact_ranked_action(
            _mapping(value, name="ranked node"),
            name="ranked node",
        )
        for value in ranked[:top_k]
    ]
    audit = {
        "schema_version": 3,
        "station_id": int(station_id),
        "belief_after_station_id": belief_after_station_id,
        "selected_pose_index": int(result.next_pose_index),
        "selection_mode": "pf_dss_pp",
        "selected_pose_xyz": [float(value) for value in result.next_pose],
        "selected_program": {
            "name": str(result.shield_program.name),
            "kind": str(result.shield_program.kind),
            "pair_ids": [int(value) for value in result.shield_program.pair_ids],
        },
        "selected_score": float(result.score),
        "candidate_pose_count": _required_nonnegative_integer(
            shortlist,
            "candidate_pose_count",
            name="planning_eig_shortlist",
        ),
        "exact_pose_count": _required_nonnegative_integer(
            shortlist,
            "shortlisted_pose_count",
            name="planning_eig_shortlist",
        ),
        "proxy_subset_evaluation_count": _required_nonnegative_integer(
            shortlist,
            "proxy_subset_evaluation_count",
            name="planning_eig_shortlist",
        ),
        "exact_subset_evaluation_count": _required_nonnegative_integer(
            shortlist,
            "exact_subset_evaluation_count",
            name="planning_eig_shortlist",
        ),
        "planning_particle_count": _required_nonnegative_integer(
            diagnostics,
            "planning_particle_count",
            name="planner diagnostics",
        ),
    }
    if selected_eig is not None:
        audit["selected_information_gain"] = selected_eig
    if information_leader is not None:
        audit["information_gain_leader"] = information_leader
    if compact_ranked:
        audit["top_ranked_actions"] = compact_ranked
    audit["exact_eig_seed"] = _required_nonnegative_integer(
        shortlist,
        "exact_eig_seed",
        name="planning_eig_shortlist",
    )
    if shadow is not None:
        audit["shield_view_count_shadow"] = shadow
    return audit


def build_bootstrap_planner_audit(
    *,
    station_id: int,
    pose_index: int,
    pose_xyz: Sequence[float],
    program: ShieldProgram,
    shadow_enabled: bool,
    candidate_view_counts: tuple[int, ...] = (2, 4, 8),
    retention_fraction: float = 0.95,
    per_comparison_confidence: float = 0.95,
) -> dict[str, object]:
    """Build a schema-aligned prior-only bootstrap planner audit."""
    if (
        isinstance(station_id, bool)
        or not isinstance(station_id, int)
        or station_id != 0
    ):
        raise ValueError("Bootstrap planner audit requires station_id=0.")
    if isinstance(pose_index, bool) or not isinstance(pose_index, int):
        raise ValueError("Bootstrap pose_index must be an integer.")
    pose = [float(value) for value in pose_xyz]
    if len(pose) != 3:
        raise ValueError("Bootstrap pose_xyz must contain three coordinates.")
    counts = tuple(int(value) for value in candidate_view_counts)
    if not counts or tuple(sorted(set(counts))) != counts:
        raise ValueError("Bootstrap shadow view counts must strictly increase.")
    if bool(shadow_enabled) and counts != (2, 4, 8):
        raise ValueError("Enabled bootstrap shadow counts must be (2, 4, 8).")
    pair_ids = tuple(int(value) for value in program.pair_ids)
    if not pair_ids:
        raise ValueError("Bootstrap shield pair IDs must be nonempty.")
    if bool(shadow_enabled) and (
        len(pair_ids) != counts[-1] or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError(
            "Enabled bootstrap shadow execution must use eight unique views."
        )
    shadow = None
    if shadow_enabled:
        shadow = {
            "schema_version": 2,
            "status": "bootstrap_forced",
            "truth_used": False,
            "candidate_view_counts": [int(value) for value in counts],
            "reference_view_count": int(counts[-1]),
            "retention_fraction": float(retention_fraction),
            "per_comparison_one_sided_confidence": float(
                per_comparison_confidence
            ),
            "actual_execution": {
                "view_count": int(len(program.pair_ids)),
                "fixed_to_reference_view_count": True,
            },
            "health": _unavailable_shadow_health(
                None,
                reason="bootstrap_prior_only",
            ),
        }
    audit = {
        "schema_version": 3,
        "station_id": 0,
        "belief_after_station_id": None,
        "selected_pose_index": int(pose_index),
        "selection_mode": (
            "external_control_bootstrap"
            if program.kind == "external_control"
            else "pf_prior_balanced_bootstrap"
        ),
        "selected_pose_xyz": pose,
        "selected_program": {
            "name": str(program.name),
            "kind": str(program.kind),
            "pair_ids": [int(value) for value in program.pair_ids],
        },
        "candidate_pose_count": 0,
        "exact_pose_count": 0,
        "proxy_subset_evaluation_count": 0,
        "exact_subset_evaluation_count": 0,
        "planning_particle_count": 0,
    }
    if shadow is not None:
        audit["shield_view_count_shadow"] = shadow
    return audit


def _resolved_shield_view_count_shadow(
    *,
    raw: object,
    result: DSSPPResult,
    selected_information_gain: float | None,
    belief_after_station_id: int | None,
    posterior_health: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Join truth-free PF health to one planner-owned shadow calculation."""
    executed = _executed_shadow_action(
        pose_index=int(result.next_pose_index),
        pose_xyz=[float(value) for value in result.next_pose],
        program=result.shield_program,
        information_gain=selected_information_gain,
        score=float(result.score),
    )
    health = (
        _unavailable_shadow_health(
            belief_after_station_id,
            reason="posterior_health_unavailable",
        )
        if posterior_health is None
        else dict(posterior_health)
    )
    if raw is None:
        if posterior_health is None:
            return None
        raise ValueError(
            "Enabled shield view-count shadow omitted its planner diagnostics."
        )
    payload = dict(_mapping(raw, name="shield_view_count_shadow"))
    if payload.get("status") != "evaluated":
        raise ValueError("A present shield view-count shadow must be evaluated.")
    if payload.get("truth_used") is not False:
        raise ValueError("Shield view-count shadow diagnostics must be truth-free.")
    exact = _mapping(payload.get("exact"), name="shield_view_count_shadow.exact")
    policy = _mapping(payload.get("policy"), name="shadow policy")
    candidate_counts_raw = policy.get("candidate_view_counts")
    if not isinstance(candidate_counts_raw, Sequence) or isinstance(
        candidate_counts_raw,
        (str, bytes),
    ):
        raise TypeError("Shadow candidate view counts must be a sequence.")
    candidate_counts = tuple(int(value) for value in candidate_counts_raw)
    reference_count = int(policy.get("reference_view_count", 8))
    if candidate_counts != (2, 4, 8) or reference_count != 8:
        raise ValueError("Evaluated shadow audit requires the (2, 4, 8) policy.")
    if len(result.shield_program.pair_ids) != reference_count:
        raise ValueError("Evaluated shadow audit requires fixed-eight execution.")
    if len(set(result.shield_program.pair_ids)) != reference_count:
        raise ValueError("Executed fixed-eight shield pairs must be unique.")
    exact_pose_indices = _validated_pose_rows(
        exact.get("pose_indices"),
        exact.get("pose_xyz"),
        name="exact shadow",
    )
    exact_by_count = _mapping(exact.get("by_view_count"), name="exact.by_view_count")
    _validate_view_count_pair_rows(
        exact_by_count,
        candidate_view_counts=candidate_counts,
        pose_count=len(exact_pose_indices),
        name="exact shadow",
    )
    proxy = _mapping(payload.get("proxy"), name="shield_view_count_shadow.proxy")
    if proxy.get("status") == "evaluated":
        proxy_pose_indices = _validated_pose_rows(
            proxy.get("pose_indices"),
            proxy.get("pose_xyz"),
            name="proxy shadow",
        )
        proxy_by_count = _mapping(
            proxy.get("by_view_count"),
            name="proxy.by_view_count",
        )
        _validate_view_count_pair_rows(
            proxy_by_count,
            candidate_view_counts=candidate_counts,
            pose_count=len(proxy_pose_indices),
            name="proxy shadow",
        )

    raw_point_action = _mapping(
        exact.get("point_rule_action"),
        name="point_rule_action",
    )
    raw_lcb_action = _mapping(
        exact.get("paired_lcb_rule_action"),
        name="paired_lcb_rule_action",
    )
    point_action = _shadow_action(
        raw_point_action,
        score_field="pose_score",
    )
    lcb_action = _shadow_action(
        raw_lcb_action,
        score_field="pose_score",
    )
    health_passed = bool(health.get("available")) and bool(health.get("passed"))
    health_action = lcb_action if health_passed else executed
    try:
        exact_pose_indices.index(int(result.next_pose_index))
    except ValueError as error:
        raise ValueError(
            "Executed pose is absent from the exact shadow union."
        ) from error
    retention_fraction = float(policy["retention_fraction"])
    confidence = float(policy["per_comparison_one_sided_confidence"])
    if not 0.0 < retention_fraction <= 1.0:
        raise ValueError("Shadow retention fraction must lie in (0, 1].")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Shadow confidence must lie in (0, 1).")
    return {
        "schema_version": 2,
        "status": "evaluated",
        "truth_used": False,
        "candidate_view_counts": [int(value) for value in candidate_counts],
        "reference_view_count": reference_count,
        "retention_fraction": retention_fraction,
        "per_comparison_one_sided_confidence": confidence,
        "actual_execution": {
            "view_count": int(len(result.shield_program.pair_ids)),
            "fixed_to_reference_view_count": True,
        },
        "health": health,
        "proxy": _compact_proxy_shadow(
            proxy,
            candidate_view_counts=candidate_counts,
        ),
        "exact": _compact_exact_shadow(
            exact,
            candidate_view_counts=candidate_counts,
            include_pose_xyz=proxy.get("status") != "evaluated",
        ),
        "hypothetical_actions": {
            "point_rule": point_action,
            "paired_lcb_rule": lcb_action,
            "health_gated_rule": health_action,
        },
    }


class PlannerAuditWriter:
    """Append one fsync-backed planner decision record per station."""

    def __init__(self, path: str | Path) -> None:
        """Initialize a new append-only audit file."""
        self.path = Path(path).expanduser().resolve()
        if self.path.exists():
            raise FileExistsError(f"Refusing to replace planner audit {self.path}.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = DurableJSONLWriter(self.path, mode=0o644)

    def append(self, payload: Mapping[str, object]) -> None:
        """Durably append one finite JSON object."""
        self._writer.append(dict(payload))

    def close(self) -> None:
        """Close the shared durable writer exactly once."""
        self._writer.close()

    def __enter__(self) -> "PlannerAuditWriter":
        """Return this writer for one deterministic audit lifetime."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the audit writer when leaving a managed lifetime."""
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "PlannerAuditWriter",
    "SHIELD_VIEW_COUNT_SHADOW_HEALTH_GATES",
    "build_bootstrap_planner_audit",
    "build_planner_audit",
]
