"""Durable, truth-free audit records for PF action selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from runtime.artifacts import DurableJSONLWriter

from planning.dss_pp import DSSPPResult


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Return one mapping or fail on a malformed planner diagnostic."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _leader(
    leaders: Mapping[str, object],
    name: str,
) -> dict[str, object] | None:
    """Return one JSON-safe planner component leader when available."""
    value = leaders.get(name)
    if value is None:
        return None
    return dict(_mapping(value, name=f"component_leaders.{name}"))


def build_planner_audit(
    *,
    station_id: int,
    result: DSSPPResult,
    top_k: int = 10,
    mc_rank_stability: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one compact audit of the action domain and selected EIG."""
    if isinstance(station_id, bool) or not isinstance(station_id, int):
        raise TypeError("station_id must be an integer.")
    if station_id < 0:
        raise ValueError("station_id must be nonnegative.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer.")
    diagnostics = _mapping(result.diagnostics, name="planner diagnostics")
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
    best_exact_eig = (
        selected_eig
        if information_leader is None
        else float(information_leader["information_gain"])
    )
    stability = (
        {
            "status": "not_evaluated_in_control_loop",
            "reason": (
                "Independent-seed EIG repetition is an offline diagnostic "
                "because it doubles expensive planning work."
            ),
        }
        if mc_rank_stability is None
        else dict(mc_rank_stability)
    )
    return {
        "schema_version": 1,
        "station_id": int(station_id),
        "selected_pose_xyz": [float(value) for value in result.next_pose],
        "selected_program": {
            "name": str(result.shield_program.name),
            "kind": str(result.shield_program.kind),
            "pair_ids": [int(value) for value in result.shield_program.pair_ids],
        },
        "selected_score": float(result.score),
        "selected_information_gain": selected_eig,
        "best_exact_information_gain": best_exact_eig,
        "selected_pose_best_exact_information_gain": diagnostics.get(
            "selected_pose_exact_information_gain_leader"
        ),
        "selected_program_is_exact_eig_leader_at_selected_pose": bool(
            diagnostics.get(
                "selected_program_is_exact_eig_leader_at_selected_pose",
                False,
            )
        ),
        "selected_pose_exact_program_count": int(
            diagnostics.get("selected_pose_exact_program_count", 0)
        ),
        "total_action_count": int(shortlist.get("total_action_count", 0)),
        "shortlisted_pose_count": int(shortlist.get("shortlisted_pose_count", 0)),
        "programs_per_shortlisted_pose": int(
            shortlist.get("programs_per_shortlisted_pose", 0)
        ),
        "full_program_sweep_per_shortlisted_pose": bool(
            shortlist.get("full_program_sweep_per_shortlisted_pose", False)
        ),
        "selected_proxy_rank": int(shortlist.get("shortlist_selected_proxy_rank", 0)),
        "exact_action_count": int(shortlist.get("exact_action_count", 0)),
        "proxy_action_count": int(shortlist.get("proxy_action_count", 0)),
        "planning_particle_count": int(diagnostics.get("planning_particle_count", 0)),
        "score_leader": _leader(leaders, "score"),
        "information_gain_leader": information_leader,
        "top_ranked_actions": [
            dict(_mapping(value, name="ranked node")) for value in ranked[:top_k]
        ],
        "shortlist_certificate": {
            "available": bool(
                shortlist.get(
                    "shortlist_formal_recall_certificate_available",
                    False,
                )
            ),
            "winner_exceeds_excluded_bound": bool(
                shortlist.get(
                    "shortlist_mc_winner_exceeds_universal_excluded_bound",
                    False,
                )
            ),
            "evaluated_objective_lower_bound": shortlist.get(
                "shortlist_evaluated_objective_lower_bound"
            ),
            "excluded_objective_upper_bound": shortlist.get(
                "shortlist_max_excluded_universal_objective_upper_bound"
            ),
        },
        "exact_eig_seed": shortlist.get("exact_eig_seed"),
        "mc_seed_rank_stability": stability,
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


__all__ = ["PlannerAuditWriter", "build_planner_audit"]
