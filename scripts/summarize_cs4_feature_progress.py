"""Summarize persistent Cs4 feature-validation progress from logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+pose=")
URL_RE = re.compile(r"CUI split visualization URL:\s+(\S+)")
PF_ISOTOPES_RE = re.compile(r"PF candidate isotopes:\s+(\[[^\]]*\])")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize Cs4 feature-validation progress.",
    )
    parser.add_argument("--log", type=Path, required=True, help="run_all log path.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Feature-validation manifest CSV path.",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=None,
        help="Optional JSON status output path.",
    )
    return parser.parse_args()


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    """Read manifest rows as dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _output_tag(row: dict[str, str]) -> str:
    """Return the output tag for one manifest row."""
    return f"{row['case']}_{row['variant']}_seed_{row['seed']}"


def _summary_path(row: dict[str, str]) -> Path:
    """Return the expected result summary path for one manifest row."""
    return Path("results") / f"result_summary_{_output_tag(row)}.json"


def _completed_payload(row: dict[str, str]) -> dict[str, Any] | None:
    """Return a compact completion payload when the summary exists."""
    path = _summary_path(row)
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"summary_path": path.as_posix(), "readable": False}
    metrics = summary.get("metrics", {})
    mission = summary.get("mission_metrics", {})
    return {
        "summary_path": path.as_posix(),
        "readable": True,
        "measurements_completed": summary.get("measurements_completed"),
        "runtime_wall_s": summary.get("runtime_wall_s"),
        "max_pose_stop_unresolved": mission.get("max_pose_stop_unresolved"),
        "metrics": metrics,
    }


def _read_log(path: Path) -> str:
    """Read the run log when available."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _last_segment(log_text: str) -> str:
    """Return the current run segment from a sequential run_all log."""
    marker = "Loaded "
    index = log_text.rfind(marker)
    if index < 0:
        return log_text
    return log_text[index:]


def _last_step(segment: str) -> int | None:
    """Return the largest step index observed in the current segment."""
    steps = [int(match.group(1)) for match in STEP_RE.finditer(segment)]
    return max(steps) if steps else None


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the last regex group match from text."""
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _build_status(
    *,
    manifest_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Build a compact status summary for the feature-validation run."""
    rows = _manifest_rows(manifest_path)
    completed: dict[str, Any] = {}
    for row in rows:
        payload = _completed_payload(row)
        if payload is not None:
            completed[_output_tag(row)] = payload
    pending = [_output_tag(row) for row in rows if _output_tag(row) not in completed]
    log_text = _read_log(log_path)
    segment = _last_segment(log_text)
    current_variant = pending[0] if pending else None
    return {
        "manifest": manifest_path.as_posix(),
        "log": log_path.as_posix(),
        "total_variants": len(rows),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "completed_variants": completed,
        "pending_variants": pending,
        "current_variant": current_variant,
        "current_segment_last_step": _last_step(segment),
        "cui_url": _last_match(URL_RE, log_text),
        "pf_candidate_isotopes": _last_match(PF_ISOTOPES_RE, log_text),
    }


def _write_status(path: Path | None, payload: dict[str, Any]) -> None:
    """Write status JSON when requested."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    """Print and optionally persist the current progress summary."""
    args = _parse_args()
    status = _build_status(manifest_path=args.manifest, log_path=args.log)
    _write_status(args.status_json, status)
    current = status.get("current_variant")
    print(
        "cs4_feature_progress "
        f"completed={status['completed_count']}/{status['total_variants']} "
        f"current={current} "
        f"step={status.get('current_segment_last_step')} "
        f"pending={status['pending_variants']} "
        f"url={status.get('cui_url')}"
    )


if __name__ == "__main__":
    main()
