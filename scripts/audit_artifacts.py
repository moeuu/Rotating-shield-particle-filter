"""Read-only audit of generated output layout and Git index hygiene."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = frozenset({
    "runs", "ral_ablation", "diagnostics", "benchmarks", "ral_figure_review",
    "ral_isaac_figures", "ral_supplementary_video",
})


def audit(root: Path) -> tuple[list[str], list[str]]:
    """Return an output inventory and violations without changing any files."""
    tracked_ignored = subprocess.check_output(
        ["git", "ls-files", "-z", "-ci", "--exclude-standard"], cwd=root,
    ).decode().split("\0")
    problems = [f"Tracked generated/local file: {p}" for p in tracked_ignored if p]
    inventory = []
    for name in ("results", "logs", "tmp"):
        base = root / name
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            problems.append(f"Expected a local directory: {name}")
            continue
        for entry in sorted(base.iterdir()):
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                problems.append(f"Output root must not be a symlink: {relative}")
                continue
            files = [entry] if entry.is_file() else entry.rglob("*")
            size = sum(p.stat().st_size for p in files
                       if not p.is_symlink() and p.is_file())
            inventory.append(f"{size / 1024**2:9.2f} MiB  {relative}")
            if name == "results" and entry.name not in RESULT_ROOTS:
                problems.append(f"Unexpected results root: {relative}")
            if name in {"results", "logs"} and not entry.is_dir():
                problems.append(f"Loose output; use a run directory: {relative}")
    return inventory, problems


def main() -> int:
    """Print the inventory and optionally fail on hygiene violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory, problems = audit(ROOT)
    for line in inventory:
        print(line)
    for problem in problems:
        print(f"ERROR: {problem}")
    print(f"{len(problems)} hygiene violation(s); no files changed.")
    return 1 if args.check and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
