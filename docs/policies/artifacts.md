# Artifact and Git Policy

## Version control

Track source code, reusable public configuration, tests, documentation, dependency
locks, and pinned project skill sources (including their upstream lock).
Do not track generated experiment results, logs, checkpoints, observation arrays,
rendered review images, videos, local datasets, caches, or temporary exports.
`results/`, `logs/`, and `tmp/` have no force-added Git exceptions. Historical
tracked artifacts remain available in Git history; removing them from the index
does not delete local copies. Git history is not a backup for new ignored data.

Manchester environment downloads, original assets, converted USD, and textures
belong exclusively in the sibling runtime's ignored
`data/manchester_nuclear_assets/`. Do not recreate `sim/` or an asset copy here.

Keep final paper figures and submission assets in the manuscript workspace under
its own versioning policy. Keep their raw evidence and provenance in the durable
run bundle and a separately backed-up artifact store before local cleanup.
Do not move or delete source bundles merely because their figures were exported.

## Output layout

Use these repository-relative output roots:

| Purpose | Location |
| --- | --- |
| Standalone PF acquisition | `results/runs/<opaque-run-id>/` |
| RA-L acquisition | `results/ral_ablation/runs/<opaque-run-id>/` |
| RA-L runtime-published observations | `results/ral_ablation/measurement_logs/<opaque-run-id>/` |
| Diagnostics | `results/diagnostics/<opaque-run-id>/<diagnostic-name>/` |
| Performance experiments | `results/benchmarks/<unique-benchmark-id>/` |
| Figure review, Isaac captures, supplementary video | `results/ral_figure_review/`, `results/ral_isaac_figures/`, `results/ral_supplementary_video/` |
| Console logs | `logs/<opaque-run-id>/` |
| Disposable previews and temporary exports | `tmp/` |

The existing RA-L generator owns its sibling `configs`, `run_plans`, and
`control_policies` directories; associate their entries by exact run ID. Preserve
its path contract instead of manually moving live bundles. Runtime private truth,
scene seeds, acquisition scripts, and private manifests stay under the sibling
runtime's `private_runs/`, never in PF input or public Git.

For a new run, use a fresh ID/output directory and record the command, PF and
runtime revisions, dirty-working-tree status, resolved public configuration,
start/end timestamps, and completion/failure status with the run. Do not overwrite
a completed bundle or reuse a directory for a new implementation. Resuming an
existing acquisition requires its recorded identity and the supported protocol.
Keep stdout/stderr and monitor metadata together under `logs/<opaque-run-id>/`;
use persistent sessions for long acquisitions. Tests must use pytest `tmp_path`
or a temporary directory, never persistent `results/pf-test-*` folders.

## Retention and checks

Build with `uv build`. The explicit Hatchling package list reads current source
directly and does not reuse a persistent `build/lib/` tree. Keep release wheels
and source distributions in ignored `dist/`; remove superseded distributions
instead of treating them as source backups. Packaging tests build disposable
source copies with deliberately stale build files and compare every packaged
Python file and its bytes against the current source, including a wheel rebuilt
from the source distribution. Do not restore incremental setuptools build trees.

After each completed or failed experiment, report its exact paths and status.
Retain complete evidence for active paper results, current acceptance runs, and
failures still under investigation. Once superseded and approved for cleanup,
remove the result, associated observations, config/plan/policy, and logs as one
run-ID group. Check for active processes and paper references before doing so.
Use the trash for local cleanup; never automatically purge data based on age.

Run `uv run python scripts/audit_artifacts.py` to inventory output roots and find
unexpected directories, loose outputs, or ignored files accidentally in Git.
Run its `--check` mode before committing. The audit is read-only: it does not
assert scientific validity or decide which run may be deleted. Review nested
RA-L entries by run ID; allowed directory names alone do not prove relevance.
