# Documentation Map

This directory contains only current PF-owned architecture, policy, and RA-L
experiment documentation. Runtime simulation and Geant4 implementation details
belong to the sibling `Rotating-shield-simulation-runtime` repository. Run-specific
commands, truth, results, and interpretation belong in the durable run bundle or
the manuscript workspace rather than in this directory.

## Architecture

- [Pure PF architecture](architecture/pf.md): current estimator state,
  likelihood, exact-RJ/SMC behavior, reporting, and live-session boundary.
- [Conditional-greedy planner](architecture/planner.md): current DSS-PP shield
  program and pose-selection design.
- [Exact transport cache](architecture/transport_cache.md): fixed-capacity CUDA
  storage and exact proposal-overlay contract.

Architecture documents describe implemented behavior. Historical alternatives,
retired paths, and unlinked benchmark results are intentionally omitted; Git
history retains them.

## Policies

- [PF compute policy](policies/compute.md): batching, device execution, and
  equivalence requirements for PF-owned heavy code.
- [PF inference fidelity policy](policies/inference_fidelity.md): causal
  ingestion, likelihood integrity, truth isolation, and full-run discipline.
- [Post-run evaluation policy](policies/post_run_evaluation.md): split-aware
  truth association, merged-source metrics, and independent result statuses.

Policies are normative. `AGENTS.md` points to them before work in the relevant
scope. A policy change must be made before, not in response to, the result batch
that it will govern.

## RA-L

- [Experiment protocol](ral/experiment_protocol.md): paired four-variant task,
  seeds, acquisition contract, execution, and reporting scope.
- [Manuscript policy](ral/manuscript.md): invariant claim boundaries,
  acknowledgment, evidence handling, and page allocation.
- [Figure policy](ral/figures.md): result content, source-data preservation,
  rendering, and visual QA.

RA-L documents contain reusable current rules only. Opaque run IDs, private
paths, one-off reconstruction commands, and predecessor-run commentary must be
stored with the corresponding private run or manuscript working notes.

## Maintenance Rules

- Keep one authoritative document for each rule. `AGENTS.md` may repeat concise
  high-priority guardrails but should link here for the complete policy.
- Update links and the index in the same change as any move or rename.
- Do not create `docs/archive/`. Remove superseded documents after preserving
  any still-current rule; Git history is the archive.
- Keep measured performance numbers only when the document links to the raw
  benchmark artifact, command/configuration, hardware, and code revision.
- Keep local Markdown links valid. The documentation integrity tests enforce
  both link validity and index coverage.
