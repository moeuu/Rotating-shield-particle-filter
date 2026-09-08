# Proportionate Verification Policy

## Test scope

Choose checks from the changed behavior and its callers, not from the number of
edited files. During iteration, run the affected test files or individual cases.
Run the full PF suite before publishing changes to ingestion, likelihood,
SMC/exact-RJ, transport caches, planning, serialization, or package dependencies,
and before a release. Once checks pass, repeat them only after a relevant change
or a new failure. Do not rerun the runtime's native physics suite for PF-only
changes, documentation, ignore rules, or movement of already verified local data.

For documentation or artifact-layout changes, documentation and artifact-hygiene
checks are sufficient. For visualization changes, run the relevant visualization
checks and required rendered-image review. For packaging changes, test both the
wheel and source-distribution round trip. Native runtime changes require the
corresponding native integration tests in the runtime repository.

`uv run pytest` continues to mean the full suite; no tests are silently disabled.
Use `--durations=20` when auditing test cost. Share expensive immutable fixtures
such as a compiled executable within one test session, while keeping simulation
processes, requests, outputs, and mutable estimator state isolated per test.

## What tests should enforce

Prefer externally observable behavior, statistical invariants, corruption
rejection, and regression cases over private helper names, textual call order,
or exact internal dictionaries. Keep structural checks when they enforce a real
ownership or truth-isolation boundary. Replace brittle implementation checks as
the corresponding behavior is changed; do not discard meaningful coverage just
to reduce the test count. Small reversible documentation/formatting changes do
not require new tests.

Use exact comparison for identities, integer event counts, schemas, and causal
ordering. For floating-point scientific outputs, state a tolerance justified by
the numerical operation and physical quantity. Do not require bitwise equality
of general floating-point results across different execution schedules.

## Digest and validation boundaries

Keep SHA-256 for persisted artifact identities, source/configuration provenance,
observation ordering, and scientific cache consistency. Authenticate external
inputs and bind completed results to the observations they actually consumed.
Cheap identity comparisons inside the live loop should stay cheap.

Compute an unchanged value once per operation and pass its validated digest to
internal helpers. Do not repeatedly serialize the same published observation
sequence within a single bind. Retain independent validation of live and
published records. Reuse must be scoped to the operation or an explicitly
immutable object; do not add a global path cache or trust modification times in
place of content validation. Repeated public reads/binds must still detect
changed on-disk artifacts.

Measure before changing a digest format or weakening a check. JSON conversion,
model reconstruction, filesystem reads, and compilation can cost more than the
hash primitive. Optimize repeated work first, preserving existing digest bytes,
algorithm IDs, and failure behavior.
