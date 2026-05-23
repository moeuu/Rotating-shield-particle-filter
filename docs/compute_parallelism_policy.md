# Compute Parallelism Policy

Runtime fidelity is not negotiable, but compute-heavy code must be implemented
in a parallel or batched form from the first version. Parallelism here means
same inputs, same mathematical model, same statistical interpretation, and
numerically equivalent results, with only the execution schedule changed.

## Required Default For New Heavy Code

New code must use vectorized, batched-GPU, batched-NumPy, Geant4 multithreaded,
or process-parallel evaluation when it operates over any of these dimensions:

- PF particles.
- Source slots inside particles.
- Candidate source locations.
- Measurement stations or shield postures.
- Fe/Pb orientation programs.
- Spectrum bins, response-matrix columns, or isotope count channels.
- Obstacle ray/segment/material components.

The first implementation should not be a scalar Python loop that is later
expected to be parallelized. If a scalar reference implementation is useful, it
must be kept as a small test oracle or explicit debug fallback, not as the
runtime path.

## Acceptable Parallel Forms

- Geant4 native multithreading for transport histories.
- Batched GPU kernels for PF expected counts, EIG, obstacle attenuation, and
  candidate scoring.
- Batched NumPy linear algebra for per-particle fixed-geometry solves.
- Process-level parallelism for independent CPU-bound candidate or baseline
  evaluations when GPU batching is not available.
- Thread-level parallelism only for I/O-bound work or code that releases the
  GIL; do not assume Python threads speed up CPU-bound Python loops.

## Required Tests

Every new parallel or batched runtime path must include at least one regression
test that compares it against a serial or scalar oracle on a small deterministic
case. The test must assert numerical equivalence within an explicit tolerance.

If an existing scalar path remains as a fallback, add a test that proves the
standard runtime selects the batched/parallel path when the required backend is
available.

## Exceptions

A scalar runtime loop is allowed only when all of the following are true:

- The iteration count is provably tiny in full simulations.
- Batching would not reduce asymptotic cost or wall time.
- The reason is documented in the function docstring or adjacent comment.
- The code path is covered by a test so it does not silently become a large
  runtime bottleneck later.

## Review Checklist

Before merging a feature that touches PF, planning, spectrum processing,
obstacle attenuation, or Geant4 orchestration, verify:

- The operation dimensions are identified explicitly.
- The runtime path is batched/parallel by default.
- The scalar path, if any, is debug-only or a tested fallback.
- Parallel execution does not change physics, geometry, source-rate semantics,
  likelihood equations, or random sampling meaning.
- A serial-vs-parallel equivalence test was added or updated.
- Full `uv run pytest` passes.
