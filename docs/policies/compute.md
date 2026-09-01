# PF Compute Policy

PF runtime fidelity is not negotiable, but PF-owned heavy operations must be
implemented in a parallel or batched form from their first production version.
Parallelization may change execution schedule and wall time only; it must retain
the same geometry, likelihood, proposal density, random-variable semantics, and
posterior target.

Geant4 transport, spectrum generation, and shared physical-response execution
are owned by `Rotating-shield-simulation-runtime` and follow that repository's
compute policy.

## Required Default

Use batched NumPy, batched Torch/CUDA, or process-parallel execution when an
operation spans any of these dimensions:

- PF particles or aligned particle rows;
- isotope source slots or structural proposal states;
- candidate source locations or continuous surface charts;
- detector poses, Fe/Pb pairs, or candidate shield subsets;
- stations, views, spectrum bins, transport lines, or response features;
- predictive samples, likelihood candidates, or obstacle components used by
  PF-owned planning and diagnostics.

Do not leave a scalar Python runtime loop over one of these dimensions for later
optimization. A scalar form may exist only as a small deterministic test oracle
or an explicitly selected debug fallback.

## Accepted Execution Forms

- CUDA float64 kernels for production likelihood, exact-RJ scoring, predictive
  sampling, EIG, and response-cache operations.
- Batched NumPy or multithreaded Torch-CPU execution with the same statistical
  model as the GPU path.
- Process-level parallelism for independent CPU-bound analyses when device
  batching is unavailable and process scheduling cannot alter random streams.
- Small control-plane loops that serialize bounded summaries or coordinate
  already-batched kernels.

Python threads are appropriate only for I/O-bound work or native operations
that release the GIL. They are not a substitute for batching CPU-bound Python
loops.

## Memory and Randomness

Chunking must be deterministic with respect to input order. Changing a batch or
chunk size must not change candidate support, likelihood equations, proposal
probabilities, predictive sample identity, or acceptance decisions beyond an
explicitly tested floating-point tolerance.

Device paths must bound resident state and scratch allocation before execution.
A lower-memory retry may reduce only scheduling dimensions; it must not reduce
particles, predictive samples, views, energy bins, response fidelity, or dtype.

## Required Tests

Every new batched or parallel runtime path must include at least one of:

- a serial-versus-parallel equivalence test on a small deterministic case;
- a batch-size or chunk-size invariance test; or
- a test proving that the production configuration selects the batched path.

The test must state its tolerance and verify the scientific output, not only
array shapes or kernel invocation. Run `uv run pytest` after the change.

## Narrow Exception

A scalar runtime loop is acceptable only when its full-simulation iteration
count is provably tiny, batching cannot materially reduce cost, the reason is
documented next to the code, and the path is covered by a regression test.
