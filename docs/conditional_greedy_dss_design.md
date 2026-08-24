# Conditional-greedy DSS planning

## Production decision flow

The standard `pf_strict_3d` planner separates shield-program selection from
robot-pose selection:

1. The shared runtime supplies at most 256 reachable 3-D poses and separate
   horizontal, mast-vertical, and settling times.
2. For every pose, PF planning requests source-resolved float64 responses for
   all `num_orientations ** 2` Fe/Pb pairs. The canonical octant geometry has
   64 pairs.
3. The shared full-spectrum model generates one 64-view virtual station. The
   latent PF state and station-shared nuisance variables are shared across all
   views. These spectra exist only inside planning; acquisition still executes
   exactly the selected eight views.
4. An opaque runtime cache evaluates arbitrary pair subsets with the exact PF
   likelihood. Shared Gamma-Poisson terms are rebuilt from the selected
   subset's sufficient statistics; an unselected view cannot affect the
   likelihood.
5. Proxy conditional greedy selects eight distinct pairs at every pose. It
   evaluates `64 + 63 + ... + 57 = 484` candidate subsets in eight device
   batches and preserves greedy order as execution order.
6. Proxy pose utility is full-spectrum EIG plus spatial terms minus the same
   horizontal, mast, settling, path, and robot-turn terms used by final pose
   scoring.
7. The exact pose budget starts at eight. Independent proxy seeds re-evaluate
   the top 24 poses. A positive one-sided 95% paired gap lower bound and mean
   top-k Jaccard of at least 0.75 stop the search; otherwise the budget expands
   to 12 and then 16 poses. One coverage-leading pose is reserved.
8. Exact search retains 512 PF planning particles, 50 predictive samples,
   float64 arithmetic, the production full-spectrum law, and 121-ray detector
   aperture integration. Poses are processed in batches of at most four.
9. Every exact pose compares conditional greedy, all 448 one-pass 1-swap
   neighbors, and the EIG leader of the legacy 48-program library on the same
   Monte Carlo observations. The largest EIG is retained.
10. Only contenders or poses whose paired EIG/score gap is not separated from
    zero are checked with an independent seed. This confirmation does not
    lower particle count, sample count, or response fidelity.
11. The shield program is fixed by EIG alone. Final pose choice then maximizes
    that EIG plus spatial utility minus physical robot-motion costs.

Shield rotation angle has zero utility cost. If rotation later becomes a
measured operational constraint, it must enter as measured time rather than an
angular heuristic.

## Runtime and PF ownership

The sibling simulation-runtime repository owns transport physics, predictive
sampling, nuisance integration, and arbitrary-subset likelihood preparation.
The PF repository owns posterior sampling, conditional-greedy subset search,
adaptive pose budgeting, spatial utility, and diagnostics. PF code receives an
opaque cache and cannot extract or approximate the runtime nuisance axes.

The 48-program implementation is not imported by the conditional-greedy core.
It remains available only through the legacy policy, RA-L baseline paths, and
the narrow `legacy_program_guard` adapter. The generic `ShieldProgram` value is
defined independently, so removing the compatibility guard does not affect the
all-pairs search. Live acquisition's prior-only first station also uses a
standalone balanced traversal that reproduces the former first program without
importing the 48-program builder. The old builder is imported lazily only when
a legacy or shadow policy explicitly requests it.

## Compute contract

The response is kept as 64 unique views rather than expanded to 48 programs by
eight views. Full-spectrum view terms are prepared once in bounded view slabs;
greedy, one-swap, and incumbent evaluation then use selection-matrix reductions
without a scalar Python loop over pair candidates.

The default scheduling is memory-aware:

- proxy: up to 32 poses per response/cache batch, 16 particles, 2 samples;
- exact: nominally 2--4 poses per response batch, cache action slab 1,
  512 particles, 50 samples; a clearly diagnosed one-pose fallback is used
  only when the configured budget or current free VRAM cannot hold two;
- likelihood preparation: sample slab at most 10, state slab at most 128, and
  view slab at most 8.

The existing exact/proxy memory fields remain total phase budgets, not
per-pose allowances. The response phase and subset-search phase are
sequential. During response construction the planner first reserves its six
source-resolved destination fields and the runtime's eight retained transport
fields. The runtime streams source chunks into preallocated component buffers,
so it never retains every chunk and a second concatenated copy at once. The
residual phase budget then bounds exact physical-response scratch by source
rows, accounting for aperture rays, obstacle boxes, shield pairs, and line
count. A separate materialization bound covers the derived transport-feature
assembly; the CPU debug path conservatively includes its host concatenation.
The subset phase separately uses the runtime's per-pose resident-cache and
candidate-workspace estimate, including latent-indexed transport copies. The
pose chunk is the minimum allowed by both phase bounds, allocator-aware free
VRAM (driver free memory plus reclaimable Torch cache), and the proxy/exact cap.
If a runtime source row cannot fit the residual scratch budget, the operation
fails closed. A recognized allocation failure retries the same canonical poses
and random streams with chunks reduced from four or three to two and then one.
The strict profile assigns 4 GiB to both phases; the former 256 MiB proxy
default described only the old likelihood path and cannot efficiently hold the
new 64-pair physical response. Diagnostics record the response-field,
destination, runtime-retained, materialization, residual-scratch, subset-cache,
and device-capacity inputs and the selected chunk, including retries and
low-memory fallback use.

Independent program confirmation first copies its small KL result to the host
and releases the initial opaque cache. If every pose is ambiguous it reuses the
original response components; otherwise it creates one zero-copy pose view at
a time. It therefore never overlaps the first cache with a second cache or a
full copied ambiguous-pose component batch, and it does not recompute physical
response.

These slabs alter only scheduling. They do not change physics, likelihood,
particle support, predictive samples, or floating-point dtype.

## Measured planning cost

Historical completed-run logs contained 38 planner decisions with a mean of
133.1 s per decision: 64.4 s proxy, 23.0 s exact-48 evaluation, and 45.7 s
other geometry/orchestration. The 95th percentile was 177.7 s.

An RTX 5090 benchmark of the final memory-bounded path used 256 real reachable
runtime poses, separate motion components, 246 material obstacle boxes,
Co/Cs/Eu transport, 121 detector-aperture rays, float64, 512 exact particles,
and 50 exact samples. The exact shortlist was forced to its maximum of 16 poses
and 15 of the 16 program decisions required an independent ambiguity check. It
measured:

| Exact poses | Proxy 256 | Stability top 24 | Exact search | Core total | Peak live allocated |
|---:|---:|---:|---:|---:|---:|
| 16 | 16.67 s | 1.73 s | 55.53 s | 73.93 s | 2.80 GiB |

The exact pose chunks were `3, 3, 3, 3, 3, 1`; no OOM retry occurred. Peak
PyTorch reserved memory was 6.00 GiB, but that includes reclaimable allocator
cache rather than simultaneously live tensors. The configured 4 GiB phase
budget governs estimated and observed live allocations.

The measured planner core is below the 180 s target without reducing response
fidelity, particles, samples, or dtype. Adding the historical mean 45.7 s of
candidate geometry and orchestration gives a rough 119.6 s engineering
estimate, not an end-to-end guarantee. The benchmark combines a real runtime
candidate/obstacle snapshot with independently resampled saved PF marginals;
its spatial terms were held at zero because no single saved artifact matches
the current joint-state, surface-chart, and runtime contracts. It is therefore
a compute and decision-reproducibility benchmark, not an RA-L accuracy result
or a replacement for a fresh independent full simulation.

## Limits and claims

Conditional greedy followed by one 1-swap pass is not exhaustive search over
all `C(64, 8)` subsets and has no global-optimality guarantee. The legacy floor
prevents regression relative to the current 48 programs on the same finite MC
sample, but does not prove optimality outside those candidates. The 8--16 pose
shortlist is an uncertainty-controlled compute budget, not proof that the
selected pose is globally optimal among all reachable poses.

The planner may use detailed runtime diagnostics internally, but the durable audit
keeps only physical pose/subset counts, selected action/EIG, the EIG leader, compact
top-k actions, relevant seeds, and the statistical evidence needed by the
fixed-eight shadow audit. Runtime chunk/memory telemetry and derivable counters are
not persisted.

## Fixed-eight shield-view shadow audit

The production acquisition contract still executes exactly eight shield-pair
views at every station. The strict PF profile additionally evaluates a
non-controlling shadow policy over view counts `{2, 4, 8}`:

- The depth-eight conditional-greedy order supplies nested prefixes
  `G2 subset G4 subset G8`. Proxy EIG for every valid pose is already available
  from the greedy stages, so the proxy audit adds no response or likelihood
  approximation.
- Exact K=2, K=4, and greedy-prefix K=8 are evaluated through one independent
  holdout all-64-view cache after their ordered pairs have been fixed by the
  selection cache. The three horizons therefore share holdout PF particles,
  predictive samples, latent states, and station-shared nuisance draws without
  reusing the samples that selected the prefix. This avoids an in-sample
  winner's-curse interpretation of the paired interval. Physical response is
  reused; only virtual observations and likelihood reductions are repeated.
  The executed K=8 one-swap/legacy-guard result remains separate and unchanged.
- The executed K=8 pose shortlist remains controlling. When it contains fewer
  than the configured maximum of 16 poses, K-specific proxy leaders fill only
  the unused audit capacity. These extra poses cannot affect the executed pose
  or shield program.
- For each exact pose and short view count `k`, the audit forms paired samples
  `D_q(k) = KL_q(k) - 0.95 KL_q(8)`. It recommends two views when the one-sided
  95% Student-t lower bound for `D(2)` is strictly positive, otherwise four
  when the corresponding `D(4)` bound is positive, and otherwise eight. The
  confidence is per comparison; no simultaneous or global 95% claim is made.
- PF-owned truth-free health is joined only when writing `planner_audit.jsonl`.
  Particle-diversity warnings, sampler failures, upper-cardinality-boundary
  mass, latest full-spectrum innovation failure, newly activated isotopes, or
  unavailable health force the hypothetical health-gated action to eight.
- `measurement_time_weight` does not enter the shadow decision, pose score used by
  that decision, the returned `ShieldProgram`, runtime station completion, or
  measurement budget. The compact audit retains marginal EIG per added live second,
  but drops the uncalibrated time-weight counterfactual and derivable elapsed-time
  arrays.

The audit stores point-rule, paired-LCB, and health-gated hypothetical actions. The
top-level selected action plus `actual_execution` records that acquisition remained
fixed at eight views. Enabling or disabling the shadow path must leave the
controlling pose, ordered pair IDs, EIG, score, and planner RNG stream unchanged.

The holdout adds one virtual-observation cache construction and three batched
fixed-prefix likelihood reductions per exact pose chunk. It does not repeat
the full-spectrum physical response. This audit overhead has not yet been
measured in a completed full-simulation run and is not included in the older
benchmark table above.
