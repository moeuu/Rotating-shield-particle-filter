# Conditional-Greedy DSS Planning

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
9. Every exact pose compares conditional greedy with all 448 one-pass 1-swap
   neighbors on the same Monte Carlo observations. The largest EIG is retained.
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

Production exposes only the conditional all-pairs search. Explicit RA-L
comparison variants may supply one fixed `ShieldProgram`, but they cannot
select another program library. Live acquisition's prior-only first station
uses an independent balanced bootstrap traversal.

Schema-v2 represents disabled compound spatial terms explicitly. A disabled
local-orbit term requires an empty radius list and a null sigma; a disabled
elevation term requires null scale and angle fields. Coverage-floor weight and
quantile are either both positive or both zero, and disabling coverage also
requires a zero exact-shortlist coverage reserve. The production loader rejects
mixed active/inactive representations before connecting to the runtime. The
RA-L `eig_only_path` uses these disabled states and retains only full-spectrum
EIG and runtime-authored motion-time costs in its controlling pose score.

## Compute contract

The response is kept as 64 unique views rather than expanded to 48 programs by
eight views. Full-spectrum view terms are prepared once in bounded view slabs;
greedy and one-swap evaluation then use selection-matrix reductions
without a scalar Python loop over pair candidates.

The default scheduling is memory-aware:

- proxy: up to 32 poses per response/cache batch, 16 particles, 2 samples;
- exact: nominally 2--4 poses per response batch, cache action slab 1,
  512 particles, 50 samples; a clearly diagnosed one-pose fallback is used
  only when the configured budget or current free VRAM cannot hold two;
- likelihood preparation: sample slab at most 10, state slab at most 128, and
  view slab at most 8.

Exact/proxy memory settings are total phase budgets, not per-pose allowances.
Response construction streams source chunks into preallocated fields; subset
search uses a separately bounded resident cache and candidate workspace. The
pose chunk is the minimum allowed by both budgets, allocator-aware free VRAM,
and the configured cap. An allocation retry may reduce only that chunk while
preserving canonical pose order and random streams. A single source row that
cannot fit fails closed.

Independent confirmation releases the initial opaque cache before constructing
another cache and reuses the already-computed physical response. Diagnostics
record the selected chunks, budget inputs, and any scheduling-only retry.

These slabs alter only scheduling. They do not change physics, likelihood,
particle support, predictive samples, or floating-point dtype.

## Limits and claims

Conditional greedy followed by one 1-swap pass is not exhaustive search over
all `C(64, 8)` subsets and has no global-optimality guarantee. There is no
compatibility floor that may override this decision. The 8--16 pose shortlist
is an uncertainty-controlled compute budget, not proof that the selected pose
is globally optimal among all reachable poses.

The planner may use detailed runtime diagnostics internally, but the durable audit
keeps only physical pose/subset counts, selected action/EIG, the EIG leader, compact
top-k actions, relevant seeds, and the statistical evidence needed by the
fixed-eight shadow audit. Runtime chunk/memory telemetry and derivable counters are
not persisted.

## Fixed-eight shield-view shadow audit

The production acquisition contract still executes exactly eight shield-pair
views at every station. The strict PF profile additionally evaluates a
non-controlling shadow policy over view counts `{2, 4, 8}`:

- The greedy order supplies nested prefixes `G2 subset G4 subset G8`.
- After selection, one independent all-64-view holdout cache evaluates K=2,
  K=4, and greedy-prefix K=8 with shared PF particles, predictive samples,
  latent states, and nuisance draws. The executed K=8 one-swap result remains
  separate.
- For short count `k`, paired samples are
  `D_q(k) = KL_q(k) - 0.95 KL_q(8)`. A positive one-sided 95% Student-t lower
  bound recommends two views, then four, otherwise eight. This is per-comparison
  confidence, not a simultaneous 95% claim.
- Truth-free health warnings force the hypothetical health-gated action to
  eight. No uncalibrated measurement-time utility is applied.

The executed K=8 shortlist and action always remain controlling. Extra audit
poses fill only unused shortlist capacity, and enabling the audit must not
change pose/program selection or the planner RNG stream. The durable audit
stores point-rule, paired-LCB, health-gated actions, marginal EIG per live
second, and explicit fixed-eight execution. It reuses physical response and
adds only batched virtual-observation likelihood reductions.
