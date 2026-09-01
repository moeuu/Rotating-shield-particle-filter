# Fixed-Horizon Exact Transport Cache

## Scope

The production PF remains a causal 16-station particle filter. It does not
rerun inference over a finalized log and it does not replace the hierarchical
full-spectrum likelihood with an MLE or an approximate sufficient statistic.
The optimization changes only how source-resolved transport is stored and how
an exact-RJ proposal is materialized.

The live contract has fixed capacity for 16 stations and 128 views. Before the
first acquisition, the PF reserves two CUDA float64 buffers for that complete
capacity: one accepted buffer and one resampling destination. The allocation
contains only isotopes declared by the runtime handshake. An acquisition
contract above either fixed bound is rejected, and insufficient CUDA memory is
an error. Preflight includes the two persistent buffers and conservative
headroom for the minimum bounded exact-likelihood slab; there is no CPU or
full-clone fallback.

## Accepted state

`JointTransportCache` owns these aligned arrays:

```text
total[N, 128, source_slot, line]
uncollided[N, 128, source_slot, line]
features[N, 128, source_slot, line, feature]
station_log_likelihood[N, 16]
```

It also owns station offsets and signatures, the valid view count, the accepted
state digest, and the CUDA row-generation identifier. Only the acquired prefix
is exposed to likelihood code. A new station is copied into its unused slab;
prior slabs are never concatenated or recopied.

Accepted untempered likelihoods are retained per station. On a normal station
append, only the new station column is evaluated; earlier station likelihoods
are neither rescanned nor reconstructed. A changed proposal still evaluates
its exact response over every acquired view before it can be accepted. The
complete replacement block is overlaid once, and the one-stage MH/RJ decision
uses the resulting exact full-history likelihood difference and one uniform
draw. No representative station, history screening, or block-size weight
exists. Accepted changed rows are rebased per station before the sweep
completes.

Every station signature covers its detector position, integer spectrum, energy
axis, Fe/Pb program, live times, station identifiers, and model-contract hash.
The cache is rejected if this history identity or the accepted PF generation
does not match.

The second transport buffer is permanent. Resampling gathers every active
transport component and every station likelihood with the same ancestor vector
into that buffer, synchronizes CUDA, and swaps buffers only after all gathers
succeed. Thus resampling does not allocate a history-sized temporary and cannot
publish a partially reindexed cache.

Required bytes are computed from the authenticated particle count, source-slot
capacity, active line basis, feature count, and acquisition limits at handshake
time. Profile-specific memory totals are not stored as architecture constants.

## Exact slot overlay

For a proposal that changes one isotope, PF recomputes the complete fixed slot
block for that isotope over the acquired views. Unchanged isotope blocks are
read from accepted storage. The runtime-owned observation model gathers only a
bounded proposal-state slab, overlays the replacement slots, and invokes the
same hierarchical likelihood used by a fully materialized state.

The source/line tensors are not collapsed. Finite-Monte-Carlo and component
uncertainty therefore retain exactly the same source-resolved semantics.
Birth, death, canonical source reordering, position, strength, split/merge, and
cross-isotope block moves all use fixed slot blocks rather than source identity
heuristics.

Before evaluation or commit, PF verifies row, slot, shape, dtype, device, and
active-slot alignment. Inactive slots must be exactly zero. Accepted proposal
rows commit their CUDA state, transport block, station-likelihood invalidation,
and target value under the same mask. Rejected rows remain untouched. Invalid
station likelihood rows are then recomputed from committed transport and must
match the proposal target within the declared float64 tolerance.

The production CUDA lifecycle does not clone or rebuild the complete accepted
history. Small NumPy/materialized forms remain only as deterministic test
oracles.

## Verification

Tests compare slot overlay with a materialized full replacement for both the
plain renewal likelihood and hierarchical finite-MC/component uncertainty.
They also cover proposal families, accepted/rejected masks, cache commit,
ancestor alignment, fixed capacity, inactive slots, CPU oracle/CUDA numerical
equivalence, and production-path selection. Performance claims belong in a
versioned benchmark artifact that records its command, configuration, hardware,
raw measurements, and repository revision.

This is a storage and materialization change. It does not change Geant4
physics, transport values, likelihood equations, priors, the one-stage RJ
transition rule, or the posterior target. Numerical equivalence and transition
tests are therefore required, but a new Geant4 acceptance campaign is not.
