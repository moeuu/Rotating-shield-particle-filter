# Fixed-horizon exact transport cache

## Scope

The production PF remains a causal 16-station particle filter. It does not
rerun inference over a finalized log and it does not replace the hierarchical
full-spectrum likelihood with an MLE or an approximate sufficient statistic.
The optimization changes only how source-resolved transport is stored and how
an exact-RJ proposal is scheduled.

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
[Transition-Preserving History Tree](transition_preserving_history_tree.md)
first evaluates the actual latest station, then generates transport for exact
dyadic child blocks only while the proposal remains viable. No representative
station or block-size weight exists. Accepted changed rows are rebased per
station before the sweep completes.

Every station signature covers its detector position, integer spectrum, energy
axis, Fe/Pb program, live times, station identifiers, and model-contract hash.
The cache is rejected if this history identity or the accepted PF generation
does not match.

The second transport buffer is permanent. Resampling gathers every active
transport component and every station likelihood with the same ancestor vector
into that buffer, synchronizes CUDA, and swaps buffers only after all gathers
succeed. Thus resampling does not allocate a history-sized temporary and cannot
publish a partially reindexed cache.

For the current 4,096-particle Cs-137/Co-60 profile (16 slots, three lines,
13 transport features), the two persistent buffers occupy 5.626 GiB. Including
the conservative minimum exact-overlay headroom makes the live preflight
require 5.709 GiB. These byte counts are computed from the authenticated active
line basis at handshake time rather than stored as profile-specific constants.

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

The former complete-history clone and accepted-history rebuild are not part of
the production CUDA lifecycle. Small NumPy/materialized forms remain only as
deterministic test oracles.

## Verification and performance evidence

Tests compare slot overlay with a materialized full replacement for both the
plain renewal likelihood and hierarchical finite-MC/component uncertainty.
They also cover proposal families, accepted/rejected masks, cache commit,
ancestor alignment, fixed capacity, inactive slots, CPU oracle/CUDA numerical
equivalence, and production-path selection.

A local RTX 5090 benchmark used 4,096 accepted states, 16 source slots, three
gamma lines, and the hierarchical model. At 4,096 simultaneous proposal states:

| History | Materialized | Slot overlay | Overlay change | Peak scratch change |
| --- | ---: | ---: | ---: | ---: |
| 64 views | 1.0935 s | 1.0988 s | 0.995x | 11,817 to 8,578 MiB |
| 128 views | 2.6159 s | 2.1911 s | 1.194x faster | 16,139 to 9,658 MiB |

The main demonstrated benefit is bounded peak memory and removal of allocator
failures. The 128-view case also improved likelihood wall time by about 19%.
Position moves still recompute the changed source across all acquired views and
the exact likelihood still reads the bounded history, so no larger speed claim
is made without end-to-end profiling.

This is an inference scheduling change. It does not change Geant4 physics,
transport values, likelihood equations, priors, or the posterior target. TPHT
does change the RJ transition kernel and consumes a second delayed-acceptance
uniform, so finite-chain samples need not be bitwise identical. Its invariant
target remains the exact full-history posterior. It therefore requires
transition and numerical tests rather than a new Geant4 acceptance campaign.
