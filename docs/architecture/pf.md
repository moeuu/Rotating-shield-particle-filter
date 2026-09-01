# Pure PF Architecture

## Estimator boundary

The scientific runtime accepts one estimator profile, `pf_strict`, under
`pure_pf_schema_version: 2`. Its sequential data flow is:

```text
unit-weight Geant4 detector events
        -> sampled detector-response marks
        -> event-time non-paralyzable dead time
        -> one integer full-spectrum station record
        -> joint full-spectrum SMC/RJ update
        -> one aligned multi-isotope PF posterior
             |-> PF-only DSS-PP belief
             `-> coherent posterior-particle report
```

The MeasurementLog append precedes the update. Ground truth is stored outside
the inference log and is used only for evaluation. Source cardinality,
continuous surface position, and strength remain particle state throughout
live ingestion, planning, and reporting.

There are no position MLE, batch fit, strength refit, surface-map rescue,
count-extraction rescue, or report-time optimization paths. Unknown or
incomplete runtime settings fail closed.

## Joint full-spectrum observation model

The PF does not estimate isotope counts before inference. One immutable
source-resolved generative model owns:

- the exact analysis energy axis and positive gamma-line order;
- detector-response marking and physical background;
- non-paralyzable renewal total-count statistics;
- conditional full-spectrum mark statistics;
- declared, training-only transport-model uncertainty;
- NumPy and Torch likelihoods;
- predictive means and samples used by DSS; and
- posterior-predictive innovation diagnostics.

The model contract is identified by a SHA-256 digest covering its physics,
parameters, energy axis, line order, and validation provenance. Runtime use is
allowed only after predeclared independent Geant4 holdouts pass production
gates. PF, exact RJ moves, live ingestion, and DSS call this same model. No
additional isotope-count, contrast, view-ratio, Poisson, or covariance
likelihood is added.

Production observations are nonnegative integer histograms produced from
unit-weight detector events. Fractional, weighted, truncated, energy-axis
mismatched, or contract-mismatched spectra are rejected.

## Continuous physical source support

PF reconstructs the authenticated exposed physical surface union supplied by
the runtime environment contract:

- floor and ceiling portions not hidden by touching transport solids;
- all exposed room-wall portions;
- all exposed faces of transport components, including the underside of a
  raised component.

Blocked navigation cells do not invent source surfaces. The sibling runtime
owns private truth sampling and any predeclared experiment-level geometric
conditioning. PF receives neither those truth locations nor their RNG
provenance; its configured surface-area prior and proposal support remain
truth-free.

Rectangular surface charts define atlas topology only. A PF source stores
`(chart_id, u, v, strength)` with continuous `u,v` in the chart unit square;
the chart edge limit controls atlas topology and batching, not localization
resolution. XYZ is derived directly from chart coordinates and passed to the
continuous physical response kernel. Missing or inconsistent chart
coordinates fail instead of being reconstructed or projected.

Birth and global-position proposals include chart mass and continuous UV
density. Local moves uniformly select one predeclared 3 cm, 15 cm, or 50 cm
physical tangent scale and unfold the displacement across valid shared-edge
portals. They jointly map source strength to preserve the source's integrated
all-history response, including the coordinate-area ratio and strength-map
Jacobian in the exact MH ratio. All target responses are evaluated at
continuous XYZ, without patch-center interpolation.

## Joint SMC and exact structural moves

One aligned particle row represents the simultaneous state of every isotope.
The initial distribution is the independent product of each isotope's
predeclared cardinality, surface-area, and strength priors. All isotope
containers share one outer weight and one resampling ancestry.

The strict profile uses a shifted-Gamma source-strength prior with support
`[300 kcps, infinity)`. A physically supported multi-merge is therefore not
rejected by an artificial upper strength ceiling. The Gamma shape and scale are
predeclared physical-prior parameters, not fitted to a diagnostic run. Finite
strength grids are proposal design points only and never truncate PF state.

Same-isotope pair and multi-component split/merge kernels remain reversible.
Pair merge selection favors a near-floor donor and a receiver with similar
all-history line response as well as short intrinsic surface distance.
Multi-merge anchor selection mixes frozen data-informed evidence with a
positive uniform component. Every complete normalized selector is included in
the RJ ratio, so distant or strong donors retain positive support. A separate
all-isotope strength block changes every active strength in one batched
likelihood evaluation. It imposes no conservation law between isotopes:
simultaneous increases and decreases are merely proposals judged by the shared
mixed-spectrum posterior.

The standard cardinality policy is
`independent_poisson_with_thin_geometric_capacity_tail_v1`: independently for
each isotope, a Poisson source-count prior with predeclared mean 2.0 defines
`K = 0, ..., 5`, followed by the fixed geometric capacity tail through `K = 8`.
It is fixed before observation, retains positive mass above the ordinary range,
and is fully recorded in result provenance. A failed holdout does not authorize
retuning it on that holdout.

Each complete station joint likelihood is tempered from `beta = 0` to
`beta = 1` through one bridge. Views are never assimilated as ordered prefixes
and cannot trigger intermediate view-order-dependent resampling. If a beta
increment would cross the target ESS, the PF applies only the admissible
increment, resamples all isotope states with one ancestor vector, and
rejuvenates at the current intermediate target before continuing. Reaching a
temper-step safety bound before `beta = 1`, or finishing below the ESS contract,
is an error rather than a forced likelihood application.

Exact-RJ history evaluation uses the fixed-capacity, source-resolved CUDA cache
described in [Fixed-horizon exact transport cache](transport_cache.md).
It replaces changed isotope slot blocks without copying unchanged history and
evaluates the complete acquired history once for every supported proposal. The
one-stage MH/RJ decision uses that exact target difference and one uniform
draw. No station represents another station, and no spectrum is weighted,
averaged, screened, or discarded.

Within each intermediate target, conditional isotope moves evaluate the full
joint spectrum with all other isotope states held fixed. The reversible-jump
kernel includes:

- full-support data-informed birth and global proposals with explicit
  forward/reverse densities;
- exact birth/death cardinality, position, strength, move-direction, and
  Jacobian terms;
- continuous local surface moves;
- proper shifted-Gamma strength moves without an artificial upper bound;
- exact pair and multi-component split/merge moves that jointly refresh
  cardinality, continuous positions, and all surviving strengths without
  imposing a false strength-conservation law; and
- an exact joint isotope-state block whose isotope priors remain independent
  while the shared full-spectrum likelihood decides one simultaneous move.

Proposal scoring may use observations to improve mixing, but it does not alter
the target likelihood and every non-prior proposal density appears in the MH/RJ
ratio. Accepted rejuvenation moves leave outer particle weights unchanged.

An in-process caller may stage an immutable surface-density grid bound to one
exact incoming record prefix. PF maps it to its chart atlas and mixes it only
into the structural proposal. A positive area-prior component retains full
support, and the same frozen density appears in every forward/reverse MH/RJ
term. The grid never becomes a likelihood or directly changes particle weights.

Diagnostics retain ESS, surviving station-start ancestry, direction-resolved
cardinality transitions, and attempted/accepted posterior mass for structural
moves. Structural mixing is gated independently per isotope. Hard-cap
saturation, lineage collapse, incomplete recovery, and incomplete mixing are
sampler-quality evidence rather than reasons to discard a completed
acquisition. Invalid support, non-finite MH terms, or malformed provenance are
kernel-integrity errors and fail closed.

Rejuvenation continues until movement and lineage diagnostics pass, progress
stalls, or the explicit station wall-time guard is reached. The latter two
outcomes produce a sampler-quality warning and allow acquisition to continue;
they are not relabelled as execution failures.

## Posterior reporting and stopping

Cardinality probabilities are computed from the aligned joint weights. The
official point report selects one joint MAP cardinality vector and one existing
posterior particle as the configuration representative, so reported sources
coexisted and remain on their original continuous surfaces. It never averages
positions into a solid or projects a synthetic mean back to a surface.

Uncertainty and stopping diagnostics use the joint MAP cardinality-vector mass,
ordinary-capacity boundary mass, connected 95% intrinsic surface-path radius,
model-native full-spectrum posterior-predictive innovation, and the completion
state of exact-RJ rejuvenation. Current ESS remains a particle-adequacy
diagnostic, not evidence that the physical posterior has converged. A
rank-deficient three-dimensional covariance determinant is not treated as
surface convergence.

The standard controller assesses adaptive stopping from station 10 and requires
three consecutive ready generations, making station 12 the earliest adaptive
stop. The runtime acquisition contract owns the hard station limit.

## Planner

The [conditional-greedy DSS-PP planner](planner.md) receives aligned PF
particles without renormalizing away `P(K=0)`.
Modes use existing surface representatives, and the planner retains at least
the PF source-slot limit for every isotope.

Candidate spectra are sampled and scored with the same joint generative model
used by the PF. The standard planner uses `horizon = 1`; a longer horizon is
invalid until belief and coverage are conditionally rolled forward. Candidate
generation covers reachable three-dimensional free space and scores each
station's own shield program. PF-independent continuous-surface observability
coverage prevents an absent high-wall or ceiling mode from becoming an
exploration absorbing state.

## Execution model

Particle, source-slot, line, view, spectrum-bin, candidate, and shield-program
dimensions use batched NumPy/Torch kernels with explicit chunks and caches.
CUDA uses float64 for the production likelihood. CPU and GPU paths implement
the same distribution and have equivalence tests. Chunking cannot change the
random stream or scientific result. Scalar forms are limited to deterministic
test/debug oracles. New heavy paths follow the
[PF compute policy](../policies/compute.md).

## Live MeasurementLog ingestion

A published MeasurementLog bundle binds the resolved runtime configuration,
environment, forward-model manifest, repository commit, ordered station
records, and joint-spectrum contract hash. Each record stores the exact raw
integer analysis spectrum, energy-axis identity, detector pose, Fe/Pb indices,
live time, station completion marker, and native statistical provenance.
Isotope-fitted counts and truth are not part of the inference record.

During a live session, the runtime durably appends a completed station before
returning its typed event. PF validates and assimilates that station before it
proposes the next reachable pose and shield program. Production acquisition is
fresh-run only: an interruption aborts without publishing a partial log, and no
resume, prefix-replay, or finalized-log inference entrypoint exists.

`PFLiveSession` owns this in-process causal boundary and rejects record delivery
after completion. Optional proposal guidance must bind the exact station and
ordered-prefix digest. Its planning DTO copies particle arrays into read-only
storage and preserves native PF semantics; it does not invent an
estimator-neutral grid or deterministic spectrum surrogate.

At termination `complete_live_state()` seals inference before runtime
publication. PF then validates the immutable MeasurementLog's ordered-record
digest and `bind_published_log()` changes only final provenance identities.
Contract mismatches fail before publication.
