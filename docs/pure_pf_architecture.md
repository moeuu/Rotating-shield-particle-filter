# Pure PF architecture

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
count-extraction rescue, or report-time optimization paths. Unknown, retired,
or incomplete runtime settings fail closed.

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

Random ground truth and the PF use the same exposed physical surface union:

- floor and ceiling portions not hidden by touching transport solids;
- all exposed room-wall portions;
- all exposed faces of transport components, including the underside of a
  raised component.

Blocked navigation cells do not invent source surfaces. Truth positions are
independent continuous draws from normalized physical surface area, with no
height, visibility, ceiling-count, separation, or response-observability
conditioning.

Rectangular surface charts define atlas topology only. A PF source stores
`(chart_id, u, v, strength)` with continuous `u,v` in the chart unit square;
the chart edge limit controls atlas topology and batching, not localization
resolution. XYZ is derived directly from chart coordinates and passed to the
continuous physical response kernel. Missing or inconsistent chart
coordinates fail instead of being reconstructed or projected.

Birth and global-position proposals include chart mass and continuous UV
density. Local moves draw a symmetric physical tangent displacement and unfold
it across valid shared-edge portals; the coordinate-area ratio is included in
the MH proposal ratio. All target responses are evaluated at continuous XYZ,
without patch-center interpolation.

## Joint SMC and exact structural moves

One aligned particle row represents the simultaneous state of every isotope.
The initial distribution is the independent product of each isotope's
predeclared cardinality, surface-area, and strength priors. All isotope
containers share one outer weight and one resampling ancestry.

The strict profile uses a shifted-Gamma source-strength prior.  Its support is
`[300 kcps, infinity)`, so a physically supported multi-merge cannot be rejected
only because the combined source exceeds the former 2 Mcps ceiling.  The Gamma
shape and scale are predeclared physical-prior parameters, not fitted to a
diagnostic run. Finite strength grids are proposal design points only and never
truncate the PF state space.

Same-isotope pair and multi-component split/merge kernels remain reversible.
Multi-merge anchor selection mixes frozen data-informed evidence with a positive
uniform component, and the complete normalized selector is included in the RJ
ratio.  A separate all-isotope strength block changes every active strength in
one batched likelihood evaluation.  It imposes no conservation law between
isotopes: simultaneous increases and decreases are merely proposals judged by
the shared mixed-spectrum posterior. The retired scalar isotope-identity
transfer kernel is not present in the production package.

The standard cardinality policy is
`independent_poisson_with_thin_geometric_capacity_tail_v1`: independently for
each isotope, a Poisson source-count prior with predeclared mean 2.0 defines
`K = 0, ..., 5`, followed by the fixed geometric capacity tail through `K = 8`.
This encodes the design assumption of sparse surface contamination while
keeping positive mass above the ordinary range; it is fixed before any
observation and was not selected from a failed diagnostic run. Every result
manifest records the policy name, mean, support, tail ratio, and complete
normalized probability vector. Ground-truth isotope counts and strengths are
checked against PF support before any external Geant4 process starts. The
policy remains subject to the designated independent holdout acceptance gate;
a failed holdout does not authorize retuning it on that holdout.

Each station is tempered to `beta = 1`. If a partial likelihood would cross the
target ESS, the PF applies only the admissible increment, resamples all isotope
states with one ancestor vector, and rejuvenates at the current intermediate
target before continuing. Reaching a temper-step safety bound before
`beta = 1`, or finishing below the ESS contract, is an error rather than a
forced likelihood application.

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

The ordinary cardinality model is defined through `K=5`. A proper geometric
tail gives `K=6..8` nonzero support so `K=5` is not an artificial absorbing
boundary; `K=8` is the explicit memory/capacity limit. The standard tail ratio
is fixed before evaluation and is not changed in response to a run.

Proposal scoring may use observations to improve mixing, but it does not alter
the target likelihood and every non-prior proposal density appears in the MH/RJ
ratio. Accepted rejuvenation moves leave outer particle weights unchanged.

An in-process caller may optionally stage an immutable external surface-density grid
for one exact incoming record prefix. This is a proposal contract, not an estimator
contract: PF maps the grid to its own chart atlas in one batched neighbor query and
mixes it with the native residual proposal. The positive area-prior component retains
full support, and the same frozen proposal density is used in every forward/reverse
MH/RJ term. Standalone PF does not stage this value; the MLE-guided hybrid orchestrator
uses it to alter the accepted finite particle realization without changing the PF
target or weights directly.

Diagnostics report the current ESS after all applied likelihood, the number of
surviving station-start ancestors, and attempted/accepted posterior weight mass
for every structural move. Rejections also retain quantiles of the likelihood,
prior, reverse-minus-forward proposal, Jacobian, support, nonfinite, and MH
random terms. Counts of unweighted moved particles are retained only as
secondary diagnostics.

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

The controller starts adaptive-stop assessment at the configured station and
requires a configured number of consecutive ready posterior generations. The
standard contract assesses stations 10 onward and requires three consecutive
ready generations, so the earliest adaptive stop is station 12. The runtime
acquisition contract remains the sole owner of the hard station limit.

## Planner

DSS-PP receives aligned PF particles without renormalizing away `P(K=0)`.
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
the same distribution and have equivalence tests. CPU execution uses batched
NumPy or multithreaded Torch-CPU kernels; there is no separate Python-worker
count. Geant4 uses native worker threads. Scalar implementations are limited
to tiny deterministic test or debug oracles.

GPU DSS keeps selected-pair transport, source/line packing, exact predictive
spectra, cross likelihoods, and EIG aggregation on one Torch device. Only the
final action-score vector returns to the host. A bounded control-plane loop
constructs one canonical generator per action so memory chunking and action
ordering cannot change its random stream; it does not iterate over particles,
source slots, views, or spectrum bins.

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

`PFLiveSession` owns this in-process causal boundary. It holds the estimator and
ordered durable records together, uses the same canonical station assimilation
helper as the standalone controller, and rejects record delivery after completion.
Optional surface guidance is accepted only at a station boundary and must identify the
run, station, final step, record count, and ordered-prefix digest exactly. The returned
receipt records the mapped chart count and all isotopes for which the proposal evaluator
consumed the guide.
Its planning DTO copies particle arrays into read-only storage and includes a
canonical, truth-free PF posterior summary. The DTO deliberately preserves PF
particle semantics; it is not reshaped into an MLE-style surface grid. The current
estimator has no public deterministic latest predicted-spectrum snapshot, so the
facade does not derive a surrogate from private model state or stochastic posterior
predictive diagnostics.

At termination PF asks the runtime to publish the immutable MeasurementLog,
validates its exact ordered-record digest against the stations assimilated in
that live session, and binds the posterior provenance to the published bundle
digest. `complete_live_state()` seals the inference state before publication;
`bind_published_log()` then changes only final provenance identities and exposes
canonical posterior/state bytes through `PFBoundLiveState`. Contract mismatches
fail before publication.
