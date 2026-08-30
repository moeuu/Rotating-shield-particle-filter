# Current RA-L Manuscript Plan

This document is the source of truth for rewriting the RA-L paper from the
current implementation. It separates the proposed method, supporting
engineering, experiment interventions, and evidence that has not yet been
collected.

## Working title and central claim

Working title:

> Posterior-Adaptive Attenuation Coding for Online Multi-Isotope Source-Term
> Estimation

The paper should not present a shield-aware path planner wrapped around a
generic particle filter. The proposed method is the coupled loop in which:

1. the current joint posterior predicts how every reachable detector pose and
   all 64 Fe/Pb orientation pairs would encode competing source hypotheses;
2. the planner chooses an eight-view attenuation code and a reachable 3-D
   detector pose; and
3. the exact shield pose used for every acquired spectrum conditions the next
   full-spectrum SMC weighting and reversible-jump transition.

The concise message is: **the planner designs a physical attenuation code and
the transdimensional SMC decodes that same code**.

## Problem and state

The input is a causal stream of eight-view stations. Each view contains the raw
energy-binned count spectrum, detector pose, Fe orientation, Pb orientation,
live time, and the authenticated environment/obstacle contract. The active
isotope catalog and its line energies and branching weights are supplied.

One aligned particle represents all configured isotopes:

```text
X = {K_i, (surface_i,j, u_i,j, v_i,j, a_i,j) for j=1..K_i}_i.
```

Here `K_i` is the unknown source count, `(surface,u,v)` is a continuous point
on a room or obstacle surface, and `a` is net detector count rate at 1 m. All
isotopes share one outer particle weight and one resampling ancestry. This is
not a collection of independent isotope filters.

For energy bin `b` and shielded view `v`, the expected source spectrum has the
form

```text
mu_vb(X) = background_vb + sum_i sum_j a_i,j
           sum_l q_i,l T_v,l(x_i,j) K_b(E_i,l, xi_v,i,j).
```

`q` is the supplied isotope line catalog, `T` is shared air/obstacle/Fe/Pb
transport, and `K` is the isotope-independent detector response operator. The
likelihood also carries the current hierarchical background, mark, component,
and finite-Monte-Carlo uncertainty terms. The paper should state this factor
boundary and cite standard detector-response and finite-template uncertainty
work instead of spending space on implementation detail.

## Proposed method content

### 1. Shield-conditioned joint full-spectrum SMC

Every measured Fe/Pb pair enters the physical response and therefore the joint
full-spectrum likelihood used for particle weighting. A complete station is
tempered from beta 0 to 1 as one joint factor. The eight views are not ingested
as an arbitrary ordered prefix, preventing view-order-dependent mode deletion.
All isotope states are resampled with the same ancestor vector.

This subsection must make explicit that the shield is used by inference, not
only by planning. A no-shield observation or the wrong shield-pair identity
defines a different likelihood and cannot be silently substituted.

### 2. Shield-aware exact transdimensional rejuvenation

The rejuvenation kernel targets the same shield-conditioned posterior and uses
the following exact MH/RJ proposals:

- full-support, data-informed birth/global proposals;
- continuous surface moves with a fixed mixture of 3 cm, 15 cm, and 50 cm
  tangent scales;
- a coupled position-strength map that approximately preserves integrated
  all-history response and includes its area and strength Jacobian;
- shifted-Gamma strength moves with no artificial upper ceiling;
- response-aware pair and multi-component death/merge/split proposals; and
- joint isotope-strength/state blocks evaluated against the mixed spectrum.

Every non-prior selector, forward/reverse density, and Jacobian remains in the
acceptance ratio. Observation-guided proposals improve exploration but do not
change the posterior target. Nearby raw components may describe one physical
cluster; remote response-distinct components and hard-cap saturation are the
failures of interest.

### 3. Posterior-adaptive attenuation-code and pose design

For each reachable detector pose, the model constructs a virtual 64-view
station using all Fe/Pb orientation pairs and joint posterior particles. The
planner selects eight distinct pairs by batched conditional greedy, then tests
all 448 one-pass 1-swap neighbours at each exact shortlisted pose. The selected
program maximizes full-spectrum expected information gain; final pose utility
adds declared coverage, bearing, frontier, local-orbit, elevation, revisit, and
runtime-authored motion-time terms.

Conditional greedy plus one swap is not exhaustive over `C(64,8)` and has no
global-optimality claim. Its contribution is a tractable posterior-adaptive
physical code over the complete 64-pair hardware alphabet, evaluated with the
same full-spectrum probability law as inference.

### 4. Exact transition-preserving history scheduling

TPHT is a supporting computational contribution, not the definition of the
whole proposed method. It keeps the complete acquired history as the posterior
target but evaluates a proposal in stages:

```text
latest exact station -> recent exact stations -> dyadic exact old blocks
                     -> full exact replay only for survivors.
```

It uses a reversible two-factor delayed-acceptance decision. An unresolved row
can be rejected by a valid probability upper bound, but it can never be
accepted without evaluating every required exact factor. Accepted rows are
replayed and checked before an atomic CUDA-cache commit.

Tree metadata is logarithmic, but proposal work is data dependent from one to
all `T` stations. The manuscript must retain the worst-case `O(T)` work per
proposal and `O(T^2)` acquisition statement; it may report the measured
reduction in evaluated stations and ESJD per second. It must not claim an
unconditional `O(T log T)` algorithm.

## What is and is not novel

The paper's clearest novelty is the coupling of posterior-adaptive physical
attenuation-code design with shield-conditioned joint transdimensional
full-spectrum SMC. Shield hardware, particle filters, reversible-jump MCMC,
expected information gain, conditional greedy selection, and delayed
acceptance each have prior art and require citations.

The defensible implementation-specific contribution is their exact coupling:

- an eight-of-64 Fe/Pb code is selected from the joint multi-isotope posterior;
- the actual code conditions particle weights and structural moves;
- continuous surface position and strength are moved jointly under that coded
  likelihood; and
- TPHT skips unnecessary old transport while preserving the same posterior.

Do not claim the first directional detector, first active radiation search,
first use of shielding, first RJ particle filter, or universal radionuclide
validation. The current application acceptance is isotope-profile specific.

GPU batching, fixed-capacity caches, strict schemas, provenance binding,
fail-close lifecycle checks, and CUI rendering are important reproducibility
engineering but should receive only compact implementation text.

## Four full-simulation comparisons

The main paper uses exactly four paired high-fidelity closed-loop runs on one
fresh `4 Cs-137 + 3 Co-60 + 2 Eu-154` scene:

| Variant | Physical shield | Eight-pair code | Pose policy | Question |
| --- | --- | --- | --- | --- |
| `proposed` | Fe/Pb | posterior-adaptive | native DSS-PP | complete method |
| `no_shield_native_path` | absent | physically ineffective | native DSS-PP | value of attenuation coding |
| `round_robin_shield` | Fe/Pb | independent round robin | native DSS-PP using forced code | value of adaptive code design |
| `eig_only_path` | Fe/Pb | posterior-adaptive | EIG minus measured motion time | value of spatial guidance |

All variants retain the same exact RJSMC, catalog, detector response, candidate
contract, maximum 16 stations, eight views per station, and 20 s per view. The
no-shield variant retains the native planner algorithm but need not select the
same physical poses because its predictive distribution is different.

TPHT versus one-stage exact full-history RJPF is a separate computation
ablation at matched saved states and proposals. It is not a fifth or sixth
Geant4 run. Report sweep time, evaluated stations per proposal, full-history
fraction, peak GPU memory, position/log-strength ESJD per second, exact-target
agreement, and detailed-balance tests.

## Evaluation contract

The primary unit is a true physical source, not a raw RJ component. Apply the
predeclared post-run clustering and one-to-one truth matching separately for
each isotope.

Primary outcomes:

- true-source cluster recall;
- per-source 3-D Euclidean position error;
- per-source relative cluster-strength error;
- fraction passing both 0.5 m and 25% thresholds;
- count of response-distinct remote components;
- posterior mass at the `K=8` hard capacity;
- station and detector-live-time to the first stable pass; and
- physical mission time, with motion/settling reported separately from the
  fixed maximum 2560 s detector live time.

Report medians and ranges or IQRs across sources as descriptive summaries, but
do not pretend that nine sources in one scene are independent repeated trials.
The four runs form one paired batch. Additional independent batches are needed
for inferential statistics.

Raw component cardinality is not an accuracy outcome. A raw `K=6` estimate can
be successful if all four physical source clusters have correct aggregate
strength and no remote response-distinct component remains. Hard-cap mass above
0.05 remains a sampler failure.

## Figure and page plan

Use three vector PDF figures with text at least 7 pt at final size:

1. **Problem and attenuation code** (about 0.20--0.23 page, two columns): room,
   surface sources, mobile detector, the actual Fe/Pb octant geometry, and
   several selected pair views. It explains the physical measurement code.
2. **Coupled method** (about 0.24--0.28 page, two columns): a compact
   `64 pairs -> selected 8 -> shield-conditioned spectra -> full-station
   SMC/RJ -> posterior -> next code/pose` loop. Every graphical stage must
   correspond to an implemented operation; do not invent a response heatmap
   before the fresh batch provides an auditable response artifact.
3. **Auditable result** (about 0.32--0.36 page, two columns): equal-metric x-y
   and height projections with truth, estimates, stations, and obstacles;
   cardinality evolution; and per-source position/strength error thresholds.
   Do not draw straight station-to-station routes unless obstacle-aware
   waypoints were persisted.

Planned text allocation is page 1 introduction and Fig. 1; pages 2--4 model and
method with Fig. 2; page 5 TPHT and experiment contract; pages 6--7 Fig. 3,
tables, results, and discussion; page 8 limitations, conclusion, and references.

## Current evidence boundary

The most recent fully completed durable run available before the fresh paper
batch is `cs4-co3-20260827-175045`. It used predecessor estimator code, 16
stations, 128 spectra, and 20 s per view. A nearest-mode diagnostic gives five
of seven sources within 0.5 m and three of seven within both the position and
strength thresholds. Co-60 ends near `P(K=3)=0.977`; Cs-137 ends at the `K=8`
capacity with approximately 0.998 mass. The run therefore demonstrates honest
failure visualization, not the accuracy or superiority of the current method.

It predates the current evaluator artifact needed for response-distinct remote
component certification. The manuscript must label it “completed
predecessor-code diagnostic,” omit comparative bars, and leave the final
four-variant result table explicitly pending rather than fabricating values.

## Citation boundary

Use compact citations for: active radiation-source localization with particle
filters; active planning/EIG; RJMCMC; SMC tempering; delayed acceptance;
finite-Monte-Carlo template uncertainty; Geant4; evaluated nuclear line data;
and detector response matrices. Cite the authors' prior shielded-search work to
make the incremental claim explicit. Standards and simulator implementation
details belong in the experiment section, not the novelty statement.

## Venue compliance snapshot

The target is the double-anonymous IEEE Robotics and Automation Letters initial
submission format. The official author instructions were checked on 2026-08-31:
the paper uses US Letter, the `ieeeconf` 10-point template, and at most eight
pages including figures, tables, appendices, and references (six regular pages
plus at most two excess pages). The first-page masked acknowledgment remains
exactly `This work was in part supported by XXX.`. Generated vector figures use
3.5-inch or 7.16-inch IEEE widths, embedded fonts, and a minimum final-size text
of 7 pt. These venue details must be rechecked against the official site before
submission because external requirements may change.
