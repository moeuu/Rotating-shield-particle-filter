# Transition-Preserving History Tree (TPHT-RJPF)

## Scope

TPHT reduces repeated proposal work inside online exact-RJ rejuvenation. It
does not change Geant4 transport, detector or shield physics, DSS/EIG planning,
the eight-pose station likelihood, isotope lines, priors, or source-rate
semantics. The posterior target remains the complete acquired history.

TPHT is an exact-target transition scheduler. It is not a coreset, an averaged
spectrum, a weighted pseudo-likelihood, or a replacement observation model.

## Exact history tree

Stations are represented by a newest-first dyadic forest. With 16 acquired
stations, its roots are:

```text
[15] [14] [12:14] [8:12] [0:8]
```

Old roots are split into exact child blocks of at most four stations before
proposal evaluation. The production order is therefore:

```text
[15] [14] [12:14] [8:12] [4:8] [0:4]
```

Every evaluated child contains the real observations, geometry, Fe/Pb poses,
dwell times, and exact source-resolved full-spectrum likelihood for those
stations. No station represents another station, and no block-size weight is
applied. The latest station is evaluated first because it is a real factor of
the online posterior, not because it is a proxy for old data.

Candidate transport is generated only for the child currently being examined.
Unchanged source slots are read from the fixed accepted CUDA cache. A proposal
rejected at the latest factor therefore never generates transport for old
stations. A survivor descends through the exact children until rejection is
proved or every station has been evaluated.

## Reversible two-factor decision

Let `Delta_new` be the exact likelihood change at the latest station and let
`Delta_old` be the sum of exact changes at all older stations. Prior,
forward/reverse proposal, Jacobian, and other non-likelihood terms are grouped
in `Delta_rj`. The production acceptance probability is:

```text
alpha(x, y) = min(1, exp(Delta_new + Delta_rj))
              * min(1, exp(Delta_old))
```

The two factors use independent uniforms. In the reverse direction both log
ratios change sign, so the ratio of forward and reverse acceptance
probabilities is:

```text
alpha(x, y) / alpha(y, x)
    = exp(Delta_new + Delta_old + Delta_rj)
```

This is the complete-history RJ/MH ratio. The transition kernel differs from a
one-stage MH kernel and can have lower movement per sweep, but it is reversible
for the same exact posterior and introduces no stationary approximation bias.

During old-history refinement, each unevaluated station uses only the universal
discrete-PMF fact `log p(observation | state) <= 0`. If the resulting upper
bound is already below the second uniform threshold, rejection is exact and
remaining children are skipped. Otherwise refinement continues. An ambiguous
row is never accepted from a bound: it must reach all exact station leaves.

Accepted structural rows are replayed once over the full acquired interval to
stage an aligned transport-cache commit. The replay must match the independently
refined station likelihoods within the declared float64 tolerance or the sweep
fails closed.

## Complexity

Tree metadata has `O(log T)` roots, but exact proposal work is data dependent.
For one proposal, it lies between one station and all `T` stations. The
worst-case acquisition cost remains `O(T^2)` because an exact method must allow
every proposal to remain ambiguous through the full history. TPHT does not
claim an unconditional `O(T log T)` bound.

The practical objective is to prevent every rejected proposal from paying the
worst-case cost. The live diagnostics record:

- evaluated stations per proposal;
- latest-factor rejection fraction;
- refinement-bound rejection fraction;
- full-history fraction;
- exact rejection fraction after full refinement;
- exact block calls, staged replay rows, and maximum tree level.

From station eight onward, two consecutive sweeps with mean evaluated history
at least 80% of the acquired station count set
`tpht.history_scaling_warning=1`. This is a visible performance warning, not an
accuracy fallback. The fixed 16-station/128-view capacity remains fail-closed.

## Quantitative reference benchmark

The paired benchmark uses the same persisted Cs-137/Co-60 posterior, 4,096
particles, 16 stations, exact hierarchical likelihood, proposal configuration,
and RTX 5090. The one-stage full-history reference took 111.495 s. Three TPHT
runs took 74.176 s, 76.014 s, and 81.541 s; the median is 76.014 s.

| Metric | One-stage exact | Hierarchical TPHT | Change |
| --- | ---: | ---: | ---: |
| Sweep wall time | 111.495 s | 76.014 s median | 1.467x faster |
| Mean evaluated stations/proposal | 16 nominal | 4.385 | 72.6% fewer |
| Full-history proposal fraction | 100% nominal | 22.63% | 77.37 points lower |
| Position ESJD | 2.3658 m2 | 2.1423 m2 | 90.55% retained |
| Log-strength ESJD | 0.013264 | 0.011674 | 88.02% retained |
| Position ESJD/s | reference | measured | 1.328x higher |
| Log-strength ESJD/s | reference | measured | 1.291x higher |

The latest exact factor rejected 33,496 of 43,871 proposal rows (76.35%). In
this checkpoint the universal old-history bound produced no additional early
rejections; its role was correctness-preserving progressive refinement. The
speed gain therefore comes from avoiding old transport for latest-factor
rejects, not from pretending that a coarse old block is exact.

ESJD is a finite-sweep exploration diagnostic, not posterior accuracy. TPHT
retained less movement per sweep but more movement per second. Its asymptotic
posterior accuracy is unchanged by construction; finite-run convergence must
still be monitored with the existing mixing diagnostics.

## Required verification

Tests must cover:

- disjoint history coverage and exact 16-station root/child schedules;
- latest-station evaluation without a representative or block weight;
- batched decisions against the scalar two-factor formula;
- detailed balance for the factorized acceptance probability;
- progressive exact refinement of an ambiguous row;
- certified early rejection without accepting an unresolved row;
- accepted replay equality and station-cache alignment;
- strength, structural, and cross-isotope production-path wiring; and
- fail-close behavior for cache, shape, dtype, device, PMF, and history
  violations.

Because this changes only PF transition scheduling and keeps the physics and
likelihood target unchanged, it requires numerical and transition tests, not a
new Geant4 acceptance campaign.

