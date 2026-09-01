# Post-run PF cluster accuracy policy

This policy is the standard detailed evaluation for every completed full
simulation. It is an evaluation contract, not part of the PF target. Private
truth may be joined only after the run has completed and only when the exact
`run_id` matches.

## What counts as a source estimate

The 0.5 m value is a localization target, not the cutoff for deciding whether a
true source was detected. Truth association and target attainment are separate
results.

For each true source, the maximum split-assignment radius is

```text
min(4 * position_target, 0.5 * nearest_same_isotope_truth_separation).
```

With the standard 0.5 m target, the unconstrained maximum is 2.0 m. The
same-isotope-separation term prevents assignment regions from overlapping when
two physical sources are close. If there is no other true source of that
isotope, the separation term is infinite.

Every PF component is assigned to a true source only when that source is its
unique nearest same-isotope truth and the component lies within that source's
split-assignment radius. A component is assigned at most once. Exact
nearest-truth ties remain unassigned. Response signatures are not used to pick
which components are favorable to a truth source.

Assigned components at most 0.5 m from truth are labelled `core`; assigned
components farther than 0.5 m are labelled `extended_split`. Both groups
contribute fully to strength and position scoring. Thus a component just beyond
the position target is not reported as a missing detection, while a broad split
still carries an explicit accuracy cost.

The raw PF cardinality is reported but is not an accuracy target. A result with
four physical source clusters may therefore pass with raw K=5 or K=6 when one
or more clusters contain a corner or surface split.

## Position and strength of an assigned split cluster

For assigned component positions `x_i` and positive reported strengths `w_i`,
the merged position is the strength-weighted centroid

```text
x_bar = sum(w_i * x_i) / sum(w_i).
```

The centroid-to-truth distance is reported as the merged localization bias, but
it is not sufficient by itself: components on opposite sides of truth can
cancel to an apparently perfect centroid. The position target is therefore
scored with the strength-weighted RMS distance to truth

```text
e_rms = sqrt(sum(w_i * ||x_i - x_truth||^2) / sum(w_i)).
```

The report also records the strength-weighted spatial dispersion about the
centroid. These values obey

```text
e_rms^2 = ||x_bar - x_truth||^2 + dispersion^2.
```

This decomposition distinguishes a coherently displaced cluster from a
truth-centered but widely fragmented cluster. The strength-weighted medoid is
retained only as a display/audit component and is never used for position
scoring. Cluster strength is the sum of every assigned component strength.

For every isotope and every true source, the report records:

- true position, merged centroid, and signed centroid error;
- centroid error, spatial dispersion, and strength-weighted RMS position error;
- the fraction of assigned strength lying within the 0.5 m target;
- true strength and summed cluster strength;
- absolute and relative strength error;
- core and extended-split raw component indices and their truth distances; and
- individual association, position-target, strength-target, and joint results.

The predeclared accuracy targets are at most 0.5 m strength-weighted RMS
position error and at most 25% relative cluster-strength error. Association is
reported separately as `truth_source_detection_status`. Consequently, an
associated source can be described as detected even when one or both accuracy
targets are not met.

## Remote components

An estimate is remote when it has no same-isotope truth, has an equidistant
nearest-truth ambiguity, or lies outside its nearest truth's split-assignment
radius. The report records the exclusion reason and applicable radius. The PF
publication includes a truth-free response signature for every reported mode:
the normalized same-isotope expected-count vector across all completed
measurements.

A remote component is response-distinct when its cosine similarity to every
covered source-cluster response is below 0.995. Any response-distinct remote
component fails the post-run accuracy assessment. It does not retroactively
make the completed acquisition an execution failure. A spatially remote but
response-indistinguishable component is reported explicitly as an
observability ambiguity; it is not silently counted as a new physical source
and does not fail the raw-cardinality criterion.

## Sampler capacity

Raw K is not scored, but posterior mass at the configured hard source cap must
not exceed 0.05 for any isotope. Exceeding this limit is a sampler/model-capacity
failure, not evidence that the hard-cap cardinality is correct.

The value 0.05 is defined once in `pf.cardinality_policy` and is consumed by
both live health checks and this post-run evaluator. It is not a live-profile
setting and cannot be changed per run. Ordinary K=5/K=6 mass remains a
diagnostic only. When that mass is material, live health verifies that inward
proposals were attempted, had support, and produced a finite MH ratio; a finite
batch is allowed to reject every such proposal.

## Independent result statuses

Every result reports three separate statuses:

- `execution_status`: whether the requested acquisition and posterior
  publication completed;
- `sampler_quality_status`: whether truth-free hard-cap, lineage, and mixing
  diagnostics passed; and
- `accuracy_status`: the private-truth position, strength, source-coverage, and
  remote-component assessment performed only after completion.

`accuracy_status=pass` requires all of the following:

1. every true same-isotope source has an assigned estimated cluster;
2. every cluster meets the RMS-position and summed-strength targets;
3. no response-distinct remote component remains.

`truth_source_detection_status=pass` requires only item 1. A position-target
failure does not erase evidence that the source was detected; it states that
the requested localization quality was not achieved.

Hard-cap saturation belongs only to `sampler_quality_status`. A run may
therefore have `execution_status=complete`, `sampler_quality_status=failed`,
and `accuracy_status=pass`, or any other evidence-consistent combination. No
aggregate legacy `passed` field is emitted.

The criteria payload and its SHA-256 digest are written into every evaluation
artifact. Changing a criterion requires changing this policy before evaluating
the next comparison batch; per-run or per-seed overrides are not accepted by
the standard CLI.

The truth-free `pf_post_run_evaluation_input.json` has one exact schema-v1
contract. Unknown or missing fields, a mismatched run or MeasurementLog hash,
reordered modes, duplicate mode labels, and zero or non-normalized response
columns are errors. They are never ignored or replaced with defaults.

The split-aware truth-bearing evaluation report uses schema v3 and
cross-checks its hard-cap evidence against the published sampler-quality
status. A contradiction is an artifact-integrity error rather than a status
that can be silently reconciled.

Schema-v3 rules apply prospectively to comparison batches evaluated after this
policy change. Existing schema-v2 reports remain immutable and must not be
overwritten or relabelled. A separately named retrospective schema-v3 diagnostic
may explain an older run, but it is not independent acceptance evidence and must
not be mixed with a prospective comparison table.

The private RA-L session orchestrator runs this evaluation after every
successful full simulation. It does not pass truth to the PF controller and it
does not modify the already-published atomic PF result bundle. The detailed
truth-bearing report is written below the sibling runtime's ignored
`private_runs/ral_ablation/evaluations/` directory. A pre-existing report is
not overwritten.

## Standard command

After the runtime has finalized the `MeasurementLog` and PF has atomically
published its artifacts, run:

```bash
uv run python scripts/evaluate_completed_pf_run.py \
  --result RUN/pf_output/closed_loop_result.json \
  --posterior RUN/pf_output/pf_posterior.json \
  --evaluation-input RUN/pf_output/pf_post_run_evaluation_input.json \
  --truth-manifest RUN/truth_manifest.json \
  --output PRIVATE_RUN/evaluations/RUN.json
```

The command rejects incomplete runs, cross-run truth, mismatched MeasurementLog
hashes, missing response signatures, and reordered posterior modes.
