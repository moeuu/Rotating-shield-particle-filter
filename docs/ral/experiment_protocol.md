# RA-L Experiment Protocol

This document defines the reusable comparison contract for the current RA-L
paper. It contains no private truth, opaque run ID, or result interpretation.
Run-specific provenance belongs below the sibling runtime's ignored
`private_runs/ral_ablation/` tree and in each durable result bundle.

## Task

Use one paired high-fidelity task:

- scenario: `cs4_co3_multi_source_cardinality`;
- truth: four Cs-137 and three Co-60 sources;
- support: continuous exposed room and transport-component surfaces;
- placement: uniform physical-surface measure conditioned on a predeclared
  3.0 m minimum 3-D separation between sources of the same isotope;
- no visibility, height, ceiling-count, detector-response, or observed-count
  screening.

The sibling runtime is the sole owner of environment and truth realization.
The PF consumes only the truth-free environment contract and causal
MeasurementLog stream.

## Seeds and Pairing

Generate a fresh private scene seed and opaque batch ID for every new comparison
batch by omitting `--seeds` from the ablation generator. Generate PF and Geant4
transport seeds independently; none of the three seeds may be equal.

All four variants in one batch must share byte-canonical comparison-bearing
environment, obstacle geometry, source positions and strengths, isotope set,
scene RNG provenance, and acquisition contract. Reuse the PF seed across those
four variants. Reuse a recorded scene only to repeat that same live batch, and
then supply its recorded scene, PF, transport, and batch identities together.
Partial replay provenance is an error.

Private scene seeds, truth manifests, and source RNG provenance must not enter a
PF config, adaptive event, MeasurementLog, planner input, or PF artifact.

## Acquisition and PF Contract

Every variant uses:

- `pure_pf_schema_version: 2` and profile `pf_strict`;
- the same joint full-spectrum reversible-jump particle filter;
- ordinary source-count support through five sources per isotope and the same
  fixed geometric capacity tail through the hard limit of eight;
- at most 16 complete stations;
- exactly eight shield views per acquired station;
- 20.0 s live time per view and at most 128 measurements;
- at most 2560 s detector live time;
- 3.0 m station-separation penalty scale and 3.0 m coverage radius; and
- the same runtime-authored reachable poses and motion costs.

Station separation is a planner term, not a hard reachability constraint.
Physical reachability and collision clearance come only from the runtime.

## Four Variants

| Variant | Shield | Shield program | Pose policy | Scientific question |
| --- | --- | --- | --- | --- |
| `proposed` | Fe/Pb | posterior-adaptive eight-of-64 | native DSS-PP | complete method |
| `no_shield_native_path` | absent | physically ineffective | native DSS-PP | value of attenuation coding |
| `round_robin_shield` | Fe/Pb | independent deterministic round robin | native DSS-PP using the forced code | value of adaptive code design |
| `eig_only_path` | Fe/Pb | posterior-adaptive | full-spectrum EIG minus runtime motion time | value of spatial guidance |

Variants change only the declared shield or planning intervention. They retain
the same observation model, exact-RJ/SMC target, candidate contract, particle
configuration, measurement budget, and spectral fidelity. The no-shield
variant need not follow the same path because removing attenuation changes its
predictive distribution.

`eig_only_path` disables coverage, coverage reserve, bearing, frontier,
revisit, turn, local-orbit, and elevation terms through their strict inactive
schema states. `round_robin_shield` forces only the shield program; native pose
selection remains active.

## Generation and Execution

Generate a fresh exhaustive batch and its four-run paper subset with:

```bash
PYTHONPATH=src uv run python -m baselines.ral_ablation.cli
uv run python scripts/build_ral_paper_subset.py
```

When multiple recorded batches exist, select one explicitly with `--batch-id`.
Scene-seed-only selection is not supported. The generated private manifest and
run script live under the sibling runtime's ignored
`private_runs/ral_ablation/` directory.

Execute the selected subset with:

```bash
bash ../Rotating-shield-simulation-runtime/private_runs/ral_ablation/run_paper_subset.sh
```

Before acquisition, the private batch contract must verify equality of every
comparison-bearing physical field. Each PF controller receives an opaque Unix
socket, truth-free configuration, independently sealed control policy, and no
private scenario path. Long runs follow the persistent-session and monitoring
requirements in the [PF inference fidelity policy](../policies/inference_fidelity.md).

The runtime may expose private truth to its asynchronous CUI renderer through a
dedicated owner-only endpoint. That payload must remain inaccessible to PF and
must never be serialized into estimator state or published PF artifacts.

## Artifacts and Evaluation

Each variant must publish a unique truth-free MeasurementLog, PF result bundle,
particle snapshot, station/planner trace, performance trace, diagnostics, and
artifact inventory. Targets must not be overwritten. Private truth and the
truth-bearing evaluation remain in the sibling runtime's ignored private tree.

After every successful full simulation, apply the fixed
[post-run evaluation policy](../policies/post_run_evaluation.md). The evaluator
must join exact matching run identities and preserve both its criteria payload
and digest. Existing reports remain immutable when a later schema or policy is
introduced.

## Reporting Contract

Do not place placeholder comparison numbers in the paper. An older or
predecessor-code run may appear only as a clearly labelled diagnostic and may
not establish current accuracy or comparative advantage.

For the paired four-run batch, report:

- true-source association recall;
- split-aware merged estimated-source count;
- per-source merged-centroid bias, strength-weighted RMS 3-D error, split
  dispersion, and relative summed-strength error;
- joint 0.5 m RMS-position and 25% strength-target fraction;
- response-distinct remote-component count;
- hard-cap posterior mass and sampler-quality status;
- station and detector-live-time to stable success; and
- mission motion/settling time separately from detector live time.

Raw RJ component count is diagnostic, not the physical source-count outcome.
Treat the four variants as one paired scene and report descriptive paired
differences. Seven sources in one scene are not independent experimental
replicates and do not justify unsupported inferential statistics.

Generate paper figures only from authenticated source artifacts according to
the [RA-L figure policy](figures.md).
