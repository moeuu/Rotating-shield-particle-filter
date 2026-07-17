# External relocation boundary

`python -m pf.hybrid_replay` is an opt-in, estimator-neutral boundary for a
hybrid controller.  It does not change `pf.replay`, `PurePFEstimator`, or the
scientific meaning of `pf_strict`/`pf_profiled`.

The separate [hybrid DSS-PP recommendation boundary](hybrid_planning_boundary.md)
can causally replay these directives and expose eligible external modes to the
planner without granting them PF-state authority.

## Statistical operation

An external directive may relocate one existing source slot per eligible PF
particle.  It may not add or remove a slot, replace a strength, replace the
background, change a particle weight, hard-prune a state, or add an external
objective to the PF likelihood.

The proposal is

```text
q(y | x) = d Normal_box(y; x, sigma_defensive)
           + (1 - d) sum_j w_j Normal_box(y; candidate_j, sigma_j)
```

where every Gaussian is independently truncated and normalized on the PF XYZ
box.  The Metropolis-within-Gibbs acceptance rule is

```text
log alpha = min(0,
                log p_pf(y, fixed remainder | observations 0:cutoff)
                - log p_pf(x, fixed remainder | observations 0:cutoff)
                + log q(x | y) - log q(y | x)).
```

The old and proposed targets are evaluated together with the production
continuous response kernel in one particle/proposal batch.  `log_weight` is
asserted unchanged after the move.  A serial evaluator exists only as a test
oracle.

The supported v1 target is the complete configured count likelihood through
the cutoff plus the uniform volume-position prior.  The implementation fails
closed when a surface-projected position prior, shield contrast likelihood,
shield-view-ratio likelihood, or correlated station-view likelihood is active.
Projecting a Gaussian onto a union of surfaces would not have the density used
by the MH correction, so it is never done silently.

## Causal binding

Each directive binds all of these values:

- `source_run_id`;
- `covered_records_sha256`;
- `pf_resolved_config_sha256`;
- `data_cutoff_step` and `data_cutoff_station`;
- the exact `covered_step_ids`;
- an explicitly writer-marked station-complete record.

`covered_records_sha256` is the SHA-256 of canonical JSON for the exact neutral
MeasurementLog records through the cutoff.  It excludes the directory digest,
manifest record count, and every unseen suffix.  The relocation RNG is derived
from this prefix hash, the proposal mixture, isotope, and explicit relocation
seed.  A future directive or a changed unseen suffix therefore cannot alter a
prefix state.

The full-log `source_measurement_log_sha256` may be retained as provenance, but
it is intentionally not an RNG input or causal binding.

## Orchestrator PFDirective v1

The loader accepts either the native schedule representation or an
orchestrator `PFDirective` with `directive_kind="proposal_only_mh"`.  The latter
must provide `source_run_id`, `covered_records_sha256`, and
`pf_resolved_config_sha256` either at top level or in `provenance`.  Its proposal
kernel is:

```json
{
  "family": "defensive_truncated_gaussian_position",
  "position_sigma_xyz_m": [0.2, 0.2, 0.2],
  "defensive_weight": 0.25,
  "candidate_weight": 1.0
}
```

All proposals in one v1 directive use the same defensive weight and XYZ sigma.
`strength_cps_1m` remains snapshot metadata and is not copied into the PF.

## Command and outputs

```bash
uv run python -m pf.hybrid_replay \
  --measurement-log /path/to/log \
  --config /path/to/pf-config.json \
  --directive-schedule /path/to/pf-directive-schedule.json \
  --profile pf_strict \
  --seed 7 \
  --relocation-seed 11 \
  --output-dir /path/to/output
```

The standard `pf_posterior.json`, `pf_trace.jsonl`, and `pf_diagnostics.json`
remain byte-compatible with standard replay when the schedule is empty.  The
wrapper additionally writes:

- `external_directive_schedule.json`;
- `external_relocation_receipts.json`, including per-particle target deltas,
  proposal corrections, log acceptance ratios, random draws, and accept flags;
- `pf_directive_receipts/receipt-*.json`, one orchestrator-contract receipt per
  applied directive.  Every candidate records `mh_attempt_count`,
  `mh_accepted_count`, `mh_rejected_count`, `not_sampled_count`, and the total
  eligible-particle count.  `outcome` is an aggregate category
  (`mh_accepted`, `mh_rejected`, `mh_mixed`, or `not_applied`), never a selected
  representative particle.  The scalar log-ratio and uniform draw are present
  only for a singleton attempt; many-particle evidence remains in the detailed
  particle receipt.  Counts are candidate-local: `not_sampled_count` includes
  defensive draws and draws assigned to another external candidate;
- `pf_pre_update_predictive.jsonl`, containing observation-independent PF
  predictive count mean, epistemic variance, and Poisson predictive variance
  before each row is processed;
- `hybrid_pf_posterior.json`, an honest wrapper around the base PF posterior
  identifying the externally proposed target-preserving move;
- `hybrid_diagnostics.json`.

`--stop-after N` leaves later directives explicitly pending.  Directives apply
only immediately after their cutoff station update and before record `N + 1`.
This v1 performs schedule-driven replay from the beginning and does not expose
a resumable PF checkpoint.

## Deliberate limitations

- Fixed-cardinality relocation cannot discover a source when every particle
  has zero slots for that isotope.
- It is not reversible-jump birth/death and must not be described as such.
- Candidate corroboration, quarantine, pruning, conflict planning, and final
  batch reporting remain controller responsibilities.
- Later observations are processed once by the ordinary PF update.  They are
  not reweighted through the external snapshot.
