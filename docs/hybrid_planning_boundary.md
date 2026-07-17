# Hybrid DSS-PP recommendation boundary

`python -m pf.hybrid_planning` is an opt-in planning boundary for an external
hybrid controller. It returns an **algorithmic recommendation only**. It does
not command a robot, move a detector, actuate Fe/Pb shields, append an
observation, or authorize execution.

The normal `pf.replay`, `PurePFEstimator`, and pure-PF DSS-PP path are
unchanged. A private fail-closed capability token prevents a normal
`select_dss_pp_next_station` call from reading a
`planner_only_external_signature_modes` hook. Only this explicit
recommendation boundary supplies that token.

## Processing boundary

The command performs the following operations:

1. Validate a schema-v1 planning request.
2. Bind it to one exact, writer-marked completed station using
   `source_run_id`, `data_cutoff_step`, `data_cutoff_station`,
   `covered_records_sha256`, and the resolved PF configuration digest.
3. Create an in-memory `MeasurementLog` prefix ending at that station.
4. Remove later relocation directives before parsing or applying the causal
   directive schedule.
5. Replay `pf_strict` or `pf_profiled` plus the existing target-preserving
   fixed-cardinality relocation directives through the cutoff.
6. Wrap the replayed estimator in a read-only planning view.
7. Add only `pending` and `verified` external modes as DSS-PP hypotheses.
   `quarantined` modes are recorded as excluded and never reach DSS-PP.
8. Run the existing `select_dss_pp_next_station` implementation over the
   supplied continuous XYZ candidates and Fe/Pb shield programs.
9. Verify that the canonical PF particle state is byte-identical before and
   after planning.
10. Emit the selected XYZ (including height), shield-program pair IDs, ranked
    diagnostics, causal provenance, exact belief sources, and safety
    attestation.

External mode strength, weight, and spread are planner metadata. They do not
alter PF particle weights, positions, strengths, backgrounds, or cardinality.
The earlier relocation boundary remains the only external PF-state operation,
and it remains a target-preserving Metropolis-within-Gibbs position proposal.

## Candidate safety attestation

The PF repository does not reconstruct the orchestrator's collision workspace.
The request must therefore provide an already filtered ordered candidate set
and this attestation:

```json
{
  "candidate_poses_sha256": "<64 lowercase hex>",
  "workspace_sha256": "<64 lowercase hex>",
  "planning_config_sha256": "<64 lowercase hex>",
  "collision_checked": true,
  "reachability_filtered": true
}
```

`candidate_poses_sha256` is recomputed over the exact ordered matrix after
conversion to a JSON list of IEEE-754 float64 values. Candidate order is part
of the digest because the recommendation reports the original candidate
index. The workspace and planning-config hashes are opaque upstream
attestations: the PF validates their SHA-256 syntax and echoes them, but does
not claim to have rerun those safety checks.

In Python, the shared digest operation is:

```python
sha256(canonical_json_bytes(np.asarray(candidate_xyz, dtype=np.float64).tolist()))
```

`canonical_json_bytes` is the repository-wide sorted, two-space-indented JSON
encoding with a trailing newline. This matches the orchestrator's existing
canonical JSON helper.

The DSS-PP request must explicitly set `augment_candidates` to `false`.
Otherwise DSS-PP could create poses that are absent from the attested set.
Legacy runtime-rescue and surface-rescue planner inputs must also remain
disabled.

## Request shape

```json
{
  "schema_version": 1,
  "request_id": "planning-request-001",
  "source_run_id": "run-001",
  "data_cutoff_step": 39,
  "data_cutoff_station": 7,
  "covered_records_sha256": "<prefix digest>",
  "pf_resolved_config_sha256": "<resolved PF digest>",
  "current_pose_xyz": [1.0, 2.0, 0.6],
  "current_pair_id": null,
  "visited_poses_xyz": [[1.0, 2.0, 0.6]],
  "candidate_poses_xyz": [
    [1.5, 2.5, 0.4],
    [2.0, 3.0, 1.1]
  ],
  "candidate_attestation": {
    "candidate_poses_sha256": "<candidate digest>",
    "workspace_sha256": "<workspace digest>",
    "planning_config_sha256": "<collision-planning config digest>",
    "collision_checked": true,
    "reachability_filtered": true
  },
  "dsspp_config": {
    "augment_candidates": false,
    "include_runtime_rescue_modes": false,
    "include_global_surface_rescue_modes": false
  },
  "external_modes": [
    {
      "mode_id": "candidate-Cs-137-001",
      "isotope": "Cs-137",
      "position_xyz": [3.0, 2.0, 1.4],
      "strength_cps_1m": 1200.0,
      "weight": 0.35,
      "spread_m": 0.25,
      "verification_state": "pending",
      "source_snapshot_id": "snapshot-004"
    }
  ],
  "bounds_xyz": {
    "min": [0.0, 0.0, 0.0],
    "max": [10.0, 10.0, 3.0]
  },
  "continuous_height_bounds_m": [0.2, 2.5]
}
```

`current_pair_id` may be `null`. External verification states are exactly
`pending`, `verified`, or `quarantined`. Quarantined entries are accepted for
an auditable exclusion receipt even when their isotope or position is outside
the active PF domain; they are never converted into a DSS-PP mode.

## Command

```bash
uv run python -m pf.hybrid_planning \
  --measurement-log /path/to/measurement-log \
  --config /path/to/pf-strict-config.json \
  --directive-schedule /path/to/pf-directives.json \
  --planning-request /path/to/planning-request.json \
  --profile pf_strict \
  --seed 37 \
  --relocation-seed 101 \
  --output /path/to/planning-recommendation.json
```

`--directive-schedule` is optional; omitting it means no external relocation
directives. Future schedule items are removed before directive validation, so
adding an unseen log or directive suffix cannot change the recommendation at
the bound cutoff.

## Result guarantees

The result records:

- `algorithmic_recommendation_only: true`;
- `robot_actuation_authorized: false`;
- the original attested `candidate_index`, exact XYZ, and detector height;
- the selected shield-program name, kind, and Fe/Pb pair IDs;
- the full DSS-PP sequence and diagnostics;
- exact belief-source labels (`pf_posterior`, `pf_tentative`, and only the
  eligible external verification states that were present);
- included external mode IDs and quarantined/excluded IDs;
- all three candidate-attestation hashes;
- the exact record-prefix and resolved-PF digests, without a full-log digest in
  causal identity;
- causal relocation-directive provenance;
- identical before/after PF-state SHA-256 digests.

Execution remains the responsibility of a separate runtime that must recheck
current workspace safety and decide whether to accept the recommendation.
