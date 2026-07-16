# PF purity audit

## Scope and definition

This audit records the mixed baseline at commit
`87961934add7aec893f41782e7488ddc50f2c636`. A pure PF state at observation
step `t`, its planner belief, dwell/stop decision, and its report may depend on
observations `1:t` only through sequential PF operations. Count extraction by
the configured `response_poisson` spectrum front-end is retained; the method
is therefore **count-domain PF-only localization without batch refinement**.

The continuous three-dimensional workspace, detector-height candidates,
collision checks, retract/translate/extend routes, shared spherical-octant
shield geometry, obstacle attenuation, and Geant4 transport are outside the
purity refactor and remain unchanged.

## Mixed estimator inventory

The table uses these abbreviations: `H` reads the accumulated measurement
history, `P` mutates/protects/prunes particles, `L` affects planner beliefs,
`D` affects adaptive dwell, `S` affects mission stopping, and `R` affects the
reported estimate.

| Facility | Definition and direct callers | H | P | L | D | S | R |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Sparse Poisson evidence | `pf.sparse_evidence.fit_sparse_poisson_evidence`; called by `RotatingShieldPFEstimator._update_sparse_poisson_evidence_for_isotope`, then `refresh_sparse_poisson_evidence` from `update_pair`, `update_pair_sequence`, and `finalize_deferred_pose_update` | yes | via authoritative sync/prune | via diagnostics and rescue | yes | yes | yes |
| Spectral sparse evidence | `pf.sparse_evidence.fit_sparse_poisson_spectral_evidence`; called by `_spectral_sparse_poisson_evidence_payload` during sparse refresh | yes, raw spectrum bins | indirectly | indirectly | indirectly | indirectly | yes |
| Joint sparse evidence | `pf.sparse_evidence.fit_joint_sparse_poisson_evidence`; called by `_refresh_joint_sparse_poisson_evidence` during sparse refresh | yes, all isotopes | indirectly | indirectly | indirectly | indirectly | yes |
| Sparse off-grid refinement | `pf.sparse_evidence.refine_sparse_poisson_evidence_offgrid`; called by count, spectral, and joint `_sparse_offgrid_refinement_payload` helpers | yes | through selected evidence sources | through rescued candidates | no direct call | through readiness | yes |
| Authoritative sparse cardinality/source sync | `_authoritative_sparse_evidence_cardinality`, `_authoritative_sparse_evidence_sources`, `_sync_sparse_evidence_cardinality_protection`, and `IsotopeParticleFilter.sync_particles_to_evidence_sources`; called by `_run_isotope_structural_update` | yes | yes; replaces source slots and protects strata | yes, because later PF modes change | indirectly | indirectly | yes |
| Sparse-driven verification pruning | `_prune_candidate_verification_queue_with_sparse_evidence`; called while merging sparse diagnostics | yes | can change later injected hypotheses | yes | no | indirectly | yes |
| Report MLE rescue | `_augment_report_candidates_with_mle_rescue`, `_rank_residual_surface_candidates`, `_rank_global_surface_candidates`; called from report construction and runtime rescue | yes | through runtime injection | yes | indirectly | indirectly | yes |
| Runtime report rescue | `_runtime_report_rescue_estimate`, `_merge_runtime_report_rescue_memory`, `_inject_runtime_report_rescue`; called after each isotope structural update | yes | yes, via `inject_runtime_report_rescue_particles` | yes | indirectly | indirectly | yes |
| All-history dictionary proposal | `_all_history_dictionary_candidates`; called by `_runtime_report_rescue_estimate` when enabled | yes | through rescue injection | yes | no direct call | no direct call | yes |
| Global birth rescue | `_runtime_global_birth_rescue_candidates`, `_inject_runtime_global_birth_quarantine`; called by `_run_isotope_structural_update` | yes | yes | yes through resulting PF/rescue modes | indirectly | indirectly | yes |
| Surface-map reconstruction | `pf.surface_map.fit_surface_map_poisson`; wrapped by `RotatingShieldPFEstimator.fit_surface_map`; called by `realtime_demo._fit_final_surface_map` | yes, count or spectrum history | no direct mutation | hotspots can enter legacy reporting/planning paths | no | no | yes, final surface result |
| Report BIC/model order | `_select_report_clusters_by_model_order`, `_select_report_clusters_by_refit_after_remove`, `_report_cluster_log_likelihood`; called by `_refit_reported_strengths`/report construction | yes | optional `_apply_report_model_order_particle_prune` | yes through selected modes/readiness | yes | yes | yes |
| Report strength refit | `_solve_report_strengths`, `_solve_report_strengths_batch`, `_refit_reported_strengths`; called by `estimates`, model-order trials, pre-finalize guard, and rescue | yes | no direct mutation | planner sees refitted strengths | indirectly | indirectly | yes |
| Surface local refinement | `_report_surface_local_candidates`, `_refine_report_surface_positions`; called during report refinement | yes | optional downstream prune | planner sees refined position | no direct call | indirectly | yes |
| Batch-driven particle pruning | `_apply_sparse_poisson_particle_prune`, `_apply_report_model_order_particle_prune`, `IsotopeParticleFilter.apply_report_model_order_cluster_prune`; called from evidence/report paths | yes | yes | yes through changed posterior | indirectly | indirectly | yes |
| Rescue modes in DSS-PP | `planning.dss_pp.extract_signature_modes` calls `runtime_report_rescue_modes` and `planning_surface_rescue_modes`; `select_dss_pp_next_station` enables both from `DSSPPConfig` | yes upstream | no direct planner mutation | yes | no | no | no |
| Sparse cardinality pressure in DSS-PP | `_cardinality_evidence_gap_pressure` calls sparse/report-model-order diagnostics and feeds `_static_station_program_score`/node scoring | yes | no | yes | no | no | no |
| Report evidence in adaptive dwell | `realtime_demo._source_cardinality_measurement_ready` and `_source_cardinality_readiness_reason` call report model-order/sparse-gap helpers from `mission_control` | yes | no | no | yes | no | no |
| Report evidence in mission stop | `mission_control.report_model_order_ready_for_stop`, `report_model_order_simple_ready_for_stop`, and `sparse_cardinality_evidence_gap_unresolved`; called by the runtime mission-stop block | yes | no | no | no | yes | no |
| Count/SNR absent-isotope final filter | `realtime_demo._filter_absent_final_estimates`; called by the legacy final-summary path | yes | no | no | no | no | yes |
| Mixed final report | `RotatingShieldPFEstimator.estimates`, `final_report_estimate`, `pruned_estimates`, and `realtime_demo._fit_final_surface_map` | yes | optional | no | no | no | yes |

## Configuration keys in the mixed baseline

The following keys enabled or parameterized batch-derived behavior and must
not be authoritative in `pf_strict` or `pf_profiled`:

- `sparse_poisson_evidence_enable`, `sparse_poisson_evidence_authoritative`,
  all `sparse_poisson_evidence_*`, `sparse_poisson_spectral_*`,
  `sparse_poisson_joint_*`, and `sparse_poisson_offgrid_*` keys;
- `report_mle_rescue_*`, `runtime_report_rescue_*`,
  `all_history_dictionary_proposal_*`, and `birth_global_rescue_*` keys;
- `report_cluster_model_selection`, all `report_model_order_*`,
  `report_strength_refit*`, and `report_surface_local_refine*` keys;
- `surface_map_reconstruction_enable` and all `surface_map_*` solver keys;
- `final_absent_isotope_filter` and its count/SNR/strength thresholds;
- `adaptive_cardinality_dwell_enable` and its BIC/model-order thresholds;
- `adaptive_mission_stop`, `mission_stop_require_model_order_ready`, and
  `mission_stop_report_simple_*`.

PF-internal residual birth/death/split/merge, particle-state strength updates,
robust likelihoods, covariance, tempering, resampling, and roughening are not
batch estimators. Their configuration remains part of the PF family.

## Required active-path disposition

`pf_strict` is the repository default. All batch capabilities above resolve to
false even when a legacy Boolean requests them. `pf_profiled` differs only by
allowing causal conditional strength profiling on the current sequential PF
state. The active estimator does not import sparse-evidence or surface-map
modules. Historical implementations remain available at the recorded Git
commit, not as an implicit active runtime path.
