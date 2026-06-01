# Online PF Improvement Notes

These notes capture follow-up ideas for improving intermediate online
estimates without changing the current RA-L ablation runs.

The primary evaluation target remains the final reported source-term estimate.
Any online-PF change must be accepted only if it does not degrade final report
accuracy, model-order selection, or calibrated source-strength refit quality.

## Candidate Improvements

1. Weakly inject station-level BIC/rescue candidates into the PF.

   After each station window, use the BIC/report-rescue candidate set as a
   low-weight proposal source for the next PF state. Inject these candidates as
   tentative or quarantined modes, not as immediately verified sources.

2. Use BIC-selected candidates as a planner input.

   The planner should not depend only on the current MAP particle when the MAP
   is over-split or collapsed. Candidate stations and shield programs should
   also be scored against the station-level BIC/rescue source set.

3. Preserve low-weight but spatially distinct modes.

   During resampling, protect a small quota of independent spatial modes so
   that plausible sources are not lost only because their current posterior
   weight is temporarily low.

4. Add cardinality-conditioned resampling.

   When the weighted particle source-count distribution drifts toward 4 or 5
   sources while station-level BIC repeatedly supports fewer sources, keep a
   protected quota of particles at the BIC-supported cardinalities. This should
   prevent online particle populations from diverging from the all-history
   model-order evidence.

## Acceptance Criteria

- Final source count error must not worsen.
- Final matched localization error must not worsen on the RA-L cases.
- Final strength refit error must not worsen materially.
- Online improvements should reduce time spent in collapsed or over-split MAP
  states, but not at the cost of final report accuracy.
- Changes should be tested first on the current
  `mix9_multi_isotope_cardinality` proposed run. If the proposed measurement
  log is available, estimator-only changes should also be checked through the
  logged replay ablations before they are used for RA-L tables.
