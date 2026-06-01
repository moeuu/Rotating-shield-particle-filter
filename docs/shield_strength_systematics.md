# Shield Strength Systematics

This note documents the reversible implementation that separates shield-coded
localization from shield-sensitive strength estimation.

## Implemented Behavior

1. Shield-pose-conditioned response calibration is supported through
   `NetResponseCalibration.scale_by_isotope_and_pair`. Calibration records with
   `shield_pair_id` now fit per-isotope, per-pair response scales while keeping
   the original isotope-wide scale as a fallback.
2. Runtime `response_poisson` count ingestion adds an optional
   shield-systematic variance floor. The floor is applied after the spectrum
   response regression and existing diagnostic variance floors, so the same
   Geant4 spectrum and response-Poisson count estimate are still used.
3. Weakly shielded anchor pairs can be configured with
   `response_poisson_shield_systematic_anchor_pair_ids`. Anchor pairs use
   `response_poisson_shield_systematic_anchor_rel_sigma`; other shielded pairs
   use `response_poisson_shield_systematic_rel_sigma`.
4. DSS-PP already evaluates temporal separation with normalized and whitened
   program signatures. The new variance floor makes the downstream strength
   fit less sensitive to shield-pose spectral bias without removing those
   normalized temporal signatures from localization and source-count selection.
5. Final report strength refit continues to use the full-spectrum
   `response_poisson` count-ingestion standard. Shield-dependent mismatch is
   propagated through measurement variances instead of replacing Geant4
   transport or switching to a lower-fidelity count extractor.

## Runtime Metadata

Geant4 and the Python diagnostic transport now attach these fields to each
observation metadata payload:

- `fe_orientation_index`
- `pb_orientation_index`
- `shield_num_orientations`
- `shield_pair_id`
- `shield_thickness_scale`
- `shield_thickness_fe_cm`
- `shield_thickness_pb_cm`

The no-shield baselines set `shield_thickness_scale = 0`, so the
shield-systematic variance floor is skipped rather than used to make the
baseline artificially worse.

## Rollback

To return to the previous behavior without deleting code, set:

```json
"response_poisson_shield_systematic_variance_enable": false
```

The pair-conditioned calibration support is passive unless a calibration file
contains `scale_by_isotope_and_pair` or calibration records include
`shield_pair_id`. Existing isotope-only calibration files retain their old
behavior.

If the user asks to revert to the state before "using the shield as a temporal
code for localization while modeling shield-derived strength systematics",
disable the variance flag above and ignore or remove
`scale_by_isotope_and_pair` from calibration payloads.
