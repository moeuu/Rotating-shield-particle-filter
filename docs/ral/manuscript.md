# RA-L Manuscript Policy

These rules apply whenever editing the RA-L manuscript in
`/home/moeu/research/ai-latex-workspace/projects/ieee-ra-l-letter`. The actual
manuscript is authoritative for its current wording; this document records
stable project constraints and claim boundaries.

For manuscript work, recheck current RA-L/IEEE submission rules against the
official source for the exact submission stage. Do not rely on a venue snapshot
embedded in repository notes.

## Central claim

The manuscript presents posterior-adaptive attenuation coding for online
multi-isotope source-term estimation. The coupled loop is the method:

1. the joint posterior predicts how reachable detector poses and all 64 Fe/Pb
   orientation pairs distinguish competing source configurations;
2. the planner selects an eight-view physical attenuation code and reachable
   detector pose; and
3. the realized shield identities condition the full-spectrum SMC weighting
   and exact transdimensional rejuvenation that produce the next posterior.

Describe this as a coupled inference-and-design method, not as a shield-aware
path planner wrapped around a generic particle filter.

## Method content

Allocate most method space to:

- the source-resolved shield-conditioned joint full-spectrum likelihood;
- complete-station tempered SMC with aligned multi-isotope ancestry;
- continuous-surface birth, death, position/strength, split, and merge moves;
- complete forward/reverse selector, density, area, and Jacobian terms in the
  exact MH/RJ ratio;
- conditional-greedy selection of eight views from all 64 Fe/Pb pairs plus the
  one-swap refinement; and
- posterior-adaptive pose selection using the same generative model.

State the model boundary precisely: isotope line identities and branching
weights are supplied, while the detector response operator is
isotope-independent. Do not claim isotope-blind identification or universal
application validation.

Present fixed-capacity CUDA caches, slot overlays, batching, strict schemas,
provenance binding, and fail-closed lifecycle checks as reproducibility and
implementation engineering, not separate methodological contributions.

## Novelty and claim boundaries

The defensible contribution is the exact coupling of a posterior-adaptive
eight-of-64 Fe/Pb code with shield-conditioned joint transdimensional
full-spectrum SMC. Shield hardware, particle filters, reversible-jump MCMC,
SMC tempering, expected information gain, and greedy subset selection all have
prior art and require citations.

Do not claim the first directional detector, first active radiation search,
first use of shielding, first RJ particle filter, global optimality of the
conditional-greedy program, or universal radionuclide validation. Separate
implemented facts, measured results, inferences, and limitations.

## Evidence discipline

- Use only completed, artifact-verified runs governed by the
  [experiment protocol](experiment_protocol.md) for the main comparison.
- Do not fill missing results with placeholders or silently mix batch IDs.
- A predecessor-code result must be labelled diagnostic and cannot support a
  current-method or comparative claim.
- A completed proposed-only run is a case study, not the four-variant headline
  comparison. Keep its audit figure out of the live paper until the comparison
  is complete or the manuscript explicitly reframes the claim.
- Seven sources in one paired scene are descriptive observations, not seven
  independent experimental replicates.
- Keep detailed implementation traces and run-specific failure analysis out of
  the manuscript unless they directly support a stated claim.
- Ensure every quantitative claim is traceable to a durable artifact and a
  declared evaluation rule.

## Anonymous funding acknowledgment

- Keep the anonymous sponsor statement exactly as
  `This work was in part supported by XXX.` for review submissions.
- Place it in the first-page unnumbered author footnote using `\thanks`, not as
  a numbered section in the main text.
- Keep the real grant name masked until the user explicitly requests a
  non-anonymous or camera-ready version.
- Retain `\IEEEoverridecommandlockouts`, which the `ieeeconf` class needs for
  the footnote.

## Page and figure budget

- Page 1: abstract, motivation, three contribution statements, and the compact
  problem/attenuation-code figure.
- Pages 2--4: related boundary, model, and coupled inference/design method.
- Page 5: experiment and evaluation contract.
- Pages 6--7: four-variant comparison and discussion.
- Page 8: limitations, conclusion, and references.

Use the eighth page effectively while staying within the eight-page limit. Do
not create artificial white space by removing necessary explanation, and do
not compress figure or table text below the readable limits in the
[figure policy](figures.md).

The current manuscript has two live figures with distinct roles:

1. the robotic problem and physical Fe/Pb attenuation code; and
2. the coupled inference and planning mechanism.

Do not use the same primary scene rendering for both roles. The completed-run
3-D case audit is a review/supplementary artifact for now. After all four
variants are complete, decide from the quantitative evidence whether a result
figure should replace or restructure one live figure, or whether the result is
clearer as a table. Do not add a third live figure by default.

If a result figure becomes live, follow the figure policy's evidence grammar:
actual obstacle geometry in a contextual 3-D view, companion orthogonal
projections or explicit 3-D error metrics, saved routes only, redundant marker
encoding, and a comparison-first presentation shared by all variants.

## Figure and table selection

Choose a figure when spatial geometry, occlusion, uncertainty, temporal
cardinality, or a multi-variant relationship is materially easier to understand
visually. Choose a table for a small number of exact metrics. Do not place the
same numbers in both without a distinct analytical purpose.

Before promoting a generated figure:

1. verify the complete comparison and its batch identity;
2. verify every plotted value against the stored evaluation artifacts;
3. inspect the standalone render at final paper size;
4. compile and inspect the actual manuscript page; and
5. confirm that the caption distinguishes measured results, display
   transformations, and interpretation.

All scientific imagery must be constructed from authenticated simulation,
measurement, PF, or evaluation data. AI-generated or AI-edited imagery is not
permitted in the manuscript.
