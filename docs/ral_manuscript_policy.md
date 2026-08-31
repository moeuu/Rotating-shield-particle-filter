# RA-L Manuscript Policy

These rules apply whenever editing the RA-L manuscript in
`/home/moeu/research/ai-latex-workspace/projects/ieee-ra-l-letter`.

## Anonymous Funding Acknowledgment

- Keep the anonymous sponsor statement exactly as
  `This work was in part supported by XXX.` for review submissions.
- Place sponsor support in the first-page unnumbered author footnote using
  `\thanks`, not as a numbered section in the main text.
- Keep the real grant name masked during double-anonymous review. Replace
  `XXX` with the full support statement only for a non-anonymous or
  camera-ready version when explicitly requested.
- Do not delete `\IEEEoverridecommandlockouts`, because the `ieeeconf` class
  needs it for `\thanks`.

## Content Balance

- Treat the proposal as one coupled inference-and-design method, not as a path
  planner with a generic PF underneath. The central novelty is that selected
  Fe/Pb pose pairs form a posterior-adaptive attenuation code and the same code
  conditions full-spectrum particle weights and exact transdimensional moves.
- Allocate most method space to: the shield-conditioned joint multi-isotope
  likelihood; full-station tempered SMC; shield-aware birth, coupled
  position/strength, death/merge RJ moves; conditional-greedy selection of
  eight poses from all 64 Fe/Pb pairs; and one-stage exact full-history MH/RJ
  evaluation through the source-resolved CUDA slot overlay.
- State the boundary precisely: isotope line catalogs are supplied, while the
  detector response operator is isotope-independent. Do not claim isotope-blind
  identification or universal application validation.
- Present the fixed-capacity CUDA cache and slot overlay as implementation
  engineering, not as an independent methodological contribution. Every
  proposal uses one exact full-history target difference and one MH uniform.
- Move implementation-only details, simulator details, and long result
  interpretation out of the method section.
- Keep the experimental section compact. It should define the evaluation setup,
  the ablation meaning, and the evidence needed to support the claims, without
  repeating limitations already handled in the discussion.
- Never fill missing ablation results with placeholders. A prior completed run
  may be reported only as an explicitly provisional predecessor-code diagnostic.
- Use the eighth page effectively while staying within the eight-page RA-L
  limit. Do not leave large blank space if essential explanation has been
  removed.

## Planned Page and Figure Budget

- Page 1: abstract, motivation, three contribution statements, and the compact
  problem/shield-coding figure.
- Pages 2--4: related boundary, model, and the coupled inference/design method;
  the method figure may occupy about one quarter page across both columns.
- Page 5: experiment and evaluation contract, including the paired Cs4/Co3
  comparison and truth-matching criteria.
- Pages 6--7: main paired-projection result figure, four-variant design table,
  provisional diagnostic or final paired results, and discussion.
- Page 8: limitations, conclusion, and references. Use the full page without
  compressing labels below the figure-quality policy.

The three main figures have distinct roles: Fig. 1 explains the physical
attenuation code, Fig. 2 explains how SMC/RJ decodes and the planner redesigns
that code, and Fig. 3 makes 3-D localization and strength errors auditable from
paired metric projections and compact diagnostics.
