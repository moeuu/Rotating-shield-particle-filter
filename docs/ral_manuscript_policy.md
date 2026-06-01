# RA-L Manuscript Policy

These rules apply whenever editing the RA-L manuscript in
`/home/moeu/research/latex/projects/ieee-ra-l-letter`.

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

- The proposed method section should explain only the paper's novel mechanism:
  active shield-time coding, surface/obstacle-aware PF likelihoods,
  residual/global source birth with verification, DSS-PP station selection, and
  remaining-view guidance.
- Move implementation-only details, simulator details, and long result
  interpretation out of the method section.
- Keep the experimental section compact. It should define the evaluation setup,
  the ablation meaning, and the evidence needed to support the claims, without
  repeating limitations already handled in the discussion.
- Use the eighth page effectively while staying within the eight-page RA-L
  limit. Do not leave large blank space if essential explanation has been
  removed.

