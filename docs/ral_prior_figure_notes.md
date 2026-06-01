# RA-L Prior Figure Notes

This note records the prior-figure scan used when redesigning RA-L figures.

## Papers Checked

- Kemp et al., "Real-time radiological source term estimation for multiple
  sources in cluttered environments," IEEE TNS, 2023.
- Vavrek et al., "Reconstructing the position and intensity of multiple
  gamma-ray point sources with a sparse parametric algorithm," IEEE TNS, 2020.
- Bandstra et al., "Improved gamma-ray point source quantification in three
  dimensions by modeling attenuation in the scene," IEEE TNS, 2021.
- Anderson et al., "Mobile robotic radiation surveying with recursive Bayesian
  estimation and attenuation modeling," IEEE T-ASE, 2022, from the citation
  record and related recursive Bayesian surveying context.

## Design Lessons

- Gmail comments from the May 2026 RA-L review thread emphasize that Fig. 2 was
  hard to understand, that maps must look like maps with obstacles and floor
  context, and that method modules should be tied to ablation results rather
  than shown as generic workflow boxes.
- Prior PF papers do not rely only on text-box workflow figures. They pair
  algorithm summaries with ray-tracing diagrams, measurement paths, PF stage
  snapshots, convergence plots, and result tables.
- Scene-attenuation papers show geometry explicitly: point clouds, voxelized
  obstacles, detector trajectories, source truth, estimates, and count traces.
- Sparse reconstruction papers show why a model is identifiable by plotting
  detector paths, source estimates, forward-predicted count traces, and
  likelihood/model-order diagnostics.
- Earlier thesis/conference result figures were strongest when the environment,
  measurement points, true sources, and reconstructed source field were shown
  explicitly rather than reduced to a metric table. Their weakness for this
  RA-L paper is that 2-D maps alone hide ceiling/high-wall errors, so the RA-L
  result figure must add a 3-D view and a height-preserving projection.

## Decision for Fig. 2

Fig. 2 should not be a text-only method flowchart. It should visualize the
mechanism that is unique to this manuscript:

1. A rendered 3-D station view with surface-constrained hypotheses and
   source-detector paths through obstacles.
2. A rendered Fe/Pb posture sequence around the CeBr3 detector.
3. A station-level temporal response matrix for same-isotope source hypotheses
   and a PF/replanning diagnostic.

Short labels are acceptable, but the figure must be carried by geometry,
response signatures, particles, spectra, or diagnostics.

## Decision for Fig. 3

Fig. 3 should be a proposed-method PF result figure rendered from the actual
run summaries and traces. It should not be a compact dashboard of bar charts.
Each task panel group must show the 3-D room, obstacle geometry, stations,
ground truth, final estimates, and match segments, with paired floor and
depth-height projections to make metric and height errors visible. Ablation
bars and isotope-wise aggregate errors should stay in tables unless a separate
figure is explicitly allocated for them.
