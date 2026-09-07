# scene02 Paper Integration (2026-08-26)

Source result directory:
`results/paper_scene02_mc_mse_20260826/`.

Formal setting: 20 paired Monte Carlo realizations, 100 frames, fixed process
noise, CV covariance propagation, 3 m range-noise standard deviation, and
0.01-degree azimuth-noise standard deviation. The nominal tracker setting is
`Pd=0.99`, clutter mean `2`.

The ablation, robustness, and historical-comparison tables, the scene02 result
subsection, and the evidence-qualified conclusion are all inlined in
`GNN_IEEE_REVISED_SINGLE.tex`. There are no external TeX table or section
inputs.

All eight plot assets were copied into `figures/` with a `scene02_` prefix.
The main text currently includes the ablation OSPA/position curves and the two
robustness OSPA sweeps. Velocity, robustness-position, and protocol-qualified
historical-comparison plots remain available in the same directory for a
supplement or later page-budget revision.

The historical MCST--BP, MTP--GO, and BP--R--RGTT rows are descriptive only:
they retain the original-paper observation protocol and fewer Monte Carlo
realizations, and therefore are not a paired or protocol-matched comparison.

Build products:

- sole manuscript source: `GNN_IEEE_REVISED_SINGLE.tex`;
- compiled preview: `GNN_IEEE_REVISED_SINGLE.pdf`;
- static audit: `../PAPER_MANUSCRIPT_AUDIT_20260826.md`.
