# Paper comparison/ablation experiment preflight

Date: 2026-08-22 (Asia/Shanghai)

Status: **protocol confirmed; smoke and formal single-training-seed experiments completed**

## 1. User-specified evaluation source

The requested legacy evaluation entry is:

`E:/PycharmProjects/Neural Network/GNN_groupTargetTracking_v9_norm_without_struct/evaluation/evaluation.py`

SHA-256: `3b032caa71732d609a465b8a72e585dfc1d732f04df3a420fbfad0b2403d73b4`

It loads this truth trajectory:

`E:/PycharmProjects/Neural Network/GNN_groupTargetTracking_v9_norm_without_struct/generate_trajectory/scene02/trajectory01.pt`

- SHA-256: `fa7b5f5ef29c01c7d2c2918af2afa16bf09306b2ecb915601d2cb27a585634bc`
- Frames/targets/sample interval: 100 / 13 / 2 s
- `scene02` also contains `trajectory_measurement.mat` (SHA-256
  `67b0990a262c3346ffe0c7ff9690d01695abea354ba357a435ba8983f0b2bd54`),
  which stores one set of measurements for the same trajectory. It is not a
  second truth trajectory, and the supplied `evaluation.py` does not load it.

The current repository has another file with the same relative name, but its
SHA-256 is `1e971c3b7f523ba3a9449ada9b4a8554c2878ead2a2b2ff1d72b9fe782df219a`.
It is not interchangeable: targets 9--13 and many framewise truth states differ
materially (maximum state difference reaches 460 m). Existing paper-revision BP
reports used this current-repository copy and therefore cannot be reused for the
user-specified comparison/ablation scenario.

## 2. Frozen historical comparison assets

Machine-readable legacy results exist under
`SGGT-Net_paper/GNN_version3/data`; the unchanged comparison algorithms do not
need to be executed again.

For the nominal comparison plotted by the old notebook:

| Paper method | Frozen asset | MC runs | Raw RMS of the MC-mean OSPA curve (m) |
|---|---|---:|---:|
| MCST-BP | `trajectory1_MCST_lambda_3.pkl` | 5 | 22.5795 |
| MTP-GO | `trajectory1_proposed_al_without_struct_MCU_lambda_3.pkl` | 5 | 27.8958 |
| BP-R-RGTT | `trajectory1_BP_R_RGTT_lambda_3.pkl` | 10 | 25.3390 |

The old SGGT result (`trajectory1_proposed_al_lambda_3.pkl`) belongs to the
superseded model and must be replaced by the current P4 result.

The historical values reproduce the old paper tables, but their metadata is
not submission-grade:

- run counts differ (2, 5, or 10) and random seeds are absent;
- filenames `lambda_3`, `lambda_10`, and `lambda_15` were reported in the paper
  as clutter means 10, 20, and 30, but the stored pickle does not record the
  actual parameter;
- the current legacy code sets `Pd=0.99`, clutter mean 10, range noise 3 m, and
  azimuth noise 0.01 degrees, whereas the old paper states `Pd=0.95` and
  azimuth noise 0.1 degrees;
- at the lowest detection-probability row, the paper's MCST-BP and BP-R-RGTT
  numbers are swapped relative to the corresponding raw pickle files;
- the old notebook modifies one part of the MCST curve before plotting and
  manually edits several group-count samples. Neither manipulation will be
  used in the revised statistics.

Consequently, the legacy baselines may be reused descriptively from the raw
pickle files, but paired confidence intervals against P4 cannot be claimed.

## 3. Current checkpoints that must be evaluated on the specified scene

| ID | Meaning | Checkpoint SHA-256 |
|---|---|---|
| P2 | frozen Temporal predictor | `c2c0dfdde7ccef1a7d892459c1e4a1a5d5c96cbc539f771ab09e862f9d359022` |
| P3 | own-innovation correction only | `18c471b6b45648f4d0b63641fb9f5341c61deb030ccaa79fb29441f00cc27da4` |
| P4 | complete own + group-common correction | `6177ab5834c0388c74d8d75e5bdbfb90c56f41bc6ce4baf1ad07258cb0385581` |
| P5 | P4 without association-context input | `ca6498fb5e9c302fdd42bfed6696030aba49fa489b78252ba756532b3cbeb395` |

Old SEAN/GASF ablation pickle files describe a superseded architecture and
must not be presented as ablations of CGIC. The revised ablation is P2/P3/P4/P5.

## 4. Required implementation changes before launch

1. Add a paper-scene wrapper around
   `evaluation/run_bp_tracker_robustness_v3.py` that accepts the external
   trajectory explicitly and refuses any trajectory hash except the frozen
   manifest hash.
2. Freeze actual/nominal `Pd`, clutter mean, range/azimuth noise, OSPA
   parameters, Monte Carlo count, and seed list in a JSON manifest. Do not infer
   these values from legacy pickle filenames.
3. Run all current P2/P3/P4/P5 checkpoints with the same generated measurement
   seeds, keeping per-run and per-frame OSPA, cardinality, matched-state RMSE,
   NIS, runtime, checkpoint hash, and trajectory hash.
4. Add an aggregator that computes both (a) the legacy table statistic, RMS of
   the MC-mean framewise OSPA curve, and (b) mean/run-level uncertainty. It must
   import frozen baseline pickle values without curve smoothing or manual edits.
5. Store new files under a separate directory such as
   `results/paper_scene02_20260822`; do not overwrite the v9-pack IID/OOD/BP
   evidence, which remains useful for generalization and robustness sections.
6. First run one scene/one Monte Carlo smoke test for P2 and P4; only after
   finite outputs, identical measurement counts/seeds, and valid checkpoint
   hashes should the full current-model matrix start.

## 5. Proposed experiment matrix after confirmation

### A. Algorithm comparison on the designed scenario

- Frozen, no rerun: MCST-BP, MTP-GO, BP-R-RGTT raw pickle curves.
- Rerun: P4 complete CGIC on the exact user-specified trajectory.
- Primary metric: OSPA (`c=100`, `p=1`) using the same legacy aggregation.
- Secondary diagnostics: localization/cardinality OSPA components, matched
  position/velocity RMSE, cardinality MAE, and runtime.

### B. Current mechanism ablation on the designed scenario

- P2 Temporal, P3 own-only, P4 full, P5 no association context.
- Shared measurement seeds and paired per-run comparison.
- Report whole-trajectory and maneuver-event results; do not reuse old
  SEAN/GASF ablation numbers.

### C. Detection/clutter robustness

Run only after the historical condition labels are resolved. The current
3-by-3 v9-pack BP matrix cannot be silently substituted because it used a
different trajectory. If the legacy baselines remain frozen, robustness claims
must clearly distinguish historical descriptive values from newly paired
P2/P4 tests.

## 6. Resolved protocol decisions

The user resolved both launch blockers on 2026-08-23:

1. Use only the external legacy `scene02/trajectory01.pt`; do not evaluate
   `scene01` and do not substitute the current repository's same-name file.
2. When code and manuscript parameters conflict, the old paper is authoritative.
   The frozen protocol therefore uses `Pd=0.95`, clutter mean 10, range noise
   3 m, azimuth noise 0.1 degrees, and OSPA `(c=100 m, p=1)`.

The machine-readable source of truth is
`results/paper_scene02_protocol_v1.json`. It freezes trajectory, checkpoint and
historical-result hashes, a smoke count of one, a formal count of 20, and shared
measurement seeds 20260823--20260842.

## 7. Execution result (2026-08-23)

The P2/P4 one-run smoke test completed for all 100 frames with identical seeds
and framewise measurement counts and no NaN/Inf. The four formal models then
completed 20 paired Monte Carlo runs each. Strict report revalidation confirmed
20 unique seeds, 100 frames per run, finite values, and the frozen hashes.

| ID | Legacy OSPA curve RMS (m) | Global matched position RMSE (m) | Global matched velocity RMSE (m/s) |
|---|---:|---:|---:|
| P2 | 33.2196 | 29.3316 | 10.1285 |
| P3 | 33.2646 | 29.2262 | 10.0923 |
| P4 | **32.2229** | 27.7540 | **9.3297** |
| P5 | 32.2602 | **27.4746** | 9.5056 |

Paired 10,000-sample bootstrap on the run-level RMS metrics found:

- P4 over P2: OSPA gain 0.990 m (95% CI 0.465--1.498), position gain
  1.587 m (0.784--2.310), and velocity gain 0.802 m/s (0.489--1.085).
- P4 over P3: OSPA gain 1.049 m (0.438--1.613), position gain
  1.485 m (0.672--2.212), and velocity gain 0.765 m/s (0.397--1.079).
- P4 over P5: all three confidence intervals cross zero. P5 has lower global
  position RMSE, while P4 has slightly lower legacy OSPA RMS and velocity RMSE.
  The independent benefit of association-context input is therefore not
  supported on this scene.

The frozen historical paper values (MCST-BP 22.64 m, MTP-GO 27.89 m and
BP-R-RGTT 25.33 m) are numerically lower than the current P4 legacy OSPA RMS.
Because their original seeds and complete runtime protocol metadata are absent,
they remain descriptive and cannot support a paired or strictly controlled
superiority claim. This mismatch must be resolved or disclosed before revising
the paper's comparative-performance conclusion.

Auditable outputs:

- `results/paper_scene02_20260823/formal/P2.json` through `P5.json`
- `results/paper_scene02_20260823/summary/SUMMARY.json`
- `results/paper_scene02_20260823/summary/ablation_table.csv`
- `results/paper_scene02_20260823/summary/comparison_table.csv`
- `results/paper_scene02_20260823/formal_STATUS.json`
