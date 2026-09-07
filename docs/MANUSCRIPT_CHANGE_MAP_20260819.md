# Manuscript Change Map — 2026-08-19

Source manuscript:
`E:/PycharmProjects/Neural Network/SGGT-Net_paper/GNN - IEEE.tex`  
Source SHA-256:
`e20862b005de7841e5a7d2455575ecd380b6d449f444d4a9d527ee291ad0d8f2`  
Source size/length: 77,042 bytes / 930 lines

The actual source filename contains spaces (`GNN - IEEE.tex`). All future
editing and compilation commands must use that exact filename unless a copied,
explicitly named revision file is created.

## Overall decision

The manuscript requires a structural rewrite, not local patching. The current
abstract, contributions, method core, loss, ablation, robustness tables, and
conclusion describe a different model from the frozen P4 checkpoint. Old
numerical tables cannot be retained as evidence for the revised architecture.

## Section-by-section disposition

| Current lines/section | Main incompatibility | Disposition in revision |
|---|---|---|
| 33 title | Foregrounds spectral-graph module combination | Replace with a title foregrounding label-free causal group innovation; retain BP tracker context without claiming BP novelty |
| 45--50 abstract | Claims SEAN, graph GASF and Neural ODE combination; no exact evidence boundary | Rewrite last, after statistics/multi-seed; state problem, mechanism, protocol and measured scope |
| 56--82 introduction | Vague combined limitations and module-list novelty | Rewrite related work baseline-by-baseline; motivate strict causal innovation under association uncertainty; give falsifiable contributions tied to P2/P3/P4/P5 |
| 84--151 problem formulation | Reuses state notation and defines variables after use | Rewrite symbols; define lifecycle/real mask, causal information set, target/measurement sets, clutter intensity and association variables before equations |
| 152--233 BP association | Does not clearly distinguish adopted BP; factorization omits one association family; covariance update is over-simplified | Keep a shorter adopted-backend section; correct joint variables and clutter intensity; derive soft posterior moments including missed detection/spread; identify only interface changes |
| 234--301 encoder/decoder | Workflow and online/offline roles unclear | Retain concise Temporal backbone description and streaming eight-frame encoder; add intuitive full closed-loop workflow and online/offline table |
| 302--337 SEAN | Uses group-label adjacency supervision; module disabled in formal P4 | Remove from main method. Mention only as an abandoned/rejected label-dependent design if needed in discussion |
| 338--433 GASF | Describes graph propagation and learned polynomial-bin gates inconsistent with frozen temporal-only `a,b,c` filter | Replace with the exact Temporal GASF used by P2/P4; present it as frozen backbone, not principal novelty |
| 447--479 ODE mean | Describes SEAN/GASF gated acceleration fusion inconsistent with concat formal checkpoint | Rewrite to match actual Temporal decoder. Neural ODE mean prediction is adopted backbone functionality, not the main contribution |
| 480--546 covariance/Q | Claims SEAN-controlled Q and gives no matching loss; conflates neural mean and CV covariance | Separate CV covariance approximation from learned mean; document retained legacy adaptive-Q path without novelty claim; list diagnostics/limitations |
| 547--622 loss | Uses group-label focal structure loss; formal P4 uses no structure loss and trains only corrector | Replace entirely with causal-corrector/state-improvement objective, masks, physical scaling, NLL weight, optimizer and frozen-backbone policy |
| 623--693 experiment setup | Old dataset/training LR/optimizer/counts and vague convergence | Replace with V9 1000/200/300/400, seed status, AdamW/cosine/warmup, batch 4, all-time supervision, checkpoint criterion and exact evaluation counts |
| 694--723 accuracy | Uses old baselines and unverifiable “numerous MC” | Replace with P0--P5 clean/IID/OOD/association results and paired trajectory intervals |
| 724--732 group number | Depends on disabled SEAN/group topology truth | Remove from main paper |
| 729--771 spectrum analysis | Supports an obsolete graph-gated GASF claim | Remove or reduce to Temporal-backbone diagnostic; do not use as evidence for CGIC novelty |
| 772--815 robustness | Old one-factor tables may use matched/oracle parameters and old model | Replace with full 3×3 mismatched nominal-vs-actual BP matrix; decompose OSPA and state metrics |
| 816--876 ablation | “w/o GASF” and “w/o SEAN” descriptions are swapped; artifacts are obsolete | Delete and replace with immutable P2/P3/P4/P5 mechanism table |
| 877--907 complexity | Unsupported qualitative ranking and unrelated old timing table | Replace with exact P2/P4 parameter counts, timed interface, exclusions and controlled 5/10/20/40 timing-only scaling; add buffer asymptotics |
| 908--916 conclusion | Repeats unsupported SEAN/GASF/ODE combined novelty and broad robustness | Rewrite only after final statistics; include simulation, single-backend and calibration limitations |

## New proposed manuscript structure

1. Introduction and related work.
2. Problem formulation and causal information set.
3. Adopted BP association and probabilistic state update.
4. Temporal motion-prior backbone.
5. Label-free causal group-innovation correction.
6. Association-uncertainty interface.
7. Offline training and online inference.
8. Experiments.
   - data/protocol and reproducibility;
   - deterministic and learned baselines;
   - clean IID/OOD and association-shift results;
   - own/peer/context ablations;
   - event-stratified mechanism analysis;
   - full BP Pd×clutter robustness;
   - calibration and complexity;
   - limitations.
9. Conclusion.

## Equations that require replacement

- Current Eqs. (1)--(4): disambiguate stacked versus per-target states.
- Current association factorization: include both association-variable sets or
  eliminate the unused set consistently.
- Current clutter likelihood: expose spatial clutter intensity/rate.
- Current state/covariance update: use soft marginal moment matching, including
  missed detection and between-candidate covariance.
- Current hard-selected innovation: replace with strict prior soft-association
  innovation and validity.
- Current SEAN adjacency/focal loss equations: remove.
- Current graph-GASF equations: replace with frozen temporal-only filter.
- Current gated ODE fusion: replace with actual concat backbone and additive
  CGIC state correction.
- Current Q/SEAN scale equation: audit against `adaptive_q_mode=legacy` and
  document only what is actually used.
- Current loss: replace with the actual corrector training objective and masks.

## Figures/tables to retire

- Existing encoder-decoder diagram unless redrawn for the frozen model.
- SEAN group-number estimation figure.
- Obsolete graph-GASF diagram and spectrum plots as main-contribution evidence.
- Old one-dimensional Pd and clutter tables.
- Old GASF/SEAN ablation curves and table.
- Old cross-method runtime table unless every method is rerun under a verified,
  common timing interface.

## New figures/tables required

1. BP → Temporal prior → soft update → strict innovation buffer → CGIC → next
   prior workflow.
2. Own versus leave-one-out peer innovation construction.
3. Adopted-versus-proposed component table.
4. P0--P5 switch/flag table.
5. Main val/IID/OOD/association result table.
6. P2/P3/P4/P5 mechanism and paired-CI table.
7. Pd×clutter P2/P4 position and OSPA gain heatmaps.
8. Event-stratified steady/switch/strong result plot.
9. NIS/NLL/coverage table with explicit negative/null findings.
10. Parameters, storage complexity and measured scaling table.

## Claims allowed before multi-seed completion

- Seed-123 P4 improves position prediction/update over P2 on val, IID, OOD and
  fixed-association validation.
- Seed-123 P4 substantially outperforms P3, supporting aligned peer history.
- P4 and P5 are close, so association context has only a small current effect.
- In the full BP matrix P4 improves position, velocity and OSPA in 8/9 cells,
  with a small regression at Pd 0.75/clutter 30.
- P4 does not improve BP NIS coverage.

These are single-training-seed statements. They must not be generalized to
random-initialization robustness until P2/P4 multi-seed experiments complete.

