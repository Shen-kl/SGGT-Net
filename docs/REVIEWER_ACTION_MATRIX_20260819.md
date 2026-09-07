# Reviewer Action Matrix — TAES-2026-1968

Source review: `E:/PycharmProjects/Neural Network/SGGT-Net_paper/reviewer.md`  
Source SHA-256: `73dff0adf162be22db2b1a899aeb60834418ee766ff787c6a4457b4f8f77cd73`  
Status date: 2026-08-19

This is the authoritative evidence/action ledger. “Complete” means that the
implementation or result artifact exists; it does not mean that the revised
manuscript paragraph has already been written. The reference response letter
was read for tone and structure only. None of its unrelated methods, evidence,
or numerical claims will be reused.

## Response style learned from the reference letter

Each final response will use this order:

1. acknowledge the technical concern without generic praise;
2. identify the exact methodological or textual change;
3. report only verified quantitative evidence;
4. cite the revised section, equation, figure, table, and artifact;
5. state remaining limitations explicitly.

Page numbers and yellow-highlight claims will be inserted only after final PDF
compilation. The response letter will not promise experiments that are merely
planned.

## Senior and Associate Editors

| Concern | Required manuscript action | Evidence/artifact | State |
|---|---|---|---|
| Novelty appears incremental | Replace the SEAN/GASF/ODE module list with the new problem of label-free, strictly causal group-innovation correction under uncertain association | P2/P3/P4/P5 counterfactuals; `PAPER_MODEL_FREEZE_20260819.md` | mechanism and seed-123 formal evidence complete; writing pending |
| Problem definition lacks rigor | Define indexed and stacked states, measurements, lifecycle mask, causal information set, association variables, innovations, and valid-history mask before use | code/protocol invariants and mask tests | evidence complete; notation rewrite pending |
| Simulation-only evidence | Add IID/OOD motion shifts, mismatched Pd/clutter BP Monte Carlo, uncertainty intervals, and a frank simulation-only limitation | 200/300/400 val/IID/OOD; 18 BP cells, 1080 rollouts | single-seed/trajectory evidence complete; training-seed replication pending |
| Strong baselines missing | Add validation-tuned CV, two-mode IMM, Temporal, own-only, full, and no-context models under consistent masks and updates | P0--P5 artifacts | seed-123 and deterministic baselines complete |
| Learning blocks unclear | Begin method with an intuitive online workflow and then give exact tensors, histories, masks, offline/online roles | frozen architecture and code interfaces | figure/text pending |
| Baseline fairness and group labels | Remove group labels and structure supervision from training/inference; disclose that state truth is used only for supervised loss and offline event scoring | `decoder_use_SEAN=False`, `lambda_struct=0` | complete |
| Oracle robustness parameters | Hold nominal BP Pd/clutter fixed while changing actual measurement generation | formal BP protocol and all nine cells | complete |
| Mathematical consistency | Separate learned state-mean propagation from CV covariance approximation and clean up state/transition notation | P2--P5 use `covariance_transition=cv` | rewrite pending |

## Reviewer 1

| Reviewer point | Response/change | Evidence | State |
|---|---|---|---|
| Only simulated scenarios | Do not claim real-radar deployment. Report clean IID/OOD, association-augmented validation, full BP missed-detection/clutter tests, and simulation-only limitation | formal seed-123 results and BP matrix | complete except final prose |
| Combination of known GNN/BP/spectral/ODE ideas | Claim neither the components nor their combination as the novelty. Formalize causal own/peer innovation correction and test information sources counterfactually | P4 vs P3 and P5 bootstrap | complete; statistical synthesis pending |
| Limited strong baselines | Add P0 CV, P1 IMM, P2 Temporal and counterfactual learned baselines | physical tuning plus P0--P5 results | complete |
| Group structure needs truth labels | Remove SEAN and focal structure loss from the final model and main evidence | no-SEAN checkpoints; zero structure weight | complete |
| Baseline comparison unfair | Use identical V9 splits, lifecycle masks, recursive feedback, and update rules; tune physical hyperparameters on validation only | frozen protocol and physical tuning JSON | complete |
| Ablation descriptions are confused | Replace old “w/o GASF/SEAN” text with immutable P0--P5 flag table | experiment freeze | complete; manuscript table pending |

## Reviewer 2

| No. | Reviewer point | Response/change | Evidence | State |
|---:|---|---|---|---|
| 1 | Workflow is difficult for tracking readers | Add an intuitive prediction → BP soft association → probabilistic update → strict innovation buffer → next prior diagram before equations | frozen online sequence | figure pending |
| 2 | BP originality/adaptation is unclear | Keep a concise BP background section, cite the adopted algorithm, and add an adopted-versus-proposed table. Only the soft-marginal-to-causal-history interface is proposed integration | BP protocol and adapter | writing pending |
| 3 | Variables and history length appear after use | Define 4-D per-target state, 2-D measurement, stacked sets, eight-frame history, 4/8 plus recent-1/3 rule, and constant buffer size before method equations | checkpoint metadata | complete; rewrite pending |
| 4 | CV matrix may mismatch CT/CA motion | Explain that the neural/ODE backbone predicts the mean while the main fast protocol approximates covariance with CV; include IMM baseline and label ODE Jacobian as diagnostic only | P1 and CV-covariance formal protocol | complete; discussion pending |
| 5 | Offline versus online adaptation unclear | State that weights are learned offline and frozen; states, covariances, graphs, BP marginals, innovation histories, eligibility, and legacy adaptive Q update online | implementation | complete; table pending |

## Reviewer 3

| Reviewer point | Required correction | Evidence/state |
|---|---|---|
| Discuss baselines [15]--[17] one at a time | Rewrite related work as method → capability → exact limitation, avoiding one shared vague limitation | writing pending |
| `x_k` denotes both one and many targets | Reserve `x_k^i` for one target and bold stacked `X_k` (or an explicitly distinct symbol) for the set | notation rewrite pending |
| `n_k^j` absent from left-hand side of factorization | Write the joint posterior over both target- and measurement-oriented association variables or marginalize one family consistently | factor-graph rewrite pending |
| `p_c` and clutter rate unclear | Define spatial clutter intensity as expected clutter count divided by surveillance volume/area and use it in the likelihood ratio | BP configuration available; equation rewrite pending |
| Covariance omits missed detection and candidate spread | Replace simple EKF covariance equation with soft-association Gaussian-mixture moment matching including missed detection and between-candidate spread | association uncertainty implementation/tests | complete; derivation pending |
| Neural mean versus CV `F` mismatch | State explicitly that CV is the covariance-transition approximation, not the learned mean model | formal CV protocol | complete; prose pending |
| SEAN uses ground-truth adjacency | Remove SEAN/group labels/structure loss from the final method | complete |
| Group labels make comparison unfair | Use label-free P2--P5 protocol and identical masks/splits | complete |
| Eq. (23) uses hard maximum association | Use soft BP marginal moments to update state and form strict prior innovation; report detection mass, entropy, spread | complete |
| Learned Q loss is missing | Do not claim Q learning as the new contribution. Document retained legacy adaptive-Q parameterization and the exact losses that train/freeze it | loss audit pending in result/method phase |
| “Numerous Monte Carlo realizations” is vague | Give exact split sizes, training seed, three BP trajectories/cell, 20 measurement MC runs/trajectory and all seeds | counts frozen; multi-training-seed incomplete |
| Robustness may use matched/oracle Pd/clutter | Main BP holds nominal 0.85/20 while actual values cross 3×3; matched-parameter runs excluded from main table | complete |
| BP-R-RGTT called low complexity despite ranking second slowest | Remove qualitative ranking claim and report measured values/interface/exclusions exactly | scaling reports complete; old table will be removed |

## Reviewer 4

| Reviewer point | Response/change | Evidence | State |
|---|---|---|---|
| No rigorously defined new problem or improved classic method | Define association-uncertainty-aware causal group-innovation correction, exact Temporal fallback, eligibility, and own/peer counterfactual | formal method and ablations | complete; writing pending |
| No general performance explanation | Add mechanism-aligned event/IID/OOD analysis and paired intervals; avoid universal claims | results and bootstrap | synthesis pending |
| Only about 20 targets; complexity/storage unclear | Give parameter/buffer asymptotics and timing-only 5/10/20/40 controlled scaling with explicit exclusions | scaling P2/P4 complete | analytical complexity prose pending |
| Figures/tables unreadable | Regenerate final plots as vector or high-resolution graphics at target column width with consistent fonts | pending |
| Necessity of polar conversion | Explain polar radar sensing versus Cartesian tracking/neural features; present conversion as sensor preprocessing, not a contribution | measurement model/code | pending |

## Evidence that must not be overstated

1. Current complete P3/P4/P5 training evidence is seed 123 only.
2. Trajectory-paired bootstrap does not substitute for independent training
   seeds.
3. P4 and P5 are nearly tied on several clean/IID/OOD metrics; association
   context is a secondary claim unless multi-seed evidence strengthens it.
4. P4 substantially outperforming P3 supports aligned peer information, but
   the result must be reported with paired intervals.
5. BP NIS 95% coverage is not improved by P4 and remains poorly calibrated.
6. Full BP cardinality is affected by the adopted birth/confirmation manager;
   total OSPA must be decomposed before attributing changes to the motion prior.

## Final response-letter completion rule

A row may be described as “addressed” in the response letter only when the
revised manuscript location and verified artifact are both available. Until
then, its state remains “pending” even if the code implementation is complete.

