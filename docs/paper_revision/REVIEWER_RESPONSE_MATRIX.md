# Reviewer Response Matrix

This is an evidence ledger, not the final response letter. Each response must
state exactly what changed and point to a manuscript section, figure/table, and
machine-readable result. The reference response letter is used only for tone
and organization; none of its unrelated claims or results are reused.

## Editorial concerns

| Concern | Manuscript action | Required evidence | Current state |
|---|---|---|---|
| Novelty appears incremental | Replace module-list positioning with the strictly causal, label-free group-innovation problem and counterfactual formulation | own-only vs own+peer, oracle ceiling, fixed Temporal counterfactual | mechanism implemented; formal matrix pending |
| Problem definition lacks rigor | Define lifecycle, state/measurement sets, association variables, causal information set, and evaluation masks before use | notation audit and protocol hash | outline complete; full rewrite pending |
| Simulated data only | Add IID/OOD shifts, mismatched Pd/clutter, uncertainty intervals, and explicit limitation | three seeds plus trajectory bootstrap; full BP Monte Carlo | pending |
| Strong baselines missing | Add CV, two-mode IMM, Temporal, own-only and no-context under one lifecycle/rollout protocol | same splits, masks, measurement updates, trajectory statistics and runtime | CV/IMM evaluators implemented; formal runs pending |
| Learning blocks unclear | Open the method with an intuitive workflow, then give exact tensors, histories, masks, and online/offline status | architecture diagram and tensor table | pending figure |

## Reviewer 1

| Comment | Planned response |
|---|---|
| Real radar uncertainty | Do not claim real-radar generalization. Stress-test missed detections, clutter and OOD motion; state simulation-only limitation. |
| Combination of known ideas | Remove GNN/ODE/GASF/SEAN enumeration as the novelty claim. Claim only the new causal innovation problem, peer counterfactual and soft association interface if formal evidence supports them. |
| Group-label assumption and fairness | Remove structure loss and all group labels from model inputs/training. Truth is used only for state loss and offline event evaluation. |
| Confusing ablation description | Use immutable experiment IDs P0--P5 and one flag table. |

## Reviewer 2

| Comment | Planned response |
|---|---|
| Need intuitive workflow | Add a one-page prediction--BP--soft innovation--history update diagram at the start of the method. |
| Is BP original? | State that BP is adopted; summarize the factorization and identify only the soft-marginal-to-CGIC adapter as new integration work. |
| Undefined history length | State eight frames, partial-history rule 4/8 plus 1/3 recent, and constant memory complexity. |
| CV transition versus maneuver motion | Separate learned mean propagation from covariance approximation; use CV covariance as main speed-controlled protocol and ODE Jacobian only as a diagnostic. |
| Offline versus online components | Parameters are trained offline and fixed online; graph, BP marginals, innovations, history buffers, states, and covariances update online. |

## Reviewer 3

| Comment | Planned response |
|---|---|
| State notation reused | Reserve bold stacked symbols for all targets and indexed symbols for one target. |
| Association factor notation/clutter rate | Define both association-variable families on the left-hand side and express clutter intensity explicitly. |
| Covariance update omits ambiguity | Use and derive Gaussian-mixture moment matching, including missed-detection and candidate-spread terms. |
| Neural mean / CV covariance mismatch | Name it as an approximation and compare with one-step Neural-ODE Jacobian covariance. |
| Hard maximum association | Replace CGIC history with soft BP-marginal innovation moments and report entropy/spread. |
| Learned Q loss unclear | Do not claim adaptive Q as the new contribution; list its parameterization and all losses, and retain Q diagnostics/ablation. |
| Reproducibility | Publish exact counts, seeds, manifests, commands, checkpoint rule and runtime. |
| Oracle robustness parameters | Keep nominal tracker Pd/clutter fixed while actual conditions change; matched runs are appendix-only. |
| Runtime wording | Report measured ranks and scaling without qualitative ``low complexity'' claims unsupported by data. |

## Adopted BP disclosure

The revised manuscript will keep a concise BP section because the marginal
association variables, missed-detection hypothesis, clutter intensity, and
state moment matching define the interface seen by CGIC. It will explicitly
label factorization, loopy messages, existence/birth/pruning and the standard
Kalman candidate updates as adopted backend components. Only the conversion of
soft BP marginals into strictly causal innovation/entropy/spread histories is
part of the proposed integration. Complete-tracker results will separate
motion-prior localization from track-manager birth/confirmation errors.

## Reviewer 4

| Comment | Planned response |
|---|---|
| No new problem/mechanism | Formalize causal group-innovation correction under association uncertainty and test each claimed information source counterfactually. |
| Only about 20 targets | Report analytical buffer/model complexity and measured inference scaling for 5/10/20/40 targets. |
| Figures unreadable | Regenerate plots at final column width with vector output and consistent typography. |
| Polar transform necessity | Explain the radar measurement domain and why Cartesian neural features are obtained by projection; remove unnecessary claims if it is merely coordinate conversion. |

## Response-writing rule

Use the order: acknowledge the technical issue, describe the concrete change,
report only verified evidence, and give exact revised section/table locations.
Do not answer a novelty objection with accuracy alone, and do not describe BP,
EKF, FFT, GNN, GRU, or Neural ODE as individually novel.
