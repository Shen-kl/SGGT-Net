# Response to the Editor and Reviewers — Working Revision

Manuscript: TAES-2026-1968

Dear Editor and Reviewers,

Thank you for identifying that the original manuscript did not adequately separate an actual methodological contribution from a combination of established BP, GNN, spectral, Neural-ODE, and EKF components. We have therefore changed the problem statement, network, training protocol, evaluation, and manuscript organization rather than only revising the wording.

The revised manuscript studies strictly causal group-innovation correction under uncertain association. The learned corrector sees only information available through the preceding frame. It keeps a target's own innovation separate from a leave-one-out peer innovation, uses whitening/NIS and soft-association statistics as reliability context, and predicts a bounded correction to a frozen Temporal prior. It uses no group labels, group-structure loss, association truth, current-frame measurement, confidence gate, or dynamic activation. BP is retained and cited as an adopted association backend, not claimed as a new algorithm.

All numbers below are generated from a frozen protocol and machine-audited result files. The current complete learned-model evidence uses training seed 123; we state this limitation in every relevant table and do not represent trajectory Bootstrap as a substitute for independent training seeds.

## Response to the Senior and Associate Editors

### Comment: The novelty appears incremental and the work resembles a stack of existing modules.

We agree that the original SEAN/GASF/Neural-ODE/BP contribution list did not establish a falsifiable new mechanism. We removed that list and reformulated the paper around one question: whether strictly past soft-association innovations of a target and leave-one-out spatial peers improve the next motion prior without group labels or association truth.

The revised Sec. “Causal Group-Innovation Prior Correction” defines the causal information set, own/peer separation, history eligibility, bounded correction, and exact Temporal fallback (Eqs. (soft innovation), (whitened NIS), (peer innovation), and (bounded correction)). An adopted-versus-proposed paragraph explicitly classifies BP, GNN–GRU, the temporal FFT filter, Neural ODE, RK4, EKF, and CV covariance propagation as backend/backbone elements. The proposed contribution is the soft-marginal-to-causal-own/peer-innovation interface and its counterfactual correction.

The new P2–P5 experiments test that mechanism. P4 versus P3 own-only improves position-prior RMSE by 1.357, 1.566, and 2.231 m on Validation, IID, and OOD, with trajectory-paired 95% intervals entirely above zero. P4 versus P5 without association context is small and mixed across splits; we therefore do not claim that the current context features provide an established independent gain.

### Comment: The problem definition and mathematical presentation lack rigor.

The revised problem section now distinguishes the per-target physical state, the internally permuted learned state, the measurement set, the lifecycle validity mask, and the strictly causal information set. A scored forecast requires the target to exist at both the previous and current frame; birth initializes the filter and pre-birth/post-death frames participate in neither loss nor evaluation.

The BP section now defines both target-oriented and measurement-oriented association variables, the one-to-one factor, missed detection, clutter intensity, soft marginal probability, candidate innovation, and the full Gaussian-mixture moment-matched posterior covariance. The latter contains missed-detection, within-candidate, and between-candidate covariance terms. The former hard maximum-association update has been removed.

### Comment: Evidence is simulation-only and strong baselines/statistical analysis are insufficient.

We now compare a Validation-tuned CV model (P0), a two-mode IMM (P1), the frozen Temporal learned prior (P2), an own-only correction (P3), full CGIC (P4), and CGIC without association context (P5), under identical lifecycle masks and recursive predict–update feedback. The frozen splits contain 1000/200/300/400 training/Validation/IID/OOD trajectories. IID/OOD test data are never used for model selection.

P4 reduces P2 position-prior RMSE from 17.965 to 16.396 m on IID and from 22.987 to 20.755 m on OOD. The paired improvements are 1.569 m [1.344, 1.798] and 2.232 m [1.963, 2.502], respectively. We additionally report event-stratified results, NIS calibration, a fixed association-perturbation rollout, parameters/runtime, and a complete BP missed-detection/clutter matrix. We explicitly retain simulation-only validation as a limitation and make no real-radar deployment claim.

## Reviewer 1

### Comment: The method combines existing BP, GNN, spectral, and ODE components and its novelty is unclear.

We replaced module-by-module novelty with the causal innovation problem and counterfactual described above. The revised introduction no longer claims that combining standard blocks is itself novel. The method section states exactly which components are adopted and which interface/correction is proposed. The own-only and no-context counterfactuals test the incremental information sources instead of merely disabling large named blocks.

### Comment: Group structure requires truth labels and may make comparisons unfair.

SEAN, group labels, structural focal loss, and structural temporal consistency loss are removed from the final method. The formal settings use `decoder_use_SEAN=False` and a zero structure-loss weight. Historical edges are constructed from online state estimates with a 150-m distance rule, not from group membership. Ground-truth state is used only for supervised tracking loss, and truth-derived steady/switch/strong labels are used only to partition evaluation statistics after inference.

### Comment: Baselines and ablations are limited or confusing.

The old “w/o SEAN/w/o GASF” table has been replaced by immutable IDs P0–P5. The revised comparison table gives the motion prior, peer input, and association-context input for every group. CV/IMM parameters are selected only on Validation. All learned variants use the same V9 splits, Temporal checkpoint, lifecycle mask, recursive evaluator, and CV covariance transition.

## Reviewer 2

### Comment 1: The workflow is difficult to follow for a tracking reader.

We added an online flow diagram before the detailed equations: causal buffer through frame (k-1) → frozen Temporal prior → CGIC correction → BP soft association using the current measurement → mixture-matched update → append the new innovation for frame (k+1). A second diagram shows the separate own and leave-one-out peer channels. The current measurement has no path to CGIC before the frame-(k) prior is fixed.

### Comment 2: It is unclear whether BP is original and how it is adapted.

BP is now explicitly called an adopted backend and cited. We retain a concise description because its marginal probabilities determine the soft measurement, missed-detection probability, candidate spread, and innovation stored for CGIC. The proposed integration is the use of these strict causal marginal moments in the next prior; BP message updates themselves are not new.

### Comment 3: History length and variables are introduced too late.

The revised method defines the eight-frame history before the network equations. A correction is enabled only with at least four valid own/peer pairs and at least one valid pair in the latest three frames. The buffer length is constant, so storage does not grow with trajectory duration.

### Comment 4: CV covariance propagation may be inconsistent with CT/CA motion.

We now separate mean and covariance models. The frozen backbone uses a Neural-ODE/RK4 mean predictor, whereas the main speed-controlled protocol uses a CV transition to approximate covariance propagation. This approximation is common to P2–P5 and therefore cannot explain their relative mean improvement. We add a two-mode IMM baseline and report poor BP-stress NIS coverage as evidence that covariance approximation remains a limitation; ODE-Jacobian covariance is not presented as part of the final main model.

### Comment 5: Offline training and online adaptation are unclear.

All neural weights are learned offline and frozen at inference. Posterior states, covariances, radius-graph edges, BP marginals, innovation histories, and eligibility masks update online. The retained legacy adaptive-Q head is trained only with the Temporal backbone and then frozen for P3–P5. Its special rank/surprise/smooth/baseline loss weights are all zero; only the posterior and (10^{-4})-weighted Gaussian NLL paths provide covariance-related gradients.

## Reviewer 3

### Comment: Notation for one target and the target set is inconsistent.

The revised text reserves indexed lower-case state symbols for a target and separately defines the target set/stack. It also explains the deterministic permutation between the physical interleaved state and the learned position-first state. Every measurement and association index is defined before use.

### Comment: The factorization omits an association variable and clutter intensity is unclear.

The revised BP section introduces both (a_{i,k}) and (b_{j,k}) and their one-to-one consistency factor. It states that the single-target likelihood ratio contains detection probability and Poisson spatial clutter intensity. The complete robustness experiment reports both nominal and actual detection/clutter parameters.

### Comment: The covariance ignores missed detection/association spread, and a hard maximum association is used.

We replaced the old covariance equation with Gaussian-mixture moment matching over missed detection and every candidate association. The covariance includes conditional covariance plus between-component mean spread. The state and stored innovation use soft BP marginal moments; maximum-probability association is not used by CGIC.

### Comment: The learned process-noise objective is missing.

The revised training section lists the exact Temporal objective: prior loss + posterior loss + (0.1) auxiliary velocity-increment loss + (10^{-4}) Gaussian NLL. Structure and all dedicated Q regularizers have zero weight. The legacy adaptive-Q output is not claimed as new and is frozen during corrector training. The corrector objective is separately given as corrected-state Huber loss plus a 0.25 counterfactual no-regression term.

### Comment: Reproducibility, Monte Carlo counts, and robustness parameters are vague or may be oracle matched.

We report 1000/200/300/400 trajectories, batch size 4, optimizer/schedule, stopping rule, code commit, protocol hash, checkpoint epochs and hashes, and GPU. The main BP test fixes nominal (P_D=0.85) and clutter mean 20, disables online (P_D) adaptation, and changes actual (P_Din{0.75,0.85,0.95}) and clutter mean ({10,20,30}). Each condition contains three trajectories and 20 paired Monte Carlo realizations per trajectory. A hierarchical Bootstrap resamples scene first and Monte Carlo realization second.

### Comment: Runtime/complexity claims are unsupported and the previous method was not actually fastest.

We removed the qualitative “low complexity” ranking. The revised table reports 317,540 P2 and 335,209 P4 parameters and controlled prediction-interface latency/memory at 5, 10, 20, and 40 targets. It explicitly excludes model loading, data I/O, BP association, measurement update, and track management, and does not infer end-to-end asymptotic complexity from the nearly flat controlled latency.

### Comment: Polar conversion appears unnecessary.

The revised measurement section explains that the radar supplies range/azimuth noise while the tracker and neural model operate in Cartesian coordinates. The first-order Jacobian converts the scene-specific noise covariance to Cartesian form. We state that this is sensor preprocessing, not a methodological contribution.

## Reviewer 4

### Comment: The manuscript does not rigorously define a new problem or mechanism and lacks general analysis.

The revised problem is association-uncertainty-aware causal group-innovation correction, with a precise information set, exact no-correction counterfactual, validity rule, and bounded correction. Mechanism evidence is no longer inferred only from a global average. On Validation, position-prior improvement grows from 1.033 m [0.835, 1.227] in steady frames to 1.512 m [1.198, 1.837] at switches and 1.717 m [1.328, 2.133] under strong maneuver. IID/OOD and trajectory-level win fractions are also reported.

### Comment: Target-count generality, storage, and figures are inadequate.

The causal history is fixed at eight frames, so per-target innovation storage is constant in trajectory length. We add controlled 5/10/20/40-target timing and CUDA-memory measurements, parameter counts, and exact benchmark exclusions. Mechanism and result figures are regenerated as vector TikZ/SVG assets with a common font and explicit statistical unit.

## Negative and boundary results retained in the revision

We have not removed results that weaken a broad claim. P4 versus P5 is nearly tied and directionally mixed, so association context is not claimed as independently beneficial. In the complete BP matrix, matched-state NIS 95% coverage changes from 0.4160 to 0.4096. The hardest (P_D=0.75), clutter-30 condition slightly regresses. Hierarchical BP Bootstrap gives a matched-position macro improvement of 0.394 m with interval [-0.519, 0.687], while matched velocity and OSPA intervals are positive. These results are presented as limitations and failure analysis.

## Reproducibility and current completion status

The formal protocol SHA-256 is `63b0ee5c33d4a8df6ad4c8802293def1e61ce737cd8ba319f12be95a6f1b917e`, and the code commit is `e192aa680069673ec93bd47d153f989fcc6bf930`. The P2/P4 seed-123 checkpoint hashes are recorded in the manuscript. An automated audit verifies 22 tracking reports, eight NIS reports, 12 paired trajectory-Bootstrap reports, 18 BP condition summaries, all BP scene-source hashes, and complexity files with no missing/non-finite/count/hash errors.

The revised manuscript currently labels learned results as seed 123. Seed 456 has a Temporal checkpoint but not a P4 checkpoint; seed 789 P2/P4 are not complete. In accordance with the present experimental decision, we do not claim three-seed completion. Final page/line numbers and any final multi-seed summary will be inserted only after those assets exist and the paper is compiled.

Sincerely,

Kailun Shen and Weiming Yuan
