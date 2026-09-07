# Response Letter Working Draft

Dear Editor and Reviewers,

Thank you for identifying the manuscript's central weaknesses: the original
version framed a composition of established modules as the contribution, used
group-label supervision that weakened baseline fairness, did not represent
association uncertainty consistently in its innovation and covariance update,
and did not provide a sufficiently reproducible or statistically supported
evaluation. We have therefore redefined the technical problem and rebuilt both
the method and experimental protocol rather than making a cosmetic revision.

The revised manuscript studies a strictly causal and label-free group
innovation prior under uncertain association. The learned model receives no
group labels or association truth. It separates a target's own past innovation
from leave-one-out peer innovation and uses soft BP marginal statistics to
represent detection mass, association entropy, and candidate spread. The BP
tracker is now stated explicitly as an adopted backend; standard BP, EKF, GNN,
FFT, GRU, and Neural ODE components are not claimed as individually novel.

All numerical statements below remain placeholders until the frozen
three-random-seed and BP Monte Carlo experiments finish.

## Response to the Senior and Associate Editors

Comment: The novelty appears incremental, the problem definition needs more
rigor, only simulated scenarios are used, strong baselines are missing, and the
learning blocks need clearer explanation.

Response: We agree that the former module-by-module positioning did not
establish a research-level contribution. In the revised manuscript, Sec. II
defines the states, measurements, lifecycle mask, uncertain association
variables, and causal information set before use. Sec. III distinguishes the
adopted BP tracker from the proposed interface. Sec. IV formulates the causal
group-innovation correction and an exact Temporal counterfactual. The new
experiments compare CV, IMM, Temporal, own-only CGIC, full CGIC, and CGIC
without association context under identical lifecycle masks and recursive
prediction/update feedback. CV/IMM process noise is selected on validation
only and frozen for IID/OOD tests. We also
add IID/OOD motion tests, unmatched detection/clutter conditions, three seeds,
and trajectory-paired bootstrap intervals. Because the data remain simulated,
we state this limitation and do not claim real-radar deployment.

## Reviewer 1

Comment: Simulation-only evidence, unclear novelty, weak baselines, unfair use
of group labels, and confusing ablations.

Response: Group-structure supervision has been removed. Neither group labels
nor association truth enter training or inference; ground-truth state is used
only in the supervised tracking loss and truth-derived maneuver classes only
for offline evaluation. The contribution is reformulated as a causal
innovation-estimation problem rather than a combination of named modules. All
ablations now use immutable IDs P0--P5, identical splits, and an exact
alpha-zero counterfactual. Results and confidence intervals will be inserted
from Table [TBD] after the frozen experiments complete.

## Reviewer 2

Comment: The overall workflow is difficult to follow; BP originality,
historical length, non-CV motion, and offline/online adaptation are unclear.

Response: The beginning of Sec. IV now gives an intuitive online sequence:
prior prediction, BP soft association, probabilistic update, innovation-buffer
update, and next-frame correction. BP is cited and described as adopted. The
new integration consists only of exposing soft marginal moments to the causal
prior. History length is fixed at eight frames with a documented partial
history fallback, so storage does not grow with time. Learned parameters are
trained offline and held fixed; states, covariances, graphs, BP marginals, and
innovation buffers update online. The learned mean and covariance approximation
are described separately, with CV covariance in the main speed-controlled
protocol and Neural-ODE Jacobian covariance as an explicit diagnostic.

## Reviewer 3

Comment: Notation, clutter intensity, association-aware covariance, hard
association, group-label supervision, the Q objective, reproducibility,
robustness parameters, and runtime claims need correction.

Response: The revision reserves indexed symbols for one target and stacked
symbols for the target set, defines both association-variable families, and
includes clutter intensity explicitly. The posterior covariance is now
computed by Gaussian-mixture moment matching over missed detection and all
candidate associations, including between-candidate spread. CGIC uses the soft
association innovation rather than a maximum-probability measurement. Group
labels and the structure loss are removed. The process-noise parameterization
and every weighted loss are listed in a reproducibility table. The main
robustness experiment holds the tracker's nominal detection probability and
clutter mean fixed while actual conditions change; matched-parameter runs are
appendix-only, and adaptive online Pd learning is disabled in this mismatch
test. Complete-tracker OSPA is decomposed into localization/cardinality and
reported together with missed/false tracks, fragmentation, identity switches,
and matched-state calibration, so track-management errors are not attributed
to the motion prior. Runtime and scaling are reported numerically without an
unsupported qualitative complexity claim.

## Reviewer 4

Comment: The work does not rigorously define a new problem or mechanism, lacks
general analysis, tests few targets, and has readability issues.

Response: Sec. II--IV now formulate causal group-innovation correction under
association uncertainty, establish an exact no-correction counterfactual, and
test the incremental information in aligned peers and association context. We
add event-stratified IID/OOD results, paired uncertainty intervals, analytical
buffer/model complexity, and measured scaling for 5, 10, 20, and 40 targets.
Figures will be regenerated as vector graphics at final column width. The polar
measurement conversion is described as a sensor-coordinate operation, not as
a methodological contribution.

## Finalization checklist

- Replace every `[TBD]` with a table/section identifier only after the result
  artifact exists.
- State the code commit, protocol hash, seeds, split counts, hardware,
  checkpoint criterion, and early-stopping threshold.
- Add a concise adopted-versus-proposed table for BP and CGIC.
- Do not reuse numerical results from the rejected protocol.
