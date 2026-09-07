# 2026-08-13 Association-Aware CGIC Debug Record

The legacy `debug_record.md` contains mixed text encoding, so this UTF-8 record
is kept separately instead of risking a destructive recode while experiments
are active.

## Problems

- Hard maximum association discarded missed-detection probability and candidate
  spread before constructing the innovation history.
- Requiring all eight history frames made the corrector unavailable after a
  single missed detection.
- Directly mixing other targets/global clutter into fixed-ID supervised
  predictor training produced identity-breaking feedback because that rollout
  has no existence/reacquisition manager.
- The development evaluation manifest was created before the 50-trajectory
  subset and falsely recorded 200 trajectories.
- Epoch -1 was overwritten by later `last/best` checkpoints, and checkpoint
  selection used clean rather than fixed-association validation.

## Measures

- Added soft PDA moment matching and causal association statistics; the BP
  adapter consumes soft marginals and stores strict prior innovations.
- Added 4-of-8 plus recent-1-of-3 partial history and an own-only ablation.
- Restricted training perturbations to misses plus local identity-consistent
  ambiguity; full global clutter stays in the BP tracker evaluation.
- Built the manifest after subsetting and added a subset regression test.
- Added immutable `initial.pth` / `epoch_minus_one_metrics.json`, plus an
  independent `best_association_position.pth` selected on deterministic
  perturbations.
- Added an executable 0.05 m development gate. The formal three-seed runner
  refuses to start without an approved gate result.

## Verification and next action

- 78 unit/regression tests pass; GPU smoke and 32/16 training are finite.
- The 32/16 fixed-association gain is only 0.0139 m prediction and 0.0073 m
  update, so it is not sufficient evidence for the paper.
- The corrected 200/50 run is the next active experiment. If it fails the
  0.05 m gate, stop and revisit the association-context learning signal rather
  than launching formal seeds or writing a positive conclusion.

## Formal multi-seed launch audit

### Additional problems found

- The first seed-789 Temporal attempt stopped at 12% of epoch -1 validation.
  Its process remained attached to the command-execution job and was terminated
  when that job ended; no NaN and no trained checkpoint were produced.
- That attempt exposed a validation-protocol schema regression. Seeds 123/456
  stored version 2 with `covariance_transition=cv`; `main_v3.py` had later
  emitted version 1 without that field, changing the hash despite the same 200
  validation trajectories.

### Measures and evidence

- The incomplete attempt and logs were first moved to
  `checkpoint/incomplete_archives/seed789_protocol_v1_20260813_1750`, then
  deleted during the 2026-09-06 cleanup after confirming that no trained
  checkpoint was present.
- The version-2 protocol schema was restored. A regenerated manifest exactly
  matches the seed-123/456 hash
  `36e5157d5c49cc99c29f7b650e32a7f3c0347770c19d7839142a03a81b91346f`.
- Strict multi-seed checkpoint verification now covers architecture switches,
  state-tensor shapes, finite metrics, validation files and protocol identity.
- Evaluator, calibration, bootstrap and BP stages now resume only when report
  checkpoint hashes match. Temporal and full-CGIC BP reports have separate
  model roots and cannot overwrite one another.
- V9 contains 5--14 simultaneous targets. The 5/10/20/40 complexity benchmark
  is therefore explicitly timing-only: it copies a real five-target causal
  window into disconnected graph blocks, and no replicated-graph accuracy is
  reported.
- All 93 unit tests pass and `git diff --check` reports no whitespace errors.
- Frozen formal protocol SHA-256 is
  `63b0ee5c33d4a8df6ad4c8802293def1e61ce737cd8ba319f12be95a6f1b917e`;
  code-manifest SHA-256 is
  `f03d17d6e926dbf644b2add96d2a6e43568cedf862bf7ebfab490baa5c7f00c9`.

### Current execution

Windows task `SGGT_Formal_Paper_Experiments_20260813` launched the resumable
pipeline at 20:47 Asia/Shanghai. Seed-789 Python PID 16608 was alive and
responsive at 20:50, with increasing CPU time, about 4.7 GiB host RAM and 2.3
GiB GPU memory. Its validation protocol is the correct version 2/hash above.
P3/P4/P5 cannot begin until `best_prediction.pth` exists and all three Temporal
checkpoints pass the strict verifier.

## 2026-08-17 NIS evaluation failure and auditable resume

### Failure

- P0--P5 training and clean/association/IID/OOD rollout completed for seed 123.
- The pipeline then failed on the first P2 Temporal NIS trajectory because the
  disabled causal branch returns an empty one-dimensional eligibility tensor,
  while the evaluator indexed it as a populated `[N, 1, 1]` tensor.
- The failure was an evaluator shape-contract bug. It was not a NaN, covariance
  failure, checkpoint incompatibility, or training failure.

### Fix and evidence

- Empty optional causal outputs now map to one false eligibility value per
  node; populated outputs are shape-checked and flattened safely.
- When the legacy GASF decomposition diagnostic is absent, NIS evaluation now
  reads the actual strict-causal own and leave-one-out-peer histories from the
  rolling online window. This matches the final CGIC architecture, which keeps
  own and peer channels separate.
- Four targeted regression tests and all 97 unit tests pass.
- Real V9 CUDA one-trajectory checks pass for both P2 and P4. Each records 828
  current NIS samples and 819 causal own/peer history samples without NaN.
- The original frozen protocol is preserved. The evaluator-only repair is
  recorded in `results/formal_association_cgic/protocol_amendment_nis_fix_20260817.json`.
- `scripts/resume_formal_association_cgic_from_nis.ps1` resumes only NIS,
  paired bootstrap, summary, scaling and full-BP robustness stages; it never
  launches training or rewrites completed rollout reports.

## 2026-08-18 BP group-partition failure and continuation repair

### Reproduction and diagnosis

- The requested single-scene/single-Monte-Carlo case at actual Pd=0.75 and
  clutter=20 passed, ruling out an immediate checkpoint, CUDA, or parameter
  compatibility failure.
- The same fixed seed sequence with 20 Monte Carlo runs failed after eight
  completed measurement generations. The complete traceback is preserved in
  `log/bp_repro_pd075_clutter20_scene01_mc20.stderr.log` and ends at
  `gmComponent.X_group_partition[:, group_index]` with an axis-size mismatch.
- BP may retain multiple Gaussian-mixture components. The neural rescaler
  refreshed and expanded only `gmComponents[0]`; a later component therefore
  retained one partition column while the current group manager exposed two or
  more hypotheses. Measurement evaluation correctly iterated all components
  and all current hypotheses, revealing the stale axis.
- Other considered causes were GM weight/count bookkeeping, track-to-group
  indexing, covariance NaN/singularity, and seed-specific measurement input.
  The deterministic Python `IndexError`, successful first run, and successful
  post-fix repetition identify stale component/partition synchronization as the
  actual cause.

### Repair and verification

- The shared neural state prior is now assigned to every active GM component;
  each component preserves and predicts its own covariance, then both state and
  covariance are replicated over every current group-partition hypothesis.
- GM component/count/weight consistency is checked before applying the prior.
  Measurement evaluation also reports target, component, expected hypothesis
  count and observed tensor shapes if the invariant is violated again.
- The BP matrix runner now saves per-cell stdout and full stderr tracebacks.
- Seven targeted tests and all 99 repository tests pass. The exact failed
  single-scene condition then completed 20/20 Monte Carlo runs (indices 0--19)
  with no invalid values and empty stderr. Its matched position/velocity RMSE
  is 19.8409/5.4790 m(/s), and mean OSPA is 70.1545.
- The frozen training protocol and checkpoints are unchanged. The evaluator
  repair and hashes are audited in
  `results/formal_association_cgic/protocol_amendment_bp_partition_fix_20260818.json`.
