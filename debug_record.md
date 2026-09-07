# SGGT-Net 调试记录

### 2026-08-05: Q-v2 因果适配器探针

#### 继续优化：因果尾部 Q

依次验证了事件标签排序、因果惊讶度四分位排序、绝对锚点、位置-only Q、绝对尾部监督和确定性单调尾部反事实。自由 MLP/标签排序的主要问题是尾部样本稀少，模型倾向于整体降低 Q 来满足相对排序；位置-only Q 则破坏跨步位置—速度相关性，使后续预测恶化。

当前代码同版本、完整 200 条验证轨迹的可靠对照如下。基线为原 epoch 12 checkpoint；候选为严格因果尾部倍率 `m=min(5, 1+4*relu(s-0.8))`，其中 `s` 只由预测时刻之前的残差历史计算。

| 指标 | 基线 | 尾部 Q | 相对变化 |
|---|---:|---:|---:|
| 全局位置预测 RMSE | 112.1669 | 111.8819 | -0.25% |
| 全局位置更新 RMSE | 50.1463 | 49.7416 | -0.81% |
| 全局速度预测 RMSE | 30.7498 | 30.8866 | +0.44% |
| 全局速度更新 RMSE | 31.8621 | 31.9881 | +0.40% |
| 全局 NIS 均值 | 229.17 | 223.61 | -2.42% |
| switch 位置更新 RMSE | 85.5672 | 84.7707 | -0.93% |
| strong 位置更新 RMSE | 52.4899 | 52.0049 | -0.92% |

steady 指标变化小于约 0.01%。阈值 0.7/0.8/0.9 以及倍率系数 4/8 已因最大倍率 5 饱和，结果几乎一致；继续提高倍率没有意义。独立位置尾部 Q 在 8 条轨迹上虽然略改善当前更新，但使 strong 后续位置预测恶化约 1.6 m，已否决。最终保留加速度空间的单调尾部 Q，默认 legacy 行为仍不变；正式启用参数为 `adaptive_q_tail_boost=4`、`adaptive_q_tail_threshold=0.8`。

注意：早期 Q-v2 小探针曾误用 `main.py` 的 `S_VEL=300`。这些探针只用于机制/梯度排查，不作为精度证据。入口已修复为 v3 pack 自动同步 `S_VEL=40`、`S_EDGE_V_DIST=40`。上表两组均由同一当前评估器和正确 v9 配置重新完整 rollout，才是可比较结论。

新增可选且兼容旧 checkpoint 的 `adaptive_q_mode=innovation`：只读取预测帧之前的残差历史、decoder hidden 和 GASF latent，输出有界的加速度 Q 倍率以及独立 PSD 位置协方差；legacy 仍为默认模式。

验证：16 项单元测试通过；legacy Q 与 Q-v2 的真实 v9 前向、反向均通过且无 NaN。

探针 1（32 train/8 val、Q-only 3 epoch、lr 3e-4）发现 `softplus(-8)^2` 导致位置 Q 梯度近乎消失：额外位置 Q trace 约 2.19e-7 m^2，strong/steady 倍率仅相差约 0.14%。已否决该参数化。

探针 2 改为有界 sigmoid 方差（每轴上限 400 m^2，初始总 trace 0.02 m^2），冻结主干并置 eval，仅训练 Q 头；1 epoch，lr 1e-4，lambda_nll 1e-3，lambda_q_rank 1，耗时 9.7 分钟。epoch -1 位置预测/更新为 44.8113/22.1745 m；epoch 0 为 44.8112/22.1746 m，位置和没有改善，因此 `best_position` 正确保留 epoch -1。

| 事件 | 加速度倍率 | 额外位置 Q trace (m^2) | NIS 均值 |
|---|---:|---:|---:|
| steady | 0.998599 | 0.019870 | 0.932 |
| switch | 0.998673 | 0.019872 | 0.841 |
| strong | 0.998749 | 0.019877 | 374.55 |

结论：数值梯度问题已修复，但按事件标签做批内排序仍几乎没有产生 Q 分离。事件窗口较宽、因果残差分布重叠，且只含一种事件的 batch 会令排序损失失活。当前版本不扩大训练。下一单因素应按因果历史惊讶度排序（批内或跨批队列中的高/低残差或 NIS 代理），再由 NLL/logdet 约束绝对尺度；只有 strong/steady Q 出现实质分离且位置不退化后，才扩大到 200 场景。

## 2026-08-03：速度改善未转化为位置更新改善

### 现象

- v9 全量训练中，SEAN+GASF 最佳点为 epoch 7：位置预测/更新 RMSE 为 30.495/23.052，速度预测/更新 RMSE 为 4.877/4.202。
- 继续训练到 epoch 15 后，速度预测/更新改善到 4.212/3.627，但位置预测/更新恶化到 32.857/25.823。
- 验证 loss 持续下降，但完整轨迹的综合 RMSE 在 epoch 7 后上升。

### 三项诊断结果

诊断覆盖验证集前 8 条完整轨迹（736 个在线预测—更新步骤）以及 48 个梯度采样时刻。

1. **损失梯度失衡**：速度预测梯度是位置预测梯度的约 214～262 倍；速度更新梯度是位置更新梯度的约 325～365 倍。`S_POS=1500`、`S_VEL=40` 下的等权归一化损失实际上严重偏向速度。
2. **EKF 轻微过度置信**：改进模型的位置 Kalman 增益由基础模型的 0.8003 降至 0.7838，位置方差由 201.20 降至 194.69。位置先验变好，但滤波器减少了量测修正，抵消了位置预测收益。
3. **RK4 不是主因**：SEAN+GASF 的平均运动学一致性误差仅 0.0068 m，95% 分位仅 0.0207 m；同时一步位移误差比基础模型改善约 4.1%。

限制：训练器只保存刷新最佳值的 checkpoint，没有 epoch 15 权重。因此 epoch 15 的内部梯度、Kalman 增益和协方差无法直接恢复，不能将日志趋势冒充 checkpoint 诊断结果。

### 分阶段措施

1. 第一轮只修复已确认的梯度失衡：增加独立的 `position_loss_weight`，初始尝试位置/速度权重为 16/1；CV guard 同步使用位置/速度 16/1。
2. 使用固定种子和相同 64/20 场景进行短程单因素 A/B，检查位置 RMSE、速度 RMSE、NaN 和梯度比例；不同时修改网络结构或 RK4。
3. 如果位置改善且速度没有明显崩溃，再扩大到 200 场景，并测试位置权重 8/16/32。
4. 只有在损失平衡后仍出现“先验改善、后验不改善”，才单独测试协方差膨胀或位置方差下限。
5. 轻量运动学一致性损失暂不启用，因为现有误差只有厘米级，不是当前瓶颈。

诊断结果：`evaluation_results/v9_base_epoch2_dynamics.json`、`evaluation_results/v9_sean_gasf_epoch7_dynamics.json`。

### 第一轮修改结果：位置权重 16

已实现 `position_loss_weight`，并在相同 64/20 场景、seed 123、基础模型、CV guard、3 epoch 协议下完成单因素对照。

| 配置（epoch 2） | 位置预测 | 速度预测 | 位置更新 | 速度更新 | 综合分数 |
|---|---:|---:|---:|---:|---:|
| 位置/速度权重 1/1 | 25.8748 | 4.6242 | 18.6587 | 3.8744 | 53.0320 |
| 位置/速度权重 16/1 | 25.7923 | 4.6134 | 18.5943 | 3.8664 | 52.8664 |

位置权重 16 相对等权配置在四项指标上均未退化，综合分数改善约 0.31%；相对训练前综合分数 53.6403 改善约 1.44%。真实数据前向、反向和 12 项单元测试通过，未出现 NaN。该结果支持继续扩大实验，但提升幅度较小，尚不能证明 16 是最优权重，也不能直接外推到 SEAN+GASF 全量训练。

### SEAN+GASF 扩大验证与方案选择

64 场景、8 epoch 时，位置权重 1 和 16 均未超过 epoch -1，结构 F1 也保持不变，说明 64 场景不足以训练正式模型。权重 16 在训练中后段退化较少，但不据此作最终选择。

随后扩大到 200 训练场景、20 验证场景、8 epoch，其他配置保持一致。两组最佳点均为 epoch 7：

| 配置 | 位置预测 | 速度预测 | 位置更新 | 速度更新 | 综合分数 |
|---|---:|---:|---:|---:|---:|
| 位置/速度权重 1/1 | 25.7301 | 4.5694 | 18.6439 | 3.8287 | 52.7721 |
| 位置/速度权重 16/1 | 25.5137 | 4.5664 | 18.4365 | 3.8221 | 52.3386 |

权重 16 的综合分数改善约 0.82%，位置预测改善约 0.84%，位置更新改善约 1.11%，速度指标没有明显损失。由此选择 16/1 进入全量 SEAN+GASF 训练；暂不扫描 8/32，避免在单一 20 场景验证子集上继续调参。

### 全量位置权重 16：人工停止决定

全量训练在 epoch 14 后人工停止（2026-08-03），原因是论文目标更侧重位置估计，而 epoch 7 后的主要收益来自速度、位置反而退化。进程停止时两个候选 checkpoint 均完整保留。

- epoch 7：位置预测 29.781、位置更新 22.184、速度预测 5.030、速度更新 4.315，综合分数 61.310。
- epoch 11：位置预测 29.882、位置更新 22.733、速度预测 4.667、速度更新 4.012，综合分数 61.294。

epoch 11 的简单综合分数只改善约 0.026%，但位置预测/更新分别恶化约 0.10 m/0.55 m，因此位置主任务选用 epoch 7。下一步先改为并行保存“最佳位置、最佳综合、最后一轮”checkpoint，再在 200 场景上单独测试 `position_loss_weight=32`、保持 `cv_guard_position_weight=16`，避免混淆轨迹位置监督与 CV 保护的贡献。

### 200 场景：位置损失权重 32、CV guard 位置权重 16

新 checkpoint 策略已验证，可并行保存 `best_position.pth`、`best_composite.pth` 和 `last.pth`，本实验按位置主指标选择模型。两组最佳点均为 epoch 7：

| 配置 | 位置预测 | 位置更新 | 位置和 | 速度预测 | 速度更新 | 综合分数 |
|---|---:|---:|---:|---:|---:|---:|
| loss 16 / guard 16 | 25.5137 | 18.4365 | 43.9502 | 4.5664 | 3.8221 | 52.3386 |
| loss 32 / guard 16 | 25.3315 | 18.2502 | 43.5817 | 4.5664 | 3.8201 | 51.9683 |

loss 权重 32 使位置预测改善约 0.71%、位置更新改善约 1.01%、位置和改善约 0.84%，综合分数改善约 0.71%；速度没有可见损失。方向稳定，但位置总体改善略低于预设的约 1% 全量训练门槛，属于边界结果。若论文愿意为约 0.37 m 的位置和收益承担一次约 10 小时全量训练，可继续权重 32；否则保留全量权重 16，并转向协方差校准。

### 全量位置损失权重 32最终结果

全量训练于 2026-08-04 23:29 完成，在 epoch 20 因位置指标连续 8 个 epoch 未改善而早停，总耗时约 13 小时 54 分。`best_position` 为 epoch 12，`best_composite`/`last` 为 epoch 20。

- epoch 12（位置最佳）：位置预测 28.003、位置更新 20.955、速度预测 4.690、速度更新 4.035、F1 0.9521，位置和 48.959。
- epoch 20（综合最佳）：位置预测 27.931、位置更新 21.296、速度预测 4.385、速度更新 3.814、F1 0.9536，综合分数 57.426。

相对全量权重 16 的位置最佳 epoch 7，权重 32 的位置预测/更新分别改善约 5.97%/5.54%，位置和改善约 5.79%。位置更新 20.955 也比量测位置 RMSE 22.947 改善约 8.68%。因此全量证据明确支持将位置损失权重 32、CV guard 位置权重 16 固定为位置主任务的正式候选协议。

### 2026-08-05：过程噪声 Q 全量反事实评估

使用全量权重 32 的 epoch 12 `best_position` checkpoint，在完整 200 条验证轨迹上从起点重新执行在线 rollout。比较学习 Q、Q=0、固定平均 Q、0.5/1.5/2.0 倍学习 Q；每种模式都会改变后验并继续影响后续状态与残差，不是单步离线重算。

核心全局结果（节点时刻汇总，包含机动尾部，故数值大于按轨迹平均的训练日志）：

| Q 模式 | 位置预测 | 位置更新 | 速度预测 | 速度更新 | NIS 均值 | 95%覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| learned | 100.88 | 83.31 | 11.11 | 11.36 | 637.4 | 90.8% |
| zero | 277.15 | 261.69 | 12.14 | 11.85 | 1939.1 | 32.2% |
| fixed mean | 98.15 | 66.86 | 21.92 | 22.69 | 294.4 | 90.8% |
| 0.5× | 105.95 | 89.80 | 9.56 | 9.71 | 744.3 | 87.1% |
| 1.5× | 98.44 | 79.65 | 12.45 | 12.79 | 579.5 | 92.6% |
| 2.0× | 96.69 | 77.02 | 13.24 | 13.64 | 540.7 | 93.8% |

学习 Q 在平稳段校准良好（NIS 2.04、95%覆盖 95.3%），但切换/强机动段 NIS 均值分别达到 2086.8/827.7；Q trace 仅从平稳 0.9404 增至切换 0.9434、强机动 0.9489，变化不足 1%，未随误差量级显著增长。Q=0 的灾难性退化证明 Q 必不可少；固定平均 Q 改善位置但使速度约翻倍，证明节点/时变 Q 对速度有用，却没有达到位置主任务的合适校准。增大 Q 可稳定改善位置与覆盖，但会持续损害速度。

结构原因：当前 Q 是二维加速度协方差，经 `G q G^T` 同时映射到位置和速度；T=2 时位置系数 `T^2/2` 与速度系数 `T` 都等于 2，因此两类物理协方差增量被强制同尺度，无法独立校准。Q 只读取 decoder hidden，不读取 GASF 残差/机动特征；协方差跨步又被 detach，缺少多步校准信用。

可靠结论：当前自适应 Q 并非无效，而是“平稳段有效、机动段响应不足、位置—速度耦合不合理”。下一步不应简单全局放大 Q；应保留加速度基础 Q，并增加由 GASF/创新驱动的 PSD 位置模型失配项，允许位置与速度不确定性分别校准，同时使用事件平衡的创新 NLL/NIS 监督。
## 2026-08-09: spectral-SEAN refactor and demand-label audit

Problem: the embedded SEAN activation remained effectively constant even
after graph-expert auxiliary training. A first optimizer audit also found that
`interaction_gate_lr_multiplier` had been inserted in the dataset-selection
branch, so the selector never received its requested learning-rate multiplier.

Changes and checks:

- Moved the selector parameters into a verified AdamW group. Joint SEAN+GASF
  startup now prints and asserts `backbone` and `interaction_selector` groups.
- Removed group-structure label dependence. SEAN controls only the graph
  interaction increment inside GASF; the temporal counterfactual remains the
  reference path.
- Added causal residual-magnitude/change context, fixed physical maneuver
  demand supervision, trajectory-batch class balancing, and pooled confusion
  metrics.
- Fixed a label-alignment error: maneuver demand had used current truth minus
  the learned previous posterior, so its positive fraction changed with the
  checkpoint. It now uses current truth velocity minus previous truth velocity.
- On the first 16 validation trajectories the corrected >2 m/s maneuver prior
  is fixed (pooled value about 0.272 over 13,013 valid points). The previous
  model-dependent label reported roughly 0.58--0.65 and is invalid.
- 25 unit tests and the real V9 forward/backward smoke test pass.

Controlled full-time pilot (`8 train / 8 val`, seed 123, three epochs,
backbone LR 1e-4, selector LR 1e-3, demand weight 0.1, additive fusion):

| epoch | pos pred | pos update | demand BCE | demand balanced acc | alpha mean | alpha std |
|---:|---:|---:|---:|---:|---:|---:|
| -1 | 28.1209 | 18.9154 | 0.6926 | 0.5438 | 0.49956 | 0.000043 |
| 0 | 28.4337 | 19.1455 | 0.6880 | 0.5000 | 0.49552 | 0.000065 |
| 1 | 27.6393 | 18.5834 | 0.6844 | 0.5000 | 0.49231 | 0.000087 |
| 2 | 27.4118 | 18.4214 | 0.6832 | 0.5000 | 0.49114 | 0.000097 |

At the best epoch, mean activation is 0.491343 for maneuver positives and
0.491105 for negatives (separation 0.000238). The trajectory model trains and
remains finite, but this separation is too small to support a claim of dynamic
on-demand interaction. Do not scale this configuration to 200/50 or the full
split. The next bounded experiment should improve the selector input/target
identifiability rather than merely increasing LR or loss weight again.

### 2026-08-09 paired SEAN-versus-temporal intervention

The epoch-2 checkpoint above was evaluated on the same first eight validation
trajectories (6,148 valid online points) under three closed-loop interventions.
All weights, ABC temporal filtering, streaming encoder, ODE, covariance and
measurement updates were shared. Only the multiplier on the SEAN graph
interaction increment changed: temporal=0, learned=predicted alpha, expert=1.

| mode | pos pred | vel pred | pos update | vel update | effective/temporal feature norm |
|---|---:|---:|---:|---:|---:|
| Temporal-only | 27.3059 | 6.5670 | 18.3410 | 4.8592 | 0.0000 |
| Learned SEAN | 27.4118 | 6.5763 | 18.4214 | 4.8693 | 0.1151 |
| Full graph expert | 27.5352 | 6.5884 | 18.5142 | 4.8816 | 0.2341 |

Learned SEAN worsened position prediction/update by 0.1058/0.0804 m relative
to Temporal-only. In strong windows, Temporal-only was 42.8151/26.7333 m and
learned SEAN was 43.0276/26.9106 m. Under the paired common-state comparison,
the full expert won only 43.77% of valid nodes overall and 40.04% in strong
windows; its position RMSE improvement was negative in every event class.
Learned alpha was nearly event invariant (steady 0.49106, switch 0.49110,
strong 0.49140).

Conclusion: this checkpoint falsifies, rather than verifies, the hypothesis
that SEAN is better than Temporal GASF. The monotone degradation from alpha 0
to learned alpha to alpha 1 identifies the graph expert increment as the first
failure; selector calibration is secondary. Do not train a larger SEAN
ablation from this design. Next modify/form the graph expert until alpha=1
beats alpha=0 in strong-window paired evaluation, then train the selector.
Raw result: `evaluation_results/sean_vs_temporal_dev8_epoch2.json`.

### 2026-08-09 graph-expert formation and strong-maneuver refinement

Problem: the original full graph expert was worse than Temporal GASF in every
event class.  A state-direction diagnostic showed that its effective state
correction was only about 0.05 m and initially pointed in the wrong direction.
The original graph-expert pilot also let the learned mixed branch drive the
online state history, so changing the expert changed the future residual and
the nominal temporal baseline at the same time.

Changes:

- Added a dedicated `graph_expert` stage that freezes Temporal GASF, the ODE,
  Q and activation head and trains only SEAN edge/node parameters plus the
  graph interaction projection (31,810 parameters).
- Fixed the online training history to alpha=0 while evaluating/training the
  alpha=1 same-state expert counterfactual.
- Replaced broad preprocessing-scale expert loss by task-scale physical loss
  (25 m position, 5 m/s velocity), with direct direction and one-sided
  improvement losses.
- Added strong-only expert/base loss, advantage logging and
  `expert_strong` checkpoint selection.
- Preserved the causal candidate graph and changed SEAN to a bounded residual
  reweighting (`candidate_weight * (0.5 + sigmoid(logit))`) rather than allowing
  a small-sample free sigmoid to suppress the graph.
- Added `evaluation/analyze_graph_prior_v3.py`.  On 32 training scenes and
  38,550 strong candidate pairs, maneuver-increment cosine had median 0.9996;
  among pairs within 100 m and 5 m/s, 99.25% had cosine > 0.5.  V9 therefore
  contains strong neighbour-consensus signal; lack of data signal is not the
  primary failure.

Training used 32 train / 8 fixed validation trajectories, batch 4 and complete
online trajectories.  A general expert stage (epochs 0--4), then strong-only
refinement at 2e-5 (epochs 5--6) and 1e-5 (epochs 7--8), reduced validation
strong advantage from -0.000821 to +0.000013 while temporal RMSE remained
27.305938 m.  The validation same-state direction loss fell to 0.2815 and the
overall node-level expert-better fraction reached 60.01%.

Strict alpha=1 closed-loop rollout at epoch 8 (6,148 valid points):

| event | temporal pos pred | expert pos pred | temporal pos update | expert pos update |
|---|---:|---:|---:|---:|
| all | 27.3059 | 27.2562 | 18.3410 | 18.3098 |
| steady | 19.5289 | 19.4151 | 14.4180 | 14.3469 |
| switch | 17.8935 | 17.7786 | 14.3898 | 14.3179 |
| strong | 42.8151 | 42.8425 | 26.7333 | 26.7596 |

Strong velocity prediction improved slightly (11.43542 -> 11.43470 m/s), but
strong position prediction/update remain worse by 0.0273/0.0263 m.  Therefore
the expert has become globally useful and its same-state strong correction now
beats the temporal counterfactual, but the stricter always-on strong-window
closed-loop position gate is not yet passed.  Do not claim final SEAN
superiority yet.  Next test should distinguish one more small strong-only
refinement from training the dynamic activation to prevent unnecessary
always-on accumulation.

Checkpoint:
`checkpoint/graph_expert_prior_strong_refine2_train32_val8_seed123/GNN_groupTargetTracking/best_expert_strong.pth`

Results:
`evaluation_results/graph_prior_stats_train32.json`,
`evaluation_results/graph_expert_strong_refine2_epoch8_rollout.json`.

## 2026-08-09: on-demand activation training started

Problem: the existing `gate` stage froze the correct activation head, but eval
mode did not request paired temporal/expert counterfactuals.  Its labels also
used the normalized four-state error although the current failure is position
rollout degradation.  A first real-data smoke with a 0.05 m dead zone exposed
another scale issue: the repaired expert's local position advantage is usually
millimetric, so every label was discarded and gate loss was exactly zero.

Changes and checks:

- Gate validation now always computes alpha=0 and alpha=1 from the same causal
  state; the online path keeps learned alpha.
- Only the 4,353 activation-head parameters train.  Frozen dropout is kept in
  eval mode, while the activation head remains differentiable.
- Gate labels use paired radial position error in metres, a 0.0001 m dead zone,
  balanced positive/negative BCE, event weights 1.0/1.5/2.0 and a mixed-path
  counterfactual regret penalty.
- Added class-count, balanced-accuracy and positive/negative activation-score
  diagnostics plus a dedicated `best_gate.pth` selection score.
- 30 unit tests pass.  A 2-train/2-val, 12-step V9 smoke had 98.8% supervised
  points, 40.7% positive labels, finite parameters, and changed exactly the
  four activation-head tensors; every frozen tensor remained bit-identical.

Active development run: 32 train / 8 fixed validation trajectories, full
causal supervision, batch 4, streaming encoder, AdamW peak LR 3e-4 with two
warmup epochs, 20-epoch cap and 6-epoch gate-score patience.  Epoch -1 position
prediction/update are 27.1788/18.2691 m; gate balanced accuracy is 0.5 and
activation mean is 0.4911, establishing the intended near-constant baseline.

Output: `checkpoint/activation_gate_position_train32_val8_seed123`  
Log: `log/activation_gate_position_train32_val8_seed123.log`

### Result (completed 2026-08-09)

Training stopped at epoch 8 after six epochs without gate-score improvement
(about 73 minutes).  `best_gate.pth` is epoch 3; the lowest position checkpoint
is epoch 8.  Training remained finite and the frozen temporal/expert weights
were unchanged, but the selector stayed effectively constant:

- validation balanced accuracy: 0.5000 at initialization, 0.5171 at epoch 8;
- activation mean/std: 0.4911/0.00010 initially, 0.4997/0.00013 at epoch 8;
- positive-minus-negative score separation: 0.000065 at epoch 8;
- epoch-8 event activation means (complete rollout): steady 0.49978, switch
  0.49970, strong 0.49943.  Strong demand therefore did not increase alpha.

Correct `concat` closed-loop comparison on the fixed eight validation
trajectories (`best_gate`, epoch 3):

| path | pos pred | pos update | strong pos pred | strong pos update |
|---|---:|---:|---:|---:|
| alpha=0 temporal | 27.180320 | 18.270029 | 43.228883 | 27.093594 |
| learned alpha | 27.178817 | 18.269069 | 43.225723 | 27.091354 |
| alpha=1 expert | 27.177257 | 18.267978 | 43.221289 | 27.087915 |

The learned path is nearly the midpoint between the two fixed branches.  It
does not outperform the full expert.  The graph expert helps globally and on
strong windows by only millimetres, while it worsens switch position prediction
by about 0.00323 m.  These branch differences are too small/noisy for the
current causal selector features and binary per-node labels to learn a robust
policy.  This is a valid negative development result, not evidence for
on-demand SEAN.

The earlier `graph_expert_strong_refine2_epoch8_rollout.json` used the
evaluator's old default `add` fusion.  The formal model uses `concat`; rerunning
the old checkpoint under `concat` exactly reproduces the current temporal
27.180320 m baseline.  Do not compare or cite the old 27.3059 m result.

Results: `evaluation_results/activation_gate_train32_best_gate_rollout.json`,
`evaluation_results/activation_gate_train32_best_position_rollout.json`, and
`evaluation_results/graph_expert_epoch8_temporal_recheck.json`.

### Fixed-alpha and oracle feasibility test (2026-08-09)

The fixed eight validation trajectories were re-evaluated under the formal
`concat` protocol with alpha 0/0.25/0.5/0.75/1. Two non-causal diagnostics
were added: a pointwise oracle on a fixed temporal history, and a greedy oracle
that rolls each selected prior, EKF posterior, covariance and innovation into
the next frame. The greedy result is not a global dynamic-programming bound.

| alpha | global pos pred | global pos update | steady pred | switch pred | strong pred |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 27.180320 | 18.270029 | 19.003844 | **17.261478** | 43.228883 |
| 0.25 | 27.179720 | 18.269705 | 19.003103 | 17.262392 | 43.227837 |
| 0.50 | 27.178809 | 18.269064 | 19.002672 | 17.263253 | 43.225703 |
| 0.75 | 27.177887 | 18.268423 | **19.002576** | 17.264286 | 43.223139 |
| 1.00 | **27.177255** | **18.267977** | 19.002700 | 17.264705 | **43.221286** |

The curve is nearly monotone globally and on strong windows, so learned
alpha~=0.5 is not an optimum; always-on is the best fixed choice. However,
alpha=1 improves global prediction by only 0.00307 m and strong prediction by
0.00760 m, while worsening switch prediction by 0.00323 m.

The fixed-history local oracle gives 27.179414/18.269582 m position
prediction/update. The closed-loop greedy oracle gives 27.171617/18.264937 m,
only 0.00870/0.00509 m better than temporal and 0.00564/0.00304 m better than
always-on. Even non-causal target access therefore exposes less than 1 cm of
global position opportunity. Do not spend more training budget on the current
activation head. The next diagnostic should scale the graph increment beyond
one and measure ODE/state sensitivity; the monotone [0,1] curve suggests
under-utilization by the frozen concat/ODE path rather than over-correction.

Result: `evaluation_results/activation_alpha_sweep_oracle_val8.json`.

## 2026-08-10: latent graph fusion root cause and physical repair

### Problem

The repaired latent graph expert still offered less than one centimetre of
closed-loop opportunity, so training a better alpha selector could not produce
a meaningful gain.  The key question was whether V9 lacks interaction signal,
or whether the network loses a useful signal while mapping it through the
latent GASF/ODE fusion path.

### Evidence and rejected explanations

- A causal direction diagnostic shows that the latest SEAN-weighted neighbour
  innovation is predictive of the required correction.  On val8--15 its mean
  cosine is 0.3422 and linear explained fraction is 0.3591; on strong nodes
  these rise to 0.4750 and 0.5794.  On val16--19 the independent cosine remains
  0.3092.  The dataset therefore contains usable graph-consensus signal.
- The old latent expert produces only about 0.024 m of same-state correction
  against an ideal correction near 21 m, with cosine about 0.017.  Expanding
  and retraining the latent graph projection did not generalize.
- Injecting projected neighbour innovation into the GASF latent, adding a
  direct ODE conditioning feature, and training a zero-initialized context
  bridge all failed closed-loop validation.  This rules out a simple lack of
  latent dimension or a single LayerNorm attenuation point.
- The failure is therefore primarily a representation/actuation mismatch:
  a physically directional residual is converted into a free latent vector
  and must be rediscovered by the ODE backbone.  Small useful directions are
  attenuated; unconstrained amplification destabilizes the recurrent rollout.

### Implemented repair

- In formal physical mode, GASF still performs causal temporal filtering and
  SEAN graph construction, but the graph-weighted latest innovation enters the
  ODE acceleration derivative directly.  The old latent graph increment is
  disabled in this mode, avoiding duplicate interaction injection.
- The scalar physical gain is learnable but bounded:
  `gain = 0.075 * sigmoid(raw_gain)`, initialized at 0.05.  A gain scan showed
  that 0.025--0.075 is stable and beneficial, while 0.10 is split-dependent and
  0.25 or larger can diverge in closed loop.
- A post-graph causal selector uses eight rotation-invariant magnitude,
  agreement and contrast features.  Offline cross-trajectory fitting on 32
  training trajectories reaches validation balanced accuracy 0.6228 and score
  separation 0.1227 on val48--63, proving that demand is predictable.  The
  prior online 8-trajectory selector failed because of insufficient diversity
  and a misleading sparse balanced-loss aggregation, not because the gate is
  intrinsically unpredictable.
- Position-first calibration uses
  `alpha' = 0.75 + 0.25 * alpha`.  Exact alpha=0/1 interventions bypass the
  formal floor, so temporal and expert ablations remain exact.  The floor,
  physical-gain settings and GASF fusion mode are now checkpoint metadata.
- Gate BCE sentinel zeros from time steps without both classes are excluded
  from the reported mean and gate checkpoint score.

### Independent closed-loop result

Protocol: V9 val80--95 (16 trajectories, never used for gate fitting or floor
selection), batch 1, streaming encoder, complete online prediction/update,
lifecycle-correct `real_mask`, and `concat` fusion.

| path | position prediction | position update | velocity prediction | velocity update |
|---|---:|---:|---:|---:|
| temporal, alpha=0 | 29.370698 | 20.682845 | 6.080806 | 4.665422 |
| full expert, alpha=1 | 26.500070 | 18.984977 | 5.502640 | 4.212708 |
| calibrated dynamic | **26.469845** | **18.952816** | **5.493294** | **4.197807** |

The calibrated dynamic path improves position prediction by 2.900853 m
(9.8767%) and update by 1.730028 m (8.3646%) versus temporal.  It also beats
the full expert globally by 0.030225/0.032161 m and improves both velocity
metrics.  Position prediction by event is 25.098460/22.492354/30.482509 m for
steady/switch/strong.  The strong subset is still 0.246 m worse than the fixed
expert, so the selector is a small global calibration improvement, not yet a
claim of per-event oracle behavior.

The formal checkpoint reproduces the old evaluation-floor result within
`2e-7 m`.  An initially observed 0.018 m discrepancy was traced to the old
sweep hardcoding `concat` while the comparison CLI defaulted to `add`; the
evaluation interface now reads fusion mode from checkpoint metadata and
records it in every result protocol.

Artifacts:

- formal checkpoint:
  `checkpoint/physical_context_gate_floor075_offline_train32_val48_63_seed123/best_model.pth`
- selector report:
  `evaluation_results/physical_context_gate_floor075_offline_pretrain.json`
- independent formal rollout:
  `evaluation_results/physical_context_gate_floor075_formal_concat_val80_95.json`
- temporal/expert baselines:
  `evaluation_results/physical_context_gate_baselines_val80_95.json`
- causal feature cache/diagnostic:
  `evaluation_results/activation_features_train32_val48_63.pt` and
  `evaluation_results/activation_predictability_train32_val48_63.json`

This is a validated mechanism-development result based on a smoke source
checkpoint, not a paper-ready full-data result.  Do not rewrite paper
conclusions yet.  The next formal experiment must train the temporal baseline,
fixed physical expert and calibrated dynamic model from scratch under one
fixed split/seed/protocol before attribution or significance claims.

## 2026-08-10: formal CV Expert and dynamic training started

The full V9 Temporal checkpoint can be reused, but the physical residual must
be recalibrated to that backbone.  A shared initialization replaces 151
compatible non-physical tensors with
`paper_a1_no_sean_gasf_lifecycle_fixed_converged_seed123/best_prediction.pth`
weights and retains 31 new SEAN/physical tensors from the validated mechanism.
Loading the resulting checkpoint has zero missing and zero unexpected keys.

An initial alpha=0/1 gate prevented an invalid long run: the old gain 0.05,
which was selected on a different development backbone, worsened val0--7
position prediction/update by 0.427/0.224 m.  Direct gain scans showed that the
graph direction remains useful but the scale changed:

- val0--7: gain 0 gives 19.61698/14.99983 m; gain 0.02 gives
  19.42126/14.89138 m position prediction/update.
- independent val8--15: gain 0 gives 23.80509/17.95714 m; gain 0.02 gives
  23.45106/17.80438 m.

The formal gain therefore initializes at 0.02 with a hard maximum of 0.075.
A real one-trajectory optimization smoke moved it to 0.0200088, kept all
tensors finite, trained exactly one parameter and saved the complete 200-val
protocol manifest.  Thirty-seven unit tests pass.

The formal sequential job started at 2026-08-10 10:39 CST.  Stage 1 trains the
fixed always-on Expert on all 1000 train / 200 validation trajectories, CV
covariance, batch 4, streaming encoder, all-time supervision, AdamW/cosine,
position checkpoint selection, 0.05 m early-stop delta, four-epoch patience
and a 12-epoch cap.  Only the bounded physical gain is trainable.  Stage 2
will collect causal counterfactual features once over 1000/200 trajectories,
fit the 8-32-1 dynamic selector offline, apply the prevalidated 0.75 activation
floor, and run a complete 200-trajectory temporal/dynamic/expert comparison.

Artifacts:

- shared initialization:
  `checkpoint/formal_cv_physical_shared_init_seed123/shared_init.pth`
- runner: `scripts/run_formal_cv_physical_models.ps1`
- active fixed-Expert log: `log/formal_cv_fixed_expert_full_seed123.log`
- gain scans:
  `evaluation_results/formal_cv_shared_init_gain_scan_val0_7.json` and
  `evaluation_results/formal_cv_shared_init_gain_scan_val8_15.json`

## 2026-08-10: full-data fixed Expert and dynamic selector result

The formal CV job finished normally. Fixed-Expert calibration used all
1000/200 train/validation trajectories and stopped after epoch 3 under the
0.05 m minimum-improvement, patience-4 policy. Its validation position
prediction/update are 21.268532/15.986454 m. The offline dynamic selector
reaches 0.585102 validation balanced accuracy and 0.052148 score separation.

Full 200-validation online rollout ranks position prediction as Dynamic
21.261715 m, Fixed Expert 21.268532 m, Temporal 21.621178 m. Position update
is Fixed Expert 15.986454 m, Dynamic 15.989510 m, Temporal 16.189333 m.
Dynamic activation is 0.872900 +/- 0.023338 and changes only from 0.869995 on
steady nodes to 0.876730 on strong nodes. Conclusion: the physical graph
expert yields a repeatable roughly 1.2--1.7% position benefit over Temporal,
but the selector is still nearly always-on; its global advantage over the
fixed expert is millimetric and it loses 0.029350/0.022978 m on strong-event
position prediction/update. Do not describe this as successful strong-event
activation without further evidence.

The selector was fitted/selected using validation data. Freeze all settings
and use IID-test 300 plus OOD-test 400 for unbiased paper evidence.

## 2026-08-10: whitened innovation and soft-graph decomposition prototype

Problem: the embedded SEAN graph expert remained largely redundant with the
Encoder GNN because both propagated generic hidden features. Raw residual
magnitudes also mixed maneuver evidence with measurement/prediction
uncertainty, so the same residual could have different statistical meaning.

Implemented on branch `feature/sggt-v2-whitened-innovation`:

- expose the EKF's physical innovation `r` and innovation covariance `S`;
- form the rotation-equivariant standardized innovation `S^{-1/2} r` and NIS,
  using a symmetric eigendecomposition, relative eigenvalue floor and clipping;
- maintain causal histories of raw/whitened innovation, `S`, NIS and validity;
  only continuing targets are valid, while births initialize the filter;
- use SEAN's label-free soft graph to compute an information-weighted common
  innovation from each target plus its neighbours, with the individual term
  defined by exact subtraction; isolated targets use common=0, individual=own;
- temporally filter common and individual standardized streams with the same
  `(a,b,c)` low-pass family, append uncertainty statistics, and fuse the result
  through a zero-initialized residual feature projection without a confidence
  gate or direct acceleration correction;
- add an `innovation_expert` stage that freezes Encoder, Temporal GASF, ODE and
  covariance paths and trains against the exact alpha=0 comparator.

Validation:

- 30 CPU unit tests pass, including all legacy V2 tests, NIS equivalence,
  rotation equivariance, singular-covariance safety, isolated-node behavior,
  exact reconstruction, graph-batch invariance and zero-initialized baseline
  preservation;
- real V9 two-step CPU and RTX 3060 forward/backward smoke tests pass;
- 32 train / 16 validation trajectories, batch 4, full-time online rollout,
  CV covariance, seed 123, Temporal checkpoint initialization, 10 expert
  epochs (first 5 at 1e-4, next 5 at 5e-5) complete without NaN;
- validation counterfactual expert loss decreases monotonically from 1.979537
  at epoch -1 to 1.966258 at epoch 9; strong-event loss decreases from
  1.504963 to 1.498874 and the expert-better fraction is 0.931630;
- independent forced-alpha closed-loop comparison on the same 16 development
  trajectories gives Temporal versus Expert position prediction
  31.403008/31.297400 m (improvement 0.105607 m) and position update
  19.986940/19.960078 m (improvement 0.026861 m). Strong-event prediction and
  update improve by 0.071559/0.019204 m.

Evidence boundary: this is a mechanism-development result on a selected 32/16
subset initialized from a previously converged Temporal checkpoint. It is not
paper-ready evidence, and no paper conclusion should be changed yet. Before
full training, add formal innovation/NIS diagnostics and freeze one protocol;
then run Temporal, innovation expert and no-SEAN controls with multiple seeds.

Artifacts:

- best epoch-9 expert:
  `checkpoint/innovation_decomp_selfinfo_train32_val16_continue10_seed123/GNN_groupTargetTracking/best_expert.pth`
- training logs:
  `log/innovation_decomp_selfinfo_train32_val16_seed123.log` and
  `log/innovation_decomp_selfinfo_train32_val16_continue10_seed123.log`
- closed-loop report:
`results/innovation_decomp_selfinfo_train32_val16_epoch9_compare.json`

## 2026-08-10: NIS calibration, innovation statistics and frozen formal comparison

Before any paper-scale retraining, the epoch-9 development Expert was rolled
out causally on all 200 validation trajectories (172,710 current valid
innovations).  The current two-dimensional EKF NIS has mean 2.280806 versus
the chi-square expectation 2, PIT-KS 0.039540, 95% coverage 0.925169 and 99%
coverage 0.978536.  Calibration is therefore usable but mildly
overconfident.  The mismatch is concentrated in strong maneuver samples:
mean NIS 2.545784, PIT-KS 0.085790 and 95% coverage 0.904016.  This is a
diagnostic of insufficient predictive uncertainty during maneuvers, not a
reason to add a direct NIS optimization term that could be satisfied by
inflating Q.

The strictly causal history ends at t-1.  Across all events, the
information-weighted group-common/individual NIS means are 8.433788/0.882905
and the common energy fraction is 0.484796.  For strong maneuvers these become
9.744200/0.922578 and 0.499374, so the shared innovation stream contains more
maneuver-dependent change than the individual stream.  Group NIS is not
compared to chi-square(2), because overlapping neighbor innovations are
correlated.  Exact common+individual reconstruction has maximum error
1.57284e-5 m.

A separate 16-trajectory graph-statistics pass found mean SEAN edge weight
0.684156, 8.508704 candidate neighbors and 8.495328 effective neighbors.
The near equality of candidate and effective counts shows that the current
development SEAN is close to dense, almost uniform averaging rather than a
sparse/selective graph.  Therefore the no-SEAN fixed-graph control is a
required formal comparison: if the full model does not beat it, improvement
cannot be attributed to learned SEAN structure.

Artifacts:

- `results/innovation_calibration_full_val200_epoch9.json`
- `results/innovation_calibration_val16_with_graph_stats.json`
- `evaluation/evaluate_innovation_calibration_v3.py`

The paper protocol is frozen in
`results/formal_innovation_protocol_v1.json`, SHA-256
`2b99127d5d53bd3362045ff20937223a072b9ffa7f5d77c2800538d550e5e381`.
It uses all 1000/200 train/validation trajectories, seeds 123/456/789,
full-time causal supervision, batch 4, streaming Encoder and CV covariance.
The three groups are A Temporal-only trained from scratch, B no-SEAN fixed
graph initialized from same-seed A, and C full SEAN whitened
group-common/individual innovation decomposition initialized from same-seed
A.  IID-test 300 and OOD-test 400 stay untouched until all validation-based
checkpoint selection is complete.

The first two detached launch attempts were reclaimed by the hosted process
job.  They reached at most 10% of epoch -1 validation, produced no checkpoint,
and were archived under
`checkpoint/_aborted_formal_innovation_v1_20260810_182225`.  The independent
formal runner was then started at 2026-08-10 19:36 CST with PID 26900 and
entered seed-123 A Temporal.  Its current training log is
`log/formal_innovation_v1_seed123_A_temporal.log`.

## 2026-08-12: formal run stopped and group-common maneuver oracle

At the user's request, the formal multi-seed runner (PID 26900) and its active
seed-456 Temporal child (PID 33984) were stopped before starting this
diagnostic.  No formal-training Python process remains.  Seed 123 A/B/C is
complete; seed-456 A is partial and its checkpoint/log are preserved only for
audit.  It must not be treated as a completed configuration or silently
skipped by a resumed sequential launcher.  The evaluation source was also
corrected to honor `args.gasf_graph_mode` instead of forcing `graph`, so the
old frozen source hash is no longer current.  Re-audit and re-freeze the
protocol before any formal run is resumed.

The new `evaluation/evaluate_group_maneuver_oracle_v3.py` rolls out the
converged seed-123 Temporal checkpoint causally and collects leave-one-out
signals.  Candidate topology is from t-1; the target's own future truth is
never included.  Other targets' truth at t is deliberately available only to
form a non-deployable oracle upper bound.  Synthetic true-group labels are a
second, theoretical ceiling; the geometric graph is the label-free topology
control.  A scalar or 2x2 map is calibrated on validation trajectories 0--99
and evaluated on the disjoint trajectories 100--199 (84,934 calibration and
87,776 held-out valid node-steps).  This avoids an in-sample oracle estimate.

The first full pass exposed lifecycle NaNs: multiplying a masked adjacency by
an invalid source value still evaluates `0 * NaN = NaN`.  The aggregator now
zeros invalid sources before matrix multiplication.  A regression test uses
NaN lifecycle padding explicitly.  Three `unittest` tests pass for structure
reconstruction, leave-one-out/lifecycle handling and the explicit 2x2 fit.
The corrected report contains no NaN or Infinity.  The corrected and original
Temporal baselines differ by only 0.000002 m, confirming that the fix did not
change model rollout.

Held-out Temporal baseline and oracle results:

| event | Temporal pos/vel RMSE | true-group delta-v pos gain | true-group delta-v vel RMSE | true-group shared-error pos gain | shared-error pos RMSE | shared-error vel RMSE |
|---|---:|---:|---:|---:|---:|---:|
| all | 17.6636 m / 3.1614 m/s | +0.0186 m | 2.8599 m/s | +1.8311 m (10.37%) | 15.8325 m | 1.5784 m/s |
| steady | 16.4876 / 2.2617 | +0.0262 m | 2.1284 | +0.7055 m (4.28%) | 15.7821 | 1.4962 |
| switch | 15.9080 / 2.5892 | +0.0661 m | 2.3660 | +1.4663 m (9.22%) | 14.4417 | 1.4744 |
| strong | 20.1817 / 4.3495 | -0.0100 m | 3.8618 | +3.4414 m (17.05%) | 16.7403 | 1.7586 |

The physical common maneuver is real: the leave-one-out true-group mean
delta-v explains 96.22% of own delta-v energy overall and 97.73% on strong
maneuvers; strong-event cosine similarity is 0.9563.  Nevertheless, a
held-out linear use of that delta-v barely changes position RMSE and slightly
worsens strong-event position.  Even the target's own future delta-v ceiling
has only +0.0150 m global position gain and -0.0276 m strong-event gain under
this simple state injection.  Therefore high maneuver coherence alone does
not validate direct acceleration/ODE injection.

The useful oracle variable is instead the targets' *shared prediction error*.
True-group peer error reduces position RMSE by 1.8311 m globally and 3.4414 m
on strong maneuvers.  Geometric peers already provide 1.4734/2.5113 m of
those gains.  Perfect true-group topology adds 0.3577 m globally and 0.9302 m
on strong events beyond geometry.  Thus this dataset does contain valuable
group-common information, and topology matters most during strong maneuvers;
the present failure is primarily that GASF/ODE targets a filtered motion
increment rather than estimating common prior error/innovation.  These are
oracle upper bounds using peer truth at t, not achievable model results and
not paper evidence.

Next decision gate: measure how much of this upper bound survives with only
strictly causal peer innovations through t-1 (and, separately, contemporaneous
peer measurements at t if the output is explicitly named group-aided
pre-update estimation rather than pure prediction).  If the lagged oracle
retains a material gain, replace direct delta-v fusion with a robust
whitened-common-innovation predictor and counterfactual Temporal protection.
If the lagged gain collapses, simultaneous group motion has no usable advance
warning for pure prediction; group information should then improve the
measurement update/uncertainty path rather than be claimed as a predictor.

Artifacts:

- formal report: `results/group_maneuver_oracle_val200_seed123.json`
- evaluator: `evaluation/evaluate_group_maneuver_oracle_v3.py`
- regression tests: `tests/test_group_maneuver_oracle.py`

## 2026-08-12: strictly causal group-innovation oracle

The next decision-gate experiment was completed without training a new model.
The same seed-123 epoch-48 Temporal checkpoint was rolled out online on all
200 validation trajectories.  For a prediction at t, every diagnostic feature
ends at t-1 after its measurement update.  The current measurement/truth at t
is not a feature.  Each historical peer consensus uses topology from the same
historical frame and excludes the target itself.  Synthetic true-group
topology remains an oracle ceiling; the geometric graph is the label-free
control.  Linear no-intercept ridge maps were calibrated on trajectories
0--99 and evaluated on trajectories 100--199.

The evaluator compares 1/4/8/16-frame raw and whitened innovation histories,
own-only, geometric-peer joint, true-group-peer joint and an event-stratified
shuffled-peer negative control.  To avoid lifecycle selection bias, its main
population metric always covers all 87,776 held-out target-steps: targets
without the requested history retain their unmodified Temporal prediction.
The aligned history coverage is 98.89/95.57/91.13/82.26% for 1/4/8/16 frames.
The saved 150.6 MB tensor cache permits further offline analysis without
another ten-minute rollout.

The best position-first setting is eight raw-innovation frames:

| held-out population | Temporal position RMSE | causal true-group joint | gain | 95% trajectory-bootstrap CI | trajectory win fraction |
|---|---:|---:|---:|---:|---:|
| all, 100 trajectories / 87,776 steps | 17.6636 m | 16.0390 m | 1.6246 m (9.20%) | [1.2979, 1.9763] m | 0.890 |
| strong, 70 trajectories / 28,822 steps | 20.1817 m | 17.6611 m | 2.5206 m (12.49%) | [1.8035, 3.2784] m | 0.914 |

This is genuinely group-derived rather than target autocorrelation.  With
eight frames, own raw innovation alone gains only 0.0051 m globally and loses
0.0919 m on strong events.  Adding correctly aligned true-group peers beyond
own history gains 1.6196 m globally and 2.6125 m on strong events.  The
event/validity-stratified shuffled-peer control gains only 0.0038 m globally
(95% CI crosses zero) and loses 0.0923 m on strong events.  Thus event-class
marginals and linear fitting do not explain the aligned result.

Geometry retains a real deployable-topology ceiling: eight-frame raw
own+geometric-peer history gains 1.0728 m globally, 95% CI [0.7065, 1.4098],
and 1.4012 m on strong events, CI [0.8057, 2.0040].  Perfect true-group
topology adds another 0.5518 m globally, CI [0.2648, 0.9018], and 1.1194 m on
strong events, CI [0.4121, 2.0026].  This independently supports learning a
selective group relation, especially for strong maneuvers.

Eight frames are preferred over four, despite the modest margin.  In a paired
trajectory bootstrap, eight raw frames improve over four by 0.1170 m globally
(95% CI [0.0210, 0.2040]) and 0.1352 m on strong events ([0.0041, 0.2506]).
Sixteen frames no longer help: relative to eight, the global change is
-0.0853 m (CI [-0.2131, 0.0236]) and strong-event change is significantly
-0.2062 m ([-0.3943, -0.0499]).

The strictly causal oracle retains about 88.7% of the former contemporaneous
peer-error upper bound globally (1.6246/1.8311 m) and 73.2% on strong events
(2.5206/3.4414 m).  Therefore useful group information is already available
before the current measurement; the innovation signal does not need to be
limited to the measurement-update claim.

Raw innovation is materially better for position than whitening alone:
eight-frame raw/whitened gains are 1.6246/1.0138 m globally and
2.5206/1.5067 m on strong events.  Whitened history has slightly better
velocity improvement in this linear diagnostic, so uncertainty must not be
discarded.  The next model should preserve raw physical amplitude and append
NIS/log-covariance as conditioning variables rather than replace raw values
with whitened values.

This also identifies why the present SEAN innovation expert underperformed.
Its `_decompose_innovation` mixes the target's own innovation into the common
term and defines individual as `own - common`; the successful oracle instead
keeps `own` and leave-one-out `peer consensus` as separate channels.  The
current expert then filters primarily group/individual whitened streams,
discarding the raw amplitude that carries the larger position signal.

Recommended minimum development experiment before any formal retraining:

1. freeze the converged Temporal backbone;
2. use exactly eight frames of raw own innovation and leave-one-out peer
   consensus as separate channels;
3. append whitened innovation, log-NIS/log-S only as uncertainty context;
4. predict a bounded normalized prior-state correction, not acceleration and
   not a confidence gate, with zero initialization preserving Temporal;
5. train first with the geometric graph and an exact alpha=0 counterfactual
   loss/performance guard; only after it beats Temporal test SEAN topology;
6. run a 32/16 development test and require positive global and strong-event
   closed-loop position gains before expanding.

Artifacts:

- `results/causal_group_innovation_oracle_val200_seed123.json`
- `results/causal_group_innovation_oracle_val200_seed123_bootstrap.json`
- `results/causal_group_innovation_oracle_val200_seed123_tensors.pt`
- `evaluation/evaluate_causal_group_innovation_oracle_v3.py`
- `evaluation/bootstrap_causal_group_innovation_oracle_v3.py`
- `tests/test_causal_group_innovation_oracle.py`
- `tests/test_causal_group_innovation_bootstrap.py`
## 2026-08-12 — 8-frame causal own/peer innovation corrector

### Problem

The causal oracle showed a reproducible upper bound from an 8-frame raw
innovation history, but the existing innovation decomposition mixed the target
into the common component and reused the latest graph over historical frames.
It therefore did not implement the oracle mechanism and could not isolate the
incremental value of group-common innovation.

### Measures

- Created branch `feature/sggt-v2-causal-innovation` from commit `e192aa6`.
- Added a strictly causal online buffer containing separate target innovation
  and same-frame, leave-one-out geometric-peer innovation.
- Added a 12,996-parameter corrector after the frozen Temporal prior and before
  the EKF measurement update. Raw own/peer channels are encoded independently;
  whitened innovation, NIS and innovation covariance are reliability context.
- Zero-initialized the output projection, bounded physical correction to 50 m
  and 5 m/s, and exposed an exact `causal_innovation_alpha=0` Temporal
  counterfactual.
- Kept Temporal GASF, encoder, ODE and Q frozen; covariance propagation is
  unchanged in this diagnostic phase.

### Verification

- 19 causal/whitening/lifecycle/oracle tests passed with `unittest`.
- Real V9 CUDA smoke passed without NaN. All 167 common non-corrector checkpoint
  tensors remained bitwise unchanged; only the corrector learned.
- 32-train/16-validation, full-time supervision, batch 4, CV covariance:
  epoch -1 Temporal prediction/update RMSE = 18.8564/14.9320 m; epoch 19 best =
  18.1651/14.5370 m. Prediction gain = 0.6913 m (3.67%), update gain = 0.3950 m.
  Velocity prediction/update also improved by 0.0273/0.0085 m/s.
- Same-checkpoint closed-loop alpha comparison on validation trajectories 0-15:
  strong/steady/switch prediction gains = 0.9697/0.5871/0.4515 m. Correction
  eligibility = 91.15%, RMS = 1.39 m, maximum = 3.20 m; no saturation.

### Next decision

Evaluate the unchanged best checkpoint on validation trajectories 16-47, which
were not used for checkpoint selection. Expand training only if alpha=1 retains
at least a 0.05 m prediction gain without degrading update or velocity metrics.

### Holdout and velocity-protection follow-up

- On untouched validation trajectories 16-47, alpha 0 -> 1 changed position
  prediction/update RMSE from 18.0622/14.3450 to 17.2609/13.8565 m. Strong and
  switch gains were 0.8911 and 1.0266 m. Velocity prediction/update regressed by
  0.0319/0.0400 m/s.
- A fixed alpha sweep exposed a monotone Pareto curve. Alpha 0.25 retained
  0.2923/0.1787 m position prediction/update gains with only 0.0011/0.0028 m/s
  velocity regression.
- Removing direct velocity correction made both position and velocity worse,
  showing that the learned velocity output partly compensates the EKF/rollout
  effect of the position correction.
- A low-LR one-step velocity counterfactual guard was rejected: on trajectories
  16-47 it marginally improved position by another 0.0133 m but increased the
  velocity regressions to 0.0459/0.0531 m/s. This local loss does not protect
  closed-loop velocity and remains disabled by default.
- Next controlled test: repeat 32/16 training from the same Temporal checkpoint
  with expert velocity weight 1.0 instead of 0.25, all other settings fixed.

### Equal physical-scale target result and expansion decision

- The equal-weight 32/16 run completed 20 epochs. On selection validation it
  improved position prediction/update from 18.8564/14.9320 to
  18.2087/14.5677 m and velocity prediction/update from 3.4114/3.1669 to
  3.3848/3.1606 m/s.
- On untouched trajectories 16-47, position prediction/update improved by
  0.7598/0.4624 m. Velocity still regressed by 0.0269/0.0359 m/s, but less than
  the position-prioritized objective. All automatic checkpoint criteria selected
  epoch 19, so this was not a checkpoint-selection artifact.
- The mechanism exceeds the 0.05 m expansion threshold on both selection and
  untouched trajectories. The formal protocol therefore uses equal physical
  position/velocity weights, starts the corrector from zero on the frozen
  Temporal checkpoint, disables the rejected local velocity guard, and reports
  the remaining position/velocity Pareto explicitly.

## 2026-08-12 ? Formal causal-innovation training launch

- Dataset sizes were verified from the real V9 directories: train/val/IID/OOD =
  1000/200/300/400 trajectories.
- Formal seed-123 configuration: all-time supervision, batch 4, streaming
  encoder, CV covariance, frozen Temporal checkpoint, 12,996 trainable corrector
  parameters, equal physical position/velocity task weights, LR 1e-4 with one
  warmup epoch and cosine decay, position checkpoint selection, 0.005 m internal
  plateau threshold and patience 4.
- The full 200-trajectory epoch -1 baseline completed at position prediction
  RMSE 18.378831 m, matching the previous Temporal reference 18.378819 m within
  1.2e-5 m. This verifies the formal protocol and exact alpha-zero degeneration.
- The managed long-running training cell is active and has entered epoch 0.

## 2026-08-23 — scene02 P4 precision and covariance diagnosis

### Problem

P4 was much worse than the old-paper SGGT result on the designed scene02, even
under Pd=0.99 and zero clutter. Candidate causes were overfitting, inconsistent
R/Q, BP/track-management errors, metric mismatch, and a scene motion tail.

### Verified causes

- BP inference multiplied neural measurement R by 100 although v3 training did
  not. R=1 reduced cluttered-protocol matched position RMSE from 27.68 to
  14.10 m over the same ten seeds.
- The formal P4 checkpoint uses legacy Q: its acceleration multiplier is always
  1 and its direct position-Q output is zero on scene02.
- BP `predict()` bypassed the process-noise selector and direct-position-Q path
  used by training `forward()`; this was repaired.
- Known-identity rollout still produced prediction/update/measurement RMSE of
  22.83/12.30/4.95 m, so BP is secondary to prior/confidence mismatch.
- Only 2.18% of scene02 states exceed the train-set maximum speed; acceleration
  and turn-rate support do not indicate a primary distribution failure.
- Epoch 13 and 19 were statistically indistinguishable on scene02, ruling out
  checkpoint selection as the large-error source.

### Counterfactual evidence

- Known-identity Q=1/8/128 position update RMSE was 12.22/6.99/5.03 m; velocity
  was best at Q=8 (4.81 m/s), demonstrating a position-velocity Pareto tradeoff.
- Corrected full BP Q=1/8/128 position RMSE was 11.45/7.19/5.72 m; mean OSPA
  was 11.33/7.46/6.27 m. Q=8 improved position in all ten paired runs.
- Fixed multipliers are diagnostic upper bounds, not a deployable adaptation.

### Measures and next task

- Default neural R scale is now 1; legacy R=100 remains explicit.
- BP neural NIS uses physically denormalized neural P rather than stale track.P.
- Added known-identity and Q/K/NIS scene02 evaluators; 14 BP tests pass.
- Next: train separate causal acceleration-Q and direct-position-Q heads using
  previous whitened innovation/log-NIS, calibration losses, and fixed-Q
  counterfactual baselines. Do not begin formal ablations until that mechanism
  is validated.

Detailed artifacts and tables are in
`results/scene02_diagnosis_20260823/DIAGNOSIS_SUMMARY.md`.

## 2026-08-23 - Causal adaptive-Q validation boundary

### Problem

The scene02 Q scan showed that larger process noise can improve position, but
that observation did not prove a causal time-varying Q mechanism. The required
test was whether a Q head driven only by lagged innovations/NIS could beat a
fixed Q with the exact same marginal mean.

### Measures

- Implemented separate causal acceleration-Q and direct-position-Q heads with
  trainable geometric parameterizations and strictly lagged innovation inputs.
- Added fixed-Q, centered response, event-conditioned Q response, and paired
  trajectory-bootstrap diagnostics.
- Trained two frozen-backbone 200/50 Q-only variants: positive surprise
  supervision and tracking-only calibration.
- Evaluated on independent `val[50:200]`, which was not used for checkpoint
  selection.

### Conclusion

- Positive surprise supervision learned the requested Q/NIS correlation but
  was significantly worse than its exact same-mean fixed-Q counterfactual.
- Removing surprise supervision substantially improved all tracking metrics by
  learning a better mean Q operating point, but its remaining time variation
  was again no better than exact same-mean fixed Q.
- Therefore the present lagged-NIS causal adaptive-Q mechanism is rejected.
  The reliable gain is covariance operating-point calibration, not dynamic
  maneuver adaptation. Do not make an adaptive-Q paper claim from these runs.

Full evidence is recorded in
`results/scene02_diagnosis_20260823/P4_q/CAUSAL_Q_VALIDATION_SUMMARY.md`.

## 2026-09-03 - Explicit association-context retention test

### Problem

P4 explicitly encodes BP detection mass, normalized association entropy, and
candidate-spread summaries, but earlier known-identity and standard-scenario
results did not show a stable motion-prediction gain over P5, which removes
those inputs. It was therefore unclear whether P5 could become the simpler full
model or whether association-context information was needed under missed
detections and clutter.

### Measures

- Froze a six-condition paired protocol before examining the new results:
  `Pd={0.99,0.90,0.80}` at clutter 3 and clutter `{2,10,15}` at `Pd=0.99`.
- Used the same 20 measurement seeds, fixed calibrated Q, fixed nominal tracker
  parameters, full 100-frame BP rollout, and frozen P4/P5 checkpoints.
- Reused three hash-validated raw reports and ran the nine missing cells. Total
  new evaluation time was 52 min 45 s; all stderr logs were empty.
- Verified a rerun against the old aggregate to sub-micrometre numerical
  tolerance and rejected any P4/P5 pair with mismatched seed or per-frame
  measurement-count sequence.
- Computed condition-wise paired bootstrap intervals and a six-condition
  equal-weight macro bootstrap with 20,000 samples.

### Conclusion

- Macro P5-minus-P4 differences were `+0.109 m` OSPA (95% CI
  `[-0.046,0.259]`), `-0.070 m` matched position
  (`[-0.181,0.040]`), and `+0.006 m/s` velocity
  (`[-0.072,0.084]`): none establishes a primary-metric P4 advantage.
- P4 has a very small cardinality-RMSE advantage of `0.023`
  (`[0.0004,0.045]`), but no primary metric is significantly better in the
  predeclared high-ambiguity conditions.
- The explicit association-context encoder is rejected as a main-model
  component. The P5 architecture should become the selected full model, so P5
  should disappear as an ablation label.
- This does not justify removing BP, soft associated measurements, innovations,
  covariance normalization, NIS, or validity. Only the explicit reliability
  feature encoder is rejected; adapter computations can be pruned only after a
  forward-equivalence test confirms they are unused.

Full evidence is in
`results/association_context_decision_20260903/DECISION.md`.
