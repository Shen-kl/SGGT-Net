# 论文正式结果审计（2026-08-19）

## 结论

当前 seed 123 的正式证据通过自动硬审计，可以用于撰写论文方法、实验和讨论初稿；但摘要与最终结论仍需等待核心 P2/P4 多随机种子验证。

审计程序为 `evaluation/build_paper_revision_artifacts.py`，机器可读审计记录为
`results/paper_revision_20260819/RESULT_AUDIT.json`，论文工作摘要为
`results/paper_revision_20260819/RESULT_AUDIT.md`。

## 已核验范围

- 22 份跟踪报告：P0--P5 的 Validation、IID、OOD，以及 P2--P5 的固定关联扰动闭环。
- 8 份 NIS 报告：P2/P4 在上述四类评估上的事件分层校准统计。
- 12 份轨迹级配对 Bootstrap 报告，共 240 个“对比--数据划分--事件--指标”统计项。
- 18 个 BP 条件汇总：P2/P4 各 9 个 Pd×clutter 条件，包含 54 个场景结果和 1080 次 Monte Carlo rollout。
- P2/P4 参数量与 5、10、20、40 目标受控复制下的推理时间和显存。
- 正式协议、原始报告、BP 场景源文件及所有引用 checkpoint 的 SHA-256。

自动审计检查了文件存在性、JSON 可解析性、全部数值有限性、轨迹数、事件有效点数守恒、关联协议标志、NIS 重建误差、Bootstrap 配对单位及样本数、BP rollout 数和文件/checkpoint 哈希。当前错误数为 0。

## 当前主要结果

P4 相对 P2 的位置预测 RMSE：Validation 从 18.378 m 降至 17.018 m，IID 从
17.965 m 降至 16.396 m，OOD 从 22.987 m 降至 20.755 m。在固定关联扰动闭环中，
位置预测 RMSE 从 25.255 m 降至 23.525 m。

P4 相对 P2 的位置更新 RMSE：Validation、IID、OOD 和关联扰动闭环分别改善
0.724 m、0.838 m、1.225 m 和 1.058 m。速度更新的改善明显更弱，Validation 上还存在
0.003 m/s 的轻微退化，必须如实报告。

九个 BP 条件宏平均中，P4 相对 P2 的匹配位置 RMSE 从 28.209 m 降至 27.814 m，
匹配速度 RMSE 从 7.943 m/s 降至 7.764 m/s，OSPA 从 63.431 降至 62.389；九个条件中
八个位置/OSPA 方向改善，Pd=0.75、clutter=30 是唯一退化条件。

## 论文解释边界

- P4 相对 P3 的差异支持群公共创新的主要增量作用。
- P4 相对 P5 的差异很小且方向混合，不能声称关联上下文已经带来稳定、显著的独立收益。
- BP 压力测试下的匹配状态 NIS 95% coverage 明显低于标称值，且 P4 宏平均略差；这属于必须讨论的校准局限，不得包装成改善。
- 结果文件中的 `f1=0` 是已禁用的旧结构监督指标，不是当前方法的有效性能指标，论文应删除结构 F1 叙述。
- 复杂度实验只覆盖模型预测接口，不包括模型加载、数据 I/O、BP 关联和量测更新，表注必须明确。
- 当前学习模型完整证据只有 seed 123。单种子结果足以确定章节结构和撰写初稿，不足以冻结摘要中的普遍性结论。

## 自动生成表格

- `tables/main_tracking.csv`：P0--P5 主跟踪指标。
- `tables/event_tracking.csv`：steady/switch/strong/unclassified 事件分层指标。
- `tables/p4_relative_gains.csv`：P4 相对各基线的绝对和相对增益。
- `tables/paired_bootstrap.csv`：轨迹级配对 Bootstrap、95% CI 和获胜比例。
- `tables/nis_calibration.csv`：当前/历史 NIS 的事件分层校准。
- `tables/bp_robustness.csv`：完整 Pd×clutter BP 压力测试。
- `tables/complexity_scaling.csv`：参数量、推理时间和显存。

以上相对路径均位于 `results/paper_revision_20260819/`。
