# ToolSched 实验整理

> **Placement 结果口径更新：** 旧版 `swe_terminal.placement.json` 使用相同的
> resource-class factor 同时生成策略排序与 synthetic oracle，因此 Top-1=1.000
> 是循环论证，已废弃。新版动作是带预启动 core/SMT/cluster 遥测的具体核心，
> 默认只评估真实 `placement_costs`；synthetic 仅作为独立、显式标注的压力测试。
> 方法与数据格式见 `docs/placement_design.md`。

数据来源：本页数字来自当前 `artifacts/` 下已经生成的实验结果，主结果优先使用
`agent_non_bfcl.*`，即 DeepResearchBench + SWE-ReBench + SWE-Bench Verified +
Terminal-Bench，且过滤 `duration_ms >= 1` 的真实工具调用。

## 1. 实验数据与切分

| 文件 | 训练样本 | 测试样本 | 切分方式 | 说明 |
|---|---:|---:|---|---|
| `artifacts/agent_non_bfcl.supervised.v4.json` | 12942 | 2858 | case holdout | 延迟分桶、next tool、延迟分位数 |
| `artifacts/agent_non_bfcl.metrics.json` | 12942 | 2858 | case holdout | 传统 quantile baseline、resource class |
| `artifacts/agent_non_bfcl.remaining.v3.json` | 12942 rows / 315 episodes | 2858 rows / 78 episodes | case holdout | Agent 剩余工具时间 |
| `artifacts/agent_non_bfcl.calibration.json` | - | 2858 | 测试集顺序 replay | 在线校准模拟 |
| `artifacts/agent_non_bfcl.speculation.json` | - | 2858 | 测试集 replay | speculative tool admission |

注意：当前 remaining-time 标签是 normalized tool-call 样本上的“剩余工具调用时间”，不包含 LLM 思考时间、排队时间、隐藏 harness overhead。

## 2. 当前包含的问题

| 问题 | 是否适合 ML | 当前方案 | Baseline | 输出 |
|---|---|---|---|---|
| 单次工具延迟分桶 `latency_bucket` | 是，有真实 `duration_ms` 标签 | Logistic Regression；Historical Random Forest | per-tool most-common bucket | 5 类时间桶 |
| 单次工具延迟分位数 `latency_quantiles` | 更适合统计画像，不是监督 ML | empirical P50/P90/P99 by group | global / per-tool / EWMA quantile | P50/P90/P99 |
| 下一步工具 `next_tool` | 是，有 trace 中的 `next_tool` 标签 | Logistic Regression | first-order Markov；history Markov | top-k next tool |
| Agent 剩余工具时间 `agent_remaining_time` | 部分可学，但噪声很大 | log-space Random Forest + OOB residual calibration | global quantile；step-conditioned；EWMA；compositional | P50/P90/P99 remaining ms |
| resource class | 暂不应当当作真实 ML | rule taxonomy | 无独立 telemetry label | resource class |
| placement | 目前是 evaluator/policy | synthetic 或已有 counterfactual cost 上选最小 regret | fixed placement policies | placement + regret |
| speculation | policy，不是训练模型 | next-tool confidence + latency cost rule | 无训练 baseline | admit/hit/waste/hidden latency |

## 3. 方法与公式

### 3.1 延迟分桶

标签：

`bucket(duration_ms) in {<100ms, 0.1-1s, 1-10s, 10-60s, >60s}`

Logistic Regression：

`P(y=k|x) = softmax_k(Wx + b)`

其中 `x` 包含 `dataset/tool/operation/tool_family/resource_class_hint`、命令长度、argv 数、flag 数、是否 pipe、是否递归、call index、history length、last 5 tools 等在线可见特征。

Historical Random Forest：

在上述特征外加入训练集统计先验：

`prior_bucket_i = count(bucket=i | tool or operation) / count(tool or operation)`

以及：

`prior_log_p50 = log(1 + P50(duration | group))`

`prior_log_p90 = log(1 + P90(duration | group))`

Baseline：

`y_hat = argmax_k count(bucket=k | tool)`

指标：

`accuracy = correct / n`

`adjacent_accuracy = mean(|y - y_hat| <= 1)`

`severe_underprediction = mean(y_hat <= y - 2)`

`long_task_recall = recall(y >= 10s as y_hat >= 10s)`

### 3.2 延迟分位数画像

Group empirical quantile：

`Q_q(g) = empirical_quantile({duration_i: group_i = g}, q)`

当前主模型使用：

`g = (operation, resource_class)`

并输出：

`p50 = Q_0.50(g), p90 = Q_0.90(g), p99 = Q_0.99(g)`

Pinball loss：

`L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))`

Coverage：

`coverage_p90 = mean(y <= p90)`

在线 tail calibration：

按 tool family 维护窗口内 P90 violation：

`violation_t = 1[y_t > p90_t]`

若窗口覆盖率低于目标，则放大 tail scale；若覆盖率高于目标太多，则回落：

`p90'_t = p90_t * scale_group`

### 3.3 下一步工具预测

Logistic Regression：

`P(next_tool=k|x) = softmax_k(Wx + b)`

特征：当前 `tool/operation/tool_family` 与最近工具历史。

First-order Markov baseline：

`P(next_tool=j | current_tool=i) = count(i -> j) / sum_j count(i -> j)`

History Markov baseline：

使用最长可匹配的最近工具 suffix：

`P(next | suffix) = count(suffix -> next) / sum count(suffix -> *)`

指标：

`top-k accuracy = mean(y in TopK(P(next_tool|x)))`

### 3.4 Agent 剩余工具时间

标签定义：在 episode 第 `t` 个工具完成后：

`remaining_t = total_tool_time_episode - cumulative_tool_time_t`

RandomForestRemainingRegressor：

训练目标：

`z_t = log(1 + remaining_t)`

模型：

`z_hat_t = RF(x_t)`

点预测：

`p50_t = exp(z_hat_t) - 1`

OOB residual calibration：

`r_t = z_t - z_hat_t^OOB`

`p90_t = exp(z_hat_t + scale * Q_0.90(r)) - 1`

`p99_t = exp(z_hat_t + scale * Q_0.99(r)) - 1`

特征：dataset/tool/operation/family/resource、step index、累计工具耗时、当前工具耗时、历史工具、工具多样性、命令结构特征等。不使用 `remaining_steps`、`total_time`、`next_tool` 等未来信息。

指标：

`MAE = mean(|y - y_hat|)`

`WAPE = sum(|y - y_hat|) / sum(|y|)`

`SMAPE = mean(|y - y_hat| / ((|y| + |y_hat|)/2))`

`R2 = 1 - SSE / SST`

### 3.5 Placement

当前 placement 不是从真实 counterfactual 数据学出的模型。若样本没有 `placement_costs`，使用 synthetic cost：

`cost(p) = base_latency * factor(resource_class, placement)`

选择：

`p_hat = argmin_p predicted_cost(p)`

Regret：

`regret = (cost(p_hat) - cost(p_oracle)) / cost(p_oracle)`

因此该部分只能验证 evaluator 和策略接口，不能声称模型已学到真实 placement 效果。

### 3.6 Speculation

Admission rule：

`hidden = min(predicted_p50_latency, llm_slack)`

`benefit = P(next_tool matches) * hidden`

`waste = (1 - P(next_tool matches)) * predicted_p50_latency`

`interference = 0.15 * predicted_p90_latency`

`cost = waste + interference`

`admit = is_safe_read_only and benefit > lambda * cost`

## 4. 主实验效果：agent_non_bfcl

### 4.1 延迟分桶

| 模型 | Accuracy | Macro F1 | Weighted F1 | Adjacent Acc | Long-task Recall >=10s | Severe Underprediction |
|---|---:|---:|---:|---:|---:|---:|
| Per-tool baseline | 0.653 | 0.455 | 0.622 | 0.918 | 0.584 | 0.049 |
| Logistic Regression | 0.701 | 0.590 | 0.709 | 0.907 | 0.646 | 0.040 |
| Historical Random Forest | 0.687 | 0.577 | 0.699 | 0.910 | 0.642 | 0.036 |
| Logistic + online calibration | 0.725 | - | - | - | 0.730 | 0.037 |
| Historical RF + online calibration | 0.735 | - | - | - | 0.758 | 0.036 |

结论依据：分桶问题是可学习的。Logistic offline accuracy 由 0.653 提升到 0.701；online calibration 后 Historical RF 到 0.735，long-task recall 从 0.642 到 0.758。代价是 adjacent accuracy 略低于 per-tool baseline，说明模型更积极地区分长尾类别。

### 4.2 每个桶的召回率

| Bucket | n | Per-tool baseline recall | Logistic recall | Historical RF recall |
|---|---:|---:|---:|---:|
| `<100ms` | 1280 | 0.940 | 0.864 | 0.843 |
| `0.1-1s` | 627 | 0.292 | 0.654 | 0.667 |
| `1-10s` | 389 | 0.602 | 0.599 | 0.589 |
| `10-60s` | 472 | 0.523 | 0.409 | 0.356 |
| `>60s` | 90 | 0.000 | 0.689 | 0.778 |

结论依据：ML 模型最大的价值是识别 `>60s` 极长工具调用；per-tool baseline 在该桶 recall 为 0。

### 4.3 延迟分位数画像

| 模型 | MAE ms | MAPE | P90 Coverage | P99 Coverage | Pinball P90 |
|---|---:|---:|---:|---:|---:|
| EWMA by tool | 11926 | 14.755 | 0.827 | 0.849 | 3873 |
| Global quantile | 7997 | 13.800 | 0.921 | 0.995 | 5349 |
| Per-tool quantile | 6534 | 4.452 | 0.900 | 0.983 | 3403 |
| Operation + resource quantile | 6645 | 6.820 | 0.908 | 0.985 | 3277 |
| Operation + resource + online calibration | 6242 | - | 0.955 | - | 8738 |

结论依据：延迟分位数画像不需要复杂 ML，group empirical quantile 已经能达到约 0.90 的 P90 coverage。在线 tail calibration 把 P90 coverage 从 0.908 提到 0.955，但 P90 pinball loss 从 3277 增到 8738，说明它更保守。

### 4.4 下一步工具预测

| 模型 | n | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|
| First-order Markov baseline | 2785 | 0.435 | 0.743 | 0.859 | 0.952 |
| Logistic Regression | 2785 | 0.455 | 0.792 | 0.888 | 0.964 |
| History Markov | 2785 | 0.420 | 0.654 | 0.699 | 0.715 |

结论依据：next-tool 有学习信号，但提升主要体现在 top-k。Top1 从 0.435 到 0.455，Top3 从 0.743 到 0.792。History Markov 在当前 case holdout 下不如 first-order Markov。

### 4.5 Agent 剩余工具时间

| 模型 | MAE ms | WAPE | SMAPE | R2 | P90 Coverage | P99 Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Global quantile | 287321 | 0.876 | 1.082 | -0.116 | 0.975 | 1.000 |
| Step-conditioned | 283894 | 0.865 | 1.066 | -0.111 | 0.963 | 1.000 |
| Random forest log | 258429 | 0.788 | 0.990 | 0.033 | 0.951 | 0.995 |
| EWMA family | 338087 | 1.031 | 1.385 | -0.347 | 0.406 | 0.443 |
| Compositional | 320239 | 0.976 | 1.674 | -0.380 | 0.297 | 0.388 |

结论依据：剩余时间可以学到一部分信号，但难度高。Random forest log 是唯一 R2 为正的模型，MAE 相比 global quantile 从 287.3s 降到 258.4s，WAPE 从 0.876 降到 0.788。P90/P99 coverage 也保持可用。不过 R2 只有 0.033，不能宣称高精度预测，只能说比强统计 baseline 有稳定改善。

### 4.6 Resource class

| 方法 | n | Accuracy |
|---|---:|---:|
| Rule taxonomy | 2858 | 1.000 |

结论依据：这个 1.000 不是监督模型效果，而是规则生成标签与规则预测之间的一致性。没有独立 telemetry label 前，不应作为 ML 准确率宣传。

### 4.7 Speculation

| n_candidates | n_admitted | Admission rate | Hit rate admitted | Estimated hidden latency | Estimated wasted CPU |
|---:|---:|---:|---:|---:|---:|
| 2858 | 115 | 0.040 | 0.600 | 75192.8 ms | 82802.3 ms |

结论依据：当前策略非常保守，只 admit 4.0% 的候选；admitted hit rate 为 60%。估计隐藏延迟与浪费 CPU 量级接近，因此这部分需要更好的 next-tool confidence 或更强安全/收益约束后才能作为优化主结论。

### 4.8 Placement

当前 `agent_non_bfcl` 没有 placement artifact。已有 `swe_terminal.placement.json`：

| n | Mode | Top1 | Normalized regret | Chosen placements |
|---:|---|---:|---:|---|
| 2707 | real_counterfactual_if_available_else_synthetic | 1.000 | 0.000 | compact_l3, spread_numa |

结论依据：因为模式允许 synthetic fallback，且 cost model 和 synthetic oracle 使用同一套 resource-class cost 因子，这个 1.000 只能说明 placement evaluator/接口可运行，不能证明真实 placement 预测已解决。

## 5. 子集对照

### 5.1 DeepResearchBench

| 问题 | 模型 | 指标 |
|---|---|---|
| latency bucket | baseline accuracy 0.926；Historical RF 0.857；Logistic 0.801 | baseline 更高 |
| latency bucket online | Historical RF 0.857 -> 0.908；Logistic 0.801 -> 0.904 | online calibration 有效但仍低于 baseline |
| next tool | baseline top1 0.538；Logistic top1 0.565 | 小幅提升 |
| quantile | P90 coverage 0.904；online 后 0.960 | coverage 更保守 |

解释：DeepResearchBench 测试集只有 272 个 latency bucket 样本，并且 bucket 分布偏集中，per-tool baseline 已经很强。因此不应把 agent_non_bfcl 上的 ML 改善直接泛化为所有子集都胜出。

### 5.2 SWE/Terminal

| 问题 | 结果 |
|---|---|
| latency bucket | Logistic accuracy 0.690 vs baseline 0.653；long-task recall 0.639 vs 0.584 |
| latency bucket online | accuracy 0.690 -> 0.718；long-task recall 0.639 -> 0.679 |
| next tool | Logistic top1 0.493 vs Markov baseline 0.473 |
| quantile | P90 coverage 0.887；online 后 0.949 |

解释：SWE/Terminal 与 agent_non_bfcl 主结论一致：ML 对 latency bucket 和 next-tool 有小到中等收益，online calibration 能提升 coverage 或分类稳定性。

## 6. 总结性结论

| 结论 | 数据依据 |
|---|---|
| 延迟分桶值得用 ML | agent_non_bfcl accuracy 0.653 -> 0.701/0.687，online 后到 0.725/0.735；`>60s` recall 从 0 到 0.689/0.778 |
| 延迟 P50/P90/P99 更适合统计画像 + 校准 | operation+resource quantile P90 coverage 0.908，online 后 0.955 |
| next-tool 可学但 Top1 提升有限 | Top1 0.435 -> 0.455，Top3 0.743 -> 0.792 |
| Agent remaining-time 只能说“有改善”，不能说高精度 | Random forest WAPE 0.788 vs global 0.876，R2 0.033 |
| resource_class 和 placement 不应包装成已验证 ML | resource_class 是规则标签；placement 使用 synthetic fallback |
| speculation 当前是可运行策略原型 | admission 4.0%，hit rate 60%，hidden/waste 量级接近 |
