下面按当前 `main` 分支解释。先给出最重要的结论：

**ToolSched 不是一个单一的“工具执行时间预测模型”，而是一个离线实验框架。**它把 Agent trace 转换成统一样本，然后分别研究：

1. 当前工具调用属于哪个时延区间；
2. 下一步可能调用什么工具；
3. 当前工具结束后，Agent 还剩多少工具执行时间；
4. 如何根据预测结果进行核放置；
5. 是否值得投机执行下一工具；
6. 模型部署到新机器、不同负载环境后，如何在线校准。

其中只有一部分是机器学习；资源分类、放置和投机准入主要是手工规则与决策模型。项目目前也没有真正接入 Agent runtime，而是离线读取 trace、训练、评估和回放在线校准。([GitHub][1])

---

# 一、统一数学表示：ToolSched 的一个样本是什么

假设某个 Agent 的工具调用序列为

[
\mathcal E=
\left(
a_0,a_1,\ldots,a_{n-1}
\right).
]

第 (i) 个工具调用记作

[
a_i=(x_i,d_i,h_i,y_i),
]

其中：

* (x_i)：执行该工具之前可以观察到的信息，即特征；
* (d_i)：该工具调用的实际执行时间，单位为毫秒；
* (h_i=(t_{i-k},\ldots,t_{i-1}))：最近 (k) 个历史工具；
* (y_i=t_{i+1})：下一工具标签。

代码中的 `ToolSample` 保存以下主要字段：

[
\begin{aligned}
s_i = (&\text{dataset},
\text{case},
\text{attempt},
\text{tool},
\text{operation},
\text{family},\
&\text{duration},
\text{features},
\text{labels},
\text{resources},
\text{history},
\text{next_tool}).
\end{aligned}
]

其中 `dataset/case/attempt` 共同标识一次完整 Agent 运行，`duration_ms` 是监督标签，`history` 默认只保留最近五个工具。([GitHub][2])

---

# 二、`tool`、`operation` 和 `tool_family` 分别是什么

这是理解项目最关键的三个概念。

| 层次               | 含义               | 示例          |
| ---------------- | ---------------- | ----------- |
| `tool`           | trace 中原始工具名称    | `exec-grep` |
| `operation`      | 从工具名和命令中识别出的实际操作 | `grep`      |
| `tool_family`    | 更粗粒度的工具类别        | `search`    |
| `resource_class` | 对主要资源需求的规则判断     | `io_search` |

例如，Agent 执行：

```text
tool = exec-command
command = rg --recursive "ToolSample" src/
```

经过归一化后可能得到：

[
\text{tool}=\texttt{exec-command},
\quad
\text{operation}=\texttt{grep},
\quad
\text{tool_family}=\texttt{search},
\quad
\text{resource_class}=\texttt{io_search}.
]

## 2.1 operation 的识别规则

项目会读取命令文本，根据关键词进行归一化：

* 包含 `pytest`、`tox`、`unittest`：测试操作；
* 包含 `grep`、`rg`、`find`：搜索操作；
* 包含 `git diff`、`git status`：Git 操作；
* 包含 `make`、`cmake`、`ninja`、`npm`、`cargo`：构建操作；
* 包含 `python`：Python 执行；
* 包含 `curl` 或 `wget`：下载操作；
* 其他 `exec-*` 工具则去掉 `exec-` 前缀。

这样做的目的是把不同 Agent harness 中名字不同但语义相近的工具合并起来。([GitHub][3])

## 2.2 tool_family 的划分

当前代码把工具粗略划分为：

[
\mathcal F=
{
\text{file},
\text{network},
\text{search},
\text{test},
\text{terminal},
\text{control},
\text{tool}
}.
]

例如：

* `read_file/write_file/edit_file/list_dir` 属于 `file`；
* `web_search/web_fetch/download` 属于 `network`；
* `grep/find` 属于 `search`；
* `pytest/test` 属于 `test`；
* `build/python/exec-*` 属于 `terminal`；
* `ls/cd/pwd` 属于 `control`。([GitHub][3])

## 2.3 resource_class 不是机器学习标签

`resource_class` 当前完全由规则产生：

[
r_i=\phi(\text{family}_i,\text{operation}_i,x_i).
]

具体包括：

* 测试：`cpu_memory_mixed`；
* 构建和 Python：`cpu`；
* 递归搜索：`io_search`；
* 普通搜索：`search`；
* 文件操作：`file_io`；
* 网络操作：`network`；
* 控制操作：`light_control`；
* 其他：`unknown`。

因此，当前所谓“资源分类准确率”并不是模型和独立真实性能计数器标签之间的准确率，而主要是在验证规则本身是否被一致复现。([GitHub][4])

---

# 三、项目究竟设计了哪些特征

特征可以分成五组。

## 3.1 工具身份特征

这些离散变量采用 one-hot 编码：

[
x_i^{\text{id}}=
[
\mathbf 1(\text{dataset}=d),
\mathbf 1(\text{tool}=t),
\mathbf 1(\text{operation}=o),
\mathbf 1(\text{family}=f),
\mathbf 1(\text{resource}=r)
].
]

它们分别回答：

* 数据来自哪个 benchmark；
* 原始工具叫什么；
* 实际执行什么操作；
* 属于哪类工具；
* 预期主要消耗哪种资源。

例如 `tool=exec-grep` 会对应一个独立维度，该维度取 1，其他工具维度取 0。([GitHub][5])

## 3.2 命令复杂度特征

代码会从命令或参数中提取：

| 特征                   | 含义                     |   |
| -------------------- | ---------------------- | - |
| `command_len`        | 命令文本字符数                |   |
| `argv_count`         | 命令参数数量                 |   |
| `flag_count`         | `-r`、`--include` 等选项数量 |   |
| `has_pipe`           | 是否包含管道 `               | ` |
| `has_recursive_hint` | 是否可能递归扫描               |   |
| `include_count`      | `--include` 出现次数       |   |
| `path_token_count`   | 命令中路径形式 token 数量       |   |
| `input_key_count`    | 工具输入字典字段数量             |   |
| `preview_len`        | 返回结果预览字符数              |   |
| `preview_line_count` | 返回结果预览行数               |   |
| `exit_code_nonzero`  | 是否疑似失败                 |   |

这些是低成本、执行前后容易获取的结构化特征。([GitHub][3])

需要注意：**提取了不等于所有模型都使用了。**

主时延分档和下一工具模型实际使用的数值特征只有：

[
\begin{aligned}
&\text{command_len},
\text{argv_count},
\text{flag_count},
\text{include_count},\
&\text{path_token_count},
\text{input_key_count},
\text{call_index},
\text{history_len},
\end{aligned}
]

以及 `has_pipe` 和 `has_recursive_hint`。例如 `preview_len` 虽然被提取，但不进入主 `SampleFeatureEncoder`。([GitHub][5])

## 3.3 对数变换

对于非负数值特征 (v)，代码使用

[
\tilde v=\log(1+\max(v,0)).
]

这样做是因为命令长度、调用序号等变量可能具有长尾分布。对数变换可以压缩极端大值，例如：

[
10,\ 100,\ 10000
\quad\longrightarrow\quad
\log 11,\ \log 101,\ \log 10001.
]

于是一个非常长的命令不会在数值尺度上完全支配模型。([GitHub][5])

## 3.4 历史工具序列特征

默认保留最近五个工具：

[
h_i=(t_{i-5},\ldots,t_{i-1}).
]

编码形式为位置相关的 one-hot：

[
\mathbf 1(\text{prev1}=t),
\mathbf 1(\text{prev2}=t),
\ldots,
\mathbf 1(\text{prev5}=t).
]

其中：

* `prev1` 表示紧邻当前工具的上一个工具；
* `prev2` 表示倒数第二个历史工具；
* `latest_tool` 与 `prev1` 基本重复。

位置相关编码使模型能够区分：

[
(\texttt{read},\texttt{grep})
\quad\text{和}\quad
(\texttt{grep},\texttt{read}).
]

([GitHub][5])

## 3.5 调用进度特征

每个调用还包含：

[
\text{call_index}=i,
\qquad
\text{history_len}=\min(i,5).
]

`call_index` 表示这是当前 episode 中第几个工具调用。它提供了非常粗糙的 Agent 进度信息。([GitHub][6])

---

# 四、数学问题一：单次工具时延分档

项目没有首先直接预测精确毫秒数，而是定义五个时延区间：

[
B(d)=
\begin{cases}
0,&0\le d<100\text{ ms},\
1,&100\le d<1000\text{ ms},\
2,&1\le d<10\text{ s},\
3,&10\le d<60\text{ s},\
4,&d\ge 60\text{ s}.
\end{cases}
]

因此监督学习问题是：

[
f_\theta(x_i)
\longrightarrow
P(B(d_i)=k\mid x_i),
\quad k\in{0,1,2,3,4}.
]

它回答的不是“需要 12.7 秒”，而是：

> 这个调用更可能小于 100 ms，还是超过 10 秒？

这种粗分类更适合准入、抢占、投机和放置等离散调度决策。([GitHub][7])

## 4.1 基线：每个工具最常见区间

对每个工具 (t)，统计其训练样本的分档众数：

[
\hat k_t
========

\arg\max_k
N(t,k),
]

其中 (N(t,k)) 表示工具 (t) 落入区间 (k) 的训练样本数。

预测时：

[
\hat B_i=
\begin{cases}
\hat k_{t_i},&t_i\text{ 在训练集中出现过},\
\hat k_{\text{global}},&\text{否则}.
\end{cases}
]

这不使用命令长度、历史序列等信息，只记住每个工具通常有多慢。([GitHub][7])

## 4.2 多分类 Logistic Regression

模型为 softmax 分类器：

[
z_{i,k}=w_k^\top x_i+b_k,
]

[
p_{i,k}
=======

# P(B_i=k\mid x_i)

\frac{\exp(z_{i,k})}
{\sum_{j=0}^{4}\exp(z_{i,j})}.
]

预测类别为：

[
\hat k_i=\arg\max_k p_{i,k}.
]

训练目标是加权交叉熵：

[
\mathcal L(\theta)
==================

-\sum_{i=1}^{N}
\omega_{y_i}
\log p_{i,y_i}
+
\lambda\sum_k|w_k|_2^2.
]

其中 (\omega_{y_i}) 用于平衡不同区间的样本量。否则，由于短调用通常较多，模型可能始终预测最短区间。代码使用 `class_weight="balanced"`、(C=1) 和最多 2000 次迭代。([GitHub][7])

## 4.3 加历史先验的 Random Forest

这个模型首先从训练集构造三类历史统计：

1. 全局统计；
2. 同一 `tool` 的统计；
3. 同一 `operation` 的统计。

对于分组 (g)，计算每个区间的频率：

[
\pi_{g,k}
=========

\frac{N_{g,k}}{N_g},
]

以及：

[
\log(1+Q_{0.5}(D_g)),
\qquad
\log(1+Q_{0.9}(D_g)),
\qquad
\log(1+\operatorname{mean}(D_g)).
]

这些统计量作为额外特征加入 (x_i)。

例如对 `pytest` 操作，模型可能看到：

[
\begin{aligned}
P(<100\text{ms})&=0.01,\
P(0.1\text{-}1\text{s})&=0.04,\
P(1\text{-}10\text{s})&=0.25,\
P(10\text{-}60\text{s})&=0.55,\
P(>60\text{s})&=0.15.
\end{aligned}
]

然后使用 120 棵决策树组成 Random Forest：

[
\hat p_k(x)
===========

\frac{1}{M}
\sum_{m=1}^{M}
p_k^{(m)}(x),
\qquad M=120.
]

树的最大深度为 14，叶节点至少包含 8 个训练样本。([GitHub][7])

该模型的实质是：

> 不仅知道这是 `pytest`，还知道历史上 `pytest` 通常落在哪些时延区间。

但当前先验由整个训练集计算，没有使用 leave-one-out 方式，因此训练样本的先验统计中包含它自身；测试样本仍然只使用训练集统计。

---

# 五、数学问题二：经验分位数时延模型

这是项目里最直接的“工具 cost distribution”。

对于某个分组

[
g=(\text{operation},\text{resource_class}),
]

收集训练时延：

[
D_g={d_i:g_i=g}.
]

输出：

[
\hat Q_{0.5}(g),
\qquad
\hat Q_{0.9}(g),
\qquad
\hat Q_{0.99}(g).
]

分别对应：

* P50：中位数；
* P90：约 90% 的调用不超过该值；
* P99：约 99% 的调用不超过该值。

代码使用线性插值经验分位数。设排序后数据为

[
d_{(0)}\le d_{(1)}\le\cdots\le d_{(n-1)},
]

位置为

[
u=(n-1)q,
\qquad
l=\lfloor u\rfloor,
\quad
h=\min(l+1,n-1),
]

则

[
Q_q(D)
======

(1-u+l)d_{(l)}
+
(u-l)d_{(h)}.
]

若测试时出现从未见过的分组，则退化为全局分位数。代码还把

[
U=Q_{0.9}-Q_{0.5}
]

作为一个简单不确定性指标。([GitHub][8])

这不是监督机器学习模型，因为它没有参数化函数

[
f_\theta(x)\to y;
]

它只是根据分组查询历史统计。

---

# 六、数学问题三：下一工具预测

设当前工具为 (t_i)，最近历史为

[
h_i=(t_{i-k},\ldots,t_{i-1}),
]

目标是预测：

[
t_{i+1}.
]

即：

[
P(t_{i+1}=v\mid t_i,h_i,x_i).
]

这个预测可以用于投机执行：当 LLM 还在生成下一步工具调用时，系统提前启动一个高概率工具。

## 6.1 一阶 Markov 模型

仅考虑当前工具：

[
P(t_{i+1}=b\mid t_i=a)
======================

\frac{N(a\rightarrow b)}
{\sum_cN(a\rightarrow c)}.
]

预测：

[
\hat t_{i+1}
============

\arg\max_b
P(b\mid t_i).
]

例如训练数据中：

[
\texttt{edit_file}
\rightarrow
\begin{cases}
\texttt{pytest}:70%,\
\texttt{read_file}:20%,\
\texttt{git_diff}:10%,
\end{cases}
]

则当前工具为 `edit_file` 时预测下一工具是 `pytest`，置信度为 0.7。([GitHub][8])

## 6.2 最长后缀 History Markov

考虑最近最多五个工具：

[
s_i=(t_{i-k},\ldots,t_i).
]

模型从最长历史开始查找：

[
P(t_{i+1}\mid t_{i-k},\ldots,t_i).
]

如果完整五阶历史没有出现过，则退化为四阶、三阶，直到一阶：

[
\begin{aligned}
&P(t_{i+1}\mid t_{i-4},\ldots,t_i),\
&\downarrow\
&P(t_{i+1}\mid t_{i-3},\ldots,t_i),\
&\downarrow\
&P(t_{i+1}\mid t_i).
\end{aligned}
]

这类似一个变阶 n-gram 模型。([GitHub][9])

## 6.3 下一工具 Logistic Regression

输入仍然是前面介绍的结构化特征：

[
x_i=
[
\text{当前工具},
\text{operation},
\text{family},
\text{命令特征},
\text{调用位置},
\text{最近工具历史}
].
]

对每个候选下一工具 (v)，计算：

[
P(t_{i+1}=v\mid x_i)
====================

\frac{\exp(w_v^\top x_i+b_v)}
{\sum_u\exp(w_u^\top x_i+b_u)}.
]

代码只把训练集中出现次数不少于 5 次的下一工具作为 Logistic Regression 类别。低频工具不会成为该模型的输出类别；如果整个数据中不足两个常见类别，才整体退化到 Markov 模型。([GitHub][9])

---

# 七、数学问题四：Agent 剩余时间预测

这是最容易被误解的部分。

当前实现预测的不是完整 Agent 端到端剩余时间，而是：

> 当前工具调用完成后，当前 episode 中后续所有**已观察工具调用时延之和**。

它不包括：

* 后续 LLM 推理时间；
* GPU 排队时间；
* CPU 调度等待；
* Agent harness 内部开销；
* 工具之间未记录的空闲时间。

README 也明确说明，剩余时间标签是由归一化工具调用样本构造的，而非完整 wall-clock Agent 时间。([GitHub][1])

## 7.1 标签定义

一个 episode 按

[
(\text{dataset},\text{case},\text{attempt})
]

分组，并按照 `call_index` 排序。

假设工具时延为：

[
d_0,d_1,\ldots,d_{n-1}.
]

当前工具 (i) 已执行完成时：

[
C_i=\sum_{j=0}^{i}d_j
]

是累计工具时间，

[
R_i=\sum_{j=i+1}^{n-1}d_j
]

是剩余工具时间，

[
M_i=n-1-i
]

是剩余工具调用数。

并且：

[
T=C_i+R_i=\sum_{j=0}^{n-1}d_j.
]

因此模型输入点是**当前工具执行完成后**。所以当前工具时延 (d_i) 是可用特征。([GitHub][10])

---

# 八、默认运行的剩余时间模型

当前 `evaluate-remaining` 默认运行五类模型：

1. 全局分位数；
2. 按步骤位置分组的分位数；
3. 对数空间 Random Forest；
4. 按工具家族的 EWMA；
5. 工具序列分解模型。

其他线性分位数、进度分组、剩余步数分类等模型虽然已经实现，但 CLI 默认不会运行。([GitHub][11])

## 8.1 全局剩余时间分位数

忽略当前状态，对所有训练步骤的 (R_i) 计算：

[
\hat R_{0.5}=Q_{0.5}({R_i}),
]

[
\hat R_{0.9}=Q_{0.9}({R_i}),
]

[
\hat R_{0.99}=Q_{0.99}({R_i}).
]

所有测试步骤都输出相同结果。它的作用不是获得高精度，而是提供最弱基线。([GitHub][12])

## 8.2 Step-conditioned 分位数

按当前步骤编号分组：

[
g_i=\min(i,50).
]

在第 (i) 步预测：

[
\hat R_{i,q}
============

Q_q\left(
{R_j:\min(j,50)=g_i}
\right).
]

例如：

* 第 0 次工具调用结束后，参考所有训练 episode 的第 0 步；
* 第 5 次工具调用结束后，参考所有第 5 步；
* 第 50 步以后全部放在同一组。

它隐含假设：

> Agent 执行到相同工具步数时，剩余工作量具有相似分布。

([GitHub][12])

## 8.3 对数空间 Random Forest

这是当前默认剩余时间模型中最主要的学习模型。

目标先变换为：

[
z_i=\log(1+R_i).
]

模型学习：

[
\hat z_i=f_{\mathrm{RF}}(x_i).
]

最终点预测为：

[
\hat R_i=\exp(\hat z_i)-1.
]

采用对数空间是因为剩余时间可能从几毫秒到几千秒，跨度非常大。

### 使用的特征

Random Forest 使用：

[
\begin{aligned}
x_i={&
\text{dataset},
\text{tool},
\text{operation},
\text{family},
\text{resource},\
&\log(1+i),
\log(1+C_i),
\log(1+d_i),\
&\log\left(1+\frac{C_i}{i+1}\right),\
&\log(1+\text{command length}),
\log(1+\text{argument count}),\
&\text{pipe},
\text{recursive},
\text{history length},
\text{recent tool diversity},
\text{last five tools}
}.
\end{aligned}
]

其中：

[
\frac{C_i}{i+1}
]

表示当前为止每个工具的平均执行时间。

模型参数为：

* 80 棵树；
* 最大深度 14；
* 最小叶节点样本数 6；
* 启用 out-of-bag 预测。([GitHub][12])

严格来说，代码把该点预测命名为 `p50`，但模型优化的是 log-space Random Forest 回归误差，并不是直接优化中位数或 pinball loss，因此它不是严格意义上的条件 P50。

## 8.4 按工具家族的 EWMA

对每个工具家族 (f)，维护一个状态：

[
m_{f,t}.
]

看到一个训练样本的真实剩余时间 (R_t) 后更新：

[
m_{f,t+1}
=========

(1-\alpha)m_{f,t}
+
\alpha R_t,
\qquad
\alpha=0.25.
]

预测为：

[
\hat R_{0.5}=m_f,
\qquad
\hat R_{0.9}=1.5m_f,
\qquad
\hat R_{0.99}=2m_f.
]

后两个分位数只是手工倍数，并非通过统计覆盖率学习得到。当前评估流程是在训练集上依次更新 EWMA，然后固定状态预测测试集，并没有在测试期间继续在线更新。([GitHub][12])

## 8.5 Compositional 分解模型

该模型不直接回归 (R_i)，而是分解为：

1. 下一工具是什么；
2. 该工具要执行多久；
3. 沿预测序列累加时延。

首先学习一阶工具转移概率：

[
P(t_{j+1}=b\mid t_j=a).
]

同时为每种工具计算：

[
Q_{0.5}(D_t),
\quad
Q_{0.9}(D_t),
\quad
Q_{0.99}(D_t).
]

然后从当前工具出发，最多 rollout 10 步。每一步只选择概率最大的下一工具：

[
t_{h+1}
=======

\arg\max_vP(v\mid t_h).
]

代码的预测近似为：

[
\hat R_q
========

\sum_{h=1}^{H}
\gamma^{h-1}
P(t_h\mid t_{h-1})
Q_q(D_{t_h}),
]

其中

[
H=10,
\qquad
\gamma=0.95.
]

需要注意，这不是完整地对所有可能工具路径求期望，因为每一步只保留概率最大的下一工具，然后再乘它的概率。若某一步没有已知转移，则使用全局时延分位数填满剩余 rollout 深度。([GitHub][12])

---

# 九、已经实现但默认不运行的剩余时间模型

## 9.1 线性分位数回归

对每个分位数 (q\in{0.5,0.9,0.99})，学习：

[
\hat R_{i,q}=w_q^\top\tilde x_i+b_q.
]

损失是 pinball loss：

[
\rho_q(e)=
\begin{cases}
qe,&e\ge 0,\
(q-1)e,&e<0,
\end{cases}
\qquad
e=R_i-\hat R_{i,q}.
]

完整目标为：

[
\min_{w_q,b_q}
\sum_i
\rho_q(R_i-w_q^\top\tilde x_i-b_q)
+
\lambda|w_q|_2^2.
]

P90 模型会对低估施加更大惩罚，因此倾向于输出一个能覆盖约 90% 样本的上界。代码使用 mini-batch 梯度下降，并同时标准化输入和目标。([GitHub][12])

## 9.2 对数线性回归

学习：

[
\log(1+R_i)=w^\top\tilde x_i+b+\epsilon_i.
]

预测：

[
\hat R_i=\exp(w^\top\tilde x_i+b)-1.
]

但 P90 和 P99 是：

[
\hat R_{0.9}=2\hat R,
\qquad
\hat R_{0.99}=4\hat R,
]

仍是手工倍数。([GitHub][12])

## 9.3 Progress-conditioned 模型

真实进度定义为：

[
p_i=\frac{C_i}{T}.
]

训练时知道完整 episode 的 (T)，先将样本按真实进度分成十组；同时训练一个线性模型：

[
\hat p_i=w^\top x_i+b.
]

推理时根据预测进度 (\hat p_i) 查询对应组内的剩余时间分位数。该方法训练分组依赖完整 episode 的 oracle 总时间，但在线推理不直接使用真实总时间。([GitHub][12])

## 9.4 剩余时间区间分类

将剩余时间分成：

[
\le15s,\quad
15\text{-}60s,\quad
60\text{-}180s,\quad
180\text{-}600s,\quad
600\text{-}1800s,\quad

> 1800s.
> ]

使用 softmax 分类：

[
P(c=k\mid x)
============

\frac{\exp(w_k^\top x+b_k)}
{\sum_j\exp(w_j^\top x+b_j)}.
]

再用每个类别中剩余时间的中位数 (m_k) 转成点预测：

[
\hat R=\sum_kP(c=k\mid x)m_k.
]

这些边界是按调度语义手工设计的，而不是从数据自动学习。([GitHub][12])

## 9.5 剩余步数分解模型

先预测剩余工具数量 (M_i) 属于哪个区间：

[
M_i\in
[0,3],\ (3,10],\ (10,25],\ (25,50],\ (50,\infty).
]

得到期望剩余步数：

[
\hat M_i
========

\sum_kP(c=k\mid x)m_k^{\text{steps}}.
]

然后乘以当前工具家族的平均单步时间：

[
\hat R_i
========

\hat M_i
\cdot
\overline d_{\text{family}(i)}.
]

这是典型的“先预测工作量，再预测单位成本”的分解方法。([GitHub][12])

---

# 十、在线校准算法

校准解决的问题是：

> 离线模型可能在训练机器上表现正常，但部署到不同机器、不同 placement 或不同系统负载后，预测会整体偏快、偏慢，或者 P90 覆盖率不再是 90%。

当前项目包含三种不同含义的校准。

---

## 10.1 时延整体倍率的 EWMA 校准

设离线模型输出：

[
\hat Q_{0.5,t},
\quad
\hat Q_{0.9,t},
\quad
\hat Q_{0.99,t}.
]

按照以下键分组：

[
g=
(
\text{tool family},
\text{machine profile},
\text{placement class}
).
]

每个组维护一个尺度因子 (c_{g,t})，初始化为 1。

校准后的预测是：

[
\tilde Q_{q,t}
==============

c_{g,t}\hat Q_{q,t}.
]

工具执行完成后得到真实时延 (d_t)，计算：

[
r_t
===

\operatorname{clip}
\left(
\frac{d_t}{\hat Q_{0.5,t}},
0.2,
5.0
\right).
]

然后进行指数移动平均更新：

[
c_{g,t+1}
=========

(1-\alpha)c_{g,t}
+
\alpha r_t,
\qquad
\alpha=0.15.
]

例如，模型一直预测 1 秒，但当前机器实际需要约 2 秒，则：

[
r_t\approx2,
]

随着观测增加，(c_g) 会逐渐接近 2。此后 P50、P90、P99 都乘约 2。([GitHub][13])

这是**整体速度偏差校准**，主要修正机器快慢、placement 差异等造成的乘性偏差。

---

## 10.2 P90 覆盖率反馈校准

即使 P50 准确，尾部也可能过窄。例如模型声称是 P90，但实际上只有 70% 的调用低于预测值。

定义违反事件：

[
v_t=
\mathbf 1(d_t>\tilde Q_{0.9,t}).
]

在最近 (W=100) 个同家族样本中，经验覆盖率为：

[
\widehat{\operatorname{Cov}}_{0.9}
==================================

1-
\frac{1}{W}
\sum_{j=t-W+1}^{t}v_j.
]

每个 `tool_family` 维护尾部倍率 (s_f)，校准后：

[
Q_{0.9,t}^{\mathrm{final}}
==========================

s_f\tilde Q_{0.9,t},
]

[
Q_{0.99,t}^{\mathrm{final}}
===========================

s_f\tilde Q_{0.99,t}.
]

P50 不变。

更新规则为：

[
s_f\leftarrow
\begin{cases}
1.05s_f,
&\widehat{\operatorname{Cov}}*{0.9}<0.90,[4pt]
\max(1,0.95s_f),
&\widehat{\operatorname{Cov}}*{0.9}>0.95,[4pt]
s_f,
&\text{其他}.
\end{cases}
]

也就是说：

* 覆盖率不足：扩大尾部；
* 覆盖率明显过高：缩小尾部；
* 但尾部倍率不会低于 1。

这是一个简单的反馈控制器，而不是标准 conformal prediction。([GitHub][13])

整体流程为：

[
\text{原始预测}
\rightarrow
\text{EWMA 整体缩放}
\rightarrow
\text{尾部覆盖率缩放}.
]

---

## 10.3 时延分档概率的在线校准

对时延分档模型，原始概率为：

[
p_{t,k}=P(B_t=k\mid x_t).
]

每个工具家族 (g) 和区间 (k) 维护权重：

[
w_{g,k},
\qquad
w_{g,k}^{(0)}=1.
]

先乘权重并重新归一化：

[
\tilde p_{t,k}
==============

\frac{
w_{g,k}p_{t,k}
}{
\sum_jw_{g,j}p_{t,j}
}.
]

真实标签出现后，更新：

[
w_{g,k}
\leftarrow
w_{g,k}
\left[
1+\alpha
\left(
\mathbf 1(y_t=k)-\tilde p_{t,k}
\right)
\right],
]

其中

[
\alpha=0.05,
\qquad
w_{g,k}\in[0.2,5].
]

如果某个家族的长任务持续被低估，那么长时延区间对应的 (w_{g,k}) 会逐渐增大，从而提高其校准后概率。([GitHub][14])

这更准确地说是**在线类别先验修正**，而不是保证概率满足严格可靠性定义的 Platt scaling 或 isotonic calibration。

---

## 10.4 Random Forest 的 OOB 残差尾部校准

剩余时间 Random Forest 还使用一种离线残差校准。

设：

[
z_i=\log(1+R_i),
]

Random Forest 对训练样本产生 out-of-bag 预测：

[
\hat z_i^{\mathrm{OOB}}.
]

计算残差：

[
e_i=z_i-\hat z_i^{\mathrm{OOB}}.
]

取残差分位数：

[
r_{0.9}=Q_{0.9}({e_i}),
\qquad
r_{0.99}=Q_{0.99}({e_i}).
]

新样本的预测上界为：

[
\hat R_{0.9}
============

\exp\left(
\hat z+
1.1\max(0,r_{0.9})
\right)-1,
]

[
\hat R_{0.99}
=============

\exp\left(
\hat z+
1.1\max(0,r_{0.99})
\right)-1.
]

其中 1.1 是额外的 `tail_scale`。([GitHub][12])

它与前两种在线校准不同：这是训练阶段利用 OOB 残差构造尾部上界，部署后不会自动更新。

---

# 十一、核放置模型

核放置当前不是训练出来的模型，而是一个手工 cost function。

对于每个候选核心 (a)，系统需要在工具启动前观察：

[
s_a=
(
u^{core},
u^{smt},
u^{cluster},
p^{llc},
p^{mem},
q,
f
),
]

分别表示：

* 当前核心利用率；
* SMT sibling 利用率；
* cluster 利用率；
* LLC 压力；
* 内存带宽压力；
* run queue 长度；
* 当前频率相对值。

候选动作是：

[
a\in\mathcal A_i,
]

即把工具 (i) 放到哪个具体核心。([GitHub][15])

## 11.1 工具需求向量

根据规则资源类别生成：

[
d_i=
(
d_i^{cpu},
d_i^{mem},
d_i^{cache},
d_i^{io},
d_i^{parallel}
).
]

每一维通常归一化到 ([0,1])，并由手工表格赋值。例如：

* `cpu`：CPU 需求高；
* `cpu_memory_mixed`：CPU、内存、Cache 都高；
* `file_io`：I/O 需求高；
* `network`：网络需求高。

并行度默认取 1，除非 trace 中提供 `cpu_parallelism` 等观测。([GitHub][15])

## 11.2 候选核心预测代价

基础时延使用工具预测 P90：

[
L_i^{base}=\hat Q_{0.9}(d_i).
]

定义自身干扰：

[
\begin{aligned}
I_{\text{self}}
={}&
d^{cpu}(0.50u^{core}+0.55u^{smt})\
&+
d^{cache}(0.55p^{llc}+0.15u^{smt})\
&+
d^{mem}(0.60p^{mem}+0.20u^{cluster})\
&+
0.12\min(q,4)\
&+
d^{cpu}\max(0,1/f-1)\
&+
0.45p^{parallel}u^{cluster}.
\end{aligned}
]

其中：

[
p^{parallel}
============

\operatorname{clip}
\left(
\frac{d^{parallel}-1}{7},
0,1
\right).
]

再定义对同机其他任务造成的外部性：

[
\begin{aligned}
I_{\text{peer}}
={}&
d^{cpu}(0.55u^{smt}+0.15u^{cluster})\
&+
0.50d^{cache}p^{llc}\
&+
0.55d^{mem}p^{mem}.
\end{aligned}
]

最终预测代价：

[
\hat C(i,a)
===========

L_i^{base}
\left[
1+
I_{\text{self}}(i,a)
+
\lambda I_{\text{peer}}(i,a)
\right],
]

其中

[
\lambda=0.20.
]

选择：

[
a_i^\star
=========

\arg\min_{a\in\mathcal A_i}
\hat C(i,a).
]

([GitHub][15])

因此这一模块不是从 placement 数据学习 (\hat C(i,a))，而是人为指定需求权重和干扰系数。

## 11.3 评估指标：regret

若对同一个工具真实重放到多个候选核心，得到反事实成本：

[
C(i,a),
]

则 oracle 核心为：

[
a_i^{oracle}
============

\arg\min_aC(i,a).
]

归一化 regret 为：

[
\operatorname{regret}_i
=======================

\frac{
C(i,a_i^\star)-C(i,a_i^{oracle})
}{
C(i,a_i^{oracle})
}.
]

* regret (=0)：选到最优核心；
* regret (=0.1)：比最优核心慢 10%。

项目明确区分真实反事实重放和 synthetic stress test；合成结果不能被当成真实 placement 加速证据。([GitHub][15])

---

# 十二、投机工具执行决策

设下一工具预测正确的概率为：

[
p=P(\hat t_{i+1}=t_{i+1}).
]

预测工具的 P50 时延为 (L_{50})，P90 为 (L_{90})，LLM 仍需执行的时间为 (S)。

若提前执行正确，最多可以隐藏：

[
H=\min(L_{50},S).
]

期望收益：

[
B=pH.
]

若预测错误，浪费的执行时间：

[
W=(1-p)L_{50}.
]

代码额外加入干扰成本：

[
I=0.15L_{90}.
]

总成本：

[
C=W+I.
]

准入条件：

[
\text{admit}
============

\text{safe}
\land
B>\lambda C,
]

默认：

[
\lambda=1.
]

只有只读工具，例如 `read_file`、`grep`、`find`、`web_search` 等，被认为可以安全投机。([GitHub][16])

一个简单例子：

[
p=0.8,\quad
L_{50}=1000\text{ ms},\quad
L_{90}=1800\text{ ms},\quad
S=1500\text{ ms}.
]

则：

[
B=0.8\times1000=800\text{ ms},
]

[
C=0.2\times1000+0.15\times1800
=470\text{ ms}.
]

若工具只读，则：

[
800>470,
]

可以投机执行。

但当前 `speculation_metrics` 中有一个实现边界：虽然模型预测了 `next_tool`，成本和安全判断仍直接使用当前 `ToolSample`，并没有构造一个代表“预测下一工具”的新样本。因此，当前离线投机评估更接近框架验证，还不是严格的下一工具执行成本评估。([GitHub][16])

---

# 十三、如何训练和评估

项目按 `(dataset, case_id)` 划分训练集和测试集：

[
80%\text{ case 用于训练},
\qquad
20%\text{ case 用于测试}.
]

同一个 case 的所有工具调用不会一部分进入训练、一部分进入测试，因此比随机按工具调用拆分更能避免同一 case 泄漏。([GitHub][17])

## 13.1 时延分档指标

除了普通准确率，还包括：

### Macro F1

对每个区间分别计算 F1，再等权平均：

[
\operatorname{MacroF1}
======================

\frac{1}{K}
\sum_{k=1}^{K}F1_k.
]

它不会让大量短任务掩盖少量长任务。

### Adjacent bucket accuracy

预测与真实区间最多差一档：

[
\frac{1}{N}
\sum_i
\mathbf 1(|\hat y_i-y_i|\le1).
]

### Severe underprediction

预测至少比真实值低两档：

[
\frac{1}{N}
\sum_i
\mathbf 1(\hat y_i\le y_i-2).
]

### Long-task recall

对真实时延不少于 10 秒的样本：

[
\frac{
#{\text{真实长任务且预测为长任务}}
}{
#{\text{真实长任务}}
}.
]

这些指标比单纯 accuracy 更符合调度风险，因为把 70 秒任务预测成 20 秒，通常比预测成 8 秒严重得多。([GitHub][7])

## 13.2 连续时间指标

### MAE

[
\operatorname{MAE}
==================

\frac{1}{N}
\sum_i|R_i-\hat R_i|.
]

### WAPE

[
\operatorname{WAPE}
===================

\frac{
\sum_i|R_i-\hat R_i|
}{
\sum_i|R_i|
}.
]

### Pinball loss

[
L_q(y,\hat y)
=============

\max
\left[
q(y-\hat y),
(q-1)(y-\hat y)
\right].
]

P90 的 pinball loss 会严重惩罚低估。

### Coverage

[
\operatorname{Coverage}_{0.9}
=============================

\frac{1}{N}
\sum_i
\mathbf 1(R_i\le\hat Q_{0.9,i}).
]

理想情况下：

[
\operatorname{Coverage}_{0.9}\approx0.90.
]

覆盖率越高不一定越好。如果 P90 永远预测一个极大值，覆盖率可以达到 100%，但对调度没有信息量，因此需要同时观察 pinball loss 和区间宽度。([GitHub][18])

---

# 十四、用一个完整例子串起来

假设一个 Agent episode 有三个工具调用：

[
\begin{array}{c|c|c|c}
i & \text{tool} & \text{operation} & d_i\
\hline
0 & \texttt{read_file} & \texttt{read_file} & 80\text{ ms}\
1 & \texttt{exec-grep} & \texttt{grep} & 600\text{ ms}\
2 & \texttt{exec-pytest} & \texttt{pytest} & 20000\text{ ms}
\end{array}
]

对于第 1 个调用 `exec-grep`：

[
\text{tool}=\texttt{exec-grep},
]

[
\text{operation}=\texttt{grep},
]

[
\text{tool_family}=\texttt{search}.
]

如果命令带递归参数：

[
\text{resource_class}=\texttt{io_search}.
]

其历史为：

[
h_1=[\texttt{read_file}],
]

下一工具标签为：

[
y_1=\texttt{exec-pytest}.
]

当前调用的时延区间标签为：

[
B(600\text{ ms})=\texttt{0.1-1s}.
]

在 `exec-grep` 执行完成后：

[
C_1=80+600=680\text{ ms},
]

[
R_1=20000\text{ ms},
]

[
M_1=1.
]

因此同一条样本可以同时用于：

* 时延分档：预测 `exec-grep` 是否落在 0.1–1 秒；
* 下一工具预测：预测下一步是否为 `exec-pytest`；
* 剩余时间预测：预测后续还剩约 20 秒工具时间；
* 投机决策：是否提前启动预测的下一工具；
* placement：将当前或下一工具放在哪个核心。

---

# 十五、这个项目目前真正完成到什么程度

从数学上看，当前系统可以概括为：

[
\boxed{
\text{Trace}
\rightarrow
\text{结构化样本}
\rightarrow
\begin{cases}
\text{时延区间分类}\
\text{经验时延分位数}\
\text{下一工具分类}\
\text{剩余工具时间回归}
\end{cases}
\rightarrow
\begin{cases}
\text{在线校准}\
\text{核放置}\
\text{投机准入}
\end{cases}
}
]

但需要准确把握以下边界：

1. **主要被预测的成本仍然是 latency。**`ToolCostDistribution` 虽然预留了 CPU time、memory、working set、I/O 等字段，但当前主要模型通常没有为这些字段提供非零预测。([GitHub][2])

2. **resource class 是规则推断，不是根据独立硬件计数器训练的分类器。**

3. **placement cost 是手工干扰函数，不是当前数据学习出来的 response surface。**

4. **remaining time 是剩余工具时延之和，不是完整 Agent 剩余完成时间。**

5. **Random Forest 的所谓 P50 是对数空间点预测，不是严格分位数回归。**

6. **部分 P90/P99 是经验分位数或残差校准得到的，但部分模型只是用 (1.5\times)、(2\times)、(4\times) 等手工倍数。**

7. **当前框架的核心研究价值不是“已经有一个很准的模型”，而是把多个调度相关预测问题、特征、校准和决策指标统一到同一套可复现实验管线中。**([GitHub][1])

[1]: https://github.com/w20chen/ToolSched "GitHub - w20chen/ToolSched · GitHub"
[2]: https://github.com/w20chen/ToolSched/blob/main/toolsched/schema.py "ToolSched/toolsched/schema.py at main · w20chen/ToolSched · GitHub"
[3]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/features/command.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/features/resource_class.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/features/ml.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/data/loader.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/models/buckets.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/models/baselines.py "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/models/next_tool.py "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/episodes.py "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/cli.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/models/remaining.py "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/calibration/online.py "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/calibration/bucket.py "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/scheduler/placement.py "raw.githubusercontent.com"
[16]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/scheduler/speculation.py "raw.githubusercontent.com"
[17]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/evaluation/split.py "raw.githubusercontent.com"
[18]: https://raw.githubusercontent.com/w20chen/ToolSched/main/toolsched/evaluation/metrics.py "raw.githubusercontent.com"
