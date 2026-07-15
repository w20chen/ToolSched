# Prediction Problems in ToolSched

ToolSched is an **offline experiment framework for agent tool-cost modeling**.
It normalizes benchmark traces from several agent datasets into a uniform
schema, then evaluates prediction models across five distinct problems.

This document explains each problem, the features available to models, how
each model works, and the terminology used throughout the project.

---

## Table of Contents

1. [Data Foundation: The `ToolSample`](#1-data-foundation-the-toolsample)
2. [Problem 1: Latency Bucket Classification](#2-problem-1-latency-bucket-classification)
3. [Problem 2: Next Tool Prediction](#3-problem-2-next-tool-prediction)
4. [Problem 3: Latency Quantile Profiling](#4-problem-3-latency-quantile-profiling)
5. [Problem 4: Resource Class Taxonomy](#5-problem-4-resource-class-taxonomy)
6. [Problem 5: Agent Remaining Time Prediction](#6-problem-5-agent-remaining-time-prediction)
7. [Features: What Models Can Learn From](#7-features-what-models-can-learn-from)
8. [Calibration: Online Correction at Inference Time](#8-calibration-online-correction-at-inference-time)
9. [Evaluation: How Models Are Scored](#9-evaluation-how-models-are-scored)
10. [Glossary of Terms](#10-glossary-of-terms)

---

## 1. Data Foundation: The `ToolSample`

Every problem starts from a common normalized row called a **`ToolSample`**.
These are produced by `toolsched.data.loader.load_attempt()`, which reads raw
benchmark output directories and converts each tool call into a sample.

### The `ToolSample` Schema

```python
@dataclass
class ToolSample:
    sample_id: str       # unique ID, e.g. "swe-rebench/case_123/attempt_1/tool_0"
    dataset: str         # source benchmark name, e.g. "swe-rebench", "terminal-bench"
    case_id: str         # task/case identifier within the dataset
    attempt_id: str      # which attempt at solving the case
    tool: str            # the tool invoked, e.g. "read_file", "exec-grep"
    operation: str       # load-oriented operation, e.g. "text_search_recursive"
    tool_family: str     # functional category, e.g. "search_text_processing"
    timestamp: str | None    # ISO-8601 timestamp of when the tool was invoked
    duration_ms: float | None  # wall-clock duration of the tool call in milliseconds
    end_timestamp: str | None  # ISO-8601 timestamp of when the tool completed
    input: dict          # the arguments/parameters passed to the tool
    result_preview: str  # truncated preview of the tool's output
    features: dict       # engineered features (see §7)
    labels: dict         # ground-truth labels for supervised tasks
    resources: dict      # machine-level resource measurements (CPU, memory, etc.)
    history: list[str]   # list of tools called before this one (recent window)
    next_tool: str | None  # the tool that was called immediately after this one
```

### How Raw Data Becomes Samples

The loader (`toolsched/data/loader.py`) walks a dataset directory tree looking
for `tool_calls.json` files via `discover_attempts()`. Each attempt directory
represents one agent run on one task case. The JSON array of tool calls is
processed sequentially:

1. **Normalize the operation and family**: `normalize_operation()` maps the
   concrete `tool` into a load-oriented `operation` and a functional
   `tool_family`. The family says what the tool is for; the operation says
   what kind of load it tends to create.
2. **Extract features**: `extract_command_features()` parses the command text
   to produce numeric and boolean features (command length, flag count, etc.).
3. **Infer resource class**: `infer_resource_class()` maps each operation to
   one fixed coarse resource class.
4. **Build history**: A sliding window of the last `k` tool names is tracked
   as the `history` field.
5. **Set `next_tool`**: The tool name of the *next* call in the array becomes
   the label for next-tool prediction.

### Supported Input Datasets

The framework is designed for datasets under an `agent_datasets` directory,
including:

| Dataset | Description |
|---------|-------------|
| **SWE-ReBench (p1/p2)** | Real-world software engineering agent traces (two partitions) |
| **SWE-Bench Verified** | Curated SWE-bench traces with verified solutions |
| **Terminal-Bench (p1/p2/p3)** | Terminal-based agent interaction traces (three partitions) |
| **BFCL Multi-Turn Base** | Berkeley Function Calling Leaderboard — multi-turn base |
| **BFCL Multi-Turn Long Context** | BFCL multi-turn with long-context scenarios |
| **BFCL Memory** | BFCL memory-augmented function calling traces |
| **BFCL Web Search** | BFCL web-search-augmented function calling traces |
| **DeepResearchBench** | Deep research agent traces |
| **ScienceAgentBench Verified** | Science agent benchmark traces with verified solutions |

Resource sampling intervals vary across datasets.  Use `toolsched inspect` to
see per-dataset estimates:

| Sampling Rate | Datasets |
|---------------|----------|
| ~0.50 s | swe-bench-verified, swe-rebench-p1, terminal-bench-p2 |
| ~0.54 s | bfcl-memory, bfcl-multi-turn-*, bfcl-web-search, deep-research-bench |
| ~2.00 s | science-agent-bench-verified, swe-rebench-p2, terminal-bench-p1, terminal-bench-p3 |

The core pipeline (`_load_resources`) is unaffected by sampling-rate
differences because it aggregates *all* samples unconditionally.  The
`plot_operation_resources.py` script auto-detects the sampling interval and
scales its baseline / counter-gap thresholds to 3× the observed interval
(minimum 2.0 s).

---

## 2. Problem 1: Latency Bucket Classification

**Goal**: Predict which **duration bucket** a tool call's latency will fall
into, *before* the tool is executed.

This is a **supervised multi-class classification** task.

### Label Definition

The continuous `duration_ms` value is discretized into **5 buckets**:

| Bucket Index | Range       | Label       |
|:------------:|-------------|-------------|
| 0            | 0 – 100 ms  | `<100ms`    |
| 1            | 100 ms – 1 s | `0.1-1s`   |
| 2            | 1 s – 10 s  | `1-10s`     |
| 3            | 10 s – 60 s | `10-60s`    |
| 4            | > 60 s      | `>60s`      |

The function `bucket_index(duration_ms)` maps a duration to its bucket index,
and `bucket_label(index)` maps back to the string label.

### Models

#### A. `PerToolBucketBaseline` (Baseline)

A simple **most-common-bucket-per-tool** model. For each tool name (e.g.,
`"read_file"`, `"exec-grep"`), it finds the bucket that appears most often in
the training data. For unseen tools, it falls back to the global most-common
bucket.

This is a *non-learned* baseline — it doesn't use any features beyond the tool
name.

#### B. `LatencyBucketModel` (Logistic Regression)

Uses `LogisticRegression` from scikit-learn with `class_weight="balanced"`
to handle imbalanced buckets (most tool calls are fast, few are slow).

**Features**: The full feature set from `SampleFeatureEncoder` (see §7).
This includes the tool name, operation, tool family, dataset, command
properties, recent history tools, and resource class hint — all encoded as
a sparse feature vector by `DictVectorizer`.

**Training**: A `LogisticRegression` (multi-class, `lbfgs` solver, 2000 max
iterations) is trained on the encoded feature matrix with bucket index as
the target.

**Prediction**: `predict_index(sample)` returns the bucket index (0–4).
`predict_proba_dict(sample)` returns per-bucket probabilities.

#### C. `HistoricalBucketFeatureModel` (Random Forest)

A `RandomForestClassifier` (120 trees, max depth 14, min samples per leaf 8)
that augments the standard feature set with **historical duration priors**.

**Historical priors** are computed from the training set:

| Prior Feature | Description |
|--------------|-------------|
| `prior_bucket_0` ... `prior_bucket_4` | Empirical fraction of samples in each bucket (global, per-tool, per-operation) |
| `prior_log_p50` | Log-transformed P50 of prior durations |
| `prior_log_p90` | Log-transformed P90 of prior durations |
| `prior_log_mean` | Log-transformed mean of prior durations |

These priors are injected into the sample's `features` dict before encoding.
The idea is that the Random Forest can use these priors to improve recall on
long-running tasks (buckets 3 and 4) without "seeing" future labels — the
priors could be maintained online.

### Metrics

See `evaluate_bucket_model()` in `toolsched/models/buckets.py`:

| Metric | Definition |
|--------|-----------|
| **Accuracy** | Exact bucket match rate |
| **Macro F1** | F1 score averaged across buckets (each bucket weighted equally) |
| **Weighted F1** | F1 score weighted by bucket frequency |
| **Adjacent Bucket Accuracy** | Fraction of predictions off by at most 1 bucket (e.g., true=3, pred=4 counts as correct) |
| **Severe Underprediction Rate** | Fraction of predictions that are 2+ buckets *below* the true bucket (underestimating task duration) |
| **Long Task Recall (≥10s)** | Recall on buckets 3 and 4 combined (tools taking ≥10 seconds) |
| **Per-Bucket Recall** | Recall broken down by each individual bucket |

---

## 3. Problem 2: Next Tool Prediction

**Goal**: Predict which **tool the agent will call next**, given the current
tool call and the history of recent calls.

This is a **supervised multi-class classification** task over a tool vocabulary.

### Label Definition

The `next_tool` field of a `ToolSample` — the `tool` value of the immediately
following tool call in the same attempt. If a sample is the last call in an
episode, it has no `next_tool` and is excluded from evaluation.

### Models

#### A. `NextToolMarkovModel` (First-Order Markov Baseline)

A **first-order Markov chain**: it counts how often each tool follows each
other tool in training data. Prediction is simply the most common next tool
given the current `sample.tool`. If the current tool has never been seen,
it falls back to the globally most common tool.

This is the simplest baseline — it only uses the current tool name.

#### B. `HistoryMarkovModel` (Longest-Suffix Markov)

A **variable-order Markov model** that looks at the longest matching suffix of
the recent tool history to predict the next tool.

For example, with `history = ["read_file", "exec-grep", "exec-grep"]` and
`current_tool = "edit_file"`, it checks sequences:
- `("exec-grep", "exec-grep", "edit_file")` — full 3-tuple
- `("exec-grep", "edit_file")` — last 2
- `("edit_file",)` — last 1

It uses the longest suffix that has been seen in training and returns the most
common next tool from that state. If no suffix matches, it falls back to the
first-order Markov model.

This captures patterns like "after two grep calls followed by an edit, the
next tool is likely another grep" even though a single "edit_file" might not
predict that.

#### C. `NextToolLogisticModel` (Logistic Regression)

Uses `LogisticRegression` over a richer feature set including the current tool,
operation, tool family, dataset, and recent history (last 5 tools) — all
one-hot encoded.

**Training**: Only tools that appear at least `min_count=5` times as a next
tool are included as classes. Samples whose next tool is rare are dropped from
LR training but are still covered by the Markov fallback.

**Fallback**: If the logistic model hasn't been fitted (fewer than 2 classes),
prediction falls back to `NextToolMarkovModel`.

### Metrics

See `evaluate_next_tool()` in `toolsched/models/next_tool.py`:

| Metric | Definition |
|--------|-----------|
| **Top-1 Accuracy** | Does the top prediction match the true next tool? |
| **Top-3 / Top-5 / Top-10 Accuracy** | Is the true next tool among the top *k* predictions? |

---

## 4. Problem 3: Latency Quantile Profiling

**Goal**: Report empirical **P50/P90/P99 latencies** for groups of similar
tool calls.

This is **not a supervised ML model** — it is a statistical baseline that
directly reports observed quantiles from grouped training data.

### Model: `GroupQuantileModel`

Groups tool calls by a configurable key (default: `("operation",)`)
and computes the **P50, P90, and P99** of the observed `duration_ms` values
within each group.

If a test sample falls into an unseen group, the model falls back to
`GlobalQuantileModel` (quantiles over all training data regardless of group).

### ToolCostDistribution Output

```python
@dataclass(frozen=True)
class ToolCostDistribution:
    latency_p50_ms: float    # median latency
    latency_p90_ms: float    # 90th percentile latency
    latency_p99_ms: float    # 99th percentile latency
    cpu_time_ms: float       # (reserved for future use)
    memory_bytes: float      # (reserved for future use)
    working_set_bytes: float # (reserved for future use)
    io_bytes: float          # (reserved for future use)
    resource_class: str      # from the taxonomy
    uncertainty: float       # P90 - P50 as a simple spread measure
```

### Additional Profiling: `tool_profiles()`

The `toolsched.evaluation.profile` module produces a richer profile table
grouped by `(tool_family, operation, resource_class)`, including:

- `count` — number of samples in the group
- `latency_p50/p90/p99_ms`
- `mean_command_len`, `mean_preview_len`
- `recursive_rate` — fraction of calls with a recursive search hint
- `pipe_rate` — fraction of calls using a pipe (`|`)

### Metrics

Since this is not a learned model, it's evaluated with **regression metrics**
on the `latency_p50_ms` prediction:

| Metric | Definition |
|--------|-----------|
| **MAE** | Mean Absolute Error of P50 prediction vs actual duration |
| **MAPE** | Mean Absolute Percentage Error |
| **Pinball Loss (P50, P90, P99)** | Quantile loss at each level |
| **Coverage (P90, P99)** | Fraction of actual durations ≤ the predicted quantile |

---

## 5. Problem 4: Resource Class Taxonomy

**Goal**: Classify each tool call into a **resource usage category** based on
its operation.

This is a **rule-based taxonomy** — not a learned ML model — because
independent telemetry labels (actual CPU/memory measurements per tool call)
are not yet available in the datasets.

### The Taxonomy Rules

Defined in `toolsched/features/resource_class.py`. The mapping is many-to-one:
each operation has exactly one resource class, while multiple operations may
share the same resource class.

| Operation | Resource Class |
|-----------|---------------|
| `data_script` | `cpu` |
| `test_run`, `project_build`, `container_operation` | `cpu_memory_mixed` |
| `package_install`, `version_control_update`, `download` | `network_disk_io` |
| `text_search_simple` | `search` |
| `text_search_recursive` | `io_search` |
| `text_transform`, `archive_operation`, `database_query` | `cpu_io_mixed` |
| `directory_list`, `file_discovery`, `version_control_status`, `system_operation` | `metadata_io` |
| `file_read`, `file_write`, `file_edit`, `file_mutation`, `version_control_diff`, `version_control_history` | `file_io` |
| `memory_read`, `memory_write`, `working_directory`, `shell_control` | `light_control` |
| `web_search`, `web_fetch` | `network` |
| `shell_script`, `unknown_command` | `unknown` |

This classification is stored in `sample.labels["resource_class"]` as derived
metadata for profiles, reports, and consistency checks. It is not used as an
independent model feature because it is determined by `operation`. Quantile
profiling groups by `operation`; `resource_class` remains useful as a coarser
readable load bucket.

### Metrics

Since this is a rule and there's no independent ground-truth label,
`classification_metrics()` in `toolsched/evaluation/metrics.py` compares the
predicted class (which is the same rule) against the stored label — this is
only useful as a consistency check.

---

## 6. Problem 5: Agent Remaining Time Prediction

**Goal**: After a tool call completes, predict how much **total agent
wall-clock time remains** before the episode finishes.

This is a **regression** (or distributional regression) task. It is the most
complex problem in ToolSched.

### Label Definition

The remaining time is computed by `AgentEpisode._build_wall_time_rows()`.

Two sources are used (in priority order):

1. **`agent_wall_time`** (preferred): When `resources.json` contains
   `attempt_start_time` and `attempt_end_time`, the remaining time is the
   actual wall-clock time from the current tool's completion to the end of
   the entire attempt (including LLM inference time, inter-tool gaps, etc.).
   
2. **`tool_timestamp_span`**: When per-tool timestamps are available but
   attempt-level timing isn't, the span from the earliest tool start to the
   latest tool end is used as the episode timeline.
   
3. **`tool_duration_sum`** (fallback): Simple sum of all future tool
   `duration_ms` values. This excludes LLM time and inter-tool gaps.

### Episode Structure

The flat `ToolSample` list is grouped into `AgentEpisode` objects by
`(dataset, case_id, attempt_id)`. Each episode produces one `EpisodeStep` per
tool call position:

```python
@dataclass
class EpisodeStep:
    sample: ToolSample        # the tool call at this position
    step_index: int           # 0-based position in the episode
    cumulative_time_ms: float # total elapsed time up to and including this step
    remaining_time_ms: float  # total time remaining after this step (the TARGET)
    remaining_steps: int      # number of steps remaining after this one
    total_time_ms: float      # total episode duration
    total_steps: int          # total number of steps in the episode
    label_source: str         # how remaining_time_ms was computed
```

### Features (`EpisodeStep.feature_vector()`)

| Feature | Description |
|---------|-------------|
| `step_index` | Current step number (0-based) |
| `cumulative_time_ms` | Total time elapsed so far |
| `last_duration_ms` | Duration of the just-completed tool call |
| `mean_duration_so_far_ms` | Average duration per step so far |
| `tool_diversity_so_far` | Number of unique tools seen so far |
| `command_len` | Length of the current tool's command text |
| `preview_len` | Length of the current tool's output preview |
| `has_pipe` | Whether the command contains a pipe |
| `has_recursive_hint` | Whether the command has a recursive flag |
| `argv_count` | Number of command-line arguments |

Additional features are added by specific models (e.g., one-hot encodings,
interaction terms).

### Models

There are **9 models** for remaining-time prediction, ranging from simple
statistical baselines to learned regressors.

#### 1. `GlobalRemainingQuantile` (Baseline)

Unconditional quantiles of remaining time across *all* training steps. Every
sample gets the same prediction. This shows how much predictive power comes
from just knowing the typical episode length.

#### 2. `StepConditionedRemaining` (Baseline)

Groups training rows by `step_index` (capped at 50), then computes per-step
quantiles. At test time, predicts using the quantiles of the matching step
bucket. This captures the fact that remaining time tends to decrease as the
episode progresses.

#### 3. `ProgressConditionedRemaining` (Intermediate)

Groups by **progress ratio** (cumulative ÷ total time). During training, the
true progress ratio is known (oracle). At inference, a simple linear
regression model predicts the progress ratio from features, and the
corresponding quantile bucket is used.

This is more adaptive than step-conditioned because episodes vary in length —
step 5 might be 50% done in one episode and 10% done in another.

#### 4. `LinearQuantileRemaining` (Pythia-style Core)

A **linear regression model trained with pinball loss** at multiple quantile
levels (P50, P90, P99). This is "Pythia-style" — lightweight, interpretable,
and producing calibrated uncertainty estimates.

**Training**: Mini-batch gradient descent with:
- Features z-score normalized
- Targets (remaining time) also z-score normalized to keep gradients stable
- L2 regularization
- Learning rate decay
- Separate weight vectors for each quantile level

**Prediction**: For each quantile level `q`, computes `w_q · x + b_q`, then
un-normalizes back to milliseconds. All predictions are clamped to ≥ 0.

#### 5. `LogSpaceRemaining`

Linear regression in **log(remaining_time + 1) space**, then exponentiate
back. This better handles the heavy-tailed distribution of remaining times
(most tools are fast, some are very slow).

#### 6. `RandomForestRemainingRegressor`

A `RandomForestRegressor` (80 trees, max depth 14) that predicts
`log1p(remaining_time_ms)` using a rich feature dict (see below).

**Key details**:
- Trained in log space to handle heavy tails
- Uses OOB (out-of-bag) residual quantiles to calibrate P90/P99 predictions:
  `p90 = expm1(log_p50 + residual_q90 * tail_scale)`
- Features include one-hot encodings of dataset, tool, operation, family,
  plus numerical features (all log-transformed):
  `step_index_log`, `cumulative_time_log`, `current_duration_log`,
  `mean_duration_so_far_log`, `command_len_log`, `argv_count_log`,
  `flag_count_log`, `has_pipe`, `has_recursive_hint`, `history_len`,
  `tool_diversity_recent`, and recent tool history one-hots.

#### 7. `EwmaRemainingByFamily`

**EWMA** (Exponentially Weighted Moving Average) grouped by `tool_family`.

For each family, maintains: `state = (1 - α) × old_state + α × observed_remaining`

At inference, predicts the family's current state. Tail quantiles are
heuristic multiples (P90 = P50 × 1.5, P99 = P50 × 2.0).

#### 8. `CompositionalRemaining`

A **decomposed predictor** that:
1. Builds a per-tool duration quantile table (P50/P90/P99)
2. Builds a first-order Markov transition matrix (tool → next tool probabilities)
3. At prediction time, **rolls out** the most likely future tool sequence up
   to `rollout_depth=10` steps, accumulating predicted durations
4. Applies a discount factor (0.95) to reduce contribution of distant steps

This mirrors the Pythia philosophy of decomposing a hard problem (total
remaining time) into predictable sub-components (what tools next, how long
each takes).

#### 9. `BinnedRemainingClassifier`

A **softmax (multinomial logistic) regression** that classifies remaining
time into **scheduling-semantic buckets**:

| Bucket | Range | Scheduling Meaning |
|--------|-------|--------------------|
| 0 | ≤ 15 s | Overlap — hideable behind LLM inference |
| 1 | 15–60 s | Short — quick scheduling |
| 2 | 60–180 s | Moderate — normal scheduling |
| 3 | 180–600 s | Long — significant resource commitment |
| 4 | 600–1800 s | Heavy — consider preemption |
| 5 | > 1800 s | Extreme — special handling |

The model is trained end-to-end with mini-batch gradient descent. At inference,
it produces a scalar prediction by weighting each bucket's median by its
predicted probability.

#### 10. `StepsDecomposedRemaining`

Another **decomposed predictor** that:
1. **Predicts remaining steps** (not time) via a softmax classifier over
   step-count buckets (≤3, 4–10, 11–25, 26–50, >50)
2. **Multiplies** by the per-tool-family mean duration per step

This decomposes the hard remaining-time regression into an easier
step-count classification problem, plus a simple duration estimate.

### Evaluation

See `remaining_time_metrics()` in `toolsched/evaluation/remaining.py`.

| Metric | Definition |
|--------|-----------|
| **MAE** | Mean Absolute Error of P50 prediction vs actual remaining time |
| **WAPE** | Weighted Absolute Percentage Error (sum of absolute errors ÷ sum of actuals) |
| **MAPE** | Mean Absolute Percentage Error (only computed when actual > 1s) |
| **SMAPE** | Symmetric MAPE |
| **MdAPE** | Median Absolute Percentage Error |
| **R²** | Coefficient of determination |
| **Pinball Loss (P50, P90, P99)** | Quantile loss at each level |
| **Coverage (P90, P99)** | Fraction of actuals ≤ predicted quantile |
| **Mean Remaining** | Average actual remaining time (for context) |

Additionally, `remaining_by_progress()` breaks down MAE by progress decile
to show where predictions are good/bad.

---

## 7. Features: What Models Can Learn From

### Command Features (`extract_command_features`)

Extracted from the raw tool call payload by `toolsched/features/command.py`:

| Feature | Type | Description |
|---------|------|-------------|
| `command_len` | numeric | Length of the command text in characters |
| `argv_count` | numeric | Number of parsed arguments |
| `flag_count` | numeric | Number of flags (`-r`, `--recursive`, etc.) |
| `has_pipe` | boolean | Whether the command contains a pipe (`\|`) |
| `has_recursive_hint` | boolean | Whether the command contains `-r`, `-R`, `--recursive`, `rg`, etc. |
| `include_count` | numeric | Number of `--include` patterns |
| `path_token_count` | numeric | Number of path-like tokens (containing `/` or `\`) |
| `input_key_count` | numeric | Number of keys in the tool's input dict |
| `preview_len` | numeric | Length of the result preview text |
| `preview_line_count` | numeric | Number of lines in the result preview |
| `exit_code_nonzero` | boolean | Whether the result suggests a non-zero exit code |

### Encoding: `SampleFeatureEncoder`

Used by the supervised models to convert a `ToolSample` into a flat numeric
feature vector. The encoder produces:

**One-hot categorical features:**
- `dataset=<name>` — which benchmark
- `tool=<name>` — which tool
- `operation=<name>` — the normalized operation
- `tool_family=<name>` — the functional category

`resource_class` is intentionally omitted from the model feature vector because
it is derived from `operation`.

**Numeric features (log1p-transformed):**
- `command_len`, `argv_count`, `flag_count`, `include_count`,
  `path_token_count`, `input_key_count`, `call_index`, `history_len`

**Boolean features:**
- `has_pipe`, `has_recursive_hint`

**Historical prior features** (used by `HistoricalBucketFeatureModel`):
- `prior_bucket_0` ... `prior_bucket_4` — per-tool bucket distribution
- `prior_log_p50`, `prior_log_p90`, `prior_log_mean` — per-tool duration stats

**History one-hot features:**
- `prev1_tool=<name>`, `prev2_tool=<name>`, ... up to `prev5_tool=<name>`
- `latest_tool=<name>` (alias for the most recent tool)

### Feature Availability

All features are **online-available** — they can be computed before the tool
is executed using only the tool name, input payload, and history. The only
exception is `call_index` (the position in the attempt), which requires
tracking across the episode.

---

## 8. Calibration: Online Correction at Inference Time

ToolShed includes **online calibration** that simulates how predictions can
be corrected as new observations arrive, without retraining the model.

### Latency Bucket Calibration

`BucketPriorCalibrator` in `toolsched/calibration/bucket.py`:

- Maintains per-`tool_family` multiplicative weights for each bucket
- After each prediction, adjusts weights using the rule:
  `weight = weight × (1 + α × (target - prediction))`
  where `target` is 1.0 for the correct bucket and 0.0 for others
- This is a streaming multiplicative correction (similar to online learning
  of a confusion matrix correction)

### Latency Quantile Calibration

Two-stage calibration in `toolsched/calibration/online.py`:

1. **`EwmsScaleCalibrator`**: Maintains a per-`(tool_family, machine_profile)`
   multiplicative scale factor via EWMA: after observing the actual duration,
   computes `ratio = actual / predicted_p50`, then updates
   `scale = (1 - α) × scale + α × ratio`.

2. **`QuantileCoverageCalibrator`**: Maintains a sliding window of P90
   coverage violations. If coverage drops below the target (90%), it
   inflates the P90/P99 tail estimates. If coverage exceeds 95%, it
   deflates them.

### Replay Simulation

`replay_calibration()` and `replay_bucket_calibration()` simulate streaming
calibration by iterating through test samples in order, applying then
updating the calibrator at each step. The metrics report "before" (raw model)
vs "after" (calibrated) values for accuracy, pinball loss, and coverage.

---

## 9. Evaluation: How Models Are Scored

### Train/Test Split

`split_by_case()` in `toolsched/evaluation/split.py`:

- Groups samples by `(dataset, case_id)`
- Shuffles cases randomly (seed=7)
- Holds out 20% of cases for testing
- All samples from held-out cases go to the test set
- This ensures no data leakage between train and test

### Evaluation Suites

#### Supervised Suite (`run_supervised_suite`)

Runs all supervised problems end-to-end:

1. Splits data by case holdout
2. Trains all models on the training split
3. Evaluates on the test split
4. Reports:

```json
{
  "latency_bucket": {
    "models": {
      "logistic": { "accuracy": ..., "macro_f1": ..., ... },
      "historical_random_forest": { ... },
      "baseline": { ... }
    },
    "online_calibration": { ... }
  },
  "next_tool": {
    "logistic": { "top1_accuracy": ..., ... },
    "history_markov": { ... },
    "markov_1": { ... }
  },
  "latency_quantile_profile": {
    "offline": { "mae_ms": ..., "coverage_p90": ..., ... },
    "online_calibration": { "mae_before_ms": ..., "mae_after_ms": ..., ... }
  }
}
```

---

## 10. Glossary of Terms

| Term | Definition |
|------|-----------|
| **Attempt** | One complete agent run on one task case. Contains multiple tool calls. |
| **Case** | A task/problem instance from a benchmark (e.g., a GitHub issue to fix). |
| **Dataset** | A collection of cases from one benchmark (e.g., SWE-Bench Verified). |
| **Episode** | Same as an attempt — the sequence of all tool calls within one (dataset, case, attempt). |
| **Tool** | A named function the agent can invoke (e.g., `read_file`, `exec-grep`, `write_file`). |
| **Operation** | A load-oriented abstraction of the tool call (e.g., `text_search_recursive`, `project_build`, `package_install`). |
| **Tool Family** | A functional grouping: `data_analysis_scripting`, `test_execution`, `package_environment_mgmt`, `search_text_processing`, `file_navigation`, `version_control`, `file_io`, `web_network`. |
| **Duration** | Wall-clock time a tool call takes, in milliseconds. |
| **Latency Bucket** | 5 discrete categories of duration: `<100ms`, `0.1-1s`, `1-10s`, `10-60s`, `>60s`. |
| **Remaining Time** | Wall-clock time from the current step to the end of the episode. |
| **EWMA** | Exponentially Weighted Moving Average — a running average that gives more weight to recent observations. |
| **Pinball Loss** | A quantile-specific loss function: `max(q × (y−ŷ), (q−1) × (y−ŷ))`. Asymmetric: penalizes under-prediction more when q > 0.5. |
| **P50 / P90 / P99** | The 50th, 90th, and 99th percentiles of a distribution. |
| **Coverage** | For a predicted quantile (e.g., P90), the fraction of actual values that fall at or below it. Ideal: 0.90 for P90. |
| **OOB (Out-of-Bag)** | For Random Forest, predictions on training samples not included in a given tree's bootstrap sample. Used as a free validation set. |
| **Prior (Historical Prior)** | Pre-computed statistics (bucket distribution, duration quantiles) from training data, per tool or per operation. |
| **Calibration** | The process of adjusting model predictions at inference time based on observed errors, without retraining. |
| **WAPE** | Weighted Absolute Percentage Error: sum of absolute errors ÷ sum of actuals. Less sensitive to near-zero actuals than MAPE. |
| **SMAPE** | Symmetric MAPE: `2 × |y−ŷ| / (|y| + |ŷ|)`. Bounded between 0% and 200%. |
| **MdAPE** | Median Absolute Percentage Error — more robust to outliers than MAPE. |
| **Rollout** | In composition models, simulating future tool calls step by step using predicted transitions. |
| **Pythia-style** | After the Pythia microbenchmarking approach: lightweight, interpretable models on engineered features, with quantile predictions and decomposition into sub-problems. |
| **DictVectorizer** | A scikit-learn transformer that converts dictionaries (with string keys) into sparse numeric feature matrices via one-hot encoding. |
| **Log1p / Expm1** | `log1p(x) = log(x + 1)` and `expm1(x) = exp(x) - 1`. Used to transform skewed distributions more symmetrically. |
