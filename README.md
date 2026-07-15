# ToolSched

ToolSched is an offline experiment framework for Agent tool-cost modeling. It
normalizes existing benchmark traces into one sample schema, evaluates several
prediction questions, and replays online calibration without integrating into an
agent runtime.

## Supported Inputs

The loader expects benchmark attempts that contain files such as:

- `tool_calls.json`
- `trace.jsonl`
- `resources.json`
- `results.json`
- `run_manifest.json`

It is designed for the datasets under `/data/share/datasets/agent_datasets`,
including SWE-ReBench (p1/p2), SWE-Bench Verified, Terminal-Bench (p1/p2/p3),
BFCL multi-turn (base, long-context), BFCL memory, BFCL web-search,
DeepResearchBench, and ScienceAgentBench Verified.

### Resource Sampling Intervals

Datasets have different resource-telemetry sampling rates. The framework
estimates the median interval by reading the `epoch` / `timestamp` fields from
the `samples` array in `resources.json` and computing the gap between
consecutive samples.

| Interval | Datasets |
|----------|----------|
| ~0.50 s | swe-bench-verified, swe-rebench-p1, terminal-bench-p2 |
| ~0.54 s | bfcl-memory, bfcl-multi-turn-\*, bfcl-web-search, deep-research-bench |
| ~2.00 s | science-agent-bench-verified, swe-rebench-p2, terminal-bench-p1, terminal-bench-p3 |

The core pipeline (`toolsched.data.loader`) aggregates all samples
unconditionally and is unaffected by sampling-rate differences. The
`scripts/plot_operation_resources.py` script auto-detects each dataset's
interval and scales its baseline / counter-gap thresholds to 3× the observed
interval (minimum 2.0 s).

Run `toolsched inspect` to see per-dataset interval estimates.

## Prediction Questions

The first version includes these tasks:

- `latency_bucket`: a supervised ML task. It predicts one of five duration
  buckets: `<100ms`, `0.1-1s`, `1-10s`, `10-60s`, `>60s`.
- `next_tool`: a supervised ML task. It predicts the next tool from the current
  tool and recent tool history.
- `latency_quantiles`: a statistical profile, not a supervised ML model. It
  reports grouped empirical P50/P90/P99.
- `resource_class`: a rule-based taxonomy until independent telemetry labels
  are available. It is a coarse load class derived from `operation`.
- `agent_remaining_time`: a post-tool prediction task. It predicts how much
  agent wall-clock time remains after the current tool has completed, including
  later LLM time and later tool time when trace timestamps are available.

## Quick Start

```bash
python -m toolsched.cli inspect --datasets /data/share/datasets/agent_datasets
python -m toolsched.cli build --datasets /data/share/datasets/agent_datasets --out artifacts/samples.jsonl
python -m toolsched.cli profile --samples artifacts/samples.jsonl --out artifacts/profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts/samples.jsonl --out artifacts/supervised.json
python -m toolsched.cli evaluate --samples artifacts/samples.jsonl --out artifacts/metrics.json
python -m toolsched.cli calibrate --samples artifacts/samples.jsonl --out artifacts/calibration.json
```

For a smaller first pass:

```bash
python -m toolsched.cli build --datasets /data/share/datasets/agent_datasets --out artifacts/samples.small.jsonl --limit-attempts 200
python -m toolsched.cli evaluate --samples artifacts/samples.small.jsonl
```

To focus on real tool latency rather than BFCL in-memory calls, include
SWE, Terminal-Bench, DeepResearchBench, and ScienceAgentBench:

```bash
python -m toolsched.cli build --datasets /data/share/datasets/agent_datasets \
  --include-dataset deep-research-bench \
  --include-dataset swe-rebench-p1 --include-dataset swe-rebench-p2 \
  --include-dataset swe-bench-verified \
  --include-dataset terminal-bench-p1 --include-dataset terminal-bench-p2 --include-dataset terminal-bench-p3 \
  --include-dataset science-agent-bench-verified \
  --min-duration-ms 1 \
  --out artifacts/agent_non_bfcl.samples.jsonl
python -m toolsched.cli profile --samples artifacts/agent_non_bfcl.samples.jsonl --out artifacts/agent_non_bfcl.profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts/agent_non_bfcl.samples.jsonl --out artifacts/agent_non_bfcl.supervised.json
python -m toolsched.cli evaluate --samples artifacts/agent_non_bfcl.samples.jsonl --out artifacts/agent_non_bfcl.metrics.json
python -m toolsched.cli evaluate-remaining --samples artifacts/agent_non_bfcl.samples.jsonl --out artifacts/agent_non_bfcl.remaining.v3.json --min-episode-steps 3
```

DeepResearchBench can also be evaluated separately:

```bash
python -m toolsched.cli build --datasets /data/share/datasets/agent_datasets --include-dataset deep-research-bench --min-duration-ms 1 --out artifacts/deep_research.samples.jsonl
python -m toolsched.cli profile --samples artifacts/deep_research.samples.jsonl --out artifacts/deep_research.profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts/deep_research.samples.jsonl --out artifacts/deep_research.supervised.json
```

## Model Boundary

This framework deliberately avoids treating every component as ML:

- Learned models:
  - `LatencyBucketModel`: `LogisticRegression(class_weight="balanced")` over
    online-available structured features.
  - `HistoricalBucketFeatureModel`: a compact `RandomForestClassifier` that
    adds training-set tool/operation latency priors as online-maintainable
    features. It is useful for testing whether a stronger model can improve
    long-task recall without using future labels.
  - `NextToolLogisticModel`: `LogisticRegression` over current tool,
    operation/family, and recent tool history.
  - `RandomForestRemainingRegressor`: `RandomForestRegressor` over online
    episode-state features, trained in `log1p(remaining_time_ms)` space. It
    uses OOB residuals to calibrate P90/P99 remaining-time upper bounds.
- Statistical baselines:
  - global and grouped empirical latency quantiles.
  - per-tool most-common latency bucket.
  - first-order Markov next-tool baseline.
  - longest-suffix history Markov next-tool baseline.
  - global, step-conditioned, EWMA, and simple compositional remaining-time
    baselines.
- Rules and policies:
  - resource taxonomy.

## Output Schema

Each normalized row has the base shape:

```json
{
  "sample_id": "swe-rebench/12rambau__sepal_ui-516/attempt_1/tool_0_L8JbQ3eyO",
  "dataset": "swe-rebench",
  "case_id": "12rambau__sepal_ui-516",
  "attempt_id": "attempt_1",
  "tool": "exec-grep",
  "operation": "text_search_simple",
  "tool_family": "search_text_processing",
  "timestamp": "2026-06-26T13:26:41.672047Z",
  "duration_ms": 510.2,
  "features": {},
  "labels": {},
  "history": ["read_file"],
  "next_tool": "exec-grep"
}
```

## Design Notes

This repo intentionally separates online-available tool-call features,
statistical latency profiling, supervised prediction tasks, and residual
calibration.

Tool taxonomy is layered:

- `tool`: the concrete invoked tool name.
- `operation`: a load-oriented abstraction, such as `text_search_simple`,
  `text_search_recursive`, `project_build`, or `package_install`.
- `resource_class`: a coarse resource bucket determined by `operation`.
  The relation is many-to-one: every operation has one resource class, while
  multiple operations may share a class. It is retained as derived metadata
  for profiles and reports, not as an independent model feature.
- `tool_family`: a functional category, not a load class. Current families are
  `data_analysis_scripting`, `test_execution`, `package_environment_mgmt`,
  `search_text_processing`, `file_navigation`, `version_control`, `file_io`,
  and `web_network`.

Remaining-time labels use attempt/tool timestamps to measure future agent
wall-clock time after each completed tool call. This includes observed gaps
between tool calls and, when attempt end time is available, final post-tool
agent time. If timestamps are unavailable, the evaluator falls back to the
legacy future-tool-duration label and reports that fallback in `label_sources`.

Latest remaining-time check on `agent_non_bfcl.samples.jsonl`:

- `random_forest_log`: MAE 258.4s, WAPE 0.788, R2 0.033, P90 coverage 0.951.
- `global_quantile`: MAE 287.3s, WAPE 0.876, R2 -0.116, P90 coverage 0.975.
- `step_conditioned`: MAE 283.9s, WAPE 0.865, R2 -0.111, P90 coverage 0.963.
