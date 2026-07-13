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

It is designed for the datasets under `C:\Users\29068\Desktop\agent_datasets`,
including SWE-ReBench, SWE-Bench Verified, Terminal-Bench, BFCL multi-turn,
BFCL memory, BFCL web-search, and DeepResearchBench.

## Prediction Questions

The first version includes these tasks:

- `latency_bucket`: a supervised ML task. It predicts one of five duration
  buckets: `<100ms`, `0.1-1s`, `1-10s`, `10-60s`, `>60s`.
- `next_tool`: a supervised ML task. It predicts the next tool from the current
  tool and recent tool history.
- `latency_quantiles`: a statistical profile, not a supervised ML model. It
  reports grouped empirical P50/P90/P99.
- `resource_class`: a rule-based taxonomy until independent telemetry labels
  are available.
- `placement`: a policy/evaluator only until controlled replay provides real
  counterfactual placement labels.
- `speculation`: a decision rule combining next-tool confidence, predicted
  cost, read-only safety, and LLM slack.
- `agent_remaining_time`: a post-tool prediction task. It predicts how much
  observed tool-call time remains in the current agent episode after the
  current tool has completed.

## Quick Start

```powershell
python -m toolsched.cli inspect --datasets C:\Users\29068\Desktop\agent_datasets
python -m toolsched.cli build --datasets C:\Users\29068\Desktop\agent_datasets --out artifacts\samples.jsonl
python -m toolsched.cli profile --samples artifacts\samples.jsonl --out artifacts\profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts\samples.jsonl --out artifacts\supervised.json
python -m toolsched.cli evaluate --samples artifacts\samples.jsonl --out artifacts\metrics.json
python -m toolsched.cli calibrate --samples artifacts\samples.jsonl --out artifacts\calibration.json
python -m toolsched.cli simulate-placement --samples artifacts\samples.jsonl --out artifacts\placement.json
python -m toolsched.cli speculate --samples artifacts\samples.jsonl --out artifacts\speculation.json
```

For a smaller first pass:

```powershell
python -m toolsched.cli build --datasets C:\Users\29068\Desktop\agent_datasets --out artifacts\samples.small.jsonl --limit-attempts 200
python -m toolsched.cli evaluate --samples artifacts\samples.small.jsonl
```

To focus on real tool latency rather than BFCL in-memory calls, include
SWE/OpenClaw, Terminal-Bench, and DeepResearchBench:

```powershell
python -m toolsched.cli build --datasets C:\Users\29068\Desktop\agent_datasets --include-dataset deep-research-bench --include-dataset swe-rebench --include-dataset swe-bench-verified --include-dataset terminal-bench --min-duration-ms 1 --out artifacts\agent_non_bfcl.samples.jsonl
python -m toolsched.cli profile --samples artifacts\agent_non_bfcl.samples.jsonl --out artifacts\agent_non_bfcl.profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts\agent_non_bfcl.samples.jsonl --out artifacts\agent_non_bfcl.supervised.json
python -m toolsched.cli evaluate --samples artifacts\agent_non_bfcl.samples.jsonl --out artifacts\agent_non_bfcl.metrics.json
python -m toolsched.cli evaluate-remaining --samples artifacts\agent_non_bfcl.samples.jsonl --out artifacts\agent_non_bfcl.remaining.v3.json --min-episode-steps 3
```

DeepResearchBench can also be evaluated separately:

```powershell
python -m toolsched.cli build --datasets C:\Users\29068\Desktop\agent_datasets --include-dataset deep-research-bench --min-duration-ms 1 --out artifacts\deep_research.samples.jsonl
python -m toolsched.cli profile --samples artifacts\deep_research.samples.jsonl --out artifacts\deep_research.profiles.json
python -m toolsched.cli evaluate-supervised --samples artifacts\deep_research.samples.jsonl --out artifacts\deep_research.supervised.json
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
  - synthetic placement evaluator.
  - cost-aware speculative admission.

## Output Schema

Each normalized row has the core shape:

```json
{
  "sample_id": "swe-rebench/12rambau__sepal_ui-516/attempt_1/tool_0_L8JbQ3eyO",
  "dataset": "swe-rebench",
  "case_id": "12rambau__sepal_ui-516",
  "attempt_id": "attempt_1",
  "tool": "exec-grep",
  "operation": "grep",
  "tool_family": "terminal",
  "timestamp": "2026-06-26T13:26:41.672047Z",
  "duration_ms": 510.2,
  "features": {},
  "labels": {},
  "history": ["read_file"],
  "next_tool": "exec-grep"
}
```

## Design Notes

This repo intentionally separates:

- hardware-independent tool demand features,
- action-conditioned runtime response,
- online residual calibration,
- decision-aware metrics such as action ranking and regret.

The default data only contains observed placements for many traces. Placement
evaluation therefore supports two modes:

- real counterfactual mode if rows include `labels.placement_costs`;
- synthetic what-if mode to validate the scheduler and metrics pipeline.

Remaining-time labels are built from normalized tool-call samples, so they
measure remaining observed tool latency rather than full wall-clock agent time
including LLM thinking, queueing, or hidden harness overhead.

Latest remaining-time check on `agent_non_bfcl.samples.jsonl`:

- `random_forest_log`: MAE 258.4s, WAPE 0.788, R2 0.033, P90 coverage 0.951.
- `global_quantile`: MAE 287.3s, WAPE 0.876, R2 -0.116, P90 coverage 0.975.
- `step_conditioned`: MAE 283.9s, WAPE 0.865, R2 -0.111, P90 coverage 0.963.
