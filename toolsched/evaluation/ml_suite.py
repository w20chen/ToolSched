from __future__ import annotations

from ..calibration.bucket import replay_bucket_calibration
from ..evaluation.metrics import regression_metrics
from ..evaluation.online import replay_calibration
from ..evaluation.split import split_by_case
from ..models.baselines import GroupQuantileModel, NextToolMarkovModel
from ..models.buckets import HistoricalBucketFeatureModel, LatencyBucketModel, PerToolBucketBaseline, evaluate_bucket_model
from ..models.next_tool import HistoryMarkovModel, NextToolLogisticModel, evaluate_next_tool
from ..schema import ToolSample


def run_supervised_suite(samples: list[ToolSample]) -> dict:
    train, test = split_by_case(samples)

    bucket_baseline = PerToolBucketBaseline().fit(train)
    bucket_model = LatencyBucketModel().fit(train)
    bucket_rf = HistoricalBucketFeatureModel().fit(train)

    markov = NextToolMarkovModel().fit(train)
    history_markov = HistoryMarkovModel().fit(train)
    next_model = NextToolLogisticModel().fit(train)

    quantile_model = GroupQuantileModel(("operation",)).fit(train)

    return {
        "split": {
            "strategy": "case_holdout",
            "train": len(train),
            "test": len(test),
        },
        "learnability": {
            "latency_bucket": "learned: true label comes from duration_ms buckets",
            "next_tool": "learned: true label comes from the following trace tool",
            "latency_quantiles": "statistical baseline: grouped empirical quantiles, not a supervised ML model",
            "resource_class": "rule taxonomy: no independent telemetry label yet",
        },
        "latency_bucket": {
            "models": {
                "logistic": "LogisticRegression(class_weight='balanced')",
                "historical_random_forest": "RandomForestClassifier with tool/operation historical bucket priors",
                "baseline": "per-tool most-common bucket",
            },
            "buckets": ["<100ms", "0.1-1s", "1-10s", "10-60s", ">60s"],
            "features": [
                "dataset",
                "tool",
                "operation",
                "tool_family",
                "resource_class_hint",
                "command_len",
                "argv_count",
                "flag_count",
                "has_pipe",
                "has_recursive_hint",
                "include_count",
                "path_token_count",
                "input_key_count",
                "call_index",
                "history_len",
                "last_5_tools",
            ],
            "offline": {
                "logistic": evaluate_bucket_model(test, bucket_model, bucket_baseline),
                "historical_random_forest": evaluate_bucket_model(test, bucket_rf, bucket_baseline),
            },
            "online_calibration": {
                "logistic": replay_bucket_calibration(test, bucket_model),
                "historical_random_forest": replay_bucket_calibration(test, bucket_rf),
            },
        },
        "next_tool": {
            "model": "LogisticRegression over current tool, operation, family, and last tools",
            "baselines": {
                "markov_1": "first-order Markov current_tool -> next_tool",
                "history_markov": "longest suffix over last tools -> next_tool",
            },
            "offline": {
                "logistic": evaluate_next_tool(test, next_model, markov),
                "history_markov": evaluate_next_tool(test, history_markov, markov),
            },
        },
        "latency_quantile_profile": {
            "model": "Group empirical P50/P90/P99 by operation",
            "offline": regression_metrics(test, quantile_model.predict),
            "online_calibration": replay_calibration(test, quantile_model.predict),
        },
    }
