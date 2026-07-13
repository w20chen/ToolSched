from __future__ import annotations

from collections import Counter, defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ..features.ml import SampleFeatureEncoder
from ..schema import ToolSample


BUCKETS = (
    (0.0, 100.0, "lt_100ms", "<100ms"),
    (100.0, 1000.0, "100ms_1s", "0.1-1s"),
    (1000.0, 10000.0, "1s_10s", "1-10s"),
    (10000.0, 60000.0, "10s_60s", "10-60s"),
    (60000.0, float("inf"), "gt_60s", ">60s"),
)


def bucket_index(duration_ms: float) -> int:
    for idx, (lo, hi, _, _) in enumerate(BUCKETS):
        if lo <= duration_ms < hi:
            return idx
    return len(BUCKETS) - 1


def bucket_label(index: int) -> str:
    return BUCKETS[index][3]


class PerToolBucketBaseline:
    def __init__(self) -> None:
        self.by_tool: dict[str, int] = {}
        self.global_bucket = 0

    def fit(self, samples: list[ToolSample]) -> "PerToolBucketBaseline":
        global_counts: Counter[int] = Counter()
        grouped: dict[str, Counter[int]] = defaultdict(Counter)
        for sample in samples:
            if sample.duration_ms is None:
                continue
            idx = bucket_index(sample.duration_ms)
            global_counts[idx] += 1
            grouped[sample.tool][idx] += 1
        self.global_bucket = global_counts.most_common(1)[0][0] if global_counts else 0
        self.by_tool = {tool: counts.most_common(1)[0][0] for tool, counts in grouped.items()}
        return self

    def predict_index(self, sample: ToolSample) -> int:
        return self.by_tool.get(sample.tool, self.global_bucket)

    def predict(self, sample: ToolSample) -> str:
        return bucket_label(self.predict_index(sample))


class LatencyBucketModel:
    """Simple supervised latency-bucket classifier."""

    def __init__(self, history_k: int = 5) -> None:
        self.encoder = SampleFeatureEncoder(history_k=history_k)
        self.model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=7,
        )
        self.classes_: list[int] = []

    def fit(self, samples: list[ToolSample]) -> "LatencyBucketModel":
        rows = [s for s in samples if s.duration_ms is not None]
        x = self.encoder.fit_transform(rows)
        y = [bucket_index(float(s.duration_ms)) for s in rows]
        self.model.fit(x, y)
        self.classes_ = [int(c) for c in self.model.classes_]
        return self

    def predict_index(self, sample: ToolSample) -> int:
        x = self.encoder.transform([sample])
        return int(self.model.predict(x)[0])

    def predict(self, sample: ToolSample) -> str:
        return bucket_label(self.predict_index(sample))

    def predict_proba_dict(self, sample: ToolSample) -> dict[str, float]:
        x = self.encoder.transform([sample])
        probs = self.model.predict_proba(x)[0]
        out = {bucket_label(i): 0.0 for i in range(len(BUCKETS))}
        for cls, prob in zip(self.model.classes_, probs):
            out[bucket_label(int(cls))] = float(prob)
        return out


class HistoricalBucketFeatureModel:
    """RandomForest classifier with online-available historical priors."""

    def __init__(self, history_k: int = 5) -> None:
        self.encoder = SampleFeatureEncoder(history_k=history_k)
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=7,
            n_jobs=-1,
        )
        self.tool_priors: dict[str, dict[str, float]] = {}
        self.operation_priors: dict[str, dict[str, float]] = {}
        self.global_prior: dict[str, float] = {}

    def fit(self, samples: list[ToolSample]) -> "HistoricalBucketFeatureModel":
        rows = [s for s in samples if s.duration_ms is not None]
        self._fit_priors(rows)
        x = self.encoder.fit_transform([self._with_priors(s) for s in rows])
        y = [bucket_index(float(s.duration_ms)) for s in rows]
        self.model.fit(x, y)
        return self

    def predict_index(self, sample: ToolSample) -> int:
        x = self.encoder.transform([self._with_priors(sample)])
        return int(self.model.predict(x)[0])

    def predict(self, sample: ToolSample) -> str:
        return bucket_label(self.predict_index(sample))

    def predict_proba_dict(self, sample: ToolSample) -> dict[str, float]:
        x = self.encoder.transform([self._with_priors(sample)])
        probs = self.model.predict_proba(x)[0]
        out = {bucket_label(i): 0.0 for i in range(len(BUCKETS))}
        for cls, prob in zip(self.model.classes_, probs):
            out[bucket_label(int(cls))] = float(prob)
        return out

    def _fit_priors(self, samples: list[ToolSample]) -> None:
        by_tool: dict[str, list[float]] = defaultdict(list)
        by_op: dict[str, list[float]] = defaultdict(list)
        all_values: list[float] = []
        for sample in samples:
            if sample.duration_ms is None:
                continue
            value = float(sample.duration_ms)
            by_tool[sample.tool].append(value)
            by_op[sample.operation].append(value)
            all_values.append(value)
        self.global_prior = _duration_prior(all_values)
        self.tool_priors = {k: _duration_prior(v) for k, v in by_tool.items()}
        self.operation_priors = {k: _duration_prior(v) for k, v in by_op.items()}

    def _with_priors(self, sample: ToolSample) -> ToolSample:
        prior = dict(self.global_prior)
        for prefix, values in (
            ("tool", self.tool_priors.get(sample.tool)),
            ("operation", self.operation_priors.get(sample.operation)),
        ):
            source = values or self.global_prior
            for key, value in source.items():
                prior[f"{prefix}_{key}"] = value
        merged = dict(sample.features)
        merged.update(prior)
        return ToolSample(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            case_id=sample.case_id,
            attempt_id=sample.attempt_id,
            tool=sample.tool,
            operation=sample.operation,
            tool_family=sample.tool_family,
            timestamp=sample.timestamp,
            duration_ms=sample.duration_ms,
            input=sample.input,
            result_preview=sample.result_preview,
            features=merged,
            labels=sample.labels,
            resources=sample.resources,
            history=sample.history,
            next_tool=sample.next_tool,
        )


def _duration_prior(values: list[float]) -> dict[str, float]:
    if not values:
        return {f"prior_bucket_{i}": 0.0 for i in range(len(BUCKETS))} | {
            "prior_log_p50": 0.0,
            "prior_log_p90": 0.0,
            "prior_log_mean": 0.0,
        }
    values = sorted(values)
    counts = Counter(bucket_index(v) for v in values)
    n = len(values)
    return {
        **{f"prior_bucket_{i}": counts[i] / n for i in range(len(BUCKETS))},
        "prior_log_p50": _log1p(_quantile(values, 0.50)),
        "prior_log_p90": _log1p(_quantile(values, 0.90)),
        "prior_log_mean": _log1p(sum(values) / n),
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _log1p(value: float) -> float:
    import math

    return math.log1p(max(0.0, value))


def evaluate_bucket_model(test: list[ToolSample], model, baseline: PerToolBucketBaseline | None = None) -> dict:
    rows = [s for s in test if s.duration_ms is not None]
    y_true = [bucket_index(float(s.duration_ms)) for s in rows]
    y_pred = [model.predict_index(s) for s in rows]
    payload = _bucket_metrics(y_true, y_pred)
    if baseline is not None:
        y_base = [baseline.predict_index(s) for s in rows]
        payload["baseline"] = _bucket_metrics(y_true, y_base)
    return payload


def _bucket_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    if not y_true:
        return {}
    adjacent = sum(abs(a - b) <= 1 for a, b in zip(y_true, y_pred)) / len(y_true)
    severe_under = sum(b <= a - 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    long_true = [i for i, y in enumerate(y_true) if y >= 3]
    long_recall = (
        sum(y_pred[i] >= 3 for i in long_true) / len(long_true)
        if long_true else 0.0
    )
    per_bucket = {}
    for idx in range(len(BUCKETS)):
        ids = [i for i, y in enumerate(y_true) if y == idx]
        if not ids:
            continue
        per_bucket[bucket_label(idx)] = {
            "n": len(ids),
            "recall": sum(y_pred[i] == idx for i in ids) / len(ids),
        }
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(range(len(BUCKETS))), average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=list(range(len(BUCKETS))), average="weighted", zero_division=0),
        "adjacent_bucket_accuracy": adjacent,
        "severe_underprediction_rate": severe_under,
        "long_task_recall_ge_10s": long_recall,
        "per_bucket": per_bucket,
    }
