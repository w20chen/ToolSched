from __future__ import annotations

from collections import Counter
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ..features.ml import SampleFeatureEncoder
from ..models.baselines import NextToolMarkovModel
from ..schema import ToolSample


class NextToolLogisticModel:
    def __init__(self, history_k: int = 5, min_count: int = 5) -> None:
        self.history_k = history_k
        self.min_count = min_count
        self.encoder = SampleFeatureEncoder(history_k=history_k)
        self.model = LogisticRegression(
            C=1.0,
            class_weight=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=7,
        )
        self.common_tools: set[str] = set()
        self.fallback = NextToolMarkovModel()

    def fit(self, samples: list[ToolSample]) -> "NextToolLogisticModel":
        rows = [s for s in samples if s.next_tool]
        counts = Counter(str(s.next_tool) for s in rows)
        self.common_tools = {tool for tool, n in counts.items() if n >= self.min_count}
        fit_rows = [s for s in rows if s.next_tool in self.common_tools]
        self.fallback.fit(samples)
        if len({s.next_tool for s in fit_rows}) < 2:
            return self
        x = self.encoder.fit_transform(fit_rows)
        y = [str(s.next_tool) for s in fit_rows]
        self.model.fit(x, y)
        return self

    def predict(self, sample: ToolSample) -> tuple[str | None, float]:
        if not hasattr(self.model, "classes_") or len(getattr(self.model, "classes_", [])) < 2:
            return self.fallback.predict(sample)
        x = self.encoder.transform([sample])
        probs = self.model.predict_proba(x)[0]
        best = int(probs.argmax())
        return str(self.model.classes_[best]), float(probs[best])

    def predict_topk(self, sample: ToolSample, k: int = 5) -> list[tuple[str, float]]:
        if not hasattr(self.model, "classes_") or len(getattr(self.model, "classes_", [])) < 2:
            return self.fallback.predict_topk(sample, k)
        x = self.encoder.transform([sample])
        probs = self.model.predict_proba(x)[0]
        order = probs.argsort()[::-1][:k]
        return [(str(self.model.classes_[idx]), float(probs[idx])) for idx in order]


class HistoryMarkovModel:
    def __init__(self, max_order: int = 5) -> None:
        self.max_order = max_order
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.fallback = NextToolMarkovModel()

    def fit(self, samples: list[ToolSample]) -> "HistoryMarkovModel":
        self.fallback.fit(samples)
        for sample in samples:
            if not sample.next_tool:
                continue
            seq = list(sample.history[-self.max_order:]) + [sample.tool]
            for order in range(1, min(len(seq), self.max_order) + 1):
                key = tuple(seq[-order:])
                self.transitions[key][str(sample.next_tool)] += 1
        return self

    def predict(self, sample: ToolSample) -> tuple[str | None, float]:
        seq = list(sample.history[-self.max_order:]) + [sample.tool]
        for order in range(min(len(seq), self.max_order), 0, -1):
            counts = self.transitions.get(tuple(seq[-order:]))
            if counts:
                total = sum(counts.values())
                tool, n = counts.most_common(1)[0]
                return tool, n / total if total else 0.0
        return self.fallback.predict(sample)

    def predict_topk(self, sample: ToolSample, k: int = 5) -> list[tuple[str, float]]:
        seq = list(sample.history[-self.max_order:]) + [sample.tool]
        for order in range(min(len(seq), self.max_order), 0, -1):
            counts = self.transitions.get(tuple(seq[-order:]))
            if counts:
                total = sum(counts.values())
                return [(tool, n / total) for tool, n in counts.most_common(k)] if total else []
        return self.fallback.predict_topk(sample, k)


def evaluate_next_tool(test: list[ToolSample], model, baseline: NextToolMarkovModel | None = None) -> dict:
    rows = [s for s in test if s.next_tool]
    if not rows:
        return {}
    y_true = [str(s.next_tool) for s in rows]
    y_pred = [model.predict(s)[0] for s in rows]
    payload = {
        "n": len(rows),
        "top1_accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": _topk_accuracy(rows, model, 3),
        "top5_accuracy": _topk_accuracy(rows, model, 5),
        "top10_accuracy": _topk_accuracy(rows, model, 10),
    }
    if baseline is not None:
        y_base = [baseline.predict(s)[0] for s in rows]
        payload["baseline"] = {
            "top1_accuracy": accuracy_score(y_true, y_base),
            "top3_accuracy": _topk_accuracy(rows, baseline, 3),
            "top5_accuracy": _topk_accuracy(rows, baseline, 5),
            "top10_accuracy": _topk_accuracy(rows, baseline, 10),
        }
    return payload


def _topk_accuracy(rows: list[ToolSample], model, k: int) -> float:
    hits = 0
    for sample in rows:
        top = []
        if hasattr(model, "predict_topk"):
            top = [tool for tool, _ in model.predict_topk(sample, k)]
        else:
            pred, _ = model.predict(sample)
            top = [pred] if pred else []
        hits += int(str(sample.next_tool) in top)
    return hits / len(rows) if rows else 0.0
