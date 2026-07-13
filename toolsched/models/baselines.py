from __future__ import annotations

from collections import Counter, defaultdict, deque
from statistics import median

from ..schema import ToolCostDistribution, ToolSample


def quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1 - frac) + clean[hi] * frac


class GlobalQuantileModel:
    def __init__(self) -> None:
        self.values: list[float] = []

    def fit(self, samples: list[ToolSample]) -> "GlobalQuantileModel":
        self.values = [s.duration_ms for s in samples if s.duration_ms is not None]
        return self

    def predict(self, sample: ToolSample) -> ToolCostDistribution:
        return ToolCostDistribution(
            latency_p50_ms=quantile(self.values, 0.50),
            latency_p90_ms=quantile(self.values, 0.90),
            latency_p99_ms=quantile(self.values, 0.99),
            resource_class=sample.labels.get("resource_class", "unknown"),
            uncertainty=max(0.0, quantile(self.values, 0.90) - quantile(self.values, 0.50)),
        )


class GroupQuantileModel:
    def __init__(self, keys: tuple[str, ...] = ("tool",)) -> None:
        self.keys = keys
        self.global_model = GlobalQuantileModel()
        self.groups: dict[tuple, list[float]] = {}

    def fit(self, samples: list[ToolSample]) -> "GroupQuantileModel":
        self.global_model.fit(samples)
        groups: dict[tuple, list[float]] = defaultdict(list)
        for s in samples:
            if s.duration_ms is not None:
                groups[self._key(s)].append(s.duration_ms)
        self.groups = dict(groups)
        return self

    def predict(self, sample: ToolSample) -> ToolCostDistribution:
        values = self.groups.get(self._key(sample))
        if not values:
            return self.global_model.predict(sample)
        return ToolCostDistribution(
            latency_p50_ms=quantile(values, 0.50),
            latency_p90_ms=quantile(values, 0.90),
            latency_p99_ms=quantile(values, 0.99),
            resource_class=sample.labels.get("resource_class", "unknown"),
            uncertainty=max(0.0, quantile(values, 0.90) - quantile(values, 0.50)),
        )

    def _key(self, sample: ToolSample) -> tuple:
        vals = []
        for key in self.keys:
            if key == "resource_class":
                vals.append(sample.labels.get("resource_class", "unknown"))
            elif key == "operation":
                vals.append(sample.operation)
            elif key == "family":
                vals.append(sample.tool_family)
            else:
                vals.append(getattr(sample, key))
        return tuple(vals)


class EwmaToolModel:
    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha
        self.global_median = 0.0
        self.state: dict[str, float] = {}

    def fit(self, samples: list[ToolSample]) -> "EwmaToolModel":
        values = [s.duration_ms for s in samples if s.duration_ms is not None]
        self.global_median = median(values) if values else 0.0
        for s in samples:
            if s.duration_ms is None:
                continue
            old = self.state.get(s.tool, self.global_median)
            self.state[s.tool] = (1 - self.alpha) * old + self.alpha * s.duration_ms
        return self

    def predict(self, sample: ToolSample) -> ToolCostDistribution:
        p50 = self.state.get(sample.tool, self.global_median)
        return ToolCostDistribution(
            latency_p50_ms=p50,
            latency_p90_ms=p50 * 1.5,
            latency_p99_ms=p50 * 2.0,
            resource_class=sample.labels.get("resource_class", "unknown"),
            uncertainty=p50 * 0.5,
        )


class NextToolMarkovModel:
    def __init__(self) -> None:
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def fit(self, samples: list[ToolSample]) -> "NextToolMarkovModel":
        for s in samples:
            if not s.next_tool:
                continue
            prev = s.tool
            self.transitions[prev][s.next_tool] += 1
            self.global_counts[s.next_tool] += 1
        return self

    def predict(self, sample: ToolSample) -> tuple[str | None, float]:
        counts = self.transitions.get(sample.tool) or self.global_counts
        if not counts:
            return None, 0.0
        total = sum(counts.values())
        tool, n = counts.most_common(1)[0]
        return tool, n / total if total else 0.0

    def predict_topk(self, sample: ToolSample, k: int = 5) -> list[tuple[str, float]]:
        counts = self.transitions.get(sample.tool) or self.global_counts
        total = sum(counts.values())
        if not counts or total <= 0:
            return []
        return [(tool, n / total) for tool, n in counts.most_common(k)]
