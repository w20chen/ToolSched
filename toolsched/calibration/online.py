from __future__ import annotations

from collections import defaultdict, deque

from ..schema import ToolCostDistribution, ToolSample


class EwmaScaleCalibrator:
    def __init__(self, alpha: float = 0.15, min_scale: float = 0.2, max_scale: float = 5.0) -> None:
        self.alpha = alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)

    def group(self, sample: ToolSample) -> tuple[str, str]:
        machine = str(sample.resources.get("machine_profile", "unknown"))
        return (sample.tool_family, machine)

    def apply(self, sample: ToolSample, pred: ToolCostDistribution) -> ToolCostDistribution:
        c = self.scale[self.group(sample)]
        return ToolCostDistribution(
            latency_p50_ms=pred.latency_p50_ms * c,
            latency_p90_ms=pred.latency_p90_ms * c,
            latency_p99_ms=pred.latency_p99_ms * c,
            cpu_time_ms=pred.cpu_time_ms,
            memory_bytes=pred.memory_bytes,
            working_set_bytes=pred.working_set_bytes,
            io_bytes=pred.io_bytes,
            resource_class=pred.resource_class,
            uncertainty=pred.uncertainty * c,
        )

    def update(self, sample: ToolSample, pred_before: ToolCostDistribution) -> None:
        if sample.duration_ms is None or pred_before.latency_p50_ms <= 0:
            return
        ratio = sample.duration_ms / pred_before.latency_p50_ms
        ratio = min(self.max_scale, max(self.min_scale, ratio))
        key = self.group(sample)
        old = self.scale[key]
        self.scale[key] = (1 - self.alpha) * old + self.alpha * ratio


class QuantileCoverageCalibrator:
    def __init__(self, window: int = 100, target_p90: float = 0.90, step: float = 0.05) -> None:
        self.window = window
        self.target_p90 = target_p90
        self.step = step
        self.violations: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))
        self.tail_scale: dict[str, float] = defaultdict(lambda: 1.0)

    def key(self, sample: ToolSample) -> str:
        return sample.tool_family

    def apply(self, sample: ToolSample, pred: ToolCostDistribution) -> ToolCostDistribution:
        scale = self.tail_scale[self.key(sample)]
        return ToolCostDistribution(
            latency_p50_ms=pred.latency_p50_ms,
            latency_p90_ms=pred.latency_p90_ms * scale,
            latency_p99_ms=pred.latency_p99_ms * scale,
            cpu_time_ms=pred.cpu_time_ms,
            memory_bytes=pred.memory_bytes,
            working_set_bytes=pred.working_set_bytes,
            io_bytes=pred.io_bytes,
            resource_class=pred.resource_class,
            uncertainty=pred.uncertainty * scale,
        )

    def update(self, sample: ToolSample, pred: ToolCostDistribution) -> None:
        if sample.duration_ms is None:
            return
        key = self.key(sample)
        self.violations[key].append(int(sample.duration_ms > pred.latency_p90_ms))
        if len(self.violations[key]) < max(10, self.window // 5):
            return
        coverage = 1.0 - sum(self.violations[key]) / len(self.violations[key])
        if coverage < self.target_p90:
            self.tail_scale[key] *= 1 + self.step
        elif coverage > self.target_p90 + 0.05:
            self.tail_scale[key] = max(1.0, self.tail_scale[key] * (1 - self.step))
