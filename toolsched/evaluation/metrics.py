from __future__ import annotations

import math
from typing import Callable

from ..schema import ToolCostDistribution, ToolSample


def pinball_loss(y: float, pred: float, q: float) -> float:
    err = y - pred
    return max(q * err, (q - 1) * err)


def regression_metrics(
    samples: list[ToolSample],
    predict: Callable[[ToolSample], ToolCostDistribution],
) -> dict:
    rows = [s for s in samples if s.duration_ms is not None]
    if not rows:
        return {}
    abs_err_p50 = []
    abs_pct = []
    pin50 = []
    pin90 = []
    pin99 = []
    cover90 = 0
    cover99 = 0
    for s in rows:
        pred = predict(s)
        y = float(s.duration_ms)
        abs_err_p50.append(abs(y - pred.latency_p50_ms))
        if y > 0:
            abs_pct.append(abs(y - pred.latency_p50_ms) / y)
        pin50.append(pinball_loss(y, pred.latency_p50_ms, 0.50))
        pin90.append(pinball_loss(y, pred.latency_p90_ms, 0.90))
        pin99.append(pinball_loss(y, pred.latency_p99_ms, 0.99))
        cover90 += int(y <= pred.latency_p90_ms)
        cover99 += int(y <= pred.latency_p99_ms)
    return {
        "n": len(rows),
        "mae_ms": mean(abs_err_p50),
        "mape": mean(abs_pct),
        "pinball_p50": mean(pin50),
        "pinball_p90": mean(pin90),
        "pinball_p99": mean(pin99),
        "coverage_p90": cover90 / len(rows),
        "coverage_p99": cover99 / len(rows),
    }


def next_tool_metrics(samples: list[ToolSample], predict_next: Callable[[ToolSample], tuple[str | None, float]]) -> dict:
    rows = [s for s in samples if s.next_tool]
    if not rows:
        return {}
    correct = 0
    confs = []
    for s in rows:
        yhat, conf = predict_next(s)
        correct += int(yhat == s.next_tool)
        confs.append(conf)
    return {"n": len(rows), "top1_accuracy": correct / len(rows), "mean_confidence": mean(confs)}


def mean(values: list[float]) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else 0.0

