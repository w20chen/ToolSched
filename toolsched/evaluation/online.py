from __future__ import annotations

from typing import Callable

from ..calibration.online import EwmaScaleCalibrator, QuantileCoverageCalibrator
from ..evaluation.metrics import mean, pinball_loss
from ..schema import ToolCostDistribution, ToolSample


def replay_calibration(
    samples: list[ToolSample],
    predict: Callable[[ToolSample], ToolCostDistribution],
) -> dict:
    rows = [s for s in samples if s.duration_ms is not None]
    scale = EwmaScaleCalibrator()
    coverage = QuantileCoverageCalibrator()
    before_abs = []
    after_abs = []
    before_pin90 = []
    after_pin90 = []
    before_cover90 = 0
    after_cover90 = 0
    for s in rows:
        raw = predict(s)
        scaled = scale.apply(s, raw)
        calibrated = coverage.apply(s, scaled)
        y = float(s.duration_ms)
        before_abs.append(abs(y - raw.latency_p50_ms))
        after_abs.append(abs(y - calibrated.latency_p50_ms))
        before_pin90.append(pinball_loss(y, raw.latency_p90_ms, 0.9))
        after_pin90.append(pinball_loss(y, calibrated.latency_p90_ms, 0.9))
        before_cover90 += int(y <= raw.latency_p90_ms)
        after_cover90 += int(y <= calibrated.latency_p90_ms)
        scale.update(s, raw)
        coverage.update(s, calibrated)
    n = len(rows)
    return {
        "n": n,
        "mae_before_ms": mean(before_abs),
        "mae_after_ms": mean(after_abs),
        "pinball_p90_before": mean(before_pin90),
        "pinball_p90_after": mean(after_pin90),
        "coverage_p90_before": before_cover90 / n if n else 0.0,
        "coverage_p90_after": after_cover90 / n if n else 0.0,
        "ewma_groups": len(scale.scale),
        "tail_groups": len(coverage.tail_scale),
    }

