from __future__ import annotations

from collections import defaultdict

from ..models.buckets import BUCKETS, bucket_index, bucket_label
from ..schema import ToolSample


class BucketPriorCalibrator:
    """Streaming multiplicative correction for bucket probabilities."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.weights: dict[tuple[str, int], float] = defaultdict(lambda: 1.0)

    def group(self, sample: ToolSample) -> str:
        return sample.tool_family

    def apply(self, sample: ToolSample, probs: dict[str, float]) -> dict[str, float]:
        group = self.group(sample)
        adjusted = {}
        for idx in range(len(BUCKETS)):
            label = bucket_label(idx)
            adjusted[label] = max(0.0, probs.get(label, 0.0) * self.weights[(group, idx)])
        total = sum(adjusted.values())
        if total <= 0:
            return {bucket_label(i): 1.0 / len(BUCKETS) for i in range(len(BUCKETS))}
        return {k: v / total for k, v in adjusted.items()}

    def update(self, sample: ToolSample, calibrated_probs: dict[str, float]) -> None:
        if sample.duration_ms is None:
            return
        group = self.group(sample)
        y = bucket_index(float(sample.duration_ms))
        for idx in range(len(BUCKETS)):
            target = 1.0 if idx == y else 0.0
            pred = calibrated_probs.get(bucket_label(idx), 0.0)
            self.weights[(group, idx)] *= 1.0 + self.alpha * (target - pred)
            self.weights[(group, idx)] = min(5.0, max(0.2, self.weights[(group, idx)]))


def replay_bucket_calibration(samples: list[ToolSample], model) -> dict:
    rows = [s for s in samples if s.duration_ms is not None]
    calibrator = BucketPriorCalibrator()
    before_correct = 0
    after_correct = 0
    before_long_recall_num = 0
    after_long_recall_num = 0
    long_den = 0
    before_severe = 0
    after_severe = 0
    for sample in rows:
        y = bucket_index(float(sample.duration_ms))
        probs = model.predict_proba_dict(sample)
        before = _argmax_label(probs)
        calibrated = calibrator.apply(sample, probs)
        after = _argmax_label(calibrated)
        before_correct += int(before == y)
        after_correct += int(after == y)
        before_severe += int(before <= y - 2)
        after_severe += int(after <= y - 2)
        if y >= 3:
            long_den += 1
            before_long_recall_num += int(before >= 3)
            after_long_recall_num += int(after >= 3)
        calibrator.update(sample, calibrated)
    n = len(rows)
    return {
        "n": n,
        "accuracy_before": before_correct / n if n else 0.0,
        "accuracy_after": after_correct / n if n else 0.0,
        "severe_underprediction_before": before_severe / n if n else 0.0,
        "severe_underprediction_after": after_severe / n if n else 0.0,
        "long_task_recall_before": before_long_recall_num / long_den if long_den else 0.0,
        "long_task_recall_after": after_long_recall_num / long_den if long_den else 0.0,
        "groups": len({sample.tool_family for sample in rows}),
    }


def _argmax_label(probs: dict[str, float]) -> int:
    label = max(probs, key=probs.get)
    for idx in range(len(BUCKETS)):
        if bucket_label(idx) == label:
            return idx
    return 0

