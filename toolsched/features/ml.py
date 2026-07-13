from __future__ import annotations

import math
from typing import Any

from sklearn.feature_extraction import DictVectorizer

from ..schema import ToolSample


NUMERIC_FEATURES = (
    "command_len",
    "argv_count",
    "flag_count",
    "include_count",
    "path_token_count",
    "input_key_count",
    "call_index",
    "history_len",
)

BOOLEAN_FEATURES = (
    "has_pipe",
    "has_recursive_hint",
)


class SampleFeatureEncoder:
    """DictVectorizer wrapper for online-available tool-call features."""

    def __init__(self, history_k: int = 5) -> None:
        self.history_k = history_k
        self.vectorizer = DictVectorizer(sparse=True)

    def fit(self, samples: list[ToolSample]) -> "SampleFeatureEncoder":
        self.vectorizer.fit([self.to_dict(s) for s in samples])
        return self

    def transform(self, samples: list[ToolSample]):
        return self.vectorizer.transform([self.to_dict(s) for s in samples])

    def fit_transform(self, samples: list[ToolSample]):
        self.fit(samples)
        return self.transform(samples)

    def to_dict(self, sample: ToolSample) -> dict[str, float]:
        row: dict[str, float] = {
            f"dataset={sample.dataset}": 1.0,
            f"tool={sample.tool}": 1.0,
            f"operation={sample.operation}": 1.0,
            f"tool_family={sample.tool_family}": 1.0,
        }
        for name in NUMERIC_FEATURES:
            row[name] = _log1p(sample.features.get(name))
        for name, value in sample.features.items():
            if name.startswith("prior_"):
                row[name] = _safe_float(value)
        for name in BOOLEAN_FEATURES:
            row[name] = 1.0 if sample.features.get(name) else 0.0
        for offset, tool in enumerate(reversed(sample.history[-self.history_k:]), start=1):
            row[f"prev{offset}_tool={tool}"] = 1.0
        if sample.history:
            row[f"latest_tool={sample.history[-1]}"] = 1.0
        return row

    @property
    def feature_names(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())


def _log1p(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return math.log1p(max(0.0, number))


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
