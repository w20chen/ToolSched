from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSample:
    sample_id: str
    dataset: str
    case_id: str
    attempt_id: str
    tool: str
    operation: str
    tool_family: str
    timestamp: str | None
    duration_ms: float | None
    input: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    next_tool: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "attempt_id": self.attempt_id,
            "tool": self.tool,
            "operation": self.operation,
            "tool_family": self.tool_family,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "result_preview": self.result_preview,
            "features": self.features,
            "labels": self.labels,
            "resources": self.resources,
            "history": self.history,
            "next_tool": self.next_tool,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "ToolSample":
        return cls(
            sample_id=str(row["sample_id"]),
            dataset=str(row.get("dataset", "")),
            case_id=str(row.get("case_id", "")),
            attempt_id=str(row.get("attempt_id", "")),
            tool=str(row.get("tool", "")),
            operation=str(row.get("operation", "")),
            tool_family=str(row.get("tool_family", "")),
            timestamp=row.get("timestamp"),
            duration_ms=_float_or_none(row.get("duration_ms")),
            input=dict(row.get("input") or {}),
            result_preview=str(row.get("result_preview") or ""),
            features=dict(row.get("features") or {}),
            labels=dict(row.get("labels") or {}),
            resources=dict(row.get("resources") or {}),
            history=list(row.get("history") or []),
            next_tool=row.get("next_tool"),
        )


@dataclass(frozen=True)
class ToolCostDistribution:
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    cpu_time_ms: float = 0.0
    memory_bytes: float = 0.0
    working_set_bytes: float = 0.0
    io_bytes: float = 0.0
    resource_class: str = "unknown"
    uncertainty: float = 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

