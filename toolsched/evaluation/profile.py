from __future__ import annotations

from collections import defaultdict

from ..models.baselines import quantile
from ..schema import ToolSample


def tool_profiles(samples: list[ToolSample]) -> dict:
    groups: dict[tuple[str, str], list[ToolSample]] = defaultdict(list)
    for s in samples:
        groups[(s.tool_family, s.operation)].append(s)
    rows = []
    for (family, operation), items in sorted(groups.items()):
        durations = [float(s.duration_ms) for s in items if s.duration_ms is not None]
        command_len = [float(s.features.get("command_len", 0)) for s in items]
        preview_len = [float(s.features.get("preview_len", 0)) for s in items]
        rows.append(
            {
                "tool_family": family,
                "operation": operation,
                "count": len(items),
                "latency_p50_ms": quantile(durations, 0.50),
                "latency_p90_ms": quantile(durations, 0.90),
                "latency_p99_ms": quantile(durations, 0.99),
                "mean_command_len": sum(command_len) / len(command_len) if command_len else 0.0,
                "mean_preview_len": sum(preview_len) / len(preview_len) if preview_len else 0.0,
                "recursive_rate": sum(1 for s in items if s.features.get("has_recursive_hint")) / len(items),
                "pipe_rate": sum(1 for s in items if s.features.get("has_pipe")) / len(items),
            }
        )
    return {"profiles": rows, "profile_count": len(rows), "sample_count": len(samples)}

