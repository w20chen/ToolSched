from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..features.command import extract_command_features, normalize_operation
from ..features.resource_class import infer_resource_class
from ..io import read_json
from ..schema import ToolSample
from .discovery import AttemptPath, discover_attempts


def load_attempt(attempt: AttemptPath, history_k: int = 5) -> list[ToolSample]:
    tool_path = attempt.path / "tool_calls.json"
    try:
        calls = read_json(tool_path)
    except Exception:
        return []
    if not isinstance(calls, list):
        return []
    attempt_resources = _load_resources(attempt.path / "resources.json")
    samples: list[ToolSample] = []
    tools_seen: list[str] = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "unknown")
        payload = dict(call.get("input") or {})
        preview = str(call.get("result_preview") or "")
        operation, family = normalize_operation(tool, payload)
        features = extract_command_features(tool, payload, preview)
        resource_class = infer_resource_class(family, operation, features)
        features.update(
            {
                "call_index": idx,
                "history_len": min(len(tools_seen), history_k),
                "resource_class_heuristic": resource_class,
                "operation": operation,
                "tool_family": family,
            }
        )
        next_tool = None
        if idx + 1 < len(calls) and isinstance(calls[idx + 1], dict):
            next_tool = str(calls[idx + 1].get("tool") or "unknown")
        sample_resources = dict(attempt_resources)
        # Placement state must be a pre-launch snapshot.  Do not silently use
        # resource samples collected during/after the call as decision input.
        prelaunch = call.get("prelaunch_resources")
        if isinstance(prelaunch, dict):
            sample_resources.update(prelaunch)
        if isinstance(call.get("placement_candidates"), (list, dict)):
            sample_resources["placement_candidates"] = call["placement_candidates"]

        labels: dict[str, Any] = {"resource_class": resource_class}
        placement_costs = call.get("placement_costs")
        if placement_costs is None and isinstance(call.get("labels"), dict):
            placement_costs = call["labels"].get("placement_costs")
        if isinstance(placement_costs, dict):
            labels["placement_costs"] = placement_costs

        sample = ToolSample(
            sample_id=f"{attempt.dataset}/{attempt.case_id}/{attempt.attempt_id}/{call.get('id') or idx}",
            dataset=attempt.dataset,
            case_id=attempt.case_id,
            attempt_id=attempt.attempt_id,
            tool=tool,
            operation=operation,
            tool_family=family,
            timestamp=call.get("timestamp"),
            duration_ms=_duration(call),
            input=payload,
            result_preview=preview[:2048],
            features=features,
            labels=labels,
            resources=sample_resources,
            history=tools_seen[-history_k:],
            next_tool=next_tool,
        )
        samples.append(sample)
        tools_seen.append(tool)
    return samples


def load_datasets(
    dataset_root: Path,
    limit_attempts: int | None = None,
    include_datasets: set[str] | None = None,
    min_duration_ms: float | None = None,
) -> Iterable[ToolSample]:
    kept = 0
    for attempt in discover_attempts(dataset_root):
        if include_datasets and attempt.dataset not in include_datasets:
            continue
        kept += 1
        if limit_attempts is not None and kept > limit_attempts:
            break
        for sample in load_attempt(attempt):
            if min_duration_ms is not None:
                if sample.duration_ms is None or sample.duration_ms < min_duration_ms:
                    continue
            yield sample


def _duration(call: dict[str, Any]) -> float | None:
    value = call.get("duration_ms")
    if value is None and call.get("timestamp") and call.get("end_timestamp"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_resources(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    # Preserve explicitly named pre-launch placement inputs.  Aggregate
    # samples below describe the attempt and are not treated as a per-call
    # placement snapshot.
    out = {
        key: payload[key]
        for key in ("machine_profile", "cpu_parallelism", "observed_parallelism", "max_thread_count")
        if key in payload
    }
    if isinstance(payload.get("placement_candidates"), (list, dict)):
        out["placement_candidates"] = payload["placement_candidates"]

    samples = payload.get("samples")
    if isinstance(samples, list) and samples:
        cpu = [_safe_float(s.get("cpu_percent")) for s in samples if isinstance(s, dict)]
        ipc = [_safe_float(s.get("ipc")) for s in samples if isinstance(s, dict)]
        ctx = [_safe_float(s.get("context_switches")) for s in samples if isinstance(s, dict)]
        out.update({
            "sample_count": len(samples),
            "cpu_percent_mean": _mean(cpu),
            "cpu_percent_max": max(cpu) if cpu else None,
            "ipc_mean": _mean(ipc),
            "context_switches_max": max(ctx) if ctx else None,
        })
    return out


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
