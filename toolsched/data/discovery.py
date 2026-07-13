from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptPath:
    dataset: str
    case_id: str
    attempt_id: str
    path: Path


def discover_attempts(dataset_root: Path, limit: int | None = None) -> list[AttemptPath]:
    attempts: list[AttemptPath] = []
    for dataset_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for tool_calls in dataset_dir.rglob("tool_calls.json"):
            attempt_dir = tool_calls.parent
            attempt_id = attempt_dir.name
            case_id = attempt_dir.parent.name
            attempts.append(AttemptPath(dataset_dir.name, case_id, attempt_id, attempt_dir))
            if limit is not None and len(attempts) >= limit:
                return attempts
    return attempts


def summarize_datasets(dataset_root: Path) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for attempt in discover_attempts(dataset_root):
        row = summary.setdefault(attempt.dataset, {"attempts": 0, "tool_calls_files": 0})
        row["attempts"] += 1
        row["tool_calls_files"] += 1
    return summary

