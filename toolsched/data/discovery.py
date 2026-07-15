from __future__ import annotations

import json
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


def estimate_sampling_interval_s(dataset_dir: Path, max_files: int = 10) -> float | None:
    """Estimate the median resource sampling interval for a dataset.

    Samples up to *max_files* ``resources.json`` files in the dataset
    directory, computes the median gap between consecutive ``epoch`` / 
    ``timestamp`` values, and returns the result in seconds.

    Returns ``None`` if no usable resource samples are found.
    """
    all_gaps: list[float] = []
    checked = 0
    for res_path in sorted(dataset_dir.rglob("resources.json")):
        if checked >= max_files:
            break
        checked += 1
        try:
            payload = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        samples = payload.get("samples") if isinstance(payload, dict) else None
        if not isinstance(samples, list) or len(samples) < 2:
            continue
        epochs: list[float] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            e = s.get("epoch")
            if e is None:
                from datetime import datetime

                t = s.get("timestamp")
                if t:
                    try:
                        e = datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        pass
            if e is not None:
                epochs.append(e)
        if len(epochs) < 2:
            continue
        gaps = [epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1)]
        all_gaps.extend(gaps)
    if not all_gaps:
        return None
    all_gaps.sort()
    return all_gaps[len(all_gaps) // 2]


def summarize_datasets(dataset_root: Path) -> dict:
    summary: dict[str, dict] = {}
    for attempt in discover_attempts(dataset_root):
        row = summary.setdefault(attempt.dataset, {"attempts": 0, "tool_calls_files": 0})
        row["attempts"] += 1
        row["tool_calls_files"] += 1

    # Annotate each dataset with its estimated resource sampling interval.
    for dataset_name in sorted(summary):
        dataset_dir = dataset_root / dataset_name
        if dataset_dir.is_dir():
            interval = estimate_sampling_interval_s(dataset_dir)
            summary[dataset_name]["resource_sampling_interval_s"] = interval

    return summary

