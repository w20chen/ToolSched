from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolsched.features.command import extract_command_features, normalize_operation


@dataclass
class ResourceSample:
    epoch: float
    cpu_percent: float | None
    mem_mib: float | None
    disk_bytes: float | None
    net_bytes: float | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--top-operations", type=int, default=18)
    args = parser.parse_args()

    root = Path(args.datasets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root, args.max_attempts)
    write_rows(out_dir / "operation_resource_rows.csv", rows)
    write_summary(out_dir / "operation_resource_summary.csv", rows)
    plot_boxplots(out_dir / "operation_resource_boxplots.png", rows, args.top_operations)
    print(json.dumps({
        "rows": len(rows),
        "operations": len({row["operation"] for row in rows}),
        "out_dir": str(out_dir),
        "boxplot": str(out_dir / "operation_resource_boxplots.png"),
        "rows_csv": str(out_dir / "operation_resource_rows.csv"),
        "summary_csv": str(out_dir / "operation_resource_summary.csv"),
    }, indent=2))


def collect_rows(root: Path, max_attempts: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempts_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "tool_calls.json" not in filenames:
            continue
        attempt_dir = Path(dirpath)
        tool_path = attempt_dir / "tool_calls.json"
        resources_path = attempt_dir / "resources.json"
        try:
            calls = json.loads(tool_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(calls, list) or not calls:
            continue
        resources = load_resource_samples(resources_path)
        dataset = dataset_name(root, attempt_dir)
        attempts_seen += 1
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            row = row_for_call(dataset, attempt_dir, idx, call, resources)
            if row is not None:
                rows.append(row)
        if max_attempts is not None and attempts_seen >= max_attempts:
            break
    return rows


def row_for_call(
    dataset: str,
    attempt_dir: Path,
    idx: int,
    call: dict[str, Any],
    resources: list[ResourceSample],
) -> dict[str, Any] | None:
    tool = str(call.get("tool") or "unknown")
    payload = dict(call.get("input") or {})
    preview = str(call.get("result_preview") or "")
    operation, family = normalize_operation(tool, payload)
    features = extract_command_features(tool, payload, preview)

    start = parse_epoch(call.get("timestamp"))
    duration_ms = safe_float(call.get("duration_ms"))
    end = parse_epoch(call.get("end_timestamp"))
    if end is None and start is not None and duration_ms is not None:
        end = start + max(0.0, duration_ms) / 1000.0
    if duration_ms is None and start is not None and end is not None:
        duration_ms = max(0.0, (end - start) * 1000.0)

    stats = resource_stats(resources, start, end)
    return {
        "dataset": dataset,
        "attempt_dir": str(attempt_dir),
        "call_index": idx,
        "tool": tool,
        "operation": operation,
        "tool_family": family,
        "duration_ms": duration_ms,
        "cpu_percent_mean": stats["cpu_percent_mean"],
        "memory_mib_mean": stats["memory_mib_mean"],
        "memory_mib_max": stats["memory_mib_max"],
        "network_io_bytes": stats["network_io_bytes"],
        "disk_io_bytes": stats["disk_io_bytes"],
        "resource_sample_count": stats["resource_sample_count"],
        "has_pipe": features.get("has_pipe"),
        "has_recursive_hint": features.get("has_recursive_hint"),
    }


def load_resource_samples(path: Path) -> list[ResourceSample]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return []
    out: list[ResourceSample] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        epoch = safe_float(sample.get("epoch"))
        if epoch is None:
            epoch = parse_epoch(sample.get("timestamp"))
        if epoch is None:
            continue
        disk_read = safe_float(sample.get("disk_read_bytes")) or 0.0
        disk_write = safe_float(sample.get("disk_write_bytes")) or 0.0
        net_rx = safe_float(sample.get("net_rx_bytes")) or 0.0
        net_tx = safe_float(sample.get("net_tx_bytes")) or 0.0
        out.append(ResourceSample(
            epoch=epoch,
            cpu_percent=safe_percent(sample.get("cpu_percent")),
            mem_mib=parse_mem_mib(sample.get("mem_usage")),
            disk_bytes=disk_read + disk_write,
            net_bytes=net_rx + net_tx,
        ))
    out.sort(key=lambda row: row.epoch)
    return out


def resource_stats(resources: list[ResourceSample], start: float | None, end: float | None) -> dict[str, Any]:
    empty = {
        "cpu_percent_mean": None,
        "memory_mib_mean": None,
        "memory_mib_max": None,
        "network_io_bytes": None,
        "disk_io_bytes": None,
        "resource_sample_count": 0,
    }
    if not resources or start is None or end is None:
        return empty
    if end < start:
        start, end = end, start
    if end == start:
        end = start + 0.001

    in_window = [row for row in resources if start <= row.epoch <= end]
    if not in_window:
        mid = (start + end) / 2.0
        nearest = min(resources, key=lambda row: abs(row.epoch - mid))
        if abs(nearest.epoch - mid) <= 1.0:
            in_window = [nearest]

    cpu_values = [row.cpu_percent for row in in_window if row.cpu_percent is not None]
    mem_values = [row.mem_mib for row in in_window if row.mem_mib is not None]
    disk_delta = counter_delta(resources, start, end, "disk_bytes")
    net_delta = counter_delta(resources, start, end, "net_bytes")
    return {
        "cpu_percent_mean": mean(cpu_values),
        "memory_mib_mean": mean(mem_values),
        "memory_mib_max": max(mem_values) if mem_values else None,
        "network_io_bytes": net_delta,
        "disk_io_bytes": disk_delta,
        "resource_sample_count": len(in_window),
    }


def counter_delta(resources: list[ResourceSample], start: float, end: float, attr: str) -> float | None:
    before = interpolated_counter(resources, start, attr)
    after = interpolated_counter(resources, end, attr)
    if before is None or after is None:
        return None
    return max(0.0, after - before)


def interpolated_counter(resources: list[ResourceSample], t: float, attr: str) -> float | None:
    previous = None
    following = None
    for row in resources:
        value = getattr(row, attr)
        if value is None:
            continue
        if row.epoch <= t:
            previous = row
        if row.epoch >= t:
            following = row
            break
    if previous is None and following is None:
        return None
    if previous is None:
        return getattr(following, attr)
    if following is None:
        return getattr(previous, attr)
    prev_value = getattr(previous, attr)
    next_value = getattr(following, attr)
    if following.epoch == previous.epoch:
        return prev_value
    frac = (t - previous.epoch) / (following.epoch - previous.epoch)
    return prev_value + frac * (next_value - prev_value)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset", "attempt_dir", "call_index", "tool", "operation", "tool_family",
        "duration_ms", "cpu_percent_mean", "memory_mib_mean", "memory_mib_max",
        "network_io_bytes", "disk_io_bytes", "resource_sample_count",
        "has_pipe", "has_recursive_hint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        "duration_ms", "cpu_percent_mean", "memory_mib_mean",
        "memory_mib_max", "network_io_bytes", "disk_io_bytes",
    ]
    columns = ["operation", "n"] + [f"{metric}_{stat}" for metric in metrics for stat in ("p50", "p90", "p99", "mean")]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["operation"]), []).append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for operation, items in sorted(grouped.items()):
            out: dict[str, Any] = {"operation": operation, "n": len(items)}
            for metric in metrics:
                values = [safe_float(item.get(metric)) for item in items]
                values = [value for value in values if value is not None and math.isfinite(value)]
                out[f"{metric}_p50"] = quantile(values, 0.50)
                out[f"{metric}_p90"] = quantile(values, 0.90)
                out[f"{metric}_p99"] = quantile(values, 0.99)
                out[f"{metric}_mean"] = mean(values)
            writer.writerow(out)


def plot_boxplots(path: Path, rows: list[dict[str, Any]], top_n: int) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["operation"])] = counts.get(str(row["operation"]), 0) + 1
    operations = [op for op, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]]
    metrics = [
        ("duration_ms", "Tool duration (ms)", True),
        ("cpu_percent_mean", "CPU percent mean", False),
        ("memory_mib_mean", "Memory footprint mean (MiB)", False),
        ("network_io_bytes", "Network I/O delta (bytes)", True),
        ("disk_io_bytes", "Disk I/O delta (bytes)", True),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(12, len(operations) * 0.75), 18), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title, log_scale) in zip(axes, metrics):
        data = []
        labels = []
        for operation in operations:
            values = [
                safe_float(row.get(metric))
                for row in rows
                if row["operation"] == operation
            ]
            values = [value for value in values if value is not None and math.isfinite(value)]
            if log_scale:
                values = [math.log10(value + 1.0) for value in values]
            if values:
                data.append(values)
                labels.append(f"{operation}\n(n={counts[operation]})")
        ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        ax.set_title(title + (" [log10(x+1)]" if log_scale else ""))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Operation-level tool time and resource distributions", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def dataset_name(root: Path, attempt_dir: Path) -> str:
    try:
        return attempt_dir.relative_to(root).parts[0]
    except ValueError:
        return "unknown"


def parse_epoch(value: Any) -> float | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_mem_mib(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re_match_mem(text)
    if match is None:
        return safe_float(text)
    number, unit = match
    factor = {
        "b": 1 / (1024 * 1024),
        "kib": 1 / 1024,
        "kb": 1 / 1024,
        "mib": 1.0,
        "mb": 1.0,
        "gib": 1024.0,
        "gb": 1024.0,
    }.get(unit.lower(), 1.0)
    return number * factor


def re_match_mem(text: str) -> tuple[float, str] | None:
    import re

    match = re.match(r"^\s*([0-9.]+)\s*([A-Za-z]+)?\s*$", text)
    if not match:
        return None
    return float(match.group(1)), match.group(2) or "MiB"


def safe_percent(value: Any) -> float | None:
    if value is None:
        return None
    return safe_float(str(value).replace("%", ""))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


if __name__ == "__main__":
    main()
