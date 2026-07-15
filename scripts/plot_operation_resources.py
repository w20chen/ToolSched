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


MEMORY_BASELINE_MAX_AGE_S = 2.0
CPU_BASELINE_MAX_AGE_S = 2.0
COUNTER_INTERPOLATION_MAX_GAP_S = 2.0
# Minimum safe multiplier: thresholds should be at least this many times the
# observed sampling interval to avoid missing valid baseline/counter samples.
_MIN_SAMPLING_MULTIPLIER = 3.0
DEFAULT_MIN_OPERATION_P50_DURATION_MS = 500.0
RESOURCE_PLOT_METRICS = (
    "cpu_percent_delta_mean",
    "memory_bytes_delta_max",
    "network_io_bytes_per_s",
    "disk_io_bytes_per_s",
)


@dataclass
class ResourceSample:
    epoch: float
    cpu_percent: float | None
    mem_bytes: float | None
    disk_bytes: float | None
    net_bytes: float | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument(
        "--top-operations",
        type=int,
        help=(
            "Optionally limit CPU/memory/network/disk panels to the most common "
            "eligible operations. The duration panel always shows all operations."
        ),
    )
    parser.add_argument(
        "--min-operation-p50-duration-ms",
        type=float,
        default=DEFAULT_MIN_OPERATION_P50_DURATION_MS,
        help=(
            "Exclude operations with duration P50 below this threshold from "
            "CPU/memory/network/disk boxplots. Duration is always plotted."
        ),
    )
    args = parser.parse_args()

    root = Path(args.datasets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset_intervals = collect_rows(root, args.max_attempts)
    write_rows(out_dir / "operation_resource_rows.csv", rows)
    summary_rows = summarize_rows(rows, args.min_operation_p50_duration_ms)
    write_summary(out_dir / "operation_resource_summary.csv", summary_rows)
    plot_boxplots(
        out_dir / "operation_resource_boxplots.png",
        rows,
        summary_rows,
        args.top_operations,
        args.min_operation_p50_duration_ms,
    )
    print(json.dumps({
        "rows": len(rows),
        "operations": len({row["operation"] for row in rows}),
        "boxplot_min_operation_p50_duration_ms": args.min_operation_p50_duration_ms,
        "dataset_sampling_intervals_s": {
            ds: round(iv, 3) for ds, iv in sorted(dataset_intervals.items()) if iv > 0
        },
        "threshold_multiplier": _MIN_SAMPLING_MULTIPLIER,
        "out_dir": str(out_dir),
        "boxplot": str(out_dir / "operation_resource_boxplots.png"),
        "rows_csv": str(out_dir / "operation_resource_rows.csv"),
        "summary_csv": str(out_dir / "operation_resource_summary.csv"),
    }, indent=2))


def collect_rows(root: Path, max_attempts: int | None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    attempts_seen = 0
    dataset_intervals: dict[str, float] = {}  # dataset -> median sampling interval
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

        # Cache only valid intervals so an empty/bad first attempt does not
        # permanently force default thresholds for the whole dataset.
        interval = dataset_intervals.get(dataset, 0.0)
        if interval <= 0:
            observed_interval = sampling_interval(resources)
            if observed_interval is not None and observed_interval > 0:
                dataset_intervals[dataset] = observed_interval
                interval = observed_interval
        eff_max_age = max(CPU_BASELINE_MAX_AGE_S, interval * _MIN_SAMPLING_MULTIPLIER)
        eff_max_gap = max(COUNTER_INTERPOLATION_MAX_GAP_S, interval * _MIN_SAMPLING_MULTIPLIER)

        attempts_seen += 1
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            row = row_for_call(dataset, attempt_dir, idx, call, resources, eff_max_age, eff_max_gap)
            if row is not None:
                rows.append(row)
        if max_attempts is not None and attempts_seen >= max_attempts:
            break
    return rows, dataset_intervals


def row_for_call(
    dataset: str,
    attempt_dir: Path,
    idx: int,
    call: dict[str, Any],
    resources: list[ResourceSample],
    baseline_max_age_s: float = CPU_BASELINE_MAX_AGE_S,
    counter_max_gap_s: float = COUNTER_INTERPOLATION_MAX_GAP_S,
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

    stats = resource_stats(resources, start, end,
                           baseline_max_age_s=baseline_max_age_s,
                           counter_max_gap_s=counter_max_gap_s)
    network_io_bytes_per_s = bytes_per_second(stats["network_io_bytes"], duration_ms)
    disk_io_bytes_per_s = bytes_per_second(stats["disk_io_bytes"], duration_ms)
    return {
        "dataset": dataset,
        "attempt_dir": str(attempt_dir),
        "call_index": idx,
        "tool": tool,
        "operation": operation,
        "tool_family": family,
        "duration_ms": duration_ms,
        "cpu_percent_mean": stats["cpu_percent_mean"],
        "cpu_percent_baseline": stats["cpu_percent_baseline"],
        "cpu_percent_delta_mean": stats["cpu_percent_delta_mean"],
        "ambient_memory_bytes_mean": stats["ambient_memory_bytes_mean"],
        "ambient_memory_bytes_max": stats["ambient_memory_bytes_max"],
        "memory_bytes_baseline": stats["memory_bytes_baseline"],
        "memory_bytes_delta_mean": stats["memory_bytes_delta_mean"],
        "memory_bytes_delta_max": stats["memory_bytes_delta_max"],
        "network_io_bytes": stats["network_io_bytes"],
        "disk_io_bytes": stats["disk_io_bytes"],
        "network_io_bytes_per_s": network_io_bytes_per_s,
        "disk_io_bytes_per_s": disk_io_bytes_per_s,
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
        disk_bytes = sum_optional_counters(
            sample.get("disk_read_bytes"),
            sample.get("disk_write_bytes"),
        )
        net_bytes = sum_optional_counters(
            sample.get("net_rx_bytes"),
            sample.get("net_tx_bytes"),
        )
        out.append(ResourceSample(
            epoch=epoch,
            cpu_percent=safe_percent(sample.get("cpu_percent")),
            mem_bytes=parse_memory_bytes(sample.get("mem_usage")),
            disk_bytes=disk_bytes,
            net_bytes=net_bytes,
        ))
    out.sort(key=lambda row: row.epoch)
    return out


def resource_stats(resources: list[ResourceSample], start: float | None, end: float | None,
                  baseline_max_age_s: float = CPU_BASELINE_MAX_AGE_S,
                  counter_max_gap_s: float = COUNTER_INTERPOLATION_MAX_GAP_S) -> dict[str, Any]:
    empty = {
        "cpu_percent_mean": None,
        "cpu_percent_baseline": None,
        "cpu_percent_delta_mean": None,
        "ambient_memory_bytes_mean": None,
        "ambient_memory_bytes_max": None,
        "memory_bytes_baseline": None,
        "memory_bytes_delta_mean": None,
        "memory_bytes_delta_max": None,
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
    cpu_values = [row.cpu_percent for row in in_window if row.cpu_percent is not None]
    mem_values = [row.mem_bytes for row in in_window if row.mem_bytes is not None]
    cpu_mean = mean(cpu_values)
    cpu_baseline = value_at_or_before(resources, start, "cpu_percent", baseline_max_age_s)
    mem_mean = mean(mem_values)
    mem_max = max(mem_values) if mem_values else None
    mem_baseline = value_at_or_before(resources, start, "mem_bytes", max(baseline_max_age_s, MEMORY_BASELINE_MAX_AGE_S))
    disk_delta = counter_delta(resources, start, end, "disk_bytes", counter_max_gap_s)
    net_delta = counter_delta(resources, start, end, "net_bytes", counter_max_gap_s)
    return {
        "cpu_percent_mean": cpu_mean,
        "cpu_percent_baseline": cpu_baseline,
        "cpu_percent_delta_mean": non_negative_delta(cpu_mean, cpu_baseline),
        "ambient_memory_bytes_mean": mem_mean,
        "ambient_memory_bytes_max": mem_max,
        "memory_bytes_baseline": mem_baseline,
        "memory_bytes_delta_mean": non_negative_delta(mem_mean, mem_baseline),
        "memory_bytes_delta_max": non_negative_delta(mem_max, mem_baseline),
        "network_io_bytes": net_delta,
        "disk_io_bytes": disk_delta,
        "resource_sample_count": len(in_window),
    }


def value_at_or_before(resources: list[ResourceSample], t: float, attr: str, max_age_s: float) -> float | None:
    previous = None
    for row in resources:
        value = getattr(row, attr)
        if value is None:
            continue
        if row.epoch <= t:
            previous = row
        else:
            break
    if previous is None or t - previous.epoch > max_age_s:
        return None
    return getattr(previous, attr)


def non_negative_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return max(0.0, value - baseline)


def bytes_per_second(delta_bytes: float | None, duration_ms: float | None) -> float | None:
    if delta_bytes is None or duration_ms is None or duration_ms <= 0:
        return None
    return delta_bytes / (duration_ms / 1000.0)


def counter_delta(resources: list[ResourceSample], start: float, end: float, attr: str,
                  max_gap_s: float = COUNTER_INTERPOLATION_MAX_GAP_S) -> float | None:
    before = interpolated_counter(resources, start, attr, max_gap_s)
    after = interpolated_counter(resources, end, attr, max_gap_s)
    if before is None or after is None or after < before:
        return None
    return after - before


def interpolated_counter(resources: list[ResourceSample], t: float, attr: str,
                         max_gap_s: float = COUNTER_INTERPOLATION_MAX_GAP_S) -> float | None:
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
    if previous is None or following is None:
        return None
    prev_value = getattr(previous, attr)
    next_value = getattr(following, attr)
    if following.epoch == previous.epoch:
        return prev_value
    if following.epoch - previous.epoch > max_gap_s:
        return None
    if next_value < prev_value:
        return None
    frac = (t - previous.epoch) / (following.epoch - previous.epoch)
    return prev_value + frac * (next_value - prev_value)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset", "attempt_dir", "call_index", "tool", "operation", "tool_family",
        "duration_ms", "cpu_percent_mean", "cpu_percent_baseline", "cpu_percent_delta_mean",
        "ambient_memory_bytes_mean", "ambient_memory_bytes_max", "memory_bytes_baseline",
        "memory_bytes_delta_mean", "memory_bytes_delta_max",
        "network_io_bytes", "disk_io_bytes",
        "network_io_bytes_per_s", "disk_io_bytes_per_s",
        "resource_sample_count",
        "has_pipe", "has_recursive_hint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, Any]], min_operation_p50_duration_ms: float) -> list[dict[str, Any]]:
    metrics = [
        "duration_ms",
        "cpu_percent_mean",
        "cpu_percent_delta_mean",
        "ambient_memory_bytes_mean",
        "ambient_memory_bytes_max",
        "memory_bytes_delta_mean",
        "memory_bytes_delta_max",
        "network_io_bytes",
        "disk_io_bytes",
        "network_io_bytes_per_s",
        "disk_io_bytes_per_s",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["operation"]), []).append(row)
    summary_rows = []
    for operation, items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "operation": operation,
            "n": len(items),
            "resource_complete_n": sum(has_complete_resource_metrics(item) for item in items),
        }
        for metric in metrics:
            values = finite_values(item.get(metric) for item in items)
            out[f"{metric}_n"] = len(values)
            out[f"{metric}_p50"] = quantile(values, 0.50)
            out[f"{metric}_p90"] = quantile(values, 0.90)
            out[f"{metric}_p99"] = quantile(values, 0.99)
            out[f"{metric}_mean"] = mean(values)
        duration_p50 = safe_float(out.get("duration_ms_p50"))
        out["eligible_for_resource_boxplot"] = bool(
            duration_p50 is not None and duration_p50 >= min_operation_p50_duration_ms
        )
        summary_rows.append(out)
    return summary_rows


def write_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    metrics = [
        "duration_ms",
        "cpu_percent_mean",
        "cpu_percent_delta_mean",
        "ambient_memory_bytes_mean",
        "ambient_memory_bytes_max",
        "memory_bytes_delta_mean",
        "memory_bytes_delta_max",
        "network_io_bytes",
        "disk_io_bytes",
        "network_io_bytes_per_s",
        "disk_io_bytes_per_s",
    ]
    columns = ["operation", "n", "resource_complete_n", "eligible_for_resource_boxplot"]
    columns += [f"{metric}_{stat}" for metric in metrics for stat in ("n", "p50", "p90", "p99", "mean")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for out in summary_rows:
            writer.writerow(out)


def plot_boxplots(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    top_n: int,
    min_operation_p50_duration_ms: float,
) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["operation"])] = counts.get(str(row["operation"]), 0) + 1
    eligible = {
        str(row["operation"])
        for row in summary_rows
        if row.get("eligible_for_resource_boxplot") is True
    }
    duration_operations = [
        op
        for op, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    resource_operations = [
        op
        for op, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if op in eligible
    ]
    if top_n is not None:
        resource_operations = resource_operations[:top_n]
    complete_resource_rows = [row for row in rows if has_complete_resource_metrics(row)]
    metrics = [
        ("duration_ms", "Tool duration (ms)", True, False, duration_operations),
        ("cpu_percent_delta_mean", "CPU mean excess above pre-call baseline (percentage points)", False, False, resource_operations),
        ("memory_bytes_delta_max", "Memory peak excess above pre-call baseline (bytes)", True, False, resource_operations),
        ("network_io_bytes_per_s", "Network IO: RX + TX (byte/s)", True, False, resource_operations),
        ("disk_io_bytes_per_s", "Disk IO: read + write (byte/s)", True, False, resource_operations),
    ]

    plot_width = max(12, max(len(duration_operations), len(resource_operations)) * 0.75)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(plot_width, 18), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for index, (ax, (metric, title, log_scale, show_fliers, operations)) in enumerate(zip(axes, metrics)):
        metric_rows = rows if index == 0 else complete_resource_rows
        data = []
        labels = []
        positions = []
        missing_positions = []
        for position, operation in enumerate(operations, start=1):
            values = finite_values(
                row.get(metric)
                for row in metric_rows
                if row["operation"] == operation
            )
            if log_scale:
                values = [math.log10(value + 1.0) for value in values]
            labels.append(f"{operation}\n(n={counts[operation]}, m={len(values)})")
            if values:
                data.append(values)
                positions.append(position)
            else:
                missing_positions.append(position)
        if not data:
            ax.set_axis_off()
            continue
        ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            showfliers=show_fliers,
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
            patch_artist=True,
        )
        ax.set_xticks(range(1, len(operations) + 1), labels=labels)
        for position in missing_positions:
            ax.text(
                position,
                0.03,
                "NA",
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
            )
        if log_scale:
            set_readable_log_ticks(ax, metric)
        suffix = " (log scale)" if log_scale else ""
        filter_note = "" if index == 0 else f"; operation P50 duration >= {min_operation_p50_duration_ms:g} ms"
        ax.set_title(title + suffix + filter_note)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Environment telemetry attributed to exclusive tool windows.\n"
        "Duration: all operations. Resources: operation P50 duration "
        f">= {min_operation_p50_duration_ms:g} ms. "
        "n=all calls; m=calls with all four resource metrics.",
        fontsize=12,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def set_readable_log_ticks(ax: Any, metric: str) -> None:
    if metric == "duration_ms":
        ticks = [
            (0.0, "0 ms"),
            (10.0, "10 ms"),
            (100.0, "100 ms"),
            (1_000.0, "1 s"),
            (10_000.0, "10 s"),
            (100_000.0, "100 s"),
            (1_000_000.0, "1000 s"),
        ]
    elif metric.endswith("_per_s"):
        ticks = [
            (0.0, "0 B/s"),
            (1_000.0, "1 KB/s"),
            (1_000_000.0, "1 MB/s"),
            (1_000_000_000.0, "1 GB/s"),
            (1_000_000_000_000.0, "1 TB/s"),
        ]
    else:
        ticks = [
            (0.0, "0 B"),
            (1_000.0, "1 KB"),
            (1_000_000.0, "1 MB"),
            (1_000_000_000.0, "1 GB"),
            (1_000_000_000_000.0, "1 TB"),
        ]

    lower, upper = ax.get_ylim()
    visible = [
        (math.log10(value + 1.0), label)
        for value, label in ticks
        if lower <= math.log10(value + 1.0) <= upper
    ]
    ax.set_yticks(
        [position for position, _ in visible],
        labels=[label for _, label in visible],
    )


def dataset_name(root: Path, attempt_dir: Path) -> str:
    try:
        return attempt_dir.relative_to(root).parts[0]
    except ValueError:
        return "unknown"


def sampling_interval(resources: list[ResourceSample]) -> float | None:
    """Estimate the median sampling interval from a list of ResourceSample."""
    if len(resources) < 2:
        return None
    epochs = [r.epoch for r in resources if r.epoch is not None]
    if len(epochs) < 2:
        return None
    epochs.sort()
    gaps = sorted(epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1) if epochs[i + 1] > epochs[i])
    if not gaps:
        return None
    return gaps[len(gaps) // 2]


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


def parse_memory_bytes(value: Any) -> float | None:
    if value is None:
        return None
    # Docker-style telemetry commonly reports "used / limit". Resource
    # attribution needs the used value, not the full compound string.
    text = str(value).split("/", 1)[0].strip()
    match = re_match_mem(text)
    if match is None:
        number = safe_float(text)
        return None if number is None else number * 1024 * 1024
    number, unit = match
    factor = {
        "b": 1.0,
        "kb": 1000.0,
        "kib": 1024.0,
        "mb": 1000.0 ** 2,
        "mib": 1024.0 ** 2,
        "gb": 1000.0 ** 3,
        "gib": 1024.0 ** 3,
    }.get(unit.lower())
    if factor is None:
        return None
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


def finite_values(values: Any) -> list[float]:
    out = [safe_float(value) for value in values]
    return [value for value in out if value is not None and math.isfinite(value)]


def has_complete_resource_metrics(row: dict[str, Any]) -> bool:
    return all(safe_float(row.get(metric)) is not None for metric in RESOURCE_PLOT_METRICS)


def sum_optional_counters(*values: Any) -> float | None:
    parsed = [safe_float(value) for value in values]
    present = [value for value in parsed if value is not None]
    return sum(present) if present else None


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
