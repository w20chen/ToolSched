"""Linux controlled-replay collector for real core-placement datasets."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..io import write_samples
from ..schema import ToolSample
from .linux_topology import LogicalCpu, discover_linux_topology, select_candidate_cpus, topology_summary
from .placement_manifest import load_and_validate_manifest


@dataclass(frozen=True)
class InterferenceScenario:
    name: str
    busy_cpus: tuple[int, ...]
    worker_kind: str
    pressure_source: str = "controlled_corunner_proxy"


def collect_placement_study(
    manifest_path: Path,
    raw_out: Path,
    samples_out: Path,
    repeats: int = 7,
    max_candidates: int = 4,
    max_corunners: int = 2,
    scenarios_requested: tuple[str, ...] = ("idle", "smt_busy", "cluster_a_busy", "cluster_b_busy"),
    warmup_seconds: float = 0.20,
    utilization_window_seconds: float = 0.15,
    peer_baseline_seconds: float = 1.0,
    perf_mode: str = "auto",
    seed: int = 7,
    min_success_per_candidate: int = 3,
) -> dict[str, Any]:
    if platform.system() != "Linux" or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("real placement collection requires Linux sched_setaffinity")
    if repeats < 1 or max_candidates < 2:
        raise ValueError("repeats must be positive and at least two candidates are required")
    if perf_mode not in {"auto", "required", "off"}:
        raise ValueError("perf_mode must be auto, required, or off")
    if perf_mode == "required" and shutil.which("perf") is None:
        raise RuntimeError("--perf-mode required but Linux perf is not installed")
    manifest = load_and_validate_manifest(manifest_path, require_approved=True)
    topology = discover_linux_topology()
    candidates = select_candidate_cpus(topology, max_candidates=max_candidates)
    if len(candidates) < 2:
        raise RuntimeError("fewer than two allowed physical-core candidates")
    scenarios = build_interference_scenarios(topology, candidates, max_corunners)
    scenarios = [scenario for scenario in scenarios if scenario.name in set(scenarios_requested)]
    missing = sorted(set(scenarios_requested) - {scenario.name for scenario in scenarios})
    if not scenarios:
        raise RuntimeError("none of the requested interference scenarios are supported by this topology")

    machine = _machine_metadata(topology, candidates, missing)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    hardware_perf_rows = 0
    rng = random.Random(seed)
    with raw_out.open("w", encoding="utf-8") as handle:
        for entry in manifest["entries"]:
            entry_scenarios = list(scenarios)
            rng.shuffle(entry_scenarios)
            for scenario in entry_scenarios:
                for repeat in range(repeats):
                    action_order = list(candidates)
                    rng.shuffle(action_order)
                    for order_index, action in enumerate(action_order):
                        row = _collect_one(
                            entry=entry,
                            action=action,
                            candidates=candidates,
                            topology=topology,
                            scenario=scenario,
                            repeat=repeat,
                            order_index=order_index,
                            warmup_seconds=warmup_seconds,
                            utilization_window_seconds=utilization_window_seconds,
                            peer_baseline_seconds=peer_baseline_seconds,
                            perf_mode=perf_mode,
                            machine=machine,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        completed += 1
                        failed += int(not row["success"])
                        hardware_perf_rows += int(any(
                            state.get("pressure_measurement") == "hardware_perf"
                            for state in row.get("candidate_states", [])
                        ))

    samples, aggregation = aggregate_placement_rows(raw_out, min_success_per_candidate)
    write_samples(samples_out, samples)
    gates = {
        "at_least_20_independent_invocations": len(manifest["entries"]) >= 20,
        "at_least_5_repeats": repeats >= 5,
        "at_least_3_candidate_cores": len(candidates) >= 3,
        "at_least_2_supported_scenarios": len(scenarios) >= 2,
        "failure_rate_at_most_10_percent": (failed / completed if completed else 1.0) <= 0.10,
        "hardware_perf_pressure_available": hardware_perf_rows > 0,
        "aggregated_counterfactual_samples_available": len(samples) > 0,
    }
    return {
        "manifest": str(manifest_path),
        "raw_out": str(raw_out),
        "samples_out": str(samples_out),
        "machine": machine,
        "replay_entries": len(manifest["entries"]),
        "candidates": [row.cpu_id for row in candidates],
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "unsupported_scenarios": missing,
        "replay_rows": completed,
        "failed_rows": failed,
        "hardware_perf_rows": hardware_perf_rows,
        "aggregated_samples": len(samples),
        "aggregation": aggregation,
        "minimum_design_gates": gates,
        "minimum_design_gates_passed": all(gates.values()),
        "claim_boundary": (
            "passing these engineering gates does not replace multi-machine replication, "
            "case-block bootstrap, or external validation"
        ),
    }


def build_interference_scenarios(
    topology: list[LogicalCpu],
    candidates: list[LogicalCpu],
    max_corunners: int = 2,
) -> list[InterferenceScenario]:
    """Construct fixed load placements shared by all actions in a replay block."""

    scenarios = [InterferenceScenario("idle", (), "cpu", "measured_idle_state")]
    candidate_ids = {row.cpu_id for row in candidates}
    by_cpu = {row.cpu_id: row for row in topology}

    # Load an SMT sibling while keeping its paired candidate available.
    for candidate in candidates:
        siblings = [cpu for cpu in candidate.thread_siblings if cpu != candidate.cpu_id and cpu not in candidate_ids]
        if siblings:
            scenarios.append(InterferenceScenario("smt_busy", (siblings[0],), "cpu"))
            break

    # Use non-candidate physical cores so the candidate set remains runnable.
    candidate_physical = {(row.package_id, row.core_id) for row in candidates}
    spare_by_cluster: dict[str, list[int]] = {}
    for row in topology:
        if row.cpu_id in candidate_ids or (row.package_id, row.core_id) in candidate_physical:
            continue
        spare_by_cluster.setdefault(row.cluster_id, []).append(row.cpu_id)
    if spare_by_cluster:
        anchor_cluster = max(spare_by_cluster, key=lambda key: len(spare_by_cluster[key]))
        busy = tuple(sorted(set(spare_by_cluster[anchor_cluster]))[:max_corunners])
        if busy:
            scenarios.append(InterferenceScenario("cluster_a_busy", busy, "cache"))

        other = [key for key in spare_by_cluster if key != anchor_cluster and spare_by_cluster[key]]
        if other:
            other_cluster = max(other, key=lambda key: len(spare_by_cluster[key]))
            busy_other = tuple(sorted(set(spare_by_cluster[other_cluster]))[:max_corunners])
            scenarios.append(InterferenceScenario("cluster_b_busy", busy_other, "cache"))
    return scenarios


def aggregate_placement_rows(
    raw_path: Path,
    min_success_per_candidate: int = 3,
) -> tuple[list[ToolSample], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["invocation_id"]), str(row["scenario"]))
            groups.setdefault(key, []).append(row)

    samples: list[ToolSample] = []
    excluded: dict[str, int] = {}
    for (invocation_id, scenario), rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("success")]
        if not successful:
            _bump(excluded, "no_successful_replays")
            continue
        entry = successful[0]["entry"]
        candidate_ids = sorted({str(row["chosen_candidate"]) for row in rows})
        states = []
        costs: dict[str, dict[str, float]] = {}
        incomplete = False
        for candidate_id in candidate_ids:
            action_rows = [
                row for row in successful
                if str(row["chosen_candidate"]) == candidate_id
            ]
            if len(action_rows) < min_success_per_candidate:
                incomplete = True
                break
            snapshots = []
            for row in successful:
                match = next(
                    (state for state in row["candidate_states"] if str(state["candidate_id"]) == candidate_id),
                    None,
                )
                if match:
                    snapshots.append(match)
            if not snapshots:
                incomplete = True
                break
            state = dict(snapshots[0])
            for field in (
                "core_util", "smt_sibling_util", "cluster_util", "llc_pressure",
                "memory_bw_pressure", "run_queue", "frequency_ratio",
            ):
                state[field] = statistics.median(float(snapshot[field]) for snapshot in snapshots)
            pressure_sources = {str(snapshot.get("pressure_measurement", "unknown")) for snapshot in snapshots}
            state["pressure_measurement"] = (
                "hardware_perf" if pressure_sources == {"hardware_perf"}
                else "+".join(sorted(pressure_sources))
            )
            states.append(state)
            latency = statistics.median(float(row["tool_latency_ms"]) for row in action_rows)
            peer_values = [
                float(row["peer_slowdown_ms"])
                for row in action_rows
                if row.get("peer_measurement_valid") and row.get("peer_slowdown_ms") is not None
            ]
            costs[candidate_id] = {
                "tool_latency_ms": latency,
                "peer_slowdown_ms": statistics.median(peer_values) if peer_values else 0.0,
            }
        if incomplete or len(costs) < 2:
            _bump(excluded, "insufficient_success_per_candidate")
            continue

        all_latencies = [float(row["tool_latency_ms"]) for row in successful]
        features = {
            "command_len": len(str(entry.get("command", ""))),
            "call_index": 0,
            "history_len": 0,
            "placement_repeats": len(successful),
        }
        samples.append(
            ToolSample(
                sample_id=f"placement/{invocation_id}/{scenario}",
                dataset=str(entry.get("dataset", "placement-replay")),
                case_id=str(entry.get("case_id", invocation_id)),
                attempt_id=f"placement-{scenario}",
                tool=str(entry.get("tool", "shell")),
                operation=str(entry.get("operation", "replay")),
                tool_family=str(entry.get("tool_family", "terminal")),
                timestamp=None,
                duration_ms=statistics.median(all_latencies),
                input={"command": entry["command"], "working_dir": entry["cwd"]},
                features=features,
                labels={
                    "resource_class": str(entry.get("resource_class", "unknown")),
                    "placement_costs": costs,
                    "placement_evidence": "controlled_real_replay",
                },
                resources={
                    "placement_candidates": states,
                    "machine": successful[0]["machine"],
                    "scenario": scenario,
                    "state_provenance": successful[0]["state_provenance"],
                },
            )
        )
    return samples, {"groups": len(groups), "included": len(samples), "excluded": excluded}


def _collect_one(
    entry: dict[str, Any],
    action: LogicalCpu,
    candidates: list[LogicalCpu],
    topology: list[LogicalCpu],
    scenario: InterferenceScenario,
    repeat: int,
    order_index: int,
    warmup_seconds: float,
    utilization_window_seconds: float,
    peer_baseline_seconds: float,
    perf_mode: str,
    machine: dict[str, Any],
) -> dict[str, Any]:
    reset = entry.get("reset_command")
    if reset:
        reset_result = _run_command(
            str(reset), Path(str(entry["cwd"])), dict(entry.get("environment") or {}),
            cpu=None, timeout=float(entry.get("reset_timeout_seconds", 300.0)),
        )
        if reset_result["return_code"] != 0:
            return _failed_row(entry, action, scenario, repeat, order_index, machine, "reset_failed")

    solo_peer: list[dict[str, Any]] = []
    if scenario.busy_cpus:
        baseline_workers = _start_workers(scenario, max_seconds=peer_baseline_seconds + 2.0)
        time.sleep(max(0.10, peer_baseline_seconds))
        solo_peer = _stop_workers(baseline_workers)

    peers = _start_workers(
        scenario,
        max_seconds=float(entry["timeout_seconds"]) + warmup_seconds + 5.0,
    )
    try:
        if peers:
            time.sleep(max(0.0, warmup_seconds))
        candidate_states = _capture_candidate_states(
            candidates, topology, scenario, utilization_window_seconds, perf_mode
        )
        start = time.perf_counter()
        result = _run_command(
            str(entry["command"]),
            Path(str(entry["cwd"])),
            dict(entry.get("environment") or {}),
            cpu=action.cpu_id,
            timeout=float(entry["timeout_seconds"]),
        )
        tool_elapsed = time.perf_counter() - start
    except BaseException:
        _stop_workers(peers)
        raise
    concurrent_peer = _stop_workers(peers)
    peer_slowdown = None
    peer_valid = False
    if scenario.busy_cpus and tool_elapsed >= 0.10 and concurrent_peer:
        peer_slowdown = _peer_slowdown_ms(concurrent_peer, solo_peer, tool_elapsed)
        peer_valid = peer_slowdown is not None

    expected = {int(code) for code in entry.get("expected_exit_codes", [0])}
    success = not result["timed_out"] and int(result["return_code"]) in expected
    return {
        "schema_version": 1,
        "invocation_id": entry["invocation_id"],
        "scenario": scenario.name,
        "repeat": repeat,
        "order_index": order_index,
        "chosen_candidate": f"cpu-{action.cpu_id}",
        "candidate_states": candidate_states,
        "tool_latency_ms": result["elapsed_seconds"] * 1000.0,
        "peer_slowdown_ms": peer_slowdown,
        "peer_measurement_valid": peer_valid,
        "concurrent_peer": concurrent_peer,
        "solo_peer": solo_peer,
        "return_code": result["return_code"],
        "timed_out": result["timed_out"],
        "stdout_tail": result["stdout_tail"],
        "stderr_tail": result["stderr_tail"],
        "success": success,
        "failure_reason": None if success else "tool_failed_or_timed_out",
        "entry": entry,
        "machine": machine,
        "state_provenance": {
            "cpu_util": "/proc/stat pre-launch delta while co-runners are active",
            "frequency": "Linux cpufreq sysfs",
            "llc_pressure": "perf cache-miss proxy when available; otherwise controlled co-runner proxy",
            "memory_bw_pressure": "perf stalled-cycle proxy when available; otherwise controlled co-runner proxy",
            "run_queue": "/proc/loadavg runnable count normalized by allowed CPUs",
        },
    }


def _capture_candidate_states(
    candidates: list[LogicalCpu],
    topology: list[LogicalCpu],
    scenario: InterferenceScenario,
    interval: float,
    perf_mode: str,
) -> list[dict[str, Any]]:
    by_cpu = {row.cpu_id: row for row in topology}
    cluster_cpus: dict[str, list[int]] = {}
    for row in topology:
        cluster_cpus.setdefault(row.cluster_id, []).append(row.cpu_id)
    utilization, perf_pressure = _sample_cpu_and_cluster_perf(
        cluster_cpus, max(0.02, interval), perf_mode
    )
    if perf_mode == "required" and set(perf_pressure) != set(cluster_cpus):
        missing = sorted(set(cluster_cpus) - set(perf_pressure))
        raise RuntimeError(f"perf counters unavailable for LLC clusters: {missing}")
    run_queue = _normalized_run_queue(len(topology))
    busy_clusters = {by_cpu[cpu].cluster_id for cpu in scenario.busy_cpus if cpu in by_cpu}
    states = []
    for candidate in candidates:
        sibling_values = [utilization.get(cpu, 0.0) for cpu in candidate.thread_siblings if cpu != candidate.cpu_id]
        cluster_values = [utilization.get(cpu, 0.0) for cpu in cluster_cpus[candidate.cluster_id]]
        cluster_util = statistics.mean(cluster_values) if cluster_values else utilization.get(candidate.cpu_id, 0.0)
        controlled_pressure = cluster_util if candidate.cluster_id in busy_clusters else 0.0
        measured = perf_pressure.get(candidate.cluster_id)
        if measured:
            llc_pressure = measured["llc_pressure"]
            memory_pressure = measured["memory_pressure"]
            pressure_quality = "hardware_perf"
        else:
            llc_pressure = controlled_pressure if scenario.worker_kind == "cache" else 0.20 * cluster_util
            memory_pressure = 0.70 * controlled_pressure if scenario.worker_kind == "cache" else 0.10 * cluster_util
            pressure_quality = scenario.pressure_source
        states.append(
            {
                "candidate_id": f"cpu-{candidate.cpu_id}",
                "core_id": str(candidate.core_id),
                "logical_cpu_id": candidate.cpu_id,
                "cluster_id": candidate.cluster_id,
                "numa_node": str(candidate.numa_node),
                "core_util": _unit(utilization.get(candidate.cpu_id, 0.0)),
                "smt_sibling_util": _unit(statistics.mean(sibling_values) if sibling_values else 0.0),
                "cluster_util": _unit(cluster_util),
                "llc_pressure": _unit(llc_pressure),
                "memory_bw_pressure": _unit(memory_pressure),
                "run_queue": run_queue,
                "frequency_ratio": _frequency_ratio(candidate.cpu_id),
                "pressure_measurement": pressure_quality,
            }
        )
    return states


def _run_command(
    command: str,
    cwd: Path,
    environment: dict[str, Any],
    cpu: int | None,
    timeout: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in environment.items()})

    def set_affinity() -> None:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})

    start = time.perf_counter()
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        preexec_fn=set_affinity,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return {
        "return_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.perf_counter() - start,
        "stdout_tail": stdout[-4096:],
        "stderr_tail": stderr[-4096:],
    }


def _start_workers(
    scenario: InterferenceScenario,
    max_seconds: float = 3600.0,
) -> list[subprocess.Popen]:
    workers = []
    try:
        for cpu in scenario.busy_cpus:
            def set_affinity(target: int = cpu) -> None:
                os.sched_setaffinity(0, {target})

            workers.append(
                subprocess.Popen(
                    [
                        sys.executable, "-m", "toolsched.scheduler.placement_worker",
                        "--kind", scenario.worker_kind, "--max-seconds", str(max_seconds),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                    preexec_fn=set_affinity,
                )
            )
    except BaseException:
        _stop_workers(workers)
        raise
    return workers


def _stop_workers(workers: list[subprocess.Popen]) -> list[dict[str, Any]]:
    rows = []
    for process in workers:
        if process.poll() is None:
            process.terminate()
    for process in workers:
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        parsed = None
        for line in reversed(stdout.splitlines()):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            rows.append({"error": stderr[-1000:] or "worker produced no metrics"})
    return rows


def _peer_slowdown_ms(
    concurrent: list[dict[str, Any]],
    solo: list[dict[str, Any]],
    window_seconds: float,
) -> float | None:
    if len(concurrent) != len(solo) or not concurrent:
        return None
    slowdown = 0.0
    for with_tool, without_tool in zip(concurrent, solo):
        solo_rate = float(without_tool.get("operations_per_second", 0.0))
        concurrent_elapsed = float(with_tool.get("elapsed_seconds", 0.0))
        concurrent_operations = float(with_tool.get("operations", 0.0))
        # Co-runners start before the tool so state can be measured pre-launch.
        # Remove expected solo work from that warm-up interval before comparing
        # throughput over the actual overlap window.
        pre_tool_seconds = max(0.0, concurrent_elapsed - window_seconds)
        overlap_operations = concurrent_operations - solo_rate * pre_tool_seconds
        concurrent_rate = max(0.0, overlap_operations) / max(1e-9, window_seconds)
        if concurrent_rate <= 0 or solo_rate <= 0:
            return None
        slowdown += 1000.0 * window_seconds * max(0.0, solo_rate / concurrent_rate - 1.0)
    return slowdown


def _sample_cpu_and_cluster_perf(
    cluster_cpus: dict[str, list[int]],
    interval: float,
    perf_mode: str,
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    before = _read_proc_stat()
    perf_processes: dict[str, subprocess.Popen] = {}
    if perf_mode != "off" and shutil.which("perf"):
        events = "cycles,instructions,cache-references,cache-misses,stalled-cycles-backend"
        for cluster, cpus in cluster_cpus.items():
            perf_processes[cluster] = subprocess.Popen(
                [
                    "perf", "stat", "-a", "-x,", "-C", ",".join(str(cpu) for cpu in sorted(cpus)),
                    "-e", events, "--", "sleep", str(interval),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
    time.sleep(interval)
    after = _read_proc_stat()
    result = {}
    for cpu, (total_after, idle_after) in after.items():
        total_before, idle_before = before.get(cpu, (total_after, idle_after))
        total_delta = total_after - total_before
        idle_delta = idle_after - idle_before
        result[cpu] = _unit(1.0 - idle_delta / total_delta) if total_delta > 0 else 0.0
    perf_pressure: dict[str, dict[str, float]] = {}
    for cluster, process in perf_processes.items():
        try:
            _, stderr = process.communicate(timeout=max(2.0, interval * 4))
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
        counters = _parse_perf_stat(stderr)
        references = counters.get("cache-references", 0.0)
        misses = counters.get("cache-misses", 0.0)
        cycles = counters.get("cycles", 0.0)
        stalled = counters.get("stalled-cycles-backend", 0.0)
        if process.returncode == 0 and references > 0 and cycles > 0:
            perf_pressure[cluster] = {
                "llc_pressure": _unit(misses / references),
                "memory_pressure": _unit(stalled / cycles),
            }
    return result, perf_pressure


def _parse_perf_stat(stderr: str) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in stderr.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        raw = fields[0].replace(" ", "")
        event = fields[2]
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            counters[event] = value
    return counters


def _read_proc_stat() -> dict[int, tuple[int, int]]:
    rows = {}
    with Path("/proc/stat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or not parts[0].startswith("cpu") or not parts[0][3:].isdigit():
                continue
            values = [int(value) for value in parts[1:]]
            total = sum(values)
            idle = (values[3] if len(values) > 3 else 0) + (values[4] if len(values) > 4 else 0)
            rows[int(parts[0][3:])] = (total, idle)
    return rows


def _normalized_run_queue(cpu_count: int) -> float:
    try:
        fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()
        runnable = int(fields[3].split("/", 1)[0])
        return max(0.0, runnable / max(1, cpu_count))
    except (OSError, ValueError, IndexError):
        return 0.0


def _frequency_ratio(cpu: int) -> float:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    current = _read_float(root / "scaling_cur_freq")
    reference = _read_float(root / "cpuinfo_max_freq") or _read_float(root / "scaling_max_freq")
    if current and reference and reference > 0:
        return min(2.0, max(0.1, current / reference))
    return 1.0


def _machine_metadata(
    topology: list[LogicalCpu],
    candidates: list[LogicalCpu],
    unsupported_scenarios: list[str],
) -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "topology": topology_summary(topology),
        "candidate_topology": [row.to_json() for row in candidates],
        "unsupported_scenarios": unsupported_scenarios,
    }


def _failed_row(
    entry: dict[str, Any], action: LogicalCpu, scenario: InterferenceScenario,
    repeat: int, order_index: int, machine: dict[str, Any], reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "invocation_id": entry["invocation_id"],
        "scenario": scenario.name,
        "repeat": repeat,
        "order_index": order_index,
        "chosen_candidate": f"cpu-{action.cpu_id}",
        "candidate_states": [],
        "tool_latency_ms": None,
        "peer_slowdown_ms": None,
        "peer_measurement_valid": False,
        "return_code": None,
        "timed_out": False,
        "success": False,
        "failure_reason": reason,
        "entry": entry,
        "machine": machine,
        "state_provenance": {},
    }


def _read_float(path: Path) -> float | None:
    try:
        value = float(path.read_text(encoding="utf-8").strip())
        return value if math.isfinite(value) else None
    except (OSError, ValueError):
        return None


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
