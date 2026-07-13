"""Build and validate explicit replay manifests for placement experiments."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from ..schema import ToolSample


SAFE_PROGRAMS = {"find", "grep", "rg", "head", "tail", "wc", "sort", "uniq", "cut", "ls", "cat", "pwd"}
SAFE_GIT_SUBCOMMANDS = {"status", "diff", "log", "show", "grep", "rev-parse", "ls-files"}
UNSAFE_SHELL_TOKENS = (";", "&&", "||", ">", "`", "$(", "${")


def build_replay_manifest(
    samples: list[ToolSample],
    path_maps: list[tuple[str, str]] | None = None,
    max_entries: int = 50,
    min_observed_duration_ms: float = 50.0,
    approve_safe_read_only: bool = False,
) -> dict[str, Any]:
    """Select strictly read-only, locally reproducible commands from traces."""

    path_maps = path_maps or []
    entries = []
    skipped: dict[str, int] = {}
    seen_commands: set[tuple[str, str]] = set()
    # Long calls are most likely to expose placement effects.  Dataset/case
    # diversity is preserved by allowing at most one identical command/cwd.
    ordered = sorted(samples, key=lambda s: float(s.duration_ms or 0.0), reverse=True)
    for sample in ordered:
        if len(entries) >= max_entries:
            break
        if (sample.duration_ms or 0.0) < min_observed_duration_ms:
            _bump(skipped, "below_min_duration")
            continue
        command, cwd = _sample_replay_command(sample)
        if not command:
            _bump(skipped, "not_replayable")
            continue
        command = apply_path_maps(command, path_maps)
        cwd = apply_path_maps(cwd, path_maps)
        safe, reason = validate_read_only_command(command)
        if not safe:
            _bump(skipped, f"unsafe:{reason}")
            continue
        key = (command, cwd)
        if key in seen_commands:
            _bump(skipped, "duplicate")
            continue
        if not Path(cwd).is_dir():
            _bump(skipped, "cwd_missing")
            continue
        seen_commands.add(key)
        entries.append(
            {
                "invocation_id": sample.sample_id,
                "dataset": sample.dataset,
                "case_id": sample.case_id,
                "tool": sample.tool,
                "operation": sample.operation,
                "tool_family": sample.tool_family,
                "resource_class": sample.labels.get("resource_class", "unknown"),
                "command": command,
                "cwd": cwd,
                "environment": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "timeout_seconds": max(30.0, min(900.0, 4.0 * float(sample.duration_ms or 1000.0) / 1000.0)),
                "expected_exit_codes": [0],
                "read_only": True,
                "approved": bool(approve_safe_read_only),
                "observed_duration_ms": sample.duration_ms,
            }
        )
    return {
        "schema_version": 1,
        "kind": "toolsched-placement-replay-manifest",
        "entries": entries,
        "selection": {
            "max_entries": max_entries,
            "min_observed_duration_ms": min_observed_duration_ms,
            "strict_read_only": True,
            "auto_approved": bool(approve_safe_read_only),
            "path_maps": [{"from": old, "to": new} for old, new in path_maps],
            "skipped": skipped,
        },
    }


def build_toolsched_study_manifest(
    samples_path: Path,
    cwd: Path,
    shard_count: int = 4,
    workload_names: set[str] | None = None,
) -> dict[str, Any]:
    """Create approved real repository workloads across independent case shards."""

    if not samples_path.is_file():
        raise ValueError(f"samples file does not exist: {samples_path}")
    if not cwd.is_dir():
        raise ValueError(f"study cwd does not exist: {cwd}")
    if shard_count < 1 or shard_count > 32:
        raise ValueError("shard_count must be between 1 and 32")
    workloads = (
        ("bucket_logistic", "cpu_memory_mixed", 600),
        ("bucket_forest", "cpu_memory_mixed", 900),
        ("next_tool", "cpu_memory_mixed", 900),
        ("remaining_forest", "cpu_memory_mixed", 1200),
        ("quantile", "cpu_memory_mixed", 300),
    )
    available = {name for name, _, _ in workloads}
    selected_names = workload_names or available
    unknown = selected_names - available
    if unknown:
        raise ValueError(f"unknown study workloads: {sorted(unknown)}")
    workloads = tuple(row for row in workloads if row[0] in selected_names)
    if not workloads:
        raise ValueError("at least one study workload is required")
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LOKY_MAX_CPU_COUNT": "1",
    }
    entries = []
    absolute_samples = str(samples_path.resolve())
    for workload, resource_class, timeout in workloads:
        for shard_index in range(shard_count):
            invocation_id = f"toolsched/{workload}/shard-{shard_index}-of-{shard_count}"
            entries.append(
                {
                    "invocation_id": invocation_id,
                    "dataset": "toolsched-real-replay",
                    "case_id": invocation_id,
                    "tool": "python",
                    "operation": "python",
                    "tool_family": "terminal",
                    "resource_class": resource_class,
                    "command": (
                        "python -m toolsched.scheduler.study_workloads "
                        f"--samples {shlex.quote(absolute_samples)} --workload {workload} "
                        f"--shard-index {shard_index} --shard-count {shard_count}"
                    ),
                    "cwd": str(cwd.resolve()),
                    "environment": dict(environment),
                    "timeout_seconds": timeout,
                    "expected_exit_codes": [0],
                    "read_only": True,
                    "approved": True,
                }
            )
    return {
        "schema_version": 1,
        "kind": "toolsched-placement-replay-manifest",
        "description": "Real ToolSched computations partitioned by case shard; co-runners are added by the collector.",
        "entries": entries,
        "study_design": {
            "case_shards": shard_count,
            "workload_families": [name for name, _, _ in workloads],
            "independent_invocations": len(entries),
        },
    }


def load_and_validate_manifest(path: Path, require_approved: bool = True) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("kind") != "toolsched-placement-replay-manifest":
        raise ValueError("not a ToolSched placement replay manifest")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest has no replay entries")
    ids: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {idx} is not an object")
        invocation_id = str(entry.get("invocation_id", ""))
        if not invocation_id or invocation_id in ids:
            raise ValueError(f"manifest entry {idx} has missing or duplicate invocation_id")
        ids.add(invocation_id)
        command = str(entry.get("command", ""))
        cwd = Path(str(entry.get("cwd", "")))
        if not command or not cwd.is_dir():
            raise ValueError(f"manifest entry {invocation_id} has missing command or cwd")
        if require_approved and entry.get("approved") is not True:
            raise ValueError(f"manifest entry {invocation_id} is not explicitly approved")
        if entry.get("read_only") is not True and not entry.get("reset_command"):
            raise ValueError(f"stateful entry {invocation_id} requires reset_command")
        timeout = float(entry.get("timeout_seconds", 0.0))
        if timeout <= 0 or timeout > 3600:
            raise ValueError(f"manifest entry {invocation_id} has invalid timeout")
    return payload


def validate_read_only_command(command: str) -> tuple[bool, str]:
    """Conservative shell validator used only for automatic approval."""

    if not command.strip() or "\n" in command or "\r" in command:
        return False, "empty_or_multiline"
    if any(token in command for token in UNSAFE_SHELL_TOKENS):
        return False, "shell_control_or_redirection"
    # A single pipe is allowed only when every segment starts with a known
    # read-only program. xargs is intentionally excluded.
    for segment in command.split("|"):
        try:
            argv = shlex.split(segment.strip(), posix=True)
        except ValueError:
            return False, "invalid_shell_syntax"
        if not argv:
            return False, "empty_pipeline_segment"
        program = Path(argv[0]).name
        if program == "git":
            if len(argv) < 2 or argv[1] not in SAFE_GIT_SUBCOMMANDS:
                return False, "unsafe_git_subcommand"
        elif program not in SAFE_PROGRAMS:
            return False, f"program_not_allowlisted:{program}"
        if program == "find" and any(arg in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for arg in argv):
            return False, "unsafe_find_action"
    return True, "safe"


def parse_path_maps(values: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"path map must be OLD=NEW: {value}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"path map must be OLD=NEW: {value}")
        out.append((old, new))
    return out


def apply_path_maps(value: str, mappings: list[tuple[str, str]]) -> str:
    for old, new in mappings:
        value = value.replace(old, new)
    return value


def _sample_replay_command(sample: ToolSample) -> tuple[str, str]:
    payload = sample.input
    cwd = str(payload.get("working_dir") or payload.get("cwd") or ".")
    if isinstance(payload.get("command"), str):
        return str(payload["command"]), cwd
    path = payload.get("path")
    if sample.operation == "read_file" and path:
        quoted = shlex.quote(str(path))
        return f"head -c 1048576 {quoted}", cwd
    if sample.operation == "list_dir" and path:
        quoted = shlex.quote(str(path))
        return f"find {quoted} -maxdepth 1 -mindepth 1", cwd
    return "", cwd


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
