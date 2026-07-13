"""Dependency-free Linux CPU topology discovery for placement replay."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogicalCpu:
    cpu_id: int
    core_id: int
    package_id: int
    numa_node: int
    cluster_id: str
    thread_siblings: tuple[int, ...]

    def to_json(self) -> dict:
        return asdict(self) | {"thread_siblings": list(self.thread_siblings)}


def discover_linux_topology(
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
    allowed_cpus: set[int] | None = None,
) -> list[LogicalCpu]:
    if allowed_cpus is None:
        allowed_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
    rows = []
    for cpu_dir in sorted(sysfs_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cpu_id = int(cpu_dir.name[3:])
        if allowed_cpus and cpu_id not in allowed_cpus:
            continue
        topology = cpu_dir / "topology"
        core_id = _read_int(topology / "core_id", cpu_id)
        package_id = _read_int(topology / "physical_package_id", 0)
        siblings = tuple(
            sorted(
                cpu
                for cpu in parse_cpu_list(_read_text(topology / "thread_siblings_list", str(cpu_id)))
                if not allowed_cpus or cpu in allowed_cpus
            )
        )
        numa_node = _numa_node(cpu_dir)
        cluster_id = _last_level_cache_id(cpu_dir, package_id, core_id)
        rows.append(LogicalCpu(cpu_id, core_id, package_id, numa_node, cluster_id, siblings or (cpu_id,)))
    if not rows:
        raise RuntimeError(f"no allowed online CPUs found under {sysfs_root}")
    return rows


def select_candidate_cpus(topology: list[LogicalCpu], max_candidates: int = 4) -> list[LogicalCpu]:
    """Select one logical CPU per physical core, balanced across LLC domains."""

    physical: dict[tuple[int, int], LogicalCpu] = {}
    for row in topology:
        key = (row.package_id, row.core_id)
        if key not in physical or row.cpu_id < physical[key].cpu_id:
            physical[key] = row
    groups: dict[str, list[LogicalCpu]] = {}
    for row in physical.values():
        groups.setdefault(row.cluster_id, []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: row.cpu_id)
    selected = []
    ordered_groups = [groups[key] for key in sorted(groups)]
    offset = 0
    while len(selected) < min(max_candidates, len(physical)):
        made_progress = False
        for values in ordered_groups:
            if offset < len(values):
                selected.append(values[offset])
                made_progress = True
                if len(selected) >= max_candidates:
                    break
        if not made_progress:
            break
        offset += 1
    return selected


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def topology_summary(topology: list[LogicalCpu]) -> dict:
    return {
        "logical_cpus": len(topology),
        "physical_cores": len({(row.package_id, row.core_id) for row in topology}),
        "packages": len({row.package_id for row in topology}),
        "numa_nodes": len({row.numa_node for row in topology}),
        "llc_clusters": len({row.cluster_id for row in topology}),
        "smt_enabled": any(len(row.thread_siblings) > 1 for row in topology),
    }


def _last_level_cache_id(cpu_dir: Path, package_id: int, core_id: int) -> str:
    caches = []
    for index in (cpu_dir / "cache").glob("index*"):
        level = _read_int(index / "level", -1)
        cache_type = _read_text(index / "type", "").lower()
        if level >= 0 and cache_type in {"unified", "data"}:
            shared = _read_text(index / "shared_cpu_list", "")
            cache_id = _read_text(index / "id", "")
            caches.append((level, cache_id or shared))
    if not caches:
        return f"package-{package_id}/core-{core_id}"
    level, identifier = max(caches, key=lambda item: item[0])
    normalized = re.sub(r"\s+", "", identifier)
    return f"package-{package_id}/L{level}-{normalized}"


def _numa_node(cpu_dir: Path) -> int:
    nodes = sorted(cpu_dir.glob("node[0-9]*"))
    return int(nodes[0].name[4:]) if nodes else 0


def _read_text(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def _read_int(path: Path, default: int) -> int:
    try:
        return int(_read_text(path, str(default)))
    except ValueError:
        return default
