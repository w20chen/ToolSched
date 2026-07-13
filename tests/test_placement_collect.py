from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolsched.scheduler.linux_topology import LogicalCpu, parse_cpu_list, select_candidate_cpus
from toolsched.scheduler.placement_collect import (
    _parse_perf_stat,
    _peer_slowdown_ms,
    aggregate_placement_rows,
    build_interference_scenarios,
)
from toolsched.scheduler.placement_manifest import (
    build_toolsched_study_manifest,
    validate_read_only_command,
)


def cpu(cpu_id: int, core_id: int, cluster: str, siblings: tuple[int, ...]) -> LogicalCpu:
    return LogicalCpu(cpu_id, core_id, 0, 0, cluster, siblings)


class PlacementCollectionTests(unittest.TestCase):
    def test_cpu_list_parser(self) -> None:
        self.assertEqual(parse_cpu_list("0-2,5,8-9"), {0, 1, 2, 5, 8, 9})

    def test_candidate_selection_balances_clusters_and_avoids_smt_duplicates(self) -> None:
        topology = [
            cpu(0, 0, "L3-a", (0, 4)), cpu(4, 0, "L3-a", (0, 4)),
            cpu(1, 1, "L3-a", (1, 5)), cpu(5, 1, "L3-a", (1, 5)),
            cpu(2, 2, "L3-b", (2, 6)), cpu(6, 2, "L3-b", (2, 6)),
            cpu(3, 3, "L3-b", (3, 7)), cpu(7, 3, "L3-b", (3, 7)),
        ]
        selected = select_candidate_cpus(topology, max_candidates=4)

        self.assertEqual({row.cluster_id for row in selected}, {"L3-a", "L3-b"})
        self.assertEqual(len({row.core_id for row in selected}), 4)

    def test_scenarios_do_not_place_corunners_on_candidate_logical_cpus(self) -> None:
        topology = [
            cpu(0, 0, "L3-a", (0, 4)), cpu(4, 0, "L3-a", (0, 4)),
            cpu(1, 1, "L3-a", (1, 5)), cpu(5, 1, "L3-a", (1, 5)),
            cpu(2, 2, "L3-b", (2, 6)), cpu(6, 2, "L3-b", (2, 6)),
            cpu(3, 3, "L3-b", (3, 7)), cpu(7, 3, "L3-b", (3, 7)),
            cpu(8, 4, "L3-a", (8,)), cpu(9, 5, "L3-b", (9,)),
        ]
        candidates = [topology[0], topology[2], topology[4], topology[6]]
        scenarios = build_interference_scenarios(topology, candidates, max_corunners=2)
        candidate_ids = {row.cpu_id for row in candidates}

        self.assertIn("idle", {scenario.name for scenario in scenarios})
        for scenario in scenarios:
            self.assertFalse(candidate_ids.intersection(scenario.busy_cpus))

    def test_automatic_command_approval_is_conservative(self) -> None:
        self.assertTrue(validate_read_only_command("rg TODO toolsched | head -20")[0])
        self.assertTrue(validate_read_only_command("git status --short")[0])
        self.assertFalse(validate_read_only_command("pytest -q")[0])
        self.assertFalse(validate_read_only_command("find . -name '*.py' -delete")[0])
        self.assertFalse(validate_read_only_command("cat a > b")[0])

    def test_perf_csv_parser(self) -> None:
        parsed = _parse_perf_stat(
            "100000,,cycles,100.00,100.00\n"
            "2000,,cache-references,100.00,100.00\n"
            "500,,cache-misses,100.00,100.00\n"
        )
        self.assertEqual(parsed["cycles"], 100000.0)
        self.assertEqual(parsed["cache-misses"], 500.0)

    def test_peer_slowdown_removes_prelaunch_warmup_work(self) -> None:
        concurrent = [{
            "operations": 150.0,
            "elapsed_seconds": 2.0,
            "operations_per_second": 75.0,
        }]
        solo = [{"operations_per_second": 100.0}]
        self.assertAlmostEqual(_peer_slowdown_ms(concurrent, solo, 1.0), 1000.0)

    def test_aggregation_keeps_real_repeats_and_nested_costs(self) -> None:
        entry = {
            "invocation_id": "inv-1", "dataset": "study", "case_id": "case-1",
            "tool": "python", "operation": "python", "tool_family": "terminal",
            "resource_class": "cpu", "command": "echo ok", "cwd": ".",
        }
        states = [
            {
                "candidate_id": candidate_id, "core_id": candidate_id[-1],
                "logical_cpu_id": idx, "cluster_id": f"L3-{idx}", "numa_node": "0",
                "core_util": 0.1 + idx * 0.2, "smt_sibling_util": 0.2,
                "cluster_util": 0.3, "llc_pressure": 0.4,
                "memory_bw_pressure": 0.2, "run_queue": 0.1, "frequency_ratio": 1.0,
            }
            for idx, candidate_id in enumerate(("cpu-0", "cpu-1"))
        ]
        rows = []
        for repeat in range(3):
            for idx, candidate_id in enumerate(("cpu-0", "cpu-1")):
                rows.append(
                    {
                        "invocation_id": "inv-1", "scenario": "idle", "repeat": repeat,
                        "chosen_candidate": candidate_id, "candidate_states": states,
                        "tool_latency_ms": 100.0 + 50.0 * idx + repeat,
                        "peer_slowdown_ms": 0.0, "peer_measurement_valid": True,
                        "success": True, "entry": entry, "machine": {"hostname": "test"},
                        "state_provenance": {"cpu_util": "test"},
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            samples, summary = aggregate_placement_rows(raw, min_success_per_candidate=3)

        self.assertEqual(summary["included"], 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].labels["placement_costs"]["cpu-0"]["tool_latency_ms"], 101.0)
        self.assertEqual(len(samples[0].resources["placement_candidates"]), 2)

    def test_study_manifest_expands_workloads_by_case_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            samples.write_text("{}\n", encoding="utf-8")
            payload = build_toolsched_study_manifest(samples, root, shard_count=3)

        self.assertEqual(len(payload["entries"]), 15)
        self.assertEqual(len({entry["invocation_id"] for entry in payload["entries"]}), 15)
        self.assertTrue(all(entry["approved"] for entry in payload["entries"]))

    def test_study_manifest_can_select_a_quick_workload_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            samples.write_text("{}\n", encoding="utf-8")
            payload = build_toolsched_study_manifest(
                samples, root, shard_count=1, workload_names={"quantile"}
            )

        self.assertEqual(len(payload["entries"]), 1)
        self.assertIn("--workload quantile", payload["entries"][0]["command"])


if __name__ == "__main__":
    unittest.main()
