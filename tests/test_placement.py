from __future__ import annotations

import unittest

from toolsched.schema import ToolCostDistribution, ToolSample
from toolsched.scheduler.placement import (
    CoreCandidate,
    choose_placement,
    placement_metrics,
    predicted_candidate_cost,
    synthetic_hidden_costs,
)


def sample(sample_id: str = "case/tool-1") -> ToolSample:
    return ToolSample(
        sample_id=sample_id,
        dataset="test",
        case_id="case",
        attempt_id="attempt",
        tool="python",
        operation="python",
        tool_family="terminal",
        timestamp=None,
        duration_ms=1000.0,
        labels={"resource_class": "cpu"},
    )


def predict(_: ToolSample) -> ToolCostDistribution:
    return ToolCostDistribution(800.0, 1000.0, 1500.0, resource_class="cpu")


def candidate(
    name: str,
    core: float,
    sibling: float,
    cluster: float,
    llc: float = 0.2,
    memory: float = 0.2,
    queue: float = 0.0,
) -> CoreCandidate:
    return CoreCandidate(
        candidate_id=name,
        core_id=name,
        cluster_id=name,
        numa_node="0",
        core_util=core,
        smt_sibling_util=sibling,
        cluster_util=cluster,
        llc_pressure=llc,
        memory_bw_pressure=memory,
        run_queue=queue,
        frequency_ratio=1.0,
    )


class PlacementTests(unittest.TestCase):
    def test_same_single_threaded_tool_moves_when_interference_moves(self) -> None:
        s = sample()
        state_a = [
            candidate("core-a", 0.05, 0.05, 0.20),
            candidate("core-b", 0.80, 0.90, 0.85),
        ]
        state_b = [
            candidate("core-a", 0.80, 0.90, 0.85),
            candidate("core-b", 0.05, 0.05, 0.20),
        ]
        costs = {"core-a": 1000.0, "core-b": 1100.0}

        first = choose_placement(s, predict, state_a, costs)
        second = choose_placement(s, predict, state_b, costs)

        self.assertEqual(first.chosen, "core-a")
        self.assertEqual(second.chosen, "core-b")

    def test_busy_smt_sibling_can_outweigh_local_core_idle(self) -> None:
        s = sample()
        idle_logical_busy_sibling = candidate("core-a", 0.0, 1.0, 0.3)
        moderately_busy_isolated = candidate("core-b", 0.25, 0.0, 0.3)

        a = predicted_candidate_cost(s, idle_logical_busy_sibling, 1000.0)
        b = predicted_candidate_cost(s, moderately_busy_isolated, 1000.0)

        self.assertLess(b, a)

    def test_real_mode_never_falls_back_to_synthetic_ground_truth(self) -> None:
        result = placement_metrics([sample()], predict, mode="real")

        self.assertEqual(result["real_counterfactual"]["n"], 0)
        self.assertIn("unavailable", result["real_counterfactual"]["status"])
        self.assertNotIn("synthetic_stress_test", result)

    def test_real_counterfactual_uses_matching_candidate_ids(self) -> None:
        s = sample()
        s.resources["placement_candidates"] = [
            {
                "candidate_id": "core-a",
                "core_id": 0,
                "cluster_id": 0,
                "numa_node": 0,
                "core_util": 0.05,
                "smt_sibling_util": 0.05,
                "cluster_util": 0.10,
                "llc_pressure": 0.10,
                "memory_bw_pressure": 0.10,
                "run_queue": 0,
                "frequency_ratio": 1.0,
            },
            {
                "candidate_id": "core-b",
                "core_id": 1,
                "cluster_id": 1,
                "numa_node": 1,
                "core_util": 0.80,
                "smt_sibling_util": 0.80,
                "cluster_util": 0.80,
                "llc_pressure": 0.80,
                "memory_bw_pressure": 0.80,
                "run_queue": 2,
                "frequency_ratio": 0.9,
            },
        ]
        s.labels["placement_costs"] = {"core-a": 900.0, "core-b": 1400.0}

        result = placement_metrics([s], predict, mode="real")

        self.assertEqual(result["real_counterfactual"]["n"], 1)
        self.assertEqual(result["real_counterfactual"]["top1_accuracy"], 1.0)
        self.assertEqual(result["real_counterfactual"]["normalized_regret_mean"], 0.0)

    def test_synthetic_oracle_is_not_the_policy_cost_function(self) -> None:
        s = sample("case/independent-oracle")
        candidates = [
            candidate("core-a", 0.10, 0.75, 0.25, llc=0.20, memory=0.20),
            candidate("core-b", 0.35, 0.10, 0.70, llc=0.65, memory=0.65),
        ]
        predicted = [predicted_candidate_cost(s, c, 1000.0) for c in candidates]
        hidden = synthetic_hidden_costs(s, candidates, base_ms=1000.0)
        predicted_ratio = predicted[0] / predicted[1]
        hidden_ratio = hidden["core-a"] / hidden["core-b"]

        self.assertNotAlmostEqual(predicted_ratio, hidden_ratio, places=5)

    def test_synthetic_results_are_explicitly_labeled(self) -> None:
        rows = [sample(f"case/tool-{i}") for i in range(20)]
        result = placement_metrics(rows, predict, mode="synthetic")

        self.assertNotIn("real_counterfactual", result)
        self.assertEqual(result["synthetic_stress_test"]["n"], 20)
        self.assertEqual(result["synthetic_stress_test"]["status"], "stress_test_only")
        self.assertIn("os_proxy", result["synthetic_stress_test"]["baselines"])
        self.assertEqual(len(result["synthetic_stress_test"]["normalized_regret_ci95"]), 2)

    def test_real_objective_can_include_peer_slowdown(self) -> None:
        s = sample("case/peer-impact")
        s.resources["placement_candidates"] = [
            {
                "candidate_id": name,
                "core_id": idx,
                "cluster_id": idx,
                "numa_node": 0,
                "core_util": util,
                "smt_sibling_util": util,
                "cluster_util": util,
                "llc_pressure": util,
                "memory_bw_pressure": util,
                "run_queue": 0,
                "frequency_ratio": 1.0,
            }
            for idx, (name, util) in enumerate((("core-a", 0.05), ("core-b", 0.70)))
        ]
        # core-a is best for the new tool but harms existing co-runners enough
        # that the declared system objective prefers core-b.
        s.labels["placement_costs"] = {
            "core-a": {"tool_latency_ms": 900.0, "peer_slowdown_ms": 1000.0},
            "core-b": {"tool_latency_ms": 1000.0, "peer_slowdown_ms": 0.0},
        }

        result = placement_metrics([s], predict, mode="real")["real_counterfactual"]

        self.assertEqual(result["top1_accuracy"], 0.0)
        self.assertAlmostEqual(result["normalized_regret_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
