from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..schema import ToolCostDistribution, ToolSample


PLACEMENTS = ("os_default", "single_core", "compact_l3", "spread_numa", "cross_socket")


@dataclass(frozen=True)
class PlacementDecision:
    chosen: str
    oracle: str
    chosen_cost: float
    oracle_cost: float
    regret: float


def synthetic_costs(sample: ToolSample, base_ms: float | None = None) -> dict[str, float]:
    base = base_ms if base_ms is not None else (sample.duration_ms or 1.0)
    cls = sample.labels.get("resource_class", "unknown")
    factors = {
        "os_default": 1.0,
        "single_core": 1.15,
        "compact_l3": 0.95,
        "spread_numa": 1.05,
        "cross_socket": 1.25,
    }
    if cls in {"cpu", "cpu_memory_mixed"}:
        factors.update({"single_core": 1.30, "compact_l3": 0.88, "spread_numa": 1.00})
    elif cls in {"io_search", "file_io", "network"}:
        factors.update({"single_core": 1.02, "compact_l3": 0.98, "spread_numa": 0.96, "cross_socket": 1.05})
    elif cls == "light_control":
        factors.update({"single_core": 1.0, "compact_l3": 1.0, "spread_numa": 1.0, "cross_socket": 1.0})
    return {p: max(0.0, base * f) for p, f in factors.items()}


def choose_placement(sample: ToolSample, predict: Callable[[ToolSample], ToolCostDistribution]) -> PlacementDecision:
    costs = sample.labels.get("placement_costs")
    if not isinstance(costs, dict):
        costs = synthetic_costs(sample)
    pred_base = predict(sample).latency_p90_ms
    pred_costs = synthetic_costs(sample, pred_base)
    chosen = min(pred_costs, key=pred_costs.get)
    oracle = min(costs, key=costs.get)
    chosen_cost = float(costs[chosen])
    oracle_cost = float(costs[oracle])
    regret = (chosen_cost - oracle_cost) / oracle_cost if oracle_cost > 0 else 0.0
    return PlacementDecision(chosen, oracle, chosen_cost, oracle_cost, regret)


def placement_metrics(samples: list[ToolSample], predict: Callable[[ToolSample], ToolCostDistribution]) -> dict:
    rows = [s for s in samples if s.duration_ms is not None]
    if not rows:
        return {}
    decisions = [choose_placement(s, predict) for s in rows]
    top1 = sum(d.chosen == d.oracle for d in decisions) / len(decisions)
    regret = sum(d.regret for d in decisions) / len(decisions)
    baselines = {
        "os_default": _fixed_policy_regret(rows, "os_default"),
        "compact_l3": _fixed_policy_regret(rows, "compact_l3"),
        "spread_numa": _fixed_policy_regret(rows, "spread_numa"),
        "cross_socket": _fixed_policy_regret(rows, "cross_socket"),
        "cost_model": regret,
        "oracle": 0.0,
    }
    return {
        "n": len(decisions),
        "top1_accuracy": top1,
        "normalized_regret_mean": regret,
        "baseline_regret_mean": baselines,
        "placements": sorted({d.chosen for d in decisions}),
        "mode": "real_counterfactual_if_available_else_synthetic",
    }


def _fixed_policy_regret(samples: list[ToolSample], placement: str) -> float:
    regrets = []
    for sample in samples:
        costs = sample.labels.get("placement_costs")
        if not isinstance(costs, dict):
            costs = synthetic_costs(sample)
        oracle_cost = min(float(v) for v in costs.values())
        chosen_cost = float(costs.get(placement, costs.get("os_default", oracle_cost)))
        regrets.append((chosen_cost - oracle_cost) / oracle_cost if oracle_cost > 0 else 0.0)
    return sum(regrets) / len(regrets) if regrets else 0.0
