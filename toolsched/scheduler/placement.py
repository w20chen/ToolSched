"""Interference-aware core placement and honest counterfactual evaluation.

The scheduling action is a concrete core/cluster candidate.  A decision uses
only state observable before launch.  Real counterfactual costs and synthetic
stress tests are reported separately; synthetic costs are never silently used
as ground truth for a real evaluation.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..schema import ToolCostDistribution, ToolSample


PLACEMENT_STATE_FIELDS = (
    "core_util",
    "smt_sibling_util",
    "cluster_util",
    "llc_pressure",
    "memory_bw_pressure",
    "run_queue",
    "frequency_ratio",
)


@dataclass(frozen=True)
class CoreCandidate:
    """One placement action and its pre-launch machine state."""

    candidate_id: str
    core_id: str
    cluster_id: str
    numa_node: str
    core_util: float
    smt_sibling_util: float
    cluster_util: float
    llc_pressure: float
    memory_bw_pressure: float
    run_queue: float
    frequency_ratio: float = 1.0


@dataclass(frozen=True)
class ToolDemand:
    """Coarse, online-available resource demand in the range [0, 1]."""

    cpu: float
    memory: float
    cache: float
    io: float
    parallelism: float


@dataclass(frozen=True)
class PlacementDecision:
    chosen: str
    oracle: str
    chosen_cost: float
    oracle_cost: float
    regret: float
    predicted_cost: float
    source: str


def parse_core_candidates(sample: ToolSample) -> list[CoreCandidate]:
    """Parse pre-launch candidate telemetry from ``sample.resources``.

    Accepted shape::

        resources.placement_candidates = [
          {"candidate_id": "cpu-2", "core_id": 2, "cluster_id": 0,
           "numa_node": 0, "core_util": .1, "smt_sibling_util": .8,
           "cluster_util": .5, "llc_pressure": .4,
           "memory_bw_pressure": .3, "run_queue": 1,
           "frequency_ratio": .95}
        ]

    A mapping keyed by candidate id is accepted as well.  Missing telemetry is
    not invented for real evaluation: malformed candidates are skipped.
    """

    raw = sample.resources.get("placement_candidates")
    if raw is None:
        raw = sample.resources.get("core_candidates")
    items: list[tuple[str | None, Any]] = []
    if isinstance(raw, dict):
        items = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, list):
        items = [(None, value) for value in raw]

    candidates: list[CoreCandidate] = []
    for fallback_id, value in items:
        if not isinstance(value, dict):
            continue
        candidate_id = str(value.get("candidate_id", value.get("id", fallback_id or "")))
        if not candidate_id:
            continue
        if any(field not in value for field in PLACEMENT_STATE_FIELDS):
            continue
        candidates.append(
            CoreCandidate(
                candidate_id=candidate_id,
                core_id=str(value.get("core_id", candidate_id)),
                cluster_id=str(value.get("cluster_id", "unknown")),
                numa_node=str(value.get("numa_node", "unknown")),
                core_util=_unit(value.get("core_util")),
                smt_sibling_util=_unit(value.get("smt_sibling_util")),
                cluster_util=_unit(value.get("cluster_util")),
                llc_pressure=_unit(value.get("llc_pressure")),
                memory_bw_pressure=_unit(value.get("memory_bw_pressure")),
                run_queue=max(0.0, _number(value.get("run_queue"))),
                frequency_ratio=min(2.0, max(0.1, _number(value.get("frequency_ratio"), 1.0))),
            )
        )
    return candidates


def infer_tool_demand(sample: ToolSample) -> ToolDemand:
    """Infer a conservative demand vector without using outcome latency."""

    cls = str(sample.labels.get("resource_class", "unknown"))
    by_class = {
        "cpu": (1.00, 0.35, 0.55, 0.05),
        "cpu_memory_mixed": (0.85, 0.85, 0.80, 0.10),
        "io_search": (0.45, 0.65, 0.55, 0.70),
        "search": (0.35, 0.45, 0.45, 0.55),
        "file_io": (0.25, 0.40, 0.30, 0.90),
        "network": (0.20, 0.25, 0.20, 1.00),
        "light_control": (0.15, 0.10, 0.10, 0.10),
        "unknown": (0.45, 0.45, 0.40, 0.35),
    }
    cpu, memory, cache, io = by_class.get(cls, by_class["unknown"])

    # Prefer measured historical parallelism when supplied by a runtime.  A
    # missing value defaults to one: most command-line tools in current traces
    # must be treated as single-threaded until telemetry proves otherwise.
    parallelism = _first_number(
        sample.resources,
        ("cpu_parallelism", "observed_parallelism", "max_thread_count"),
        default=1.0,
    )
    return ToolDemand(cpu, memory, cache, io, min(64.0, max(1.0, parallelism)))


def predicted_candidate_cost(
    sample: ToolSample,
    candidate: CoreCandidate,
    base_latency_ms: float,
    externality_lambda: float = 0.20,
) -> float:
    """Predict self latency plus a proxy for interference imposed on peers."""

    d = infer_tool_demand(sample)
    parallel_pressure = min(1.0, max(0.0, d.parallelism - 1.0) / 7.0)
    frequency_penalty = max(0.0, 1.0 / candidate.frequency_ratio - 1.0)
    self_interference = (
        d.cpu * (0.50 * candidate.core_util + 0.55 * candidate.smt_sibling_util)
        + d.cache * (0.55 * candidate.llc_pressure + 0.15 * candidate.smt_sibling_util)
        + d.memory * (0.60 * candidate.memory_bw_pressure + 0.20 * candidate.cluster_util)
        + 0.12 * min(candidate.run_queue, 4.0)
        + d.cpu * frequency_penalty
        + parallel_pressure * 0.45 * candidate.cluster_util
    )
    # This term discourages improving the new tool at the expense of tasks
    # already sharing its physical core/LLC/memory domain.
    peer_externality = (
        d.cpu * (0.55 * candidate.smt_sibling_util + 0.15 * candidate.cluster_util)
        + d.cache * 0.50 * candidate.llc_pressure
        + d.memory * 0.55 * candidate.memory_bw_pressure
    )
    base = max(1e-6, float(base_latency_ms))
    return base * (1.0 + self_interference + externality_lambda * peer_externality)


def choose_placement(
    sample: ToolSample,
    predict: Callable[[ToolSample], ToolCostDistribution],
    candidates: list[CoreCandidate] | None = None,
    actual_costs: dict[str, float] | None = None,
    source: str = "real_counterfactual",
) -> PlacementDecision:
    """Choose a candidate using only pre-launch state, then score if costs exist."""

    candidates = candidates if candidates is not None else parse_core_candidates(sample)
    if not candidates:
        raise ValueError("placement requires core-level pre-launch candidate telemetry")
    if not actual_costs:
        raise ValueError("placement evaluation requires counterfactual costs")

    available = [c for c in candidates if c.candidate_id in actual_costs]
    if not available:
        raise ValueError("candidate telemetry and placement_costs have no matching ids")
    pred_base = max(1e-6, float(predict(sample).latency_p90_ms))
    pred_costs = {c.candidate_id: predicted_candidate_cost(sample, c, pred_base) for c in available}
    chosen = min(pred_costs, key=pred_costs.get)
    oracle = min((c.candidate_id for c in available), key=lambda key: float(actual_costs[key]))
    chosen_cost = float(actual_costs[chosen])
    oracle_cost = float(actual_costs[oracle])
    regret = (chosen_cost - oracle_cost) / oracle_cost if oracle_cost > 0 else 0.0
    return PlacementDecision(
        chosen=chosen,
        oracle=oracle,
        chosen_cost=chosen_cost,
        oracle_cost=oracle_cost,
        regret=regret,
        predicted_cost=pred_costs[chosen],
        source=source,
    )


def placement_metrics(
    samples: list[ToolSample],
    predict: Callable[[ToolSample], ToolCostDistribution],
    mode: str = "real",
    synthetic_clusters: int = 2,
    synthetic_cores_per_cluster: int = 2,
) -> dict:
    """Evaluate placement without mixing real and synthetic evidence.

    ``mode='real'`` is the scientifically valid default.  ``synthetic`` is an
    explicit stress test of policy mechanics.  ``both`` reports two separate
    strata and never aggregates their metrics.
    """

    if mode not in {"real", "synthetic", "both"}:
        raise ValueError("mode must be one of: real, synthetic, both")
    rows = [s for s in samples if s.duration_ms is not None]
    payload: dict[str, Any] = {
        "mode": mode,
        "n_input": len(rows),
        "action": "concrete core candidate under pre-launch machine state",
        "objective": "tool_latency_ms + 0.20 * peer_slowdown_ms when peer slowdown is available",
        "required_state_fields": list(PLACEMENT_STATE_FIELDS),
        "warning": "synthetic stress tests are not evidence of real placement speedup",
    }

    if mode in {"real", "both"}:
        scenarios = []
        excluded_no_state = 0
        excluded_no_costs = 0
        for sample in rows:
            candidates = parse_core_candidates(sample)
            costs = _real_costs(sample)
            if not candidates:
                excluded_no_state += 1
            if not costs:
                excluded_no_costs += 1
            if not candidates or not costs:
                continue
            scenarios.append((sample, candidates, costs, "real_counterfactual"))
        real = _evaluate_scenarios(scenarios, predict)
        real.update(
            {
                "n_excluded_no_candidate_state": excluded_no_state,
                "n_excluded_no_counterfactual_costs": excluded_no_costs,
                "evidence": "real counterfactual replay",
            }
        )
        if not scenarios:
            real["status"] = "unavailable: collect candidate state and placement_costs"
        payload["real_counterfactual"] = real

    if mode in {"synthetic", "both"}:
        scenarios = []
        for sample in rows:
            candidates = synthetic_machine_state(
                sample,
                n_clusters=synthetic_clusters,
                cores_per_cluster=synthetic_cores_per_cluster,
            )
            costs = synthetic_hidden_costs(sample, candidates)
            scenarios.append((sample, candidates, costs, "synthetic_stress_test"))
        synthetic = _evaluate_scenarios(scenarios, predict)
        synthetic.update(
            {
                "evidence": "synthetic nonlinear hidden response surface",
                "status": "stress_test_only",
            }
        )
        payload["synthetic_stress_test"] = synthetic
    return payload


def synthetic_machine_state(
    sample: ToolSample,
    n_clusters: int = 2,
    cores_per_cluster: int = 2,
) -> list[CoreCandidate]:
    """Generate deterministic heterogeneous load for explicit stress tests."""

    rng = random.Random(_stable_seed(sample.sample_id, "machine-state"))
    candidates: list[CoreCandidate] = []
    for cluster in range(max(1, n_clusters)):
        cluster_util = rng.uniform(0.08, 0.92)
        llc = _unit(0.15 + 0.70 * cluster_util + rng.uniform(-0.20, 0.20))
        memory_bw = _unit(0.10 + 0.75 * cluster_util + rng.uniform(-0.25, 0.25))
        for local_core in range(max(1, cores_per_cluster)):
            core_id = cluster * max(1, cores_per_cluster) + local_core
            core_util = _unit(cluster_util + rng.uniform(-0.45, 0.35))
            sibling = _unit(0.45 * cluster_util + rng.uniform(-0.15, 0.55))
            candidates.append(
                CoreCandidate(
                    candidate_id=f"cluster-{cluster}/core-{core_id}",
                    core_id=str(core_id),
                    cluster_id=str(cluster),
                    numa_node=str(cluster),
                    core_util=core_util,
                    smt_sibling_util=sibling,
                    cluster_util=cluster_util,
                    llc_pressure=llc,
                    memory_bw_pressure=memory_bw,
                    run_queue=float(max(0, round(core_util * 3 + rng.uniform(-0.5, 1.0)))),
                    frequency_ratio=rng.uniform(0.78, 1.08),
                )
            )
    return candidates


def synthetic_hidden_costs(
    sample: ToolSample,
    candidates: Iterable[CoreCandidate],
    base_ms: float | None = None,
) -> dict[str, float]:
    """Independent nonlinear oracle used only to stress-test policy ranking.

    Its functional form deliberately differs from ``predicted_candidate_cost``
    and includes stable unobserved heterogeneity.  This prevents the previous
    tautology where policy and oracle minimized identical hand-written factors.
    """

    d = infer_tool_demand(sample)
    base = max(1e-6, float(base_ms if base_ms is not None else sample.duration_ms or 1.0))
    costs: dict[str, float] = {}
    for c in candidates:
        rng = random.Random(_stable_seed(sample.sample_id, c.candidate_id, "hidden-cost"))
        frequency_penalty = max(0.0, 1.0 / c.frequency_ratio - 1.0)
        nonlinear = (
            d.cpu * (0.62 * c.core_util**1.6 + 0.72 * c.smt_sibling_util**2.0)
            + d.cache * (0.82 * c.llc_pressure**1.35 + 0.18 * c.smt_sibling_util)
            + d.memory * (0.76 * c.memory_bw_pressure**1.45 + 0.16 * c.cluster_util**2)
            + 0.16 * math.log1p(c.run_queue)
            + 0.85 * d.cpu * frequency_penalty
        )
        # Latent effects represent unmeasured co-runner phase and microthermal
        # variation.  They are bounded so observable state remains informative.
        latent = rng.uniform(-0.06, 0.06)
        costs[c.candidate_id] = max(1e-6, base * (1.0 + nonlinear + latent))
    return costs


def _evaluate_scenarios(
    scenarios: list[tuple[ToolSample, list[CoreCandidate], dict[str, float], str]],
    predict: Callable[[ToolSample], ToolCostDistribution],
) -> dict:
    if not scenarios:
        return {"n": 0, "top1_accuracy": None, "normalized_regret_mean": None, "baselines": {}}

    decisions = [choose_placement(s, predict, c, costs, source) for s, c, costs, source in scenarios]
    baseline_names = ("os_proxy", "least_core_util", "smt_aware", "least_cluster_load", "random")
    baseline_metrics = {
        name: _policy_metrics(scenarios, name)
        for name in baseline_names
    }
    mean_cost = sum(d.chosen_cost for d in decisions) / len(decisions)
    os_cost = baseline_metrics["os_proxy"]["mean_cost_ms"]
    regrets = [d.regret for d in decisions]
    paired_reductions = []
    by_resource: dict[str, list[float]] = {}
    for decision, (sample, candidates, costs, _) in zip(decisions, scenarios):
        os_choice = _baseline_choice(
            sample,
            [c for c in candidates if c.candidate_id in costs],
            "os_proxy",
        )
        os_row_cost = float(costs[os_choice])
        paired_reductions.append((os_row_cost - decision.chosen_cost) / os_row_cost if os_row_cost > 0 else 0.0)
        cls = str(sample.labels.get("resource_class", "unknown"))
        by_resource.setdefault(cls, []).append(decision.regret)
    return {
        "n": len(decisions),
        "top1_accuracy": sum(d.chosen == d.oracle for d in decisions) / len(decisions),
        "normalized_regret_mean": sum(regrets) / len(regrets),
        "normalized_regret_ci95": _bootstrap_mean_ci(regrets, "policy-regret"),
        "mean_cost_ms": mean_cost,
        "relative_cost_reduction_vs_os_proxy": (os_cost - mean_cost) / os_cost if os_cost > 0 else 0.0,
        "paired_cost_reduction_vs_os_proxy_ci95": _bootstrap_mean_ci(
            paired_reductions, "policy-vs-os"
        ),
        "chosen_candidates": sorted({d.chosen for d in decisions}),
        "baselines": baseline_metrics,
        "by_resource_class": {
            cls: {
                "n": len(values),
                "normalized_regret_mean": sum(values) / len(values),
            }
            for cls, values in sorted(by_resource.items())
        },
        "oracle": {"normalized_regret_mean": 0.0},
    }


def _policy_metrics(
    scenarios: list[tuple[ToolSample, list[CoreCandidate], dict[str, float], str]],
    policy: str,
) -> dict[str, float]:
    regrets: list[float] = []
    costs_out: list[float] = []
    correct = 0
    for sample, candidates, costs, _ in scenarios:
        available = [c for c in candidates if c.candidate_id in costs]
        chosen = _baseline_choice(sample, available, policy)
        oracle = min((c.candidate_id for c in available), key=lambda key: float(costs[key]))
        chosen_cost = float(costs[chosen])
        oracle_cost = float(costs[oracle])
        regrets.append((chosen_cost - oracle_cost) / oracle_cost if oracle_cost > 0 else 0.0)
        costs_out.append(chosen_cost)
        correct += int(chosen == oracle)
    return {
        "top1_accuracy": correct / len(scenarios),
        "normalized_regret_mean": sum(regrets) / len(regrets),
        "normalized_regret_ci95": _bootstrap_mean_ci(regrets, f"baseline-{policy}"),
        "mean_cost_ms": sum(costs_out) / len(costs_out),
    }


def _baseline_choice(sample: ToolSample, candidates: list[CoreCandidate], policy: str) -> str:
    if not candidates:
        raise ValueError("baseline policy requires at least one candidate")
    if policy == "least_core_util":
        chosen = min(candidates, key=lambda c: (c.core_util, c.run_queue, c.candidate_id))
    elif policy == "smt_aware":
        chosen = min(
            candidates,
            key=lambda c: (c.core_util + 0.8 * c.smt_sibling_util + 0.1 * c.run_queue, c.candidate_id),
        )
    elif policy == "least_cluster_load":
        chosen = min(
            candidates,
            key=lambda c: (c.cluster_util + 0.4 * c.memory_bw_pressure + 0.3 * c.llc_pressure, c.core_util),
        )
    elif policy == "random":
        chosen = random.Random(_stable_seed(sample.sample_id, "random-baseline")).choice(candidates)
    else:  # os_proxy: load balance using only conventional scheduler-visible pressure
        chosen = min(candidates, key=lambda c: (c.core_util + 0.15 * c.run_queue, c.candidate_id))
    return chosen.candidate_id


def _real_costs(sample: ToolSample) -> dict[str, float]:
    raw = sample.labels.get("placement_costs")
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            latency = _number(value.get("tool_latency_ms"), float("nan"))
            peer = _number(value.get("peer_slowdown_ms"), 0.0)
            number = latency + 0.20 * max(0.0, peer)
        else:
            number = _number(value, float("nan"))
        if math.isfinite(number) and number > 0:
            out[str(key)] = number
    return out


def _bootstrap_mean_ci(values: list[float], seed_key: str, draws: int = 1000) -> list[float] | None:
    """Deterministic percentile bootstrap CI for paired scenario summaries."""

    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(_stable_seed(seed_key, str(len(values))))
    means = []
    n = len(values)
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def _stable_seed(*parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _first_number(values: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in values:
            return _number(values[key], default)
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unit(value: Any) -> float:
    return min(1.0, max(0.0, _number(value)))
