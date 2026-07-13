from __future__ import annotations

from typing import Callable

from ..schema import ToolCostDistribution, ToolSample


SAFE_READ_ONLY = {
    "read_file",
    "list_dir",
    "web_search",
    "web_fetch",
    "exec-grep",
    "grep",
    "find",
    "ls",
    "pwd",
}


def is_speculation_safe(sample: ToolSample) -> bool:
    if sample.tool in SAFE_READ_ONLY:
        return True
    if sample.operation in {"grep", "find", "git_status", "git_diff", "read_file"}:
        return True
    return False


def admit_speculation(
    sample: ToolSample,
    predict: Callable[[ToolSample], ToolCostDistribution],
    match_probability: float,
    llm_slack_ms: float,
    cost_lambda: float = 1.0,
) -> tuple[bool, dict]:
    pred = predict(sample)
    hidden = min(pred.latency_p50_ms, llm_slack_ms)
    benefit = match_probability * hidden
    waste = (1 - match_probability) * pred.latency_p50_ms
    interference = 0.15 * pred.latency_p90_ms
    cost = waste + interference
    safe = is_speculation_safe(sample)
    admit = safe and benefit > cost_lambda * cost
    return admit, {
        "benefit_ms": benefit,
        "cost_ms": cost,
        "safe": safe,
        "predicted_latency_ms": pred.latency_p50_ms,
    }


def speculation_metrics(
    samples: list[ToolSample],
    predict: Callable[[ToolSample], ToolCostDistribution],
    predict_next: Callable[[ToolSample], tuple[str | None, float]],
    llm_slack_ms: float = 1500.0,
) -> dict:
    candidates = []
    for s in samples:
        next_tool, conf = predict_next(s)
        if not next_tool:
            continue
        proxy = s
        admit, info = admit_speculation(proxy, predict, conf, llm_slack_ms)
        hit = int(next_tool == s.next_tool) if s.next_tool else 0
        candidates.append((admit, hit, info))
    if not candidates:
        return {}
    admitted = [c for c in candidates if c[0]]
    hits = sum(hit for _, hit, _ in admitted)
    wasted = sum(info["predicted_latency_ms"] for admit, hit, info in admitted if not hit)
    hidden = sum(info["benefit_ms"] for admit, hit, info in admitted if hit)
    return {
        "n_candidates": len(candidates),
        "n_admitted": len(admitted),
        "admission_rate": len(admitted) / len(candidates),
        "hit_rate_admitted": hits / len(admitted) if admitted else 0.0,
        "estimated_hidden_latency_ms": hidden,
        "estimated_wasted_cpu_ms": wasted,
    }

