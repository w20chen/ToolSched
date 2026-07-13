"""Evaluation metrics for agent remaining-time prediction."""

from __future__ import annotations

import math
from typing import Callable

from ..episodes import AgentEpisode, EpisodeStep, build_episodes
from ..evaluation.metrics import mean, pinball_loss
from ..schema import ToolSample


def remaining_time_metrics(
    samples: list[ToolSample],
    predict_scalar: Callable[[EpisodeStep], float],
    predict_dist: Callable[[EpisodeStep], dict[str, float]] | None = None,
    min_remaining_ms: float = 0.0,
    min_episode_steps: int = 2,
) -> dict:
    """Evaluate remaining-time prediction across all steps in all test episodes.

    Args:
        samples: Flat list of ToolSamples (test set).
        predict_scalar: Predict remaining_time_ms (point estimate).
        predict_dist: Predict {"p50", "p90", "p99"} dict. If None, derived from scalar.
        min_remaining_ms: Only evaluate steps where actual remaining > this threshold.
        min_episode_steps: Skip episodes with fewer than this many steps.

    Returns:
        Dict with MAE, MAPE, R², pinball scores, coverage, etc.
    """
    episodes = build_episodes(samples)
    episodes = [e for e in episodes if e.total_steps >= min_episode_steps]
    if not episodes:
        return {"n_episodes": 0, "n_steps": 0, "error": "no qualifying episodes"}

    all_rows: list[EpisodeStep] = []
    for ep in episodes:
        all_rows.extend(ep.build_training_rows())

    # Filter: optionally exclude steps with trivial remaining time
    rows = [r for r in all_rows if r.remaining_time_ms >= min_remaining_ms]
    if not rows:
        return {"n_episodes": len(episodes), "n_steps": 0, "error": "no steps above threshold"}

    abs_err = []
    abs_pct = []
    smape_vals = []
    pin50_vals = []
    pin90_vals = []
    pin99_vals = []
    cover90 = 0
    cover99 = 0
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = sum(r.remaining_time_ms for r in rows) / len(rows)
    y_abs_sum = 0.0

    for r in rows:
        y = r.remaining_time_ms
        yhat = predict_scalar(r)

        abs_err.append(abs(y - yhat))
        y_abs_sum += abs(y)
        # Only compute percentage error when actual > 1s to avoid explosion.
        if y > 1000.0:
            abs_pct.append(abs(y - yhat) / y)
        denom = (abs(y) + abs(yhat)) / 2.0
        if denom > 1e-9:
            smape_vals.append(abs(y - yhat) / denom)

        ss_res += (y - yhat) ** 2
        ss_tot += (y - y_mean) ** 2

        if predict_dist is not None:
            dist = predict_dist(r)
            p50 = dist.get("p50", yhat)
            p90 = dist.get("p90", yhat * 1.5)
            p99 = dist.get("p99", yhat * 2.0)
        else:
            p50 = yhat
            p90 = yhat * 1.5
            p99 = yhat * 2.0

        pin50_vals.append(pinball_loss(y, p50, 0.50))
        pin90_vals.append(pinball_loss(y, p90, 0.90))
        pin99_vals.append(pinball_loss(y, p99, 0.99))
        cover90 += int(y <= p90)
        cover99 += int(y <= p99)

    n = len(rows)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    med_ape = median_sorted(sorted(abs_pct)) if abs_pct else 0.0

    return {
        "n_episodes": len(episodes),
        "n_steps": n,
        "mae_ms": mean(abs_err),
        "wape": sum(abs_err) / y_abs_sum if y_abs_sum > 0 else 0.0,
        "mape": mean(abs_pct),
        "smape": mean(smape_vals),
        "mdape": med_ape,
        "r_squared": r_squared,
        "pinball_p50": mean(pin50_vals),
        "pinball_p90": mean(pin90_vals),
        "pinball_p99": mean(pin99_vals),
        "coverage_p90": cover90 / n,
        "coverage_p99": cover99 / n,
        "mean_remaining_ms": y_mean,
    }


def remaining_by_progress(
    samples: list[ToolSample],
    predict_scalar: Callable[[EpisodeStep], float],
    n_buckets: int = 10,
    min_episode_steps: int = 2,
) -> dict:
    """Break down remaining-time MAE by progress decile."""
    episodes = build_episodes(samples)
    episodes = [e for e in episodes if e.total_steps >= min_episode_steps]
    rows: list[EpisodeStep] = []
    for ep in episodes:
        rows.extend(ep.build_training_rows())

    buckets: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_buckets)}
    for r in rows:
        b = min(int(r.progress_ratio * n_buckets), n_buckets - 1)
        yhat = predict_scalar(r)
        buckets[b].append((r.remaining_time_ms, yhat))

    breakdown = {}
    for b in range(n_buckets):
        pairs = buckets[b]
        if not pairs:
            continue
        y_vals = [p[0] for p in pairs]
        yhat_vals = [p[1] for p in pairs]
        mae = sum(abs(y - yh) for y, yh in pairs) / len(pairs)
        breakdown[f"decile_{b}"] = {
            "count": len(pairs),
            "mae_ms": mae,
            "mean_actual_ms": sum(y_vals) / len(y_vals),
            "mean_pred_ms": sum(yhat_vals) / len(yhat_vals),
        }

    return {"by_progress_decile": breakdown}


def median_sorted(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
