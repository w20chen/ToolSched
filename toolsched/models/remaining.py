"""Remaining-time prediction models for agent episodes.

Pythia-style approach: lightweight, interpretable models that predict
how much wall-clock time remains in an agent episode at each step.

Models implemented:
- GlobalRemainingQuantile: unconditional global baseline
- StepConditionedRemaining: grouped by progress decile
- LinearQuantileRemaining: pinball-loss linear regression (Pythia core)
- LogSpaceRemaining: linear regression in log(y+1) space
- EwmaRemainingByFamily: EWMA per tool_family
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import median
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from ..episodes import AgentEpisode, EpisodeStep, build_episodes
from ..models.baselines import quantile
from ..schema import ToolSample


# ──────────────────────────────────────────────
#  helpers
# ──────────────────────────────────────────────

def _feature_matrix(rows: list[EpisodeStep]) -> list[list[float]]:
    """Convert rows to float feature matrix (list of lists)."""
    names = EpisodeStep.feature_names()
    X = []
    for r in rows:
        fv = r.feature_vector()
        X.append([fv[name] for name in names])
    return X


def _targets(rows: list[EpisodeStep]) -> list[float]:
    return [r.remaining_time_ms for r in rows]


def _normalize_features(
    train_X: list[list[float]],
    test_X: list[list[float]] | None = None,
) -> tuple[list[list[float]], list[list[float]] | None, list[float], list[float]]:
    """Z-score normalize each feature column. Returns (train_norm, test_norm, means, stds)."""
    n_features = len(train_X[0]) if train_X else 0
    means = []
    stds = []
    for j in range(n_features):
        col = [row[j] for row in train_X]
        mu = sum(col) / len(col)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in col) / len(col)) if len(col) > 1 else 1.0
        if sigma < 1e-9:
            sigma = 1.0
        means.append(mu)
        stds.append(sigma)

    def _norm(matrix: list[list[float]]) -> list[list[float]]:
        result = []
        for row in matrix:
            result.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])
        return result

    train_norm = _norm(train_X)
    test_norm = _norm(test_X) if test_X is not None else None
    return train_norm, test_norm, means, stds


# ──────────────────────────────────────────────
#  1. Global baseline
# ──────────────────────────────────────────────

class GlobalRemainingQuantile:
    """Unconditional remaining-time quantiles from all training steps."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def fit(self, rows: list[EpisodeStep]) -> "GlobalRemainingQuantile":
        self.values = [r.remaining_time_ms for r in rows]
        return self

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        return {
            "p50": quantile(self.values, 0.50),
            "p90": quantile(self.values, 0.90),
            "p99": quantile(self.values, 0.99),
        }

    def predict_scalar(self, row: EpisodeStep) -> float:
        return quantile(self.values, 0.50)


# ──────────────────────────────────────────────
#  2. Step-conditioned quantile
# ──────────────────────────────────────────────

class StepConditionedRemaining:
    """Group rows by step_index (capped at max_steps), use per-group quantiles."""

    def __init__(self, max_steps: int = 50) -> None:
        self.max_steps = max_steps
        self.global_model = GlobalRemainingQuantile()
        self.groups: dict[int, list[float]] = {}

    def fit(self, rows: list[EpisodeStep]) -> "StepConditionedRemaining":
        self.global_model.fit(rows)
        groups: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            bucket = min(r.step_index, self.max_steps)
            groups[bucket].append(r.remaining_time_ms)
        self.groups = dict(groups)
        return self

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        bucket = min(row.step_index, self.max_steps)
        vals = self.groups.get(bucket)
        if not vals:
            return self.global_model.predict(row)
        return {
            "p50": quantile(vals, 0.50),
            "p90": quantile(vals, 0.90),
            "p99": quantile(vals, 0.99),
        }

    def predict_scalar(self, row: EpisodeStep) -> float:
        bucket = min(row.step_index, self.max_steps)
        vals = self.groups.get(bucket)
        if vals:
            return quantile(vals, 0.50)
        return self.global_model.predict_scalar(row)


# ──────────────────────────────────────────────
#  3. Progress-conditioned quantile
# ──────────────────────────────────────────────

class ProgressConditionedRemaining:
    """Group by progress decile (cumulative_time / total_time), per-group quantiles.

    This uses oracle knowledge of total_time for training, but only cumulative_time
    at inference. During training, we compute the true progress ratio to assign buckets.
    At inference, we predict progress first using a simple regressor.
    """

    def __init__(self, n_buckets: int = 10) -> None:
        self.n_buckets = n_buckets
        self.global_model = GlobalRemainingQuantile()
        self.groups: dict[int, list[float]] = {}
        # Simple linear model to predict progress ratio from features at inference
        self._progress_weights: list[float] = []
        self._progress_bias: float = 0.0
        self._feat_means: list[float] = []
        self._feat_stds: list[float] = []

    def fit(self, rows: list[EpisodeStep]) -> "ProgressConditionedRemaining":
        self.global_model.fit(rows)

        # Train progress predictor
        X = _feature_matrix(rows)
        y_progress = [r.progress_ratio for r in rows]
        Xn, _, self._feat_means, self._feat_stds = _normalize_features(X)
        self._progress_weights, self._progress_bias = _linear_fit(Xn, y_progress, lr=0.05, epochs=200)

        # Group by oracle progress decile for remaining-time quantiles
        groups: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            bucket = min(int(r.progress_ratio * self.n_buckets), self.n_buckets - 1)
            groups[bucket].append(r.remaining_time_ms)
        self.groups = dict(groups)
        return self

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        # Predict progress ratio, then use corresponding bucket
        X = _feature_matrix([row])
        Xn = _apply_norm(X, self._feat_means, self._feat_stds)
        pred_progress = _linear_predict(Xn[0], self._progress_weights, self._progress_bias)
        pred_progress = max(0.0, min(1.0, pred_progress))
        bucket = min(int(pred_progress * self.n_buckets), self.n_buckets - 1)
        vals = self.groups.get(bucket)
        if not vals:
            return self.global_model.predict(row)
        return {
            "p50": quantile(vals, 0.50),
            "p90": quantile(vals, 0.90),
            "p99": quantile(vals, 0.99),
        }

    def predict_scalar(self, row: EpisodeStep) -> float:
        return self.predict(row)["p50"]


# ──────────────────────────────────────────────
#  4. Linear quantile regression (Pythia core)
# ──────────────────────────────────────────────

class LinearQuantileRemaining:
    """Linear model trained with pinball loss at multiple quantiles.

    Features are z-score normalized; targets are also z-score normalized
    during training to keep gradient magnitudes reasonable.  Predictions
    are un-normalised back to milliseconds.

    This is the Pythia-style core: a lightweight linear regressor on engineered
    features, producing quantile predictions for calibrated uncertainty.
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 500,
        batch_size: int = 256,
        l2: float = 0.0001,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.weights: dict[float, list[float]] = {}
        self.biases: dict[float, float] = {}
        self.feat_means: list[float] = []
        self.feat_stds: list[float] = []
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def fit(
        self, rows: list[EpisodeStep], quantiles: tuple[float, ...] = (0.50, 0.90, 0.99)
    ) -> "LinearQuantileRemaining":
        X = _feature_matrix(rows)
        y_raw = _targets(rows)
        Xn, _, self.feat_means, self.feat_stds = _normalize_features(X)

        # Normalize targets so gradients are well-behaved
        self.target_mean = sum(y_raw) / len(y_raw) if y_raw else 0.0
        var = sum((v - self.target_mean) ** 2 for v in y_raw) / len(y_raw) if len(y_raw) > 1 else 1.0
        self.target_std = math.sqrt(var) if var > 1e-9 else 1.0
        y_norm = [(v - self.target_mean) / self.target_std for v in y_raw]

        for q in quantiles:
            w, b = _quantile_regression_fit(
                Xn, y_norm, q,
                lr=self.lr, epochs=self.epochs,
                batch_size=self.batch_size, l2=self.l2,
            )
            self.weights[q] = w
            self.biases[q] = b
        return self

    def _predict_raw(self, row: EpisodeStep, q: float) -> float:
        X = _feature_matrix([row])
        Xn = _apply_norm(X, self.feat_means, self.feat_stds)
        w = self.weights.get(q, [])
        b = self.biases.get(q, 0.0)
        pred_norm = _linear_predict(Xn[0], w, b)
        # Un-normalize
        pred_ms = pred_norm * self.target_std + self.target_mean
        return max(0.0, pred_ms)

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        return {
            "p50": self._predict_raw(row, 0.50),
            "p90": self._predict_raw(row, 0.90),
            "p99": self._predict_raw(row, 0.99),
        }

    def predict_scalar(self, row: EpisodeStep) -> float:
        return self._predict_raw(row, 0.50)


# ──────────────────────────────────────────────
#  5. Log-space linear regression
# ──────────────────────────────────────────────

class LogSpaceRemaining:
    """Linear regression on log(remaining_time + 1), then exponentiate.

    Better handles the heavy-tailed distribution of remaining times.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 300, l2: float = 0.001) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.feat_means: list[float] = []
        self.feat_stds: list[float] = []

    def fit(self, rows: list[EpisodeStep]) -> "LogSpaceRemaining":
        X = _feature_matrix(rows)
        y_log = [math.log(r.remaining_time_ms + 1.0) for r in rows]
        Xn, _, self.feat_means, self.feat_stds = _normalize_features(X)
        self.weights, self.bias = _mse_regression_fit(
            Xn, y_log, lr=self.lr, epochs=self.epochs, l2=self.l2,
        )
        return self

    def predict_scalar(self, row: EpisodeStep) -> float:
        X = _feature_matrix([row])
        Xn = _apply_norm(X, self.feat_means, self.feat_stds)
        log_pred = _linear_predict(Xn[0], self.weights, self.bias)
        return max(0.0, math.exp(log_pred) - 1.0)

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        p50 = self.predict_scalar(row)
        # Heuristic tail: same as EWMA approach, p90 = 2x, p99 = 3x in log space
        # means roughly p50 * (p50 ratio) for tail
        return {"p50": p50, "p90": p50 * 2.0, "p99": p50 * 4.0}


class RandomForestRemainingRegressor:
    """Log-space random forest for post-tool remaining-time prediction.

    Prediction point: after the current tool call has completed. Therefore the
    current tool duration is available, but future episode length/remaining
    steps are not.
    """

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 14,
        min_samples_leaf: int = 6,
        tail_scale: float = 1.10,
    ) -> None:
        self.tail_scale = tail_scale
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=7,
            n_jobs=-1,
            oob_score=True,
        )
        self.residual_q90 = 0.0
        self.residual_q99 = 0.0
        self.global_p50 = 0.0

    def fit(self, rows: list[EpisodeStep]) -> "RandomForestRemainingRegressor":
        fit_rows = [r for r in rows if r.remaining_time_ms >= 0]
        self.global_p50 = median([r.remaining_time_ms for r in fit_rows]) if fit_rows else 0.0
        if not fit_rows:
            return self
        X = self.vectorizer.fit_transform([_remaining_feature_dict(r) for r in fit_rows])
        y = np.asarray([math.log1p(r.remaining_time_ms) for r in fit_rows], dtype=float)
        self.model.fit(X, y)

        pred = getattr(self.model, "oob_prediction_", None)
        if pred is None or len(pred) != len(y) or not np.isfinite(pred).all():
            pred = self.model.predict(X)
        residuals = sorted((yt - yp) for yt, yp in zip(y, pred))
        self.residual_q90 = quantile(residuals, 0.90)
        self.residual_q99 = quantile(residuals, 0.99)
        return self

    def _predict_log(self, row: EpisodeStep) -> float:
        if not hasattr(self.model, "estimators_"):
            return math.log1p(self.global_p50)
        X = self.vectorizer.transform([_remaining_feature_dict(row)])
        return float(self.model.predict(X)[0])

    def predict_scalar(self, row: EpisodeStep) -> float:
        return max(0.0, math.expm1(self._predict_log(row)))

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        log_p50 = self._predict_log(row)
        p50 = max(0.0, math.expm1(log_p50))
        p90 = max(p50, math.expm1(log_p50 + max(0.0, self.residual_q90) * self.tail_scale))
        p99 = max(p90, math.expm1(log_p50 + max(0.0, self.residual_q99) * self.tail_scale))
        return {"p50": p50, "p90": p90, "p99": p99}


def _remaining_feature_dict(row: EpisodeStep) -> dict[str, float]:
    s = row.sample
    duration = s.duration_ms or 0.0
    mean_so_far = row.cumulative_time_ms / max(1, row.step_index + 1)
    history = list(s.history or [])
    out: dict[str, float] = {
        f"dataset={s.dataset}": 1.0,
        f"tool={s.tool}": 1.0,
        f"operation={s.operation}": 1.0,
        f"family={s.tool_family}": 1.0,
        f"resource={s.labels.get('resource_class', 'unknown')}": 1.0,
        "step_index_log": math.log1p(row.step_index),
        "cumulative_time_log": math.log1p(max(0.0, row.cumulative_time_ms)),
        "current_duration_log": math.log1p(max(0.0, duration)),
        "mean_duration_so_far_log": math.log1p(max(0.0, mean_so_far)),
        "command_len_log": math.log1p(max(0.0, float(s.features.get("command_len", 0)))),
        "argv_count_log": math.log1p(max(0.0, float(s.features.get("argv_count", 0)))),
        "flag_count_log": math.log1p(max(0.0, float(s.features.get("flag_count", 0)))),
        "has_pipe": 1.0 if s.features.get("has_pipe") else 0.0,
        "has_recursive_hint": 1.0 if s.features.get("has_recursive_hint") else 0.0,
        "history_len": float(len(history)),
        "tool_diversity_recent": float(len(set(history))),
    }
    for idx, tool in enumerate(reversed(history[-5:]), start=1):
        out[f"prev{idx}_tool={tool}"] = 1.0
    return out


# ──────────────────────────────────────────────
#  6. EWMA remaining by tool_family
# ──────────────────────────────────────────────

class EwmaRemainingByFamily:
    """EWMA over remaining time, grouped by the current step's tool_family."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha
        self.global_median_remaining: float = 0.0
        self.state: dict[str, float] = {}

    def fit(self, rows: list[EpisodeStep]) -> "EwmaRemainingByFamily":
        values = [r.remaining_time_ms for r in rows]
        self.global_median_remaining = median(values) if values else 0.0
        for r in rows:
            family = r.sample.tool_family
            old = self.state.get(family, self.global_median_remaining)
            self.state[family] = (1 - self.alpha) * old + self.alpha * r.remaining_time_ms
        return self

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        p50 = self.state.get(row.sample.tool_family, self.global_median_remaining)
        return {"p50": p50, "p90": p50 * 1.5, "p99": p50 * 2.0}

    def predict_scalar(self, row: EpisodeStep) -> float:
        return self.state.get(row.sample.tool_family, self.global_median_remaining)


# ──────────────────────────────────────────────
#  7. Compositional remaining-time predictor (Pythia decomposition)
# ──────────────────────────────────────────────

class CompositionalRemaining:
    """Predict remaining time by decomposing into:
    1. Future tool sequence (Markov rollout)
    2. Per-tool duration (group quantile model)
    3. Sum predicted durations

    This mirrors Pythia's approach of predicting execution properties
    by decomposing into predictable sub-components rather than
    directly regressing the total.
    """

    def __init__(
        self,
        rollout_depth: int = 10,
        discount: float = 0.95,
    ) -> None:
        self.rollout_depth = rollout_depth
        self.discount = discount  # decay for longer rollouts
        self.tool_durations: dict[str, dict[str, float]] = {}  # tool -> {p50, p90, p99}
        self.transitions: dict[str, list[tuple[str, float]]] = {}  # tool -> [(next, prob)]
        self.global_p50: float = 0.0
        self.global_p90: float = 0.0
        self.global_p99: float = 0.0

    def fit(self, rows: list[EpisodeStep]) -> "CompositionalRemaining":
        from collections import Counter, defaultdict

        # Fit per-tool duration quantiles from all individual samples
        tool_durations_raw: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            tool_durations_raw[r.sample.tool].append(
                r.sample.duration_ms if r.sample.duration_ms is not None else 0.0
            )

        all_durations = []
        for tool, durs in tool_durations_raw.items():
            self.tool_durations[tool] = {
                "p50": quantile(durs, 0.50),
                "p90": quantile(durs, 0.90),
                "p99": quantile(durs, 0.99),
            }
            all_durations.extend(durs)

        self.global_p50 = quantile(all_durations, 0.50)
        self.global_p90 = quantile(all_durations, 0.90)
        self.global_p99 = quantile(all_durations, 0.99)

        # Fit Markov transitions from episode sequences
        trans_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            if r.sample.next_tool:
                trans_counts[r.sample.tool][r.sample.next_tool] += 1

        for tool, counts in trans_counts.items():
            total = sum(counts.values())
            # Keep top-5 transitions sorted by probability
            sorted_trans = counts.most_common(5)
            self.transitions[tool] = [(t, c / total) for t, c in sorted_trans]

        return self

    def _predict_tool_duration(self, tool: str) -> dict[str, float]:
        d = self.tool_durations.get(tool)
        if d:
            return dict(d)
        return {"p50": self.global_p50, "p90": self.global_p90, "p99": self.global_p99}

    def _rollout_remaining(
        self, current_tool: str
    ) -> tuple[float, float, float]:
        """Simulate future tool sequence and accumulate predicted durations."""
        total_p50 = 0.0
        total_p90 = 0.0
        total_p99 = 0.0
        tool = current_tool
        weight = 1.0

        for step in range(self.rollout_depth):
            # Predict next tool
            trans = self.transitions.get(tool, [])
            if not trans:
                # No transitions known: use global average for remaining
                avg_remaining_steps = max(0, self.rollout_depth - step)
                total_p50 += self.global_p50 * avg_remaining_steps * weight
                total_p90 += self.global_p90 * avg_remaining_steps * weight
                total_p99 += self.global_p99 * avg_remaining_steps * weight
                break

            # Expected duration of the predicted next tool
            next_tool, prob = trans[0]  # most likely next tool
            dur = self._predict_tool_duration(next_tool)
            total_p50 += dur["p50"] * prob * weight
            total_p90 += dur["p90"] * prob * weight
            total_p99 += dur["p99"] * prob * weight

            tool = next_tool
            weight *= self.discount

        return total_p50, total_p90, total_p99

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        p50, p90, p99 = self._rollout_remaining(row.sample.tool)
        return {"p50": p50, "p90": p90, "p99": p99}

    def predict_scalar(self, row: EpisodeStep) -> float:
        return self._rollout_remaining(row.sample.tool)[0]


# ──────────────────────────────────────────────
#  8. Binned remaining-time classifier (softmax)
# ──────────────────────────────────────────────

# Scheduling-semantic bucket boundaries in seconds.
# Chosen so each bucket captures a meaningful scheduling decision:
#   <= 15s  : "overlap"  – hideable behind LLM inference
#   15-60s  : "short"    – quick, light scheduling
#   60-180s : "moderate" – normal scheduling
#   180-600s: "long"     – significant resource commitment
#   600-1800s:"heavy"    – long-running, consider preemption
#   > 1800s : "extreme"  – very long, special handling
BUCKET_BOUNDARIES_SEC = [15.0, 60.0, 180.0, 600.0, 1800.0]


class BinnedRemainingClassifier:
    """Multinomial logistic regression (softmax) for remaining-time buckets."""

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 500,
        batch_size: int = 256,
        l2: float = 0.0001,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.boundaries = list(BUCKET_BOUNDARIES_SEC)
        self.n_classes = len(self.boundaries) + 1
        self.W: list[list[float]] = []
        self.b: list[float] = []
        self.feat_means: list[float] = []
        self.feat_stds: list[float] = []
        self.bucket_medians: list[float] = []

    @staticmethod
    def _bucket_index(remaining_sec: float) -> int:
        for i, bound in enumerate(BUCKET_BOUNDARIES_SEC):
            if remaining_sec <= bound:
                return i
        return len(BUCKET_BOUNDARIES_SEC)

    def fit(self, rows: list[EpisodeStep]) -> "BinnedRemainingClassifier":
        X = _feature_matrix(rows)
        Xn, _, self.feat_means, self.feat_stds = _normalize_features(X)

        y_bucket = [self._bucket_index(r.remaining_time_ms / 1000.0) for r in rows]

        bucket_values: dict[int, list[float]] = {i: [] for i in range(self.n_classes)}
        for r, b_idx in zip(rows, y_bucket):
            bucket_values[b_idx].append(r.remaining_time_ms)
        self.bucket_medians = []
        for i in range(self.n_classes):
            vals = sorted(bucket_values[i])
            self.bucket_medians.append(quantile(vals, 0.50) if vals else 5000.0)

        n_feat = len(Xn[0])
        n_samples = len(Xn)
        self.W = [[0.0] * n_feat for _ in range(self.n_classes)]
        self.b = [0.0] * self.n_classes

        indices = list(range(n_samples))
        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.epochs):
            random.shuffle(indices)
            decay = 1.0 / (1.0 + epoch * 0.005)
            step_lr = self.lr * decay

            for bi in range(n_batches):
                start = bi * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                if not batch_idx:
                    continue

                m = len(batch_idx)
                grad_W = [[0.0] * n_feat for _ in range(self.n_classes)]
                grad_b = [0.0] * self.n_classes

                for idx in batch_idx:
                    scores = [self.b[c] + _dot(Xn[idx], self.W[c]) for c in range(self.n_classes)]
                    max_score = max(scores)
                    exp_scores = [math.exp(s - max_score) for s in scores]
                    sum_exp = sum(exp_scores)
                    probs = [es / sum_exp for es in exp_scores]

                    y_true = y_bucket[idx]
                    for c in range(self.n_classes):
                        grad_c = probs[c] - (1.0 if c == y_true else 0.0)
                        for j in range(n_feat):
                            grad_W[c][j] += grad_c * Xn[idx][j]
                        grad_b[c] += grad_c

                for c in range(self.n_classes):
                    for j in range(n_feat):
                        g = grad_W[c][j] / m + self.l2 * self.W[c][j]
                        self.W[c][j] -= step_lr * g
                    self.b[c] -= step_lr * (grad_b[c] / m)

        return self

    def predict_proba(self, row: EpisodeStep) -> list[float]:
        X = _feature_matrix([row])
        Xn = _apply_norm(X, self.feat_means, self.feat_stds)
        scores = [self.b[c] + _dot(Xn[0], self.W[c]) for c in range(self.n_classes)]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [es / sum_exp for es in exp_scores]

    def predict_scalar(self, row: EpisodeStep) -> float:
        probs = self.predict_proba(row)
        return sum(p * self.bucket_medians[i] for i, p in enumerate(probs))

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        probs = self.predict_proba(row)
        p50 = sum(p * self.bucket_medians[i] for i, p in enumerate(probs))
        cum = 0.0
        p90_bucket = self.n_classes - 1
        for i, p in enumerate(probs):
            cum += p
            if cum >= 0.90:
                p90_bucket = i
                break
        p90 = self.bucket_medians[p90_bucket] * 1.5
        p99_bucket = min(self.n_classes - 1, p90_bucket + 1)
        p99 = self.bucket_medians[p99_bucket] * 2.0
        return {"p50": p50, "p90": max(p50, p90), "p99": max(p50, p99)}


# ──────────────────────────────────────────────
#  9. Steps-based remaining predictor (Pythia-style decomposition)
# ──────────────────────────────────────────────

# Step-count bucket boundaries (coarse, scheduling-semantic)
STEPS_BUCKETS = [3, 10, 25, 50]  # boundaries for remaining steps


class StepsDecomposedRemaining:
    """Predict remaining_steps first (via softmax classifier), then
    convert to remaining time by multiplying with per-tool-family
    mean duration.

    This decomposes the hard remaining-time problem into:
    1. Classify remaining steps into coarse buckets (easier, more signal)
    2. Multiply by tool-specific mean duration

    This mirrors Pythia's decompose-and-compose philosophy.
    """

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 500,
        batch_size: int = 256,
        l2: float = 0.0001,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.step_boundaries = list(STEPS_BUCKETS)
        self.n_step_classes = len(self.step_boundaries) + 1
        self.W: list[list[float]] = []
        self.b: list[float] = []
        self.feat_means: list[float] = []
        self.feat_stds: list[float] = []
        # Per-step-class median steps (from training)
        self.step_class_medians: list[float] = []
        # Per-tool-family mean duration per step (ms)
        self.family_mean_duration: dict[str, float] = {}
        self.global_mean_duration: float = 0.0

    @staticmethod
    def _bucket_index(remaining_steps: int) -> int:
        for i, bound in enumerate(STEPS_BUCKETS):
            if remaining_steps <= bound:
                return i
        return len(STEPS_BUCKETS)

    def _enriched_features(self, row: EpisodeStep) -> list[float]:
        """Feature vector with categorical one-hot encoding for tool_family and operation."""
        base = row.feature_vector()
        # Add one-hot for tool_family (top families)
        family = row.sample.tool_family
        for f in ["file", "search", "test", "terminal", "network", "control", "tool"]:
            base[f"family_{f}"] = 1.0 if family == f else 0.0
        # Add one-hot for operation (top ops)
        op = row.sample.operation
        for o in ["read_file", "grep", "find", "pytest", "python", "build", "git", "edit_file", "list_dir"]:
            base[f"op_{o}"] = 1.0 if op == o else 0.0
        # Interaction: step_index * mean_duration_so_far
        base["step_x_mean_dur"] = base["step_index"] * base["mean_duration_so_far_ms"]
        # Progress proxy: cumulative / (cumulative + small_constant)
        base["progress_proxy"] = base["cumulative_time_ms"] / (base["cumulative_time_ms"] + 60000.0)
        return base

    @staticmethod
    def enriched_feature_names() -> list[str]:
        base = EpisodeStep.feature_names()
        families = ["family_file", "family_search", "family_test", "family_terminal",
                     "family_network", "family_control", "family_tool"]
        ops = ["op_read_file", "op_grep", "op_find", "op_pytest", "op_python",
               "op_build", "op_git", "op_edit_file", "op_list_dir"]
        extra = ["step_x_mean_dur", "progress_proxy"]
        return base + families + ops + extra

    def _feature_matrix_enriched(self, rows: list[EpisodeStep]) -> list[list[float]]:
        names = self.enriched_feature_names()
        X = []
        for r in rows:
            fv = self._enriched_features(r)
            X.append([fv.get(name, 0.0) for name in names])
        return X

    def fit(self, rows: list[EpisodeStep]) -> "StepsDecomposedRemaining":
        # Compute per-family mean duration
        from collections import defaultdict
        fam_durs: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r.sample.duration_ms is not None:
                fam_durs[r.sample.tool_family].append(r.sample.duration_ms)
        all_durs = []
        for fam, durs in fam_durs.items():
            self.family_mean_duration[fam] = sum(durs) / len(durs) if durs else 0.0
            all_durs.extend(durs)
        self.global_mean_duration = sum(all_durs) / len(all_durs) if all_durs else 0.0

        # Prepare feature matrix with categorical encoding
        X = self._feature_matrix_enriched(rows)
        Xn, _, self.feat_means, self.feat_stds = _normalize_features(X)

        # Assign step bucket labels
        y_bucket = [self._bucket_index(r.remaining_steps) for r in rows]

        # Compute per-class median remaining steps
        class_steps: dict[int, list[int]] = {i: [] for i in range(self.n_step_classes)}
        for r, b_idx in zip(rows, y_bucket):
            class_steps[b_idx].append(r.remaining_steps)
        self.step_class_medians = []
        for i in range(self.n_step_classes):
            vals = sorted(class_steps[i])
            self.step_class_medians.append(float(quantile([float(v) for v in vals], 0.50)) if vals else 0.0)

        # Train softmax regression
        n_feat = len(Xn[0])
        n_samples = len(Xn)
        self.W = [[0.0] * n_feat for _ in range(self.n_step_classes)]
        self.b = [0.0] * self.n_step_classes

        indices = list(range(n_samples))
        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.epochs):
            random.shuffle(indices)
            decay = 1.0 / (1.0 + epoch * 0.005)
            step_lr = self.lr * decay

            for bi in range(n_batches):
                start = bi * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                if not batch_idx:
                    continue

                m = len(batch_idx)
                grad_W = [[0.0] * n_feat for _ in range(self.n_step_classes)]
                grad_b = [0.0] * self.n_step_classes

                for idx in batch_idx:
                    scores = [self.b[c] + _dot(Xn[idx], self.W[c]) for c in range(self.n_step_classes)]
                    max_score = max(scores)
                    exp_scores = [math.exp(s - max_score) for s in scores]
                    sum_exp = sum(exp_scores)
                    probs = [es / sum_exp for es in exp_scores]

                    y_true = y_bucket[idx]
                    for c in range(self.n_step_classes):
                        grad_c = probs[c] - (1.0 if c == y_true else 0.0)
                        for j in range(n_feat):
                            grad_W[c][j] += grad_c * Xn[idx][j]
                        grad_b[c] += grad_c

                for c in range(self.n_step_classes):
                    for j in range(n_feat):
                        g = grad_W[c][j] / m + self.l2 * self.W[c][j]
                        self.W[c][j] -= step_lr * g
                    self.b[c] -= step_lr * (grad_b[c] / m)

        return self

    def _predict_steps_proba(self, row: EpisodeStep) -> list[float]:
        X = self._feature_matrix_enriched([row])
        Xn = _apply_norm(X, self.feat_means, self.feat_stds)
        scores = [self.b[c] + _dot(Xn[0], self.W[c]) for c in range(self.n_step_classes)]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [es / sum_exp for es in exp_scores]

    def predict_remaining_steps(self, row: EpisodeStep) -> float:
        """Expected remaining steps."""
        probs = self._predict_steps_proba(row)
        return sum(p * self.step_class_medians[i] for i, p in enumerate(probs))

    def predict_scalar(self, row: EpisodeStep) -> float:
        """Expected remaining time = predicted_steps * mean_duration_per_step."""
        pred_steps = self.predict_remaining_steps(row)
        mean_dur = self.family_mean_duration.get(
            row.sample.tool_family, self.global_mean_duration
        )
        return max(0.0, pred_steps * mean_dur)

    def predict(self, row: EpisodeStep) -> dict[str, float]:
        p50 = self.predict_scalar(row)
        # Conservative tail: add 50% / 100% margin
        return {"p50": p50, "p90": p50 * 1.5, "p99": p50 * 2.0}


# ──────────────────────────────────────────────
#  Pure-Python linear algebra helpers
# ──────────────────────────────────────────────

def _dot(x: list[float], w: list[float]) -> float:
    return sum(xi * wi for xi, wi in zip(x, w))


def _linear_predict(x: list[float], w: list[float], b: float) -> float:
    return _dot(x, w) + b


def _apply_norm(
    X: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    n_feat = len(means)
    result = []
    for row in X:
        result.append([(row[j] - means[j]) / stds[j] for j in range(n_feat)])
    return result


def _pinball_gradient(error: float, q: float) -> float:
    """Subgradient of pinball loss w.r.t. prediction."""
    if error > 0:
        return -q
    elif error < 0:
        return 1.0 - q
    return 0.0


def _quantile_regression_fit(
    X: list[list[float]],
    y: list[float],
    q: float,
    lr: float = 0.01,
    epochs: int = 300,
    batch_size: int = 128,
    l2: float = 0.001,
) -> tuple[list[float], float]:
    """Mini-batch gradient descent for linear quantile regression."""
    n = len(X)
    if n == 0:
        return [], 0.0
    n_feat = len(X[0])
    w = [0.0] * n_feat
    b = 0.0

    indices = list(range(n))
    n_batches = max(1, n // batch_size)

    for epoch in range(epochs):
        random.shuffle(indices)
        decay = 1.0 / (1.0 + epoch * 0.01)
        step_lr = lr * decay

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            if not batch_idx:
                continue

            grad_w = [0.0] * n_feat
            grad_b = 0.0
            m = len(batch_idx)

            for idx in batch_idx:
                pred = _dot(X[idx], w) + b
                error = y[idx] - pred
                g = _pinball_gradient(error, q)
                for j in range(n_feat):
                    grad_w[j] += g * X[idx][j]
                grad_b += g

            # Average + L2 regularization
            for j in range(n_feat):
                grad_w[j] = grad_w[j] / m + l2 * w[j]
                w[j] -= step_lr * grad_w[j]
            grad_b = grad_b / m
            b -= step_lr * grad_b

    return w, b


def _mse_regression_fit(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.01,
    epochs: int = 300,
    l2: float = 0.001,
) -> tuple[list[float], float]:
    """Mini-batch gradient descent for MSE linear regression."""
    n = len(X)
    if n == 0:
        return [], 0.0
    n_feat = len(X[0])
    w = [0.0] * n_feat
    b = 0.0
    indices = list(range(n))
    n_batches = max(1, n // 128)

    for epoch in range(epochs):
        random.shuffle(indices)
        decay = 1.0 / (1.0 + epoch * 0.01)
        step_lr = lr * decay

        for bi in range(n_batches):
            start = bi * 128
            end = min(start + 128, n)
            batch_idx = indices[start:end]
            if not batch_idx:
                continue

            grad_w = [0.0] * n_feat
            grad_b = 0.0
            m = len(batch_idx)

            for idx in batch_idx:
                pred = _dot(X[idx], w) + b
                error = pred - y[idx]  # MSE: dL/dpred = pred - y
                for j in range(n_feat):
                    grad_w[j] += error * X[idx][j]
                grad_b += error

            for j in range(n_feat):
                grad_w[j] = 2.0 * grad_w[j] / m + l2 * w[j]
                w[j] -= step_lr * grad_w[j]
            grad_b = 2.0 * grad_b / m
            b -= step_lr * grad_b

    return w, b


def _linear_fit(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.05,
    epochs: int = 200,
) -> tuple[list[float], float]:
    """Simple MSE linear fit (for progress prediction)."""
    return _mse_regression_fit(X, y, lr=lr, epochs=epochs, l2=0.0)
