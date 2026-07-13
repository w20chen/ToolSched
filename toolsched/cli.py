from __future__ import annotations

import argparse
from pathlib import Path

from .data.discovery import summarize_datasets
from .data.loader import load_datasets
from .episodes import build_episodes
from .evaluation.metrics import classification_metrics, next_tool_metrics, regression_metrics
from .evaluation.ml_suite import run_supervised_suite
from .evaluation.online import replay_calibration
from .evaluation.profile import tool_profiles
from .evaluation.remaining import remaining_by_progress, remaining_time_metrics
from .evaluation.split import split_by_case
from .features.resource_class import infer_resource_class
from .io import read_samples, write_json, write_samples
from .models.baselines import EwmaToolModel, GlobalQuantileModel, GroupQuantileModel, NextToolMarkovModel
from .models.remaining import (
    BinnedRemainingClassifier,
    CompositionalRemaining,
    EwmaRemainingByFamily,
    GlobalRemainingQuantile,
    LinearQuantileRemaining,
    LogSpaceRemaining,
    ProgressConditionedRemaining,
    RandomForestRemainingRegressor,
    StepConditionedRemaining,
    StepsDecomposedRemaining,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="toolsched")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--datasets", required=True)

    p = sub.add_parser("build")
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit-attempts", type=int)
    p.add_argument("--include-dataset", action="append")
    p.add_argument("--min-duration-ms", type=float)

    p = sub.add_parser("evaluate")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")
    p.add_argument("--holdout-dataset")

    p = sub.add_parser("evaluate-supervised")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")

    p = sub.add_parser("profile")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")

    p = sub.add_parser("calibrate")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")

    p = sub.add_parser("evaluate-remaining")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")
    p.add_argument("--min-remaining-ms", type=float, default=0.0)
    p.add_argument("--min-episode-steps", type=int, default=2)

    args = parser.parse_args()
    if args.cmd == "inspect":
        payload = summarize_datasets(Path(args.datasets))
        print_json(payload)
    elif args.cmd == "build":
        include = set(args.include_dataset or []) or None
        samples = list(load_datasets(Path(args.datasets), args.limit_attempts, include, args.min_duration_ms))
        count = write_samples(Path(args.out), samples)
        print_json({"out": args.out, "samples": count})
    elif args.cmd == "evaluate":
        samples = read_samples(Path(args.samples))
        train, test = split_by_case(samples)
        models = {
            "global_quantile": GlobalQuantileModel().fit(train),
            "per_tool_quantile": GroupQuantileModel(("tool",)).fit(train),
            "per_operation_quantile": GroupQuantileModel(("operation",)).fit(train),
            "ewma_tool": EwmaToolModel().fit(train),
        }
        next_model = NextToolMarkovModel().fit(train)
        payload = {
            "split": {"train": len(train), "test": len(test), "strategy": "case_holdout"},
            "latency": {name: regression_metrics(test, model.predict) for name, model in models.items()},
            "resource_class": classification_metrics(
                test, lambda s: infer_resource_class(s.tool_family, s.operation, s.features)
            ),
            "next_tool": next_tool_metrics(test, next_model.predict),
        }
        emit(payload, args.out)
    elif args.cmd == "evaluate-supervised":
        samples = read_samples(Path(args.samples))
        payload = run_supervised_suite(samples)
        emit(payload, args.out)
    elif args.cmd == "profile":
        samples = read_samples(Path(args.samples))
        payload = tool_profiles(samples)
        emit(payload, args.out)
    elif args.cmd == "calibrate":
        samples = read_samples(Path(args.samples))
        train, test = split_by_case(samples)
        model = GroupQuantileModel(("operation",)).fit(train)
        payload = replay_calibration(test, model.predict)
        emit(payload, args.out)
    elif args.cmd == "evaluate-remaining":
        samples = read_samples(Path(args.samples))
        train_samples, test_samples = split_by_case(samples)
        train_episodes = build_episodes(train_samples)
        train_rows = []
        for ep in train_episodes:
            train_rows.extend(ep.build_training_rows())
        test_rows = []
        test_episodes = build_episodes(test_samples)
        for ep in test_episodes:
            test_rows.extend(ep.build_training_rows())

        if not train_rows:
            emit({"error": "no training rows"}, args.out)
            return

        # Fit fast, useful defaults. Slower experimental classes remain
        # available in toolsched.models.remaining but are not run by default.
        models: dict[str, object] = {
            "global_quantile": GlobalRemainingQuantile().fit(train_rows),
            "step_conditioned": StepConditionedRemaining().fit(train_rows),
            "random_forest_log": RandomForestRemainingRegressor().fit(train_rows),
            "ewma_family": EwmaRemainingByFamily().fit(train_rows),
            "compositional": CompositionalRemaining().fit(train_rows),
        }

        # Evaluate
        results = {}
        for name, model in models.items():
            predict_scalar = getattr(model, "predict_scalar")
            predict_dist = getattr(model, "predict", None)
            metrics = remaining_time_metrics(
                test_samples,
                predict_scalar,
                predict_dist,
                min_remaining_ms=args.min_remaining_ms,
                min_episode_steps=args.min_episode_steps,
            )
            # Add per-progress-decile breakdown for top models
            if name in ("random_forest_log", "step_conditioned", "compositional"):
                metrics["by_progress"] = remaining_by_progress(
                    test_samples, predict_scalar, min_episode_steps=args.min_episode_steps,
                )
            results[name] = metrics

        payload = {
            "split": {
                "train_episodes": len(train_episodes),
                "test_episodes": len(test_episodes),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "strategy": "case_holdout",
            },
            "models": results,
        }
        emit(payload, args.out)


def emit(payload: dict, out: str | None) -> None:
    if out:
        write_json(Path(out), payload)
    print_json(payload)


def print_json(payload: dict) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
