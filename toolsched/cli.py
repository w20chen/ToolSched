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
from .scheduler.placement import placement_metrics
from .scheduler.placement_collect import aggregate_placement_rows, collect_placement_study
from .scheduler.placement_manifest import (
    build_replay_manifest,
    build_toolsched_study_manifest,
    parse_path_maps,
)
from .scheduler.speculation import speculation_metrics


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

    p = sub.add_parser("simulate-placement")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")
    p.add_argument(
        "--mode",
        choices=("real", "synthetic", "both"),
        default="real",
        help="real is the default; synthetic is an explicitly labeled policy stress test",
    )
    p.add_argument("--synthetic-clusters", type=int, default=2)
    p.add_argument("--synthetic-cores-per-cluster", type=int, default=2)

    p = sub.add_parser("prepare-placement-manifest")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--path-map", action="append", help="rewrite trace paths as OLD=NEW")
    p.add_argument("--max-entries", type=int, default=50)
    p.add_argument("--min-observed-duration-ms", type=float, default=50.0)
    p.add_argument(
        "--approve-safe-read-only",
        action="store_true",
        help="approve only commands accepted by the strict read-only allowlist",
    )

    p = sub.add_parser("collect-placement")
    p.add_argument("--manifest", required=True)
    p.add_argument("--raw-out", required=True)
    p.add_argument("--samples-out", required=True)
    p.add_argument("--metrics-out")
    p.add_argument("--summary-out")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--max-corunners", type=int, default=2)
    p.add_argument(
        "--scenario",
        action="append",
        choices=("idle", "smt_busy", "cluster_a_busy", "cluster_b_busy"),
    )
    p.add_argument("--warmup-seconds", type=float, default=0.20)
    p.add_argument("--utilization-window-seconds", type=float, default=0.15)
    p.add_argument("--peer-baseline-seconds", type=float, default=1.0)
    p.add_argument("--perf-mode", choices=("auto", "required", "off"), default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-success-per-candidate", type=int, default=3)
    p.add_argument(
        "--execute-approved-manifest",
        action="store_true",
        help="required acknowledgement that approved manifest commands will execute",
    )

    p = sub.add_parser("prepare-placement-study")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cwd", default=".")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument(
        "--workload",
        action="append",
        choices=("bucket_logistic", "bucket_forest", "next_tool", "remaining_forest", "quantile"),
        help="repeat to select a subset; default is all workloads",
    )

    p = sub.add_parser("aggregate-placement")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-success-per-candidate", type=int, default=3)

    p = sub.add_parser("speculate")
    p.add_argument("--samples", required=True)
    p.add_argument("--out")
    p.add_argument("--llm-slack-ms", type=float, default=1500.0)

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
            "per_operation_resource_quantile": GroupQuantileModel(("operation", "resource_class")).fit(train),
            "ewma_tool": EwmaToolModel().fit(train),
        }
        next_model = NextToolMarkovModel().fit(train)
        payload = {
            "split": {"train": len(train), "test": len(test), "strategy": "case_holdout"},
            "latency": {name: regression_metrics(test, model.predict) for name, model in models.items()},
            "resource_class": classification_metrics(test, lambda s: s.features.get("resource_class_heuristic", "unknown")),
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
        model = GroupQuantileModel(("operation", "resource_class")).fit(train)
        payload = replay_calibration(test, model.predict)
        emit(payload, args.out)
    elif args.cmd == "simulate-placement":
        samples = read_samples(Path(args.samples))
        train, test = split_by_case(samples)
        model = GroupQuantileModel(("operation", "resource_class")).fit(train)
        payload = placement_metrics(
            test,
            model.predict,
            mode=args.mode,
            synthetic_clusters=args.synthetic_clusters,
            synthetic_cores_per_cluster=args.synthetic_cores_per_cluster,
        )
        emit(payload, args.out)
    elif args.cmd == "prepare-placement-manifest":
        samples = read_samples(Path(args.samples))
        payload = build_replay_manifest(
            samples,
            path_maps=parse_path_maps(args.path_map),
            max_entries=args.max_entries,
            min_observed_duration_ms=args.min_observed_duration_ms,
            approve_safe_read_only=args.approve_safe_read_only,
        )
        write_json(Path(args.out), payload)
        print_json({
            "out": args.out,
            "entries": len(payload["entries"]),
            "approved": sum(entry.get("approved") is True for entry in payload["entries"]),
            "selection": payload["selection"],
        })
    elif args.cmd == "collect-placement":
        if not args.execute_approved_manifest:
            parser.error("collect-placement requires --execute-approved-manifest")
        payload = collect_placement_study(
            manifest_path=Path(args.manifest),
            raw_out=Path(args.raw_out),
            samples_out=Path(args.samples_out),
            repeats=args.repeats,
            max_candidates=args.max_candidates,
            max_corunners=args.max_corunners,
            scenarios_requested=tuple(args.scenario or (
                "idle", "smt_busy", "cluster_a_busy", "cluster_b_busy"
            )),
            warmup_seconds=args.warmup_seconds,
            utilization_window_seconds=args.utilization_window_seconds,
            peer_baseline_seconds=args.peer_baseline_seconds,
            perf_mode=args.perf_mode,
            seed=args.seed,
            min_success_per_candidate=args.min_success_per_candidate,
        )
        if args.metrics_out:
            collected = read_samples(Path(args.samples_out))
            if collected:
                train, test = split_by_case(collected)
                # With a tiny pilot, case holdout can leave too little evidence.
                # Keep the split for model fitting but report which rows were evaluated.
                fit_rows = train or collected
                eval_rows = test or collected
                model = GroupQuantileModel(("operation", "resource_class")).fit(fit_rows)
                metrics = placement_metrics(eval_rows, model.predict, mode="real")
            else:
                metrics = {"error": "collector produced no aggregatable placement samples"}
            write_json(Path(args.metrics_out), metrics)
            payload["metrics_out"] = args.metrics_out
            payload["metrics"] = metrics
        if args.summary_out:
            payload["summary_out"] = args.summary_out
            write_json(Path(args.summary_out), payload)
        print_json(payload)
    elif args.cmd == "prepare-placement-study":
        payload = build_toolsched_study_manifest(
            Path(args.samples),
            Path(args.cwd),
            shard_count=args.shards,
            workload_names=set(args.workload) if args.workload else None,
        )
        write_json(Path(args.out), payload)
        print_json({
            "out": args.out,
            "entries": len(payload["entries"]),
            "study_design": payload["study_design"],
        })
    elif args.cmd == "aggregate-placement":
        samples, summary = aggregate_placement_rows(
            Path(args.raw), args.min_success_per_candidate
        )
        count = write_samples(Path(args.out), samples)
        print_json({"out": args.out, "samples": count, "aggregation": summary})
    elif args.cmd == "speculate":
        samples = read_samples(Path(args.samples))
        train, test = split_by_case(samples)
        cost_model = GroupQuantileModel(("operation", "resource_class")).fit(train)
        next_model = NextToolMarkovModel().fit(train)
        payload = speculation_metrics(test, cost_model.predict, next_model.predict, args.llm_slack_ms)
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
