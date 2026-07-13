"""Representative ToolSched workloads for an end-to-end placement study.

These are real repository computations over the selected artifact, not
synthetic spin loops. Co-runner processes remain controlled stressors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ..episodes import build_episodes
from ..evaluation.split import split_by_case
from ..io import read_samples
from ..models.baselines import GroupQuantileModel
from ..models.buckets import HistoricalBucketFeatureModel, LatencyBucketModel
from ..models.next_tool import NextToolLogisticModel
from ..models.remaining import RandomForestRemainingRegressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument(
        "--workload",
        choices=("bucket_logistic", "bucket_forest", "next_tool", "remaining_forest", "quantile"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    samples = read_samples(Path(args.samples))
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < count")
    if args.shard_count > 1:
        samples = [
            sample for sample in samples
            if _case_shard(sample.dataset, sample.case_id, args.shard_count) == args.shard_index
        ]
    if not samples:
        parser.error("selected shard contains no samples")
    train, test = split_by_case(samples)
    started = time.perf_counter()
    if args.workload == "bucket_logistic":
        model = LatencyBucketModel().fit(train)
        checksum = sum(model.predict_index(row) for row in test[:500])
    elif args.workload == "bucket_forest":
        model = HistoricalBucketFeatureModel().fit(train)
        checksum = sum(model.predict_index(row) for row in test[:500])
    elif args.workload == "next_tool":
        model = NextToolLogisticModel().fit(train)
        checksum = sum(model.predict(row)[1] for row in test[:500])
    elif args.workload == "remaining_forest":
        train_rows = [row for episode in build_episodes(train) for row in episode.build_training_rows()]
        model = RandomForestRemainingRegressor().fit(train_rows)
        checksum = sum(model.predict_scalar(row) for row in train_rows[:500])
    else:
        model = GroupQuantileModel(("operation", "resource_class")).fit(train)
        checksum = sum(model.predict(row).latency_p90_ms for row in test[:500])
    print(
        json.dumps(
            {
                "workload": args.workload,
                "train": len(train),
                "test": len(test),
                "checksum": checksum,
                "elapsed_seconds": time.perf_counter() - started,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
            sort_keys=True,
        )
    )


def _case_shard(dataset: str, case_id: str, count: int) -> int:
    digest = hashlib.sha256(f"{dataset}\x1f{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


if __name__ == "__main__":
    main()
